"""Model registry and automatic selection.

Several local video models are installed, each good at different things.
Selecting one by hand per piece is the kind of decision an operator should
never have to make, so each model DECLARES its envelope and strengths and
the router matches those against what the piece needs.

Every number here is a measured or published constraint, not a preference.
The two that matter most:

* ``max_pixels`` — the per-frame budget beyond which the model returns a
  BLANK frame rather than a degraded one. LTX-Video renders 704x480 and
  returns a flat fill at 704x1280; nothing about the output says which
  happened. Exceeding this is not a quality trade, it is a silent failure.
* ``vram_gb`` — peak measured with CPU offload enabled. A model that does
  not fit is not slower, it raises.

Selection never invents capability: if nothing fits, it says so rather
than picking the least-bad option and producing garbage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from clipforge.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    """What one local generation model can do."""

    key: str
    model_id: str
    label: str
    #: Per-frame pixel budget. Above this the model returns blank frames.
    max_pixels: int
    #: Peak VRAM in GB with CPU offload, measured.
    vram_gb: float
    #: Inference steps that produce usable output.
    steps: int
    guidance_scale: float
    #: Relative seconds per generated second of video, on an RTX 3090.
    #: Lower is faster. Used only to break ties.
    cost: float
    #: What this model is GOOD at. Matched against a piece's needs.
    strengths: frozenset[str]
    #: Frame counts must be this many plus one (temporal VAE grouping).
    frame_group: int = 8
    #: Dimensions must be a multiple of this (latent patch grid).
    dim_multiple: int = 32
    notes: str = ""

    def fits(self, vram_gb: float) -> bool:
        return self.vram_gb <= vram_gb

    def supports(self, width: int, height: int) -> bool:
        return (width * height <= self.max_pixels
                and width % self.dim_multiple == 0
                and height % self.dim_multiple == 0)


#: LTX-Video: fast, low VRAM, modest fidelity. Measured envelope — 512x896
#: renders, 704x1280 is blank. Good for abstract/graphic work and for
#: iterating, poor at photoreal human motion.
LTX_VIDEO = ModelSpec(
    key="ltx",
    model_id="Lightricks/LTX-Video",
    label="LTX-Video 0.9",
    max_pixels=460_000,
    vram_gb=10.0,
    steps=30,
    guidance_scale=3.0,
    cost=1.0,
    strengths=frozenset({"fast", "abstract", "motion_graphics", "draft",
                         "atmospheric", "landscape"}),
    notes=("Blank above ~460k px/frame — measured, not estimated. Fastest "
           "option; use for iteration and for graphic/abstract work."),
)

#: Wan 2.2 TI2V 5B: the consumer-card variant Wan publishes for 24 GB.
#: Markedly stronger on photoreal motion and human subjects. The A14B
#: sibling is deliberately NOT registered — it does not fit 24 GB at
#: useful video resolutions, and a model that raises is worse than absent.
WAN22_TI2V_5B = ModelSpec(
    key="wan22",
    model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    label="Wan 2.2 TI2V 5B",
    max_pixels=720 * 1280,
    vram_gb=18.0,
    steps=40,
    guidance_scale=5.0,
    cost=3.2,
    strengths=frozenset({"photoreal", "human", "detail", "cinematic",
                         "motion", "hero", "landscape"}),
    notes=("Higher fidelity, ~3x the render time of LTX. Envelope is the "
           "published 720p target and is UNVERIFIED on this machine until "
           "a real render is inspected."),
)

REGISTRY: dict[str, ModelSpec] = {
    m.key: m for m in (LTX_VIDEO, WAN22_TI2V_5B)
}


def _generation_dims_for(aspect_ratio: str) -> tuple[int, int]:
    """The size generation will actually request for this aspect.

    Selection has to score models against the size they will be ASKED for,
    not the delivery size — the envelope that matters is the generation
    one, and asking past it returns blank frames rather than an error.
    """
    from clipforge.genvideo.providers import _generation_dims

    return _generation_dims(aspect_ratio)


def available_models(vram_gb: float = 24.0) -> list[ModelSpec]:
    """Registered models whose weights are present and which fit the card."""
    out = []
    for spec in REGISTRY.values():
        if not spec.fits(vram_gb):
            continue
        if not weights_present(spec):
            continue
        out.append(spec)
    return out


def weights_present(spec: ModelSpec) -> bool:
    """Whether the model is downloaded, without importing torch.

    Checks the HF cache directly: constructing a pipeline to find out
    would download tens of gigabytes as a side effect of a availability
    check.
    """
    from pathlib import Path

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        root = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        root = Path.home() / ".cache" / "huggingface" / "hub"
    folder = root / ("models--" + spec.model_id.replace("/", "--"))
    if not folder.is_dir():
        return False
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    # A folder with only .incomplete blobs is a half-download, not a model.
    return any(any(s.iterdir()) for s in snapshots.iterdir() if s.is_dir())


def select_model(*, needs: frozenset[str] | set[str] | None = None,
                 width: int, height: int, vram_gb: float = 24.0,
                 prefer: str | None = None) -> ModelSpec:
    """Best installed model for a piece with these needs and dimensions.

    ``needs`` are tags from the niche ("photoreal", "abstract", ...).
    Scoring is deliberately simple and explainable: overlap with the
    model's strengths, then lower cost as the tie-break. An operator can
    read the log line and understand the choice, which matters more than
    a cleverer ranking nobody can debug.
    """
    candidates = [m for m in available_models(vram_gb)
                  if m.supports(width, height)]
    if prefer:
        forced = REGISTRY.get(prefer)
        if forced is None:
            raise ValueError(
                f"unknown model {prefer!r}; known: {', '.join(sorted(REGISTRY))}")
        if forced not in candidates:
            why = ("weights not downloaded" if not weights_present(forced)
                   else f"does not fit {vram_gb} GB" if not forced.fits(vram_gb)
                   else f"cannot render {width}x{height} "
                        f"({width * height} px > {forced.max_pixels})")
            raise ValueError(f"{forced.label} was requested but {why}")
        return forced

    if not candidates:
        # Say WHICH constraint failed. "No model fits" sends the operator
        # looking for a bigger GPU when the real problem is that 720 is not
        # a multiple of 32.
        reasons = []
        for m in REGISTRY.values():
            if not weights_present(m):
                reasons.append(f"{m.label}: not downloaded")
            elif not m.fits(vram_gb):
                reasons.append(f"{m.label}: needs {m.vram_gb} GB")
            elif width % m.dim_multiple or height % m.dim_multiple:
                reasons.append(
                    f"{m.label}: {width}x{height} is off the latent grid "
                    f"(both dimensions must be multiples of {m.dim_multiple})")
            else:
                reasons.append(
                    f"{m.label}: {width * height:,} px exceeds its "
                    f"{m.max_pixels:,} px budget")
        raise ValueError(
            f"no installed model can render {width}x{height}. "
            + "; ".join(reasons))

    wanted = frozenset(needs or ())
    scored = sorted(
        candidates,
        key=lambda m: (-len(wanted & m.strengths), m.cost, m.key))
    chosen = scored[0]
    log.info("genvideo.model_selected", model=chosen.key,
             label=chosen.label, matched=sorted(wanted & chosen.strengths),
             considered=[m.key for m in candidates],
             size=f"{width}x{height}")
    return chosen


def describe_registry(vram_gb: float = 24.0) -> list[dict[str, object]]:
    """Dashboard-shaped view of every model and whether it is usable."""
    rows = []
    for spec in sorted(REGISTRY.values(), key=lambda m: m.cost):
        present = weights_present(spec)
        rows.append({
            "key": spec.key,
            "label": spec.label,
            "model_id": spec.model_id,
            "installed": present,
            "fits_gpu": spec.fits(vram_gb),
            "usable": present and spec.fits(vram_gb),
            "max_pixels": spec.max_pixels,
            "vram_gb": spec.vram_gb,
            "steps": spec.steps,
            "relative_cost": spec.cost,
            "strengths": sorted(spec.strengths),
            "notes": spec.notes,
        })
    return rows
