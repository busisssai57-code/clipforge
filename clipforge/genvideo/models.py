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
    #: Weight quantization this model REQUIRES to fit the card at all.
    #: "" means it runs unquantized. Unlike the [genvideo] quantize knob,
    #: which is an operator preference, this is a property of the model:
    #: a 22B transformer is ~38 GB at bf16 and simply does not fit 24 GB.
    requires_quantization: str = ""
    #: Failure modes of THIS checkpoint, appended to the preset's own
    #: `avoid` list. Separate because the preset's list is the operator's
    #: editorial choice and this is a property of the weights.
    default_negative: str = ""
    #: Whether the model generates sound as well as picture. A model that
    #: claims to and then returns none is a broken run, not a quiet one.
    generates_audio: bool = False
    #: Appended when an audio model is given a prompt that says nothing
    #: about sound. MEASURED, not a precaution: the pipeline's own
    #: 50-word style block silences LTX-2.5 (-52.8 LUFS against -18.7 for
    #: the bare brief), and one plain sentence about the scene's sound
    #: brings it back louder than either (-12.8).
    audio_prompt_hint: str = ""
    #: Path, relative to the repo, of a Python that can load this model
    #: when THIS interpreter cannot. Empty for every model that runs in
    #: the pipeline venv. It is a property of the model's dependency
    #: graph, not a deployment detail: LTX-2.5 needs diffusers 0.40,
    #: which needs huggingface-hub>=1.23, which whisperx forbids.
    interpreter: str = ""
    #: False until a real render on THIS machine has been inspected.
    #: An unverified model is never auto-selected — every number below it
    #: is from a model card, and this project has been burned twice by
    #: envelopes that were published rather than measured.
    verified: bool = True
    notes: str = ""

    def fits(self, vram_gb: float) -> bool:
        return self.vram_gb <= vram_gb

    def supports(self, width: int, height: int) -> bool:
        return (width * height <= self.max_pixels
                and width % self.dim_multiple == 0
                and height % self.dim_multiple == 0)


# LTX-Video 0.9 was RETIRED on 2026-08-13, replaced by LTX-2.5 at the
# operator's request. Its measured envelope is recorded here rather than
# deleted, because it was measured on THIS machine and is the only such
# number the project has for an LTX VAE: 512x896 rendered, 704x1280 came
# back blank, so max_pixels was 460_000 at 10 GB peak with CPU offload,
# 30 steps, guidance 3.0. If LTX-2.5 does not work out on Ampere, that is
# the spec to restore.

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

#: LTX-2.5: 22B audio+video DiT, the current Lightricks open-weights
#: release (August 2026). It HAS rendered on this machine now — but not
#: in this environment, which is why `verified=False` still stands. See
#: "the environment is the blocker" below; the flag means "the pipeline
#: cannot select this", and that is still true.
#:
#: Why it needs quantization rather than merely benefiting from it: the
#: bf16 transformer weights are 35.37 GB (the checkpoint index's own
#: total_size) against 24 GB of card. NF4 brings that to 9.13 GB —
#: MEASURED on the real shards on 2026-08-18, not estimated. There is no
#: unquantized path on this hardware.
#:
#: MEASURED 2026-08-20, first real render: 512x896, 25 frames at 24 fps,
#: 8 steps, CFG 1.0, seed 1234 — 61.7 s of generation after a 92.5 s
#: load, **13.27 GB peak VRAM**, and a re-run at the same seed produced
#: byte-identical frames (sha256 over frames 0, 12 and 24). The
#: Determinism Law holds on this route. What is NOT established is the
#: envelope: 512x896 is 459k pixels against the 921k this spec declares
#: from the card, so `max_pixels` and `vram_gb` stay as they were — the
#: measurement below them does not license the numbers above them.
#:
#: **Transformer-only quantization does not fit this model.** The Gemma4
#: text encoder is 23.92 GB at bf16 and the connectors are 6.34 GB, so
#: quantizing only the transformer leaves ~30 GB of bf16 companions for a
#: 24 GB card. NF4 across transformer + text_encoder + connectors gives
#: 9.51 / 7.50 / 1.59 GB, which is what fit. `_quantization_config()`
#: maps `transformer` alone; this model would need that widened.
#: `enable_model_cpu_offload()`, not `enable_sequential_cpu_offload()` —
#: accelerate's per-submodule hooks raise `Cannot copy out of meta
#: tensor` on a bitsandbytes 4-bit parameter.
#:
#: **The environment is the blocker, and it is not one an upgrade fixes.**
#: The checkpoint wants diffusers 0.40.x and `Gemma4UnifiedForConditional
#: Generation`; both exist — diffusers 0.40.0 and transformers 5.15.1 —
#: and the render above was made with them in `.venv-ltx25`. They cannot
#: come into the pipeline venv: diffusers 0.40 requires
#: `huggingface-hub>=1.23`, and whisperx pins `huggingface-hub<1.0`.
#: S1 is whisperx, so LTX-2.5 and transcription cannot share one
#: interpreter today. Wiring this route live means a second interpreter
#: behind the provider, which is an architecture decision, not a
#: dependency bump.
#:
#: The earlier note here claimed connectors, duration_head and vocoder
#: come from "an `ltx2` package that is not installed". That was WRONG:
#: no such package exists on PyPI, and diffusers 0.40.0 ships all three
#: under `diffusers.pipelines.ltx2` — the `"ltx2"` library name in
#: `model_index.json` resolves to the pipeline's own submodule. The
#: 0.39.0 meta-tensor death was real and is unchanged.
#:
#: `diffusion_decoder` is reported as unexpected by `LTX2Pipeline` and
#: ignored. That is correct, not damage: it belongs to
#: `LTX2DiffusionDecodePipeline`, a separate second-stage decode.
#:
#: Repeatability MEASURED 2026-08-20: the same seeded generation ten
#: times back to back — **10/10 succeeded, all ten frame-0 hashes
#: identical to each other and to the first render**. Peak VRAM was
#: 13.27 GB on every run, not a range; load 81-103 s, generation
#: 56-61 s, 161-188 s wall per clip. One earlier load DID die with a
#: Windows access violation at 79% of the connector weights (1 failure
#: in 13 attempts), and free RAM does not explain it — one of the ten
#: succeeded with 12.9 GB free, below the ~19 GB available when the
#: failure happened. Peak host commit is 54-59 GB against 32 GB of RAM,
#: which is the suspected variable and is not proven.
#:
#: The int8 and NVFP4 checkpoints Lightricks ships are NOT usable here:
#: int8-convrot is ComfyUI-only, and NVFP4 needs Blackwell — this is an
#: Ampere RTX 3090 (sm_86) with no native FP4/FP8 tensor cores, so 4-bit
#: is a storage format that is dequantized per layer. It buys VRAM, not
#: speed, and this model will be far slower than LTX 0.9 either way.
LTX_25 = ModelSpec(
    key="ltx25",
    model_id="Lightricks/LTX-2.5-Diffusers",
    label="LTX-2.5 22B (distilled)",
    #: PROVISIONAL. The VAE's compression ratios could not be read — the
    #: repo is gated and the config download 403s until the licence is
    #: accepted — so this is the published 720p target, not a measured
    #: envelope. Same status the Wan 2.2 entry carries.
    max_pixels=720 * 1280,
    #: NF4 transformer + model CPU offload. NOT measured; the 0.9 entry's
    #: 10 GB was, and the difference is the point of `verified`.
    vram_gb=16.0,
    #: MEASURED 2026-08-20, and the reason the first pieces looked like
    #: melting wax. This said 8 steps at CFG 1.0 — "the distilled
    #: schedule from the model card" — and nothing here had ever run the
    #: two schedules against each other. diffusers' own LTX2 example uses
    #: 30 steps at guidance 3.0, and on the same prompt and seed that is
    #: not a subtle difference: sharpness (Laplacian variance of the
    #: middle frame) 110.9 -> 134.8, spatial std 41.8 -> 58.5, and by eye
    #: the last frame goes from blobby fur and a smeared human to real
    #: coat texture, legible market stalls and correct anatomy. The white
    #: speckle over the sky disappears with it. Costs 2.7x the time
    #: (104.7 s -> 283.9 s for 49 frames), which is what the picture is
    #: worth.
    steps=30,
    guidance_scale=3.0,
    #: A 22B model dequantized per layer under CPU offload. The number is
    #: a placeholder ordering hint, not a measurement.
    cost=8.0,
    strengths=frozenset({"photoreal", "human", "detail", "cinematic",
                         "motion", "hero", "landscape", "audio"}),
    requires_quantization="nf4",
    #: The vendor's own example negates exactly these, and they name the
    #: failures the operator reported.
    default_negative="worst quality, inconsistent motion, blurry, jittery, "
                     "distorted",
    #: `audio_in_channels: 128` and a vocoder are in this checkpoint's own
    #: config; the pipeline returns `(video, audio)`.
    generates_audio=True,
    #: Deliberately plain and content-free. It has to say "there IS
    #: sound here" without inventing what the sound is — a hint naming
    #: voices would put a crowd in an empty desert.
    #:
    #: It is NOT a complete fix, and the measurements say where it stops.
    #: All at 704x1280, 30 steps, seed 1234, delivered LUFS:
    #:
    #:   2 s  bare brief                          -18.7   audible
    #:   2 s  brief + the preset's 50-word style  -52.8   silent
    #:   2 s  the same, plus this hint            -12.8   audible
    #:   6 s  short prompt                        -18.0   audible
    #:   6 s  brief + style                       -inf    silent
    #:   6 s  brief + style + this hint           -inf    silent
    #:
    #: So the house style suppresses this model's audio, length makes the
    #: suppression worse, and the hint overcomes it at two seconds and
    #: not at six. Raising `audio_guidance_scale` to 12 did not rescue it
    #: either (-44.5 dBFS). The provider warns on a silent track rather
    #: than pretending; a six-second shot in a styled preset may still
    #: come back mute.
    audio_prompt_hint="The scene has its own natural sound.",
    #: Its own interpreter. `SubprocessModelProvider` runs the worker
    #: there; `CLIPFORGE_ALT_PYTHON` overrides the location.
    interpreter=".venv-ltx25/Scripts/python.exe",
    verified=False,
    notes=("22B audio+video DiT. Gated on Hugging Face: accept the licence "
           "at huggingface.co/Lightricks/LTX-2.5-Diffusers first. MEASURED "
           "on disk: 116.52 GB fetched, 72.19 GB referenced — the repo "
           "ships the transformer in TWO shardings and connectors twice, "
           "so 44.33 GB of what a naive `transformer/*` allow-pattern "
           "pulls is referenced by no index (deleted 2026-08-20; a "
           "re-fetch brings them back). Renders here only under a "
           "SEPARATE interpreter (diffusers 0.40 + transformers 5.x, which "
           "whisperx's huggingface-hub<1.0 pin keeps out of the pipeline "
           "venv), needs NF4 on the text encoder and connectors as well as "
           "the transformer, and is much slower than LTX 0.9. Twelve "
           "renders: 512x896x25 in ~58 s at 13.27 GB VRAM, every one "
           "byte-identical at the same seed. "
           "Envelope numbers are still the card's — select it explicitly "
           "with --model ltx25."),
)

REGISTRY: dict[str, ModelSpec] = {
    m.key: m for m in (WAN22_TI2V_5B, LTX_25)
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
        # Unverified models are opt-in only. Auto-selecting one would put
        # a published envelope in charge of a render, which is how blank
        # frames ship silently — `prefer` bypasses this deliberately.
        if not spec.verified:
            continue
        out.append(spec)
    return out


def hf_cache_dir(model_id: str):
    """Where this model's blobs live, without importing torch or hitting
    the network.

    Shared with `preflight.check_ranking_weights`, which asks the same
    question about S3's vision-language model: two copies of this path
    arithmetic would drift the moment HF changes its layout.
    """
    from pathlib import Path

    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        root = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        root = Path.home() / ".cache" / "huggingface" / "hub"
    return root / ("models--" + model_id.replace("/", "--"))


def weights_present(spec: ModelSpec) -> bool:
    """Whether the model is downloaded, without importing torch.

    Checks the HF cache directly: constructing a pipeline to find out
    would download tens of gigabytes as a side effect of a availability
    check.
    """
    folder = hf_cache_dir(spec.model_id)
    if not folder.is_dir():
        return False
    snapshots = folder / "snapshots"
    if not snapshots.is_dir():
        return False
    # "Any file at all" was the old test, and it is how 16 MB of config
    # and tokenizer JSON reported a 7 GB model as downloaded: selection
    # said installed, the run started, and S3 blocked mid-pipeline
    # fetching weights. MEASURED on
    # models--Qwen--Qwen2.5-VL-7B-Instruct-AWQ, 2026-08-20.
    #
    # A download in flight also leaves `.incomplete` blobs, so those are
    # checked here too — which is what the old comment claimed and the
    # old code never did.
    if any((folder / "blobs").glob("*.incomplete")):
        return False
    return any(_snapshot_has_weights(s) for s in snapshots.iterdir()
               if s.is_dir())


#: Extensions a checkpoint's actual weights arrive in. Config, tokenizer
#: and README files are not weights and must not vote.
_WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".onnx")


def _snapshot_has_weights(snapshot: Path) -> bool:
    """Whether this snapshot holds every weight file it says it has.

    Where a shard index exists it is the authority: it names each shard,
    so a half-fetched sharded model (1.6 GB of a 16 GB checkpoint, which
    is the state the plain Qwen VL cache was in) is missing files the
    index lists and answers False. Without an index, one weight file of
    non-zero size is the most that can be asked.
    """
    import json as _json  # noqa: PLC0415 - only this check parses indexes

    found_index = False
    for index in snapshot.rglob("*.index.json"):
        try:
            weight_map = _json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        except Exception:  # noqa: BLE001 - an unreadable index is not proof
            continue
        found_index = True
        for name in set(weight_map.values()):
            shard = index.parent / name
            if not shard.is_file() or shard.stat().st_size == 0:
                return False
    if found_index:
        return True
    return any(f.is_file() and f.stat().st_size > 0
               for f in snapshot.rglob("*")
               if f.suffix in _WEIGHT_SUFFIXES)


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
            # An unverified model that is present and fits is allowed
            # THROUGH an explicit request — that request is how it gets
            # verified in the first place — but it says so.
            if (not forced.verified and weights_present(forced)
                    and forced.fits(vram_gb) and forced.supports(width, height)):
                log.warning("genvideo.unverified_model", model=forced.key,
                            note="envelope is from the model card, not "
                                 "measured here; inspect the output")
                return forced
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
