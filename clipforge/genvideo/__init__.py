"""Text-to-video generation with metered-cloud → local failover.

Public surface is deliberately small: build a router with
``build_router``, ask it for a sequence, get files back.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clipforge.genvideo.presets import PRESETS, Preset, get_preset
from clipforge.genvideo.providers import (GenResult, LocalDiffusersProvider,
                                          Provider, ProviderError,
                                          ProviderUnavailable, QuotaExhausted,
                                          VeoProvider)
from clipforge.genvideo.quota import QuotaLedger
from clipforge.genvideo.router import (GenerationRouter, SequenceResult,
                                       ShotOutcome)
from clipforge.log import get_logger

log = get_logger(__name__)

__all__ = [
    "PRESETS", "Preset", "get_preset", "GenResult", "Provider",
    "ProviderError", "ProviderUnavailable", "QuotaExhausted", "VeoProvider",
    "LocalDiffusersProvider", "QuotaLedger", "GenerationRouter",
    "SequenceResult", "ShotOutcome", "build_router",
]


def build_router(cfg, ws, *, api_key: str | None = None,
                 needs: set[str] | frozenset[str] | None = None,
                 prefer: str | None = None,
                 aspect_ratio: str | None = None) -> GenerationRouter:
    """Router wired from config + workspace.

    Order is the policy: the premium provider is tried first and the local
    one catches everything it drops. Cloud is opt-in — with
    ``[genvideo] use_cloud = false`` (the default) the premium provider is
    not even constructed, so no prompt can leave the machine by accident.

    The LOCAL provider is chosen from the model registry rather than being
    hardcoded to one config value. That registry existed with an unused
    ``select_model`` beside it, so an operator with a stronger model
    downloaded still got whichever id happened to be in config — including
    at resolutions outside that model's measured envelope, where these
    models return blank frames rather than failing. Selection now happens
    where the provider is actually built, which is the only place it can
    take effect.
    """
    from clipforge.genvideo.models import (_generation_dims_for,
                                           select_model)

    def _wan_controls(gv: Any) -> dict[str, Any]:
        """Wan2GP-style knobs, passed to BOTH provider constructions.

        A dict rather than repeated keyword arguments because there are two
        construction sites and the fallback one — taken by any machine
        without downloaded weights — is the branch a new argument is
        forgotten on. That is exactly how the 2026-08-05 seed fix nearly
        shipped covering only the selected-model path.

        `getattr` defaults keep an older config.toml loadable: these fields
        are new, and a config without them must run, not fail to parse.
        """
        return {
            "loras": list(getattr(gv, "loras", []) or []),
            "lora_scale": float(getattr(gv, "lora_scale", 1.0)),
            "step_cache_threshold": float(
                getattr(gv, "step_cache_threshold", 0.0)),
            "quantize": str(getattr(gv, "quantize", "none")),
        }

    gv = cfg.genvideo
    providers: list[Provider] = []
    # Same registry the chokepoint uses — the flag check and the key
    # grant (in the caller, via clipforge.cloud.gemini_key) cannot
    # disagree about what "genvideo cloud" means.
    from clipforge.cloud import cloud_enabled  # noqa: PLC0415

    if cloud_enabled(cfg, "genvideo"):
        providers.append(VeoProvider(api_key, model=gv.cloud_model))

    aspect = aspect_ratio or gv.aspect_ratio
    width, height = _generation_dims_for(aspect)
    try:
        spec = select_model(needs=needs, width=width, height=height,
                            prefer=prefer)
        controls = _wan_controls(gv)
        # A model that CANNOT fit the card unquantized overrides the
        # operator's preference, rather than being loaded at bf16 and
        # dying on an allocation. `quantize` in config is a preference;
        # `requires_quantization` on the spec is a fact about the model.
        if spec.requires_quantization:
            if controls["quantize"] != spec.requires_quantization:
                log.info("genvideo.quantization_forced",
                         model=spec.key, requested=controls["quantize"],
                         applied=spec.requires_quantization,
                         note="this model does not fit the card unquantized")
            controls["quantize"] = spec.requires_quantization
        if spec.interpreter:
            # This model cannot be loaded by the interpreter running this
            # line — not a deployment preference, a dependency conflict
            # (see ModelSpec.interpreter). It is reached only by an
            # explicit `--model`, because `verified=False` keeps it out of
            # automatic selection.
            from clipforge.genvideo.subproc import (  # noqa: PLC0415
                SubprocessModelProvider)

            providers.append(SubprocessModelProvider(
                spec, seed=gv.local_seed, quantize=controls["quantize"]))
        else:
            providers.append(LocalDiffusersProvider(
                spec.model_id, steps=spec.steps,
                guidance_scale=spec.guidance_scale, seed=gv.local_seed,
                **controls))
    except ValueError as exc:
        # No registered model fits (nothing downloaded, or the size is out
        # of every envelope). Fall back to the configured id rather than
        # refusing outright — but say so, because a silent fallback here is
        # how "auto-selection" becomes decoration again.
        log.warning("genvideo.model_selection_failed", error=str(exc)[:300],
                    fallback=gv.local_model_id)
        # The fallback branch matters MORE than the selected one: it is what
        # any machine without downloaded weights takes, and it is the branch
        # the seed fix was nearly missed on in 2026-08-05.
        providers.append(LocalDiffusersProvider(
            gv.local_model_id, steps=gv.local_steps,
            guidance_scale=gv.local_guidance_scale, seed=gv.local_seed,
            **_wan_controls(gv)))

    ledger = QuotaLedger.load(Path(ws.root) / "genvideo_quota.json")
    return GenerationRouter(providers, ledger)
