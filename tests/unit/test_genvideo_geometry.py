"""Frame/dimension quantization for local video models, pinned.

Measured against LTX-Video: these constraints are not stylistic, they are
what the model accepts. A plain `seconds * fps` frame count is REJECTED —
the temporal VAE compresses in groups of 8 plus a keyframe — and off-grid
dimensions are rejected rather than resized, because the transformer
patches a latent grid.

The first version of the provider guessed at both and would have failed on
the first real call.
"""

from __future__ import annotations

import pytest

from clipforge.genvideo.presets import PRESETS
from clipforge.genvideo.providers import _latent_frames, _round_to


@pytest.mark.parametrize("value,expected", [
    (704, 704), (1280, 1280), (700, 704), (1281, 1280), (1, 32), (0, 32),
])
def test_dimensions_snap_to_the_latent_grid(value, expected):
    assert _round_to(value, 32) == expected


def test_dimensions_are_never_zero_or_negative():
    """A zero dimension is an ffmpeg filter error much later, in a place
    that says nothing about where it came from."""
    for v in (-100, 0, 3):
        assert _round_to(v, 32) >= 32


@pytest.mark.parametrize("seconds,fps", [
    (6.0, 24), (5.0, 24), (4.0, 30), (5.0, 30), (8.0, 24), (0.1, 24),
])
def test_frame_counts_are_always_eight_n_plus_one(seconds, fps):
    n = _latent_frames(seconds, fps)
    assert (n - 1) % 8 == 0, f"{n} is not 8n+1; the model rejects it"
    assert n >= 9, "fewer than one temporal group cannot be generated"


@pytest.mark.parametrize("seconds,fps", [(6.0, 24), (5.0, 24), (4.0, 30)])
def test_quantization_stays_close_to_the_requested_duration(seconds, fps):
    """Snapping must not silently change the edit. Within one group
    (8 frames = a third of a second at 24fps) is acceptable; more is not.
    """
    got = _latent_frames(seconds, fps) / fps
    assert abs(got - seconds) <= 8.0 / fps, (
        f"asked {seconds}s at {fps}fps, got {got:.2f}s")


def test_every_preset_produces_a_legal_frame_count():
    """The presets are the real inputs — if one of them quantizes to an
    illegal count, generation fails on that creative mode only, which is
    exactly the kind of bug that hides."""
    for name, preset in PRESETS.items():
        n = _latent_frames(preset.shot_seconds, preset.fps)
        assert (n - 1) % 8 == 0, f"{name}: {n} frames is not 8n+1"
        assert n >= 9, f"{name}: {n} frames is under one temporal group"
        drift = abs(n / preset.fps - preset.shot_seconds)
        assert drift <= 8.0 / preset.fps, f"{name}: drifts {drift:.2f}s"


def test_a_sub_group_request_still_yields_one_group():
    """Asking for a fraction of a temporal group cannot be honoured; the
    floor is one group, not zero frames."""
    assert _latent_frames(0.01, 24) == 9


def test_default_guidance_scale_is_within_ltx_safe_range():
    """LTX-Video produces blank brown frames above ~4.0 and ignores the
    prompt below ~2.0.  The config default must stay in the measured safe
    band."""
    from clipforge.config import GenVideoConfig

    default = GenVideoConfig().local_guidance_scale
    assert 2.0 <= default <= 4.0, (
        f"guidance_scale {default} is outside the LTX-Video safe range "
        f"(2.0–4.0); blank or over-saturated frames are likely")


# ----------------------------------------- the in-venv model's envelope

def test_the_local_provider_generates_at_the_spec_envelope_not_the_global_cap():
    """A registry model rendered in THIS venv gets its own envelope too.

    `SubprocessModelProvider` was corrected on 2026-08-20 to generate
    inside the selected model's `max_pixels`; `LocalDiffusersProvider`
    was not, and it is the path every auto-selected model takes. The
    result was that Wan 2.2 -- the only auto-selectable entry in the
    registry -- rendered 512x896 under LTX-Video 0.9's retired 460k cap
    and was upscaled 2.14x to 1080x1920, instead of the 704x1280 its own
    spec authorises.

    A soft picture is not an error, so nothing but this test says it
    happened.
    """
    from clipforge.genvideo.models import REGISTRY
    from clipforge.genvideo.providers import (MAX_GEN_PIXELS,
                                              LocalDiffusersProvider,
                                              _generation_dims)

    for key, spec in REGISTRY.items():
        provider = LocalDiffusersProvider(spec.model_id, spec=spec)
        w, h = _generation_dims("9:16", budget=provider.spec.max_pixels,
                                multiple=provider.spec.dim_multiple)
        assert spec.supports(w, h), f"{key}: {w}x{h} is outside its envelope"
        if spec.max_pixels > MAX_GEN_PIXELS:
            assert w * h > MAX_GEN_PIXELS, (
                f"{key} declares a {spec.max_pixels:,}px envelope but would "
                f"still generate {w}x{h} ({w * h:,}px), inside the retired "
                "global cap")


def test_the_local_provider_keeps_the_global_cap_without_a_spec():
    """The fallback branch has no registry entry, so it keeps the floor.

    `build_router` falls back to the configured model id when nothing in
    the registry fits or is downloaded. There is no measured envelope for
    that id, and inventing a larger one would be exactly the guess the
    460k cap exists to prevent.
    """
    from clipforge.genvideo.providers import (MAX_GEN_PIXELS,
                                              LocalDiffusersProvider,
                                              _generation_dims)

    provider = LocalDiffusersProvider("some/unregistered-model")
    assert provider.spec is None
    w, h = _generation_dims("9:16")
    assert w * h <= MAX_GEN_PIXELS


def test_the_router_hands_the_selected_spec_to_the_local_provider():
    """The wiring itself, not just the provider's behaviour given a spec.

    The envelope fix is worthless if `build_router` keeps constructing
    the provider from two scalars: that is how the model's own numbers
    went missing in the first place.
    """
    import inspect

    from clipforge.genvideo import build_router

    source = inspect.getsource(build_router)
    assert "spec=spec" in source, (
        "build_router must pass the selected ModelSpec to "
        "LocalDiffusersProvider, or the model's envelope and VRAM budget "
        "are dropped on the way in")
