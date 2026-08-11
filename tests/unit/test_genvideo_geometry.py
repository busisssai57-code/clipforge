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
