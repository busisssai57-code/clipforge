"""The blank-video bug, pinned.

Every piece the Generation Studio produced was a flat brown fill. The file
was structurally perfect — correct frame count, correct resolution,
correct duration, valid mp4 — and contained no picture. It passed every
check that existed because every check looked at STRUCTURE and none looked
at PIXELS.

Measured cause, by bisection on identical prompts and seeds:

    704x480  (338k px)  spatial std 45.67  -> real footage
    704x1280 (901k px)  spatial std  2.67  -> blank
    512x896  (459k px)  spatial std 41.34  -> real footage

So it is the PIXEL BUDGET, not the orientation. A resolution outside the
model's trained envelope does not degrade gracefully; it returns an empty
rectangle. The decode arguments that were initially blamed made no
difference at all — the runs with and without them were identical.
"""

from __future__ import annotations

import numpy as np
import pytest

from clipforge.genvideo.providers import (MAX_GEN_PIXELS, ProviderError,
                                          _generation_dims, _write_video)


# --------------------------------------------------- generation budget

@pytest.mark.parametrize("aspect", ["9:16", "16:9"])
def test_generation_stays_inside_the_measured_envelope(aspect):
    w, h = _generation_dims(aspect)
    assert w * h <= MAX_GEN_PIXELS, (
        f"{aspect} generates {w}x{h} = {w * h} px, over the budget that "
        "produced blank frames")


@pytest.mark.parametrize("aspect", ["9:16", "16:9"])
def test_generation_dims_are_on_the_latent_grid(aspect):
    w, h = _generation_dims(aspect)
    assert w % 32 == 0 and h % 32 == 0, f"{w}x{h} is off-grid"


def test_the_budget_is_below_the_size_that_produced_blank_frames():
    """704x1280 = 901,120 px rendered blank. The budget must never allow
    it — that is the entire bug."""
    assert MAX_GEN_PIXELS < 704 * 1280


def test_the_budget_is_at_least_the_largest_proven_good_size():
    """512x896 = 458,752 px is measured-good. Dropping below it would
    trade real quality for caution that was never needed."""
    assert MAX_GEN_PIXELS >= 458_752 - 32 * 32


def test_portrait_is_not_treated_as_the_problem():
    """The first diagnosis was 'portrait is unsupported'. It was wrong —
    512x896 portrait renders fine. If a future change disables portrait
    generation, this fails."""
    w, h = _generation_dims("9:16")
    assert h > w, "9:16 must still generate a portrait frame"


def test_the_two_aspects_are_transposes_of_each_other():
    pw, ph = _generation_dims("9:16")
    lw, lh = _generation_dims("16:9")
    assert (pw, ph) == (lh, lw)


# ------------------------------------------------------- blank guard

def _frames(value, count=9, w=64, h=64, noise=False):
    rng = np.random.default_rng(0)
    if noise:
        return rng.integers(0, 255, size=(count, h, w, 3), dtype=np.uint8)
    return np.full((count, h, w, 3), value, dtype=np.uint8)


def test_a_flat_fill_is_rejected(tmp_path):
    """The exact artefact that shipped: uniform brown, valid geometry."""
    with pytest.raises(ProviderError) as err:
        _write_video(_frames(94), tmp_path / "blank.mp4", 24)
    assert "blank frame" in str(err.value).lower()
    assert not (tmp_path / "blank.mp4").exists(), (
        "a blank video must not be written to disk")


def test_the_rejection_names_resolution_as_the_measured_cause(tmp_path):
    """A guard that fires without explaining sends the next person to the
    wrong suspect — as happened here with the decode arguments."""
    with pytest.raises(ProviderError) as err:
        _write_video(_frames(94), tmp_path / "blank.mp4", 24)
    msg = str(err.value).lower()
    assert "resolution" in msg or "envelope" in msg


def test_real_content_passes_the_guard(tmp_path):
    """Control: the guard must not reject actual footage, or it would
    just be an outage with a good error message."""
    out = tmp_path / "real.mp4"
    _write_video(_frames(0, noise=True), out, 24)
    assert out.is_file() and out.stat().st_size > 0


def test_a_dark_but_real_scene_is_not_mistaken_for_blank(tmp_path):
    """A night shot is legitimately dark. The floor keys on variance, not
    brightness, so low-mean footage with detail must survive."""
    rng = np.random.default_rng(1)
    dark = rng.integers(0, 40, size=(9, 64, 64, 3), dtype=np.uint8)
    out = tmp_path / "dark.mp4"
    _write_video(dark, out, 24)
    assert out.is_file()
