"""Camera-path behaviour ACROSS a jump-cut seam (S6), pinned.

Three fixes from the pacing panel lived inside ``S6Render._execute`` where
nothing could reach them, so all three were revert-safe: removing any of
them left 433 tests green and QA blessing the result. They are extracted
into ``_remap_path_frames`` precisely so these tests can fail without them.

S4 emits 9:16 crops, so ``_path`` below builds real ones — an earlier draft
of this file used height-clamped square crops, and the rate limit and the
source-bounds clamp then contradicted each other on input the pipeline
cannot actually produce. The degenerate case still gets a test, but only
for the property that matters there (legality), not for easing.
"""

from __future__ import annotations

from clipforge.pacing import TimeMap
from clipforge.schemas.campath import CropFrame
from clipforge.stages.s6_render import (SEAM_MAX_W_STEP_FRAC,
                                        SEAM_MIN_W_STEP_PX,
                                        _remap_path_frames)

FPS = 30.0
SRC_W, SRC_H = 1920, 1080
CAP = max(SEAM_MIN_W_STEP_PX, int(SEAM_MAX_W_STEP_FRAC * SRC_W))


def _even(v: int) -> int:
    return v - (v % 2)


def _path(specs: list[tuple[int, int]]) -> list[CropFrame]:
    """(frame, w) -> centred TRUE 9:16 crop frames, as S4 emits them."""
    out = []
    for frame, w in specs:
        w = _even(w)
        h = _even(int(w * 16 / 9))
        assert h <= SRC_H, f"w={w} cannot be 9:16 inside {SRC_W}x{SRC_H}"
        out.append(CropFrame(frame=frame, x=_even((SRC_W - w) // 2),
                             y=_even((SRC_H - h) // 2), w=w, h=h))
    return out


def test_frames_inside_a_cut_are_dropped_and_survivors_renumbered():
    # Cut 1.0-2.0 s (frames 30-59) out of a 3 s path.
    tmap = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    frames = _path([(f, 540) for f in range(90)])
    out = _remap_path_frames(frames, tmap, FPS, SRC_W, SRC_H)
    idx = [f.frame for f in out]
    assert idx == sorted(idx), "compressed indices must be monotonic"
    assert len(out) == 60, f"expected 60 kept frames, got {len(out)}"
    assert idx[0] == 0 and idx[-1] == 59
    assert len(set(idx)) == len(idx), "duplicate compressed indices"


def test_a_seam_collision_keeps_the_later_source_frame():
    """The first compressed frame after a cut must carry the NEW scene's
    camera position. Keep-first renders it at the OLD position for one
    frame — a flick the panel could see.

    Collisions need a keep boundary that is NOT frame-aligned, which is
    why the CLI now quantizes keeps to the pts grid; this is the
    defence-in-depth path, since S6 is a stage and can be handed raw
    keeps. Cutting at 0.98 s puts frame 29 (t=0.9667, pre-cut) and frame
    60 (t=2.0, post-cut) both on compressed index 29. The two widths
    differ by less than the rate-limit cap, so the heal cannot mask which
    one won.
    """
    assert 540 - 520 < CAP, "widths must differ by less than the seam cap"
    tmap = TimeMap([(0.0, 0.98), (2.0, 3.0)])
    frames = _path([(f, 540) for f in range(30)]
                   + [(f, 520) for f in range(60, 90)])
    out = _remap_path_frames(frames, tmap, FPS, SRC_W, SRC_H)
    by_idx = {f.frame: f for f in out}
    assert len(by_idx) == len(out), "collision was not deduplicated at all"
    # Assert the construction really collides, so this cannot pass by
    # accident on a path where nothing was contested.
    contested = [f for f in frames
                 if tmap.is_kept(f.frame / FPS)
                 and round(tmap.to_compressed(f.frame / FPS) * FPS) == 29]
    assert len(contested) == 2, f"test construction is stale: {contested}"
    assert by_idx[29].w == 520, (
        f"compressed frame 29 kept width {by_idx[29].w}: the collision "
        "resolved keep-FIRST, so the post-cut scene renders one frame at "
        "the pre-cut camera position")


def test_seam_zoom_is_rate_limited():
    """A punch-in cut mid-ramp must ease, not pop. No single compressed
    frame may change crop width by more than the cap."""
    tmap = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    # Pre-cut wide (600), post-cut deep in a punch-in (300): a 300 px step
    # at the seam if nothing rate-limits it.
    frames = _path([(f, 600) for f in range(30)]
                   + [(f, 300) for f in range(60, 90)])
    out = _remap_path_frames(frames, tmap, FPS, SRC_W, SRC_H)
    steps = [abs(b.w - a.w) for a, b in zip(out, out[1:])]
    assert steps, "no frames to compare"
    assert max(steps) > 0, "test construction is stale: no zoom to heal"
    assert max(steps) <= CAP + 2, (           # +2: even-rounding slack
        f"crop width jumped {max(steps)} px in one frame (cap {CAP}) — "
        "the seam heal is not running")
    # And it must actually ARRIVE, not ease forever: the tail is the
    # post-cut framing.
    assert out[-1].w == 300, f"eased path never reached the target: {out[-1]}"


def test_healed_frames_stay_legal_geometry():
    """Rate-limiting must not produce an out-of-bounds or odd crop: those
    render as a green edge or an ffmpeg error, not as a soft zoom."""
    tmap = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    frames = _path([(f, 600) for f in range(30)]
                   + [(f, 300) for f in range(60, 90)])
    for f in _remap_path_frames(frames, tmap, FPS, SRC_W, SRC_H):
        assert f.w % 2 == 0 and f.h % 2 == 0, f"odd crop {f}"
        assert f.x % 2 == 0 and f.y % 2 == 0, f"odd offset {f}"
        assert f.x >= 0 and f.y >= 0
        assert f.x + f.w <= SRC_W, f"crop right edge outside source: {f}"
        assert f.y + f.h <= SRC_H, f"crop bottom edge outside source: {f}"


def test_degenerate_geometry_is_clamped_into_the_source():
    """Defensive: a height-clamped (non-9:16) path must not ease into a
    crop taller than the source. Without the clamp this produced a
    1372x2436 crop inside a 1920x1080 frame — an ffmpeg error, not a pop.
    """
    tmap = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    square = [CropFrame(frame=f, x=420, y=0, w=1080, h=1080)
              for f in range(30)]
    tall = _path([(f, 300) for f in range(60, 90)])
    for f in _remap_path_frames(square + tall, tmap, FPS, SRC_W, SRC_H):
        assert f.w % 2 == 0 and f.h % 2 == 0, f"odd crop {f}"
        assert f.x + f.w <= SRC_W, f"crop right edge outside source: {f}"
        assert f.y + f.h <= SRC_H, f"crop bottom edge outside source: {f}"


def test_an_uncut_path_is_returned_unchanged():
    """No cuts, no collisions, no steps: the heal must be a no-op rather
    than quietly re-centring a path that was already correct."""
    tmap = TimeMap([(0.0, 3.0)])
    frames = _path([(f, 540) for f in range(90)])
    out = _remap_path_frames(frames, tmap, FPS, SRC_W, SRC_H)
    assert [(f.frame, f.x, f.y, f.w, f.h) for f in out] == \
           [(f.frame, f.x, f.y, f.w, f.h) for f in frames]
