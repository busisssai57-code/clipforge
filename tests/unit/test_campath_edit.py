"""The director camera: authored keyframes become a renderable path.

Camera geometry is the classic "looks right in the preview, wrong by two
pixels in the render" problem, and §S4's rules are arithmetic — even
coordinates, 9:16, inside the frame — so they are asserted rather than
eyeballed. Every test here is a property the renderer depends on: ffmpeg
rejects an odd crop origin outright, and a crop that hangs off the frame
edge fails the same way.
"""

from __future__ import annotations

import pytest

from clipforge.campath_edit import (EASINGS, CamPathError, KeyFrame,
                                    build_frames, crop_at, default_keyframes,
                                    digest, load_keyframes, parse_keyframes,
                                    sample_at)

SRC_W, SRC_H = 1920, 1080
NTSC = "30000/1001"


def _keys(*specs):
    return parse_keyframes([
        {"t": t, "cx": cx, "cy": cy, "h": h, "easing": e}
        for t, cx, cy, h, e in specs])


# ------------------------------------------------------------- parsing

def test_a_path_needs_at_least_one_keyframe():
    with pytest.raises(CamPathError):
        parse_keyframes([])


def test_keyframes_are_sorted_by_time():
    keys = _keys((3.0, 100, 100, 500, "linear"),
                 (0.0, 200, 200, 600, "linear"))
    assert [k.t for k in keys] == [0.0, 3.0]


def test_two_keyframes_at_the_same_instant_are_refused():
    """Ambiguous, not a jump cut — the renderer would silently take
    whichever happened to sort last."""
    with pytest.raises(CamPathError, match="share t="):
        _keys((1.0, 100, 100, 500, "linear"), (1.0, 900, 900, 500, "linear"))


@pytest.mark.parametrize("bad", [
    {"cx": 1, "cy": 1, "h": 1},                       # no t
    {"t": -1, "cx": 1, "cy": 1, "h": 1},              # negative time
    {"t": 0, "cx": 1, "cy": 1, "h": 0},               # zero height
    {"t": 0, "cx": 1, "cy": 1, "h": 10, "easing": "bounce"},
])
def test_malformed_keyframes_are_refused_with_a_reason(bad):
    with pytest.raises(CamPathError):
        parse_keyframes([bad])


def test_every_documented_easing_is_accepted():
    for name in EASINGS:
        assert parse_keyframes([{"t": 0, "cx": 1, "cy": 1, "h": 10,
                                 "easing": name}])


# ---------------------------------------------------------- sampling

def test_the_camera_holds_before_the_first_keyframe():
    """Extrapolating backwards would invent a move nobody authored."""
    keys = _keys((1.0, 500, 400, 600, "linear"), (2.0, 900, 400, 600, "linear"))
    assert sample_at(keys, 0.0) == (500, 400, 600)


def test_the_camera_holds_after_the_last_keyframe():
    """Extrapolating a push-in past its last key is how you get a crop
    larger than the source at the tail of a clip."""
    keys = _keys((0.0, 500, 400, 900, "linear"), (1.0, 500, 400, 400, "linear"))
    assert sample_at(keys, 99.0) == (500, 400, 400)


def test_linear_interpolation_hits_the_midpoint():
    keys = _keys((0.0, 0, 0, 200, "linear"), (2.0, 1000, 500, 400, "linear"))
    cx, cy, h = sample_at(keys, 1.0)
    assert (round(cx), round(cy), round(h)) == (500, 250, 300)


def test_hold_does_not_move_at_all():
    """A locked-off shot is a directorial choice; interpolating through it
    would make every hold a slow drift."""
    keys = _keys((0.0, 100, 100, 500, "hold"), (4.0, 900, 900, 200, "linear"))
    for t in (0.5, 2.0, 3.9):
        assert sample_at(keys, t) == (100, 100, 500)


def test_easing_stays_inside_the_endpoints():
    for name in ("ease_in", "ease_out", "ease_in_out"):
        keys = _keys((0.0, 0, 0, 100, name), (1.0, 100, 100, 200, "linear"))
        for t in (0.1, 0.3, 0.5, 0.7, 0.9):
            cx, _cy, h = sample_at(keys, t)
            assert 0 <= cx <= 100
            assert 100 <= h <= 200


# ------------------------------------------------------------ geometry

@pytest.mark.parametrize("cx,cy,h", [
    (960, 540, 1080), (0, 0, 1080), (1920, 1080, 1080),
    (-500, -500, 400), (5000, 5000, 300), (960, 540, 99999),
])
def test_every_crop_is_even_and_inside_the_frame(cx, cy, h):
    """ffmpeg rejects an odd crop origin, and a rectangle hanging off the
    edge, whatever the operator dragged."""
    x, y, w, ch = crop_at(cx, cy, h, src_width=SRC_W, src_height=SRC_H)
    assert x % 2 == 0 and y % 2 == 0 and w % 2 == 0 and ch % 2 == 0
    assert x >= 0 and y >= 0
    assert x + w <= SRC_W
    assert y + ch <= SRC_H


def test_the_crop_is_nine_by_sixteen():
    """Any other shape is squeezed when the renderer scales it to
    1080x1920."""
    _x, _y, w, h = crop_at(960, 540, 800, src_width=SRC_W, src_height=SRC_H)
    assert abs(w / h - 9 / 16) < 0.01


def test_a_crop_taller_than_the_source_is_clamped_not_moved():
    """Size is clamped BEFORE the origin. The other order lets an oversized
    crop hang off the right edge no matter where it is placed."""
    x, y, w, h = crop_at(960, 540, 99999, src_width=SRC_W, src_height=SRC_H)
    assert h <= SRC_H and x + w <= SRC_W


def test_a_tiny_crop_is_floored():
    """Past a quarter of source height the upscale to 1920 is visibly
    soft — S4's own punch-in caps at the same depth."""
    _x, _y, _w, h = crop_at(960, 540, 10, src_width=SRC_W, src_height=SRC_H)
    assert h >= 0.25 * SRC_H


def test_a_portrait_source_is_handled():
    """Phone footage: the 9:16 crop is width-limited, not height-limited."""
    x, y, w, h = crop_at(540, 960, 1920, src_width=1080, src_height=1920)
    assert x + w <= 1080 and y + h <= 1920
    assert abs(w / h - 9 / 16) < 0.01


# -------------------------------------------------------------- frames

def test_frame_count_uses_the_exact_rational_fps():
    """A float multiply drifts by a frame over a minute at 30000/1001, and
    the campath is INDEXED by frame number — drift desyncs the camera from
    the picture."""
    keys = default_keyframes(src_width=SRC_W, src_height=SRC_H, duration_s=60)
    frames = build_frames(keys, duration_s=60.0, fps_rational=NTSC,
                          src_width=SRC_W, src_height=SRC_H)
    assert len(frames) == int(60 * 30000 / 1001)


def test_frames_are_numbered_from_zero_without_gaps():
    keys = default_keyframes(src_width=SRC_W, src_height=SRC_H, duration_s=2)
    frames = build_frames(keys, duration_s=2.0, fps_rational="30",
                          src_width=SRC_W, src_height=SRC_H)
    assert [f.frame for f in frames] == list(range(len(frames)))


def test_every_authored_frame_satisfies_the_render_rules():
    keys = _keys((0.0, 200, 200, 1080, "ease_in_out"),
                 (2.0, 1800, 900, 300, "linear"))
    for f in build_frames(keys, duration_s=2.0, fps_rational="30",
                          src_width=SRC_W, src_height=SRC_H):
        assert f.x % 2 == 0 and f.y % 2 == 0 and f.w % 2 == 0 and f.h % 2 == 0
        assert f.x + f.w <= SRC_W and f.y + f.h <= SRC_H


def test_a_push_in_actually_gets_smaller():
    """The headline move. If the interpolation were dropped these would be
    equal and nobody would notice from the frame count."""
    keys = _keys((0.0, 960, 540, 1080, "linear"),
                 (2.0, 960, 540, 400, "linear"))
    frames = build_frames(keys, duration_s=2.0, fps_rational="30",
                          src_width=SRC_W, src_height=SRC_H)
    assert frames[-1].h < frames[0].h


def test_a_zero_length_clip_is_refused():
    keys = default_keyframes(src_width=SRC_W, src_height=SRC_H, duration_s=1)
    with pytest.raises(CamPathError):
        build_frames(keys, duration_s=0.0, fps_rational="30",
                     src_width=SRC_W, src_height=SRC_H)


def test_the_default_path_is_centred_and_locked_off():
    keys = default_keyframes(src_width=SRC_W, src_height=SRC_H, duration_s=5)
    assert all(k.cx == SRC_W / 2 for k in keys)
    frames = build_frames(keys, duration_s=5.0, fps_rational="30",
                          src_width=SRC_W, src_height=SRC_H)
    assert frames[0].x == frames[-1].x and frames[0].h == frames[-1].h


# -------------------------------------------------------------- digest

def test_the_digest_changes_when_the_camera_changes():
    """S6 caches on its params. Without this in them, a re-render with a
    new camera resolves to the CACHED clip and the operator watches their
    edit do nothing — exactly what the trim did before it was caught."""
    a = _keys((0.0, 100, 100, 500, "linear"))
    b = _keys((0.0, 900, 100, 500, "linear"))
    assert digest(a) != digest(b)


def test_the_digest_is_stable_for_the_same_camera():
    a = _keys((0.0, 100, 100, 500, "linear"), (1.0, 200, 200, 400, "hold"))
    b = _keys((1.0, 200, 200, 400, "hold"), (0.0, 100, 100, 500, "linear"))
    assert digest(a) == digest(b)


def test_easing_is_part_of_the_digest():
    """Two paths through the same points with different easing render
    differently, so they must not share a cache entry."""
    a = _keys((0.0, 0, 0, 500, "linear"), (1.0, 100, 0, 500, "linear"))
    b = _keys((0.0, 0, 0, 500, "ease_in"), (1.0, 100, 0, 500, "linear"))
    assert digest(a) != digest(b)


# ---------------------------------------------------------------- io

def test_a_path_file_round_trips(tmp_path):
    import json

    keys = _keys((0.0, 100, 200, 500, "linear"), (1.5, 300, 400, 600, "hold"))
    path = tmp_path / "cam.json"
    path.write_text(json.dumps({"keyframes": [k.as_dict() for k in keys]}),
                    encoding="utf-8")
    assert [k.as_dict() for k in load_keyframes(path)] == \
        [k.as_dict() for k in keys]


def test_a_bare_list_is_also_accepted(tmp_path):
    import json

    path = tmp_path / "cam.json"
    path.write_text(json.dumps([{"t": 0, "cx": 1, "cy": 1, "h": 500}]),
                    encoding="utf-8")
    assert len(load_keyframes(path)) == 1


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(CamPathError, match="no camera path"):
        load_keyframes(tmp_path / "nope.json")


def test_malformed_json_says_so(tmp_path):
    path = tmp_path / "cam.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(CamPathError, match="not valid JSON"):
        load_keyframes(path)
