"""MAR-based active-speaker selection (blueprint §5.1), pinned.

Pure-function tests — no MediaPipe at test time. The landmarker itself is
exercised by the real pipeline run; what these pin is the MATH, the crop
GEOMETRY, and the selection RULE, which are where a silent regression would
hide (the sampler degrades to None on any failure, so a broken formula
would just quietly hand every shot back to the presence heuristic).

Panel round 2026-07-28 rewrote the activity metric: per-SECOND |ΔMAR| over
timestamped samples with detection-dropout gaps excluded, sampled at 15 Hz.
Each finding that motivated a change has a test here that FAILS if the
change is reverted (the standing rule: a fix with no failing test is
indistinguishable from no fix).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clipforge.config import S4Config
from clipforge.stages.s4_tracking import (FACE_CROP_HALF_WIDTH_FRAC,
                                          FACE_CROP_TOP_FRAC,
                                          MAR_MAX_GAP_PERIODS, MAR_SAMPLE_HZ,
                                          MIN_FACE_CROP_PX, _face_crop_bounds,
                                          _mar_activity, _mar_from_landmarks,
                                          _select_shot_subject)

#: Nominal sampling period used throughout — matches production (15 Hz).
_P = 1.0 / MAR_SAMPLE_HZ


def _series(mars: list[float], period: float = _P,
            t0: float = 0.0) -> list[tuple[float, float]]:
    """Evenly-sampled (t, mar) series, the shape production emits."""
    return [(t0 + i * period, m) for i, m in enumerate(mars)]


def _pts(mouth_open: float, width: float = 0.10):
    """478-slot landmark list with only the four MAR indices populated."""
    pts = [SimpleNamespace(x=0.0, y=0.0)] * 478
    pts[61] = SimpleNamespace(x=0.45, y=0.60)             # left corner
    pts[291] = SimpleNamespace(x=0.45 + width, y=0.60)    # right corner
    pts[13] = SimpleNamespace(x=0.50, y=0.58)             # upper inner lip
    pts[14] = SimpleNamespace(x=0.50, y=0.58 + mouth_open)  # lower inner lip
    return pts


# ---------------------------------------------------------------- raw MAR

def test_mar_is_opening_over_width():
    # opening 0.03, width 0.10 -> exactly 0.3
    assert _mar_from_landmarks(_pts(0.03)) == pytest.approx(0.3)
    # closed mouth -> 0
    assert _mar_from_landmarks(_pts(0.0)) == pytest.approx(0.0)


def test_mar_guards_degenerate_geometry():
    assert _mar_from_landmarks(_pts(0.03, width=0.0)) is None
    assert _mar_from_landmarks([]) is None
    assert _mar_from_landmarks(None) is None


# ------------------------------------------------------------- activity

def test_activity_units_are_per_second():
    """The metric is |ΔMAR| per SECOND, not per sample.

    Exactly-representable numbers so this is an equality, not an approx:
    deltas of 0.125 every 0.25 s -> 0.375 change over 0.75 s = 0.5 /s.
    Per-sample would report 0.125 — a tau tuned at one sampling rate would
    silently mean something else at another (the 6 Hz -> 15 Hz bug class).
    """
    s = _series([0.0, 0.125, 0.25, 0.375], period=0.25)
    assert _mar_activity(s, sample_period=0.25) == 0.5


def test_activity_measures_change_not_openness():
    """A held-open mouth (smile) must score ~zero; oscillation must score.

    This is the reason the metric is |ΔMAR| and not openness: speech is
    frame-to-frame CHANGE.
    """
    talking = _series([0.1, 0.5, 0.15, 0.45, 0.1, 0.5])
    smiling = _series([0.48, 0.48, 0.48, 0.48, 0.48, 0.48])
    assert _mar_activity(talking, _P) > 1.0          # /s units at 15 Hz
    assert _mar_activity(smiling, _P) == pytest.approx(0.0)


def test_activity_requires_three_valid_pairs():
    """<3 valid deltas is not evidence — a 2-sample flicker proved nothing,
    and the panel measured exactly that flicker outscoring a talker."""
    assert _mar_activity([], _P) == 0.0
    assert _mar_activity(_series([0.1, 0.9]), _P) == 0.0
    # 3 samples = only 2 pairs -> still not evidence
    assert _mar_activity(_series([0.1, 0.9, 0.1]), _P) == 0.0
    # 4 samples = 3 pairs -> first admissible series
    assert _mar_activity(_series([0.1, 0.9, 0.1, 0.9]), _P) > 0.0


def test_dropout_gaps_contribute_nothing():
    """Deltas across detection dropouts are excluded, not bridged.

    Track detected at t=0..0.2, lost, re-detected at t=2.0..2.2 with a big
    MAR jump across the 1.8 s hole. Bridging that hole injected a fake
    delta (panel: flickering profile face outscored an articulating one
    3.8x purely on cross-gap jumps). With the gap excluded there are only
    4 valid pairs of 0.01 each -> activity is tiny and equals the
    valid-only computation exactly.
    """
    stable = _series([0.30, 0.31, 0.30], t0=0.0) + \
        _series([0.80, 0.81, 0.80], t0=2.0)          # 0.5 jump across gap
    got = _mar_activity(stable, _P)
    expected = (0.01 * 4) / (_P * 4)                  # gap pair excluded
    assert got == pytest.approx(expected)
    # Sanity: if the gap delta WERE bridged the score would be dominated
    # by it — assert we are nowhere near that reverted value.
    bridged = (0.01 * 4 + 0.5) / (_P * 4 + 2.0 - 0.2 - _P)
    assert got < bridged * 0.6


def test_pure_flicker_scores_zero():
    """A face seen only in isolated single detections: every delta crosses
    a dropout gap -> no valid pairs -> 0.0. Before the fix these five
    cross-gap jumps scored as furious lip motion (the 3.8x flicker win)."""
    flicker = [(0.0, 0.1), (1.0, 0.6), (2.0, 0.1),
               (3.0, 0.7), (4.0, 0.1), (5.0, 0.8)]
    assert _mar_activity(flicker, _P) == 0.0


def test_gap_tolerance_is_pinned():
    """MAR_MAX_GAP_PERIODS widened (say 1.6 -> 4) re-admits dropout
    bridges. dt of 2 periods must NOT count; dt of 1.5 periods must."""
    assert MAR_MAX_GAP_PERIODS == 1.6
    two_apart = _series([0.1, 0.6, 0.1, 0.6, 0.1], period=2 * _P)
    assert _mar_activity(two_apart, _P) == 0.0
    ok_jitter = _series([0.1, 0.6, 0.1, 0.6, 0.1], period=1.5 * _P)
    assert _mar_activity(ok_jitter, _P) > 0.0


# ------------------------------------------------------------- selection

def test_the_talker_beats_the_bigger_silent_person():
    """The whole point: presence (area x conf) says track 1, lips say 2.
    tau here is the production default, in production units (/s)."""
    tau = S4Config().mar_activity_tau
    mar = {1: _series([0.30, 0.31, 0.30, 0.31, 0.30]),   # big, still lips
           2: _series([0.10, 0.45, 0.12, 0.40, 0.15])}   # smaller, talking
    presence = {1: 900_000.0, 2: 200_000.0}
    tid, method = _select_shot_subject(mar, presence, tau=tau,
                                       sample_period=_P)
    assert (tid, method) == (2, "mar")


def test_nobody_talking_falls_back_to_presence():
    mar = {1: _series([0.30, 0.30, 0.30, 0.30]),
           2: _series([0.20, 0.201, 0.20, 0.201])}
    presence = {1: 900_000.0, 2: 200_000.0}
    tid, method = _select_shot_subject(mar, presence, tau=0.18,
                                       sample_period=_P)
    assert (tid, method) == (1, "presence")


def test_a_close_race_is_left_to_presence():
    """Two people trading lines inside one shot: MAR must not flip-flop on
    a nose-length lead — the 1.3x margin sends it back to presence."""
    mar = {1: _series([0.1, 0.4, 0.1, 0.4, 0.1]),
           2: _series([0.1, 0.38, 0.12, 0.36, 0.1])}     # ~1.06x behind
    presence = {1: 300_000.0, 2: 900_000.0}
    tid, method = _select_shot_subject(mar, presence, tau=0.18,
                                       sample_period=_P)
    assert method == "presence"
    assert tid == 2


def test_a_clear_lead_is_decided_by_mar():
    """Margin UPPER bound (panel: margin 1.3 -> 2.0 survived every test).

    Track 1 leads track 2 by ~1.5x — decisively above 1.3, below 2.0. A
    stricter margin quietly hands real talkers back to presence; this
    fails if the margin creeps past 1.5.
    """
    mar = {1: _series([0.1, 0.55, 0.1, 0.55, 0.1, 0.55]),   # deltas 0.45
           2: _series([0.1, 0.40, 0.1, 0.40, 0.1, 0.40])}   # deltas 0.30
    presence = {1: 200_000.0, 2: 900_000.0}
    tid, method = _select_shot_subject(mar, presence, tau=0.18,
                                       sample_period=_P)
    assert (tid, method) == (1, "mar")


def test_tau_boundary_is_inclusive():
    """Activity EXACTLY tau selects MAR (>= not >). Exactly-representable
    construction: 0.375 change over 0.75 s = 0.5 /s, tau = 0.5."""
    mar = {1: _series([0.0, 0.125, 0.25, 0.375], period=0.25)}
    tid, method = _select_shot_subject(mar, {1: 1.0, 2: 5.0}, tau=0.5,
                                       sample_period=0.25)
    assert (tid, method) == (1, "mar")


def test_no_faces_at_all_is_pure_presence():
    tid, method = _select_shot_subject({}, {7: 1.0}, tau=0.18,
                                       sample_period=_P)
    assert (tid, method) == (7, "presence")


def test_empty_series_entries_are_ignored():
    """A track whose face was never landmarked must not shadow a real one."""
    mar = {1: [], 2: _series([0.1, 0.5, 0.1, 0.5, 0.1])}
    tid, method = _select_shot_subject(mar, {1: 9.0, 2: 1.0}, tau=0.18,
                                       sample_period=_P)
    assert (tid, method) == (2, "mar")


# ------------------------------------------------------- crop geometry

def test_face_crop_is_top_of_box():
    """45% of box height, not the full box — legs don't have lips."""
    assert FACE_CROP_TOP_FRAC == 0.45
    y1, y2, x1, x2 = _face_crop_bounds((100, 200, 300, 600))
    assert (y1, y2) == (200, 200 + int(400 * 0.45))
    assert (x1, x2) == (100, 300)                # no head anchor: full width


def test_face_crop_narrows_around_head_anchor():
    """MAR-3: with a head keypoint the crop is ±30% of box width around it,
    so a neighbour's face at the box edge can't be credited to this track."""
    assert FACE_CROP_HALF_WIDTH_FRAC == 0.30
    y1, y2, x1, x2 = _face_crop_bounds((100, 200, 300, 600), head_x=170)
    half = int(200 * 0.30)                       # 60 px
    assert (x1, x2) == (170 - half, 170 + half)
    assert x2 - x1 < 200                         # strictly narrower than box


def test_face_crop_half_width_floors_at_min_px():
    """Tiny boxes keep at least MIN_FACE_CROP_PX/2 half-width and clamp to
    the box, so narrowing can't starve the landmarker below its floor."""
    assert MIN_FACE_CROP_PX == 40
    y1, y2, x1, x2 = _face_crop_bounds((100, 200, 130, 300), head_x=115)
    assert x1 == 100 and x2 == 130               # clamped to the 30px box
    _, _, cx1, cx2 = _face_crop_bounds((0, 0, 50, 200), head_x=25)
    assert cx2 - cx1 >= 40                       # floor beats 30% of 50px


def test_sample_rate_is_pinned():
    """15 Hz, measured: 6 Hz starved series below the 3-pair floor (a
    talker scored 0.0 and a back of a head was framed) and flipped
    talker-ordering on 10/24 shots vs 4/24 at 15 Hz."""
    assert MAR_SAMPLE_HZ == 15.0


def test_production_tau_default_is_per_second():
    """0.18 /s — the old 0.03 was per-SAMPLE units; leaving it while the
    metric moved to /s would make every shot trivially clear tau."""
    assert S4Config().mar_activity_tau == 0.18
