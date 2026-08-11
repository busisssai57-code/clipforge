"""Self-correcting repair planning, pinned.

The dangerous failure mode for a repair loop is not that it fails to fix
something â€” it is that it retries a CODE bug until something passes by
luck, and reports green. The first test in this file is the one that
matters; the rest pin that each mechanical fault gets the right edit.
"""

from __future__ import annotations

import pytest

from clipforge.repair import (EDGE_NUDGE_S, MAX_ATTEMPTS,
                              MAX_GAIN_CORRECTION_LU, MAX_TOTAL_NUDGE_S,
                              STRUCTURAL_CHECKS, plan_repair)
from clipforge.schemas.qa import QACheck
from clipforge.stages.s7_qa import MAX_CLIP_S, MIN_CLIP_S, SHIP_TP_CEILING

WIN_START, WIN_END = 100.0, 140.0
SOURCE_S = 2900.0
TARGET_I = -14.0


def _check(name: str, measured: str = "", expected: str = "") -> QACheck:
    return QACheck(name=name, severity="fail", passed=False,
                   measured=measured, expected=expected)


def _plan(*checks: QACheck, attempt: int = 0, start: float = WIN_START,
          end: float = WIN_END, source: float = SOURCE_S, **kw):
    return plan_repair(list(checks), window_start=start, window_end=end,
                       source_duration=source, attempt=attempt,
                       target_i=TARGET_I, **kw)


# ---------------------------------------------------- the important one

@pytest.mark.parametrize("name", sorted(STRUCTURAL_CHECKS))
def test_structural_failures_are_never_repaired(name):
    """A geometry or integrity failure means the CODE is wrong. Retrying
    it with different parameters is how a bug becomes a green tick."""
    plan = _plan(_check(name, "wrong", "right"))
    assert not plan.repairable
    assert name in plan.reason


def test_a_structural_failure_blocks_even_alongside_a_fixable_one():
    """Mixed rejection: the fixable one must not license a retry that
    would also re-run the broken code path."""
    plan = _plan(_check("duration-bounds", "72.00s"),
                 _check("geometry", "1920x1080", "1080x1920"))
    assert not plan.repairable
    assert "geometry" in plan.reason


def test_the_attempt_budget_is_a_hard_stop():
    fixable = _check("duration-bounds", "72.00s")
    assert _plan(fixable, attempt=MAX_ATTEMPTS - 1).repairable
    assert not _plan(fixable, attempt=MAX_ATTEMPTS).repairable
    assert not _plan(fixable, attempt=MAX_ATTEMPTS + 5).repairable


def test_an_unknown_check_is_not_silently_ignored():
    """A new QA check with no remedy must stop the loop, not fall through
    to a no-op 're-render' that changes nothing and burns a GPU minute."""
    plan = _plan(_check("some-future-check", "bad"))
    assert not plan.repairable
    assert "some-future-check" in plan.reason


def test_planning_is_deterministic():
    checks = [_check("duration-bounds", "72.00s"),
              _check("loudness-target", "-18.30 LUFS")]
    a = plan_repair(checks, window_start=WIN_START, window_end=WIN_END,
                    source_duration=SOURCE_S, attempt=0, target_i=TARGET_I)
    b = plan_repair(checks, window_start=WIN_START, window_end=WIN_END,
                    source_duration=SOURCE_S, attempt=0, target_i=TARGET_I)
    assert a == b


# ---------------------------------------------------------- duration

def test_an_overlong_clip_is_trimmed_from_the_tail():
    """The hook is at the start â€” the ranking picked this window for how
    it opens, so the tail is what gives."""
    plan = _plan(_check("duration-bounds", "72.00s"), start=100.0, end=172.0)
    assert plan.repairable
    assert plan.window_start == 100.0, "the opening must not move"
    assert plan.window_end < 172.0
    assert MIN_CLIP_S <= (plan.window_end - plan.window_start) <= MAX_CLIP_S
    assert plan.remedies[0].action == "shrink-window"


def test_shrinking_scales_by_yield_not_by_raw_overshoot():
    """With jump-cuts the rendered clip is SHORTER than its window, so the
    correction is a ratio. Subtracting the overshoot from the window (the
    first version of this code) over-trims: a 90 s window yielding 72 s
    would have been cut to 79 s and rendered ~63 s â€” still over.
    """
    plan = _plan(_check("duration-bounds", "72.00s"), start=0.0, end=90.0)
    assert plan.repairable
    window = plan.window_end - plan.window_start
    # aim 60 s of OUTPUT at a 0.8 yield -> a 75 s window
    assert window == pytest.approx(75.0, abs=0.1)
    assert window * (72.0 / 90.0) == pytest.approx(60.0, abs=0.5)


def test_a_short_clip_is_extended():
    plan = _plan(_check("duration-bounds", "24.00s"), start=100.0, end=124.0)
    assert plan.repairable
    assert plan.window_end > 124.0
    assert (plan.window_end - plan.window_start) >= MIN_CLIP_S


def test_a_source_too_short_to_reach_the_floor_gives_up():
    """Honest refusal beats a clip that will fail the same check again."""
    plan = _plan(_check("duration-bounds", "12.00s"),
                 start=0.0, end=12.0, source=12.0)
    assert not plan.repairable
    assert "floor" in plan.reason


def test_an_unparseable_measurement_gives_up_rather_than_guessing():
    plan = _plan(_check("duration-bounds", "not a number"))
    assert not plan.repairable


# ------------------------------------------------------------- splice

def test_a_splice_mismatch_drops_the_jump_cuts():
    """Pacing is editorial; a correct clip is not. When the splice
    arithmetic disagrees with the file, ship it unspliced."""
    plan = _plan(_check("splice-duration-integrity", "30.51s measured"))
    assert plan.repairable
    assert plan.drop_jumpcuts
    assert plan.remedies[0].action == "drop-jumpcuts"


# -------------------------------------------------------------- audio

def test_a_peak_violation_asks_for_more_headroom():
    plan = _plan(_check("true-peak-ceiling", "-0.20 dBTP"))
    assert plan.repairable
    assert plan.param_overrides["loudness_tp"] < SHIP_TP_CEILING


def test_loudness_is_retargeted_by_the_measured_shortfall():
    """Landed 4.3 LU quiet -> ask for 3.0 louder (clamped), not 4.3:
    loudnorm cannot beat the source's headroom and over-asking just
    trades LUFS for clipping."""
    plan = _plan(_check("loudness-target", "-18.30 LUFS"))
    assert plan.repairable
    got = plan.param_overrides["loudness_i"]
    assert got == pytest.approx(TARGET_I + MAX_GAIN_CORRECTION_LU)
    assert got > TARGET_I, "a quiet clip must be retargeted LOUDER"


def test_a_small_shortfall_is_corrected_exactly():
    plan = _plan(_check("loudness-target", "-15.50 LUFS"))
    assert plan.param_overrides["loudness_i"] == pytest.approx(-12.5)


def test_a_headroom_limited_source_is_not_chased():
    """Already on target but failing for another reason: do not emit a
    zero-sized 'correction' that re-renders identical bytes."""
    plan = _plan(_check("loudness-target", "-14.02 LUFS"))
    assert not plan.repairable
    assert "headroom" in plan.reason


# ------------------------------------------------------- dead content

def test_black_frames_shift_the_window_off_the_dead_region():
    plan = _plan(_check("black-frames", "black runs ['1.2']s"))
    assert plan.repairable
    assert plan.window_start == pytest.approx(WIN_START + EDGE_NUDGE_S)
    assert (plan.window_end - plan.window_start) == pytest.approx(
        WIN_END - WIN_START), "the duration must survive the shift"


def test_the_window_may_not_wander_away_from_the_chosen_moment():
    """Repair must not turn into a search. Past the total-nudge budget it
    is no longer the clip the ranking picked."""
    drifted = WIN_START + MAX_TOTAL_NUDGE_S
    plan = _plan(_check("silence", "silent runs ['6.0']s"),
                 start=drifted, end=drifted + 40.0,
                 original_start=WIN_START, original_end=WIN_END)
    assert not plan.repairable
    assert "ranking selected" in plan.reason


def test_a_shift_cannot_run_past_the_end_of_the_source():
    """A window at the very tail has nowhere to step to. The source ends
    0.5 s after the window, which is less than one nudge."""
    plan = _plan(_check("black-frames", "black runs ['1.2']s"),
                 start=2850.0, end=2890.5, source=2891.0)
    assert not plan.repairable
    assert "black-frames" in plan.reason
    # Control: with room to move, the same rejection IS repairable, so
    # this test is measuring the source bound and not something else.
    ok = _plan(_check("black-frames", "black runs ['1.2']s"),
               start=2850.0, end=2890.5, source=2900.0)
    assert ok.repairable


# ------------------------------------------------------------ re-mux

def test_a_stream_duration_disagreement_just_re_renders():
    plan = _plan(_check("av-duration-match", "v=30.1 a=29.2"))
    assert plan.repairable
    assert plan.remedies[0].action == "re-render"
    assert plan.window_start == WIN_START and plan.window_end == WIN_END
    assert not plan.param_overrides


def test_combined_faults_produce_combined_remedies():
    plan = _plan(_check("duration-bounds", "72.00s"),
                 _check("loudness-target", "-18.30 LUFS"))
    assert plan.repairable
    actions = {r.action for r in plan.remedies}
    assert actions == {"shrink-window", "nudge-loudness-target"}
    assert "shrink-window" in plan.summary()
