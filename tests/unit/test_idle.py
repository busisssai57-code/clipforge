"""The idle gate: when `bta watch` may take the GPU from the operator."""

from __future__ import annotations

import sys

import pytest

from clipforge.idle import (BLIND, IdleGate, PreemptCheck, seconds_since_input,
                            wait_until_idle)

FREE = {"util_pct": 2.0, "decoder_pct": 0.0, "free_vram_gb": 22.0}


def gate(idle=1e6, gpu=None, state=None, **kw) -> IdleGate:
    kw.setdefault("quiet_checks", 1)
    return IdleGate(idle_after_s=300, gpu_busy_pct=40,
                    input_idle=lambda: idle,
                    gpu=lambda: FREE if gpu is None else gpu,
                    notification=lambda: state, **kw)


def test_recent_input_means_busy():
    reason = gate(idle=12.0).reason_busy()
    assert reason and "operator active" in reason


def test_a_gamepad_game_is_caught_by_gpu_load():
    """Controller input does not reset GetLastInputInfo, so a game played
    on a pad reads as a long-idle keyboard. The GPU is what gives it away."""
    reason = gate(gpu={**FREE, "util_pct": 97.0}).reason_busy()
    assert reason and "GPU busy" in reason


def test_watching_a_video_is_not_idle():
    """A film touches no input and barely touches the 3D engine; the
    decoder is where it shows up."""
    reason = gate(gpu={**FREE, "decoder_pct": 38.0}).reason_busy()
    assert reason and "playing video" in reason


def test_a_fullscreen_game_with_no_input_is_not_idle():
    """QUNS_RUNNING_D3D_FULL_SCREEN, even with the GPU between frames."""
    reason = gate(state=3).reason_busy()
    assert reason and "fullscreen game" in reason


def test_an_ordinary_foreground_app_is_not_busy_by_itself():
    """QUNS_APP (7) is the normal desktop state — treating it as busy
    would close the gate for ever."""
    assert gate(state=7).is_idle()


def test_vram_held_by_something_else_keeps_the_gate_shut():
    """A minimized game or a loaded LLM at 0% still owns the memory the
    pipeline needs; starting a window that dies in the VRAM guard helps
    nobody."""
    reason = gate(gpu={**FREE, "free_vram_gb": 3.0}).reason_busy()
    assert reason and "VRAM free" in reason


def test_away_and_machine_quiet_is_idle():
    assert gate(idle=301.0).is_idle()


def test_thresholds_are_inclusive_the_right_way():
    assert gate(idle=300.0, gpu={**FREE, "util_pct": 39.0}).is_idle()
    assert not gate(idle=299.9).is_idle()
    assert not gate(gpu={**FREE, "util_pct": 40.0}).is_idle()


def test_one_quiet_sample_is_not_enough_by_default():
    """A game on a loading screen reads 0% for a moment. The gate opens
    only after quiet_checks consecutive clear readings."""
    g = IdleGate(idle_after_s=300, input_idle=lambda: 1e6,
                 gpu=lambda: FREE, notification=lambda: None)
    first = g.reason_busy()
    assert first and "quiet reading" in first
    assert g.is_idle(), "the second consecutive clear reading must open it"


def test_a_busy_reading_restarts_the_streak():
    readings = iter([1e6, 10.0, 1e6, 1e6])
    g = IdleGate(idle_after_s=300, input_idle=lambda: next(readings),
                 gpu=lambda: FREE, notification=lambda: None)
    assert g.reason_busy() is not None          # first clear: streak 1/2
    assert g.reason_busy() is not None          # input! streak reset
    assert g.reason_busy() is not None          # clear again: streak 1/2
    assert g.is_idle()                          # 2/2


def test_a_session_that_cannot_see_input_fails_closed():
    """The scheduled-task trap: from session 0, GetLastInputInfo answers
    for a desktop nobody uses, so it reads as permanently idle. Reading
    "nobody is here" when the answer is "I cannot see anyone" is the one
    unknown that must NOT open the gate."""
    reason = gate(idle=BLIND).reason_busy()
    assert reason and "console session" in reason


def test_unmeasurable_signals_do_not_block_forever():
    """No desktop session and no nvidia-smi: nobody to protect, and a
    missing tool must not park the clip queue for ever."""
    assert gate(idle=None, gpu=False or None).is_idle()


def test_one_unknown_does_not_mask_the_other():
    assert not gate(idle=None, gpu={**FREE, "util_pct": 90.0}).is_idle()
    assert not gate(idle=5.0, gpu=None).is_idle()


def test_wait_until_idle_returns_when_the_gate_opens():
    readings = iter([5.0, 60.0, 400.0])
    g = gate()
    g.input_idle = lambda: next(readings)
    reasons: list[str] = []
    slept: list[float] = []
    assert wait_until_idle(g, stop=lambda: False, poll_s=7,
                           sleep=slept.append, on_wait=reasons.append)
    assert slept == [7, 7] and len(reasons) == 2


def test_wait_until_idle_honours_stop():
    assert wait_until_idle(gate(idle=0.0), stop=lambda: True,
                           sleep=lambda s: None) is False


# ------------------------------------------------------- mid-job preemption

def test_preempt_fires_on_fresh_input_only():
    """A checkpoint runs between stages, where the GPU may still be
    settling from OUR last stage — so this asks about the operator, not
    the GPU."""
    assert PreemptCheck(within_s=60, input_idle=lambda: 5.0,
                        notification=lambda: None).reason_busy()
    assert PreemptCheck(within_s=60, input_idle=lambda: 120.0,
                        notification=lambda: None).reason_busy() is None


def test_preempt_catches_a_fullscreen_session_with_no_input():
    assert PreemptCheck(within_s=60, input_idle=lambda: 1e6,
                        notification=lambda: 3).reason_busy()


def test_preempt_is_blind_safe():
    assert PreemptCheck(input_idle=lambda: BLIND,
                        notification=lambda: None).reason_busy()


@pytest.mark.skipif(sys.platform != "win32", reason="GetLastInputInfo is Windows-only")
def test_the_real_input_probe_returns_a_sane_number():
    value = seconds_since_input()
    assert value is None or value == BLIND or 0.0 <= float(value) < 60 * 60 * 24
