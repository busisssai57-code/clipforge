"""The idle gate: when `bta watch` may take the GPU from the operator."""

from __future__ import annotations

import sys

import pytest

from clipforge.idle import (BLIND, IdleGate, PreemptCheck, seconds_since_input,
                            wait_until_idle)

FREE = {"util_pct": 2.0, "decoder_pct": 0.0, "free_vram_gb": 22.0}
QUIET = {"cpu_pct": 4.0, "free_ram_gb": 20.0}


def gate(idle=1e6, gpu=None, state=None, load=None, **kw) -> IdleGate:
    kw.setdefault("quiet_checks", 1)
    return IdleGate(idle_after_s=300, gpu_busy_pct=40,
                    input_idle=lambda: idle,
                    gpu=lambda: FREE if gpu is None else gpu,
                    notification=lambda: state,
                    load=lambda: QUIET if load is None else load, **kw)


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
    # Every probe injected: this is about the streak rule, and a gate
    # left reading the real machine fails whenever something else on the
    # box is busy — which, with `bta watch` installed, is often.
    g = IdleGate(idle_after_s=300, input_idle=lambda: 1e6,
                 gpu=lambda: FREE, notification=lambda: None,
                 load=lambda: QUIET)
    first = g.reason_busy()
    assert first and "quiet reading" in first
    assert g.is_idle(), "the second consecutive clear reading must open it"


def test_a_busy_reading_restarts_the_streak():
    readings = iter([1e6, 10.0, 1e6, 1e6])
    g = IdleGate(idle_after_s=300, input_idle=lambda: next(readings),
                 gpu=lambda: FREE, notification=lambda: None,
                 load=lambda: QUIET)
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


# ------------------------------------------------- heavy work that is not ours
#
# The GPU signals miss a training run on the CPU, a compile, or a game's
# simulation thread — and on Windows nvidia-smi cannot attribute VRAM per
# process, so "someone else is holding the card" has to come from the
# total. Measured 2026-09-26: eight external burners read as 80% and the
# gate closed; they exited and it opened.

def test_someone_elses_cpu_work_keeps_the_gate_shut():
    reason = gate(load={"cpu_pct": 78.0, "free_ram_gb": 20.0}).reason_busy()
    assert reason and "using the CPU" in reason


def test_a_machine_low_on_memory_waits():
    """A big model loading is RAM before it is anything else, and this
    pipeline needs headroom of its own."""
    reason = gate(load={"cpu_pct": 3.0, "free_ram_gb": 1.5}).reason_busy()
    assert reason and "RAM free" in reason


def test_our_own_load_does_not_close_the_gate_on_us():
    """system_load subtracts this process tree; the gate must trust that
    rather than adding a second, private idea of who is busy."""
    assert gate(load={"cpu_pct": 6.0, "free_ram_gb": 12.0}).is_idle()


def test_an_unreadable_load_probe_does_not_park_the_queue():
    assert gate(load=False or None).is_idle()


def test_the_cpu_threshold_is_configurable():
    busy = {"cpu_pct": 40.0, "free_ram_gb": 20.0}
    assert gate(load=busy, cpu_busy_pct=60.0).is_idle()
    assert not gate(load=busy, cpu_busy_pct=30.0).is_idle()


def test_a_running_job_yields_when_something_heavy_starts():
    """The operator does not have to touch the keyboard: a game launched
    by remote play, or a scheduled training run, takes the machine too."""
    hot = PreemptCheck(within_s=60, input_idle=lambda: 1e6,
                       notification=lambda: None,
                       load=lambda: {"cpu_pct": 90.0, "free_ram_gb": 20.0},
                       gpu=lambda: FREE)
    assert hot.reason_busy() and "CPU" in hot.reason_busy()

    grabbed = PreemptCheck(within_s=60, input_idle=lambda: 1e6,
                           notification=lambda: None,
                           load=lambda: QUIET,
                           gpu=lambda: {**FREE, "free_vram_gb": 1.0})
    assert grabbed.reason_busy() and "GPU's memory" in grabbed.reason_busy()


def test_a_quiet_machine_does_not_pause_a_running_job():
    calm = PreemptCheck(within_s=60, input_idle=lambda: 1e6,
                        notification=lambda: None, load=lambda: QUIET,
                        gpu=lambda: FREE)
    assert calm.reason_busy() is None


def test_the_mid_job_check_ignores_gpu_utilisation():
    """Between our stages the card may still be settling from OUR last
    one, so utilisation there is not evidence about anyone else."""
    ours_settling = PreemptCheck(within_s=60, input_idle=lambda: 1e6,
                                 notification=lambda: None, load=lambda: QUIET,
                                 gpu=lambda: {**FREE, "util_pct": 99.0})
    assert ours_settling.reason_busy() is None


def test_the_real_load_probe_answers_on_this_machine():
    from clipforge.idle import system_load

    load = system_load(interval_s=0.2)
    assert load is None or (0.0 <= load["cpu_pct"] <= 100.0
                            and load["free_ram_gb"] > 0)


def test_our_own_cpu_is_subtracted_before_anyone_is_called_busy():
    """The pipeline must not read its own tail as the operator's load: a
    checkpoint runs while the previous stage is still winding down, and a
    gate that counted us would never open again."""
    from clipforge.idle import system_load

    class FakeProc:
        def __init__(self, pct): self._pct = pct
        def cpu_percent(self, _=None): return self._pct
        def children(self, recursive=False): return [FakeProc(240.0)]

    class FakePsutil:
        Error = RuntimeError
        @staticmethod
        def Process(): return FakeProc(120.0)
        @staticmethod
        def cpu_percent(interval=None): return 90.0    # whole machine
        @staticmethod
        def cpu_count(): return 12
        @staticmethod
        def virtual_memory():
            class M: available = 16 * 1024 ** 3
            return M()

    load = system_load(interval_s=0.0, ps=FakePsutil)
    # 90% total, ours is (120 + 240) / 12 cores = 30 points of it.
    assert load["cpu_pct"] == pytest.approx(60.0), load
    assert load["free_ram_gb"] == pytest.approx(16.0)


def test_subtraction_never_goes_negative():
    from clipforge.idle import system_load

    class FakeProc:
        def cpu_percent(self, _=None): return 1200.0
        def children(self, recursive=False): return []

    class FakePsutil:
        Error = RuntimeError
        Process = staticmethod(lambda: FakeProc())
        cpu_percent = staticmethod(lambda interval=None: 30.0)
        cpu_count = staticmethod(lambda: 12)
        @staticmethod
        def virtual_memory():
            class M: available = 8 * 1024 ** 3
            return M()

    assert system_load(interval_s=0.0, ps=FakePsutil)["cpu_pct"] == 0.0


def test_watch_gives_the_gate_every_configured_threshold():
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli._watch_locked)
    for knob in ("idle_after_s", "gpu_busy_pct", "min_free_vram_gb",
                 "cpu_busy_pct", "min_free_ram_gb"):
        assert f"{knob}=cfg.watch.{knob}" in src, (
            f"watch no longer passes [watch] {knob} to the gate")
