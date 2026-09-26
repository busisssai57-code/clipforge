"""The ingestion → DAG seam, pinned.

`bta watch` recorded forever and never clipped, because `on_media` was
never passed. The wiring is easy; the properties that make it SAFE are
not, and those are what this file holds:

  * ingestion never waits for the GPU (a live stream is not replayable);
  * a DAG failure never kills the dispatcher;
  * a full queue drops the CLIP, never the recording.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from clipforge.dispatch import ClipDispatcher


@pytest.fixture()
def media(tmp_path):
    def _make(name: str = "w.mp4") -> Path:
        p = tmp_path / name
        p.write_bytes(b"\x00" * 8)
        return p
    return _make


def _drain(disp: ClipDispatcher, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while disp.pending and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)


def test_a_submitted_window_reaches_the_dag(media):
    seen: list[tuple[Path, float]] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append((p, t))).start()
    try:
        assert disp.submit(media(), 123.5) is True
        _drain(disp)
    finally:
        disp.stop()
    assert seen and seen[0][1] == 123.5, seen
    assert disp.stats.processed == 1


def test_submit_never_blocks_the_caller(media):
    """The property that matters most: ingestion must not wait for a
    render. A live stream is not replayable, so a blocking submit trades
    a recoverable loss for a permanent one."""
    started = threading.Event()
    release = threading.Event()

    def slow(_p, _t):
        started.set()
        release.wait(timeout=10)

    disp = ClipDispatcher(handler=slow, maxsize=4).start()
    try:
        disp.submit(media("a.mp4"), 0.0)
        assert started.wait(timeout=5), "worker never picked up the window"
        # The worker is now blocked. Submitting must still return at once.
        t0 = time.monotonic()
        disp.submit(media("b.mp4"), 1.0)
        elapsed = time.monotonic() - t0
        assert elapsed < 0.5, (
            f"submit blocked for {elapsed:.2f}s while the DAG was busy")
    finally:
        release.set()
        disp.stop()


def test_a_full_queue_drops_the_clip_and_keeps_ingesting(media):
    release = threading.Event()
    disp = ClipDispatcher(handler=lambda p, t: release.wait(timeout=10),
                          maxsize=1).start()
    try:
        disp.submit(media("a.mp4"), 0.0)          # picked up by the worker
        time.sleep(0.1)
        disp.submit(media("b.mp4"), 1.0)          # fills the queue
        dropped = [disp.submit(media(f"{i}.mp4"), 2.0) for i in range(3)]
        assert any(d is False for d in dropped), (
            "a full queue must refuse rather than block")
        assert disp.stats.dropped >= 1
    finally:
        release.set()
        disp.stop()


def test_a_dag_failure_does_not_kill_the_dispatcher(media):
    """One bad window must not end clipping for the rest of the session."""
    calls: list[str] = []

    def flaky(p: Path, _t: float) -> None:
        calls.append(p.name)
        if p.name == "bad.mp4":
            raise RuntimeError("stage exploded")

    disp = ClipDispatcher(handler=flaky).start()
    try:
        disp.submit(media("bad.mp4"), 0.0)
        _drain(disp)
        disp.submit(media("good.mp4"), 1.0)
        _drain(disp)
    finally:
        disp.stop()
    assert calls == ["bad.mp4", "good.mp4"], calls
    assert disp.stats.failed == 1 and disp.stats.processed == 1
    assert "stage exploded" in disp.stats.last_error


def test_a_vanished_window_is_skipped_not_crashed(tmp_path):
    """Retention can sweep a window while it sits in the queue."""
    seen: list[Path] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append(p)).start()
    try:
        disp.submit(tmp_path / "never-existed.mp4", 0.0)
        _drain(disp)
    finally:
        disp.stop()
    assert seen == []
    assert disp.stats.dropped == 1


def test_a_stale_backlog_entry_is_dropped(media):
    """A window that sat in the queue past the freshness window is
    evidence the machine is not keeping up, not work worth doing.

    The item has to genuinely AGE for this to mean anything: an earlier
    version set max_age_s=0 and expected an instant drop, which passed or
    failed on Windows clock granularity (the worker dequeued within one
    ~15 ms tick, so age was exactly 0.0 and `age > 0.0` was False). Here
    the worker is held busy while the second window ages behind it.
    """
    seen: list[Path] = []
    release = threading.Event()

    def handler(p: Path, _t: float) -> None:
        if p.name == "blocker.mp4":
            release.wait(timeout=10)
            return
        seen.append(p)

    disp = ClipDispatcher(handler=handler, max_age_s=0.25, maxsize=8).start()
    try:
        disp.submit(media("blocker.mp4"), 0.0)
        time.sleep(0.05)                       # let the worker take it
        disp.submit(media("stale.mp4"), 0.0)   # queues behind the blocker
        time.sleep(0.4)                        # ... and goes stale
        release.set()
        _drain(disp)
    finally:
        release.set()
        disp.stop()
    assert seen == [], "a stale window was processed anyway"
    assert disp.stats.dropped == 1


def test_stop_abandons_the_backlog_by_default(media):
    release = threading.Event()
    disp = ClipDispatcher(handler=lambda p, t: release.wait(timeout=5),
                          maxsize=8).start()
    disp.submit(media("a.mp4"), 0.0)
    time.sleep(0.1)
    for i in range(4):
        disp.submit(media(f"q{i}.mp4"), 0.0)
    release.set()
    disp.stop(timeout=5)
    assert disp.stats.dropped >= 4, (
        "stop() should abandon queued windows; they are still on disk")


def test_stop_is_idempotent_and_safe_without_start(media):
    disp = ClipDispatcher(handler=lambda p, t: None)
    disp.stop()               # never started
    disp.start()
    disp.stop()
    disp.stop()               # twice


def test_the_watch_command_actually_passes_the_seam():
    """Structural guard on the wiring itself: the bug was not a broken
    dispatcher, it was a callback nobody supplied."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli._watch_locked)
    assert "on_media=dispatcher.submit" in src, (
        "watch no longer hands the DAG seam to the monitor; it will "
        "record forever and clip nothing")
    assert "jumpcut=None" in src, (
        "watch must pass jumpcut explicitly or the Typer sentinel forces "
        "pacing on")


# ------------------------------------------------------------- idle gate
#
# Clipping waits for an idle machine; recording never does. These pin the
# three ways that could go wrong: the gate ignored, the queue refusing work
# while the gate is shut, and every window that waited for the operator
# being thrown away as "stale" the moment the operator left.


def test_a_closed_gate_holds_the_clip_but_keeps_accepting(media):
    busy = threading.Event()
    busy.set()
    seen: list[Path] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append(p), maxsize=8,
                          gate=lambda: "operator active" if busy.is_set() else None,
                          gate_poll_s=0.02).start()
    try:
        for i in range(3):
            assert disp.submit(media(f"w{i}.mp4"), 0.0), (
                "the queue must keep accepting while clipping is held")
        time.sleep(0.2)
        assert seen == [], "a clip ran while the operator was using the PC"
        busy.clear()
        _drain(disp)
        time.sleep(0.1)
    finally:
        busy.clear()
        disp.stop()
    assert [p.name for p in seen] == ["w0.mp4", "w1.mp4", "w2.mp4"]
    assert disp.stats.dropped == 0


def test_time_spent_waiting_for_idle_is_not_staleness(media):
    """max_age_s catches a machine that cannot keep up. A machine that
    yielded to its owner for longer than max_age_s is not that, and must
    not throw away everything it recorded meanwhile."""
    busy = threading.Event()
    busy.set()
    seen: list[Path] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append(p), maxsize=8,
                          max_age_s=0.2,
                          gate=lambda: "busy" if busy.is_set() else None,
                          gate_poll_s=0.02).start()
    try:
        disp.submit(media("first.mp4"), 0.0)
        disp.submit(media("second.mp4"), 0.0)
        time.sleep(0.6)             # three times the freshness window
        busy.clear()
        _drain(disp)
        time.sleep(0.1)
    finally:
        busy.clear()
        disp.stop()
    assert [p.name for p in seen] == ["first.mp4", "second.mp4"]
    assert disp.stats.dropped == 0


def test_the_freshness_rule_still_applies_while_the_gate_is_open(media):
    """Control for the test above: excluding gated time must not switch
    the staleness check off altogether."""
    release = threading.Event()
    seen: list[Path] = []

    def handler(p: Path, _t: float) -> None:
        if p.name == "blocker.mp4":
            release.wait(timeout=10)
            return
        seen.append(p)

    disp = ClipDispatcher(handler=handler, max_age_s=0.25, maxsize=8,
                          gate=lambda: None, gate_poll_s=0.02).start()
    try:
        disp.submit(media("blocker.mp4"), 0.0)
        time.sleep(0.05)
        disp.submit(media("stale.mp4"), 0.0)
        time.sleep(0.4)
        release.set()
        _drain(disp)
    finally:
        release.set()
        disp.stop()
    assert seen == []
    assert disp.stats.dropped == 1


def test_stop_wakes_a_worker_waiting_for_idle(media):
    disp = ClipDispatcher(handler=lambda p, t: None,
                          gate=lambda: "busy", gate_poll_s=30.0).start()
    disp.submit(media("held.mp4"), 0.0)
    time.sleep(0.1)
    started = time.monotonic()
    disp.stop(timeout=5)
    assert time.monotonic() - started < 2.0, (
        "stop() waited out the 30 s gate poll instead of waking the worker")


def test_a_broken_gate_does_not_park_the_queue_forever(media):
    def boom() -> str | None:
        raise OSError("nvidia-smi vanished")

    seen: list[Path] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append(p), gate=boom,
                          gate_poll_s=0.02).start()
    try:
        disp.submit(media("w.mp4"), 0.0)
        _drain(disp)
        time.sleep(0.1)
    finally:
        disp.stop()
    assert [p.name for p in seen] == ["w.mp4"]


def test_the_watch_command_gates_clipping_on_idle():
    """Structural guard, same reason as the seam guard above: a gate that
    exists and is never passed is the bug this project keeps finding."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli._watch_locked)
    assert "gate=gate" in src, "watch no longer passes the idle gate"
    assert "IdleGate(" in src
    assert "_start_telegram_retry(" in src, (
        "watch no longer retries failed Telegram deliveries")


# --------------------------------------------------- mid-job preemption
#
# The gate alone is only a STARTING condition, and one window is 6-33
# minutes of GPU work (measured from this workspace's stage_runs). The
# operator coming back partway through is the normal case, not a rare one.


def test_a_running_job_pauses_when_the_operator_comes_back(media):
    from clipforge.idle import idle_checkpoint

    back = threading.Event()
    at_checkpoint = threading.Event()
    stages: list[str] = []
    resumed = threading.Event()

    def handler(p: Path, _t: float) -> None:
        stages.append("s1")
        at_checkpoint.wait(timeout=5)   # ... and now the operator is back
        idle_checkpoint("s2")
        stages.append("s2")
        resumed.set()

    disp = ClipDispatcher(
        handler=handler, gate=lambda: "busy" if back.is_set() else None,
        preempt=lambda: "operator came back" if back.is_set() else None,
        gate_poll_s=0.02).start()
    try:
        disp.submit(media("w.mp4"), 0.0)     # gate open: the job starts
        time.sleep(0.15)
        assert stages == ["s1"]
        back.set()                            # operator touches the keyboard
        at_checkpoint.set()
        time.sleep(0.3)
        assert stages == ["s1"], "the job ran on through the checkpoint"
        back.clear()                          # they leave again
        assert resumed.wait(timeout=5), "the job never resumed"
    finally:
        back.clear()
        disp.stop()
    assert stages == ["s1", "s2"]
    assert disp.stats.processed == 1


def test_a_paused_job_lets_go_on_shutdown(media):
    """Ctrl-C during a 20-minute render must return the terminal, not
    wait the render out."""
    from clipforge.idle import idle_checkpoint

    busy = threading.Event()
    paused = threading.Event()

    def handler(p: Path, _t: float) -> None:
        busy.set()          # from here the gate and preempt both say busy
        paused.set()
        idle_checkpoint("s2")
        raise AssertionError("the checkpoint should not have returned")

    disp = ClipDispatcher(
        handler=handler,
        gate=lambda: "busy" if busy.is_set() else None,
        preempt=lambda: "busy" if busy.is_set() else None,
        gate_poll_s=30.0).start()
    disp.submit(media("w.mp4"), 0.0)
    assert paused.wait(timeout=5)
    time.sleep(0.2)
    started = time.monotonic()
    disp.stop(timeout=5)
    assert time.monotonic() - started < 2.0
    assert disp.stats.failed == 0, "a preempted job is not a failed job"


def test_a_checkpoint_does_nothing_without_a_dispatcher():
    """`bta process` by hand must behave exactly as it always did."""
    from clipforge.idle import idle_checkpoint

    idle_checkpoint("s1")      # no checkpoint installed: a no-op


def test_a_broken_preempt_probe_does_not_wedge_a_job(media):
    from clipforge.idle import idle_checkpoint

    done = threading.Event()

    def handler(p: Path, _t: float) -> None:
        idle_checkpoint("s2")
        done.set()

    def boom() -> str | None:
        raise OSError("nvidia-smi vanished")

    disp = ClipDispatcher(handler=handler, gate=lambda: None, preempt=boom,
                          gate_poll_s=0.02).start()
    try:
        disp.submit(media("w.mp4"), 0.0)
        assert done.wait(timeout=5)
    finally:
        disp.stop()


def test_stop_finishes_the_backlog_when_asked_to_drain(media):
    """The documented opt-in: with a gate configured but open, drain=True
    must finish the queue instead of dropping it at the gate."""
    seen: list[Path] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append(p), maxsize=8,
                          gate=lambda: None, gate_poll_s=0.02).start()
    for i in range(3):
        disp.submit(media(f"d{i}.mp4"), 0.0)
    disp.stop(timeout=10, drain=True)
    assert len(seen) == 3, "drain=True dropped the backlog it promised to finish"
    assert disp.stats.dropped == 0


def test_a_window_queued_during_a_long_wait_is_not_credited_twice(media):
    """Staleness is measured net of gate time, and the credit must cover
    only the part of the wait that happened AFTER the window queued."""
    busy = threading.Event()
    busy.set()
    seen: list[Path] = []
    disp = ClipDispatcher(handler=lambda p, t: seen.append(p), maxsize=8,
                          max_age_s=0.3,
                          gate=lambda: "busy" if busy.is_set() else None,
                          gate_poll_s=0.02).start()
    try:
        disp.submit(media("first.mp4"), 0.0)
        time.sleep(0.5)                       # the wait is already running
        disp.submit(media("late.mp4"), 0.0)   # queued mid-wait
        time.sleep(0.5)
        busy.clear()
        _drain(disp)
        time.sleep(0.1)
    finally:
        busy.clear()
        disp.stop()
    assert [p.name for p in seen] == ["first.mp4", "late.mp4"]
    assert disp.stats.dropped == 0
