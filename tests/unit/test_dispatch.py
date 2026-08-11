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
