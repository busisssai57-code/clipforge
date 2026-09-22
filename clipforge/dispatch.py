"""The seam between ingestion and the clip DAG.

`bta watch` records, chunks, and builds overlap-corrected windows — and
then did nothing with them, because `ChannelMonitor.on_media` defaulted to
`None` and the CLI never passed one. A very careful recorder that never
produced a clip.

The wiring is not just "call process()", for one reason that matters:
**ingestion must never wait for the GPU.** A live stream is not
replayable. If the DAG ran inline on the monitor's thread, a four-minute
render would stall recording for four minutes and that footage is gone for
good. So media is handed to a bounded queue and drained by ONE worker
thread — one, because the VRAM Law allows a single GPU stage at a time and
a second worker would only queue up behind the same lock.

Backpressure policy, stated because the alternative is worse: when the
queue is full the window is DROPPED from the clip queue, loudly, and
ingestion continues. Blocking would trade a permanent loss (unrecorded
stream) for a recoverable one — the media is still on disk and can be run
through `bta process` later. Dropping is the cheaper mistake.

**Clipping waits for an idle machine; recording never does.** An optional
``gate`` is consulted before each window is clipped. While it reports the
operator busy, the worker holds the window and waits — the queue keeps
accepting (and the monitor keeps recording) the whole time. Time spent
waiting on the gate does not count toward a window's staleness: the
freshness rule exists to catch a machine that cannot keep up, and a
machine deliberately yielding to its owner is not that.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from clipforge.idle import JobPreempted, reset_checkpoint, set_checkpoint
from clipforge.log import get_logger

log = get_logger(__name__)

#: Sentinel that tells the worker to finish and exit.
_STOP = object()


@dataclass
class DispatchStats:
    submitted: int = 0
    processed: int = 0
    failed: int = 0
    dropped: int = 0
    last_error: str = ""

    def snapshot(self) -> dict[str, int | str]:
        return {"submitted": self.submitted, "processed": self.processed,
                "failed": self.failed, "dropped": self.dropped,
                "last_error": self.last_error}


@dataclass
class ClipDispatcher:
    """Bounded queue + single worker running the clip DAG off the hot path.

    ``handler`` is injected rather than importing the CLI, so this is
    testable without the pipeline and cannot create an import cycle.
    """

    handler: Callable[[Path, float], None]
    maxsize: int = 32
    #: Seconds a queued window may wait before it is considered stale. Live
    #: content ages badly, and a backlog older than this is usually a sign
    #: the machine cannot keep up rather than work worth doing.
    max_age_s: float = 3600.0
    #: Returns why clipping must wait (operator busy), or None to proceed.
    #: None as the gate itself means "never wait" — the pre-gate behaviour.
    gate: Callable[[], str | None] | None = None
    #: Mid-job test: has the operator come back SINCE this window started?
    #: Checked at the pipeline's stage boundaries. Without it the gate is
    #: only a starting condition, and one window is 6-33 minutes of GPU.
    preempt: Callable[[], str | None] | None = None
    #: Called with a window's source key once it has been clipped (or has
    #: failed in a way retrying will not fix). This is what makes the
    #: backlog durable: the queue itself is memory, and a reboot during a
    #: long gaming session used to lose every window waiting in it.
    on_settled: Callable[[Any, str], None] | None = None
    #: Seconds between gate checks while the operator is busy.
    gate_poll_s: float = 30.0
    stats: DispatchStats = field(default_factory=DispatchStats)

    _q: queue.Queue = field(init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    #: Set by stop(): wakes a worker that is sleeping on a closed gate.
    _halt: threading.Event = field(default_factory=threading.Event, init=False)
    #: Total seconds the worker has spent held by the gate. A window's age
    #: is measured net of whatever part of this accrued while it queued.
    _gated_s: float = field(default=0.0, init=False)
    #: When the CURRENT wait began, or None. Without it, a window submitted
    #: during a wait recorded a mark that excluded the part of that wait
    #: already elapsed, and was then credited with the whole wait — so
    #: after a long session every window read as newer than it was, by a
    #: different amount each, and the freshness rule stopped meaning
    #: anything. Set and cleared under _lock.
    _gate_started: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._q = queue.Queue(maxsize=max(1, int(self.maxsize)))

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "ClipDispatcher":
        if self._thread is not None:
            return self
        self._halt.clear()
        self._thread = threading.Thread(
            target=self._run, name="clip-dispatch", daemon=True)
        self._thread.start()
        log.info("dispatch.started", queue_maxsize=self._q.maxsize)
        return self

    def stop(self, *, timeout: float = 30.0, drain: bool = False) -> None:
        """Stop the worker. ``drain=False`` abandons the backlog.

        Default is NOT to drain: Ctrl-C on a watch loop should return the
        terminal promptly, and every queued window is still a file on
        disk. Draining is opt-in for a clean shutdown that wants the
        backlog finished.
        """
        if self._thread is None:
            return
        if not drain:
            # Only abandoning the backlog halts a worker parked on the
            # gate. Setting it while draining made the documented "finish
            # the backlog" shutdown finish nothing whenever a gate was
            # configured: every remaining window was dropped at the gate.
            self._halt.set()
            self._flush_queue()
        self._q.put(_STOP)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive() and drain:
            # The drain ran out of time — most likely parked on a closed
            # gate. Now the halt is right: stop() must return.
            self._halt.set()
            self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.warning("dispatch.stop_timeout", timeout_s=timeout,
                        note="worker still running; it is a daemon thread "
                             "and will not block exit")
        self._thread = None
        log.info("dispatch.stopped", **self.stats.snapshot())

    def _flush_queue(self) -> None:
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            if item is not _STOP:
                with self._lock:
                    self.stats.dropped += 1

    # -------------------------------------------------------------- submit

    def submit(self, path: Path, abs_start_s: float,
               key: Any = None) -> bool:
        """Queue one window. Never blocks; returns False if it was dropped.

        ``key`` names the SOURCE this window came from (a recorded segment,
        a downloaded VOD) so the caller can record that it is finished.
        """
        try:
            with self._lock:
                # Both under one lock, from one clock: monotonic, so an NTP
                # step cannot age a window that is minutes old.
                gated_mark = self._gated_now()
                queued_at = time.monotonic()
            self._q.put_nowait((Path(path), float(abs_start_s), queued_at,
                                gated_mark, key))
        except queue.Full:
            with self._lock:
                self.stats.dropped += 1
            log.warning(
                "dispatch.queue_full", path=str(path),
                queue_maxsize=self._q.maxsize,
                note="window NOT clipped; the media is on disk and can be "
                     "run through `bta process` later. Ingestion continues "
                     "— stalling it would lose unrecorded stream instead.")
            return False
        with self._lock:
            self.stats.submitted += 1
        return True

    # ---------------------------------------------------------------- work

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is _STOP:
                return
            path, abs_start_s, queued_at, gated_mark, key = item
            if not self._wait_for_gate(path):
                with self._lock:
                    self.stats.dropped += 1
                continue
            with self._lock:
                gated_while_queued = self._gated_now() - gated_mark
            age = time.monotonic() - queued_at - gated_while_queued
            if age > self.max_age_s:
                with self._lock:
                    self.stats.dropped += 1
                self._settle(key, "stale")
                log.warning("dispatch.stale_window", path=str(path),
                            age_s=round(age, 1), max_age_s=self.max_age_s,
                            note="backlog older than the freshness window; "
                                 "the machine is not keeping up")
                continue
            if not path.exists():
                # Retention may have swept the window while it queued.
                with self._lock:
                    self.stats.dropped += 1
                self._settle(key, "vanished")
                log.warning("dispatch.window_vanished", path=str(path))
                continue
            started = time.monotonic()
            token = set_checkpoint(self._checkpoint) if self.gate else None
            try:
                self.handler(path, abs_start_s)
            except JobPreempted as exc:
                # Not a failure: the job yielded the GPU to the operator
                # mid-flight. Its finished stages are in the cache, so the
                # retry after this one starts where it left off.
                with self._lock:
                    self.stats.dropped += 1
                log.info("dispatch.clip_preempted", path=str(path),
                         reason=str(exc)[:200],
                         elapsed_s=round(time.monotonic() - started, 1))
                continue
            except BaseException as exc:  # noqa: BLE001
                # A DAG failure must never kill the dispatcher: the next
                # window may well succeed, and ingestion is still running.
                with self._lock:
                    self.stats.failed += 1
                    self.stats.last_error = f"{type(exc).__name__}: {exc}"[:300]
                # Settled, not retried: a window that failed the DAG fails
                # it again on the next boot, and a loop that re-clips it
                # for ever is worse than losing it.
                self._settle(key, "failed")
                log.error("dispatch.clip_failed", path=str(path),
                          error=f"{type(exc).__name__}: {exc}"[:300],
                          elapsed_s=round(time.monotonic() - started, 1))
                continue
            finally:
                if token is not None:
                    reset_checkpoint(token)
            with self._lock:
                self.stats.processed += 1
            self._settle(key, "processed")
            log.info("dispatch.clip_done", path=str(path),
                     elapsed_s=round(time.monotonic() - started, 1),
                     **self.stats.snapshot())

    def _settle(self, key: Any, outcome: str) -> None:
        """Record that this source needs no further clipping. Never raises."""
        if key is None or self.on_settled is None:
            return
        try:
            self.on_settled(key, outcome)
        except Exception as exc:  # noqa: BLE001 - bookkeeping never kills work
            log.warning("dispatch.settle_failed", key=str(key),
                        error=f"{type(exc).__name__}: {exc}"[:200])

    def _checkpoint(self, stage: str) -> None:
        """Pause a RUNNING job when the operator comes back.

        Raises JobPreempted when the dispatcher is stopping, so a Ctrl-C
        during a 20-minute render returns the terminal instead of waiting
        the render out.
        """
        if self.preempt is None or self._halt.is_set():
            if self._halt.is_set():
                raise JobPreempted("shutting down")
            return
        try:
            reason = self.preempt()
        except Exception as exc:  # noqa: BLE001 - a broken probe never stalls
            log.warning("dispatch.preempt_error", stage=stage,
                        error=f"{type(exc).__name__}: {exc}"[:200])
            return
        if reason is None:
            return
        log.info("dispatch.job_paused", stage=stage, reason=reason,
                 note="the operator is back; waiting for the machine again")
        started = time.monotonic()
        with self._lock:
            self._gate_started = started
        try:
            while not self._halt.is_set():
                if self.gate is not None and self.gate() is None:
                    log.info("dispatch.job_resumed", stage=stage,
                             paused_s=round(time.monotonic() - started, 1))
                    return
                self._halt.wait(self.gate_poll_s)
            raise JobPreempted(f"stopped while paused at {stage}")
        finally:
            with self._lock:
                self._gated_s += time.monotonic() - started
                self._gate_started = None

    def _gated_now(self) -> float:
        """Gate-held seconds including a wait in progress. Call under _lock."""
        if self._gate_started is None:
            return self._gated_s
        return self._gated_s + (time.monotonic() - self._gate_started)

    def _wait_for_gate(self, path: Path) -> bool:
        """Hold ``path`` until the gate opens. False if stopped meanwhile."""
        if self.gate is None:
            return True
        started = time.monotonic()
        announced = False
        with self._lock:
            self._gate_started = started
        try:
            while not self._halt.is_set():
                try:
                    reason = self.gate()
                except Exception as exc:  # noqa: BLE001
                    # A broken probe must not park the queue for ever.
                    log.warning("dispatch.gate_error",
                                error=f"{type(exc).__name__}: {exc}"[:200],
                                note="proceeding without the idle gate")
                    return True
                if reason is None:
                    if announced:
                        log.info("dispatch.gate_open", path=str(path),
                                 waited_s=round(time.monotonic() - started, 1))
                    return True
                if not announced:
                    log.info("dispatch.waiting_for_idle", path=str(path),
                             reason=reason, pending=self.pending)
                    announced = True
                self._halt.wait(self.gate_poll_s)
            return False
        finally:
            with self._lock:
                self._gated_s += time.monotonic() - started
                self._gate_started = None

    # -------------------------------------------------------------- report

    @property
    def pending(self) -> int:
        return self._q.qsize()
