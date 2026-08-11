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
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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
    stats: DispatchStats = field(default_factory=DispatchStats)

    _q: queue.Queue = field(init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self._q = queue.Queue(maxsize=max(1, int(self.maxsize)))

    # ------------------------------------------------------------ lifecycle

    def start(self) -> "ClipDispatcher":
        if self._thread is not None:
            return self
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
            self._flush_queue()
        self._q.put(_STOP)
        self._thread.join(timeout=timeout)
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

    def submit(self, path: Path, abs_start_s: float) -> bool:
        """Queue one window. Never blocks; returns False if it was dropped."""
        try:
            self._q.put_nowait((Path(path), float(abs_start_s), time.time()))
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
            path, abs_start_s, queued_at = item
            age = time.time() - queued_at
            if age > self.max_age_s:
                with self._lock:
                    self.stats.dropped += 1
                log.warning("dispatch.stale_window", path=str(path),
                            age_s=round(age, 1), max_age_s=self.max_age_s,
                            note="backlog older than the freshness window; "
                                 "the machine is not keeping up")
                continue
            if not path.exists():
                # Retention may have swept the window while it queued.
                with self._lock:
                    self.stats.dropped += 1
                log.warning("dispatch.window_vanished", path=str(path))
                continue
            started = time.monotonic()
            try:
                self.handler(path, abs_start_s)
            except BaseException as exc:  # noqa: BLE001
                # A DAG failure must never kill the dispatcher: the next
                # window may well succeed, and ingestion is still running.
                with self._lock:
                    self.stats.failed += 1
                    self.stats.last_error = f"{type(exc).__name__}: {exc}"[:300]
                log.error("dispatch.clip_failed", path=str(path),
                          error=f"{type(exc).__name__}: {exc}"[:300],
                          elapsed_s=round(time.monotonic() - started, 1))
                continue
            with self._lock:
                self.stats.processed += 1
            log.info("dispatch.clip_done", path=str(path),
                     elapsed_s=round(time.monotonic() - started, 1),
                     **self.stats.snapshot())

    # -------------------------------------------------------------- report

    @property
    def pending(self) -> int:
        return self._q.qsize()
