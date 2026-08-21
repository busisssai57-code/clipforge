"""Filesystem watcher — watchdog observer + stable-file debounce (spec §4).

Purpose: notice NEW media files (VOD downloads completing, operator-dropped
files) and hand them to the pipeline only once they are STABLE — the same
"never hand a growing file to the DAG" rule the chunker enforces (§S0),
applied to files we don't own the writer of.

Two layers, deliberately separated:
  * :class:`StableFileTracker` — a pure, clock-injected state machine
    (size-unchanged-for-N-seconds ⇒ stable). Fully unit-tested offline.
  * :class:`DirectoryWatcher` — watchdog Observer wiring + a polling
    sweep. watchdog events only PROD the tracker; the poll sweep is the
    correctness backstop (ReadDirectoryChangesW drops events under load,
    and a file that finished writing before startup emits no event at all).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from clipforge.log import get_logger

log = get_logger(__name__)

MEDIA_SUFFIXES = {".mp4", ".mkv", ".ts", ".webm", ".m4v", ".mov"}


@dataclass
class _Candidate:
    size: int
    stable_since: float


@dataclass
class StableFileTracker:
    """Pure debounce logic: a file is emitted once its size has been
    unchanged for ``stable_s`` seconds. Each file is emitted at most once.
    ``clock`` is injected so tests control time."""

    stable_s: float
    clock: Callable[[], float] = time.monotonic
    _candidates: dict[Path, _Candidate] = field(default_factory=dict)
    _emitted: set[Path] = field(default_factory=set)

    def observe(self, path: Path, size: int | None = None) -> None:
        """Record that ``path`` exists (event or sweep). ``size=None`` means
        stat it now; a vanished file is dropped silently."""
        path = Path(path)
        if path in self._emitted or path.suffix.lower() not in MEDIA_SUFFIXES:
            return
        if size is None:
            try:
                size = path.stat().st_size
            except OSError:
                self._candidates.pop(path, None)
                return
        now = self.clock()
        cand = self._candidates.get(path)
        if cand is None or cand.size != size:
            self._candidates[path] = _Candidate(size=size, stable_since=now)

    def harvest(self) -> list[Path]:
        """Files that just became stable, deterministic order. Call on a
        cadence (each poll sweep).

        Also evicts zero-byte candidates that have been stale for well past
        the stability window: an aborted download leaves a 0-byte file that
        can never satisfy ``size > 0``, and over days those accumulate in
        memory forever.
        """
        now = self.clock()
        ready: list[Path] = []
        stale: list[Path] = []
        for path in sorted(self._candidates):
            cand = self._candidates[path]
            age = now - cand.stable_since
            if cand.size > 0 and age >= self.stable_s:
                ready.append(path)
            elif cand.size <= 0 and age >= self.stable_s * 10:
                stale.append(path)
        for path in ready:
            del self._candidates[path]
            self._emitted.add(path)
        for path in stale:
            del self._candidates[path]  # not emitted: it never had content
        return ready


class DirectoryWatcher:
    """watchdog + poll-sweep driver around a StableFileTracker.

    ``on_stable`` is invoked from the watcher thread — callbacks must be
    quick and thread-safe (the monitor hands the path to
    ``ClipDispatcher.submit``).
    """

    def __init__(self, directory: Path, on_stable: Callable[[Path], None], *,
                 stable_s: float = 20.0, sweep_interval_s: float = 5.0) -> None:
        self.directory = Path(directory)
        self.on_stable = on_stable
        self.tracker = StableFileTracker(stable_s=stable_s)
        self.sweep_interval_s = sweep_interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._observer = None

    # ------------------------------------------------------------------ api

    def start(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._start_observer()
        self._thread = threading.Thread(target=self._sweep_loop,
                                        name=f"watcher:{self.directory.name}",
                                        daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
        if self._thread is not None:
            self._thread.join(timeout=self.sweep_interval_s + 5.0)

    # ------------------------------------------------------------- internals

    def _start_observer(self) -> None:
        """watchdog is an accelerant, not a dependency for correctness —
        if it fails to start (network drive, exotic FS), the poll sweep
        alone still finds everything, just up to one sweep later."""
        try:
            from watchdog.events import FileSystemEventHandler
            from watchdog.observers import Observer

            tracker = self.tracker

            class _Handler(FileSystemEventHandler):
                def on_created(self, event):  # noqa: ANN001
                    if not event.is_directory:
                        tracker.observe(Path(event.src_path))

                def on_modified(self, event):  # noqa: ANN001
                    if not event.is_directory:
                        tracker.observe(Path(event.src_path))

                def on_moved(self, event):  # noqa: ANN001
                    if not event.is_directory:
                        tracker.observe(Path(event.dest_path))

            self._observer = Observer()
            self._observer.schedule(_Handler(), str(self.directory),
                                    recursive=True)
            self._observer.daemon = True
            self._observer.start()
        except Exception as exc:
            log.warning("watcher.observer_unavailable", error=str(exc),
                        note="poll sweep remains the correctness backstop")
            self._observer = None

    def _sweep_loop(self) -> None:
        while not self._stop.is_set():
            try:
                for p in self.directory.rglob("*"):
                    if p.is_file():
                        self.tracker.observe(p)
                for stable in self.tracker.harvest():
                    log.info("watcher.stable_file", path=str(stable))
                    try:
                        self.on_stable(stable)
                    except Exception as exc:  # callback bug ≠ watcher death
                        log.error("watcher.callback_failed", path=str(stable),
                                  error=f"{type(exc).__name__}: {exc}")
            except OSError as exc:  # dir briefly unavailable: keep sweeping
                log.warning("watcher.sweep_error", error=str(exc))
            self._stop.wait(self.sweep_interval_s)
