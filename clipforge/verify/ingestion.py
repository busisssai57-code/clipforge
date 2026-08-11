"""CP1 deterministic gate — proves the ingestion contracts mechanically.

Checks (all real, all offline unless ffmpeg is present):
  1. Segment readiness: primary rule (successor exists) and the growing-file
     prohibition — a file still being written never reaches the DAG.
  2. Reconnect continuity: absolute media time survives a drop; segment
     indices never restart (spec §S0).
  3. T3 shutdown: streamlink is killed (delivering EOF), ffmpeg is WAITED on
     so it can finalize the tail; no zombies.
  4. T2: segments are MPEG-TS and a hard-killed writer still leaves a
     playable file (run for real when ffmpeg is installed, else SKIP).
  5. T1: virtual window = prev tail + chunk, with absolute time corrected by
     the REAL (keyframe-quantized) overlap; dedup by absolute time.
  6. T4: Kick is total — every failure mode returns None, never raises.
  7. Determinism: backoff delays are reproducible per (channel, attempt).
  8. Authorization: only enabled, operator-listed channels are ever touched.

Run: python -m clipforge.verify.ingestion
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path
from typing import Callable

from clipforge.config import AppConfig, ChannelSpec, Watchlist
from clipforge.errors import IngestError
from clipforge.ffmpeg import MediaInfo, find_binary, probe
from clipforge.ingest import kick
from clipforge.ingest.backoff import backoff_delay
from clipforge.ingest.chunker import (ChunkerConfig, ChunkerSession,
                                      ConnectLedger, ConnectResult,
                                      SegmentEvent)
from clipforge.ingest.monitor import ChannelMonitor, disk_allows
from clipforge.ingest.overlap import build_virtual_window, dedup_by_absolute_time
from clipforge.ingest.retention import (reconcile_sessions, sweep_retention,
                                        workspace_lock)
from clipforge.paths import Workspace
from clipforge.state import StateDB
from clipforge.watch.watcher import StableFileTracker

_CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(name: str):
    def deco(fn: Callable[[], None]):
        _CHECKS.append((name, fn))
        return fn
    return deco


class _Skip(Exception):
    """Raised by a check that cannot run here (missing optional binary)."""


# --------------------------------------------------------------- fake procs


class _FakeProc:
    def __init__(self, alive_ticks: int = 10 ** 6) -> None:
        self.alive_ticks = alive_ticks
        self.killed = False
        self.waited = False
        self.pid = 1234
        self.stdout = None

    def poll(self):
        if self.killed or self.alive_ticks <= 0:
            return 0
        self.alive_ticks -= 1
        return None

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        self.waited = True
        return 0


def _info(duration: float) -> MediaInfo:
    return MediaInfo(duration_s=duration, width=1920, height=1080, fps=30.0,
                     fps_rational="30/1", v_codec="h264", a_codec="aac")


def _session(root: Path, db: StateDB, procs: list, *, on_ready=None,
             durations: dict[str, float] | None = None) -> ChunkerSession:
    return ChunkerSession(
        db=db, chunks_root=root / "chunks",
        quarantine_dir=root / "quarantine", platform="twitch", handle="t",
        streamlink_args=["--stdout", "url", "best"],
        cfg=ChunkerConfig(segment_time_s=900, ready_stable_s=20.0,
                          poll_interval_s=0.0, remux_to_mp4=False),
        on_segment_ready=on_ready,
        popen=lambda cmd, **kw: procs.pop(0),
        prober=lambda p: _info((durations or {}).get(Path(p).name, 900.0)),
        clock=time.monotonic,
        resolve_tools=lambda: ("streamlink", "ffmpeg"))


def _seg_dir(root: Path, sid: int) -> Path:
    d = root / "chunks" / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _write_seg(d: Path, idx: int, size: int = 188 * 200) -> Path:
    p = d / f"chunk_{idx:05d}.ts"
    p.write_bytes(b"\x47" * size)
    return p


# ----------------------------------------------------------- 1. readiness


@check("segment ready only when provably closed")
def _readiness() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            d = _seg_dir(root, sid)
            _write_seg(d, 0)
            _write_seg(d, 1)
            events: list[SegmentEvent] = []
            sess = _session(root, db, [_FakeProc(2), _FakeProc(2)],
                            on_ready=events.append)
            sess._connect_once(threading.Event(), sid)
            assert [e.seg_index for e in events] == [0, 1], events
        finally:
            db.close()


@check("a growing segment is NEVER handed to the DAG (chunker, not a proxy)")
def _growing_tail() -> None:
    """Drives the real chunker and asserts on WHEN emission happens.

    An earlier version of this gate checked a StableFileTracker instead —
    which the chunker never uses — so a mutant that finalized the still-
    growing tail on every poll passed the entire suite.
    """
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            d = _seg_dir(root, sid)
            _write_seg(d, 0)
            writing = {"active": True}
            violations: list[int] = []

            def on_ready(ev: SegmentEvent) -> None:
                if writing["active"]:
                    violations.append(ev.seg_index)

            class _Growing(_FakeProc):
                def poll(self):
                    if writing["active"]:
                        _write_seg(d, 0, size=188 * 200 + int(self.alive_ticks) * 997)
                    alive = super().poll()
                    if alive is not None:
                        writing["active"] = False
                    return alive

            sess = _session(root, db, [_Growing(6), _FakeProc(6)],
                            on_ready=on_ready)
            sess._connect_once(threading.Event(), sid)
            assert not violations, (
                f"segments {violations} were emitted while still being written")
        finally:
            db.close()


@check("bad segments advance the clock (absolute time never shifts)")
def _quarantine_time() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            d = _seg_dir(root, sid)
            for i in range(4):
                _write_seg(d, i)
            events: list[SegmentEvent] = []

            def prober(p: Path) -> MediaInfo:
                if Path(p).name == "chunk_00001.ts":
                    raise ValueError("corrupt")
                return _info(900.0)

            sess = _session(root, db, [_FakeProc(2), _FakeProc(2)],
                            on_ready=events.append)
            sess._prober = prober
            sess._connect_once(threading.Event(), sid)

            got = {e.seg_index: e.abs_start_s for e in events}
            assert got == {0: 0.0, 2: 1800.0, 3: 2700.0}, got
            banked = db.get_session(sid)["base_offset_s"]
            assert banked == 3600.0, banked
        finally:
            db.close()


@check("any exception tears the pipe down (no orphans)")
def _teardown_on_exception() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            d = _seg_dir(root, sid)
            _write_seg(d, 0)
            _write_seg(d, 1)
            sl, ff = _FakeProc(10 ** 6), _FakeProc(10 ** 6)
            sess = _session(root, db, [sl, ff])

            def boom(*a, **kw):
                raise RuntimeError("injected bug")

            sess._finalize = boom  # type: ignore[method-assign]
            try:
                sess._connect_once(threading.Event(), sid)
                raise AssertionError("the injected error should propagate")
            except RuntimeError:
                pass
            assert sl.killed, "streamlink orphaned on an exception path"
            assert ff.waited or ff.killed, "ffmpeg orphaned on an exception path"
        finally:
            db.close()


@check("estimated time never masquerades as captured media")
def _no_phantom_media() -> None:
    """The round-2 blocker: an unprobeable tail was credited a nominal
    900 s, which both banked phantom time AND scored the connect healthy —
    so a stream that died after 2 s reset the backoff forever and the
    session never ended."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            d = _seg_dir(root, sid)
            _write_seg(d, 0)

            def bad(p: Path) -> MediaInfo:
                raise ValueError("torn TS")

            sess = _session(root, db, [_FakeProc(2), _FakeProc(2)])
            sess._prober = bad
            result = sess._connect_once(threading.Event(), sid)
            assert result.captured_s == 0.0, result
            assert result.estimated is True
            assert db.get_session(sid)["timeline_estimated"] == 1

            # And a dead stream gives up rather than looping forever.
            connects = {"n": 0}
            sess2 = _session(root, db, [])
            sess2._connect_once = lambda stop, s, deadline=None: (
                connects.__setitem__("n", connects["n"] + 1)
                or ConnectResult(captured_s=0.0, timeline_s=900.0,
                                 estimated=True))

            class _NoWait(threading.Event):
                def wait(self, timeout=None):
                    assert connects["n"] <= 20, "run() never gave up"
                    return False

            sess2.run(_NoWait(), session_id=sid)
            assert connects["n"] == 3, connects
        finally:
            db.close()


@check("reconcile never registers a file that is still being written")
def _reconcile_growing() -> None:
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "ws").ensure()
        db = StateDB(ws.state_db)
        try:
            sid = db.open_stream_session("twitch", "t")
            d = ws.chunks / "twitch_t" / f"s{sid:05d}"
            d.mkdir(parents=True, exist_ok=True)
            growing = d / "chunk_00000.ts"
            growing.write_bytes(b"\x47" * 4096)  # mtime = now
            # An ORPHANED recorder keeps writing while reconcile polls.
            # Round 8: a young mtime alone no longer proves "still being
            # written" — it also described the dead writer's final segment,
            # which is precisely the crash that must stay resumable. Proof
            # is size-growth under a bounded poll, so the orphan must grow.
            def orphan_writes(_dt: float) -> None:
                with growing.open("ab") as fh:
                    fh.write(b"\x47" * 4096)

            emitted: list[Path] = []
            recovered = reconcile_sessions(db, ws, prober=lambda p: _info(6.0),
                                           sleep=orphan_writes,
                                           on_segment=lambda p, t: emitted.append(p))
            assert recovered == 0 and emitted == [], (recovered, emitted)
            assert db.segments_for_session(sid) == []
            assert [s["id"] for s in db.open_sessions()] == [sid], \
                "a session with a live orphan writing into it must stay open"
        finally:
            db.close()


@check("retention reclaims T1 windows and stranded media")
def _retention_reclaims_everything() -> None:
    import os as _os
    import time as _time

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "ws").ensure()
        db = StateDB(ws.state_db)
        try:
            old = _time.time() - 100 * 3600
            # A T1 window (tracked by no DB row) and stranded chunk media.
            wdir = ws.tmp / "windows_s00001"
            wdir.mkdir(parents=True, exist_ok=True)
            win = wdir / "chunk_00001.window.ts"
            win.write_bytes(b"\x47" * 8192)
            _os.utime(win, (old, old))
            cdir = ws.chunks / "twitch_t" / "s00001"
            cdir.mkdir(parents=True, exist_ok=True)
            stranded = cdir / "chunk_00009.mp4"
            stranded.write_bytes(b"\x00" * 4096)
            _os.utime(stranded, (old, old))

            report = sweep_retention(db, ws, retention_hours=48.0)
            assert not win.exists(), "T1 windows are never reclaimed"
            assert not stranded.exists(), "stranded media is never reclaimed"
            assert report.files_deleted == 2, report
        finally:
            db.close()


@check("recovery + resume never destroy recovered media")
def _reconcile_resume_safety() -> None:
    """Round-3 blocker: reconcile registered recovered media but left the
    SESSION row at its pre-crash cursor, so the monitor's resume restarted
    ffmpeg's -segment_start_number on top of the recovered files."""
    import os as _os
    import time as _time

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "ws").ensure()
        db = StateDB(ws.state_db)
        try:
            sid = db.open_stream_session("twitch", "t")
            d = ws.chunks / "twitch_t" / f"s{sid:05d}"
            d.mkdir(parents=True, exist_ok=True)
            settled = _time.time() - 600
            for i in range(3):
                p = d / f"chunk_{i:05d}.ts"
                p.write_bytes(b"\x47" * 4096)
                _os.utime(p, (settled, settled))

            reconcile_sessions(db, ws, prober=lambda p: _info(900.0),
                               segment_time_s=900.0)

            sess = db.get_session(sid)
            assert int(sess["next_segment"]) == 3, dict(sess)
            assert float(sess["base_offset_s"]) == 2700.0, dict(sess)
        finally:
            db.close()


@check("cursor writeback survives an ENTIRELY unprobeable session")
def _reconcile_all_unprobeable() -> None:
    """Round-4 blocker: the writeback was gated on a counter that ignored
    estimated recoveries, so a session whose stranded media is all
    unprobeable (0-byte / zero-filled segments — the classic post-power-loss
    NTFS artifact) kept its stale cursor and a resume overwrote it."""
    import os as _os
    import time as _time

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "ws").ensure()
        db = StateDB(ws.state_db)
        try:
            sid = db.open_stream_session("twitch", "t")
            d = ws.chunks / "twitch_t" / f"s{sid:05d}"
            d.mkdir(parents=True, exist_ok=True)
            settled = _time.time() - 600
            # Substantial but unreadable: big enough to be real media that
            # the muxer closed, so its nominal length is the right credit.
            for i in (0, 1, 3):
                p = d / f"chunk_{i:05d}.ts"
                p.write_bytes(b"\x00" * (200 * 1024))
                _os.utime(p, (settled, settled))
            # A 0-byte crash artifact in the MIDDLE (not the tail, where the
            # bitrate branch would coincidentally also yield 0): crediting it
            # a full segment fabricates 15 minutes of timeline and mistimes
            # every real segment after it.
            zero = d / "chunk_00002.ts"
            zero.write_bytes(b"")
            _os.utime(zero, (settled, settled))

            def unprobeable(p: Path) -> MediaInfo:
                from clipforge.errors import FfmpegError

                raise FfmpegError("zero-filled")

            reconcile_sessions(db, ws, prober=unprobeable, segment_time_s=900.0)

            sess = db.get_session(sid)
            assert int(sess["next_segment"]) == 4, dict(sess)
            rows = {int(r["seg_index"]): r for r in db.segments_for_session(sid)}
            assert float(rows[2]["duration_s"]) == 0.0, (
                "a 0-byte artifact was credited real media time")
            # The muxer-closed ones do get their nominal length.
            assert float(rows[0]["duration_s"]) == 900.0, dict(rows[0])
            # …and the segment AFTER the artifact is not pushed out by it.
            assert float(rows[3]["abs_start_s"]) == 1800.0, dict(rows[3])
            # The real invariant: starts are non-decreasing in index order,
            # and no two segments that CONTAIN media share a start. A
            # zero-duration artifact legitimately shares its successor's
            # start — it occupies no time, which is the whole point.
            ordered = [rows[i] for i in sorted(rows)]
            starts = [float(r["abs_start_s"]) for r in ordered]
            assert starts == sorted(starts), starts
            with_media = [float(r["abs_start_s"]) for r in ordered
                          if float(r["duration_s"] or 0.0) > 0]
            assert len(set(with_media)) == len(with_media), (
                "two segments containing media share an absolute start")
        finally:
            db.close()


@check("a useless connect ends the session (no permanent loop)")
def _useless_connect_terminates() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            _seg_dir(root, sid)
            for captured in (0.0, 0.04, 3.0):
                connects = {"n": 0}
                sess = _session(root, db, [])
                sess._connect_once = lambda stop, s, deadline=None, c=captured: (
                    connects.__setitem__("n", connects["n"] + 1)
                    or ConnectResult(captured_s=c, stalled=c > 0))

                class _NoWait(threading.Event):
                    def wait(self, timeout=None):
                        assert connects["n"] <= 30, (
                            f"run() never gave up on captured_s={captured}")
                        return False

                sess.run(_NoWait(), session_id=sid)
                assert connects["n"] == 3, (captured, connects)
        finally:
            db.close()


@check("reconnect rate is structurally capped (storm cannot relocate)")
def _reconnect_rate_cap() -> None:
    """Three rounds of health-threshold fixes each MOVED the reconnect storm
    into a new band; the last sat just ABOVE the healthy threshold, where
    every counter resets and the session never ends. The cap does not care
    how healthy the connects look."""
    # Deliberately a HARDCODED bound, not MAX_CONNECTS_PER_HOUR: a check
    # that compares against the constant it is testing passes trivially
    # when that constant is neutralized, which is exactly how two earlier
    # "covered" fixes turned out to be revert-safe.
    MAX_ALLOWED_CONNECTS_PER_HOUR = 25

    class _Clock:
        t = 0.0

        def __call__(self) -> float:
            return self.t

        def advance(self, dt: float) -> None:
            self.t += dt

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            for media in (31.0, 45.0):
                clock = _Clock()
                sid = db.open_stream_session("twitch", "t")
                sess = ChunkerSession(
                    db=db, chunks_root=root / "chunks",
                    quarantine_dir=root / "q", platform="twitch", handle="t",
                    streamlink_args=[],
                    cfg=ChunkerConfig(poll_interval_s=0.0), clock=clock,
                    resolve_tools=lambda: ("streamlink", "ffmpeg"))
                connects = {"n": 0}

                def one(stop, s, deadline=None, m=media, c=clock, k=connects):
                    k["n"] += 1
                    c.advance(m)
                    return ConnectResult(captured_s=m, timeline_s=m)

                sess._connect_once = one

                class _Hour(threading.Event):
                    def wait(self, timeout=None):
                        clock.advance(timeout or 0.0)
                        if clock.t >= 3600.0:
                            self.set()
                        return False

                # CALLER-OWNED ledger, driven across TWO sessions on one
                # virtual hour. Passing no ledger= made run() fall back to
                # its standalone `ConnectLedger()`, so this check never
                # touched the caller-owned path at all: reverting the
                # monitor's per-channel ledger left the gate at 20/20 while
                # the storm returned to 73 spawns/hour.
                ledger = ConnectLedger()
                stop = _Hour()
                sess.run(stop, session_id=sid, ledger=ledger)
                if not stop.is_set():
                    sid2 = db.open_stream_session("twitch", "t")
                    sess.run(stop, session_id=sid2, ledger=ledger)
                assert connects["n"] <= MAX_ALLOWED_CONNECTS_PER_HOUR, (
                    media, connects["n"])
                assert ledger.count(clock.t) <= MAX_ALLOWED_CONNECTS_PER_HOUR

        finally:
            db.close()

    # ...and the OWNERSHIP half: the monitor must hand the SAME ledger to
    # every session of a channel. Driving ChunkerSession directly cannot see
    # this, so reverting the monitor to a fresh ConnectLedger per session
    # restored the cross-session storm (73 spawns/hour) with the gate still
    # reading 20/20 — the exact blind spot round 8 measured.
    src = inspect.getsource(ChannelMonitor._tick_live)
    assert "self._ledgers" in src, (
        "the monitor must own a per-channel ledger; one constructed inside "
        "the per-session path is reset by every session hand-back, which is "
        "how the cross-session storm laundered spawns past the cap")
    assert "_ledgers" in ChannelMonitor.__dataclass_fields__, \
        "the ledger map must be monitor state, not a local"


@check("one process per workspace (reconcile cannot close a live session)")
def _workspace_lock() -> None:
    import subprocess as _sp

    child = (
        "import sys; sys.path.insert(0, sys.argv[2]);"
        "from clipforge.paths import Workspace;"
        "from clipforge.ingest.retention import workspace_lock;"
        "ws = Workspace(sys.argv[1]);"
        "print('ACQUIRED' if workspace_lock(ws).__enter__() else 'BLOCKED')"
    )
    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "ws").ensure()
        repo = str(Path(__file__).resolve().parents[2])
        with workspace_lock(ws) as acquired:
            assert acquired is True
            proc = _sp.run([sys.executable, "-c", child, str(ws.root), repo],
                           capture_output=True, text=True, timeout=60)
            lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
            assert lines and lines[-1] == "BLOCKED", (proc.stdout, proc.stderr)


@check("retention frees space; crashed sessions are recovered")
def _retention_and_reconcile() -> None:
    import os
    import time as _time

    with tempfile.TemporaryDirectory() as td:
        ws = Workspace(Path(td) / "ws").ensure()
        db = StateDB(ws.state_db)
        try:
            sid = db.open_stream_session("twitch", "t")
            d = ws.chunks / "twitch_t" / f"s{sid:05d}"
            d.mkdir(parents=True, exist_ok=True)
            aged = d / "chunk_00000.ts"
            aged.write_bytes(b"\x47" * 4096)
            # Bank it the way the chunker does: the session's base_offset_s
            # is the DURABLE timeline anchor, and it must survive retention
            # pruning the segment row once the media itself is gone.
            db.bank_segment_and_offset(sid, 0, aged, 0.0, 900.0, next_segment=1)
            old = _time.time() - 100 * 3600
            os.utime(aged, (old, old))

            report = sweep_retention(db, ws, retention_hours=48.0)
            assert report.files_deleted == 1, report
            assert not aged.exists()

            # A crash left the session open with unregistered media on disk.
            # Age it: reconcile deliberately ignores files still being
            # written (an orphaned recorder can outlive its parent).
            orphan = d / "chunk_00001.ts"
            orphan.write_bytes(b"\x47" * 4096)
            settled = _time.time() - 600
            os.utime(orphan, (settled, settled))
            emitted: list[float] = []
            recovered = reconcile_sessions(
                db, ws, prober=lambda p: _info(900.0),
                on_segment=lambda p, t: emitted.append(t))
            assert recovered == 1, recovered
            assert emitted == [900.0], emitted
            assert db.get_session(sid)["ended_at"] is not None
        finally:
            db.close()


# -------------------------------------------------------- 2. reconnect time


@check("absolute media time survives reconnect")
def _reconnect() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            d = _seg_dir(root, sid)
            events: list[SegmentEvent] = []
            _write_seg(d, 0)
            _write_seg(d, 1)
            _session(root, db, [_FakeProc(2), _FakeProc(2)],
                     on_ready=events.append)._connect_once(threading.Event(), sid)
            row = db.get_session(sid)
            assert row["base_offset_s"] == 1800.0
            assert row["next_segment"] == 2, "must resume into a NEW index"
            _write_seg(d, 2)
            _write_seg(d, 3)
            _session(root, db, [_FakeProc(2), _FakeProc(2)],
                     on_ready=events.append)._connect_once(threading.Event(), sid)
            assert [e.abs_start_s for e in events] == [0.0, 900.0, 1800.0, 2700.0]
        finally:
            db.close()


# ------------------------------------------------------------- 3. T3 shutdown


@check("T3: streamlink killed, ffmpeg waited (tail finalized), no zombies")
def _shutdown() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        db = StateDB(root / "s.db")
        try:
            sid = db.open_stream_session("twitch", "t")
            _seg_dir(root, sid)
            sl, ff = _FakeProc(10 ** 6), _FakeProc(10 ** 6)
            stop = threading.Event()
            stop.set()
            _session(root, db, [sl, ff])._connect_once(stop, sid)
            assert sl.killed, "streamlink must be killed to deliver EOF"
            assert ff.waited and not ff.killed, "ffmpeg must finalize the tail"
        finally:
            db.close()


# ------------------------------------------------- 4. T2 real-ffmpeg segments


@check("T2: hard-killed TS writer still leaves a playable segment")
def _ts_resilience() -> None:
    ffmpeg = find_binary("ffmpeg")
    if ffmpeg is None or find_binary("ffprobe") is None:
        raise _Skip("ffmpeg/ffprobe not installed")
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        # Write a live-ish TS segment stream, then hard-kill the writer.
        proc = subprocess.Popen(
            [str(ffmpeg), "-hide_banner", "-loglevel", "error",
             "-re", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
             "-c:v", "libx264", "-preset", "ultrafast", "-g", "30",
             "-f", "segment", "-segment_format", "mpegts",
             "-segment_time", "2", str(out / "seg_%03d.ts")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=0x08000000 if sys.platform == "win32" else 0)
        try:
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if len(list(out.glob("seg_*.ts"))) >= 2:
                    break
                time.sleep(0.2)
        finally:
            proc.kill()
            proc.wait(timeout=10)
        segs = sorted(out.glob("seg_*.ts"))
        assert len(segs) >= 2, f"expected segments, got {segs}"
        # The FIRST segment (closed before the kill) must be probeable —
        # this is why TS, not MP4 (a killed MP4 has no moov atom).
        info = probe(segs[0])
        assert info.duration_s > 0.5, info
        assert info.v_codec == "h264"


# ------------------------------------------------------------ 5. T1 windows


@check("T1: virtual window + absolute-time dedup")
def _virtual_window() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        prev, chunk = root / "chunk_00000.ts", root / "chunk_00001.ts"
        for p in (prev, chunk):
            p.write_bytes(b"\x47" * 4096)

        def runner(cmd, **kw):
            outp = Path(cmd[-1])
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_bytes(b"\x47" * 1024)

        durations = {"chunk_00000.ts": 900.0, "chunk_00001.ts": 900.0,
                     "chunk_00001.tail.ts": 63.4,
                     "chunk_00001.window.ts": 963.4}
        w = build_virtual_window(
            chunk=chunk, chunk_abs_start_s=900.0, prev_chunk=prev,
            overlap_s=60.0, out_dir=root / "win",
            prober=lambda p: _info(durations.get(Path(p).name, 900.0)),
            runner=runner)
        assert w.overlap_s == 63.4, "must report the REAL overlap"
        assert abs(w.abs_start_s - (900.0 - 63.4)) < 1e-9
        # Same moment from two adjacent windows collapses to one candidate.
        kept = dedup_by_absolute_time([(930.0, 975.0), (931.5, 976.5),
                                       (1200.0, 1245.0)])
        assert kept == [(930.0, 975.0), (1200.0, 1245.0)], kept


# ------------------------------------------------------------- 6. T4 Kick


@check("T4: Kick probes are total (never raise, never block others)")
def _kick_containment() -> None:
    def boom(name, args, **kw):
        raise RuntimeError("cloudflare ate the plugin")

    assert kick.is_live("x", run=boom) is None

    def ingest_error(name, args, **kw):
        raise IngestError("streamlink has no kick plugin")

    assert kick.is_live("x", run=ingest_error) is None

    class _P:
        returncode = 1
        stdout = '{"error": "Unable to open URL: 403 Forbidden"}'
        stderr = ""

    assert kick.is_live("x", run=lambda n, a, **k: _P()) is None


# ---------------------------------------------------------- 7. determinism


@check("backoff is deterministic per (channel, attempt) and bounded")
def _backoff() -> None:
    a = [backoff_delay(i, base_s=5, cap_s=300, seed_key="twitch:x")
         for i in range(10)]
    b = [backoff_delay(i, base_s=5, cap_s=300, seed_key="twitch:x")
         for i in range(10)]
    assert a == b, "Determinism Law: same inputs, same delays"
    assert all(0 < d <= 300 for d in a), a
    assert a != [backoff_delay(i, base_s=5, cap_s=300, seed_key="twitch:y")
                 for i in range(10)], "jitter must decorrelate channels"
    assert backoff_delay(10 ** 6, base_s=5, cap_s=300, seed_key="k") <= 300


# -------------------------------------------------------- 8. authorization


@check("authorization: only enabled, listed channels; Kick gated; disk guard")
def _authorization() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ws = Workspace(root / "ws").ensure()
        db = StateDB(ws.state_db)
        try:
            cfg = AppConfig()
            chans = [
                ChannelSpec(platform="twitch", handle="on", enabled=True),
                ChannelSpec(platform="twitch", handle="off", enabled=False),
                ChannelSpec(platform="kick", handle="k", enabled=True),
            ]
            mon = ChannelMonitor(cfg=cfg, db=db, ws=ws,
                                 watchlist=Watchlist(channels=chans))
            got = [(c.platform, c.handle) for c in mon._authorized_channels()]
            assert got == [("twitch", "on")], got  # kick gated off by default
            cfg.ingest.kick_enabled = True
            got2 = sorted(c.platform for c in mon._authorized_channels())
            assert got2 == ["kick", "twitch"], got2
            # Disk guard pauses rather than filling the drive.
            assert disk_allows(ws.root, 0.001) is True
            assert disk_allows(ws.root, 10_000_000.0) is False
        finally:
            db.close()


# ---------------------------------------------------------------- runner


def main() -> int:
    failures = skipped = 0
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"[PASS] {name}")
        except _Skip as exc:
            skipped += 1
            print(f"[SKIP] {name} ({exc})")
        except Exception:
            failures += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    total = len(_CHECKS)
    print(f"ingestion verify: {total - failures - skipped}/{total} passed, "
          f"{skipped} skipped, {failures} failed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
