"""Regressions for the round-7 findings — each test fails if its fix is
reverted (the standing rule since round 4: a fix with no failing test is
indistinguishable from no fix).
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from clipforge.config import AppConfig, ChannelSpec, Watchlist
from clipforge.errors import FfmpegError, StateError
from clipforge.ffmpeg import MediaInfo
from clipforge.ingest.chunker import (MAX_CONNECT_S, MAX_CONNECTS_PER_HOUR,
                                      MAX_SESSION_S, ChunkerConfig,
                                      ChunkerSession, ConnectLedger,
                                      ConnectResult)
from clipforge.ingest.monitor import ChannelMonitor
from clipforge.ingest.retention import reconcile_sessions
from clipforge.paths import Workspace
from clipforge.state import StateDB


def _info(duration: float) -> MediaInfo:
    return MediaInfo(duration_s=duration, width=1920, height=1080, fps=30.0,
                     fps_rational="30/1", v_codec="h264", a_codec="aac")


def _age(path: Path, hours: float) -> Path:
    import os

    old = time.time() - hours * 3600
    os.utime(path, (old, old))
    return path


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ------------------------------------------------- R7-1: cross-session storm


def test_connect_ledger_survives_session_reentry(tmp_path: Path):
    """The round-7 refutation: a per-run() window reset with each session,
    so three-strike hand-backs laundered the storm past the cap at 61-82
    spawns/hour. The ledger is per CHANNEL: repeated run() calls against the
    same ledger must stay under the cap in any rolling hour."""
    db = StateDB(tmp_path / "s.db")
    clock = Clock()
    ledger = ConnectLedger()
    total_connects = {"n": 0}
    try:
        # Simulate many short sessions: each run() does 3 sub-healthy
        # connects (~25 s each) and hands back, exactly the refuting shape.
        while clock.t < 3600.0:
            sid = db.open_stream_session("twitch", "t")
            sess = ChunkerSession(
                db=db, chunks_root=tmp_path / "chunks",
                quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
                streamlink_args=[], cfg=ChunkerConfig(poll_interval_s=0.0),
                clock=clock, resolve_tools=lambda: ("streamlink", "ffmpeg"))

            def connect(stop, s, deadline=None):
                total_connects["n"] += 1
                clock.advance(25.0)
                return ConnectResult(captured_s=25.0, timeline_s=25.0)

            sess._connect_once = connect  # type: ignore[assignment]

            class Wait(threading.Event):
                def wait(self, timeout=None):
                    clock.advance(timeout or 0.0)
                    if clock.t >= 3600.0:
                        self.set()
                    return False

            sess.run(Wait(), session_id=sid, ledger=ledger)
    finally:
        db.close()

    assert total_connects["n"] <= MAX_CONNECTS_PER_HOUR + 1, (
        f"{total_connects['n']} connects in one hour ACROSS sessions - the "
        "storm laundered through session hand-backs")


def test_ledger_prunes_beyond_the_first_hour():
    """Round-7 meta finding: deleting the window pruning passed the gate
    because tests only drove hour 1. From hour 2 on the cap must still bind
    (a stale unpruned window would either block forever or not at all)."""
    ledger = ConnectLedger()
    # Hour 1: fill to the cap.
    for i in range(MAX_CONNECTS_PER_HOUR):
        ledger.record(float(i))
    assert ledger.cooldown_s(float(MAX_CONNECTS_PER_HOUR)) > 0
    # Hour 2: the old spawns age out; capacity must return...
    t2 = 3700.0
    assert ledger.cooldown_s(t2) == 0.0, "hour-2 capacity never returned"
    # ...and refilling in hour 2 must hit the cap AGAIN.
    for i in range(MAX_CONNECTS_PER_HOUR):
        ledger.record(t2 + i)
    assert ledger.cooldown_s(t2 + MAX_CONNECTS_PER_HOUR) > 0, (
        "cap does not bind in hour 2")


async def test_monitor_passes_one_ledger_per_channel(tmp_path: Path):
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    seen: list[object] = []

    class Recorder:
        def run(self, stop, *, session_id=None, ledger=None):
            seen.append(ledger)

    try:
        cfg = AppConfig()
        cfg.ingest.poll_interval_s = 0.01
        cfg.disk.free_floor_gb = 0.001
        chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
        mon = ChannelMonitor(cfg=cfg, db=db, ws=ws,
                             watchlist=Watchlist(channels=chans),
                             twitch_is_live=lambda h, **kw: True,
                             make_chunker=lambda ch: Recorder())
        await mon._tick_live(chans[0])
        await mon._tick_live(chans[0])
    finally:
        db.close()

    assert seen[0] is not None, "monitor did not pass a ledger"
    assert seen[0] is seen[1], (
        "a fresh ledger per session resets the cap - the exact refutation")


# --------------------------------------------- R7-2: MAX_SESSION_S hard bound


def test_session_deadline_cuts_a_connect_mid_flight(tmp_path: Path, ):
    """Round-7 minor: checking session age only BETWEEN connects made the
    bound soft by up to MAX_CONNECT_S (12 h total). The deadline now reaches
    into the capture loop."""
    db = StateDB(tmp_path / "s.db")
    clock = Clock()
    try:
        sid = db.open_stream_session("twitch", "t")

        d = tmp_path / "chunks" / "twitch_t" / f"s{sid:05d}"
        d.mkdir(parents=True, exist_ok=True)

        class Alive:
            """Healthy pipe: keeps GROWING the tail so neither the no-data
            timeout nor the stall rule ends the connect — only the session
            deadline can."""

            stdout, pid = None, 1
            n = 0

            def poll(self):
                clock.advance(30.0)
                Alive.n += 1
                (d / "chunk_00000.ts").write_bytes(
                    b"\x47" * (100_000 + Alive.n * 4096))
                return None

            def kill(self):
                pass

            def wait(self, timeout=None):
                return 0

        sess = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
            streamlink_args=[], cfg=ChunkerConfig(poll_interval_s=0.0),
            clock=clock, popen=lambda *a, **kw: Alive(),
            resolve_tools=lambda: ("streamlink", "ffmpeg"))

        sess._connect_once(threading.Event(), sid, deadline=300.0)
        assert clock.t < MAX_CONNECT_S, (
            "connect ignored the session deadline and ran to its own ceiling")
        assert clock.t >= 300.0
    finally:
        db.close()


# ------------------------------------- R7-3: recovery gap/quarantine anchors


@pytest.fixture()
def env(tmp_path: Path):
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    yield ws, db
    db.close()


def _mkfile(d: Path, idx: int, size: int = 400_000, *, age_h: float = 1.0,
            suffix: str = ".ts") -> Path:
    p = d / f"chunk_{idx:05d}{suffix}"
    p.write_bytes(b"\x47" * size)
    return _age(p, age_h)


def test_gap_at_index_zero_lands_at_zero(env):
    """Round-7: the cursor seeded with base_offset_s (the timeline END)
    stamped a recovered seg0 at the end of the broadcast."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    _mkfile(d, 0)  # seg0's bank failed live; 1 and 2 banked fine
    db.bank_segment_and_offset(sid, 1, d / "chunk_00001.ts", 15.0, 15.0,
                               next_segment=2)
    db.bank_segment_and_offset(sid, 2, d / "chunk_00002.ts", 30.0, 15.0,
                               next_segment=3)

    reconcile_sessions(db, ws, prober=lambda p: _info(15.0))

    rows = {int(r["seg_index"]): float(r["abs_start_s"])
            for r in db.segments_for_session(sid)}
    assert rows[0] == 0.0, f"index-0 gap stamped at {rows[0]}, expected 0.0"
    assert rows[1] == 15.0 and rows[2] == 30.0
    sess = db.get_session(sid)
    assert float(sess["base_offset_s"]) == 45.0
    assert int(sess["next_segment"]) == 3


def test_quarantined_row_still_anchors_the_walk(env):
    """Round-7: the walk was keyed on FILES, so a row whose media moved to
    quarantine never re-anchored and its slot was reused."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    # Row for idx 0 exists but its media lives in quarantine now.
    q = ws.quarantine / f"s{sid:05d}_chunk_00000.ts"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_bytes(b"\x47" * 100)
    db.record_closed_segment(sid, 0, q, 0.0, 15.0, status="quarantined")
    _mkfile(d, 1)  # crashed before banking

    reconcile_sessions(db, ws, prober=lambda p: _info(15.0))

    rows = {int(r["seg_index"]): float(r["abs_start_s"])
            for r in db.segments_for_session(sid)}
    assert rows[1] == 15.0, (
        f"recovered seg1 at {rows[1]} - the quarantined row's slot was reused")


def test_failing_boots_do_not_compound_the_cursor(env):
    """Round-7: the unconditional writeback carried an in-memory cursor, so
    every boot whose row writes failed COMPOUNDED base_offset (45→90→...).
    Progress must be recomputed from durable rows only."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    db.bank_segment_and_offset(sid, 0, _mkfile(d, 0), 0.0, 15.0,
                               next_segment=1)
    _mkfile(d, 1)  # to be recovered — but its row write will fail

    real = db.record_closed_segment

    def failing(*a, **kw):
        raise StateError("disk I/O error")

    db.record_closed_segment = failing  # type: ignore[method-assign]
    try:
        for _boot in range(3):
            reconcile_sessions(db, ws, prober=lambda p: _info(15.0))
            sess = db.get_session(sid)
            assert float(sess["base_offset_s"]) == 15.0, (
                f"boot {_boot}: cursor compounded to {sess['base_offset_s']}")
            assert [s["id"] for s in db.open_sessions()] == [sid], (
                "session closed despite unrecovered media")
    finally:
        db.record_closed_segment = real  # type: ignore[method-assign]


def test_unsettled_file_stops_later_placements(env):
    """Placing files BEYOND an unsettled one would mistime them — their
    positions depend on its final duration."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    _mkfile(d, 0)                       # settled — recoverable
    growing = d / "chunk_00001.ts"
    growing.write_bytes(b"\x47" * 400_000)  # mtime now, and STILL GROWING
    _mkfile(d, 2)                       # settled but AFTER the unsettled one

    # Round 8: "unsettled" is now proven by size-stability, not by a young
    # mtime alone, so an orphaned recorder must actually keep writing.
    def orphan_writes(_dt: float) -> None:
        with growing.open("ab") as fh:
            fh.write(b"\x47" * 4096)

    registered = reconcile_sessions(db, ws, prober=lambda p: _info(15.0),
                                    sleep=orphan_writes)

    assert registered == 1, "only the segment before the unsettled file"
    indices = {int(r["seg_index"]) for r in db.segments_for_session(sid)}
    assert indices == {0}, f"segment 2 was placed past an unsettled file: {indices}"
    assert [s["id"] for s in db.open_sessions()] == [sid]


def test_mid_session_tail_is_not_credited_full_nominal(env):
    """Round-7 quantified R6-7: a reconnected session has one tail per
    connect; an unprobeable mid-directory tail (small vs siblings) must be
    estimated from bitrate, not credited segment_time_s."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    _mkfile(d, 0, size=900_000)   # probeable: 15 s at 60 KB/s
    _mkfile(d, 1, size=90_000)    # mid-session TAIL, unprobeable (~1.5 s worth)
    _mkfile(d, 2, size=900_000)   # next connect's first segment, probeable

    def prober(p: Path):
        if p.name == "chunk_00001.ts":
            raise FfmpegError("torn tail")
        return _info(15.0)

    reconcile_sessions(db, ws, prober=prober, segment_time_s=900.0)

    rows = {int(r["seg_index"]): r for r in db.segments_for_session(sid)}
    est = float(rows[1]["duration_s"])
    assert est < 5.0, (
        f"mid-session tail credited {est}s - nominal fabrication returns")
    # And segment 2 sits right after the estimate, not 900 s later.
    assert float(rows[2]["abs_start_s"]) == pytest.approx(15.0 + est)


# ------------------------------------------- R7-4: resume positive arm


def test_fresh_reconciled_session_is_resumable(env):
    """Round-7 meta: the positive arm (fresh crash ⇒ resumable) had ZERO
    coverage, so the round-5 over-correction could silently return."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    p = _mkfile(d, 0, age_h=0.02)  # media written ~a minute ago
    # settle guard: pretend it settled (mtime 72s old > 30s settle)

    reconcile_sessions(db, ws, prober=lambda p: _info(15.0))

    assert db.resumable_session("twitch", "t", within_s=300) == sid, (
        "a crash seconds into a broadcast must resume its timeline")


def test_stale_reconciled_session_is_not_resumable(env):
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    _mkfile(d, 0, age_h=48.0)  # two-day-old crash

    reconcile_sessions(db, ws, prober=lambda p: _info(15.0))

    assert db.resumable_session("twitch", "t", within_s=300) is None, (
        "a days-old crashed broadcast must not swallow today's stream")


# --------------------------------------------- R7-5: migration tolerance


def test_migration_tolerates_partially_applied_schema(tmp_path: Path):
    """Intermediate dev builds stamped old version numbers with some new
    columns already present; the ADD COLUMN migration must skip duplicates
    instead of bricking the DB."""
    import sqlite3

    from clipforge.state import SCHEMA_VERSION, _SCHEMA

    p = tmp_path / "mixed.db"
    raw = sqlite3.connect(p)
    raw.executescript(_SCHEMA)  # FULL current schema on disk...
    raw.execute(f"PRAGMA user_version={SCHEMA_VERSION - 1}")  # ...older stamp
    raw.commit()
    raw.close()

    db = StateDB(p)  # must open and migrate the stamp, not raise
    try:
        sid = db.open_stream_session("twitch", "t")
        assert db.get_session(sid)["closed_by_reconcile"] == 0
    finally:
        db.close()
