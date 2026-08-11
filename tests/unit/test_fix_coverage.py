"""Coverage for fixes that were previously revert-safe.

Round 4 raised a sharp meta-finding: four shipped fixes had NO gate
coverage — reverting each to its pre-fix state left the whole suite green.
After three rounds in which every round refuted the previous round's fixes,
an uncovered fix is indistinguishable from no fix at all.

Each test here fails if its fix is reverted. They are grouped in one module
so the invariant "every fix has a test that would catch its removal" is
visible in one place rather than scattered.
"""

import asyncio
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from clipforge.config import Watchlist
from clipforge.errors import ConfigError
from clipforge.ingest import kick, twitch
from clipforge.ingest.chunker import ChunkerConfig, ChunkerSession
from clipforge.ingest.retention import (MEDIA_GLOBS, UNTRACKED_MEDIA_GLOBS,
                                        sweep_retention)
from clipforge.paths import Workspace
from clipforge.state import StateDB


def _age(path: Path, hours: float) -> Path:
    import os

    old = time.time() - hours * 3600
    os.utime(path, (old, old))
    return path


# ------------------------------------------------- R3-4: dedicated housekeeper


async def test_housekeeping_has_its_own_thread_pool(tmp_path: Path):
    """Reverting to the shared ingest executor must fail here.

    The realistic saturation case is YouTube-only: the pool is sized
    ``live_channels + ingest_concurrency``, which is 1 when nothing is live,
    while a VOD download holds its thread for the length of the download
    (up to an hour). Housekeeping queued behind that never runs.
    """
    from clipforge.config import AppConfig, ChannelSpec
    from clipforge.ingest.monitor import ChannelMonitor
    import clipforge.ingest.monitor as monitor_mod

    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    cfg = AppConfig()
    cfg.workspace.root = ws.root
    cfg.ingest.poll_interval_s = 0.05
    cfg.disk.free_floor_gb = 0.001
    cfg.orchestration.ingest_concurrency = 1  # ⇒ a single shared ingest thread
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    downloading = threading.Event()

    def slow_download(vid, dest, **kw):
        downloading.wait(timeout=10)  # occupies the ingest thread
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"{vid}.mp4"
        p.write_bytes(b"x")
        return p

    swept = {"n": 0}
    mon = ChannelMonitor(cfg=cfg, db=db, ws=ws,
                         watchlist=Watchlist(channels=chans),
                         discover=lambda d, h, e: ["aaaaaaaaaaa"],
                         download=slow_download)
    orig_sweep, orig_iv = monitor_mod.sweep_retention, monitor_mod.RETENTION_INTERVAL_S
    monitor_mod.sweep_retention = lambda *a, **kw: swept.__setitem__("n", swept["n"] + 1)
    monitor_mod.RETENTION_INTERVAL_S = 0.05
    try:
        task = asyncio.create_task(mon.run())
        await asyncio.sleep(0.6)
        mon.stop()
        downloading.set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        monitor_mod.sweep_retention = orig_sweep
        monitor_mod.RETENTION_INTERVAL_S = orig_iv
        db.close()

    assert swept["n"] >= 3, (
        f"housekeeping ran {swept['n']}x while the only ingest thread was "
        "held by a download - it is sharing the starved pool")


def test_housekeeping_pool_is_structurally_separate():
    """Belt and braces: the sweep must not be dispatched on the ingest pool."""
    import inspect

    from clipforge.ingest.monitor import ChannelMonitor

    src = inspect.getsource(ChannelMonitor._housekeeping_loop)
    assert "_housekeeper" in src, "housekeeping is not using its own pool"
    assert "_to_thread" not in src, (
        "housekeeping dispatches through the shared ingest executor")


# ------------------------------------------------------- R3-6: VOD reclamation


def test_vod_media_is_reclaimable(tmp_path: Path):
    """A YouTube-only deployment must have SOME reclamation path, or the
    disk floor eventually pauses ingestion permanently."""
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    try:
        d = ws.chunks / "youtube" / "somechannel"
        d.mkdir(parents=True, exist_ok=True)
        vod = d / "yt_dQw4w9WgXcQ.mp4"
        vod.write_bytes(b"\x00" * 8192)
        _age(vod, 100.0)
        part = d / "yt_abcdefghijk.mp4.part"
        part.write_bytes(b"\x00" * 4096)
        _age(part, 100.0)

        report = sweep_retention(db, ws, retention_hours=48.0)

        assert not vod.exists(), "downloaded VOD media is unreclaimable"
        assert not part.exists(), "yt-dlp partial files are unreclaimable"
        assert report.files_deleted == 2
    finally:
        db.close()


def test_untracked_globs_cover_the_vod_shapes():
    """Guards the glob list itself against silent narrowing."""
    joined = " ".join(UNTRACKED_MEDIA_GLOBS)
    assert "yt_*" in joined and ".part" in joined
    assert MEDIA_GLOBS == ("chunk_*.ts", "chunk_*.mp4")


def test_fresh_vod_download_is_not_deleted(tmp_path: Path):
    """The reclamation must not eat a download still in progress."""
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    try:
        d = ws.chunks / "youtube" / "c"
        d.mkdir(parents=True, exist_ok=True)
        active = d / "yt_inprogress.mp4.part"
        active.write_bytes(b"\x00" * 4096)  # mtime = now
        sweep_retention(db, ws, retention_hours=0.0)
        assert active.exists(), "an in-progress download was deleted"
    finally:
        db.close()


# ------------------------------------------- R3-8a: lock before partial sweep


def test_boot_can_defer_the_partial_sweep(tmp_path: Path):
    """`watch` must not delete *.partial before it owns the workspace: those
    files include a RUNNING instance's in-flight remux and window temps."""
    from clipforge.cli import _boot

    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    (cfg_dir / "config.toml").write_text(
        f"[workspace]\nroot = '{(tmp_path / 'ws').as_posix()}'\n", encoding="utf-8")
    ws_dir = tmp_path / "ws"
    ws_dir.mkdir(parents=True, exist_ok=True)
    debris = ws_dir / "chunk_00000.mp4.remux.partial"
    debris.write_bytes(b"in-flight")

    _boot(cfg_dir / "config.toml", sweep_partials=False)
    assert debris.exists(), "sweep ran before the lock was held"

    _boot(cfg_dir / "config.toml", sweep_partials=True)
    assert not debris.exists(), "deferred sweep never cleaned crash debris"


def test_process_command_does_not_sweep_without_the_lock():
    """`process` holds no lock, so it must not sweep either."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.process)
    assert "sweep_partials=False" in src, (
        "process() sweeps *.partial workspace-wide without holding the lock")


def test_watch_sweeps_only_after_taking_the_lock():
    """The ordering itself is the fix: booting with the default sweep would
    delete a RUNNING instance's in-flight remux/window temps before this
    process discovers it does not own the workspace."""
    import inspect

    from clipforge import cli

    watch_src = inspect.getsource(cli.watch)
    assert "sweep_partials=False" in watch_src, (
        "watch() sweeps *.partial before acquiring the workspace lock")
    # And the sweep does happen, once the lock is held.
    locked_src = inspect.getsource(cli._watch_locked)
    assert "discard_partials" in locked_src, (
        "crash debris is never swept at all")


# --------------------------------------------- R3-8b: duplicate channel guard


def test_duplicate_channels_are_rejected():
    """Two loops for one channel race the same downloads into the same file."""
    with pytest.raises(Exception) as exc:
        Watchlist.model_validate({"channels": [
            {"platform": "twitch", "handle": "same"},
            {"platform": "twitch", "handle": "SAME"},
        ]})
    assert "duplicate" in str(exc.value).lower()


def test_at_handle_variants_are_the_same_channel():
    with pytest.raises(Exception):
        Watchlist.model_validate({"channels": [
            {"platform": "youtube", "handle": "@chan"},
            {"platform": "youtube", "handle": "chan"},
        ]})


def test_distinct_channels_still_allowed():
    wl = Watchlist.model_validate({"channels": [
        {"platform": "twitch", "handle": "a"},
        {"platform": "youtube", "handle": "a"},   # different platform
        {"platform": "twitch", "handle": "b"},
    ]})
    assert len(wl.channels) == 3


def test_duplicate_message_is_ascii():
    """Operator-facing text must survive a cp1252 pipe (CP0 round-2 rule)."""
    try:
        Watchlist.model_validate({"channels": [
            {"platform": "twitch", "handle": "x"},
            {"platform": "twitch", "handle": "x"},
        ]})
        raise AssertionError("expected a duplicate-channel error")
    except Exception as exc:
        str(exc).encode("ascii")


# --------------------------------------- reconnect rate limit (structural)


def _rate_limit_session(tmp_path: Path, db: StateDB, media_per_connect: float,
                        clock) -> int:
    """Drive run() for one virtual hour, returning the connect count."""
    from clipforge.ingest.chunker import ConnectResult

    sid = db.open_stream_session("twitch", "t")
    connects = {"n": 0}
    sess = ChunkerSession(
        db=db, chunks_root=tmp_path / "chunks", quarantine_dir=tmp_path / "q",
        platform="twitch", handle="t", streamlink_args=[],
        cfg=ChunkerConfig(poll_interval_s=0.0), clock=clock,
        resolve_tools=lambda: ("streamlink", "ffmpeg"))

    def connect(stop, s, deadline=None):
        connects["n"] += 1
        clock.advance(media_per_connect)  # the connect takes this long
        return ConnectResult(captured_s=media_per_connect,
                             timeline_s=media_per_connect)

    sess._connect_once = connect  # type: ignore[assignment]

    class HourCap(threading.Event):
        def wait(self, timeout=None):
            clock.advance(timeout or 0.0)
            if clock.t >= 3600.0:
                self.set()
            return False

    sess.run(HourCap(), session_id=sid)
    return connects["n"]


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


@pytest.mark.parametrize("media_per_connect", [31.0, 35.0, 45.0])
def test_reconnect_rate_is_structurally_capped(tmp_path: Path,
                                               media_per_connect: float):
    """Three rounds of health-threshold fixes each MOVED the reconnect storm
    into a new band; the last landed just ABOVE the healthy threshold, where
    every counter resets and the session never ends (~107 connects/hour).

    The cap is structural: whatever the media numbers say, the process may
    not re-spawn streamlink more often than MAX_CONNECTS_PER_HOUR.
    """
    from clipforge.ingest.chunker import MAX_CONNECTS_PER_HOUR

    db = StateDB(tmp_path / f"s{media_per_connect}.db")
    try:
        count = _rate_limit_session(tmp_path, db, media_per_connect, _Clock())
    finally:
        db.close()
    assert count <= MAX_CONNECTS_PER_HOUR + 1, (
        f"{count} connects in one virtual hour at {media_per_connect}s/connect")


def test_session_has_an_absolute_age_ceiling(tmp_path: Path):
    """The three-strike rule alone could hold a channel for 3 x
    MAX_CONNECT_S (18 h measured), during which the monitor never re-polls
    is_live and channel-level policy cannot apply."""
    from clipforge.ingest.chunker import MAX_SESSION_S, ConnectResult

    db = StateDB(tmp_path / "s.db")
    clock = _Clock()
    try:
        sid = db.open_stream_session("twitch", "t")
        sess = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
            streamlink_args=[], cfg=ChunkerConfig(poll_interval_s=0.0),
            clock=clock, resolve_tools=lambda: ("streamlink", "ffmpeg"))

        def long_healthy(stop, s, deadline=None):
            clock.advance(3600.0)  # an hour of healthy capture per connect
            return ConnectResult(captured_s=3600.0, timeline_s=3600.0)

        sess._connect_once = long_healthy  # type: ignore[assignment]

        class NoWait(threading.Event):
            def wait(self, timeout=None):
                clock.advance(timeout or 0.0)
                assert clock.t < MAX_SESSION_S * 3, "session never handed back"
                return False

        sess.run(NoWait(), session_id=sid)
        assert clock.t <= MAX_SESSION_S + 3600.0, clock.t
    finally:
        db.close()


def test_no_data_connect_times_out(tmp_path: Path):
    """A pipe that stays alive but never produces a segment must not spin
    forever — that bypassed every termination guard."""
    from clipforge.ingest.chunker import FIRST_SEGMENT_TIMEOUT_S

    db = StateDB(tmp_path / "s.db")
    clock = _Clock()
    try:
        sid = db.open_stream_session("twitch", "t")

        class Alive:
            stdout = None
            pid = 1

            def poll(self):
                clock.advance(1.0)
                return None

            def kill(self): pass

            def wait(self, timeout=None): return 0

        sess = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
            streamlink_args=[], cfg=ChunkerConfig(poll_interval_s=0.0),
            clock=clock, popen=lambda *a, **kw: Alive(),
            resolve_tools=lambda: ("streamlink", "ffmpeg"))

        result = sess._connect_once(threading.Event(), sid)
        assert result.captured_s == 0.0
        assert clock.t >= FIRST_SEGMENT_TIMEOUT_S
        assert clock.t < FIRST_SEGMENT_TIMEOUT_S * 3, "timeout did not fire"
    finally:
        db.close()


# --------------------------------------------- schema migration is forward


def test_existing_db_migrates_instead_of_becoming_unopenable(tmp_path: Path):
    """A schema bump must not discard an operator's session/VOD history."""
    import sqlite3

    from clipforge.state import SCHEMA_VERSION

    from clipforge.state import _SCHEMA

    # Build a genuine PREVIOUS-version database: stream_sessions without the
    # columns this version added. Creating it first means _SCHEMA's own
    # CREATE TABLE IF NOT EXISTS leaves it alone.
    p = tmp_path / "old.db"
    raw = sqlite3.connect(p)
    raw.execute(
        "CREATE TABLE stream_sessions ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " platform TEXT NOT NULL, handle TEXT NOT NULL,"
        " started_at REAL NOT NULL, ended_at REAL,"
        " base_offset_s REAL NOT NULL DEFAULT 0,"
        " next_segment INTEGER NOT NULL DEFAULT 0,"
        " timeline_estimated INTEGER NOT NULL DEFAULT 0)")
    raw.executescript(_SCHEMA)
    raw.execute("INSERT INTO stream_sessions(platform, handle, started_at) "
                "VALUES('twitch','t',1.0)")
    raw.execute(f"PRAGMA user_version={SCHEMA_VERSION - 1}")
    raw.commit()
    raw.close()

    db2 = StateDB(p)  # must MIGRATE, not raise
    try:
        sessions = db2.open_sessions()
        assert sessions, "an operator's session history was discarded"
        assert sessions[0]["closed_by_reconcile"] == 0
        assert sessions[0]["last_media_at"] == 0
    finally:
        db2.close()


# ------------------------------------------- T1 must not bridge a hole


def test_quarantine_clears_the_t1_predecessor(tmp_path: Path):
    """Reverting `state.last_good = None` on quarantine makes T1 splice
    NON-ADJACENT media across a hole — the window would join a segment to
    one that is minutes earlier on the timeline."""
    from clipforge.ffmpeg import MediaInfo
    from clipforge.ingest.chunker import SegmentEvent

    db = StateDB(tmp_path / "s.db")
    events: list[SegmentEvent] = []
    try:
        sid = db.open_stream_session("twitch", "t")
        d = tmp_path / "chunks" / "twitch_t" / f"s{sid:05d}"
        d.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            (d / f"chunk_{i:05d}.ts").write_bytes(b"\x47" * 400_000)

        def prober(p: Path):
            if p.name == "chunk_00001.ts":
                raise ValueError("corrupt")
            return MediaInfo(duration_s=900.0, width=1920, height=1080,
                             fps=30.0, fps_rational="30/1", v_codec="h264",
                             a_codec="aac")

        class Proc:
            def __init__(self): self.n = 0
            def poll(self): self.n += 1; return 0 if self.n > 2 else None
            def kill(self): pass
            def wait(self, timeout=None): return 0
            stdout = None
            pid = 1

        sess = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
            streamlink_args=[], on_segment_ready=events.append,
            cfg=ChunkerConfig(poll_interval_s=0.0, remux_to_mp4=False),
            popen=lambda *a, **kw: Proc(), prober=prober,
            resolve_tools=lambda: ("streamlink", "ffmpeg"))
        sess._connect_once(threading.Event(), sid)
    finally:
        db.close()

    after_hole = [e for e in events if e.seg_index == 2]
    assert after_hole, "segment after the hole was never emitted"
    assert after_hole[0].prev_path is None, (
        "T1 would splice across a quarantined hole to non-adjacent media")


# --------------------------------------------- remux failure must degrade


def test_remux_failure_emits_the_playable_ts(tmp_path: Path):
    """Reverting the degrade-to-.ts path records and emits a path that does
    not exist on disk — the DAG would receive a phantom file."""
    from clipforge.errors import FfmpegError
    import clipforge.ingest.chunker as chunker_mod

    db = StateDB(tmp_path / "s.db")
    try:
        sess = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
            streamlink_args=[], cfg=ChunkerConfig(remux_to_mp4=True),
            resolve_tools=lambda: ("streamlink", "ffmpeg"))
        src = tmp_path / "chunk_00000.ts"
        src.write_bytes(b"\x47" * 4096)

        original = chunker_mod.remux_ts_to_mp4
        chunker_mod.remux_ts_to_mp4 = lambda s, d: (_ for _ in ()).throw(
            FfmpegError("no space left"))
        try:
            out = sess._remux_if_configured(src)
        finally:
            chunker_mod.remux_ts_to_mp4 = original

        assert out == src, "a failed remux must degrade to the .ts"
        assert out.exists(), "emitted path does not exist on disk"
    finally:
        db.close()


# ------------------------------------------- CLI single-instance enforcement


def test_watch_refuses_without_the_workspace_lock():
    """Reverting the CLI's lock check lets a second instance run, whose boot
    reconcile then closes the first instance's LIVE session."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.watch)
    assert "workspace_lock" in src, "watch() does not take the workspace lock"
    assert "typer.Exit(3)" in src, (
        "watch() proceeds even when another instance holds the workspace")


def test_operator_facing_exception_messages_are_ascii():
    """CP0 round-2 rule: text an operator may see through a cp1252 pipe."""
    from clipforge.errors import AtomicWriteError, FatalStageError
    from clipforge.gpu import GB, ModelClass, vram_guard
    from clipforge.stages.base import digest_params

    def message_of(fn) -> str:
        try:
            fn()
        except Exception as exc:
            return str(exc)
        raise AssertionError("expected an exception")

    messages = [
        message_of(lambda: vram_guard(8.0, ModelClass.ASR, probe=lambda: 1 * GB)),
        message_of(lambda: digest_params({1: "a"})),
        message_of(lambda: digest_params({"s": {1, 2}})),
    ]
    for msg in messages:
        msg.encode("ascii")


# ------------------------------------------------ stall-kill vs readiness


def test_stall_kill_is_more_patient_than_readiness(tmp_path: Path):
    """ffmpeg flushes in ~256 KB steps, so a healthy sub-100 kbps stream
    leaves st_size unchanged for well over the 20 s readiness window.
    Killing the pipe on that would sever a working broadcast."""
    db = StateDB(tmp_path / "s.db")
    try:
        sess = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "q", platform="twitch", handle="t",
            streamlink_args=[], cfg=ChunkerConfig(ready_stable_s=20.0),
            resolve_tools=lambda: ("streamlink", "ffmpeg"))
        assert sess._stall_kill_s() >= 60.0
        assert sess._stall_kill_s() > sess.cfg.ready_stable_s
    finally:
        db.close()
