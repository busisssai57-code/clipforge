"""Retention + startup reconciliation — the disk guard's other half (§6)."""

from pathlib import Path

import pytest

from clipforge.ffmpeg import MediaInfo
from clipforge.ingest.retention import reconcile_sessions, sweep_retention
from clipforge.paths import Workspace
from clipforge.state import StateDB


@pytest.fixture()
def env(tmp_path: Path):
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    yield ws, db
    db.close()


def _info(duration: float) -> MediaInfo:
    return MediaInfo(duration_s=duration, width=1920, height=1080, fps=30.0,
                     fps_rational="30/1", v_codec="h264", a_codec="aac")


def _age(path: Path, hours: float) -> Path:
    import os
    import time

    old = time.time() - hours * 3600
    os.utime(path, (old, old))
    return path


def _seg(ws: Workspace, db: StateDB, sid: int, idx: int, *, age_hours: float,
         status: str = "ready", size: int = 4096) -> Path:
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"chunk_{idx:05d}.ts"
    p.write_bytes(b"\x47" * size)
    _age(p, age_hours)
    db.record_closed_segment(sid, idx, p, idx * 900.0, 900.0, status=status)
    return p


# ------------------------------------------------------------------ retention


def test_aged_media_is_deleted_and_fresh_media_kept(env):
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    old = _seg(ws, db, sid, 0, age_hours=100.0)
    fresh = _seg(ws, db, sid, 1, age_hours=1.0)

    report = sweep_retention(db, ws, retention_hours=48.0)

    assert not old.exists(), "aged chunk must be deleted"
    assert fresh.exists(), "fresh chunk must survive"
    assert report.files_deleted == 1 and report.bytes_freed > 0


def test_retention_actually_frees_the_guard(env):
    """The point of retention: without it the disk floor is a permanent
    stop, since nothing ever frees space."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    for i in range(5):
        _seg(ws, db, sid, i, age_hours=100.0)
    report = sweep_retention(db, ws, retention_hours=48.0)
    assert report.files_deleted == 5


def test_rows_for_vanished_files_are_actually_deleted(env):
    """'Pruned' must mean removed. Counting a dead row while leaving it in
    place meant every later sweep re-counted it forever."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    p = _seg(ws, db, sid, 0, age_hours=1.0)
    p.unlink()
    report = sweep_retention(db, ws, retention_hours=48.0)
    assert report.rows_pruned == 1
    assert db.segments_for_session(sid) == [], "row was counted but not deleted"
    # A second sweep finds nothing left to prune.
    assert sweep_retention(db, ws, retention_hours=48.0).rows_pruned == 0


def test_t1_windows_are_reclaimed(env):
    """Round-2 BLOCKER: T1 windows and tails live under tmp/ and are tracked
    by no DB row, so nothing reclaimed them — the workspace grew at roughly
    double the chunk rate, unboundedly."""
    ws, db = env
    wdir = ws.tmp / "windows_s00001"
    wdir.mkdir(parents=True, exist_ok=True)
    old_win = wdir / "chunk_00001.window.ts"
    old_win.write_bytes(b"\x47" * 8192)
    _age(old_win, 100.0)
    old_tail = wdir / "chunk_00001.tail.ts"
    old_tail.write_bytes(b"\x47" * 2048)
    _age(old_tail, 100.0)
    fresh = wdir / "chunk_00002.window.ts"
    fresh.write_bytes(b"\x47" * 8192)

    report = sweep_retention(db, ws, retention_hours=48.0)

    assert not old_win.exists() and not old_tail.exists()
    assert fresh.exists(), "fresh windows must survive"
    assert report.files_deleted == 2


def test_stranded_media_without_db_rows_is_reclaimed(env):
    """A crash (or a segment whose DB write failed while capture continued)
    leaves media no row references. Without this pass it is an unbounded,
    unreclaimable leak."""
    ws, db = env
    d = ws.chunks / "twitch_t" / "s00001"
    d.mkdir(parents=True, exist_ok=True)
    orphan_ts = d / "chunk_00007.ts"
    orphan_ts.write_bytes(b"\x47" * 4096)
    _age(orphan_ts, 100.0)
    orphan_mp4 = d / "chunk_00008.mp4"   # remuxed, DB write never landed
    orphan_mp4.write_bytes(b"\x00" * 4096)
    _age(orphan_mp4, 100.0)

    report = sweep_retention(db, ws, retention_hours=48.0)

    assert not orphan_ts.exists() and not orphan_mp4.exists()
    assert report.files_deleted == 2


def test_tracked_media_is_not_double_deleted(env):
    """The stranded-media pass must not touch files a row already covers."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    fresh = _seg(ws, db, sid, 0, age_hours=1.0)
    report = sweep_retention(db, ws, retention_hours=48.0)
    assert fresh.exists() and report.files_deleted == 0


def test_recently_written_stranded_file_is_left_alone(env):
    """Never delete something that may still be being written."""
    ws, db = env
    d = ws.chunks / "twitch_t" / "s00001"
    d.mkdir(parents=True, exist_ok=True)
    growing = d / "chunk_00000.ts"
    growing.write_bytes(b"\x47" * 4096)  # mtime = now
    sweep_retention(db, ws, retention_hours=0.0)  # everything is "aged"
    assert growing.exists(), "an actively-written file must not be deleted"


def test_quarantine_ages_out(env):
    ws, db = env
    import os
    import time

    q = ws.quarantine / "s00001_chunk_00000.ts"
    q.parent.mkdir(parents=True, exist_ok=True)
    q.write_bytes(b"x" * 100)
    old = time.time() - 100 * 3600
    os.utime(q, (old, old))
    sweep_retention(db, ws, retention_hours=48.0)
    assert not q.exists()


def test_locked_file_is_skipped_not_fatal(env):
    """A reader/AV holding a file must not abort the sweep."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    p = _seg(ws, db, sid, 0, age_hours=100.0)
    with open(p, "rb"):  # hold it open (Windows: unlink fails)
        report = sweep_retention(db, ws, retention_hours=48.0)
    assert report.files_deleted in (0, 1)  # skipped on Windows, deleted on POSIX


# -------------------------------------------------------------- reconciliation


def test_crashed_session_media_is_recovered(env):
    """A SIGKILL leaves a session open and its media unregistered; restart
    must recover it onto the correct absolute timeline instead of stranding
    it in a directory nothing ever rescans."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    # Segment 0 was recorded before the crash; 1 and 2 were written by the
    # (orphaned) ffmpeg and never registered.
    db.record_closed_segment(sid, 0, d / "chunk_00000.ts", 0.0, 900.0)
    (d / "chunk_00000.ts").write_bytes(b"\x47" * 4096)
    for i in (1, 2):
        p = d / f"chunk_{i:05d}.ts"
        p.write_bytes(b"\x47" * 4096)
        _age(p, 1.0)  # settled: not still being written

    emitted: list[tuple[Path, float]] = []
    recovered = reconcile_sessions(db, ws, prober=lambda p: _info(900.0),
                                   on_segment=lambda p, t: emitted.append((p, t)))

    assert recovered == 2
    assert [t for _p, t in emitted] == [900.0, 1800.0], \
        "recovered media must continue the absolute timeline"
    assert db.get_session(sid)["ended_at"] is not None, "session must be closed"


def test_reconcile_writes_back_session_progress(env):
    """Round-3 BLOCKER: reconcile registered recovered media but never
    advanced the SESSION row, so a resume restarted ffmpeg's
    -segment_start_number on top of the recovered files — destroying them
    and re-using absolute times already emitted to the DAG."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        p = d / f"chunk_{i:05d}.ts"
        p.write_bytes(b"\x47" * 4096)
        _age(p, 1.0)

    reconcile_sessions(db, ws, prober=lambda p: _info(900.0))

    sess = db.get_session(sid)
    assert sess["next_segment"] == 3, (
        "a resumed capture would overwrite the recovered segments")
    assert sess["base_offset_s"] == 2700.0, (
        "a resumed capture would re-use absolute times already emitted")


def test_reconcile_leaves_session_open_when_media_is_unsettled(env):
    """Skipping a still-being-written file and THEN closing the session
    strands it forever — a closed session is never rescanned."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    settled = d / "chunk_00000.ts"
    settled.write_bytes(b"\x47" * 4096)
    _age(settled, 1.0)
    growing = d / "chunk_00001.ts"
    growing.write_bytes(b"\x47" * 4096)  # mtime = now, and still growing

    def orphan_writes(_dt: float) -> None:
        with growing.open("ab") as fh:
            fh.write(b"\x47" * 4096)

    reconcile_sessions(db, ws, prober=lambda p: _info(900.0),
                       sleep=orphan_writes)

    assert [s["id"] for s in db.open_sessions()] == [sid], \
        "session with unrecovered media must stay open for the next boot"


def test_reconcile_estimates_unprobeable_media_instead_of_zero(env):
    """A zero-duration recovery would silently re-time everything after it."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    good = d / "chunk_00000.ts"
    good.write_bytes(b"\x47" * 900_000)   # 900 KB / 900 s = 1000 B/s
    _age(good, 1.0)
    bad = d / "chunk_00001.ts"
    bad.write_bytes(b"\x47" * 450_000)    # ~450 s at that bitrate
    _age(bad, 1.0)

    def prober(p: Path):
        if p.name == "chunk_00001.ts":
            from clipforge.errors import FfmpegError

            raise FfmpegError("corrupt")
        return _info(900.0)

    reconcile_sessions(db, ws, prober=prober)

    rows = {int(r["seg_index"]): r for r in db.segments_for_session(sid)}
    assert rows[1]["duration_s"] > 100.0, "unprobeable media got zero time"
    assert db.get_session(sid)["timeline_estimated"] == 1


def test_zero_byte_artifact_is_credited_no_time(env):
    """Round-5 MAJOR: the muxer-closed nominal had no size floor, so a
    0-byte crash artifact was credited a full segment_time_s — fabricating
    15 minutes of timeline per file and mistiming every real segment after
    it. (The gate previously ENFORCED that fabrication.)"""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    real = d / "chunk_00000.ts"
    real.write_bytes(b"\x47" * 400_000)
    _age(real, 1.0)
    empty = d / "chunk_00001.ts"
    empty.write_bytes(b"")
    _age(empty, 1.0)
    tail = d / "chunk_00002.ts"
    tail.write_bytes(b"\x47" * 400_000)
    _age(tail, 1.0)

    def prober(p: Path):
        if p.name == "chunk_00001.ts":
            from clipforge.errors import FfmpegError

            raise FfmpegError("zero-filled")
        return _info(12.0)

    reconcile_sessions(db, ws, prober=prober, segment_time_s=900.0)

    rows = {int(r["seg_index"]): r for r in db.segments_for_session(sid)}
    assert float(rows[1]["duration_s"]) == 0.0, "0-byte file credited media time"
    # The real segment after it keeps its true position.
    assert float(rows[2]["abs_start_s"]) == 12.0, dict(rows[2])


def test_reconciled_session_is_never_resumed(env):
    """Round-5 MAJOR: reconcile stamps ended_at with NOW, so an ancient
    crashed session fell inside the resume window and today's unrelated
    broadcast was appended to its timeline and directory."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "chunk_00000.ts"
    p.write_bytes(b"\x47" * 400_000)
    _age(p, 1.0)

    reconcile_sessions(db, ws, prober=lambda p: _info(12.0))

    assert db.resumable_session("twitch", "t", within_s=300) is None, (
        "a crash-recovered session must not be merged into a new broadcast")
    # A normally-closed session still resumes.
    sid2 = db.open_stream_session("twitch", "t")
    db.close_stream_session(sid2)
    assert db.resumable_session("twitch", "t", within_s=300) == sid2


def test_recovered_segment_in_a_gap_lands_at_its_true_position(env):
    """Round-5 MINOR: a running cursor ignored known rows, so a segment
    recovered into a GAP was stamped AFTER the later known segments."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        p = d / f"chunk_{i:05d}.ts"
        p.write_bytes(b"\x47" * 400_000)
        _age(p, 1.0)
    # 0 and 2 are known; 1's DB write failed during the crash.
    db.record_closed_segment(sid, 0, d / "chunk_00000.ts", 0.0, 12.0)
    db.record_closed_segment(sid, 2, d / "chunk_00002.ts", 24.0, 12.0)

    reconcile_sessions(db, ws, prober=lambda p: _info(12.0))

    rows = {int(r["seg_index"]): float(r["abs_start_s"])
            for r in db.segments_for_session(sid)}
    assert rows[1] == 12.0, f"gap segment placed at {rows[1]}, expected 12.0"
    assert rows[3] == 36.0, f"segment after the known ones at {rows[3]}"


def test_reconcile_is_idempotent(env):
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "chunk_00000.ts"
    p.write_bytes(b"\x47" * 4096)
    _age(p, 1.0)

    first = reconcile_sessions(db, ws, prober=lambda p: _info(900.0))
    second = reconcile_sessions(db, ws, prober=lambda p: _info(900.0))
    assert first == 1 and second == 0, "a closed session is not re-swept"


def test_reconcile_skips_a_file_still_being_written(env):
    """Round-2 BLOCKER: reconcile registered a partial file as CLOSED,
    emitted the truncated media, and ended the session — re-introducing the
    growing-file failure in the recovery path. An orphaned recorder can
    outlive its parent, so 'settled' must be proven, not assumed.

    Round 8 sharpened HOW it is proven: a young mtime alone marked the dead
    writer's final segment unsettled, which left the session open, which made
    the FRESHEST crash unresumable. Proof is now size-stability under a
    bounded poll. So this test must simulate a file that is genuinely still
    GROWING — the orphaned-recorder case — not merely a young one."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    growing = d / "chunk_00000.ts"
    growing.write_bytes(b"\x47" * 4096)  # mtime = now ⇒ still changing

    def orphan_writes(_dt: float) -> None:
        with growing.open("ab") as fh:
            fh.write(b"\x47" * 4096)

    emitted: list[tuple[Path, float]] = []
    recovered = reconcile_sessions(db, ws, prober=lambda p: _info(6.0),
                                   sleep=orphan_writes,
                                   on_segment=lambda p, t: emitted.append((p, t)))

    assert recovered == 0 and emitted == []
    assert db.segments_for_session(sid) == [], "partial file must not be recorded"


def test_reconcile_recovers_remuxed_mp4(env):
    """Remux is the default, so recovery that globs only .ts is blind to the
    common case."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    mp4 = d / "chunk_00000.mp4"
    mp4.write_bytes(b"\x00" * 4096)
    _age(mp4, 1.0)

    emitted: list[tuple[Path, float]] = []
    recovered = reconcile_sessions(db, ws, prober=lambda p: _info(900.0),
                                   on_segment=lambda p, t: emitted.append((p, t)))
    assert recovered == 1
    assert emitted and emitted[0][0] == mp4


def test_reconcile_advances_clock_past_unprobeable_media(env):
    """A skipped segment must not silently re-time the ones after it."""
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        seg = d / f"chunk_{i:05d}.ts"
        seg.write_bytes(b"\x47" * 4096)
        _age(seg, 1.0)

    def prober(p: Path):
        if p.name == "chunk_00001.ts":
            from clipforge.errors import FfmpegError

            raise FfmpegError("corrupt")
        return _info(900.0)

    emitted: list[tuple[Path, float]] = []
    reconcile_sessions(db, ws, prober=prober,
                       on_segment=lambda p, t: emitted.append((p, t)))

    # Segment 1 is recorded (quarantined) rather than vanishing silently,
    # and the session is flagged as having an estimated timeline.
    rows = {int(r["seg_index"]): r for r in db.segments_for_session(sid)}
    assert set(rows) == {0, 1, 2}
    assert rows[1]["status"] == "quarantined"
    assert db.get_session(sid)["timeline_estimated"] == 1


_LOCK_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[2])
from clipforge.paths import Workspace
from clipforge.ingest.retention import workspace_lock

ws = Workspace(sys.argv[1])
with workspace_lock(ws) as acquired:
    print("ACQUIRED" if acquired else "BLOCKED", flush=True)
"""


def test_workspace_lock_blocks_a_second_process(tmp_path: Path):
    """Round-2 BLOCKER: with no liveness guard, a second instance's boot
    reconcile "recovered" (and closed) the FIRST instance's LIVE session.

    Exclusivity is tested across PROCESSES because that is the real hazard
    and because Windows file locks are owned per-process — a same-process
    re-lock proves nothing.
    """
    import subprocess
    import sys

    from clipforge.ingest.retention import workspace_lock

    ws = Workspace(tmp_path / "ws").ensure()
    repo = str(Path(__file__).resolve().parents[2])

    def child() -> str:
        proc = subprocess.run([sys.executable, "-c", _LOCK_CHILD,
                               str(ws.root), repo],
                              capture_output=True, text=True, timeout=60)
        # The child also logs to stdout; the verdict is its last line.
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        assert lines, f"child produced no output (stderr: {proc.stderr[-500:]})"
        return lines[-1]

    with workspace_lock(ws) as acquired:
        assert acquired is True
        assert child() == "BLOCKED", "a second process must not get the lock"

    # Released on exit — and the OS releases it even on a hard kill, so the
    # lock cannot go stale and block a legitimate restart.
    assert child() == "ACQUIRED"


def test_reconcile_skips_unprobeable_media(env):
    ws, db = env
    sid = db.open_stream_session("twitch", "t")
    d = ws.chunks / "twitch_t" / f"s{sid:05d}"
    d.mkdir(parents=True, exist_ok=True)
    bad_file = d / "chunk_00000.ts"
    bad_file.write_bytes(b"garbage")
    _age(bad_file, 1.0)

    def bad(p):
        from clipforge.errors import FfmpegError

        raise FfmpegError("not media")

    emitted: list[Path] = []
    # Unprobeable media is REGISTERED (so the timeline and the operator's
    # count reflect it) but never EMITTED to the DAG.
    registered = reconcile_sessions(db, ws, prober=bad,
                                    on_segment=lambda p, t: emitted.append(p))
    assert registered == 1 and emitted == []
    rows = db.segments_for_session(sid)
    assert len(rows) == 1 and rows[0]["status"] == "quarantined"
    assert db.get_session(sid)["timeline_estimated"] == 1


def test_reconcile_handles_missing_directory(env):
    ws, db = env
    db.open_stream_session("twitch", "gone")
    assert reconcile_sessions(db, ws, prober=lambda p: _info(1.0)) == 0
