"""Round-8 findings: coverage that FAILS when each fix is reverted.

Every assertion here compares against a HARDCODED bound, never against the
constant under test. Round 8's meta domain caught two "covered" fixes whose
tests compared a value to the very constant they were pinning, so raising
the constant kept them green.
"""

from __future__ import annotations

import threading
import time

import pytest

from clipforge.errors import FfmpegError
from clipforge.ffmpeg import MediaInfo
from clipforge.ingest import chunker as ck
from clipforge.ingest import monitor as mon
from clipforge.ingest import retention as ret
from clipforge.ingest.chunker import ChunkerConfig, ChunkerSession, ConnectResult
from clipforge.ingest.retention import reconcile_sessions
from clipforge.paths import Workspace
from clipforge.state import StateDB


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


class _VirtualStop(threading.Event):
    """stop.wait() advances the virtual clock instead of sleeping."""

    def __init__(self, clock: _Clock, horizon: float) -> None:
        super().__init__()
        self.clock, self.horizon = clock, horizon

    def wait(self, timeout=None):  # type: ignore[override]
        self.clock.advance(timeout or 0.0)
        if self.clock.t >= self.horizon:
            self.set()
        return False


def _session(tmp_path, clock, *, per_connect: float):
    db = StateDB(tmp_path / "s.db")
    sess = ChunkerSession(
        db=db, chunks_root=tmp_path / "chunks", quarantine_dir=tmp_path / "q",
        platform="twitch", handle="t", streamlink_args=[],
        cfg=ChunkerConfig(poll_interval_s=0.0), clock=clock,
        resolve_tools=lambda: ("streamlink", "ffmpeg"))
    spawns = {"n": 0}

    def one(stop, sid, deadline=None):
        spawns["n"] += 1
        clock.advance(per_connect)
        return ConnectResult(captured_s=per_connect, timeline_s=per_connect)

    sess._connect_once = one  # type: ignore[method-assign]
    return db, sess, spawns


# --------------------------------------------------------------------------
# R8-storm-2: the cap admitted 21 spawns per sliding hour, not 20.
# --------------------------------------------------------------------------


def test_waiting_the_returned_cooldown_provably_frees_a_slot():
    """Without COOLDOWN_MARGIN_S this is a LIVE-LOCK, not just an off-by-one.

    `3600 - (now - oldest)` added back to `now` lands at 3599.999999999999x
    in float — still inside the window. The old code recorded anyway (21st
    spawn); re-checking instead spins on ~1e-13 s waits. Both are caught
    here, immediately, instead of by a test-suite timeout.
    """
    led = ck.ConnectLedger()
    for i in range(20):
        led.record(1000.0 + i * 0.5)
    now = 1010.0
    # Snapshot the boundary BEFORE any call: count()/cooldown_s() prune, so
    # reading min(times) afterwards measures a different oldest entry.
    boundary = 3600.0 - (now - min(led.times))
    cooldown = led.cooldown_s(now)
    assert cooldown > 0.0, "a saturated ledger must throttle"
    # The margin must OVERSHOOT that boundary. Landing exactly on it is the
    # bug; whether float rounding bites depends on the particular values, so
    # pin the overshoot itself rather than hope a synthetic case happens to
    # be non-representable.
    assert cooldown > boundary, (
        f"cooldown {cooldown} lands on the {boundary} boundary; the oldest "
        "spawn is then still inside `now - t < 3600`, so the next spawn is "
        "either a 21st in the hour or an infinite re-check")
    assert led.count(now + cooldown) < 20, (
        "waiting the full returned cooldown did not free a slot")


def test_no_sliding_hour_admits_more_than_twenty_spawns(tmp_path):
    clock = _Clock()
    db, sess, spawns = _session(tmp_path, clock, per_connect=31.0)
    ledger = ck.ConnectLedger()
    stop = _VirtualStop(clock, horizon=6.5 * 3600.0)
    sid = db.open_stream_session("twitch", "t")
    while not stop.is_set():
        sess.run(stop, session_id=sid, ledger=ledger)
        sid = db.open_stream_session("twitch", "t")

    # Hard bound, deliberately NOT MAX_CONNECTS_PER_HOUR.
    for start in [h * 600.0 for h in range(int(clock.t / 600.0))]:
        in_window = [t for t in ledger.times if start <= t < start + 3600.0]
        assert len(in_window) <= 20, (start, len(in_window))
    assert spawns["n"] >= 100, "harness must actually exercise the cap"


# --------------------------------------------------------------------------
# R8-storm-3: MAX_SESSION_S was overshot by up to one full cooldown, then one
# guaranteed-sterile spawn was issued. Measured overshoot: +3209.6 s.
# --------------------------------------------------------------------------


def test_session_never_outlives_its_age_ceiling_by_a_cooldown(tmp_path):
    clock = _Clock()
    db, sess, spawns = _session(tmp_path, clock, per_connect=31.0)
    ledger = ck.ConnectLedger()
    stop = _VirtualStop(clock, horizon=1e9)  # only the age ceiling may stop us
    sid = db.open_stream_session("twitch", "t")
    started = clock.t
    sess.run(stop, session_id=sid, ledger=ledger)
    age = clock.t - started

    # One in-flight connect of slack is legitimate; a whole cooldown is not.
    assert age <= 6 * 3600.0 + 120.0, f"session ran {age:.1f}s"
    # ...and the run must not have ended by issuing a spawn past the ceiling.
    assert not ledger.times or max(ledger.times) - started < 6 * 3600.0


# --------------------------------------------------------------------------
# R8-meta: MAX_CONNECT_S / MAX_SESSION_S / SESSION_RESUME_WINDOW_S /
# UNPRODUCTIVE_SESSION_S were all revert-safe — neutralizing any of them to
# 1e9 (or 0.0) left the entire gate green.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name,expected", [
    ("MAX_CONNECT_S", 6 * 3600.0),
    ("MAX_SESSION_S", 6 * 3600.0),
    ("MAX_CONNECTS_PER_HOUR", 20),
])
def test_chunker_ceilings_hold_their_values(name, expected):
    """Pins the CONSTANTS themselves.

    A behavioural test cannot see MAX_CONNECT_S without running a two-hour
    capture, so the constant is pinned directly. Neutralizing it to 1e9 —
    which removes every wall-clock bound on a live capture — now fails here.
    """
    assert getattr(ck, name) == expected


@pytest.mark.parametrize("name,expected", [
    ("SESSION_RESUME_WINDOW_S", 300.0),
    ("UNPRODUCTIVE_SESSION_S", 60.0),
])
def test_monitor_windows_hold_their_values(name, expected):
    """SESSION_RESUME_WINDOW_S widened to 1e9 makes a years-old recovered
    broadcast swallow today's unrelated stream into one session id and one
    timeline. UNPRODUCTIVE_SESSION_S at 0.0 disables the barren-session
    backoff entirely — its own docstring says that produces a sustained
    reconnect storm. Neither was visible to any gate."""
    assert getattr(mon, name) == expected


def test_retention_constants_hold_their_values():
    assert ret.SETTLE_SECONDS == 30.0
    assert ret.MIN_SEGMENT_BYTES == 188 * 64


# --------------------------------------------------------------------------
# R8-recovery-1 (NOT CURED): R7-3's unsettled-file rule and R7-4's freshness
# fix cancelled out — the FRESHEST crash left the session open, and an open
# session is unresumable, so the one case resume exists for could not resume.
# --------------------------------------------------------------------------


def test_a_dead_writers_final_segment_settles_by_waiting(tmp_path):
    """A file whose mtime is younger than SETTLE_SECONDS but whose size has
    stopped changing is a DEAD writer's tail, not a live writer."""
    path = tmp_path / "chunk_00000.ts"
    path.write_bytes(b"\x47" * 20000)
    slept: list[float] = []
    assert ret._await_settled(path, settle_s=30.0, sleep=slept.append) is True
    assert slept, "must actually poll"


def test_a_live_writer_is_still_refused(tmp_path):
    """The stop-the-walk path must survive: a growing file never settles."""
    path = tmp_path / "chunk_00000.ts"
    path.write_bytes(b"\x47" * 1000)

    def grow(_dt: float) -> None:
        with path.open("ab") as fh:
            fh.write(b"\x47" * 1000)

    assert ret._await_settled(path, settle_s=2.0, sleep=grow) is False


# --------------------------------------------------------------------------
# R8-recovery-2 (NEW MAJOR): reconcile stamps ended_at with "now", so ranking
# resume candidates on ended_at made a crash-recovered session ALWAYS outrank
# a normally-closed one — the monitor resumed the staler broadcast.
# --------------------------------------------------------------------------


def _mkseg(d, idx: int, size: int, *, age_h: float = 1.0):
    import os
    p = d / f"chunk_{idx:05d}.ts"
    p.write_bytes(b"\x47" * size)
    old = time.time() - age_h * 3600
    os.utime(p, (old, old))
    return p


def _minfo(duration: float) -> MediaInfo:
    return MediaInfo(duration_s=duration, width=1920, height=1080, fps=30.0,
                     fps_rational="30/1", v_codec="h264", a_codec="aac")


# --------------------------------------------------------------------------
# R8-recovery-3: the virtual anchor was suppressed whenever the anchor
# index's MEDIA survived without its row, re-timing the whole surviving
# suffix to 0.0 — the very mistiming the anchor exists to prevent.
# --------------------------------------------------------------------------


def test_anchor_holds_when_the_anchor_indexs_file_outlived_its_row(tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    try:
        sid = db.open_stream_session("twitch", "t")
        d = ws.chunks / "twitch_t" / f"s{sid:05d}"
        d.mkdir(parents=True, exist_ok=True)
        # Banked: indices 0..5 ended at 5400 s. Retention pruned every ROW,
        # but index 5's FILE is still on disk (rows and files are pruned
        # independently). Index 6 is the surviving suffix.
        db.set_session_progress(sid, 5400.0, 6)
        _mkseg(d, 5, 900_000)
        _mkseg(d, 6, 900_000)

        reconcile_sessions(db, ws, prober=lambda p: _minfo(900.0),
                           segment_time_s=900.0)

        rows = {int(r["seg_index"]): float(r["abs_start_s"])
                for r in db.segments_for_session(sid)}
        assert 6 in rows, rows
        assert rows[6] == pytest.approx(5400.0), (
            f"suffix re-timed to {rows[6]} — the banked anchor was suppressed "
            "because index 5's media outlived its row")
    finally:
        db.close()


# --------------------------------------------------------------------------
# R8-recovery-4: the 0.6x size test meant any unprobeable tail in the UPPER
# 40% of sizes was still credited a full nominal segment.
# --------------------------------------------------------------------------


def test_a_large_unprobeable_tail_is_not_credited_a_whole_segment(tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    try:
        sid = db.open_stream_session("twitch", "t")
        d = ws.chunks / "twitch_t" / f"s{sid:05d}"
        d.mkdir(parents=True, exist_ok=True)
        _mkseg(d, 0, 900_000)          # probeable: 900 s at 1000 B/s
        _mkseg(d, 1, 900_000)          # probeable
        _mkseg(d, 2, 700_000)          # MID-directory tail, 78% of typical
        _mkseg(d, 3, 900_000)          # next connect's first segment
        # Index 2 must NOT be the last index: `idx == indices[-1]` alone
        # already routed trailing tails to the bitrate branch, so a trailing
        # tail cannot distinguish the fix from the 0.6x gate it replaced.

        def prober(p):
            if p.name == "chunk_00002.ts":
                raise FfmpegError("truncated tail: no duration")
            return _minfo(900.0)

        reconcile_sessions(db, ws, prober=prober, segment_time_s=900.0)

        rows = {int(r["seg_index"]): float(r["duration_s"] or 0.0)
                for r in db.segments_for_session(sid)}
        # 700_000 B / 1000 B/s = 700 s, NOT the 900 s nominal.
        assert rows[2] == pytest.approx(700.0, abs=1.0), (
            f"tail credited {rows[2]}s; 900s means the nominal was fabricated")
    finally:
        db.close()


def test_resume_prefers_the_genuinely_fresher_session(tmp_path):
    db = StateDB(tmp_path / "s.db")
    # `fresh` is opened FIRST so it holds the LOWER rowid. A broken ranking
    # must not be rescued by the `id DESC` tie-break: with both ended_at
    # stamps landing in the same second, ordering on ended_at alone would
    # pick the higher id, which must be the WRONG answer for this to bite.
    fresh = db.open_stream_session("twitch", "t")
    stale = db.open_stream_session("twitch", "t")

    # `stale` last wrote media 200 s ago; `fresh` wrote 10 s ago and closed
    # normally. Reconcile then closes `stale` — stamping ended_at NOW, i.e.
    # LATER than fresh's ended_at. resumable_session reads the wall clock,
    # so the stamps are wall-clock relative.
    now = time.time()
    db.freshen_last_media(fresh, now - 10.0)
    db.close_stream_session(fresh)
    db.freshen_last_media(stale, now - 200.0)
    db.close_stream_session(stale, by_reconcile=True)

    got = db.resumable_session("twitch", "t", 300.0)
    assert got == fresh, "resumed the staler, reconcile-closed broadcast"
