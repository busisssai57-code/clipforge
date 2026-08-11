"""Chunker state machine — segment-ready rules, reconnect offsets, T3 shutdown.

Fully offline: subprocesses, prober, and clock are injected. The processes
are simulated by a scripted fake that writes .ts files the way ffmpeg's
segment muxer would.
"""

import threading
from pathlib import Path

import pytest

from clipforge.ffmpeg import MediaInfo
from clipforge.ingest.chunker import (MIN_TAIL_BYTES, ChunkerConfig,
                                      ChunkerSession, ConnectResult,
                                      SegmentEvent)
from clipforge.state import StateDB


class FakeProc:
    """Minimal Popen stand-in with scripted lifetime."""

    def __init__(self, alive_ticks: int = 10 ** 6) -> None:
        self.alive_ticks = alive_ticks
        self.killed = False
        self.waited = False
        self.pid = 4242
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


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def make_session(tmp_path: Path, db: StateDB, *, procs: list, clock,
                 on_ready=None, cfg: ChunkerConfig | None = None,
                 durations: dict[str, float] | None = None,
                 disk_ok=None) -> ChunkerSession:
    def popen(cmd, **kw):
        return procs.pop(0)

    def prober(path: Path) -> MediaInfo:
        dur = (durations or {}).get(Path(path).name, 900.0)
        return MediaInfo(duration_s=dur, width=1920, height=1080, fps=30.0,
                         fps_rational="30/1", v_codec="h264", a_codec="aac")

    return ChunkerSession(
        db=db, chunks_root=tmp_path / "chunks",
        quarantine_dir=tmp_path / "quarantine", platform="twitch",
        handle="tester", streamlink_args=["--stdout", "url", "best"],
        cfg=cfg or ChunkerConfig(segment_time_s=900, ready_stable_s=20.0,
                                 poll_interval_s=0.0, remux_to_mp4=False),
        on_segment_ready=on_ready, popen=popen, prober=prober, clock=clock,
        disk_ok=disk_ok,
        # Hermetic: no real streamlink/ffmpeg needed for the state machine.
        resolve_tools=lambda: ("streamlink", "ffmpeg"))


@pytest.fixture()
def db(tmp_path: Path):
    d = StateDB(tmp_path / "state.db")
    yield d
    d.close()


def seg_dir(tmp_path: Path, session_id: int) -> Path:
    p = tmp_path / "chunks" / "twitch_tester" / f"s{session_id:05d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_segment(d: Path, idx: int, size: int = MIN_TAIL_BYTES * 2) -> Path:
    p = d / f"chunk_{idx:05d}.ts"
    p.write_bytes(b"\x47" * size)
    return p


def test_segment_ready_when_next_appears(tmp_path: Path, db: StateDB):
    """Primary readiness rule: a segment is closed once its successor exists."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    write_segment(d, 1)  # ⇒ segment 0 is provably closed
    clock = FakeClock()
    events: list[SegmentEvent] = []
    # Processes die after a couple of polls: connect ends, tail finalizes too.
    sess = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                        clock=clock, on_ready=events.append)
    stop = threading.Event()

    sess._connect_once(stop, sid)

    assert [e.seg_index for e in events] == [0, 1]
    assert events[0].abs_start_s == 0.0
    assert events[1].abs_start_s == 900.0  # exact absolute stream time
    rows = list(db.segments_with_status("ready"))
    assert len(rows) == 2


def test_growing_tail_is_never_emitted(tmp_path: Path, db: StateDB):
    """A file still being written must NEVER reach the DAG.

    The oracle is WHEN emission happens, not merely that it happened once:
    an earlier version of this test asserted only the final event list, so a
    mutant that finalized the still-growing tail on every poll passed it.
    Here the writer records whether it was still writing at emission time.
    """
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    clock = FakeClock()
    writing = {"active": True}
    emissions: list[tuple[int, bool, int]] = []  # (idx, still_writing, size_seen)

    def on_ready(ev: SegmentEvent) -> None:
        emissions.append((ev.seg_index, writing["active"],
                          ev.path.stat().st_size if ev.path.exists() else -1))

    class GrowingProc(FakeProc):
        """Grows the tail on every poll — the file is never stable."""

        def poll(self):
            if writing["active"]:
                write_segment(d, 0, size=MIN_TAIL_BYTES * 2 + int(clock.t * 1000) + 1)
                clock.advance(1.0)
            alive = super().poll()
            if alive is not None:      # process just exited …
                writing["active"] = False  # … so writing has stopped
            return alive

    sess = make_session(tmp_path, db, procs=[GrowingProc(6), FakeProc(6)],
                        clock=clock, on_ready=on_ready)
    sess._connect_once(threading.Event(), sid)

    assert emissions, "the closed tail must eventually be emitted"
    for idx, still_writing, _size in emissions:
        assert not still_writing, (
            f"segment {idx} was emitted to the DAG while the writer was "
            "still growing it")
    # And it is emitted exactly once, at its final size.
    assert [e[0] for e in emissions] == [0]
    assert emissions[0][2] == (d / "chunk_00000.ts").stat().st_size


def test_disk_floor_ends_capture_mid_broadcast(tmp_path: Path, db: StateDB):
    """§6: the guard must be able to pause a capture already in flight —
    a broadcast lasts hours, so a per-tick check in the monitor never fires."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    free = {"ok": True}

    class TickProc(FakeProc):
        polls = 0

        def poll(self):
            TickProc.polls += 1
            if TickProc.polls > 2:
                free["ok"] = False  # disk crosses the floor mid-capture
            return super().poll()

    sess = make_session(tmp_path, db, procs=[TickProc(10 ** 6), FakeProc(10 ** 6)],
                        clock=FakeClock(), disk_ok=lambda: free["ok"])
    result = sess._connect_once(threading.Event(), sid)
    assert result.captured_s >= 0  # returned rather than writing forever
    assert not free["ok"]


def test_stalled_tail_triggers_reconnect_and_finalize(tmp_path: Path, db: StateDB):
    """Secondary rule: size unchanged for ready_stable_s ⇒ stream is dead;
    kill the connection, finalize the tail, and REPORT the stall so run()
    can reconnect promptly instead of escalating its failure backoff."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    clock = FakeClock()
    events: list[SegmentEvent] = []
    sl, ff = FakeProc(10 ** 6), FakeProc(10 ** 6)  # both stay "alive"

    # The clock advances per poll via the process's own poll() — no global
    # monkeypatching of threading primitives (which would corrupt any test
    # running concurrently).
    class TickingProc(FakeProc):
        def poll(self):
            clock.advance(5.0)
            return super().poll()

    sess = make_session(tmp_path, db, procs=[TickingProc(10 ** 6), ff],
                        clock=clock, on_ready=events.append,
                        cfg=ChunkerConfig(ready_stable_s=20.0,
                                          poll_interval_s=0.0,
                                          remux_to_mp4=False))
    result = sess._connect_once(threading.Event(), sid)

    assert result.stalled is True, "a stalled tail must be reported as such"
    assert [e.seg_index for e in events] == [0], "tail finalized after shutdown"
    assert result.captured_s == 900.0


def test_stall_does_not_escalate_backoff(tmp_path: Path, db: StateDB):
    """A productive connect killed by an upstream stall is not OUR failure:
    reconnect promptly rather than climbing toward the 300 s cap."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    delays: list[float] = []
    results = [ConnectResult(captured_s=600.0, stalled=True),
               ConnectResult(captured_s=600.0, stalled=True),
               ConnectResult(captured_s=0.0), ConnectResult(captured_s=0.0),
               ConnectResult(captured_s=0.0)]

    sess = make_session(tmp_path, db, procs=[], clock=FakeClock())
    sess._connect_once = lambda stop, s, deadline=None: results.pop(0)  # type: ignore[assignment]

    stop = threading.Event()

    class RecordingEvent(threading.Event):
        def wait(self, timeout=None):
            if timeout:
                delays.append(timeout)
            return False

    sess.run(RecordingEvent(), session_id=sid)

    # The two stall-driven reconnects use the base delay, not an escalating
    # one; only the genuinely empty connects escalate.
    assert delays[0] < 12.0 and delays[1] < 12.0, delays


def test_absolute_time_survives_reconnect(tmp_path: Path, db: StateDB):
    """Spec §S0: a dropped stream resumes into a NEW segment index and
    absolute media time continues where it left off."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    clock = FakeClock()
    events: list[SegmentEvent] = []

    # --- connect 1: two segments, 900s each
    write_segment(d, 0)
    write_segment(d, 1)
    s1 = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                      clock=clock, on_ready=events.append)
    s1._connect_once(threading.Event(), sid)

    sess_row = db.get_session(sid)
    assert sess_row["base_offset_s"] == 1800.0
    assert sess_row["next_segment"] == 2, "reconnect resumes at a NEW index"

    # --- connect 2 (after the drop): indices continue at 2
    write_segment(d, 2)
    write_segment(d, 3)
    s2 = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                      clock=clock, on_ready=events.append)
    s2._connect_once(threading.Event(), sid)

    assert [e.seg_index for e in events] == [0, 1, 2, 3]
    assert [e.abs_start_s for e in events] == [0.0, 900.0, 1800.0, 2700.0]


def test_torn_tail_below_minimum_is_discarded(tmp_path: Path, db: StateDB):
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0, size=10)  # a few bytes: torn beyond use
    events: list[SegmentEvent] = []
    sess = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                        clock=FakeClock(), on_ready=events.append)
    sess._connect_once(threading.Event(), sid)
    assert events == []
    assert not (d / "chunk_00000.ts").exists()


def test_unprobeable_segment_is_quarantined(tmp_path: Path, db: StateDB):
    """A corrupt TS must not stall the session — quarantine and continue."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    events: list[SegmentEvent] = []

    def bad_prober(path):
        raise ValueError("moov/PAT missing")

    sess = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                        clock=FakeClock(), on_ready=events.append)
    sess._prober = bad_prober
    sess._connect_once(threading.Event(), sid)

    assert events == []
    # Quarantine names include the session so two sessions' chunk_00000.ts
    # cannot overwrite each other, and the DB row points at the NEW location.
    moved = tmp_path / "quarantine" / f"s{sid:05d}_chunk_00000.ts"
    assert moved.exists()
    row = db.segments_for_session(sid)[0]
    assert Path(row["path"]) == moved and Path(row["path"]).exists()
    assert row["status"] == "quarantined"


def test_quarantine_does_not_shift_later_absolute_times(tmp_path: Path,
                                                        db: StateDB):
    """A quarantined segment must still ADVANCE the clock: dropping its
    duration silently re-timed every later segment 15 minutes early and
    banked the error into the session offset."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    for i in range(4):
        write_segment(d, i)
    events: list[SegmentEvent] = []

    def prober(path: Path):
        if path.name == "chunk_00001.ts":
            raise ValueError("corrupt")
        return MediaInfo(duration_s=900.0, width=1920, height=1080, fps=30.0,
                         fps_rational="30/1", v_codec="h264", a_codec="aac")

    sess = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                        clock=FakeClock(), on_ready=events.append)
    sess._prober = prober
    sess._connect_once(threading.Event(), sid)

    got = {e.seg_index: e.abs_start_s for e in events}
    assert got == {0: 0.0, 2: 1800.0, 3: 2700.0}, got
    assert db.get_session(sid)["base_offset_s"] == 3600.0


def test_estimated_time_never_counts_as_captured_media(tmp_path: Path,
                                                       db: StateDB):
    """Round-2 BLOCKER: an unprobeable TAIL was credited a nominal 900 s,
    which (a) banked phantom media time and (b) scored the connect as
    'healthy', so the backoff never escalated, the empty-connect counter
    never advanced, and a dead stream looped forever holding its thread."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)  # the only segment, and it is unprobeable

    def bad_prober(path):
        from clipforge.errors import FfmpegError

        raise FfmpegError("torn TS")

    sess = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                        clock=FakeClock())
    sess._prober = bad_prober
    result = sess._connect_once(threading.Event(), sid)

    assert result.captured_s == 0.0, (
        "estimated seconds must NOT count as captured media")
    assert result.estimated is True
    # With no probed media to learn a bitrate from, we refuse to guess.
    assert result.timeline_s == 0.0
    assert db.get_session(sid)["timeline_estimated"] == 1


def test_unprobeable_tail_is_estimated_from_observed_bitrate(tmp_path: Path,
                                                             db: StateDB):
    """A tail is arbitrarily short, so its duration is estimated from this
    connect's real bitrate — never assumed to be a full segment."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0, size=900_000)   # 900 KB of 900 s media = 1000 B/s
    write_segment(d, 1, size=20_000)    # tail: ~20 s at that bitrate
    assert 20_000 > MIN_TAIL_BYTES      # above the torn-beyond-use floor

    def prober(path: Path):
        if path.name == "chunk_00001.ts":
            raise ValueError("torn tail")
        return MediaInfo(duration_s=900.0, width=1920, height=1080, fps=30.0,
                         fps_rational="30/1", v_codec="h264", a_codec="aac")

    clock = FakeClock()

    class TickingProc(FakeProc):
        """Advances wall time so the tail has plausibly existed a while."""

        def poll(self):
            clock.advance(30.0)
            return super().poll()

    sess = make_session(tmp_path, db, procs=[TickingProc(3), FakeProc(3)],
                        clock=clock)
    sess._prober = prober
    result = sess._connect_once(threading.Event(), sid)

    assert result.captured_s == 900.0, "only probed media counts as captured"
    # ~20 s from the bitrate bound, and nowhere near a phantom 900 s.
    assert 10.0 < (result.timeline_s - 900.0) < 40.0, result.timeline_s
    assert result.estimated is True


def test_tail_estimate_is_bounded_by_wall_clock(tmp_path: Path, db: StateDB):
    """Round-3 finding: on variable-bitrate media the bitrate estimate can
    be wildly wrong (a high-motion tail estimated 689 s for 4 s of media),
    and the nominal clamp is far too loose to catch it at a 900 s segment
    length. A tail cannot hold more media than the wall time it existed."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0, size=20_000)      # low-bitrate body: 20 KB / 900 s
    write_segment(d, 1, size=5_000_000)   # huge high-motion tail

    def prober(path: Path):
        if path.name == "chunk_00001.ts":
            raise ValueError("torn tail")
        return MediaInfo(duration_s=900.0, width=1920, height=1080, fps=30.0,
                         fps_rational="30/1", v_codec="h264", a_codec="aac")

    clock = FakeClock()

    class SlowTick(FakeProc):
        def poll(self):
            clock.advance(2.0)  # the tail exists for only a few seconds
            return super().poll()

    sess = make_session(tmp_path, db, procs=[SlowTick(3), FakeProc(3)],
                        clock=clock)
    sess._prober = prober
    result = sess._connect_once(threading.Event(), sid)

    assert result.captured_s == 900.0, "the probed body still counts"
    tail_estimate = result.timeline_s - 900.0
    # Bitrate alone would say ~225,000 s; the nominal clamp would say 900.
    # Wall clock says the handful of seconds the tail actually existed —
    # that is the honest bound, and it is what keeps this from banking
    # phantom timeline. (Measured at finalize, so it includes the shutdown.)
    assert 0.0 <= tail_estimate < 60.0, tail_estimate
    assert tail_estimate < 900.0 * 0.1, "wall-clock bound is not binding"


def test_dying_stream_does_not_loop_forever(tmp_path: Path, db: StateDB):
    """The end-to-end consequence of the blocker: three consecutive connects
    that capture nothing real must end the session."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    connects = {"n": 0}

    sess = make_session(tmp_path, db, procs=[], clock=FakeClock())

    def dead_connect(stop, s, deadline=None):
        connects["n"] += 1
        if connects["n"] > 20:
            raise AssertionError("run() never gave up on a dead stream")
        # Timeline advances (estimated) but nothing real was captured.
        return ConnectResult(captured_s=0.0, timeline_s=900.0, estimated=True)

    sess._connect_once = dead_connect  # type: ignore[assignment]

    class NoWait(threading.Event):
        def wait(self, timeout=None):
            return False

    sess.run(NoWait(), session_id=sid)
    assert connects["n"] == 3, connects["n"]


@pytest.mark.parametrize("captured", [0.0, 1e-9, 0.04, 3.0, 4.9,
                                      5.1, 10.0, 25.0, 29.9])
def test_useless_connects_end_the_session(tmp_path: Path, db: StateDB,
                                          captured: float):
    """Every sub-healthy capture must count as a strike.

    Two rounds of review showed that ANY threshold between "zero" and
    "healthy" merely MOVES the permanent loop: first 0.04 s reset the
    give-up counter, then 3 s did, then 5-30 s did. The parametrization
    deliberately spans the whole band, including the values each previous
    fix left open (5.1 through 29.9)."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    connects = {"n": 0}
    sess = make_session(tmp_path, db, procs=[], clock=FakeClock())

    def useless(stop, s, deadline=None):
        connects["n"] += 1
        if connects["n"] > 30:
            raise AssertionError(f"run() never gave up on captured_s={captured}")
        return ConnectResult(captured_s=captured, timeline_s=captured,
                             stalled=captured > 0)

    sess._connect_once = useless  # type: ignore[assignment]

    class NoWait(threading.Event):
        def wait(self, timeout=None):
            return False

    sess.run(NoWait(), session_id=sid)
    assert connects["n"] == 3, connects


def test_interleaved_tiny_captures_still_end_the_session(tmp_path: Path,
                                                         db: StateDB):
    """0,0,tiny,0,0,tiny,... defeated the CONSECUTIVE rule before the
    useless-connect threshold existed."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    seq = [0.0, 0.0, 0.02] * 12
    connects = {"n": 0}
    sess = make_session(tmp_path, db, procs=[], clock=FakeClock())

    def cycle(stop, s, deadline=None):
        connects["n"] += 1
        if connects["n"] > 30:
            raise AssertionError("run() never gave up")
        return ConnectResult(captured_s=seq[connects["n"] - 1])

    sess._connect_once = cycle  # type: ignore[assignment]

    class NoWait(threading.Event):
        def wait(self, timeout=None):
            return False

    sess.run(NoWait(), session_id=sid)
    assert connects["n"] == 3


def test_productive_connects_never_end_the_session(tmp_path: Path, db: StateDB):
    """The counterpart: a healthy stream must not be given up on."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    connects = {"n": 0}
    sess = make_session(tmp_path, db, procs=[], clock=FakeClock())

    def healthy(stop, s, deadline=None):
        connects["n"] += 1
        return ConnectResult(captured_s=600.0, timeline_s=600.0)

    sess._connect_once = healthy  # type: ignore[assignment]

    class StopAfter10(threading.Event):
        def wait(self, timeout=None):
            if connects["n"] >= 10:
                self.set()
            return False

    sess.run(StopAfter10(), session_id=sid)
    assert connects["n"] >= 10, "healthy stream was abandoned"


def test_torn_tail_advances_clock_too(tmp_path: Path, db: StateDB):
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    write_segment(d, 1, size=10)  # torn tail
    events: list[SegmentEvent] = []
    sess = make_session(tmp_path, db, procs=[FakeProc(2), FakeProc(2)],
                        clock=FakeClock(), on_ready=events.append)
    sess._connect_once(threading.Event(), sid)
    assert [e.seg_index for e in events] == [0]
    # next_segment moved past the discarded tail so the reconnect does not
    # reuse its index.
    assert db.get_session(sid)["next_segment"] == 2


def test_db_failure_does_not_kill_the_capture(tmp_path: Path, db: StateDB):
    """A transient DB failure must be survivable: the media is on disk and
    the clock advanced, so the capture keeps running rather than tearing
    down (and it certainly must not orphan the pipe)."""
    from clipforge.errors import StateError

    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    write_segment(d, 1)
    sl, ff = FakeProc(2), FakeProc(2)
    events: list[SegmentEvent] = []
    sess = make_session(tmp_path, db, procs=[sl, ff], clock=FakeClock(),
                        on_ready=events.append)

    calls = {"n": 0}
    real_bank = db.bank_segment_and_offset

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise StateError("database is locked")
        return real_bank(*a, **kw)

    sess.db = type("FlakyDB", (), {
        "get_session": staticmethod(db.get_session),
        "bank_segment_and_offset": staticmethod(flaky),
        "set_next_segment": staticmethod(db.set_next_segment),
        "close_stream_session": staticmethod(db.close_stream_session),
    })()

    sess._connect_once(threading.Event(), sid)  # must not raise

    assert calls["n"] == 2, "capture continued past the DB failure"
    assert [e.seg_index for e in events] == [1], "seg 0 failed to record"


def test_unexpected_exception_still_tears_down_the_pipe(tmp_path: Path,
                                                        db: StateDB):
    """T3: ANY exception between spawn and teardown — including a bug — must
    not orphan the processes. Previously the teardown was straight-line code
    after the loop, so any raise skipped it entirely."""
    sid = db.open_stream_session("twitch", "tester")
    d = seg_dir(tmp_path, sid)
    write_segment(d, 0)
    write_segment(d, 1)
    sl, ff = FakeProc(10 ** 6), FakeProc(10 ** 6)
    sess = make_session(tmp_path, db, procs=[sl, ff], clock=FakeClock())

    def boom(*a, **kw):
        raise RuntimeError("a bug in stage bookkeeping")

    sess._finalize = boom  # type: ignore[method-assign]

    with pytest.raises(RuntimeError):
        sess._connect_once(threading.Event(), sid)

    assert sl.killed, "streamlink orphaned by an exception path"
    assert ff.waited or ff.killed, "ffmpeg orphaned by an exception path"


def test_half_spawned_pipe_kills_the_first_half(tmp_path: Path, db: StateDB):
    """If ffmpeg fails to start, streamlink must not be left running (and
    blocking forever on a full pipe)."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    sl = FakeProc(10 ** 6)
    calls = {"n": 0}

    def popen(cmd, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return sl
        raise OSError(5, "Access is denied")

    sess = ChunkerSession(
        db=db, chunks_root=tmp_path / "chunks",
        quarantine_dir=tmp_path / "quarantine", platform="twitch",
        handle="tester", streamlink_args=["--stdout"],
        cfg=ChunkerConfig(poll_interval_s=0.0, remux_to_mp4=False),
        popen=popen, clock=FakeClock(),
        resolve_tools=lambda: ("streamlink", "ffmpeg"))

    with pytest.raises(OSError):
        sess._connect_once(threading.Event(), sid)
    assert sl.killed, "half-spawned pipe leaked the streamlink half"


def test_shutdown_kills_streamlink_but_waits_for_ffmpeg(tmp_path: Path,
                                                        db: StateDB):
    """T3: killing streamlink delivers EOF; ffmpeg is WAITED on so it can
    write the tail segment's trailer — never killed first."""
    sid = db.open_stream_session("twitch", "tester")
    seg_dir(tmp_path, sid)
    sl, ff = FakeProc(10 ** 6), FakeProc(10 ** 6)
    sess = make_session(tmp_path, db, procs=[sl, ff], clock=FakeClock())
    stop = threading.Event()
    stop.set()  # immediate shutdown path

    sess._connect_once(stop, sid)

    assert sl.killed, "streamlink must be killed to deliver EOF"
    assert ff.waited, "ffmpeg must be waited on (finalize the tail)"
    assert not ff.killed, "ffmpeg must not be killed while finalizing"
