"""Channel monitor: authorization boundary, disk guard, failure containment.

Offline: every platform probe and the chunker factory are injected.
"""

import asyncio
import threading
import time
from pathlib import Path

import pytest

from clipforge.config import AppConfig, ChannelSpec, Watchlist
from clipforge.errors import IngestError
from clipforge.ingest.chunker import SegmentEvent
from clipforge.ingest.monitor import ChannelMonitor, disk_allows
from clipforge.paths import Workspace
from clipforge.state import StateDB


@pytest.fixture()
def env(tmp_path: Path):
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    yield ws, db
    db.close()


def make_monitor(ws, db, channels: list[ChannelSpec], **kw) -> ChannelMonitor:
    cfg = AppConfig()
    cfg.ingest.poll_interval_s = 0.01
    cfg.disk.free_floor_gb = 0.001  # effectively no floor for tests
    return ChannelMonitor(cfg=cfg, db=db, ws=ws,
                          watchlist=Watchlist(channels=channels), **kw)


class DummySession:
    """Stands in for a ChunkerSession: records that run() was called."""

    def __init__(self) -> None:
        self.ran = threading.Event()

    def run(self, stop, *, session_id=None, ledger=None) -> None:
        self.ran.set()
        self.session_id = session_id


# ------------------------------------------------------------- authorization


def test_only_enabled_channels_are_touched(env):
    ws, db = env
    chans = [
        ChannelSpec(platform="twitch", handle="yes", enabled=True),
        ChannelSpec(platform="twitch", handle="no", enabled=False),
    ]
    mon = make_monitor(ws, db, chans)
    assert [c.handle for c in mon._authorized_channels()] == ["yes"]


def test_kick_requires_feature_flag(env):
    """T4: Kick is opt-in; disabled by default, never silently polled."""
    ws, db = env
    chans = [ChannelSpec(platform="kick", handle="k", enabled=True),
             ChannelSpec(platform="twitch", handle="t", enabled=True)]
    mon = make_monitor(ws, db, chans)
    assert [c.platform for c in mon._authorized_channels()] == ["twitch"]

    mon.cfg.ingest.kick_enabled = True
    assert sorted(c.platform for c in mon._authorized_channels()) == \
        ["kick", "twitch"]


# ---------------------------------------------------------------- disk guard


def test_disk_guard_blocks_below_floor(tmp_path: Path):
    assert disk_allows(tmp_path, floor_gb=0.001) is True
    assert disk_allows(tmp_path, floor_gb=10_000_000.0) is False


def test_disk_guard_fails_closed_on_probe_error(tmp_path: Path):
    """Cannot verify free space ⇒ do not write (pause, don't fill the drive)."""
    assert disk_allows(tmp_path / "does_not_exist_at_all", floor_gb=1.0) is False


async def test_tick_skipped_when_disk_full(env):
    ws, db = env
    calls: list[str] = []
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       twitch_is_live=lambda h, **kw: calls.append(h) or True)
    mon.cfg.disk.free_floor_gb = 10_000_000.0  # floor above any real disk

    task = asyncio.create_task(mon._channel_loop(chans[0]))
    await asyncio.sleep(0.05)
    mon.stop()
    await asyncio.wait_for(task, timeout=5)

    assert calls == [], "ingestion must PAUSE below the disk floor"


# ------------------------------------------------------------------- youtube


def test_youtube_downloads_only_new_ids(env):
    ws, db = env
    downloaded: list[str] = []

    def discover(db_, handle, end):
        return ["aaaaaaaaaaa"] if not downloaded else []

    def download(vid, dest, **kw):
        downloaded.append(vid)
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"yt_{vid}.mp4"
        p.write_bytes(b"x")
        return p

    media: list[tuple] = []
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans, discover=discover, download=download,
                       on_media=lambda p, t, key=None: media.append((p, t, key)))

    mon._tick_youtube(chans[0])
    mon._tick_youtube(chans[0])

    assert downloaded == ["aaaaaaaaaaa"]
    assert len(media) == 1 and media[0][1] == 0.0
    assert media[0][2] == ("video", "youtube", "@c", "aaaaaaaaaaa"), (
        "a downloaded VOD must say which id it came from, so the clip "
        "can be recorded as done and not fetched again after a restart")


def test_youtube_download_is_stoppable(env):
    """Shutdown must reach an in-flight download: without the stop event
    threaded through, Ctrl+C waited out yt-dlp's hour-long timeout."""
    ws, db = env
    seen_stop = {}

    def download(vid, dest, **kw):
        seen_stop["stop"] = kw.get("stop")
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / "a.mp4"
        p.write_bytes(b"x")
        return p

    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       discover=lambda d, h, e: ["aaaaaaaaaaa"],
                       download=download)
    mon._tick_youtube(chans[0])
    # The download stops for shutdown OR for the channel going live: a
    # broadcast cannot be fetched later, a published VOD can.
    passed = seen_stop["stop"]
    assert passed is not None and not passed.is_set()
    mon._stop.set()
    assert passed.is_set(), "download must stop when the monitor does"
    mon._stop.clear()
    mon._live_event(chans[0]).set()
    assert passed.is_set(), "download must stop when the channel goes live"
    mon._live_event(chans[0]).clear()


def test_youtube_stops_mid_batch_on_shutdown(env):
    ws, db = env
    calls: list[str] = []

    def download(vid, dest, **kw):
        calls.append(vid)
        mon.stop()  # shutdown begins during the first download
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"{vid}.mp4"
        p.write_bytes(b"x")
        return p

    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       discover=lambda d, h, e: ["aaaaaaaaaaa", "bbbbbbbbbbb",
                                                 "ccccccccccc"],
                       download=download)
    mon._tick_youtube(chans[0])
    assert calls == ["aaaaaaaaaaa"], "must not start further downloads"


def test_failed_download_marks_failed_and_continues(env):
    ws, db = env

    def discover(db_, handle, end):
        return ["aaaaaaaaaaa", "bbbbbbbbbbb"]

    got: list[str] = []

    def download(vid, dest, **kw):
        if vid.startswith("a"):
            raise IngestError("HTTP 403")
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"yt_{vid}.mp4"
        p.write_bytes(b"x")
        got.append(vid)
        return p

    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans, discover=discover, download=download)
    mon._tick_youtube(chans[0])

    assert got == ["bbbbbbbbbbb"], "one bad VOD must not stop the others"


# ---------------------------------------------------------------------- live


async def test_live_channel_starts_chunker(env):
    ws, db = env
    session = DummySession()
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       twitch_is_live=lambda h, **kw: True,
                       make_chunker=lambda ch: session)

    await mon._tick_live(chans[0])
    assert session.ran.is_set()


async def test_offline_channel_does_not_start_chunker(env):
    ws, db = env
    session = DummySession()
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       twitch_is_live=lambda h, **kw: False,
                       make_chunker=lambda ch: session)
    await mon._tick_live(chans[0])
    assert not session.ran.is_set()


async def test_kick_indeterminate_treated_as_offline(env):
    """T4: None ('cannot determine') must never start a capture."""
    ws, db = env
    session = DummySession()
    chans = [ChannelSpec(platform="kick", handle="k", enabled=True)]
    mon = make_monitor(ws, db, chans, kick_is_live=lambda h, **kw: None,
                       make_chunker=lambda ch: session)
    await mon._tick_live(chans[0])
    assert not session.ran.is_set()


# ------------------------------------------------------------- containment


async def test_one_channel_failure_does_not_kill_siblings(env):
    """A crashing channel loop must never take down the others."""
    ws, db = env
    healthy_ticks: list[int] = []
    chans = [
        ChannelSpec(platform="twitch", handle="broken", enabled=True, priority=1),
        ChannelSpec(platform="twitch", handle="healthy", enabled=True),
    ]

    def is_live(handle, **kw):
        if handle == "broken":
            raise IngestError("platform API is down")
        healthy_ticks.append(1)
        return False

    mon = make_monitor(ws, db, chans, twitch_is_live=is_live)
    mon.cfg.ingest.backoff_base_s = 0.01
    mon.cfg.ingest.backoff_max_s = 0.02

    task = asyncio.create_task(mon.run())
    await asyncio.sleep(0.2)
    mon.stop()
    await asyncio.wait_for(task, timeout=5)

    assert len(healthy_ticks) >= 2, "healthy channel must keep polling"


async def test_unexpected_exception_is_contained(env):
    ws, db = env
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]

    def is_live(handle, **kw):
        raise RuntimeError("library bug")

    mon = make_monitor(ws, db, chans, twitch_is_live=is_live)
    task = asyncio.create_task(mon._channel_loop(chans[0]))
    await asyncio.sleep(0.05)
    mon.stop()
    await asyncio.wait_for(task, timeout=5)  # loop survived, returned cleanly


async def test_live_probes_do_not_block_the_event_loop(env):
    """A slow is_live probe on one channel must not stall the others —
    review measured full serialization when it ran inline on the loop."""
    ws, db = env
    chans = [ChannelSpec(platform="twitch", handle=f"c{i}", enabled=True)
             for i in range(6)]
    ticks: list[str] = []

    def slow_is_live(handle, **kw):
        time.sleep(0.2)  # blocking, like the real subprocess call
        ticks.append(handle)
        return False

    mon = make_monitor(ws, db, chans, twitch_is_live=slow_is_live)
    started = time.monotonic()
    task = asyncio.create_task(mon.run())
    while len(ticks) < 6 and time.monotonic() - started < 5:
        await asyncio.sleep(0.02)
    mon.stop()
    await asyncio.wait_for(task, timeout=10)

    elapsed = time.monotonic() - started
    assert len(ticks) >= 6
    # Serialized would be >= 1.2 s; concurrent lands near 0.2 s.
    assert elapsed < 1.0, f"probes serialized on the event loop ({elapsed:.2f}s)"


def test_t1_window_is_built_for_each_segment(env):
    """§11/T1: the DAG receives chunk + previous tail, never a bare chunk."""
    ws, db = env
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    media: list[tuple] = []
    mon = make_monitor(ws, db, chans, on_media=lambda p, t, key=None: media.append((p, t, key)))
    mon.cfg.ingest.overlap_s = 60

    calls: list[dict] = []

    def fake_window(**kw):
        calls.append(kw)
        from clipforge.ingest.overlap import VirtualWindow

        overlap = 0.0 if kw["prev_chunk"] is None else 63.4
        return VirtualWindow(path=kw["chunk"],
                             abs_start_s=kw["chunk_abs_start_s"] - overlap,
                             overlap_s=overlap, duration_s=900.0)

    import clipforge.ingest.monitor as monitor_mod

    original = monitor_mod.build_virtual_window
    monitor_mod.build_virtual_window = fake_window
    try:
        d = ws.chunks / "seg"
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i in (0, 1):
            p = d / f"chunk_{i:05d}.ts"
            p.write_bytes(b"\x47" * 1024)
            paths.append(p)
            # The chunker reports the TRUE timeline predecessor in the
            # event; the monitor must use that, not a "last emitted" cache.
            mon._on_segment(SegmentEvent(
                session_id=1, seg_index=i, path=p, abs_start_s=i * 900.0,
                duration_s=900.0,
                prev_path=paths[i - 1] if i else None))
    finally:
        monitor_mod.build_virtual_window = original

    assert calls[0]["prev_chunk"] is None, "first segment has no predecessor"
    assert calls[1]["prev_chunk"] == d / "chunk_00000.ts", \
        "second segment must carry the previous chunk's tail (T1)"
    assert calls[1]["overlap_s"] == 60
    # Absolute start is pulled back by the real overlap.
    assert media[1][1] == 900.0 - 63.4
    assert media[0][2] == ("segment", 1, 0), (
        "a window must carry the segment it came from")


def test_window_failure_still_emits_the_chunk(env):
    """A broken window build must never cost us the segment."""
    ws, db = env
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    media: list[tuple] = []
    mon = make_monitor(ws, db, chans, on_media=lambda p, t, key=None: media.append((p, t, key)))

    import clipforge.ingest.monitor as monitor_mod
    from clipforge.errors import FfmpegError

    original = monitor_mod.build_virtual_window

    def boom(**kw):
        raise FfmpegError("concat failed")

    monitor_mod.build_virtual_window = boom
    try:
        p = ws.chunks / "c.ts"
        p.write_bytes(b"\x47" * 1024)
        mon._on_segment(SegmentEvent(session_id=1, seg_index=0, path=p,
                                     abs_start_s=0.0, duration_s=900.0))
    finally:
        monitor_mod.build_virtual_window = original

    assert media == [(p, 0.0, ("segment", 1, 0))], (
        "must fall back to the bare chunk, still naming its segment")


async def test_retention_runs_even_while_a_capture_is_live(env):
    """Round-2 MAJOR: housekeeping hung off the channel loop, which blocks
    for the WHOLE broadcast — so reclamation stopped for exactly the period
    when chunks accumulate. It must run on its own cadence."""
    ws, db = env
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    forever = threading.Event()

    class BlockingSession:
        def run(self, stop, *, session_id=None, ledger=None):
            forever.wait(timeout=5)  # occupies the channel loop, like a live stream

    mon = make_monitor(ws, db, chans, twitch_is_live=lambda h, **kw: True,
                       make_chunker=lambda ch: BlockingSession())
    swept = {"n": 0}

    import clipforge.ingest.monitor as monitor_mod

    original_sweep = monitor_mod.sweep_retention
    original_interval = monitor_mod.RETENTION_INTERVAL_S
    monitor_mod.sweep_retention = lambda *a, **kw: swept.__setitem__("n", swept["n"] + 1)
    monitor_mod.RETENTION_INTERVAL_S = 0.05
    try:
        task = asyncio.create_task(mon.run())
        await asyncio.sleep(0.4)
        mon.stop()
        forever.set()
        await asyncio.wait_for(task, timeout=10)
    finally:
        monitor_mod.sweep_retention = original_sweep
        monitor_mod.RETENTION_INTERVAL_S = original_interval

    assert swept["n"] >= 2, (
        f"retention ran {swept['n']}x while a capture held the channel loop")


async def test_session_is_resumed_after_a_brief_outage(env):
    """One broadcast must be ONE timeline: a short drop that ends the
    session must not restart absolute media time at zero."""
    ws, db = env
    chans = [ChannelSpec(platform="twitch", handle="t", enabled=True)]
    session = DummySession()
    mon = make_monitor(ws, db, chans, twitch_is_live=lambda h, **kw: True,
                       make_chunker=lambda ch: session)

    prior = db.open_stream_session("twitch", "t")
    db.close_stream_session(prior)

    await mon._tick_live(chans[0])

    assert session.session_id == prior, "must resume the recent session"
    assert [s["id"] for s in db.open_sessions()] == [prior]


async def test_on_media_callback_failure_is_contained(env):
    ws, db = env
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]

    def boom(path, abs_start):
        raise ValueError("DAG seam bug")

    mon = make_monitor(ws, db, chans,
                       discover=lambda d, h, e: ["aaaaaaaaaaa"],
                       download=lambda v, dest, **kw: (
                           dest.mkdir(parents=True, exist_ok=True),
                           (dest / "a.mp4").write_bytes(b"x"),
                           dest / "a.mp4")[-1],
                       on_media=boom)
    mon._tick_youtube(chans[0])  # must not raise


# -------------------------------------------------------------- youtube live
#
# YouTube channels took the VOD branch unconditionally, so a creator who was
# live produced nothing until the broadcast became a VOD. These pin the
# probe-first routing.


async def test_a_live_youtube_channel_is_captured_not_downloaded(env):
    ws, db = env
    session = DummySession()
    downloads: list[str] = []
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       youtube_is_live=lambda h, **kw: True,
                       discover=lambda *a, **kw: downloads.append("discover") or [],
                       make_chunker=lambda ch: session)
    task = asyncio.create_task(mon._channel_loop(chans[0]))
    await asyncio.sleep(0.2)
    mon.stop()
    await asyncio.wait_for(task, timeout=5)
    assert session.ran.is_set(), "a live YouTube broadcast was not captured"
    assert downloads == [], (
        "the VOD backlog competed with a live broadcast for bandwidth")


@pytest.mark.parametrize("state", [False, None])
async def test_an_offline_or_unknown_youtube_channel_takes_the_vod_path(env, state):
    ws, db = env
    session = DummySession()
    discovered: list[str] = []
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       youtube_is_live=lambda h, **kw: state,
                       discover=lambda *a, **kw: discovered.append("d") or [],
                       make_chunker=lambda ch: session)
    task = asyncio.create_task(mon._vod_loop(chans[0]))
    await asyncio.sleep(0.2)
    mon.stop()
    await asyncio.wait_for(task, timeout=5)
    assert not session.ran.is_set()
    assert discovered, "an off-air channel must still pick up new VODs"


async def test_the_live_loop_never_downloads_a_vod(env):
    """The live probe must stay responsive: one 54-minute VOD download
    (there were nine pending) used to hold this loop for as long as it
    ran, and a broadcast that began meanwhile was never recorded."""
    ws, db = env
    discovered: list[str] = []
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    mon = make_monitor(ws, db, chans,
                       youtube_is_live=lambda h, **kw: False,
                       discover=lambda *a, **kw: discovered.append("d") or [])
    task = asyncio.create_task(mon._channel_loop(chans[0]))
    await asyncio.sleep(0.2)
    mon.stop()
    await asyncio.wait_for(task, timeout=5)
    assert discovered == [], "the live loop went looking for VODs"


async def test_the_vod_backlog_stands_aside_when_the_channel_goes_live(env):
    """A published VOD can be fetched later; a broadcast cannot."""
    ws, db = env
    downloaded: list[str] = []
    chans = [ChannelSpec(platform="youtube", handle="@c", enabled=True)]
    ids = ["a" * 11, "b" * 11, "c" * 11]

    def discover(_db, _handle, _end, **kw):
        for v in ids:
            db.mark_video_seen("youtube", "@c", v)
        return list(ids)

    def download(vid, dest, **kw):
        downloaded.append(vid)
        mon._live_event(chans[0]).set()      # the stream starts mid-backlog
        dest.mkdir(parents=True, exist_ok=True)
        p = dest / f"yt_{vid}.mp4"
        p.write_bytes(b"\x00")
        return p

    mon = make_monitor(ws, db, chans, discover=discover, download=download)
    mon._tick_youtube(chans[0])
    assert downloaded == [ids[0]], (
        "the backlog kept downloading while the channel was broadcasting")
    still_pending = [r["video_id"] for r in
                     db.videos_needing_download("youtube", "@c")]
    assert set(ids[1:]) <= set(still_pending), (
        "the skipped ids must stay pending, with no attempt counted")


def test_a_youtube_capture_uses_youtube_streamlink_args(env):
    """The chunker built for a YouTube channel must open the channel's
    /live URL — kick's argv builder here would capture nothing."""
    ws, db = env
    chans = [ChannelSpec(platform="youtube", handle="@IShowSpeed", enabled=True)]
    mon = make_monitor(ws, db, chans)
    session = mon._build_chunker(chans[0])
    assert "https://www.youtube.com/@IShowSpeed/live" in session.streamlink_args
    assert "--hls-live-edge" in session.streamlink_args


def test_youtube_channels_get_a_capture_thread(env):
    """The pool is sized for broadcasts that hold a thread for hours; it
    used to count only non-YouTube channels."""
    import inspect

    src = inspect.getsource(ChannelMonitor.run)
    assert 'platform != "youtube"' not in src
