"""End-to-end CP1: monitor → chunker → remux → T1 window → DAG seam.

Uses a fake "live" channel (ffmpeg standing in for streamlink, exactly as
the real pipe test does) driven through the REAL ChannelMonitor, so the
wiring under test is the wiring that ships: the monitor's chunker factory,
the segment callback, the virtual-window build, and the on_media seam.

Needs real ffmpeg; skips otherwise. No GPU, no network.
"""

import asyncio
import threading
from pathlib import Path

import pytest

from clipforge.config import AppConfig, ChannelSpec, Watchlist
from clipforge.ffmpeg import find_binary, probe
from clipforge.ingest.chunker import ChunkerConfig, ChunkerSession
from clipforge.ingest.monitor import ChannelMonitor
from clipforge.paths import Workspace
from clipforge.state import StateDB

pytestmark = pytest.mark.skipif(
    find_binary("ffmpeg") is None or find_binary("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


def _fake_stream_args(ffmpeg: Path) -> list[str]:
    return ["-hide_banner", "-loglevel", "error",
            "-re", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "15",
            "-f", "mpegts", "pipe:1"]


async def test_watch_produces_windows_end_to_end(tmp_path: Path):
    ffmpeg = find_binary("ffmpeg")
    assert ffmpeg is not None
    ws = Workspace(tmp_path / "ws").ensure()
    db = StateDB(ws.state_db)
    media: list[tuple[Path, float]] = []

    cfg = AppConfig()
    cfg.workspace.root = ws.root
    cfg.ingest.poll_interval_s = 0.5
    cfg.ingest.overlap_s = 2          # small overlap for a short test
    cfg.disk.free_floor_gb = 0.001
    chans = [ChannelSpec(platform="twitch", handle="e2e", enabled=True)]

    def make_chunker(ch: ChannelSpec) -> ChunkerSession:
        return ChunkerSession(
            db=db, chunks_root=ws.chunks, quarantine_dir=ws.quarantine,
            platform=ch.platform, handle=ch.handle,
            streamlink_args=_fake_stream_args(ffmpeg),
            cfg=ChunkerConfig(segment_time_s=2, ready_stable_s=20.0,
                              poll_interval_s=0.25),
            on_segment_ready=mon._on_segment,
            log_dir=ws.logs,
            resolve_tools=lambda: (str(ffmpeg), str(ffmpeg)))

    mon = ChannelMonitor(cfg=cfg, db=db, ws=ws,
                         watchlist=Watchlist(channels=chans),
                         on_media=lambda p, t: media.append((p, t)),
                         twitch_is_live=lambda h, **kw: True,
                         make_chunker=make_chunker)
    try:
        task = asyncio.create_task(mon.run())
        # Capture ~8 s: enough for several 2 s segments plus windows.
        await asyncio.sleep(8.0)
        mon.stop()
        await asyncio.wait_for(task, timeout=30)
        # Read DB state BEFORE closing the handle.
        still_open = [s for s in db.open_sessions() if s["ended_at"] is None]
        segments = db.segments_for_session(1)
    finally:
        db.close()

    assert len(media) >= 2, f"expected multiple windows, got {media}"

    # Contract 1: everything handed to the DAG is real, playable media.
    for path, _abs in media:
        assert path.exists(), path
        info = probe(path)
        assert info.duration_s > 0.1, (path, info)

    # Contract 2 (T1): from the second segment on, the DAG receives a
    # WINDOW (prev tail + chunk), not the bare chunk — so its duration
    # exceeds the segment length and its absolute start is pulled back.
    windows = [p for p, _ in media if "window" in p.name]
    assert windows, "no virtual window was produced (T1 not wired)"
    win_dur = probe(windows[0]).duration_s
    assert win_dur > 2.0, f"window ({win_dur}s) should exceed one 2s segment"

    # Contract 3: absolute times are monotonic and start at zero.
    starts = [t for _p, t in media]
    assert starts[0] == 0.0
    assert starts == sorted(starts), starts

    # Contract 4: the session was closed cleanly on shutdown, and every
    # captured segment is durably recorded (not left at 'recording').
    assert still_open == [], "shutdown left a session open"
    assert segments, "no segments recorded in the DB"
    assert all(s["status"] == "ready" for s in segments), \
        [dict(s) for s in segments]
