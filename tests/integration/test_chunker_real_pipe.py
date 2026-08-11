"""The real streamlink→ffmpeg pipe, end to end, with a scripted fake
streamlink standing in for the network (so this is deterministic and needs
no live channel).

What this proves that the unit tests cannot:
  * the Python-wired pipe actually transports bytes (T3 wiring is correct),
  * ffmpeg really segments to MPEG-TS at the configured cadence,
  * killing streamlink delivers EOF and ffmpeg finalizes the TAIL segment
    (the whole reason for owning both halves in Python),
  * every produced segment is probeable — i.e. playable (T2).

Needs real ffmpeg; skips otherwise. No GPU, no network.
"""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from clipforge.ffmpeg import find_binary, probe
from clipforge.ingest.chunker import ChunkerConfig, ChunkerSession, SegmentEvent
from clipforge.state import StateDB

pytestmark = pytest.mark.skipif(
    find_binary("ffmpeg") is None or find_binary("ffprobe") is None,
    reason="ffmpeg/ffprobe not installed")


def _fake_streamlink_cmd(ffmpeg: Path) -> list[str]:
    """A stand-in 'streamlink': ffmpeg generating an endless TS byte stream
    on stdout, exactly like `streamlink --stdout` does."""
    return [str(ffmpeg), "-hide_banner", "-loglevel", "error",
            "-re", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-g", "15",
            "-f", "mpegts", "pipe:1"]


def test_real_pipe_segments_and_finalizes_tail(tmp_path: Path):
    ffmpeg = find_binary("ffmpeg")
    assert ffmpeg is not None
    db = StateDB(tmp_path / "state.db")
    events: list[SegmentEvent] = []
    try:
        sid = db.open_stream_session("twitch", "t")
        session = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "quarantine", platform="twitch",
            handle="t",
            # The chunker prepends the resolved streamlink path, so pass the
            # fake's arguments and resolve 'streamlink' to ffmpeg itself.
            streamlink_args=_fake_streamlink_cmd(ffmpeg)[1:],
            cfg=ChunkerConfig(segment_time_s=2, ready_stable_s=20.0,
                              poll_interval_s=0.25),
            on_segment_ready=events.append,
            resolve_tools=lambda: (str(ffmpeg), str(ffmpeg)))
        stop = threading.Event()
        # Let it capture ~7 s of media (≈3 segments at segment_time=2), then
        # stop — exercising the graceful shutdown + tail-finalize path.
        threading.Timer(7.0, stop.set).start()
        session._connect_once(stop, sid)
    finally:
        db.close()

    assert len(events) >= 2, f"expected multiple segments, got {events}"

    # Every emitted segment is REAL, probeable media, and has been remuxed
    # from the resilient recording format (.ts) to faststart MP4 now that it
    # is provably closed (§S0 / T2).
    for ev in events:
        assert ev.path.exists(), ev
        assert ev.path.suffix == ".mp4", f"segment not remuxed: {ev.path}"
        assert not ev.path.with_suffix(".ts").exists(), \
            "source .ts should be dropped after a successful remux"
        info = probe(ev.path)
        assert info.duration_s > 0.1, (ev, info)
        assert info.v_codec == "h264"

    # Absolute stream time is contiguous: each segment starts where the
    # previous one ended (within probe rounding).
    for prev, cur in zip(events, events[1:]):
        expected = prev.abs_start_s + prev.duration_s
        assert abs(cur.abs_start_s - expected) < 0.05, (prev, cur)

    # The TAIL was finalized after the kill — that is the T3 payoff. The
    # last emitted segment is the one that was still being written.
    last = events[-1]
    assert last.duration_s > 0.1, "tail segment must be finalized and playable"


def test_pipe_leaves_no_zombie_processes(tmp_path: Path):
    """T3: after a session ends, neither half of the pipe survives."""
    ffmpeg = find_binary("ffmpeg")
    assert ffmpeg is not None
    spawned: list[subprocess.Popen] = []

    real_popen = subprocess.Popen

    def tracking_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    db = StateDB(tmp_path / "state.db")
    try:
        sid = db.open_stream_session("twitch", "t")
        session = ChunkerSession(
            db=db, chunks_root=tmp_path / "chunks",
            quarantine_dir=tmp_path / "quarantine", platform="twitch",
            handle="t", streamlink_args=_fake_streamlink_cmd(ffmpeg)[1:],
            cfg=ChunkerConfig(segment_time_s=2, ready_stable_s=20.0,
                              poll_interval_s=0.25),
            popen=tracking_popen,
            resolve_tools=lambda: (str(ffmpeg), str(ffmpeg)))
        stop = threading.Event()
        threading.Timer(3.0, stop.set).start()
        session._connect_once(stop, sid)
    finally:
        db.close()

    assert len(spawned) == 2, "expected both pipe halves"
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if all(p.poll() is not None for p in spawned):
            break
        time.sleep(0.2)
    assert all(p.poll() is not None for p in spawned), \
        "zombie process survived session shutdown (T3)"
