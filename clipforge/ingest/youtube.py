"""YouTube VOD path — discovery + download via yt-dlp (spec §S0).

Dedup contract: NEVER re-download a known video id. The state DB's
``seen_videos`` table is the source of truth (``mark_video_seen`` returns
True exactly once per (platform, id)); discovery itself is stateless.

All yt-dlp interaction goes through an injectable runner so unit tests are
offline; the live network path is exercised by integration tests / CP5.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Callable

from clipforge.errors import IngestError
from clipforge.ingest.runner import run_tool
from clipforge.log import get_logger
from clipforge.state import StateDB

log = get_logger(__name__)

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

RunTool = Callable[..., object]  # (name, args, **kw) -> CompletedProcess[str]


def channel_videos_url(handle: str) -> str:
    """The uploads tab for a handle ('@name') or bare channel path."""
    handle = handle.strip("/")
    if not handle.startswith("@") and not handle.startswith("channel/"):
        handle = f"@{handle}"
    return f"https://www.youtube.com/{handle}/videos"


def discover_vod_ids(handle: str, playlist_end: int, *,
                     run: RunTool = run_tool) -> list[str]:
    """Newest ``playlist_end`` upload ids for a channel, newest first.

    ``--flat-playlist`` never downloads media; ``--print id`` emits one id
    per line. Malformed lines are dropped (yt-dlp warnings sometimes land
    on stdout) — an id must look like an 11-char YouTube id.
    """
    proc = run("yt-dlp", [
        "--flat-playlist",
        "--playlist-end", str(playlist_end),
        "--print", "id",
        "--no-warnings",
        channel_videos_url(handle),
    ], timeout_s=60.0)
    ids = [line.strip() for line in proc.stdout.splitlines()]
    return [i for i in ids if _VIDEO_ID_RE.fullmatch(i)]


def pending_vod_ids(db: StateDB, handle: str, playlist_end: int, *,
                    max_attempts: int = 5, run: RunTool = run_tool) -> list[str]:
    """Ids that still need downloading: newly discovered PLUS any previously
    seen id that never landed.

    The requeue half is the fix for a demonstrated data-loss bug: discovery
    marks an id seen the first time it appears, so an id skipped by the disk
    guard or failed by a transient yt-dlp error was previously never offered
    again — permanently lost despite the spec only forbidding re-downloading
    ids that SUCCEEDED. ``max_attempts`` stops one permanently-broken VOD
    from pinning the loop forever.
    """
    for vid in discover_vod_ids(handle, playlist_end, run=run):
        db.mark_video_seen("youtube", handle, vid)  # idempotent
    # Deterministic order (§3.2): first_seen, then id. Scoped to THIS
    # channel — a platform-wide query returned other channels' ids too,
    # so parallel loops raced the same download.
    return [row["video_id"] for row in
            db.videos_needing_download("youtube", handle,
                                       max_attempts=max_attempts)]


def download_vod(video_id: str, dest_dir: Path, *,
                 run: RunTool = run_tool,
                 stop: "threading.Event | None" = None) -> Path:
    """Download one VOD into ``dest_dir``; returns the final media path.

    ``--print after_move:filepath`` makes yt-dlp itself tell us the output
    path (no directory-diff guessing). ``--no-part`` is deliberately NOT
    used: yt-dlp's own .part files are its resume mechanism; our startup
    sweep ignores them (different suffix than .partial).

    ``stop`` is threaded into the runner so shutdown kills an in-flight
    download instead of waiting out its (hour-long) timeout.
    """
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise IngestError(f"refusing malformed video id {video_id!r}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    proc = run("yt-dlp", [
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "-o", str(dest_dir / "yt_%(id)s.%(ext)s"),
        "--no-progress", "--no-warnings",
        "--print", "after_move:filepath",
        "--no-simulate",
        f"https://www.youtube.com/watch?v={video_id}",
    ], timeout_s=3600.0, stop=stop)
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    for line in reversed(lines):  # filepath is the last printed line
        p = Path(line)
        if p.exists():
            log.info("youtube.vod_downloaded", video_id=video_id, path=str(p))
            return p
    raise IngestError(
        f"yt-dlp reported success for {video_id} but no output file found "
        f"(stdout tail: {lines[-3:] if lines else 'empty'})")
