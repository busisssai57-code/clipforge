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
    # Use exported cookies if available (for sign-in-gated videos).
    _cookie_jar = Path(__file__).resolve().parent.parent / "cookies.txt"
    _cookie_args = ["--cookies", str(_cookie_jar)] if _cookie_jar.is_file() else []
    proc = run("yt-dlp", [
        "-f", "bv*[height<=1080]+ba/b[height<=1080]/b",
        "--merge-output-format", "mp4",
        "-o", str(dest_dir / "yt_%(id)s.%(ext)s"),
        "--no-progress", "--no-warnings",
        "--print", "after_move:filepath",
        "--no-simulate",
        *_cookie_args,
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


# --------------------------------------------------------------- live
#
# YouTube live was, until now, a hole in this module and in the monitor:
# `ChannelMonitor._channel_loop` routes every youtube channel to the VOD
# path (`live_channels` literally counts `platform != "youtube"`), so a
# channel that was broadcasting produced nothing until the stream ended
# and became a VOD — hours late, if ever. The pieces to do better already
# existed: `ChunkerSession` pipes streamlink into segmenting ffmpeg for
# Twitch and Kick, and streamlink speaks YouTube. All that was missing was
# a live probe and an argv builder.

#: Accepts what an operator actually pastes: a watch URL, a share link, a
#: /live handle URL, or a bare handle.
_WATCH_RE = re.compile(
    r"(?:youtube\.com/(?:watch\?v=|live/|shorts/)|youtu\.be/)([\w-]{11})")


def live_url(target: str) -> str:
    """Normalise anything paste-able into a URL streamlink can open."""
    raw = (target or "").strip()
    if not raw:
        raise IngestError("no YouTube target given")
    if raw.startswith(("http://", "https://")):
        return raw
    handle = raw if raw.startswith("@") else f"@{raw}"
    # A channel's /live URL resolves to whatever it is broadcasting now.
    return f"https://www.youtube.com/{handle}/live"


def video_id_from(target: str) -> str | None:
    hit = _WATCH_RE.search(target or "")
    return hit.group(1) if hit else None


def is_live(target: str, *, run: RunTool = run_tool) -> bool | None:
    """True live, False confidently not, None cannot determine.

    Tri-state on purpose, matching kick.is_live: "I could not tell" and
    "it is off air" call for different behaviour from the caller, and
    collapsing them is how a monitor decides a broadcasting channel is
    dark and stays quiet through the whole stream.

    yt-dlp prints the literal string ``True``/``False``/``NA`` for
    ``is_live``; anything else means the probe itself did not work.
    """
    url = live_url(target)
    try:
        proc = run("yt-dlp", ["--no-warnings", "--skip-download",
                              "--print", "is_live", url],
                   timeout_s=45.0, ok_codes=(0, 1))
    except IngestError as exc:
        log.info("youtube.live_probe_failed", url=url, error=str(exc)[:200])
        return None
    for line in reversed([ln.strip() for ln in proc.stdout.splitlines()]):
        if line in ("True", "true"):
            return True
        if line in ("False", "false"):
            return False
    if "not currently live" in (proc.stdout + proc.stderr).lower():
        return False
    return None


def live_title(target: str, *, run: RunTool = run_tool) -> str:
    """The broadcast's title, or an empty string. Never fatal — this is
    only ever used to label a task in the UI."""
    try:
        proc = run("yt-dlp", ["--no-warnings", "--skip-download",
                              "--print", "title", live_url(target)],
                   timeout_s=45.0, ok_codes=(0, 1))
    except IngestError:
        return ""
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    return lines[-1][:200] if lines else ""


def chunker_args(target: str, quality: str) -> list[str]:
    """streamlink argv for capturing a YouTube live stream.

    Same shape as twitch/kick, and the same reason for every flag: no
    ``--retry-streams`` (it retries the lookup forever, so a stream that
    ends never returns and the capture loop pins), a bounded stream
    timeout, and segment-level retries so a CDN hiccup does not end the
    session.

    ``--hls-live-edge 3`` is YouTube-specific: its live HLS carries a
    deeper buffer than Twitch's, and sitting at the default edge adds
    tens of seconds of latency to every clip for no gain in stability.
    """
    return [
        "--stdout",
        "--loglevel", "warning",
        "--stream-timeout", "60",
        "--stream-segment-attempts", "5",
        "--stream-segment-timeout", "15",
        "--hls-live-edge", "3",
        live_url(target), quality,
    ]


#: Characters Windows forbids in a path component. The chunker builds a
#: directory as `{platform}_{handle}/sNNNNN`, so a raw URL handle made a
#: path containing ':' and '/' — WinError 123 on every connect, retried
#: forever with backoff. Measured against the live ISS stream, 2026-08-12.
_UNSAFE_PATH_RE = re.compile(r'[<>:"/\|?*\x00-\x1f]+')


def capture_handle(target: str) -> str:
    """A stable, filesystem-safe name for a capture session.

    Prefers the video id (unique and short), then the channel handle,
    then a sanitised form of whatever was given. It is also the session
    key in the state DB, so it must be stable across reconnects of the
    same broadcast — a changing handle would start a new timeline
    mid-stream and mistime every clip after the drop.
    """
    raw = (target or "").strip()
    vid = video_id_from(raw)
    if vid:
        return vid
    handle = re.search(r"/(@[\w.-]{1,80})", raw)
    if handle:
        return handle.group(1)
    if raw.startswith("@"):
        return _UNSAFE_PATH_RE.sub("_", raw)[:80]
    # Last resort: strip the scheme and flatten everything illegal.
    bare = re.sub(r"^https?://(www\.)?", "", raw)
    return (_UNSAFE_PATH_RE.sub("_", bare).strip("._") or "stream")[:80]
