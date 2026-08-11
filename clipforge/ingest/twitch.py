"""Twitch live detection via streamlink (spec §S0).

streamlink exit semantics: exit 0 + JSON with a non-empty ``streams`` map =
live; exit 1 + JSON with an ``error`` key = offline (or channel not found).
Both are NORMAL outcomes — only spawn/timeout/parse failures raise.
"""

from __future__ import annotations

import json
from typing import Callable

from clipforge.errors import IngestError
from clipforge.ingest.runner import run_tool
from clipforge.log import get_logger

log = get_logger(__name__)

RunTool = Callable[..., object]


def stream_url(handle: str) -> str:
    return f"https://www.twitch.tv/{handle.strip('/').lstrip('@')}"


def is_live(handle: str, *, run: RunTool = run_tool) -> bool:
    """True iff the channel is currently broadcasting.

    Any non-object JSON (Cloudflare interstitial, a bare ``[]``) is a parse
    failure, not "offline" — it must surface as IngestError so the monitor
    backs off instead of silently concluding the channel is dark.
    """
    proc = run("streamlink", ["--json", stream_url(handle)],
               timeout_s=30.0, ok_codes=(0, 1))
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise IngestError(
            f"streamlink --json produced non-JSON for {handle}: "
            f"{proc.stdout[:200]!r}") from exc
    if not isinstance(data, dict):
        raise IngestError(
            f"streamlink --json returned {type(data).__name__} for {handle}, "
            f"expected an object: {proc.stdout[:200]!r}")
    if data.get("error"):
        # 'No playable streams found' / 'Unable to find channel' = offline.
        return False
    return bool(data.get("streams"))


def chunker_args(handle: str, quality: str, *, disable_ads: bool = True,
                 stream_timeout_s: int = 60,
                 segment_attempts: int = 5) -> list[str]:
    """The streamlink argv (minus program name) for the live chunker pipe.

    **Ad handling (§S0).** ``--twitch-disable-ads`` is NOT passed: streamlink
    removed it (8.x warns "has been disabled and will be removed"), because
    ad filtering is now unconditional — the Twitch plugin detects
    ``stitched-ad-`` EXT-X-DATERANGEs and Amazon-titled segments and drops
    them for every session. Passing the flag would only emit a deprecation
    warning while doing nothing. ``disable_ads`` is kept in the signature
    (config still exposes it) and verified here rather than silently ignored:
    if an operator turns it OFF we have no way to re-enable ads, so we log
    that the setting is inert. Filtering leaves timestamp discontinuities in
    the stream, which the chunker's ffmpeg tolerance flags absorb.

    **Resilience (§2: drops and discontinuities are normal).** Segment-level
    retries and a bounded stream timeout keep a transient CDN hiccup from
    ending the capture and forcing a full reconnect.
    """
    if not disable_ads:
        log.info("twitch.ad_filtering_always_on",
                 note="streamlink >=6 filters Twitch ads unconditionally; "
                      "ingest.twitch_disable_ads=false cannot re-enable them")
    # NOTE: no --retry-streams. With it (and no --retry-max) streamlink
    # retries the stream lookup FOREVER, so a channel that goes offline never
    # exits and our capture loop never returns — the channel would be stuck
    # permanently. OUR reconnect loop is the retry mechanism (deterministic
    # backoff + jitter, §S0); two retry layers fight each other.
    return [
        "--stdout",
        "--loglevel", "warning",
        "--stream-timeout", str(stream_timeout_s),
        "--stream-segment-attempts", str(segment_attempts),
        "--stream-segment-timeout", "15",
        stream_url(handle), quality,
    ]
