"""Kick live detection — BEST-EFFORT behind a feature flag (trap T4).

Kick sits behind Cloudflare bot gating and streamlink's plugin support is
not guaranteed. The containment contract, enforced here and tested:

  * every public function is TOTAL — it returns a value or ``None``,
    it NEVER raises (not even IngestError), so a Kick breakage can never
    ripple into Twitch/YouTube monitoring;
  * ``is_live`` returns ``None`` for "cannot determine" (plugin missing,
    CF block, timeout) — the monitor treats None as offline and logs it;
  * everything is skipped unless ``ingest.kick_enabled = true`` in config
    (checked by the monitor, not here — this module stays policy-free).

Detection strategy: streamlink first (has a kick plugin in recent
releases), yt-dlp as fallback (its extractor sometimes survives CF when
streamlink's does not).
"""

from __future__ import annotations

import json
from typing import Callable

from clipforge.errors import ClipForgeError
from clipforge.ingest.runner import run_tool
from clipforge.log import get_logger

log = get_logger(__name__)

RunTool = Callable[..., object]


def stream_url(handle: str) -> str:
    return f"https://kick.com/{handle.strip('/').lstrip('@')}"


def is_live(handle: str, *, run: RunTool = run_tool) -> bool | None:
    """True = live, False = confidently offline, None = cannot determine.

    Total by contract — every failure path degrades to None + a log line.
    """
    # Attempt 1: streamlink --json
    try:
        proc = run("streamlink", ["--json", stream_url(handle)],
                   timeout_s=30.0, ok_codes=(0, 1))
        data = json.loads(proc.stdout)  # type: ignore[attr-defined]
        if data.get("streams"):
            return True
        err = str(data.get("error", ""))
        if "No playable streams" in err:
            return False  # plugin worked, channel is just offline
        # Any other error (403/CF/unknown-plugin) → indeterminate.
    except (ClipForgeError, json.JSONDecodeError, AttributeError) as exc:
        log.debug("kick.streamlink_probe_failed", handle=handle, error=str(exc))
    except Exception as exc:  # T4 containment: total means TOTAL
        log.warning("kick.streamlink_probe_unexpected", handle=handle,
                    error=f"{type(exc).__name__}: {exc}")

    # Attempt 2: yt-dlp is_live probe
    try:
        proc = run("yt-dlp", ["--skip-download", "--no-warnings",
                              "--print", "is_live", stream_url(handle)],
                   timeout_s=30.0, ok_codes=(0, 1))
        out = proc.stdout.strip().lower()  # type: ignore[attr-defined]
        if out == "true":
            return True
        if out == "false":
            return False
    except (ClipForgeError, AttributeError) as exc:
        log.debug("kick.ytdlp_probe_failed", handle=handle, error=str(exc))
    except Exception as exc:
        log.warning("kick.ytdlp_probe_unexpected", handle=handle,
                    error=f"{type(exc).__name__}: {exc}")

    log.info("kick.indeterminate", handle=handle,
             note="both probes failed - treating as offline (T4 best-effort)")
    return None


def chunker_args(handle: str, quality: str) -> list[str]:
    """streamlink argv for the live chunker pipe (no ad handling on Kick)."""
    # No --retry-streams: it would retry forever and pin the capture loop
    # (see twitch.chunker_args). Our own reconnect loop handles retries.
    return [
        "--stdout",
        "--loglevel", "warning",
        "--stream-timeout", "60",
        "--stream-segment-attempts", "5",
        "--stream-segment-timeout", "15",
        stream_url(handle), quality,
    ]
