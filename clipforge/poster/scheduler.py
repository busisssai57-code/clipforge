"""Social post dispatcher and execution coordinator."""

from __future__ import annotations

from pathlib import Path


from clipforge.errors import ConfigError
from clipforge.log import get_logger
from clipforge.poster.base import BaseSocialPoster
from clipforge.poster.instagram import InstagramReelsPoster
from clipforge.poster.tiktok import TikTokPoster
from clipforge.poster.x_twitter import XTwitterPoster
from clipforge.poster.youtube import YouTubeShortsPoster
from clipforge.schemas.poster import PostJob, PostResult

log = get_logger(__name__)

_POSTERS: dict[str, BaseSocialPoster] = {
    "youtube": YouTubeShortsPoster(),
    "tiktok": TikTokPoster(),
    "instagram": InstagramReelsPoster(),
    "x_twitter": XTwitterPoster(),
}

# Peak audience engagement hours per platform (local target timezone hours, e.g. EST)
PEAK_ENGAGEMENT_HOURS: dict[str, list[int]] = {
    "youtube": [12, 15, 18, 21],   # 12 PM, 3 PM, 6 PM, 9 PM
    "tiktok": [9, 12, 15, 19],     # 9 AM, 12 PM, 3 PM, 7 PM
    "instagram": [11, 14, 17, 20], # 11 AM, 2 PM, 5 PM, 8 PM
    "x_twitter": [8, 11, 13, 17],  # 8 AM, 11 AM, 1 PM, 5 PM
}


def compute_next_peak_window(platform: str, timezone_offset_hours: float = -5.0) -> str:
    """Calculate the next optimal peak audience engagement time for the target platform.

    Returns an ISO-8601 formatted datetime string with timezone offset.
    """
    from datetime import datetime, timedelta, timezone

    tz = timezone(timedelta(hours=timezone_offset_hours))
    now = datetime.now(tz)

    hours = PEAK_ENGAGEMENT_HOURS.get(platform.lower(), [12, 18])

    # Find next peak hour today
    for h in hours:
        candidate = now.replace(hour=h, minute=0, second=0, microsecond=0)
        if candidate > now:
            return candidate.isoformat()

    # Fallback to first peak hour tomorrow
    tomorrow = now + timedelta(days=1)
    next_peak = tomorrow.replace(hour=hours[0], minute=0, second=0, microsecond=0)
    return next_peak.isoformat()


def get_poster(platform: str) -> BaseSocialPoster:
    """Retrieve poster implementation by platform identifier."""
    poster = _POSTERS.get(platform.lower())
    if poster is None:
        raise ConfigError(f"Unsupported social platform: {platform!r}. Supported: {list(_POSTERS.keys())}")
    return poster


def execute_post_job(job: PostJob, auth_dir: Path, headless: bool = True,
                     timezone_offset_hours: float = -5.0, *,
                     approved: bool = False) -> PostResult:
    """Dispatch ONE approved PostJob to its platform automator.

    Two gates, both enforced here because this is the single choke point
    between ClipForge and the outside world:

      * ``approved`` must be True. It is keyword-only and defaults to False,
        so a caller cannot approve a post by accident or by argument order —
        approval has to be typed out per job. Callers obtain it from a human
        decision, never from config.
      * ``publish_mode`` must be ``"draft"``. The browser automation leaves
        the post as a draft for a person to review and publish; nothing here
        completes a publish autonomously.

    Both are re-checked at dispatch rather than trusted from config, so a
    hand-built PostJob cannot route around the config-level pins.
    """
    if not approved:
        raise ConfigError(
            f"post job {job.job_id!r} was not approved. Every clip requires "
            "explicit per-clip human approval before any browser automation "
            "runs (draft-only amendment; see VERIFICATION.md).")
    if job.publish_mode != "draft":
        raise ConfigError(
            f"post job {job.job_id!r} requests publish_mode="
            f"{job.publish_mode!r}; only 'draft' is permitted. A human "
            "completes the publish step.")

    poster = get_poster(job.platform)
    log.info(
        "poster.dispatch",
        job_id=job.job_id,
        platform=job.platform,
        publish_mode=job.publish_mode,
        approved=True,
        note="draft only - a human publishes",
    )
    return poster.upload_clip(job, auth_dir=auth_dir, headless=headless)

