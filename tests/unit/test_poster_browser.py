"""Unit tests for social poster scheduling and automator dispatch."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipforge.errors import ConfigError
from clipforge.poster import (
    BaseSocialPoster,
    InstagramReelsPoster,
    TikTokPoster,
    XTwitterPoster,
    YouTubeShortsPoster,
    get_poster,
)
from clipforge.schemas.poster import PostJob, PostResult


def test_poster_factory() -> None:
    yt = get_poster("youtube")
    assert isinstance(yt, YouTubeShortsPoster)
    assert yt.platform == "youtube"

    tt = get_poster("tiktok")
    assert isinstance(tt, TikTokPoster)
    assert tt.platform == "tiktok"

    ig = get_poster("instagram")
    assert isinstance(ig, InstagramReelsPoster)
    assert ig.platform == "instagram"

    x = get_poster("x_twitter")
    assert isinstance(x, XTwitterPoster)
    assert x.platform == "x_twitter"

    with pytest.raises(ConfigError):
        get_poster("invalid_platform")


def test_post_job_schema() -> None:
    job = PostJob(
        job_id="job_001",
        clip_path="/tmp/sample.mp4",
        platform="youtube",
        title="Test Title",
        caption="Test Caption",
        hashtags=["#Shorts", "#Viral"],
        publish_mode="draft",
    )
    assert job.job_id == "job_001"
    assert job.publish_mode == "draft"
    assert len(job.hashtags) == 2


def test_post_result_schema() -> None:
    res = PostResult(
        job_id="job_001",
        platform="youtube",
        status="draft_saved",
        post_url="https://studio.youtube.com",
        completed_at="2026-07-27T15:00:00Z",
    )
    assert res.status == "draft_saved"
    assert res.error_message is None


def test_smart_peak_scheduling() -> None:
    from clipforge.poster.scheduler import compute_next_peak_window

    yt_peak = compute_next_peak_window("youtube", timezone_offset_hours=-5.0)
    assert "T" in yt_peak
    assert "-05:00" in yt_peak or "+00:00" in yt_peak or yt_peak.endswith("00")

    tt_peak = compute_next_peak_window("tiktok", timezone_offset_hours=-5.0)
    assert "T" in tt_peak

