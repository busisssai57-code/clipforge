"""Schemas for social media post scheduling and execution results."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class PostJob(BaseModel):
    """Represents a queued or active social post task."""

    model_config = {"extra": "forbid"}

    job_id: str
    clip_path: str
    platform: Literal["youtube", "tiktok", "instagram", "x_twitter"]
    title: str
    caption: str
    hashtags: list[str] = Field(default_factory=list)
    cover_frame_offset_s: float = 0.0
    #: Single-member Literal, not a default: no job can even REPRESENT an
    #: autonomous publish. Automation prepares a draft; a human publishes.
    publish_mode: Literal["draft"] = "draft"
    #: Retained for display only. Nothing in ClipForge fires on a schedule —
    #: unattended scheduled publishing is what the human gate prevents.
    scheduled_at: str | None = None


class PostResult(BaseModel):
    """Result of an attempted social post execution."""

    model_config = {"extra": "forbid"}

    job_id: str
    platform: str
    status: Literal["success", "draft_saved", "failed"]
    post_url: str | None = None
    error_message: str | None = None
    completed_at: str
