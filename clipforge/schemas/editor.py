"""Schema for S3.5 Editor Agent decisions and social metadata."""

from __future__ import annotations

from pydantic import Field
from clipforge.schemas.base import ArtifactModel


class EditorArtifact(ArtifactModel):
    """Artifact produced by S3.5 Editor Agent for a top candidate clip."""

    schema_version: int = 1
    stage: str = "editor"
    cache_key: str

    candidate_id: str
    hook_score: float = Field(..., ge=0.0, le=1.0, description="Confidence of initial 3s hook")
    hook_text: str = Field(..., description="Transcript excerpt identified as hook")
    title: str = Field(..., description="Primary optimized viral title")
    title_variations: list[str] = Field(default_factory=list, description="Alternative A/B test titles")
    captions: dict[str, str] = Field(default_factory=dict, description="Platform-tailored post descriptions")
    hashtags: list[str] = Field(default_factory=list, description="Target hashtags including #Shorts, etc.")
    cover_frame_offset_s: float = Field(0.0, ge=0.0, description="Offset in seconds for thumbnail/cover frame")
    style_profile: str = Field("viral_fast", description="Editing profile used")
