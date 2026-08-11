"""S6 output — the final clip record (the pipeline's terminal artifact)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from clipforge.schemas.base import ArtifactModel


class ClipArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s6_render"

    mp4_path: str
    clip_start: float = Field(ge=0, description="Absolute stream seconds")
    duration_s: float = Field(gt=0)
    width: int
    height: int
    encoder_used: Literal["h264_nvenc", "libx264"]
    # Loudness verification record (Definition of Done: −14 ±1 LUFS, measured)
    measured_i_lufs: float
    measured_tp_db: float
    subtitle_path: str = Field(description="The .ass burned into this clip")
    campath_source: str = Field(description="cache_key of the S4 artifact used")
