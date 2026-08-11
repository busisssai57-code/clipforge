"""S7 output — the mechanical quality gate every clip must pass.

The QA stage is the pipeline's answer to "is this clip actually shippable"
WITHOUT a human watching it: every check is measured from the rendered bytes
by ffmpeg/ffprobe, never trusted from the producing stage's own report. A
renderer that lies about its output should be caught here, which is why S7
re-derives everything it can.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from clipforge.schemas.base import ArtifactModel


class QACheck(BaseModel):
    model_config = {"extra": "forbid"}

    name: str = Field(description="Stable check identifier, kebab-case")
    #: "fail" blocks the clip; "warn" ships it but records the deviation.
    severity: Literal["fail", "warn"]
    passed: bool
    measured: str = Field(description="What was actually observed")
    expected: str = Field(description="What the spec/config demands")


class QAArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s7_qa"

    source_clip: str = Field(description="cache_key of the ClipArtifact judged")
    clip_path: str
    #: True only if every severity="fail" check passed. Warnings do not block.
    passed: bool
    checks: list[QACheck] = []
    #: Count shortcuts for log lines and dashboards.
    failed_count: int = Field(ge=0)
    warned_count: int = Field(ge=0)
