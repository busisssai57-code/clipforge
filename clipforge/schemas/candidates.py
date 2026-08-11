"""S2 output — scored candidate windows (top-K after NMS), all scores kept."""

from __future__ import annotations

from pydantic import BaseModel, Field

from clipforge.schemas.base import ArtifactModel


class CandidateWindow(BaseModel):
    model_config = {"extra": "forbid"}

    start: float = Field(ge=0, description="Seconds, ABSOLUTE stream time (T1)")
    end: float = Field(ge=0)
    total_score: float
    # Per-heuristic breakdown persisted so S3's fallback ordering is auditable
    # and so tuning sessions can replay scoring offline (spec §S2).
    scores: dict[str, float] = {}
    text: str = Field(description="Window transcript text (S3 prompt input)")


class CandidatesArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s2_prefilter"

    source_transcript: str = Field(description="cache_key of the S1 artifact consumed")
    candidates: list[CandidateWindow] = Field(
        default=[], description="Sorted by total_score desc — S3's fallback order")
