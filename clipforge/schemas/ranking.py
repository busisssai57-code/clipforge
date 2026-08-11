"""S3 output — semantic ranking of S2 candidates (or the heuristic fallback)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from clipforge.schemas.base import ArtifactModel


class RankedItem(BaseModel):
    model_config = {"extra": "forbid"}

    candidate_index: int = Field(ge=0, description="Index into the S2 candidates list")
    rank: int = Field(ge=1, description="1 = best")
    visual_action: float | None = Field(None, ge=0, le=10)
    hook_strength: float | None = Field(None, ge=0, le=10)
    comprehensibility: float | None = Field(None, ge=0, le=10)
    justification: str = Field("", max_length=500)
    #: Copy written by the VL model that is already looking at this
    #: candidate's frames and transcript. Generated HERE rather than in the
    #: editor stage because the model is loaded, GPU-resident and mid-loop —
    #: the editor's string-casing ("Aaron Judges Toughest Challenge Yet")
    #: had no idea what was on screen. Empty on the heuristic path, and the
    #: editor falls back to its own text when these are blank.
    title: str = Field("", max_length=120,
                       description="Scroll-stopping title, VL-written")
    hook: str = Field("", max_length=160,
                      description="Opening on-screen line, VL-written")


class RankedArtifact(ArtifactModel):
    #: 2: added `ranker` and `fallback_reason` when cloud ranking landed.
    #: Both are optional, so a v1 artifact on disk still loads.
    schema_version: int = 2
    stage: str = "s3_semantic"

    source_candidates: str = Field(description="cache_key of the S2 artifact consumed")
    ranking_source: Literal["semantic", "heuristic"] = Field(description=(
        "'heuristic' = the VL model failed twice and S2 order was used (spec §S3)"))
    #: WHICH model judged, not just whether one did. Two ranking passes can
    #: both be 'semantic' and disagree because a cloud model and a local
    #: 7B looked at the same frames; a score that cannot name its judge
    #: cannot be compared against another run or trusted after the fact.
    ranker: str | None = Field(
        None, description="e.g. 'gemini:gemini-2.5-pro' or 'local:Qwen/...'")
    #: Set when the cloud ranker was configured but did not run, so the
    #: reason a weaker model produced these numbers survives in the record.
    fallback_reason: str | None = Field(None, max_length=300)
    items: list[RankedItem] = []
    visual_tokens_used: int | None = Field(None, description="T7 observability")
