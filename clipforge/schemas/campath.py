"""S4 output — per-frame virtual-camera crop path + speaker assignment audit."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from clipforge.schemas.base import ArtifactModel


class CropFrame(BaseModel):
    model_config = {"extra": "forbid"}

    frame: int = Field(ge=0, description="Frame index relative to clip start")
    x: int = Field(ge=0, description="Crop left, EVEN (spec §S4)")
    y: int = Field(ge=0, description="Crop top, EVEN")
    w: int = Field(gt=0, description="Crop width, EVEN, 9:16 with h")
    h: int = Field(gt=0)


class SpeakerAssignment(BaseModel):
    """Audit record of one (diarization turn → track) assignment, with the
    honesty requirement from §S4: confidence is always emitted."""

    model_config = {"extra": "forbid"}

    turn_speaker: str
    track_id: int | None = Field(None, description="None = no track matched")
    confidence: float = Field(ge=0, le=1)
    used_fallback: bool = Field(False, description="True ⇒ below τ, fallback framing")


class CamPathArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s4_tracking"

    source_ranking: str = Field(description="cache_key of the S3 artifact consumed")
    clip_start: float = Field(ge=0, description=(
        "MEDIA-relative seconds (what S4 seeks with and S6 passes to -ss). "
        "Add the transcript's abs_offset_s to get absolute stream time; S5 "
        "does exactly that when matching words to the window."))
    clip_end: float = Field(ge=0)
    framing_mode: Literal["speaker", "center", "dual_pane"] = Field(description=(
        "'center'/'dual_pane' are the documented low-confidence fallbacks"))
    frames: list[CropFrame] = []
    assignments: list[SpeakerAssignment] = []
    #: Per-shot subject-selection method counts. One MAR-decided shot must
    #: not brand the whole clip as visually speaker-selected — the record
    #: carries the actual split (panel: a 5/26 clip was labelled MAR_VISUAL
    #: with used_fallback=False and nothing in the artifact said otherwise).
    mar_shots: int = Field(0, ge=0)
    presence_shots: int = Field(0, ge=0)
    src_width: int = Field(gt=0)
    src_height: int = Field(gt=0)
    src_fps_rational: str = Field(description="Exact source fps (T10), e.g. 60000/1001")
