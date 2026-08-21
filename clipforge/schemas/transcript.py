"""S1 output — transcript with word timings, speakers, and turn timeline."""

from __future__ import annotations

from pydantic import BaseModel, Field

from clipforge.schemas.base import ArtifactModel


class Word(BaseModel):
    model_config = {"extra": "forbid"}

    text: str
    start: float = Field(ge=0, description="Seconds, absolute stream time")
    end: float = Field(ge=0)
    score: float | None = Field(None, ge=0, le=1, description="Alignment confidence")
    speaker: str | None = Field(None, description="Diarization label, e.g. SPEAKER_00")


class TranscriptSegment(BaseModel):
    model_config = {"extra": "forbid"}

    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str
    speaker: str | None = None
    words: list[Word] = []


class DiarizationTurn(BaseModel):
    """One contiguous speaker turn from pyannote — S4's cross-modal anchor."""

    model_config = {"extra": "forbid"}

    speaker: str
    start: float = Field(ge=0)
    end: float = Field(ge=0)


class TranscriptArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s1_transcribe"

    source_path: str = Field(description="The chunk/virtual-window this transcribes")
    abs_offset_s: float = Field(0.0, ge=0, description=(
        "Absolute stream time of this window's t=0 (T1: dedup by absolute time)"))
    language: str | None = None
    segments: list[TranscriptSegment] = []
    turns: list[DiarizationTurn] = []
    diarization_ok: bool = Field(True, description=(
        "False when pyannote failed and speakers are absent — downstream "
        "stages must degrade explicitly, not guess"))
    words_aligned: bool = Field(True, description=(
        "False when forced alignment did not run, so segments carry text "
        "and coarse times but NO word timings. Captions and the editor's "
        "word marks both key off word-level times, so this must be stated "
        "rather than inferred from an empty `words` list — which is also "
        "what a genuinely wordless segment looks like."))
    no_speech: bool = Field(False, description=(
        "True when the ASR found no speech at all in this window. A real "
        "answer, not a failure: live capture is full of music, ambience "
        "and silence, and a window with nothing said is skipped rather "
        "than crashing the run."))
