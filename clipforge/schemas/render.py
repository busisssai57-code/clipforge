"""S5/S6 outputs — the subtitle script and the rendered clip."""

from __future__ import annotations

from pydantic import Field

from clipforge.schemas.base import ArtifactModel


class SubtitleArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s5_subtitles"

    source_campath: str = Field(description="cache_key of the S4 artifact consumed")
    ass_path: str = Field(description="Absolute path of the generated .ass script")
    clip_start: float = Field(ge=0, description="Absolute stream seconds")
    clip_end: float = Field(ge=0)
    line_count: int = Field(ge=0)
    word_count: int = Field(ge=0)
    #: sha256 of the .ass bytes. The artifact is the audit record; the file is
    #: an output. Without this a corrupted or hand-edited script would still
    #: look cached-and-valid on the next run.
    ass_sha256: str = Field(description="sha256 of the .ass file bytes")


class ClipArtifact(ArtifactModel):
    schema_version: int = 1
    stage: str = "s6_render"

    source_subtitles: str = Field(description="cache_key of the S5 artifact consumed")
    clip_path: str = Field(description="Absolute path of the rendered mp4")
    clip_sha256: str = Field(description="sha256 of the mp4 bytes")
    #: MEASURED from the rendered file (ffprobe), never the prediction — the
    #: panel caught a 231 ms misreport that muted the clip's final words via
    #: an early fade-out computed from the predicted value.
    duration_s: float = Field(gt=0)
    #: What the splice arithmetic PREDICTED (TimeMap total). QA compares the
    #: two: divergence beyond a frame means concat padded seams silently.
    expected_duration_s: float | None = Field(
        None, description="Predicted duration; None when not jump-cut")
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    encoder: str = Field(description="Encoder actually used (nvenc may fall back)")
    #: Measured by ffmpeg's loudnorm analysis pass, not assumed from the
    #: filter's presence: a filter that silently no-ops still writes a file.
    loudness_i: float = Field(description="MEASURED integrated loudness (LUFS)")
    loudness_tp: float = Field(description="MEASURED true peak (dBTP)")
    #: Both the ask and the result are recorded. A source with no headroom
    #: cannot reach a loud target under a true-peak ceiling — the fixture
    #: sits at -20.05 LUFS / -0.36 dBTP, where -14 would need +6 dB and put
    #: peaks at +5.7 dBTP. Storing only the measurement would hide that the
    #: clip is off-spec; storing only the target would be a lie.
    target_loudness_i: float = Field(description="Configured target (LUFS)")
    target_loudness_tp: float = Field(description="Configured ceiling (dBTP)")
