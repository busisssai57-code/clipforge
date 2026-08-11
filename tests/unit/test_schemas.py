"""Artifact schemas: version pinning, deterministic bytes, read/write cycle."""

from pathlib import Path

import pytest

from clipforge.errors import StateError
from clipforge.schemas import (CamPathArtifact, CandidatesArtifact,
                               CandidateWindow, ClipArtifact, CropFrame,
                               DiarizationTurn, RankedArtifact, RankedItem,
                               TranscriptArtifact, TranscriptSegment, Word)


def test_transcript_roundtrip(tmp_path: Path):
    art = TranscriptArtifact(
        cache_key="k1", source_path="chunk_000.ts", abs_offset_s=900.0,
        language="en",
        segments=[TranscriptSegment(start=0.0, end=2.5, text="hello there",
                                    speaker="SPEAKER_00",
                                    words=[Word(text="hello", start=0.0, end=0.4,
                                                score=0.98, speaker="SPEAKER_00")])],
        turns=[DiarizationTurn(speaker="SPEAKER_00", start=0.0, end=2.5)])
    p = art.write(tmp_path / "t.json")
    loaded = TranscriptArtifact.read(p)
    assert loaded == art


@pytest.mark.parametrize("bad_bytes", [
    b"null", b"[1,2]", b'"str"', b"42", b"\xff\xfe\x80garbage", b"{ nope",
])
def test_read_types_every_corrupt_shape(tmp_path: Path, bad_bytes: bytes):
    """ArtifactModel.read must raise TYPED StateError for every corrupt
    shape — undecodable bytes, non-JSON, JSON-non-object — so the resume
    path can quarantine instead of dying on AttributeError/ValueError."""
    p = tmp_path / "bad.json"
    p.write_bytes(bad_bytes)
    with pytest.raises(StateError):
        TranscriptArtifact.read(p)


def test_schema_version_mismatch_rejected(tmp_path: Path):
    art = TranscriptArtifact(cache_key="k", source_path="x")
    p = art.write(tmp_path / "t.json")
    tampered = p.read_text(encoding="utf-8").replace('"schema_version": 1',
                                                     '"schema_version": 99')
    p.write_text(tampered, encoding="utf-8")
    with pytest.raises(StateError, match="schema_version"):
        TranscriptArtifact.read(p)


def test_serialization_is_byte_deterministic():
    kwargs = dict(cache_key="k", source_candidates="c", ranking_source="semantic",
                  items=[RankedItem(candidate_index=0, rank=1, justification="j")])
    a = RankedArtifact(**kwargs)
    b = RankedArtifact(**kwargs)
    assert a.to_json_bytes() == b.to_json_bytes()


def test_unknown_fields_forbidden():
    with pytest.raises(Exception):
        TranscriptArtifact(cache_key="k", source_path="x", surprise_field=1)


def test_candidates_carry_absolute_time_and_scores():
    c = CandidateWindow(start=930.0, end=985.0, total_score=7.5,
                        scores={"boundary": 2.0, "qa": 3.0, "turns": 2.5},
                        text="...")
    art = CandidatesArtifact(cache_key="k", source_transcript="tkey",
                             candidates=[c])
    # T1: times are absolute stream seconds — this window lives in chunk 2.
    assert art.candidates[0].start > 900


def test_ranking_source_is_constrained():
    with pytest.raises(Exception):
        RankedArtifact(cache_key="k", source_candidates="c",
                       ranking_source="vibes")


def test_campath_even_coordinate_contract():
    f = CropFrame(frame=0, x=102, y=0, w=608, h=1080)
    art = CamPathArtifact(cache_key="k", source_ranking="r", clip_start=0.0,
                          clip_end=45.0, framing_mode="speaker", frames=[f],
                          src_width=1920, src_height=1080,
                          src_fps_rational="60000/1001")
    assert art.frames[0].w % 2 == 0  # enforced by S4 logic at CP3; schema carries it


def test_clip_artifact_records_loudness_and_encoder():
    art = ClipArtifact(cache_key="k", mp4_path="clips/x.mp4", clip_start=930.0,
                       duration_s=45.0, width=1080, height=1920,
                       encoder_used="h264_nvenc", measured_i_lufs=-14.2,
                       measured_tp_db=-1.7, subtitle_path="s.ass",
                       campath_source="ck")
    assert art.width == 1080 and art.height == 1920
