"""Unit tests for S3.5 Editor Agent stage."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipforge.schemas.editor import EditorArtifact
from clipforge.schemas.ranking import RankedArtifact, RankedItem
from clipforge.schemas.transcript import TranscriptArtifact, TranscriptSegment, Word
from clipforge.stages.s3_5_editor import S3_5_EditorAgent
from clipforge.state import StateDB


def test_editor_agent_execution(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    artifacts_dir = tmp_path / "artifacts"
    editor = S3_5_EditorAgent(db, artifacts_dir)

    words = [
        Word(text="What", start=0.0, end=0.3, score=0.95),
        Word(text="just", start=0.3, end=0.6, score=0.95),
        Word(text="happened?", start=0.6, end=1.0, score=0.95),
        Word(text="That", start=1.2, end=1.5, score=0.95),
        Word(text="was", start=1.5, end=1.8, score=0.95),
        Word(text="insane!", start=1.8, end=2.2, score=0.95),
    ]
    segment = TranscriptSegment(start=0.0, end=2.2, text="What just happened? That was insane!", words=words, speaker="SPEAKER_00")
    transcript = TranscriptArtifact(
        schema_version=1,
        stage="s1_transcribe",
        cache_key="transcript_key_123",
        source_path="/tmp/sample.mp4",
        abs_offset_s=0.0,
        language="en",
        diarization_ok=True,
        segments=[segment],
    )

    item = RankedItem(candidate_index=0, rank=1, hook_strength=8.5, justification="High tension question")
    ranked = RankedArtifact(
        schema_version=1,
        stage="s3_rank",
        cache_key="ranked_key_123",
        source_candidates="s2_candidates_key",
        ranking_source="semantic",
        items=[item],
    )


    params = {"style_profile": "viral_fast"}

    artifact = editor.run(
        input_digest="test_digest",
        params=params,
        ranked_artifact=ranked,
        transcript_artifact=transcript,
        candidate_id="cand_001",
    )

    assert isinstance(artifact, EditorArtifact)
    assert artifact.candidate_id == "cand_001"
    assert artifact.hook_score >= 0.7
    assert "youtube" in artifact.captions
    assert "tiktok" in artifact.captions
    assert "#Shorts" in artifact.hashtags
    assert len(artifact.title_variations) >= 2
