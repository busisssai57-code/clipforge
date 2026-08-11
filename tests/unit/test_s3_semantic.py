"""Unit tests for Stage S3 Semantic Ranker."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipforge.errors import StageError
from clipforge.schemas.candidates import CandidatesArtifact, CandidateWindow
from clipforge.schemas.ranking import RankedArtifact
from clipforge.stages.s3_semantic import S3SemanticRanker
from clipforge.state import StateDB


def test_s3_fallback_when_video_missing(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    artifacts_dir = tmp_path / "artifacts"
    stage = S3SemanticRanker(db, artifacts_dir)

    cand = CandidateWindow(
        start=0.0, end=30.0, text="First candidate text",
        total_score=9.0, scores={"boundary": 1.0, "total": 9.0}
    )


    cands_artifact = CandidatesArtifact(
        cache_key="cands_cache_123",
        source_transcript="transcript_cache_123",
        candidates=[cand]
    )

    # S3 no longer degrades to heuristic order when the video is missing —
    # it RAISES. That change is deliberate and this test now pins it: a
    # silent fallback shipped clips ranked by keyword counting while the
    # run reported success, so the stage that defines this product could
    # be inert for weeks without anyone noticing. Failing loudly is the
    # contract.
    with pytest.raises(StageError) as err:
        stage.run(
            input_digest="test_digest",
            params={},
            candidates_artifact=cands_artifact,
            video_path=None,
        )
    assert "fallback disabled" in str(err.value)
