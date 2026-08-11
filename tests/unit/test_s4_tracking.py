"""Unit tests for Stage S4 Spatial Tracking & Virtual Camera Path."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipforge.errors import StageError
from clipforge.schemas.ranking import RankedArtifact, RankedItem
from clipforge.schemas.campath import CamPathArtifact
from clipforge.stages.s4_tracking import OneEuroFilter, S4Tracking
from clipforge.state import StateDB


def test_one_euro_filter_smoothing() -> None:
    filter_x = OneEuroFilter(min_cutoff=1.0, beta=0.007)
    val1 = filter_x.filter(100.0, dt=1/30.0)
    val2 = filter_x.filter(110.0, dt=1/30.0)
    val3 = filter_x.filter(120.0, dt=1/30.0)

    assert val1 == 100.0
    assert val1 < val2 < 110.0
    assert val2 < val3 < 120.0


def test_s4_camera_path_even_coordinates(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    artifacts_dir = tmp_path / "artifacts"
    stage = S4Tracking(db, artifacts_dir)

    item = RankedItem(candidate_index=0, rank=1, hook_strength=8.5, justification="Best candidate")
    ranked = RankedArtifact(
        cache_key="ranked_cache_123",
        source_candidates="cands_cache_123",
        ranking_source="heuristic",
        items=[item]
    )

    # A REAL source file. This used to pass video_path=None and still get a
    # camera path, because S4 defaulted to 1920x1080 when it could not probe
    # — so the test asserted even coordinates against fabricated geometry.
    # On the actual 1280x720 fixture those crops fall outside the frame, and
    # S6 rendered zero frames. Geometry is now probed, never assumed.
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "sample_90s.mp4"
    assert fixture.exists(), f"missing test fixture: {fixture}"

    # This fixture is TTS dialogue over a static card — there is no person
    # in it to track. S4 used to answer that with a silent centre crop and
    # a `framing_mode="speaker"` audit record, so the stage that defines
    # this product could be inert while the run reported success. It now
    # RAISES, and that is the contract this pins.
    #
    # The geometry invariants that used to live here (even coordinates,
    # crop fits inside the probed frame) moved to `clipforge verify ai`,
    # which drives S4 against footage that actually contains a subject —
    # they cannot be asserted from a run that legitimately refuses.
    with pytest.raises(StageError) as err:
        stage.run(
            input_digest="test_digest",
            params={},
            ranked_artifact=ranked,
            video_path=fixture,
            start_s=0.0,
            end_s=3.0,
        )
    assert "fallback disabled" in str(err.value)
