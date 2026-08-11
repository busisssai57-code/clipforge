"""S1 on the REAL GPU with the REAL whisperx stack — the CP2 half of the
Definition of Done's VRAM accounting.

DRAFT — moves to tests/integration/ after round 8 (frozen tree).

gpu-marked: auto-skipped without CUDA (tests/integration/conftest.py).
First run downloads model weights (~1.5 GB for large-v2); diarization
assertions are skipped when no HF token is configured (T5) — the transcript
itself must still be produced (degradation contract).
"""

import os
from pathlib import Path

import pytest

from clipforge.config import Secrets
from clipforge.gpu import GPU_LOCK
from clipforge.stages.base import digest_file
from clipforge.stages.s1_transcribe import S1Transcribe
from clipforge.state import StateDB

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_90s.mp4"

pytestmark = pytest.mark.gpu


@pytest.fixture()
def db(tmp_path: Path):
    d = StateDB(tmp_path / "s.db")
    yield d
    d.close()


def _hf_token() -> str | None:
    return Secrets().hf_token or os.environ.get("HF_TOKEN")


def test_s1_real_transcription_of_the_fixture(db, tmp_path):
    """The fixture's TTS dialogue is real speech: S1 must produce worded,
    timestamped segments, stay under the VRAM budget, and leave the GPU
    registry clean."""
    import torch

    stage = S1Transcribe(db=db, artifacts_dir=tmp_path / "art",
                         hf_token=_hf_token())
    art = stage.run(
        input_digest=digest_file(FIXTURE),
        params={"model": "small", "compute_type": "float16",
                "batch_size": 8, "abs_offset_s": 0.0},
        media_path=FIXTURE)

    # Real content: the dialogue mentions "pipeline" and "segments".
    text = " ".join(s.text.lower() for s in art.segments)
    assert len(art.segments) >= 5, f"only {len(art.segments)} segments"
    assert "pipeline" in text or "segment" in text, text[:400]

    # Word-level timestamps exist and are ordered.
    words = [w for s in art.segments for w in s.words]
    assert len(words) > 50
    assert all(w.end >= w.start for w in words)
    starts = [w.start for w in words]
    assert starts == sorted(starts)
    assert 0.0 <= starts[0] <= 20.0 and words[-1].end <= 95.0

    # VRAM Law: registry clean, residual allocation released.
    assert GPU_LOCK.registry.resident is None
    torch.cuda.empty_cache()
    residual_gb = torch.cuda.memory_allocated() / 1024 ** 3
    assert residual_gb < 1.0, f"{residual_gb:.2f} GB still allocated after S1"

    # Diarization: two TTS voices — assert only when the gated models are
    # reachable (T5); otherwise the degradation contract is the assertion.
    if _hf_token() and art.diarization_ok:
        speakers = {s.speaker for s in art.segments if s.speaker}
        assert len(speakers) >= 2, f"expected 2 voices, got {speakers}"
        assert art.turns, "no diarization turns"
    else:
        assert art.diarization_ok is False


def test_s1_then_s2_end_to_end_on_the_fixture(db, tmp_path):
    """The CP2 deliverable in one breath: fixture → transcript → scored
    candidate windows with absolute times inside the fixture's 90 s."""
    from clipforge.stages.s2_prefilter import S2Prefilter

    s1 = S1Transcribe(db=db, artifacts_dir=tmp_path / "art",
                      hf_token=_hf_token())
    transcript = s1.run(
        input_digest=digest_file(FIXTURE),
        params={"model": "small", "compute_type": "float16",
                "batch_size": 8, "abs_offset_s": 0.0},
        media_path=FIXTURE)

    s2 = S2Prefilter(db, tmp_path / "art")
    cands = s2.run(input_digest=transcript.cache_key,
                   params={"top_k": 5}, transcript=transcript)

    assert cands.candidates, "no candidate windows from the fixture dialogue"
    for c in cands.candidates:
        assert 0.0 <= c.start < c.end <= 95.0
        assert 30.0 <= (c.end - c.start) <= 60.0
        assert c.text.strip()
