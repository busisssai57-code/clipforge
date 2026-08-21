"""A quiet window is an answer, not a crash.

Found by pointing `bta live` at a real broadcast (NASA's ISS stream,
2026-08-12). The feed is mostly ambience, so whisperx reported "No active
speech found in audio", guessed Welsh at 0.57 confidence from the silence,
and S1 died with ``No default align-model for language: cy``. Every quiet
minute of every stream would have done the same — which for live capture
is most of them.

Two behaviours are pinned here, and both are about degrading with a stated
reason rather than failing or, worse, pretending:

* no speech  → empty transcript, ``no_speech=True``, alignment never run;
* no aligner → segment text and coarse times survive, ``words_aligned``
  goes False so nothing downstream mistakes "not aligned" for "silent".
"""

from __future__ import annotations

import pytest

from clipforge.stages.base import digest_bytes
from clipforge.stages.s1_transcribe import S1Transcribe
from clipforge.state import StateDB


class _Engine:
    """Minimal S1 engine. Records what was called so the test can assert
    that alignment was SKIPPED rather than merely surviving."""

    def __init__(self, *, segments, align_raises: Exception | None = None,
                 language: str = "en"):
        self._segments = segments
        self._align_raises = align_raises
        self._language = language
        self.calls: list[str] = []

    def load_asr(self, *_a, **_kw):
        self.calls.append("load_asr")
        return object()

    def transcribe(self, _asr, _path, **_kw):
        self.calls.append("transcribe")
        return {"segments": list(self._segments), "language": self._language}

    def load_align(self, language):
        self.calls.append(f"load_align:{language}")
        if self._align_raises is not None:
            raise self._align_raises
        return object()

    def align(self, _aligner, segments, _path):
        self.calls.append("align")
        # A real aligner adds word timings; mimic that so the test can tell
        # an aligned result from a passed-through one.
        return {"segments": [
            {**s, "words": [{"word": w, "start": s["start"], "end": s["end"],
                             "score": 0.9}
                            for w in str(s.get("text", "")).split()]}
            for s in segments]}

    def load_diarizer(self, _token):
        self.calls.append("load_diarizer")
        raise RuntimeError("no diarizer in this test")

    def diarize(self, *_a, **_kw):  # pragma: no cover - never reached
        raise AssertionError("diarize must not run when load_diarizer failed")

    def assign_speakers(self, *_a, **_kw):  # pragma: no cover
        raise AssertionError("assign_speakers must not run")

    def turns_of(self, *_a, **_kw):  # pragma: no cover
        return []


@pytest.fixture()
def env(tmp_path):
    db = StateDB(tmp_path / "s.db")
    media = tmp_path / "window.mp4"
    media.write_bytes(b"\x00" * 512)  # S1 only checks the file exists
    yield db, media, tmp_path
    db.close()


def _run(engine, env, *, abs_offset: float = 0.0):
    db, media, tmp = env
    stage = S1Transcribe(db=db, artifacts_dir=tmp / "art",
                         engine_factory=lambda: engine)
    return stage.run(input_digest=digest_bytes(b"m"),
                     params={"abs_offset_s": abs_offset}, media_path=media)


# ---------------------------------------------------------- no speech

def test_a_silent_window_yields_an_empty_transcript_not_an_error(env):
    art = _run(_Engine(segments=[]), env)
    assert art.segments == []
    assert art.no_speech is True


def test_alignment_is_never_attempted_when_nothing_was_said(env):
    """The bug was not only that alignment failed — it is that alignment
    ran at all. There is nothing to align, and loading the model burns
    VRAM and a model download to align an empty list."""
    engine = _Engine(segments=[])
    _run(engine, env)
    assert not any(c.startswith("load_align") for c in engine.calls), \
        engine.calls


def test_silence_does_not_claim_word_alignment(env):
    """`words_aligned` must be False, so a downstream stage cannot read an
    empty `words` list as "aligned, and nobody spoke"."""
    art = _run(_Engine(segments=[]), env)
    assert art.words_aligned is False


def test_the_exact_shape_that_broke_the_live_capture(env):
    """whisperx on silence: no segments, plus a bogus language guess. Both
    at once is what the ISS stream produced."""
    art = _run(_Engine(segments=[], language="cy"), env)
    assert art.no_speech is True
    assert art.segments == []
    assert art.language == "cy"  # reported honestly, not laundered to 'en'


# ------------------------------------------------- alignment missing

_NO_ALIGNER = ValueError("No default align-model for language: cy")


def test_a_missing_aligner_degrades_instead_of_failing(env):
    art = _run(_Engine(segments=[{"start": 1.0, "end": 3.0, "text": "hello"}],
                       align_raises=_NO_ALIGNER), env)
    assert [s.text for s in art.segments] == ["hello"]
    assert art.words_aligned is False


def test_degraded_alignment_keeps_segment_times(env):
    art = _run(_Engine(segments=[{"start": 1.5, "end": 4.25, "text": "hi"}],
                       align_raises=_NO_ALIGNER), env)
    assert (art.segments[0].start, art.segments[0].end) == (1.5, 4.25)


def test_degraded_alignment_reports_no_words_rather_than_faking_them(env):
    art = _run(_Engine(segments=[{"start": 0.0, "end": 2.0, "text": "a b c"}],
                       align_raises=_NO_ALIGNER), env)
    assert art.segments[0].words == []
    assert art.words_aligned is False


@pytest.mark.parametrize("boom", [
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUDA out of memory"),
    OSError("checkpoint is truncated"),
])
def test_a_real_fault_during_alignment_still_fails_the_stage(env, boom):
    """The degrade is NARROW, and this is the test that keeps it narrow.

    The first version of the no-speech fix caught every exception from the
    alignment phase, which meant a CUDA fault produced a plausible-looking
    wordless transcript instead of an error. That hides a broken machine
    behind a valid artifact — the exact failure this module's docstring
    forbids. Only 'no aligner for this language' may degrade.
    """
    from clipforge.errors import RetryableStageError

    with pytest.raises(RetryableStageError):
        _run(_Engine(segments=[{"start": 0.0, "end": 1.0, "text": "x"}],
                     align_raises=boom), env)


def test_a_speech_window_still_aligns_normally(env):
    """The regression guard: the degrade paths must not have turned
    alignment off for everyone."""
    engine = _Engine(segments=[{"start": 0.0, "end": 2.0, "text": "one two"}])
    art = _run(engine, env)
    assert "align" in engine.calls
    assert art.words_aligned is True
    assert [w.text for w in art.segments[0].words] == ["one", "two"]


def test_absolute_offset_still_applies_to_a_degraded_window(env):
    """T1: every time in the artifact is absolute stream time. A degraded
    path that forgot the shift would mistime every clip from it."""
    art = _run(_Engine(segments=[{"start": 2.0, "end": 5.0, "text": "x"}],
                       align_raises=_NO_ALIGNER), env, abs_offset=600.0)
    assert art.segments[0].start == 602.0
    assert art.segments[0].end == 605.0
