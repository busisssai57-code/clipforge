"""S1 transcribe: engine-boundary mechanics, no GPU/whisperx required.

DRAFT — moves to tests/unit/ once round 7 completes.
"""

import weakref
from pathlib import Path
from typing import Any

import pytest

from clipforge.errors import CoResidencyError, RetryableStageError
from clipforge.gpu import GPU_LOCK, ModelClass
from clipforge.stages.base import digest_bytes
from clipforge.stages.s1_transcribe import MAX_BATCH_SIZE, S1Transcribe
from clipforge.state import StateDB


class FakeEngine:
    def __init__(self, calls: list[str], *, diarize_fails=False,
                 transcribe_fails=False):
        self.calls = calls
        self.diarize_fails = diarize_fails
        self.transcribe_fails = transcribe_fails
        self.refs: dict[str, Any] = {}
        #: {model: still_alive} sampled when the NEXT model loads.
        self.residency_at_load: dict[str, bool] = {}
        self.align_residency_at_load: dict[str, bool] = {}

    def _obj(self, name):
        obj = type(name.upper(), (), {})()
        self.refs[name] = weakref.ref(obj)
        return obj

    def load_asr(self, model_name, compute_type):
        self.calls.append(f"load_asr({model_name},{compute_type})")
        return self._obj("asr")

    def transcribe(self, asr, media_path, *, batch_size, language):
        self.calls.append(f"transcribe(batch={batch_size},lang={language})")
        if self.transcribe_fails:
            raise RuntimeError("CUDA error: out of memory")
        return {"language": "en",
                "segments": [{"start": 0.5, "end": 2.0, "text": "Hi."}]}

    def load_align(self, language):
        self.calls.append(f"load_align({language})")
        # Same instant-of-truth sample as load_diarizer: the ASR model must
        # already be unreachable before alignment weights land on the card.
        self.align_residency_at_load = {
            name: ref() is not None for name, ref in self.refs.items()}
        # Mirrors the real engine: model AND metadata in one handle, so both
        # die together (retaining metadata pinned 369 MB of weights).
        return (self._obj("align"), {"language": language})

    def align(self, aligner, segments, media_path):
        self.calls.append("align")
        return {"segments": [{
            "start": 0.5, "end": 2.0, "text": "Hi.",
            "words": [{"word": "Hi.", "start": 0.5, "end": 2.0, "score": 0.99}],
        }]}

    def load_diarizer(self, hf_token):
        self.calls.append(f"load_diarizer(token={'y' if hf_token else 'n'})")
        # THE VRAM Law, pinned at the only instant that can prove it: the
        # moment a new model would load, every earlier model must already be
        # unreachable. A mutant that moved teardown to end-of-stage (all
        # three handles held live, unload LABELS still emitted in order)
        # passed the entire gate — labels are not evidence, dead refs are.
        self.residency_at_load = {
            name: ref() is not None for name, ref in self.refs.items()}
        if self.diarize_fails:
            raise RuntimeError("403 gated")
        return self._obj("diar")

    def diarize(self, diarizer, media_path):
        self.calls.append("diarize")
        return "D"

    def assign_speakers(self, diarization, aligned):
        self.calls.append("assign_speakers")
        for s in aligned["segments"]:
            s["speaker"] = "SPEAKER_00"
            for w in s["words"]:
                w["speaker"] = "SPEAKER_00"
        return aligned

    def turns_of(self, diarization):
        return [("SPEAKER_00", 0.5, 2.0)]


@pytest.fixture()
def env(tmp_path: Path):
    db = StateDB(tmp_path / "s.db")
    media = tmp_path / "chunk.mp4"
    media.write_bytes(b"\x00" * 512)
    yield db, media, tmp_path
    db.close()


def make_stage(db, tmp_path, engine, **kw):
    return S1Transcribe(db=db, artifacts_dir=tmp_path / "art",
                        engine_factory=lambda: engine, **kw)


def test_full_flow_order_and_artifact(env):
    db, media, tmp = env
    calls: list[str] = []
    stage = make_stage(db, tmp, FakeEngine(calls))
    art = stage.run(input_digest=digest_bytes(b"m"), params={},
                    media_path=media)

    # The §S1 pipeline order: ASR -> align -> diarize -> assign.
    names = [c.split("(")[0] for c in calls]
    assert names == ["load_asr", "transcribe", "load_align", "align",
                     "load_diarizer", "diarize", "assign_speakers"]
    assert art.language == "en"
    assert art.segments[0].speaker == "SPEAKER_00"
    assert art.diarization_ok is True
    assert art.turns[0].speaker == "SPEAKER_00"


def test_each_model_is_freed_before_the_next_loads(env):
    """The VRAM Law's strongest form: teardown INTERLEAVES with loading.

    Spec §S1 lists an unload ORDER (alignment, diarization, ASR), which
    presumes all three are resident at teardown time. Per-phase teardown
    makes that presumption false by construction — each model is released
    before the next is loaded — so what we assert is the stronger property
    the order was a proxy for: at no point are two model classes live.
    """
    db, media, tmp = env
    calls: list[str] = []
    engine = FakeEngine(calls)
    stage = make_stage(db, tmp, engine)
    stage.run(input_digest=digest_bytes(b"m"), params={}, media_path=media)

    # Freed in the order used, each before the next load.
    assert stage._last_unload_order == ["asr", "align", "diar"], \
        stage._last_unload_order

    # Interleaving proof: between any two loads there is an unload.
    loads = [c.split("(")[0] for c in calls if c.startswith("load_")]
    assert loads == ["load_asr", "load_align", "load_diarizer"], loads
    for name, ref in engine.refs.items():
        assert ref() is None, f"{name} still alive after its phase teardown"


def test_no_two_models_are_ever_resident(env):
    """§3.1's actual property, sampled at the only instants that can prove
    it — the moment each new model loads.

    The panel showed the previous suite could not see this: a mutant that
    kept ALL THREE handles alive until the stage ended (emitting the same
    unload labels in the same order) passed the entire gate. Unload
    bookkeeping is not evidence of unloading; an unreachable reference is.
    """
    db, media, tmp = env
    engine = FakeEngine([])
    stage = make_stage(db, tmp, engine)
    stage.run(input_digest=digest_bytes(b"m"), params={}, media_path=media)

    assert engine.align_residency_at_load, "alignment never loaded"
    assert engine.align_residency_at_load.get("asr") is False, (
        "ASR model was STILL RESIDENT when the alignment model loaded: "
        f"{engine.align_residency_at_load}")

    assert engine.residency_at_load, "diarizer never loaded"
    still_live = [n for n, alive in engine.residency_at_load.items() if alive]
    assert not still_live, (
        f"{still_live} still resident when the diarizer loaded - the VRAM "
        "Law permits exactly one model class at a time")


def test_a_phase_failure_frees_that_phase_before_unwinding(env):
    """A failure inside phase N must not leave phase N-1's weights resident
    while the exception propagates — that was the whole point of moving
    teardown into each phase's own finally."""
    db, media, tmp = env

    class AlignBlows(FakeEngine):
        def align(self, aligner, segments, media_path):
            raise RuntimeError("alignment CUDA fault")

    engine = AlignBlows([])
    stage = make_stage(db, tmp, engine)
    with pytest.raises(RetryableStageError):
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=media)

    # ASR was freed before alignment even started; alignment freed on the
    # way out. Nothing survives the unwind.
    assert stage._last_unload_order == ["asr", "align"], \
        stage._last_unload_order
    for name, ref in engine.refs.items():
        assert ref() is None, f"{name} leaked when a later phase failed"


def test_batch_size_is_hard_capped(env):
    db, media, tmp = env
    calls: list[str] = []
    stage = make_stage(db, tmp, FakeEngine(calls))
    stage.run(input_digest=digest_bytes(b"m"), params={"batch_size": 512},
              media_path=media)
    # Hardcoded 8, NOT MAX_BATCH_SIZE: an assertion that compares against the
    # constant under test passes trivially when that constant is neutralized.
    # Setting MAX_BATCH_SIZE = 512 left this file at 18 passed and the whole
    # pytest half green; only verify.all caught it.
    assert "transcribe(batch=8,lang=None)" in calls, calls
    assert MAX_BATCH_SIZE == 8, "the spec's §S1 VRAM protection was changed"


def test_absolute_offset_shifts_all_times(env):
    db, media, tmp = env
    stage = make_stage(db, tmp, FakeEngine([]))
    art = stage.run(input_digest=digest_bytes(b"m"),
                    params={"abs_offset_s": 900.0}, media_path=media)
    assert art.segments[0].start == 900.5
    assert art.segments[0].words[0].end == 902.0
    assert art.turns[0].end == 902.0


def test_diarization_failure_degrades_not_fatal(env):
    db, media, tmp = env
    stage = make_stage(db, tmp, FakeEngine([], diarize_fails=True))
    art = stage.run(input_digest=digest_bytes(b"m"), params={},
                    media_path=media)
    assert art.diarization_ok is False
    assert art.segments and art.turns == []
    # The diarization phase's teardown runs even though load_diarizer RAISED.
    # "It never loaded, so nothing to free" is exactly wrong for pyannote: a
    # pipeline that dies part-way through construction (OOM, a half-fetched
    # checkpoint) has already allocated. Nesting the ritual inside the try
    # that binds the handle skipped it on precisely that path.
    assert stage._last_unload_order == ["asr", "align", "diar"]


def test_teardown_reclaims_reference_CYCLES_not_just_refcounts(env):
    """`gc.collect()` is the load-bearing step of the ritual, and nothing
    used to protect it — deleting it left every gate green.

    `del` alone drops a refcount; it cannot free a cycle. A torch Module is
    exactly that shape (module <-> `_parameters`, backward hooks, optimizer
    references), so on the real pipeline the weights survive the `del` and
    stay resident until some later automatic collection — which is precisely
    the non-determinism the VRAM Law forbids. Automatic gc is disabled here
    so the only thing that can free the cycle is the ritual itself.
    """
    import gc as _gc

    db, media, tmp = env
    seen: dict[str, Any] = {}

    class Cyclic:
        def __init__(self) -> None:
            self.loop = self          # unreachable by refcounting alone

    class CyclicEngine(FakeEngine):
        def load_asr(self, model_name, compute_type):
            obj = Cyclic()
            seen["asr"] = weakref.ref(obj)
            self.refs["asr"] = seen["asr"]
            return obj

    _gc.disable()
    try:
        stage = make_stage(db, tmp, CyclicEngine([]))
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=media)
        assert seen["asr"]() is None, (
            "the ASR model survived its phase teardown: `del` dropped a "
            "refcount but the cycle needs gc.collect()")
    finally:
        _gc.enable()


def test_unload_flushes_the_cuda_allocator_when_torch_is_present(env):
    """`empty_cache()` returns freed blocks to the driver; without it the
    caching allocator keeps them and the next stage's budget assertion sees
    memory that is free in Python but not to the card. Untestable against
    real CUDA in a unit test, so the call itself is pinned via a stub."""
    import sys
    import types

    db, media, tmp = env
    calls: list[str] = []
    fake = types.ModuleType("torch")
    _GB = 1024 ** 3
    fake.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda *a: (20 * _GB, 24 * _GB),
        memory_allocated=lambda *a: 0,
        max_memory_allocated=lambda *a: 0,
        reset_peak_memory_stats=lambda *a: None,
        synchronize=lambda *a: None,
        empty_cache=lambda: calls.append("empty_cache"))
    real = sys.modules.get("torch")
    sys.modules["torch"] = fake
    try:
        stage = make_stage(db, tmp, FakeEngine([]))
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=media)
    finally:
        if real is not None:
            sys.modules["torch"] = real
        else:
            del sys.modules["torch"]

    # Three phase teardowns (asr, align, diar) plus one on gpu_session exit.
    # Pinning the exact count catches deletion of EITHER: drop the per-phase
    # flush and it falls to 1, drop the session-exit flush and it falls to 3.
    assert calls == ["empty_cache"] * 4, (
        f"expected 3 phase flushes + 1 session-exit flush, got {calls}")


def test_a_diarizer_that_allocates_then_raises_is_still_released(env):
    """The Windows leak shape: failure DURING construction, not after."""
    import weakref

    db, media, tmp = env
    leaked: dict[str, Any] = {}

    class PartialAllocEngine(FakeEngine):
        def load_diarizer(self, hf_token):
            partial = self._obj("diar")          # allocation happened...
            leaked["ref"] = weakref.ref(partial)
            self.refs["diar"] = leaked["ref"]
            raise RuntimeError("checkpoint truncated")  # ...then it died

    stage = make_stage(db, tmp, PartialAllocEngine([]))
    art = stage.run(input_digest=digest_bytes(b"m"), params={},
                    media_path=media)

    assert art.diarization_ok is False, "S1 must still emit a transcript"
    assert stage._last_unload_order == ["asr", "align", "diar"]
    assert leaked["ref"]() is None, (
        "the half-constructed diarizer is still reachable after S1 returned "
        "— on the real pipeline that is resident VRAM")


def test_transcribe_failure_is_retryable_and_residency_released(env):
    db, media, tmp = env
    stage = make_stage(db, tmp, FakeEngine([], transcribe_fails=True))
    with pytest.raises(RetryableStageError):
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=media)
    assert GPU_LOCK.registry.resident is None, "residency leaked on failure"


def test_asr_ref_dies_even_when_transcribe_fails(env):
    db, media, tmp = env
    engine = FakeEngine([], transcribe_fails=True)
    stage = make_stage(db, tmp, engine)
    with pytest.raises(RetryableStageError):
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=media)
    assert engine.refs["asr"]() is None, "asr model leaked on the error path"


def test_missing_media_is_retryable(env):
    db, _media, tmp = env
    stage = make_stage(db, tmp, FakeEngine([]))
    with pytest.raises(RetryableStageError, match="vanished"):
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=tmp / "nope.mp4")


def test_vl_coload_during_s1_raises(env):
    db, media, tmp = env

    class Spy(FakeEngine):
        def transcribe(self, asr, media_path, *, batch_size, language):
            with pytest.raises(CoResidencyError):
                GPU_LOCK.registry.register(ModelClass.VL)
            return super().transcribe(asr, media_path, batch_size=batch_size,
                                      language=language)

    stage = make_stage(db, tmp, Spy([]))
    stage.run(input_digest=digest_bytes(b"m"), params={}, media_path=media)
    assert GPU_LOCK.registry.resident is None


def test_engine_import_failure_is_retryable(env):
    db, media, tmp = env

    def broken_factory():
        raise ImportError("No module named 'whisperx'")

    stage = S1Transcribe(db=db, artifacts_dir=tmp / "art",
                         engine_factory=broken_factory)
    with pytest.raises(RetryableStageError):
        stage.run(input_digest=digest_bytes(b"m"), params={},
                  media_path=media)


def test_engine_picks_the_token_kwarg_the_install_accepts(monkeypatch):
    """whisperx renamed the diarizer's token kwarg (use_auth_token -> token
    in 3.8). Guessing wrong raises TypeError, which the stage then reports
    as a gated-model failure — sending the operator hunting for a token that
    was never the problem (observed on a real run)."""
    from clipforge.stages.s1_transcribe import WhisperXEngine

    seen: dict[str, object] = {}

    class NewStyle:  # whisperx >= 3.8
        def __init__(self, model_name=None, token=None, device=None,
                     cache_dir=None):
            seen["kwarg"] = "token"
            seen["value"] = token

    class OldStyle:  # whisperx < 3.8
        def __init__(self, use_auth_token=None, device=None):
            seen["kwarg"] = "use_auth_token"
            seen["value"] = use_auth_token

    for cls, expected in ((NewStyle, "token"), (OldStyle, "use_auth_token")):
        seen.clear()
        engine = WhisperXEngine.__new__(WhisperXEngine)  # skip whisperx import
        engine.device = "cpu"

        class FakeWx:
            DiarizationPipeline = cls

            @staticmethod
            def load_audio(path):
                return "AUDIO"

        engine._wx = FakeWx
        engine.load_diarizer("hf_tok")
        assert seen["kwarg"] == expected, seen
        assert seen["value"] == "hf_tok"


def test_engine_diarizes_from_memory_not_a_path():
    """pyannote decodes file paths through torchcodec, which supports only
    FFmpeg 4-7 and fails to load against a current ffmpeg — so every
    path-based diarization dies with a DLL error unrelated to its cause.
    Decode with whisperx's own ffmpeg loader instead."""
    from clipforge.stages.s1_transcribe import WhisperXEngine

    engine = WhisperXEngine.__new__(WhisperXEngine)
    engine.device = "cpu"
    loaded: list[str] = []

    class FakeWx:
        @staticmethod
        def load_audio(path):
            loaded.append(path)
            return "DECODED-ARRAY"

    engine._wx = FakeWx
    got: list[object] = []
    engine.diarize(lambda audio: got.append(audio) or "D",
                   Path("chunk.mp4"))

    assert loaded == ["chunk.mp4"], "audio was not decoded via ffmpeg"
    assert got == ["DECODED-ARRAY"], "a PATH was handed to pyannote"


def test_words_without_timestamps_are_dropped_not_fatal(env):
    db, media, tmp = env

    class NoTsEngine(FakeEngine):
        def align(self, aligner, segments, media_path):
            return {"segments": [{
                "start": 0.0, "end": 2.0, "text": "Hi there.",
                "words": [{"word": "Hi", "start": 0.0, "end": 1.0, "score": 0.9},
                          {"word": "there.", "start": None, "end": None}],
            }]}

    stage = make_stage(db, tmp, NoTsEngine([]))
    art = stage.run(input_digest=digest_bytes(b"m"), params={},
                    media_path=media)
    assert len(art.segments[0].words) == 1  # the timestamped one survives


# ------------------------------------------------- languages with no aligner

def test_a_language_with_no_aligner_still_produces_a_transcript(env):
    """The pipeline's most common historical failure, and the one waiting
    for the operator's own content.

    Six of the failed jobs in this workspace died on
    ``ValueError: No default align-model for language: cy`` — whisperx
    transcribed the audio and had no wav2vec2 aligner to force-align it,
    and the stage raised. Somali has no aligner either, so the Somali
    format this project is being built to copy would hit exactly this.

    A missing aligner is a fact about the LANGUAGE. The transcript is
    still worth having; only the word timings are lost, and captions
    degrade to segment level.
    """
    db, media, tmp = env
    calls: list[str] = []

    class NoAligner(FakeEngine):
        def load_align(self, language):
            calls.append(f"load_align({language})")
            raise ValueError(f"No default align-model for language: {language}")

    art = make_stage(db, tmp, NoAligner(calls)).run(
        input_digest=digest_bytes(b"m"), params={}, media_path=media,
        abs_offset_s=0.0)

    assert art.segments, "the transcript was thrown away with the alignment"
    assert art.words_aligned is False, (
        "a transcript with no word timings must SAY it has none, or S5 will "
        "build karaoke captions out of times nobody measured")


def test_a_real_fault_during_alignment_still_raises(env):
    """The narrowness that makes the degrade safe.

    A CUDA fault, an OOM or a half-fetched checkpoint means the machine
    is in trouble. Emitting a wordless transcript for those would hide a
    hardware problem behind a plausible artifact.
    """
    db, media, tmp = env
    calls: list[str] = []

    class BrokenCard(FakeEngine):
        def load_align(self, language):
            calls.append("load_align")
            raise RuntimeError("CUDA error: out of memory")

    with pytest.raises(Exception) as err:
        make_stage(db, tmp, BrokenCard(calls)).run(
            input_digest=digest_bytes(b"m"), params={}, media_path=media,
            abs_offset_s=0.0)
    assert "out of memory" in str(err.value)
