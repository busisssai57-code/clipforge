"""CP2 deterministic gate — S1 + S2 contracts, offline.

Checks (no GPU, no whisperx, no network — the S1 engine is faked):
  1. S2 determinism: identical transcript ⇒ byte-identical artifact.
  2. S2 window law: every candidate is 30–60 s, sentence-aligned, NMS'd
     below the IoU ceiling, top-K, scores persisted per component.
  3. S2 heuristics behave directionally: a Q&A window outscores dead air;
     crosstalk is not rewarded over conversation.
  4. S1 mechanics via a fake engine: unload order is alignment → diarize →
     ASR; every model reference is DEAD afterwards (weakref-proven);
     batch_size is hard-capped at 8; absolute offset shifts every time.
  5. VRAM Law inside S1: the suite runs under the ASR residency class, and
     a VL registration during the window raises.
  6. Diarization failure degrades (diarization_ok=False, no speakers) —
     it must never cost the transcript.

Run: python -m clipforge.verify.ai
"""

from __future__ import annotations

import tempfile
import traceback
import weakref
from pathlib import Path
from typing import Any, Callable

from clipforge.errors import CoResidencyError, StageError
from clipforge.gpu import GPU_LOCK, ModelClass
from clipforge.schemas import TranscriptArtifact, TranscriptSegment, Word
from clipforge.stages.base import digest_bytes
from clipforge.stages.s1_transcribe import MAX_BATCH_SIZE, S1Transcribe
from clipforge.stages.s2_prefilter import S2Prefilter
from clipforge.state import StateDB

def _repo_root() -> Path:
    """Repository root, resolved from this file rather than the cwd.

    `clipforge verify` is run from wherever the operator happens to be, so a
    relative fixture path would resolve differently per invocation.
    """
    return Path(__file__).resolve().parents[2]


_CHECKS: list[tuple[str, Callable[[], None]]] = []


def check(name: str):
    def deco(fn: Callable[[], None]):
        _CHECKS.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------------
# fixtures (pure, deterministic)
# --------------------------------------------------------------------------


def _seg(start: float, end: float, text: str, speaker: str) -> TranscriptSegment:
    words = []
    tokens = text.split()
    if tokens:
        step = (end - start) / len(tokens)
        for i, tok in enumerate(tokens):
            words.append(Word(text=tok, start=start + i * step,
                              end=start + (i + 1) * step, score=0.9,
                              speaker=speaker))
    return TranscriptSegment(start=start, end=end, text=text,
                             speaker=speaker, words=words)


def _conversation_transcript() -> TranscriptArtifact:
    """~90 s of alternating Q&A — plenty of legal 30–60 s windows."""
    segments = []
    t = 0.0
    for i in range(12):
        speaker = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
        if i % 2 == 0:
            text = f"So what do you think about topic number {i} here today?"
        else:
            text = (f"Honestly the answer to that is nuanced but here is the "
                    f"full story about point {i} and why it matters so much.")
        segments.append(_seg(t, t + 7.5, text, speaker))
        t += 7.5
    return TranscriptArtifact(cache_key="t-conv", source_path="conv.mp4",
                              segments=segments,
                              turns=[], diarization_ok=True)


class _FakeEngine:
    """Scripted S1 engine: no GPU, no network, order-recording."""

    def __init__(self, log_calls: list[str], *, diarize_fails: bool = False):
        self.calls = log_calls
        self.diarize_fails = diarize_fails
        self.model_refs: dict[str, Any] = {}
        self.residency_at_load: dict[str, bool] = {}
        self.align_residency_at_load: dict[str, bool] = {}

    def load_asr(self, model_name, compute_type):
        self.calls.append("load_asr")
        obj = type("ASR", (), {})()
        self.model_refs["asr"] = weakref.ref(obj)
        return obj

    def transcribe(self, asr, media_path, *, batch_size, language):
        self.calls.append(f"transcribe(batch={batch_size})")
        # Hardcoded 8, NOT MAX_BATCH_SIZE. Comparing a runtime value against
        # the very constant under test passes trivially when that constant is
        # neutralized — the fourth occurrence of that pattern in this project.
        assert batch_size <= 8, "batch cap violated"
        return {"language": "en", "segments": [
            {"start": 1.0, "end": 3.0, "text": "Hello there."}]}

    def load_align(self, language):
        self.calls.append("load_align")
        self.align_residency_at_load = {
            n: r() is not None for n, r in self.model_refs.items()}
        obj = type("ALIGN", (), {})()
        self.model_refs["align"] = weakref.ref(obj)
        # Model AND metadata in ONE handle, mirroring WhisperXEngine: the
        # weakref then proves BOTH die together. Retaining metadata on the
        # engine previously pinned 369 MB of alignment weights per chunk.
        return (obj, {"language": language})

    def align(self, aligner, segments, media_path):
        self.calls.append("align")
        return {"segments": [{
            "start": 1.0, "end": 3.0, "text": "Hello there.",
            "words": [{"word": "Hello", "start": 1.0, "end": 1.5, "score": 0.9},
                      {"word": "there.", "start": 1.6, "end": 3.0, "score": 0.8}],
        }]}

    def load_diarizer(self, hf_token):
        self.calls.append("load_diarizer")
        # Sample residency at the instant a new model would land on the card
        # — the only moment that can prove "never two resident". A mutant
        # deferring all teardown to end-of-stage passed the whole gate while
        # still emitting the right unload labels.
        self.residency_at_load = {
            n: r() is not None for n, r in self.model_refs.items()}
        if self.diarize_fails:
            raise RuntimeError("gated model: 403 (accept the pyannote terms)")
        obj = type("DIAR", (), {})()
        self.model_refs["diar"] = weakref.ref(obj)
        return obj

    def diarize(self, diarizer, media_path):
        self.calls.append("diarize")
        return "DIARIZATION"

    def assign_speakers(self, diarization, aligned):
        self.calls.append("assign_speakers")
        for seg in aligned["segments"]:
            seg["speaker"] = "SPEAKER_00"
            for w in seg["words"]:
                w["speaker"] = "SPEAKER_00"
        return aligned

    def turns_of(self, diarization):
        return [("SPEAKER_00", 1.0, 3.0)]


def _run_s1(tmp: Path, *, abs_offset: float = 0.0,
            diarize_fails: bool = False) -> tuple[TranscriptArtifact, list[str], _FakeEngine]:
    db = StateDB(tmp / "s1.db")
    try:
        media = tmp / "chunk.mp4"
        media.write_bytes(b"\x00" * 1024)
        calls: list[str] = []
        engine = _FakeEngine(calls, diarize_fails=diarize_fails)
        stage = S1Transcribe(db=db, artifacts_dir=tmp / "artifacts",
                             engine_factory=lambda: engine)
        art = stage.run(input_digest=digest_bytes(b"m"),
                        params={"batch_size": 99, "abs_offset_s": abs_offset},
                        media_path=media)
        return art, calls, engine, stage  # type: ignore[return-value]
    finally:
        db.close()


# ------------------------------------------------------------ 1. determinism


@check("S2 is byte-deterministic")
def _s2_determinism() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        transcript = _conversation_transcript()
        outputs = []
        for run in ("a", "b"):
            db = StateDB(tmp / f"{run}.db")
            try:
                stage = S2Prefilter(db, tmp / run)
                art = stage.run(input_digest=digest_bytes(b"t"), params={},
                                transcript=transcript)
                outputs.append(art.to_json_bytes())
            finally:
                db.close()
        assert outputs[0] == outputs[1], "same transcript, different bytes"


# ------------------------------------------------------------ 2. window law


@check("S2 windows obey 30-60 s, NMS, and top-K; scores persisted")
def _s2_window_law() -> None:
    from clipforge.stages.s2_prefilter import iou

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = StateDB(tmp / "s.db")
        try:
            stage = S2Prefilter(db, tmp)
            art = stage.run(input_digest=digest_bytes(b"t"),
                            params={"top_k": 5, "nms_iou": 0.4},
                            transcript=_conversation_transcript())
            assert art.candidates, "no candidates from 90 s of conversation"
            assert len(art.candidates) <= 5
            for c in art.candidates:
                dur = c.end - c.start
                assert 30.0 <= dur <= 60.0, f"window {dur:.1f}s out of law"
                for comp in ("boundary", "qa", "turns", "energy",
                             "laughter", "selfcont", "total"):
                    assert comp in c.scores, f"score {comp} not persisted"
            for i, a in enumerate(art.candidates):
                for b in art.candidates[i + 1:]:
                    assert iou((a.start, a.end), (b.start, b.end)) <= 0.4, \
                        "NMS ceiling violated"
        finally:
            db.close()


# ----------------------------------------------------- 3. heuristic direction


@check("S2 heuristics rank Q&A conversation above dead air")
def _s2_direction() -> None:
    from clipforge.stages.s2_prefilter import (DEFAULT_WEIGHTS,
                                               score_window, split_sentences)

    conv = split_sentences(_conversation_transcript().segments)
    dead_art = TranscriptArtifact(
        cache_key="t-dead", source_path="dead.mp4",
        segments=[_seg(0.0, 45.0, "Hmm.", "SPEAKER_00")], turns=[])
    dead = split_sentences(dead_art.segments)

    conv_score = score_window(conv[:8], conv[7].end - conv[0].start,
                              DEFAULT_WEIGHTS)["total"]
    dead_score = score_window(dead, 45.0, DEFAULT_WEIGHTS)["total"]
    assert conv_score > dead_score, (conv_score, dead_score)


# ------------------------------------------------------------- 4. S1 mechanics


@check("S1: per-phase teardown interleaves with loads; refs DEAD; batch cap")
def _s1_mechanics() -> None:
    with tempfile.TemporaryDirectory() as td:
        art, calls, engine, stage = _run_s1(Path(td), abs_offset=900.0)

        # Batch cap: the fake asserts batch<=8 inside transcribe.
        assert any(c.startswith("transcribe(batch=8") for c in calls), calls
        # Each model is released in its OWN phase, before the next loads —
        # stronger than spec §S1's teardown order, which presumed all three
        # were resident at the end.
        assert stage._last_unload_order == ["asr", "align", "diar"], \
            stage._last_unload_order
        loads = [c.split("(")[0] for c in calls if c.startswith("load_")]
        assert loads == ["load_asr", "load_align", "load_diarizer"], loads
        # The references are actually DEAD - not merely popped:
        for name, ref in engine.model_refs.items():
            assert ref() is None, f"model {name!r} survived hard_unload"
        # Absolute offset applied everywhere:
        seg = art.segments[0]
        assert seg.start == 901.0 and seg.end == 903.0
        assert seg.words[0].start == 901.0
        assert art.turns[0].start == 901.0
        assert art.abs_offset_s == 900.0


@check("never two models resident (sampled at each load, not from labels)")
def _s1_never_two_resident() -> None:
    with tempfile.TemporaryDirectory() as td:
        _art, _calls, engine, _stage = _run_s1(Path(td))
        assert engine.align_residency_at_load.get("asr") is False, (
            "ASR still resident when alignment loaded: "
            f"{engine.align_residency_at_load}")
        live = [n for n, alive in engine.residency_at_load.items() if alive]
        assert not live, f"{live} still resident when the diarizer loaded"


@check("the two VRAM-Law layers COMPOSE (orchestrator + in-stage session)")
def _law_layers_compose() -> None:
    """§6 dispatches GPU stages under GPULock while the stage itself opens a
    gpu_session. Both consult one registry, so the same class is registered
    twice — which used to clear residency on the inner exit and then raise
    CoResidencyError from the OUTER finally, on every run, masking any real
    stage exception behind it."""
    import asyncio

    from clipforge.gpu import GB, GPULock, gpu_session

    async def scenario() -> None:
        lock = GPULock(vram_probe=lambda: 24 * GB)
        async with lock.acquire(ModelClass.ASR, budget_gb=8.0):
            assert lock.registry.resident is ModelClass.ASR
            with gpu_session(ModelClass.ASR, 8.0, registry=lock.registry):
                assert lock.registry.resident is ModelClass.ASR
                # A different class is STILL refused while nested.
                try:
                    lock.registry.register(ModelClass.VL)
                    raise AssertionError("VL co-load must raise")
                except CoResidencyError:
                    pass
            # Inner exit must NOT clear residency the outer layer still owns.
            assert lock.registry.resident is ModelClass.ASR, (
                "inner gpu_session released the outer layer's residency")
        assert lock.registry.resident is None, "residency leaked"

    asyncio.run(scenario())


@check("§S1 observability: stage peak logged, then peak stats reset")
def _s1_peak_logging() -> None:
    """Spec §S1 mandates logging max_memory_allocated() then resetting.
    Deleting both was previously invisible to the entire gate."""
    import clipforge.gpu as gpu_mod
    from clipforge.gpu import gpu_session

    calls: list[str] = []

    class _FakeCuda:
        @staticmethod
        def max_memory_allocated() -> int:
            calls.append("max_memory_allocated")
            return 1234

        @staticmethod
        def reset_peak_memory_stats() -> None:
            calls.append("reset_peak_memory_stats")

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

        @staticmethod
        def mem_get_info(_i: int = 0):
            return (24 * 1024 ** 3, 24 * 1024 ** 3)

    fake_torch = type("T", (), {"cuda": _FakeCuda})
    original = gpu_mod._torch
    gpu_mod._torch = lambda: fake_torch  # type: ignore[assignment]
    try:
        with gpu_session(ModelClass.POSE, 1.0):
            pass
    finally:
        gpu_mod._torch = original  # type: ignore[assignment]

    assert "max_memory_allocated" in calls, "stage peak was never read/logged"
    assert calls.count("reset_peak_memory_stats") >= 2, (
        f"peak stats not reset before AND after the window: {calls}")
    assert "empty_cache" in calls, "cache never flushed on exit"


@check("spec constants are pinned (they were all silently editable)")
def _spec_constants() -> None:
    """Each of these was revert-safe: changing it left the whole gate green
    while violating a literal spec requirement."""
    from clipforge.config import AppConfig
    from clipforge.stages.s1_transcribe import MAX_BATCH_SIZE

    cfg = AppConfig()
    assert MAX_BATCH_SIZE == 8, "§S1: batch_size=8 is a HARD cap"
    assert cfg.s1.batch_size == 8, cfg.s1.batch_size
    assert cfg.s1.compute_type == "float16", "§S1: compute_type float16"
    assert cfg.s2.nms_iou == 0.4, "§S2: NMS on overlap IoU > 0.4"
    assert cfg.s2.window_min_s == 30.0 and cfg.s2.window_max_s == 60.0, \
        "§S2: slide 30-60 s windows"
    assert cfg.s2.top_k == 10, "§S2: take the top 10"
    assert cfg.s3.frames_per_candidate in (6, 7, 8), \
        "§S3: strictly 6-8 frames per candidate"


# ---------------------------------------------------------------- 5. VRAM law


@check("S1 runs under ASR residency; VL co-load during the window raises")
def _s1_vram_law() -> None:
    observed: dict[str, Any] = {}

    class SpyEngine(_FakeEngine):
        def load_asr(self, model_name, compute_type):
            observed["resident"] = GPU_LOCK.registry.resident
            try:
                GPU_LOCK.registry.register(ModelClass.VL)
                observed["vl_raised"] = False
            except CoResidencyError:
                observed["vl_raised"] = True
            return super().load_asr(model_name, compute_type)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = StateDB(tmp / "s.db")
        try:
            media = tmp / "c.mp4"
            media.write_bytes(b"\x00" * 128)
            stage = S1Transcribe(db=db, artifacts_dir=tmp,
                                 engine_factory=lambda: SpyEngine([]))
            stage.run(input_digest=digest_bytes(b"m"), params={},
                      media_path=media)
        finally:
            db.close()
    assert observed["resident"] is ModelClass.ASR
    assert observed["vl_raised"] is True
    assert GPU_LOCK.registry.resident is None, "residency leaked"


# --------------------------------------------------------- 6. diarize degrade


@check("diarization failure degrades; transcript survives")
def _s1_diarize_degrade() -> None:
    with tempfile.TemporaryDirectory() as td:
        art, _calls, _engine, stage = _run_s1(Path(td), diarize_fails=True)
        assert art.diarization_ok is False
        assert art.segments and art.segments[0].text == "Hello there."
        assert art.turns == []
        assert all(w.speaker is None
                   for s in art.segments for w in s.words)
        # The diarization phase's teardown runs EVEN THOUGH load_diarizer
        # raised. "It never loaded, so there is nothing to free" is wrong for
        # pyannote: a pipeline that dies part-way through construction has
        # already allocated, and that failure path — not the success path —
        # is the documented Windows leak shape.
        assert stage._last_unload_order == ["asr", "align", "diar"]


# --------------------------------------------------------- 7. S3 & S4 AI gates


@check("S3 REFUSES to rank without the video rather than degrading silently")
def _s3_fallback() -> None:
    from clipforge.schemas.candidates import CandidatesArtifact, CandidateWindow
    from clipforge.stages.s3_semantic import S3SemanticRanker

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = StateDB(tmp / "s3.db")
        try:
            cand = CandidateWindow(
                start=0.0, end=30.0, text="Sample text",
                total_score=8.5, scores={"total": 8.5}
            )

            cands_art = CandidatesArtifact(
                cache_key="cands_key", source_transcript="trans_key",
                candidates=[cand]
            )
            stage = S3SemanticRanker(db, tmp / "artifacts")
            # video_path=None used to yield a heuristic ranking and report
            # success. That meant the VL stage could be inert — clips
            # ordered by keyword counting — while every gate stayed green.
            # Refusing loudly is now the contract.
            try:
                stage.run(input_digest="test_digest", params={},
                          candidates_artifact=cands_art, video_path=None)
            except StageError as exc:
                assert "fallback disabled" in str(exc), str(exc)
            else:
                raise AssertionError(
                    "S3 ranked without the video; the silent heuristic "
                    "fallback is back")
        finally:
            db.close()


@check("S4 REFUSES to frame footage with no subject rather than centre-cropping")
def _s4_camera_path() -> None:
    from clipforge.schemas.ranking import RankedArtifact, RankedItem
    from clipforge.stages.s4_tracking import S4Tracking

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = StateDB(tmp / "s4.db")
        try:
            item = RankedItem(candidate_index=0, rank=1, hook_strength=8.0, justification="Good hook")
            ranked = RankedArtifact(
                cache_key="ranked_key", source_candidates="cands_key",
                ranking_source="heuristic", items=[item]
            )
            # A REAL source. video_path=None used to yield a camera path
            # anyway, because S4 defaulted to 1920x1080 when it could not
            # probe — so this check validated even coordinates against
            # FABRICATED geometry, and the crops it blessed fell outside the
            # actual 1280x720 fixture, making every S6 render emit 0 frames.
            fixture = _repo_root() / "tests" / "fixtures" / "sample_90s.mp4"
            assert fixture.exists(), f"missing fixture: {fixture}"
            stage = S4Tracking(db, tmp / "artifacts")
            # This fixture is TTS dialogue over a static card — no person to
            # track. S4 used to answer that with a centre crop labelled
            # framing_mode="speaker", so the active-speaker reframing that
            # defines this product could be dead while the gate read green.
            try:
                stage.run(input_digest="test_digest", params={},
                          ranked_artifact=ranked, video_path=fixture,
                          start_s=0.0, end_s=5.0)
            except StageError as exc:
                assert "fallback disabled" in str(exc), str(exc)
            else:
                raise AssertionError(
                    "S4 produced a camera path for footage with no subject; "
                    "the silent centre-crop fallback is back")
        finally:
            db.close()


@check("S4 crop geometry is even and inside the frame (pure-function level)")
def _s4_crop_geometry() -> None:
    """The invariants the old S4 check asserted, at a level that survives
    the stage refusing to run.

    They used to ride on a full S4 run against a subject-less fixture,
    which now (correctly) raises — so the geometry rules moved onto the
    helper that actually computes them. Even coordinates that fall outside
    the source frame are still unrenderable, so both halves matter.
    """
    from clipforge.stages.s6_render import _remap_path_frames
    from clipforge.pacing import TimeMap
    from clipforge.schemas.campath import CropFrame

    def _even(v: int) -> int:
        return v - (v % 2)

    src_w, src_h = 1280, 720
    frames = []
    for i in range(30):
        # Sized so the 16:9 height fits inside 720 — a taller crop would be
        # clamped and this would be testing the clamp, not the invariant.
        w = _even(360 - (i % 3) * 4)
        h = _even(min(src_h, int(w * 16 / 9)))
        # S4 emits EVEN coordinates; feeding odd ones here would fail on
        # the fixture rather than on the code under test.
        frames.append(CropFrame(frame=i, x=_even((src_w - w) // 2),
                                y=_even((src_h - h) // 2), w=w, h=h))
    out = _remap_path_frames(frames, TimeMap([(0.0, 1.0)]), 30.0, src_w, src_h)
    assert out, "no frames survived remapping"
    for f in out:
        assert f.x % 2 == 0 and f.y % 2 == 0, (f.x, f.y)
        assert f.w % 2 == 0 and f.h % 2 == 0, (f.w, f.h)
        assert f.x + f.w <= src_w, (f.x, f.w, src_w)
        assert f.y + f.h <= src_h, (f.y, f.h, src_h)


# ---------------------------------------------------------------- runner



def main() -> int:
    failures = 0
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failures += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    total = len(_CHECKS)
    print(f"ai verify: {total - failures}/{total} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
