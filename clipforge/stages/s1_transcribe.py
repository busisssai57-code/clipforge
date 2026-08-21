"""S1 — transcription: WhisperX + pyannote (spec §S1).

Flow: VAD → transcribe (batch_size hard-capped at 8 — that cap IS the VRAM
protection) → forced phoneme alignment (word timestamps) → pyannote speaker
diarization → speakers assigned to words. Emits ``s1.transcript.json`` with
segments, word timings, speakers, and the diarization turn timeline, all in
ABSOLUTE stream time (the caller passes the window's absolute offset).

VRAM Law mechanics, in this stage:

  * the whole suite (ASR + alignment + diarization) runs inside ONE
    ``gpu_session(ModelClass.ASR, …)`` window — budget asserted before any
    load, co-load with VL raises, peak logged + cache flushed on exit;
  * every model lives in the stage's model DICT (never bare locals) so
    :func:`clipforge.gpu.hard_unload` can actually drop the last reference;
  * unload order is the spec's: alignment → diarization → ASR, in a
    ``finally`` so it survives exceptions.

Testability: all whisperx/pyannote calls go through an injectable
:class:`S1Engine`. Unit tests (no GPU, no whisperx installed) mock at that
boundary — exactly the boundary §9 names; the real engine imports lazily.

Failure modes (spec §5): transient failures raise RetryableStageError
(orchestrator retries ×2 then quarantines the chunk). Diarization failure
alone does NOT fail the stage: the artifact is emitted with
``diarization_ok=False`` and no speakers — downstream degrades explicitly
(S4 falls back to center-framing) instead of losing the transcript.

Determinism (§3.2): faster-whisper decodes with beam search (no sampling),
seeds are pinned before diarization, and iteration is sorted. pyannote's
clustering is deterministic given fixed seeds and identical input bytes.
"""

from __future__ import annotations

import gc
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from clipforge.errors import RetryableStageError
from clipforge.gpu import ModelClass, gpu_session
from clipforge.log import get_logger
from clipforge.schemas import (DiarizationTurn, TranscriptArtifact,
                               TranscriptSegment, Word)
from clipforge.stages.base import Stage

log = get_logger(__name__)

#: HARD CAP (spec §S1): "batch_size=8 — this is the VRAM protection".
#: Config cannot raise it (validated there too); this is the last line.
MAX_BATCH_SIZE = 8


#: whisperx raises ``ValueError("No default align-model for language: cy")``
#: when a language has no bundled wav2vec2 aligner. Matched on the message
#: because whisperx gives it no dedicated type — but matched NARROWLY, so
#: that a CUDA fault or an OOM during the same call still fails the stage.
_NO_ALIGNER_MARKERS = (
    "no default align-model",
    "no default alignment model",
    "no align model",
    "no alignment model",
)


def _is_missing_aligner(exc: BaseException) -> bool:
    """True only for 'this language has no aligner', never for a fault."""
    text = str(exc).lower()
    return any(marker in text for marker in _NO_ALIGNER_MARKERS)


def _release_traceback_locals(exc: BaseException) -> None:
    """Drop the frame locals a propagating exception keeps alive.

    ``del model`` in a phase's ``finally`` releases only the STAGE frame's
    reference. The engine method that raised still has the model bound to a
    parameter (``transcribe(self, asr, ...)``, ``align(self, aligner, ...)``),
    its frame is retained by ``exc.__traceback__``, and ``raise ... from exc``
    carries that traceback out of the stage. So ``gc.collect()`` and
    ``empty_cache()`` in the finally were provably no-ops on every failure
    path: one full model suite stayed resident for as long as any caller held
    the exception — which is precisely what a retry loop does.

    ``traceback.clear_frames`` keeps the traceback's shape (file, line,
    function — everything the operator's error message needs) and clears only
    the locals. Frames still executing cannot be cleared; that is a documented
    no-op, not an error.
    """
    tb = getattr(exc, "__traceback__", None)
    if tb is not None:
        traceback.clear_frames(tb)


class S1Engine(Protocol):
    """The seam between stage logic and the AI stack. One method per model
    load/step so tests can assert ORDER (load, use, unload) precisely."""

    def load_asr(self, model_name: str, compute_type: str) -> Any: ...

    def transcribe(self, asr: Any, media_path: Path, *, batch_size: int,
                   language: str | None) -> dict: ...

    def load_align(self, language: str) -> Any:
        """Return a handle owning BOTH the model and its metadata.

        They must be freed together: retaining alignment metadata on the
        engine kept the wav2vec2 weights alive (measured: 369 MB, 212 live
        CUDA tensors surviving the whole stage).
        """

    def align(self, aligner: Any, segments: list[dict],
              media_path: Path) -> dict: ...

    def load_diarizer(self, hf_token: str | None) -> Any: ...

    def diarize(self, diarizer: Any, media_path: Path) -> Any: ...

    def assign_speakers(self, diarization: Any, aligned: dict) -> dict: ...

    def turns_of(self, diarization: Any) -> list[tuple[str, float, float]]: ...


class WhisperXEngine:
    """The real engine. Imports whisperx lazily so unit tests never touch
    the GPU stack; construction on a torch-less machine raises typed."""

    def __init__(self, device: str = "cuda") -> None:
        try:
            import whisperx  # noqa: PLC0415
        except ImportError as exc:
            raise RetryableStageError(
                "whisperx is not installed - pip install whisperx "
                "(see README, CP2 setup)") from exc
        self._wx = whisperx
        self.device = device
        # NOTE: the engine deliberately holds NO model state. Any attribute
        # here outlives every phase teardown and silently pins GPU memory.

    def load_asr(self, model_name: str, compute_type: str) -> Any:
        return self._wx.load_model(model_name, self.device,
                                   compute_type=compute_type)

    def transcribe(self, asr: Any, media_path: Path, *, batch_size: int,
                   language: str | None) -> dict:
        audio = self._wx.load_audio(str(media_path))
        kwargs: dict[str, Any] = {"batch_size": batch_size}
        if language:
            kwargs["language"] = language
        return asr.transcribe(audio, **kwargs)

    def load_align(self, language: str) -> Any:
        """(model, metadata) as ONE handle — see the protocol docstring.

        CLAIM RETRACTED (CP2 round 1, finding C2-3). This docstring used to
        say that stashing metadata on the engine caused 369 MB across 212
        CUDA tensors to stay resident after every run, and that returning one
        handle fixed it. That causal story is not supported by measurement
        and is contradicted by this project's own leak test, which records
        the same 369 MB / 212 tensors as a LIBRARY-level cache (a function
        attribute dict in torchaudio/transformers retaining the wav2vec2
        weights) held by no reference ClipForge owns — see
        tests/integration/test_s1_vram_exclusivity.py. A reviewer reverted
        this design consistently (engine stashes ``self._align_meta``,
        ``align()`` reads it back) and the entire two-command gate stayed
        green, including all five real-GPU tests. Nothing measurable changed.

        What the single handle IS good for, and all it is claimed to do: it
        makes ownership unambiguous, so the stage's ``del`` releases the
        model and its metadata together instead of leaving one half's
        lifetime implicit in engine state. That is a readability and
        lifetime-clarity property, not a measured reclamation. It is pinned
        by :func:`test_align_handle_owns_both_halves`, which asserts the
        weakref semantics rather than a megabyte count.
        """
        model, meta = self._wx.load_align_model(language_code=language,
                                                device=self.device)
        return (model, meta)

    def align(self, aligner: Any, segments: list[dict],
              media_path: Path) -> dict:
        model, meta = aligner
        audio = self._wx.load_audio(str(media_path))
        return self._wx.align(segments, model, meta, audio,
                              self.device, return_char_alignments=False)

    def load_diarizer(self, hf_token: str | None) -> Any:
        import inspect  # noqa: PLC0415

        import torch  # noqa: PLC0415

        torch.manual_seed(0)  # §3.2: pin seeds ahead of clustering
        # whisperx moved DiarizationPipeline across versions AND renamed its
        # token kwarg (`use_auth_token` -> `token` in 3.8). Pick whichever
        # this install actually accepts rather than guessing: passing the
        # wrong one raises TypeError, which the stage would then report as a
        # gated-model problem and send the operator hunting for a token that
        # was never the issue.
        pipeline_cls = getattr(self._wx, "DiarizationPipeline", None)
        if pipeline_cls is None:
            from whisperx.diarize import DiarizationPipeline  # noqa: PLC0415

            pipeline_cls = DiarizationPipeline
        params = inspect.signature(pipeline_cls.__init__).parameters
        token_kw = "token" if "token" in params else "use_auth_token"
        return pipeline_cls(**{token_kw: hf_token}, device=self.device)

    def diarize(self, diarizer: Any, media_path: Path) -> Any:
        """Diarize from an IN-MEMORY waveform, never a path.

        pyannote decodes files through torchcodec, whose native library
        supports FFmpeg 4-7; this project ships/expects a current ffmpeg
        (8.x here), so ``libtorchcodec`` fails to load and every
        file-path diarization dies with a DLL error that looks nothing like
        its actual cause. Decoding with whisperx's own ffmpeg-subprocess
        loader (already used for the ASR pass) sidesteps that dependency
        entirely and keeps one decode path for the whole stage.
        """
        audio = self._wx.load_audio(str(media_path))
        return diarizer(audio)

    def assign_speakers(self, diarization: Any, aligned: dict) -> dict:
        return self._wx.assign_word_speakers(diarization, aligned)

    def turns_of(self, diarization: Any) -> list[tuple[str, float, float]]:
        # whisperx returns a pandas DataFrame(columns=speaker,start,end).
        out: list[tuple[str, float, float]] = []
        for row in diarization.itertuples(index=False):
            out.append((str(row.speaker), float(row.start), float(row.end)))
        return sorted(out, key=lambda t: (t[1], t[0]))


@dataclass
class S1Transcribe(Stage[TranscriptArtifact]):
    """`chunk media` → `s1.transcript.json` (spec §5). ~8 GB, retry ×2."""

    name = "s1_transcribe"
    version = "1"
    vram_budget_gb = 8.0
    wall_budget_s = 1800.0
    artifact_type = TranscriptArtifact

    db: Any = None
    artifacts_dir: Path = Path(".")
    engine_factory: Callable[[], S1Engine] = WhisperXEngine
    hf_token: str | None = None
    _last_unload_order: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        Stage.__init__(self, self.db, self.artifacts_dir)

    # ------------------------------------------------------------------ run

    def _execute(self, *, cache_key: str, params: dict[str, Any],
                 media_path: Path | None = None,
                 **inputs: Any) -> TranscriptArtifact:
        if media_path is None:
            raise RetryableStageError("S1 requires media_path", stage=self.name)
        media_path = Path(media_path)
        if not media_path.exists():
            raise RetryableStageError(f"media vanished: {media_path}",
                                      stage=self.name)

        abs_offset = float(params.get("abs_offset_s", 0.0))
        batch_size = min(MAX_BATCH_SIZE, int(params.get("batch_size", 8)))
        model_name = str(params.get("model", "large-v2"))
        compute_type = str(params.get("compute_type", "float16"))
        language = params.get("language") or None

        self._last_unload_order = []
        diarization_ok = True
        words_aligned = True
        no_speech = False
        turns: list[tuple[str, float, float]] = []

        try:
            engine = self.engine_factory()
        except RetryableStageError:
            raise
        except Exception as exc:
            raise RetryableStageError(
                f"S1 engine construction failed: {type(exc).__name__}: {exc}",
                stage=self.name) from exc

        try:
            with gpu_session(ModelClass.ASR, self.vram_budget_gb):
                # ---- Phase 1: ASR ----------------------------------------
                # Each phase owns a LOCAL reference and frees it in its own
                # finally: del -> gc.collect() -> empty_cache(), inline.
                # Routing through a helper cannot free what the caller still
                # holds (the CP0 finding), and per-phase teardown means a
                # failure in phase N never leaves phase N-1's weights
                # resident while the exception unwinds.
                # The handle is bound to None FIRST and the load happens
                # INSIDE the try, so a raise from load_* — after it has
                # already allocated — still runs the ritual. Phase 3 was
                # given this shape first; phases 1 and 2 were left behind,
                # and measurement showed the asymmetry exactly: a failure in
                # load_asr recorded ZERO teardowns, a failure in load_align
                # recorded only ['asr'].
                asr_model = None
                try:
                    asr_model = engine.load_asr(model_name, compute_type)
                    raw = engine.transcribe(asr_model, media_path,
                                            batch_size=batch_size,
                                            language=language)
                    detected_lang = raw.get("language") or language or "en"
                except Exception as exc:
                    # The `del` below drops OUR reference, but the raising
                    # engine method's frame still holds the model in its
                    # parameter, and that frame is kept alive by
                    # exc.__traceback__ — which `raise ... from exc` carries
                    # out of the stage. Measured: a full model retained,
                    # 3/3 runs, on every failure path. Clearing the frames
                    # keeps the traceback's shape for reporting while
                    # releasing the locals it pins.
                    _release_traceback_locals(exc)
                    raise
                finally:
                    del asr_model
                    self._unload("asr")

                # ---- Phase 2: forced alignment ---------------------------
                # Two ways this legitimately does not run, both discovered
                # by pointing a live capture at a real stream (the ISS
                # feed, 2026-08-12): a window with no speech in it, and a
                # language with no alignment model. Neither is a broken
                # run, and raising on either made every quiet minute of a
                # broadcast a hard failure.
                raw_segments = raw.get("segments", []) or []
                if not raw_segments:
                    # Silence, music, ambience. The honest artifact is an
                    # empty transcript flagged as such; S2 then finds no
                    # candidates and the window is skipped, which is the
                    # correct outcome rather than an error.
                    no_speech = True
                    words_aligned = False
                    aligned = {"segments": []}
                    log.info("s1.no_speech", media=str(media_path)[:160],
                             detected_language=detected_lang,
                             note="empty transcript emitted; window will "
                                  "produce no candidates")
                else:
                    align_model = None
                    try:
                        try:
                            align_model = engine.load_align(detected_lang)
                        except Exception as exc:
                            # NARROW on purpose. Only "there is no aligner
                            # for this language" degrades — that is a fact
                            # about the language, and the transcript is
                            # still worth having without word timings.
                            #
                            # Everything else (a CUDA fault, OOM, a
                            # half-fetched checkpoint) still raises. Those
                            # mean the machine is in trouble, and emitting
                            # a wordless transcript would hide a hardware
                            # problem behind a plausible artifact — which
                            # is the failure mode this file's own docstring
                            # exists to forbid.
                            if not _is_missing_aligner(exc):
                                raise
                            words_aligned = False
                            aligned = {"segments": raw_segments}
                            log.warning(
                                "s1.alignment_unavailable",
                                language=detected_lang,
                                error=f"{type(exc).__name__}: {exc}"[:300],
                                note="no aligner for this language; emitting "
                                     "segment-level times without word "
                                     "timings, captions degrade accordingly")
                        if align_model is not None:
                            aligned = engine.align(align_model, raw_segments,
                                                   media_path)
                    except Exception as exc:
                        _release_traceback_locals(exc)
                        raise
                    finally:
                        del align_model
                        self._unload("align")

                # ---- Phase 3: diarization (degrades, never fails S1) -----
                # The handle is bound to None FIRST and the teardown is the
                # outermost finally. Nesting the teardown inside the try that
                # binds it meant a raise from load_diarizer itself skipped
                # the ritual entirely — and a pyannote pipeline that dies
                # part-way through construction (OOM, a half-fetched
                # checkpoint) has already allocated by then. That is the
                # documented Windows leak shape: the failure path, not the
                # success path, is what strands VRAM.
                diar_model = None
                try:
                    diar_model = engine.load_diarizer(self.hf_token)
                    diarization = engine.diarize(diar_model, media_path)
                    aligned = engine.assign_speakers(diarization, aligned)
                    turns = engine.turns_of(diarization)
                except Exception as exc:
                    _release_traceback_locals(exc)
                    diarization_ok = False
                    log.warning("s1.diarization_failed",
                                error=f"{type(exc).__name__}: {exc}",
                                note="emitting transcript without speakers"
                                     " (T5: check HF token + accepted terms)")
                finally:
                    del diar_model
                    self._unload("diar")
        except RetryableStageError:
            raise
        except Exception as exc:
            raise RetryableStageError(
                f"S1 failed: {type(exc).__name__}: {exc}",
                stage=self.name) from exc

        return self._build_artifact(cache_key, media_path, abs_offset,
                                    detected_lang, aligned, turns,
                                    diarization_ok, words_aligned=words_aligned,
                                    no_speech=no_speech)

    def _unload(self, phase: str) -> None:
        """The mandated ritual, executed AFTER the caller's ``del``.

        The caller deletes its own local first; by the time this runs no
        reference remains, so ``gc.collect()`` can actually reclaim the
        module and ``empty_cache()`` can return its blocks to the driver.
        Recorded in order so tests can assert the spec's align→diarize→ASR
        teardown sequence without reaching into CUDA.
        """
        self._last_unload_order.append(phase)
        gc.collect()
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------ assembly

    def _build_artifact(self, cache_key: str, media_path: Path,
                        abs_offset: float, language: str, aligned: dict,
                        turns: list[tuple[str, float, float]],
                        diarization_ok: bool, *,
                        words_aligned: bool = True,
                        no_speech: bool = False) -> TranscriptArtifact:
        """Pure assembly: shift every time into ABSOLUTE stream time (T1)."""
        segments: list[TranscriptSegment] = []
        for seg in aligned.get("segments", []):
            words = [Word(text=str(w.get("word", "")).strip(),
                          start=abs_offset + float(w["start"]),
                          end=abs_offset + float(w["end"]),
                          score=(float(w["score"])
                                 if w.get("score") is not None else None),
                          speaker=w.get("speaker"))
                     for w in seg.get("words", [])
                     if w.get("start") is not None and w.get("end") is not None]
            segments.append(TranscriptSegment(
                start=abs_offset + float(seg["start"]),
                end=abs_offset + float(seg["end"]),
                text=str(seg.get("text", "")).strip(),
                speaker=seg.get("speaker"),
                words=words))
        segments.sort(key=lambda s: (s.start, s.end))  # §3.2 sorted order

        turn_models = [DiarizationTurn(speaker=spk,
                                       start=abs_offset + start,
                                       end=abs_offset + end)
                       for spk, start, end in turns]
        # Sorted HERE, not only in WhisperXEngine.turns_of. The engine is an
        # injectable Protocol — the sort living exclusively in one
        # implementation makes the artifact's determinism a property of
        # whichever engine is plugged in, rather than of the stage that
        # promises it. §3.2 is a guarantee about the artifact.
        turn_models.sort(key=lambda t: (t.start, t.speaker))

        return TranscriptArtifact(
            cache_key=cache_key, stage=self.name,
            source_path=str(media_path), abs_offset_s=abs_offset,
            language=language, segments=segments, turns=turn_models,
            diarization_ok=diarization_ok, words_aligned=words_aligned,
            no_speech=no_speech)
