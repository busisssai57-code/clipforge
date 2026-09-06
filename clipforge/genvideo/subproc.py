"""A provider whose model lives in another interpreter.

LTX-2.5 needs diffusers 0.40 and transformers 5.x; diffusers 0.40
requires ``huggingface-hub>=1.23`` and whisperx — S1 — pins it below 1.0.
The two cannot share an environment, so this provider talks to
``_ltx_worker.py`` running under ``.venv-ltx25`` over a JSON line
protocol.

Three things this file is careful about:

**The VRAM Law crosses the process boundary.** A child holding 13 GB of
card is invisible to the residency registry, so the worker's whole
lifetime sits inside one ``gpu_session``, which takes the same reentrant
lock and the same cross-PROCESS lock every stage takes. Nothing else can
co-load while a worker is alive, and the session is released in
``close()``.

**The worker outlives a single shot.** Loading is 81-103 s against
56-61 s per 25-frame generation, so a process per shot would nearly
double the cost of a brief. ``generate()`` starts the worker on first use
and keeps it; the router closes it when the sequence ends.

**The parent owns the container.** The child returns raw uint8 frames in
a ``.npy`` and never encodes anything: the blank-frame guard, the ffmpeg
flags and the ``.partial`` discipline live in ``providers.py``, and a
second implementation of them in an environment without the ffmpeg helper
is exactly the kind of drift this project keeps finding.
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import threading
import time
from collections import deque
from contextlib import ExitStack
from pathlib import Path
from typing import Any

import structlog

from clipforge.genvideo.models import ModelSpec
from clipforge.genvideo.providers import (GenResult, ProviderError,
                                          ProviderUnavailable,
                                          _generation_dims, _latent_frames,
                                          _write_video, delivery_dims)

log = structlog.get_logger(__name__)

WORKER = Path(__file__).with_name("_ltx_worker.py")

#: Seconds per frame x step x megapixel, from three measured runs:
#: 0.39 (30 steps, 704x1216), 0.42 (30 steps, 512x896) and 0.58 (8 steps,
#: 512x896 — fewer steps, so fixed per-call cost weighs more). The slow
#: end is used, because this feeds a TIMEOUT and being wrong in the
#: generous direction costs waiting while being wrong the other way
#: kills a good render.
SECONDS_PER_FRAME_STEP_MPX = 0.6
#: On top of the estimate. A timeout should fire when something is
#: broken, not when a machine is busy.
TIMEOUT_HEADROOM = 2.0
#: Floor, and what the first call adds for the load (81-120 s measured).
MIN_CALL_TIMEOUT_S = 600.0
LOAD_ALLOWANCE_S = 300.0
START_TIMEOUT_S = 120.0


def call_timeout_s(frames: int, width: int, height: int, steps: int,
                   *, first_call: bool) -> float:
    """How long this specific generation may take before it is a fault.

    Two fixed constants used to stand here — 900 s for the first call,
    600 s after — and they were sized against a schedule that no longer
    exists. Raising quality to 30 steps at the model's own envelope made
    one six-second shot a ~26-minute job, and the 900 s cap killed the
    worker at 15:30 with the render still going. The estimate is nothing
    clever; the point is that it MOVES with the work.
    """
    work = frames * max(steps, 1) * (width * height / 1e6)
    est = work * SECONDS_PER_FRAME_STEP_MPX
    if first_call:
        est += LOAD_ALLOWANCE_S
    return max(MIN_CALL_TIMEOUT_S, est * TIMEOUT_HEADROOM)


def interpreter_for(spec: ModelSpec,
                    *, repo_root: Path | None = None) -> Path | None:
    """The Python that can load this model, or None if it is this one.

    An env var wins so an operator can put the second environment
    anywhere; otherwise the spec's own relative path is resolved against
    the repo. Returns None for models with no separate interpreter, which
    is every model but this one.
    """
    if not spec.interpreter:
        return None
    override = os.environ.get("CLIPFORGE_ALT_PYTHON")
    if override:
        return Path(override)
    root = repo_root or Path(__file__).resolve().parents[2]
    return root / spec.interpreter


class SubprocessModelProvider:
    """Generation delegated to a model in a separate interpreter."""

    def __init__(self, spec: ModelSpec, *, seed: int = 1234,
                 interpreter: Path | None = None,
                 repo_root: Path | None = None,
                 quantize: str = "",
                 loras: list[str] | None = None,
                 step_cache_threshold: float = 0.0,
                 loop_strength: float = 0.0) -> None:
        self.spec = spec
        self.model_id = spec.model_id
        self.name = f"{spec.key}-subprocess"
        self.seed = seed
        self.quantize = quantize or spec.requires_quantization
        self.python = interpreter or interpreter_for(spec, repo_root=repo_root)
        self._proc: subprocess.Popen | None = None
        self._replies: queue.Queue = queue.Queue()
        self._stderr: deque = deque(maxlen=200)
        self._stack: ExitStack | None = None
        self._loaded = False
        self._last_request: dict | None = None
        # The worker has no knob for either of these. Saying so once, at
        # construction, is the difference between a customisation that
        # was declined and one that vanished: `LocalDiffusersProvider`
        # already logs `genvideo.lora_unsupported` for the same case, and
        # this path said nothing at all.
        if loras:
            log.warning("genvideo.lora_unsupported", model=spec.key,
                        requested=len(loras),
                        note="this worker does not implement LoRA loading, "
                             "so the configured weights were NOT used. NOT a "
                             "limit of the model: LTX2Pipeline carries "
                             "LTX2LoraLoaderMixin and load_lora_weights "
                             "works. Unimplemented, not impossible -- the "
                             "same wording that hid the step cache for "
                             "months until someone checked the pipeline")
        #: Wan2GP's TeaCache idea. This used to log
        #: `step_cache_unsupported` and drop the value on the floor, which
        #: was true of the worker as written and NOT true of the pipeline:
        #: `LTX2VideoTransformer3DModel` carries diffusers' CacheMixin, so
        #: the hook was there the whole time, one level below where
        #: `LocalDiffusersProvider` looks for it. It is sent with every
        #: request and the worker reports back what actually stuck.
        self.step_cache_threshold = float(step_cache_threshold or 0.0)
        #: Pin the LAST frame as well as the first. See Niche.loop_strength.
        self.loop_strength = float(loop_strength or 0.0)

    # ------------------------------------------------------------ status

    def available(self) -> bool:
        """Interpreter, worker and weights all present. Never downloads."""
        from clipforge.genvideo.models import weights_present  # noqa: PLC0415

        return bool(self.python and Path(self.python).is_file()
                    and WORKER.is_file() and weights_present(self.spec))

    #: Words that mean the prompt already says something about sound.
    #: Deliberately short: this only has to catch a writer who HAS given
    #: audio direction, so that the hint does not argue with them.
    _AUDIO_WORDS = ("sound", "audio", "music", "voice", "voices", "chatter",
                    "ambience", "ambient", "noise", "dialogue", "speaks",
                    "speaking", "silence", "silent", "quiet", "sings",
                    "singing", "score", "hum", "roar")

    def _prompt(self, prompt: str) -> str:
        """The shot prompt, with an audio cue if this model needs one.

        MEASURED on LTX-2.5, all at 2 s and one seed: the bare brief
        gives -18.7 LUFS, and the same brief plus the pipeline's 50-word
        cinematography block gives **-52.8** — the style that makes the
        picture look right silences the track. One plain sentence about
        the scene's sound restores it (-12.8) without touching the
        style, which is why this appends rather than trims.

        A prompt that already gives audio direction is left alone: the
        writer's "no dialogue, only wind" must not be argued with by a
        generic hint.
        """
        hint = getattr(self.spec, "audio_prompt_hint", "")
        if not (self.spec.generates_audio and hint):
            return prompt
        # WORD boundaries. Plain `in` matched "hum" inside "humble" and
        # "score" inside "scoreboard", so an ordinary brief could look
        # like it already carried audio direction and lose the cue that
        # is the measured difference between -52.8 and -12.8 LUFS.
        if set(re.findall(r"[a-z']+", prompt.lower())) & set(self._AUDIO_WORDS):
            return prompt
        log.info("genvideo.audio_hint_added", model=self.spec.key)
        return prompt.rstrip().rstrip(".") + ". " + hint

    def _negative(self, operator_negative: str) -> str:
        """The preset's avoid list, plus what this MODEL gets wrong.

        Kept apart on purpose: the preset's `avoid` is an editorial
        choice ("no cartoon, no watermark") and belongs to the operator,
        while "blurry, jittery, distorted" is a property of this
        checkpoint that diffusers' own LTX2 example negates by default.
        Appended rather than replaced, and only when not already there.
        """
        extra = [t.strip() for t in self.spec.default_negative.split(",")
                 if t.strip() and t.strip().lower() not in
                 operator_negative.lower()]
        if not extra:
            return operator_negative
        return ", ".join(([operator_negative] if operator_negative else [])
                         + extra)

    def supports_start_image(self) -> bool:
        # True since 2026-09-05: the worker builds an
        # LTX2ImageToVideoPipeline from the already-loaded components with
        # `from_pipe` (no second 13 GB residency) and runs a shot that has
        # a start frame through it.
        #
        # This returned False for as long as the worker was text-to-video
        # only, and the comment here explained that saying so was "what
        # makes the router stop threading last frames through". The router
        # was not asking -- it introspected the signature, which declares
        # `start_image` to satisfy the interface -- so it threaded them
        # anyway and generate() dropped them. Both halves are fixed; this
        # one has to stay honest, because the router now believes it.
        return True

    # ------------------------------------------------------------ worker

    @staticmethod
    def _drain(stream: Any, sink: Any) -> None:
        """Pump one of the child's pipes so it can never block on a full one."""
        for line in iter(stream.readline, ""):
            sink(line.rstrip("\n"))
        stream.close()

    def _start(self) -> None:
        if self._proc is not None:
            return
        if not self.available():
            raise ProviderUnavailable(
                f"{self.spec.label} needs its own interpreter. Expected "
                f"{self.python}; set CLIPFORGE_ALT_PYTHON to point "
                "elsewhere. It exists because diffusers 0.40 (this model) "
                "and whisperx (S1) disagree about huggingface-hub, so the "
                "pipeline venv cannot load this checkpoint at all.")
        from clipforge.gpu import ModelClass, gpu_session  # noqa: PLC0415

        stack = ExitStack()
        # The child's VRAM is invisible to the registry, so the session is
        # opened HERE and held for as long as the worker lives.
        stack.enter_context(gpu_session(ModelClass.VL,
                                        budget_gb=self.spec.vram_gb))
        try:
            proc = subprocess.Popen(  # noqa: S603 - our file, our interpreter
                [str(self.python), str(WORKER)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
                cwd=str(WORKER.parent))
        except OSError as exc:
            stack.close()
            raise ProviderUnavailable(
                f"could not start {self.python}: {exc}") from exc
        self._stack, self._proc = stack, proc
        for stream, sink in ((proc.stdout, self._replies.put),
                             (proc.stderr, self._stderr.append)):
            threading.Thread(target=self._drain, args=(stream, sink),
                             daemon=True).start()
        log.info("genvideo.worker_started", model=self.spec.key,
                 python=str(self.python), pid=proc.pid)
        hello = self._call({"op": "ping"}, timeout=START_TIMEOUT_S)
        if not hello.get("ok"):
            self.close()
            raise ProviderUnavailable(f"worker did not answer ping: {hello}")

    def _call(self, request: dict, *, timeout: float) -> dict:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise ProviderError("worker is not running")
        try:
            proc.stdin.write(json.dumps(request) + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._died(f"could not send {request.get('op')}: {exc}")
        # Waited in slices rather than one long block: a child that dies
        # never puts anything in the queue, so a single `get(timeout=120)`
        # spends the whole timeout learning nothing. Measured while
        # writing these tests — a worker that exited at once still took
        # 126 s to report, and the report said "did not answer" when the
        # truth was "died with a syntax error", which was in its stderr
        # the whole time.
        deadline = time.monotonic() + timeout
        while True:
            # Checked FIRST: a worker flooding the pipe with lines this
            # loop skips (noise, blanks) could otherwise hold the call —
            # and the GPU session behind it — past its budget, because the
            # deadline was only tested when the queue went quiet.
            if time.monotonic() >= deadline:
                self.close()
                raise ProviderError(
                    f"{self.spec.label} worker did not answer "
                    f"{request.get('op')!r} within {timeout:.0f}s")
            try:
                line = self._replies.get(timeout=min(2.0, timeout))
            except queue.Empty:
                if proc.poll() is not None:
                    raise self._died("it exited before answering "
                                     f"{request.get('op')!r}")
                continue
            if not line.strip():
                continue
            if not line.lstrip().startswith("{"):
                # The worker redirects library output to stderr precisely
                # so this cannot happen; if it does, say what arrived.
                log.warning("genvideo.worker_noise", line=line[:200])
                continue
            return json.loads(line)

    def _died(self, why: str) -> ProviderError:
        """A dead worker, reported with the evidence the parent can see."""
        code = self._proc.poll() if self._proc else None
        tail = " | ".join(list(self._stderr)[-6:])
        self.close()
        # 3221225477 (-1073741819) is 0xC0000005, the access violation
        # this model has taken once in thirteen loads, when host commit
        # is tight. Naming it here saves the next reader the search.
        return ProviderError(
            f"{self.spec.label} worker died ({why}); exit={code}. "
            f"last stderr: {tail[:800]}")

    def close(self) -> None:
        """Stop the worker and give the card back. Safe to call twice."""
        proc, stack = self._proc, self._stack
        self._proc, self._stack = None, None
        if proc is not None:
            try:
                if proc.poll() is None and proc.stdin is not None:
                    proc.stdin.write(json.dumps({"op": "quit"}) + "\n")
                    proc.stdin.flush()
                proc.wait(timeout=30)
            except Exception:  # noqa: BLE001 - a stuck worker still gets killed
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception as exc:  # noqa: BLE001
                    log.warning("genvideo.worker_unkillable",
                                model=self.spec.key, pid=proc.pid,
                                error=f"{type(exc).__name__}: {exc}"[:200])
            # None when even the kill did not settle. "exit=None" hides the
            # one field that says whether this was the access violation
            # (3221225477) or a clean stop.
            code = proc.returncode
            log.info("genvideo.worker_stopped", model=self.spec.key,
                     exit=code if code is not None else "still terminating")
        if stack is not None:
            try:
                stack.close()
            except Exception as exc:  # noqa: BLE001 - releasing must not raise
                log.warning("genvideo.session_close_failed",
                            model=self.spec.key,
                            error=f"{type(exc).__name__}: {exc}"[:200])
        self._loaded = False

    def __enter__(self) -> "SubprocessModelProvider":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------- generate

    def generate(self, *, prompt: str, seconds: float, fps: int,
                 out_path: Path, negative: str = "",
                 aspect_ratio: str = "9:16",
                 start_image: Path | None = None) -> GenResult:
        self._start()
        # THIS model's envelope, not the global cap. `MAX_GEN_PIXELS` is
        # 460k because LTX-Video 0.9 returned blank frames above it —
        # measured, but measured on a different model. Applying it here
        # generated 512x896 and scaled it 2.1x to 1080x1920, which is
        # most of what "blurry" was.
        width, height = _generation_dims(aspect_ratio,
                                         budget=self.spec.max_pixels,
                                         multiple=self.spec.dim_multiple)
        # Legal BY CONSTRUCTION, which is why there is no fallback here:
        # the budget is the spec's own `max_pixels` and the grid is its
        # own `dim_multiple`, and `_generation_dims` searches at or below
        # the budget, so `spec.supports(width, height)` cannot fail.
        # Selection is the looser half — it scores every model against the
        # size the GLOBAL cap produces — and that is a conservative floor:
        # a model approved for 512x896 rendering at its own larger
        # envelope is the intended behaviour, not drift.
        # `test_every_registry_model_generates_inside_its_own_envelope`
        # pins the invariant for every model and aspect.
        deliver = delivery_dims(aspect_ratio)
        frames = _latent_frames(seconds, fps, group=self.spec.frame_group)
        npy = out_path.parent / f".{out_path.stem}.frames.npy"
        wav_npy = out_path.parent / f".{out_path.stem}.audio.npy"
        npy.parent.mkdir(parents=True, exist_ok=True)
        request = {"op": "generate", "model_id": self.model_id,
                   "quantize": self.quantize, "prompt": self._prompt(prompt),
                   "negative": self._negative(negative),
                   "width": width, "height": height,
                   "frames": frames, "fps": float(fps),
                   "steps": self.spec.steps,
                   "guidance": self.spec.guidance_scale,
                   "attention_backend": self.spec.attention_backend,
                   "stg_scale": self.spec.stg_scale,
                   "audio_stg_scale": self.spec.audio_stg_scale,
                   "modality_scale": self.spec.modality_scale,
                   "audio_modality_scale": self.spec.audio_modality_scale,
                   "seed": self.seed, "out": str(npy),
                   "step_cache": self.step_cache_threshold,
                   "start_image": str(start_image) if start_image else None,
                   "loop_strength": self.loop_strength,
                   "audio_out": str(wav_npy)}
        #: What was last asked of the worker. Kept because the size and
        #: schedule a shot was generated at are the first questions asked
        #: of a bad-looking piece, and they are otherwise only in a log
        #: line that has scrolled away.
        self._last_request = request
        log.info("genvideo.worker_call", model=self.spec.key, seed=self.seed,
                 width=width, height=height, frames=frames,
                 steps=self.spec.steps, guidance=self.spec.guidance_scale,
                 loaded=self._loaded,
                 budget_s=round(call_timeout_s(
                     frames, width, height, self.spec.steps,
                     first_call=not self._loaded)))
        # Derived from THIS shot's work, not a constant: see
        # `call_timeout_s`. The first call pays the load as well.
        timeout = call_timeout_s(frames, width, height, self.spec.steps,
                                 first_call=not self._loaded)
        try:
            reply = self._call(request, timeout=timeout)
            if not reply.get("ok"):
                # Release before raising. The router answers a
                # ProviderError by trying the NEXT provider, which loads
                # its own model on the same card — and this worker is
                # still holding 13 GB of it, invisible to everything but
                # the session this provider opened. A reload costs 120 s
                # if the operator retries; a fallback that cannot fit
                # costs the piece.
                self.close()
                raise ProviderError(
                    f"{self.spec.label} generation failed (worker released "
                    f"so the next provider can use the card): "
                    f"{reply.get('error', reply)}")
            self._loaded = True
            import numpy as np  # noqa: PLC0415

            track = None
            rate = int(reply.get("audio_sample_rate") or 0)
            if reply.get("audio_npy") and rate > 0 and wav_npy.is_file():
                track = (np.load(wav_npy), rate)
                # A track of the right length full of near-zeros is an
                # audio stream nobody can hear, and it passes every check
                # that only asks whether a stream exists. -60 dBFS is
                # inaudible against any picture; the three measured runs
                # came back at -6.4, -8.2 and -16.7.
                rms = float(reply.get("audio_rms") or 0.0)
                if rms < 0.001:
                    log.warning("genvideo.audio_silent", model=self.spec.key,
                                rms=rms, peak=reply.get("audio_peak"),
                                note="the model returned a silent track; the "
                                     "shot will have an audio stream and no "
                                     "sound")
            elif self.spec.generates_audio:
                # A model whose spec says it makes sound, returning none,
                # is a broken run rather than a silent one. Said out loud
                # because silence is exactly what nobody notices.
                log.warning("genvideo.audio_missing", model=self.spec.key,
                            note="the spec says this model generates audio "
                                 "and the worker returned none")
            try:
                _write_video(np.load(npy), out_path, fps, deliver=deliver,
                             audio=track)
            except Exception:
                # Same reason as the failed reply above, and it was
                # missing here: the router answers any ProviderError by
                # trying the NEXT provider, which loads its own model on
                # this card while this worker still holds ~13 GB of it.
                # A blank-frame rejection is exactly such a failure.
                self.close()
                raise
        finally:
            npy.unlink(missing_ok=True)
            wav_npy.unlink(missing_ok=True)
        # Requested vs applied as separate facts. A cache that quietly
        # declined to engage looks exactly like one that did, except in the
        # wall clock -- and this project has shipped that defect enough
        # times to spend a log line on it.
        applied = float(reply.get("step_cache_applied", 0.0) or 0.0)
        if self.step_cache_threshold and not applied:
            log.warning("genvideo.step_cache_unsupported", model=self.spec.key,
                        requested=self.step_cache_threshold,
                        note="the worker could not engage a step cache; "
                             "generation ran at full step count")
        elif applied:
            log.info("genvideo.step_cache", model=self.spec.key,
                     requested=self.step_cache_threshold, applied=applied)
        # Chained vs not, as a fact from the WORKER rather than from what
        # the caller hoped. A frame that was handed over and ignored is
        # exactly what produced a byte-identical "continuity" batch.
        if start_image and not reply.get("chained"):
            log.warning("genvideo.continuity_dropped", model=self.spec.key,
                        note="a start frame was supplied and the worker did "
                             "not chain from it; this shot is not continuous "
                             "with the one before it")
        log.info("genvideo.worker_done", model=self.spec.key,
                 step_cache=applied, chained=bool(reply.get("chained")),
                 load_s=reply.get("load_seconds"),
                 generate_s=reply.get("generate_seconds"),
                 audio_samples=reply.get("audio_samples", 0),
                 # The LEVEL is the part worth keeping in the log: a
                 # six-second piece shipped with a full-length track at
                 # -inf LUFS, and the sample count said nothing.
                 audio_rms=reply.get("audio_rms"),
                 audio_peak=reply.get("audio_peak"))
        return GenResult(out_path, self.name, seconds, prompt, self.model_id)
