"""LTX-2.5 generation in the interpreter that can actually load it.

**This file runs under `.venv-ltx25`, not the pipeline venv.** It may
import nothing from `clipforge` and nothing outside torch / diffusers /
numpy / the standard library, because that environment has none of it.

Why a second interpreter exists at all: the checkpoint needs diffusers
0.40 (for `LTX2Pipeline`) and transformers 5.x (for the Gemma4 text
encoder), diffusers 0.40 requires `huggingface-hub>=1.23`, and whisperx —
which is S1 — pins `huggingface-hub<1.0`. One environment cannot hold
both, so transcription and this model do not share a process. Measured
2026-08-20; see VERIFICATION.md.

Why a LONG-LIVED worker rather than one process per shot: the load is
81-103 s and a 25-frame generation is 56-61 s. Paying the load once per
brief instead of once per shot is the difference between ~57 and ~106
minutes for a 60-second piece.

Protocol — one JSON object per line, request on stdin, reply on stdout:

    {"op": "ping"}                                  -> {"ok": true, ...}
    {"op": "generate", "prompt": ..., "out": ...}   -> {"ok": true, "npy": ...}
    {"op": "quit"}                                  -> exits 0

Frames come back as a `.npy` of uint8 RGB, NOT as a video: the parent
owns encoding. Everything the DAG depends on — the blank-frame guard,
ffmpeg flags, the `.partial` discipline — lives in `providers.py` and
must not be reimplemented in an environment that has no ffmpeg helper.
"""

from __future__ import annotations

import contextlib
import json
import sys
import time
import traceback

#: The response channel. Claimed before any library can print to it:
#: diffusers, transformers and bitsandbytes all write progress, and one
#: stray stdout line would be parsed as a reply.
_REPLY = sys.stdout
sys.stdout = sys.stderr

_PIPE = None
#: What `_PIPE` actually is. The cache used to be keyed on nothing, so a
#: request naming a different model_id or quantize mode silently got the
#: first-loaded pipeline and reported success for frames the wrong
#: checkpoint made.
_PIPE_KEY: tuple[str, str] | None = None
_LOAD_SECONDS = 0.0


def _reply(payload: dict) -> None:
    _REPLY.write(json.dumps(payload) + "\n")
    _REPLY.flush()


def _load(model_id: str, quantize: str) -> object:
    """Build the pipeline once per (model, quantize), and keep it.

    Keyed, because the protocol carries both on every request: an
    unkeyed cache answers a second request for a different checkpoint
    with the first one's weights and calls it a success.
    """
    global _PIPE, _PIPE_KEY, _LOAD_SECONDS
    key = (model_id, quantize)
    if _PIPE is not None and _PIPE_KEY == key:
        return _PIPE
    if _PIPE is not None:
        # One model at a time: this process holds a card that the parent
        # has reserved for exactly one residency.
        _PIPE = None
        _PIPE_KEY = None
        import gc

        gc.collect()
    t0 = time.time()
    import torch
    from diffusers import BitsAndBytesConfig as DiffusersBnb
    from diffusers import DiffusionPipeline
    from diffusers.quantizers import PipelineQuantizationConfig
    from transformers import BitsAndBytesConfig as TransformersBnb

    load_kw = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if quantize == "nf4":
        # NOT the transformer alone. Measured 2026-08-20: the Gemma4 text
        # encoder is 23.92 GB at bf16 and the connectors 6.34 GB, so
        # quantizing only the transformer leaves ~30 GB of bf16
        # companions for a 24 GB card. All three: 9.51 / 7.50 / 1.59 GB.
        common = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                      bnb_4bit_compute_dtype=torch.bfloat16,
                      bnb_4bit_use_double_quant=True)
        load_kw["quantization_config"] = PipelineQuantizationConfig(
            quant_mapping={"transformer": DiffusersBnb(**common),
                           "text_encoder": TransformersBnb(**common),
                           "connectors": DiffusersBnb(**common)})
    pipe = DiffusionPipeline.from_pretrained(model_id, **load_kw)

    # Model offload, never sequential. `enable_sequential_cpu_offload()`
    # raises `Cannot copy out of meta tensor` on the first forward:
    # accelerate's per-submodule hooks cannot move a bitsandbytes 4-bit
    # parameter off meta. Measured on the first render attempt.
    pipe.enable_model_cpu_offload()
    vae = getattr(pipe, "vae", None)
    if vae is not None and hasattr(vae, "enable_tiling"):
        vae.enable_tiling()
    _LOAD_SECONDS = round(time.time() - t0, 1)
    _PIPE, _PIPE_KEY = pipe, key
    return pipe


def _apply_step_cache(pipe, threshold: float) -> float:
    """Set the transformer's step cache to *threshold*; return what stuck.

    Wan2GP's TeaCache idea, which this worker declined to implement for
    months while the parent logged `step_cache_unsupported`. The hook is
    real, it is just not on the PIPELINE: `LTX2Pipeline` has no CacheMixin,
    while `LTX2VideoTransformer3DModel` does, and diffusers 0.40 ships
    `FirstBlockCacheConfig(threshold=...)` -- a fixed threshold on a
    measured residual, no sampling, which is why §3.2 still holds for a
    given threshold even though CHANGING it changes output.

    Set on EVERY request rather than at load: the pipeline is cached across
    calls keyed on (model_id, quantize), so a threshold left on the
    transformer would silently outlive the run that asked for it -- the
    same unkeyed-cache defect the loader below already guards against.

    Returns the threshold actually in force, so the parent can log
    requested and applied as separate facts instead of assuming.
    """
    transformer = getattr(pipe, "transformer", None)
    if transformer is None or not hasattr(transformer, "enable_cache"):
        return 0.0
    try:
        transformer.disable_cache()
    except Exception:  # noqa: BLE001 - nothing was enabled; that is fine
        pass
    if threshold <= 0:
        return 0.0
    try:
        from diffusers import FirstBlockCacheConfig
    except ImportError:
        return 0.0
    try:
        transformer.enable_cache(FirstBlockCacheConfig(threshold=float(threshold)))
    except Exception as exc:  # noqa: BLE001 - a cache is never worth a failed render
        print(json.dumps({"log": "step_cache_failed", "error": str(exc)}),
              flush=True)
        return 0.0
    return float(threshold)


_I2V = None


@contextlib.contextmanager
def _attention(name: str):
    """Dispatch attention to *name*, or do nothing if it is unavailable.

    Wan2GP's headline speed lever. diffusers 0.40 exposes it as a context
    manager; `_native_*` backends need no extra package, `_sage_*` and
    `_flash_*` do. An unavailable backend must NOT fail the render -- a
    kernel is an optimisation, not a requirement -- so this degrades to
    torch's own choice and says which it used.
    """
    if not name:
        yield
        return
    # Enter the backend EXPLICITLY rather than wrapping the body in a
    # try/except around a `with`. The first version did the latter and
    # yielded a second time from the handler when the body raised, which
    # is illegal for a generator context manager -- every failure inside
    # the render surfaced as "generator didn't stop after throw()", and a
    # kernel that was supposed to be optional killed the shot. Setup
    # failures are tolerated here; failures INSIDE the render are not
    # ours to swallow and propagate untouched.
    cm = None
    try:
        from diffusers import attention_backend

        cm = attention_backend(name)
        cm.__enter__()
    except Exception as exc:  # noqa: BLE001 - a kernel is an optimisation
        print(json.dumps({"log": "attention_backend_failed",
                          "requested": name, "error": str(exc)[:160]}),
              flush=True)
        cm = None
    else:
        print(json.dumps({"log": "attention_backend", "applied": name}),
              flush=True)
    try:
        yield
    finally:
        if cm is not None:
            cm.__exit__(None, None, None)


def _i2v_pipe(pipe):
    """An image-to-video pipeline sharing *pipe*'s already-loaded weights.

    `from_pipe` reuses the components rather than loading a second copy:
    the transformer alone is ~13 GB on the card, and a second residency
    would not fit beside the first even if it were free.

    Cached at module level for the same reason `_PIPE` is -- rebuilding it
    per shot would re-run component wiring 33 times a batch -- and reset
    whenever the underlying pipe changes identity, so a reload of a
    different checkpoint cannot be answered with the previous one's i2v
    wrapper. That is the unkeyed-cache defect this file already carries a
    comment about; it applies twice now.
    """
    global _I2V
    if _I2V is not None and _I2V[0] is pipe:
        return _I2V[1]
    import torch
    from diffusers import LTX2ImageToVideoPipeline

    # `torch_dtype` is NOT optional here. from_pipe defaults to float32 and
    # re-casts every component it can: the 4-bit modules refuse ("conversion
    # to torch.float32 is not supported ... still in 4bit") but the VAE,
    # vocoder and audio VAE DO convert, so the transformer keeps running
    # bf16 while the VAE it feeds becomes fp32. MEASURED 2026-09-05: vae
    # bfloat16 before from_pipe, float32 after, and the shot died on
    # "Input type (struct c10::BFloat16) and bias type (float) should be
    # the same". Saying the dtype keeps the reused components as they were.
    built = LTX2ImageToVideoPipeline.from_pipe(pipe, torch_dtype=torch.bfloat16)
    _I2V = (pipe, built)
    return built


def _generate(req: dict) -> dict:
    import numpy as np
    import torch

    pipe = _load(req["model_id"], req.get("quantize", "nf4"))
    cache_applied = _apply_step_cache(pipe, float(req.get("step_cache", 0.0) or 0.0))
    t0 = time.time()
    # CPU generator: diffusers seeds the initial latents on the CPU, and
    # a device generator gives a different sample for the same integer.
    kw = dict(
        prompt=req["prompt"],
        negative_prompt=req.get("negative") or None,
        height=int(req["height"]), width=int(req["width"]),
        num_frames=int(req["frames"]), frame_rate=float(req["fps"]),
        num_inference_steps=int(req["steps"]),
        guidance_scale=float(req["guidance"]),
        # Each of these can add a whole transformer pass per step. Sent
        # explicitly rather than defaulted, because the pipeline's defaults
        # turn both extras on and nothing said so.
        stg_scale=float(req.get("stg_scale", 1.0)),
        audio_stg_scale=float(req.get("audio_stg_scale", 1.0)),
        modality_scale=float(req.get("modality_scale", 3.0)),
        audio_modality_scale=float(req.get("audio_modality_scale", 3.0)),
        generator=torch.Generator("cpu").manual_seed(int(req["seed"])),
        output_type="pil")
    # Wan2GP-style i2v chaining. The shot that has a start frame runs
    # through the image-to-video pipeline so the cut lands inside one
    # continuous scene; the first shot of a sequence, and any shot after a
    # failed one, has no frame and runs from text.
    backend = str(req.get("attention_backend") or "")
    start = req.get("start_image")
    chained = False
    if start:
        from PIL import Image

        with Image.open(start) as handle:
            seed_frame = handle.convert("RGB").copy()
        seed_frame = seed_frame.resize((int(req["width"]), int(req["height"])))
        active = _i2v_pipe(pipe)
        _apply_step_cache(active, float(req.get("step_cache", 0.0) or 0.0))
        # The i2v pipeline re-compresses the conditioning image through
        # H.264 by default, because the model was trained on compressed
        # video and a pristine frame is off-distribution. That needs PyAV.
        # Without it the pipeline raises and the SHOT dies -- measured
        # 2026-09-05, when a missing `av` turned the first chained shot of
        # a batch into "no generation provider could produce this shot".
        # A missing optional codec is not worth a failed render: fall back
        # to crf=0 (no re-compression) and say so, rather than losing the
        # beat entirely.
        try:
            import av  # noqa: F401
            with _attention(backend):
                result = active(image=seed_frame, **kw)
        except ImportError:
            print(json.dumps({"log": "i2v_no_pyav",
                              "note": "PyAV missing; conditioning frame is "
                                      "not re-compressed (image_crf=0)"}),
                  flush=True)
            with _attention(backend):
                result = active(image=seed_frame, image_crf=0, **kw)
        chained = True
    else:
        with _attention(backend):
            result = pipe(**kw)
    frames = result.frames[0]
    arr = np.stack([np.asarray(im.convert("RGB")) for im in frames])
    np.save(req["out"], arr)
    reply = {"ok": True, "step_cache_applied": cache_applied,
             "chained": chained,
             "npy": req["out"], "frames": int(arr.shape[0]),
             "height": int(arr.shape[1]), "width": int(arr.shape[2]),
             "load_seconds": _LOAD_SECONDS,
             "generate_seconds": round(time.time() - t0, 1),
             "audio_npy": None, "audio_sample_rate": 0}

    # This model generates SOUND with the picture — `audio_in_channels`
    # and a vocoder are in its own config — and the first version of this
    # worker returned `result.frames` and dropped it on the floor, which
    # is why the first pieces were silent. Measured: (2, 96480) float32
    # at 48 kHz for 49 frames, RMS 0.48.
    audio = getattr(result, "audio", None)
    audio_out = req.get("audio_out")
    if audio is not None and audio_out:
        track = audio[0]
        track = track.float().cpu().numpy() if hasattr(track, "float")             else np.asarray(track)
        np.save(audio_out, track)
        rate = getattr(getattr(pipe, "vocoder", None), "config", None)
        reply["audio_npy"] = audio_out
        reply["audio_sample_rate"] = int(
            getattr(rate, "output_sampling_rate", 0) or 0)
        reply["audio_samples"] = int(track.size)
        # LEVEL, not just length. A track of the right length full of
        # zeros is an audio stream nobody can hear, and it is exactly
        # what "there is an audio stream" tests accept. Reported from
        # the process that made it, before anything can touch it.
        flat = np.asarray(track, dtype="float32").reshape(-1)
        reply["audio_peak"] = round(float(np.abs(flat).max()), 5)
        reply["audio_rms"] = round(
            float(np.sqrt((flat ** 2).mean())), 5) if flat.size else 0.0
    return reply


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except ValueError as exc:
            _reply({"ok": False, "error": f"unparseable request: {exc}"})
            continue
        op = req.get("op")
        try:
            if op == "quit":
                return 0
            if op == "ping":
                _reply({"ok": True, "pid": __import__("os").getpid(),
                        "loaded": _PIPE is not None})
            elif op == "generate":
                _reply(_generate(req))
            else:
                _reply({"ok": False, "error": f"unknown op {op!r}"})
        except Exception as exc:  # noqa: BLE001 - a reply beats a traceback
            # The parent cannot see this process's stack. Send the type,
            # the message and the last frames, or it gets "worker failed".
            _reply({"ok": False,
                    "error": f"{type(exc).__name__}: {exc}"[:600],
                    "traceback": traceback.format_exc()[-1200:]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
