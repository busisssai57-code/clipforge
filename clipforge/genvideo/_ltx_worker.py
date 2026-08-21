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
_LOAD_SECONDS = 0.0


def _reply(payload: dict) -> None:
    _REPLY.write(json.dumps(payload) + "\n")
    _REPLY.flush()


def _load(model_id: str, quantize: str) -> object:
    """Build the pipeline once, and keep it for the life of the process."""
    global _PIPE, _LOAD_SECONDS
    if _PIPE is not None:
        return _PIPE
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
    _PIPE = pipe
    return pipe


def _generate(req: dict) -> dict:
    import numpy as np
    import torch

    pipe = _load(req["model_id"], req.get("quantize", "nf4"))
    t0 = time.time()
    # CPU generator: diffusers seeds the initial latents on the CPU, and
    # a device generator gives a different sample for the same integer.
    result = pipe(
        prompt=req["prompt"],
        negative_prompt=req.get("negative") or None,
        height=int(req["height"]), width=int(req["width"]),
        num_frames=int(req["frames"]), frame_rate=float(req["fps"]),
        num_inference_steps=int(req["steps"]),
        guidance_scale=float(req["guidance"]),
        generator=torch.Generator("cpu").manual_seed(int(req["seed"])),
        output_type="pil")
    frames = result.frames[0]
    arr = np.stack([np.asarray(im.convert("RGB")) for im in frames])
    np.save(req["out"], arr)
    reply = {"ok": True, "npy": req["out"], "frames": int(arr.shape[0]),
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
