"""One heavy checkpoint load, in a process of its own, reporting numbers.

Run as a script by ``test_quantization_real_weights.py``; never collected
(pytest collects ``test_*.py``). It exists because of what happened when
the loads ran in-process: two bf16 pipelines resident at once needed
~36 GB on a 32 GB machine, Windows paged, and a memory-mapped read inside
safetensors took an **access violation** — which does not fail a test, it
kills the interpreter. The gate exited 139 with 8 of 1267
tests reported and the rest never run, and a summary line was never printed, so a `tail`
of the output looked like a stack trace and nothing else.

A child process turns that back into a test result: a segfault here is an
exit code the parent can read and name.

Protocol: argv is ``<model_id> <mode>``; the last stdout line is JSON.
``{"skip": reason}`` means the weights or the hardware are absent — that
is a skip, not a failure. Anything else is measurements.
"""

from __future__ import annotations

import json
import sys


def _emit(payload: dict) -> None:
    print(json.dumps(payload))
    sys.stdout.flush()


def _peak_memory() -> dict:
    """High-water memory of THIS process, by the two numbers that differ.

    ``peak_wset`` counts memory-mapped checkpoint pages that Windows can
    drop on demand, so it runs far above what the load actually needs —
    24.6 GB for an 11.4 GB model. ``peak_pagefile`` is peak COMMIT, the
    number a machine can genuinely run out of, and it is what the caller
    guards on.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        return {"peak_wset_bytes": int(getattr(info, "peak_wset", 0)),
                "peak_commit_bytes": int(getattr(info, "peak_pagefile", 0))}
    except Exception:  # noqa: BLE001 - a missing number is not a failure
        return {"peak_wset_bytes": 0, "peak_commit_bytes": 0}


def main(model_id: str, mode: str) -> int:
    try:
        import torch
    except ImportError:
        _emit({"skip": "torch is not installed"})
        return 0
    if not torch.cuda.is_available():
        _emit({"skip": "CUDA is not available"})
        return 0

    from clipforge.genvideo.providers import LocalDiffusersProvider

    kw = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    try:
        if mode == "nf4":
            # The provider's own path, unaltered: the claim this side
            # makes is that the knob reaches bitsandbytes.
            from diffusers import DiffusionPipeline

            qcfg = LocalDiffusersProvider(
                model_id=model_id, quantize=mode)._quantization_config()
            assert qcfg is not None, "nf4 produced no quantization config"
            pipe = DiffusionPipeline.from_pretrained(
                model_id, quantization_config=qcfg, **kw)
            transformer, vae = pipe.transformer, pipe.vae
        else:
            # The bf16 side loads COMPONENTS, not the pipeline. Every
            # claim below is per-component, and the pipeline drags in an
            # 11 GB text encoder that no assertion here looks at — on
            # this machine that is the difference between a load that
            # fits in RAM and one that pages. `AutoModel` reads each
            # subfolder's own `_class_name`, so this is not a guess about
            # one diffusers version's module tree.
            from diffusers import AutoModel

            transformer = AutoModel.from_pretrained(
                model_id, subfolder="transformer", **kw)
            vae = AutoModel.from_pretrained(model_id, subfolder="vae", **kw)
    except Exception as exc:  # noqa: BLE001 - absence is a skip upstream
        _emit({"skip": f"{model_id} did not load: {type(exc).__name__}: "
                       f"{str(exc)[:200]}"})
        return 0

    _emit({
        # What the load actually cost this process, so the caller's RAM
        # guard can be checked against a measurement rather than trusted
        # as a model. Both numbers are Windows counters and come back
        # zero elsewhere; nothing asserts on them.
        **_peak_memory(),
        "transformer_bytes": transformer.get_memory_footprint(),
        "vae_bytes": vae.get_memory_footprint(),
        # A named path into the transformer is the one place a name is
        # right: the layer CLASS is the claim ("4-bit is a different
        # layer, not a dtype cast").
        "attn_linear_class": type(transformer.blocks[0].attn1.to_q).__name__,
        # Not a named path: the claim is about the whole component.
        "vae_4bit_types": sorted({type(m).__name__ for m in vae.modules()
                                  if type(m).__name__.endswith("4bit")}),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
