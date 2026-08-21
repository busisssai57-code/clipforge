"""NF4 against a REAL checkpoint, not a mocked `from_pretrained`.

`tests/unit/test_quantization.py` proves the provider builds the right
config object and hands it to the loader. That is a claim about wiring,
and it stays green whether or not a single weight is ever quantized —
which is precisely the shape of the bug it was written after: the
`quantize` knob was accepted, stored and never read for as long as it
existed.

So this loads Wan2.2-TI2V-5B for real and measures what came back. It is
the same class of check as `test_s1_real_gpu`: the mocked half says the
call was made, this half says the call did something.

gpu-marked (auto-skipped without CUDA) and additionally skipped when the
weights are not already in the local HF cache — a test must never be the
thing that starts a 10 GB download.

### Why the loads happen in a child process

The first version held both pipelines in this interpreter at once. That
is ~36 GB on a 32 GB machine: Windows paged, and a memory-mapped read
inside safetensors took an **access violation**, which does not fail a
test — it kills the process. On 2026-08-20 the gate exited 139 having
reported 8 of 1267 tests, the rest never run, and no summary line was ever
printed. A run that dies this way looks like a passing run to anything
that reads the exit code of a pipeline, and like a bare stack trace to
anything that tails the output.

`_quant_probe.py` moves each load into its own process, so the same crash
comes back as an exit code this file can name, and the peak footprint
belongs to a process that exits. The two loads also no longer overlap.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.gpu

MODEL = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"

_PROBE = Path(__file__).with_name("_quant_probe.py")
_REPO = Path(__file__).resolve().parents[2]
_SNAPSHOT_GLOB = ("models--" + MODEL.replace("/", "--")) + "/snapshots/*"

#: Loading is disk-bound the first time and mmap-warm afterwards; 10 GB of
#: shards through bitsandbytes is minutes, not seconds.
_TIMEOUT_S = 900


def _snapshot() -> Path:
    hub = Path.home() / ".cache/huggingface/hub"
    for snap in sorted(hub.glob(_SNAPSHOT_GLOB)):
        if (snap / "transformer").is_dir():
            return snap
    pytest.skip(f"{MODEL} is not in the local HF cache")
    raise AssertionError("unreachable")


def _bf16_gb(*subfolders: str) -> float:
    """What these components weigh ONCE LOADED, from the checkpoint itself.

    Not the file sizes: this checkpoint is stored fp32 and loaded bf16, so
    its 20.00 GB transformer is 10.02 GB of resident weights, and a guard
    built on file size demanded 28 GB for a load that needs 13. Every
    safetensors file carries a header of dtypes and shapes, so the loaded
    size is countable rather than assumed — element count times two bytes.

    Where a shard index exists only the files it names are counted; this
    repo has already met one checkpoint that ships two shardings of the
    same weights (LTX-2.5, 44 GB of it unreferenced).
    """
    snap = _snapshot()
    elements = 0
    for name in subfolders:
        d = snap / name
        if not d.is_dir():
            continue
        indexes = list(d.glob("*.index.json"))
        if indexes:
            named = set(json.loads(indexes[0].read_text())["weight_map"]
                        .values())
            files = [d / f for f in sorted(named)]
        else:
            files = [f for f in d.iterdir() if f.suffix == ".safetensors"]
        for f in files:
            with f.open("rb") as fh:
                n = int.from_bytes(fh.read(8), "little")
                header = json.loads(fh.read(n))
            for key, spec in header.items():
                if key == "__metadata__":
                    continue
                count = 1
                for dim in spec["shape"]:
                    count *= dim
                elements += count
    return elements * 2 / 1e9


def _required_gb(mode: str) -> float:
    """Free RAM this mode needs: the biggest single component it loads.

    Not the sum. The sum is what the guard asked for first, and in a full
    suite run — where earlier tests have already taken their RAM — it
    skipped all three tests on the machine they were written for, which
    is the coverage silently disappearing rather than the crash being
    prevented.

    The crash this guard exists for was ~36 GB of weights in ONE process,
    and that shape is gone: each load now runs in a child that holds one
    checkpoint and exits, and a child that dies is a named failure rather
    than the end of the run. What is left to catch is a machine that
    cannot hold a single component, so that is what is asked.

    Measured on this machine 2026-08-20, peak working set / peak commit:
    bf16 components 24.0 / 42.2 GB, NF4 pipeline 6.3 / 29.4 GB — both
    completed with ~19 GB free. Neither number is "the RAM this needs":
    most of it is memory-mapped checkpoint the OS can drop, and the NF4
    side is the cheaper because bitsandbytes quantizes layer by layer on
    the card. Modelling that precisely is not possible from here; erring
    toward running the test, with containment behind it, is.
    """
    if mode == "nf4":
        # The transformer arrives 4-bit; the text encoder does not.
        return max(_bf16_gb("text_encoder"), _bf16_gb("transformer") / 3,
                   _bf16_gb("vae")) * 1.2
    return max(_bf16_gb("transformer"), _bf16_gb("vae")) * 1.2


def _probe(mode: str) -> dict:
    """One load, one process, one JSON line back."""
    import psutil  # noqa: PLC0415 - only this guard needs it

    need = _required_gb(mode)
    free = psutil.virtual_memory().available / 1e9
    if free < need:
        pytest.skip(f"{mode}: needs ~{need:.1f} GB free RAM, {free:.1f} GB "
                    "available — this load pages and the paging fault is "
                    "an access violation, not a test failure")

    proc = subprocess.run(  # noqa: S603 - our own file, our own interpreter
        [sys.executable, str(_PROBE), MODEL, mode],
        cwd=_REPO, capture_output=True, text=True, timeout=_TIMEOUT_S)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        pytest.fail(
            f"the {mode} load produced no measurement. exit={proc.returncode}"
            f" (139/3221225477 is a segfault — almost always RAM: "
            f"{free:.1f} GB was free, ~{need:.1f} GB needed)\n"
            f"stderr tail: {proc.stderr[-800:]}")
    payload = json.loads(lines[-1])
    if "skip" in payload:
        pytest.skip(payload["skip"])
    return payload


@pytest.fixture(scope="module")
def bf16() -> dict:
    return _probe("none")


@pytest.fixture(scope="module")
def nf4() -> dict:
    return _probe("nf4")


def test_nf4_actually_replaces_the_linear_layers(nf4):
    """4-bit is a different LAYER class, not a dtype cast.

    Asserting on the footprint alone would pass for a model that merely
    loaded in fp8 or got sharded; the layer type is what says bitsandbytes
    took the weights.
    """
    assert nf4["attn_linear_class"] == "Linear4bit", nf4["attn_linear_class"]


def test_nf4_is_several_times_smaller_than_bfloat16(nf4, bf16):
    """The point of the knob: a transformer that fits a card it did not.

    Measured on this machine at 10.02 GB -> 2.54 GB. The floor is 3x
    rather than the measured 3.94x because the un-quantized norms and
    embeddings are a fixed overhead whose share moves with model shape,
    and this test should fail on a knob that stopped working, not on a
    checkpoint that changed.
    """
    big, small = bf16["transformer_bytes"], nf4["transformer_bytes"]
    assert small * 3 < big, f"{big/1e9:.2f} GB -> {small/1e9:.2f} GB"


def test_only_the_transformer_is_quantized_in_a_real_load(nf4, bf16):
    """The VAE decodes latents to pixels and is where 4-bit shows.

    The unit test pins the quant_mapping; this pins the OUTCOME, which is
    the thing a future `quantize the whole pipeline` shortcut would break.
    """
    assert nf4["vae_bytes"] == bf16["vae_bytes"]
    assert not nf4["vae_4bit_types"], nf4["vae_4bit_types"]
