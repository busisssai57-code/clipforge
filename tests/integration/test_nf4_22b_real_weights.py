"""NF4 against the real 22B LTX-2.5 checkpoint.

`test_quantization_real_weights.py` proves the provider's path end to end
on Wan 5B: `_quantization_config()` -> `from_pretrained` -> `Linear4bit`
layers -> a measured 3.94x. That is the wiring claim, and it is the right
one to make where it can be made.

It cannot be made here, and the reason is worth a test of its own. The
LTX-2.5 checkpoint is written for diffusers 0.40.0.dev0; the installed
0.39.0 silently ignores four of its config attributes, builds 96 feed-
forward biases the checkpoint does not carry, leaves them on meta, and
raises `Cannot copy out of meta tensor` on placement. So the module tree
cannot be constructed at all on this stack.

What can still be measured is the number the operator is actually asking
about -- whether 4-bit puts a 22B transformer inside a 24 GB card -- by
running the real shards through the same bitsandbytes kernel the provider
configures. That is a claim about the WEIGHTS, one layer below the claim
about the wiring, and the two are kept apart here deliberately: nothing in
this file should be read as "the LTX-2.5 route works".

Measured 2026-08-18 on this machine: 35.37 GB bf16 -> 9.13 GB NF4, 3.87x.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge.genvideo.providers import LocalDiffusersProvider

MODEL = "Lightricks/LTX-2.5-Diffusers"
_CACHE = (Path.home() / ".cache/huggingface/hub"
          / "models--Lightricks--LTX-2.5-Diffusers")

#: The card this project runs on. The whole point of the knob is this bound.
CARD_GB = 24.0


def _snapshot() -> Path:
    """The cached checkpoint, or a skip. A test never starts a 67 GB pull."""
    snaps = _CACHE / "snapshots"
    if not snaps.is_dir():
        pytest.skip(f"{MODEL} is not in the local HF cache")
    for d in snaps.iterdir():
        if (d / "transformer").is_dir():
            return d
    pytest.skip(f"{MODEL} snapshot has no transformer/")
    raise AssertionError("unreachable")


def _index() -> dict:
    idx = (_snapshot() / "transformer"
           / "diffusion_pytorch_model.safetensors.index.json")
    if not idx.is_file():
        pytest.skip("transformer shard index is missing")
    return json.loads(idx.read_text())


@pytest.fixture(scope="module")
def measured() -> dict:
    """Stream every shard through NF4 and total what comes out.

    Streamed one tensor at a time rather than loaded: the bf16 model is
    35 GB against 32 GB of host RAM, and paging it is what killed a render
    on this machine at 16:01 on 2026-08-18. The bf16 side of the ratio is
    therefore the checkpoint index's own `total_size` -- a DECLARED number,
    not one this test measured, which is why it is named as such.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("bitsandbytes")
    if not torch.cuda.is_available():
        pytest.skip("NF4 quantization requires CUDA")
    from bitsandbytes.functional import quantize_4bit
    from safetensors import safe_open

    idx = _index()
    tdir = _snapshot() / "transformer"

    # The provider's OWN config object. Reading the settings off a
    # hand-written BitsAndBytesConfig would test this test.
    qcfg = LocalDiffusersProvider(model_id=MODEL,
                                  quantize="nf4")._quantization_config()
    bnb = qcfg.quant_mapping["transformer"]
    assert bnb.load_in_4bit and bnb.bnb_4bit_quant_type == "nf4"

    quantized = kept = 0
    n_quantized = n_kept = 0
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(str(tdir / shard), framework="pt", device="cpu") as f:
            for key in f.keys():  # noqa: SIM118 - safetensors, not a dict
                tensor = f.get_tensor(key)
                if tensor.ndim != 2:
                    # Norms, biases and 1-D tensors stay bf16. Counted, not
                    # dropped: excluding them would flatter the total.
                    kept += tensor.nelement() * tensor.element_size()
                    n_kept += 1
                    continue
                gpu = tensor.to("cuda", torch.bfloat16)
                packed, state = quantize_4bit(
                    gpu, quant_type=bnb.bnb_4bit_quant_type,
                    compress_statistics=bnb.bnb_4bit_use_double_quant)
                quantized += packed.nelement() * packed.element_size()
                # Double quant's own statistics are part of what must fit.
                for absmax in (state.absmax,
                               getattr(state.state2, "absmax", None)):
                    if absmax is not None:
                        quantized += absmax.nelement() * absmax.element_size()
                n_quantized += 1
                del gpu, packed, state
                torch.cuda.empty_cache()
                del tensor
    return {"nf4_bytes": quantized + kept,
            "declared_bf16_bytes": idx["metadata"]["total_size"],
            "n_quantized": n_quantized, "n_kept": n_kept}


@pytest.mark.gpu
def test_nf4_puts_the_22b_transformer_inside_the_card(measured):
    """The knob's reason to exist, on the model that forced it.

    `requires_quantization="nf4"` on the registry entry is a claim that
    there is no unquantized path on this hardware AND that the quantized
    one fits. The second half is what this measures.
    """
    gb = measured["nf4_bytes"] / 2 ** 30
    assert gb < CARD_GB, f"NF4 transformer is {gb:.2f} GB against {CARD_GB} GB"
    # Headroom is the real claim: the transformer is not the only resident.
    # A 22B model that fits with 200 MB to spare would not run a pipeline.
    assert CARD_GB - gb > 8.0, f"only {CARD_GB - gb:.2f} GB left for the rest"


@pytest.mark.gpu
def test_the_ratio_holds_at_22b_as_it_did_at_5b(measured):
    """NF4 compresses the same whether the model is 5B or 22B.

    Bounds rather than the measured 3.87x: the un-quantized 1-D tensors are
    a fixed overhead whose share moves with model shape, so this should
    fail on a knob that stopped working, not on a re-sharded checkpoint.
    The 5B path measured 3.94x, this one 3.87x.
    """
    ratio = measured["declared_bf16_bytes"] / measured["nf4_bytes"]
    assert 3.4 < ratio < 4.4, f"{ratio:.2f}x"
    # 2-D weights are the overwhelming majority of a transformer's bytes;
    # if that stops being true the proxy above is wrong, not the kernel.
    assert measured["n_quantized"] > 1000


def test_installed_diffusers_still_cannot_build_this_checkpoint():
    """A RECORD of the blocker, written to fail when it is lifted.

    LTX-2.5's `config.json` declares `ff_bias: False` (among four
    attributes) and diffusers 0.39.0 does not know the name, so it ignores
    it and builds the feed-forwards WITH bias -- 96 tensors the checkpoint
    has no weights for. Silently: the only symptom is a meta-tensor error
    much later, at placement.

    When diffusers is upgraded this test FAILS, on purpose. That failure is
    the signal to run the real wiring test against LTX-2.5, correct the
    provisional envelope numbers on the registry entry, and flip
    `verified`. Deleting it instead is exactly the silent scope reduction
    this project treats as a build failure.
    """
    torch = pytest.importorskip("torch")
    from diffusers import LTX2VideoTransformer3DModel

    idx = _index()
    cfg = LTX2VideoTransformer3DModel.load_config(
        MODEL, subfolder="transformer", local_files_only=True)
    with torch.device("meta"):
        model = LTX2VideoTransformer3DModel.from_config(cfg)

    missing = set(model.state_dict()) - set(idx["weight_map"])
    unexpected = set(idx["weight_map"]) - set(model.state_dict())
    assert missing or unexpected, (
        "the installed diffusers now agrees with this checkpoint. "
        "Run the LTX-2.5 wiring + render checks, replace the provisional "
        "envelope numbers on the registry entry with measured ones, and "
        "delete this test with the ledger entry that explains why.")
    # Named, so the record says WHICH disagreement, not merely that one exists.
    assert any(k.endswith("ff.net.0.proj.bias") for k in missing), (
        sorted(missing)[:5])
