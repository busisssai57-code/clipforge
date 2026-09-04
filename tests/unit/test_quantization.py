"""The quantize knob must actually quantize, and models that need it get it.

`[genvideo] quantize` shipped dead: config accepted it, the provider stored
it on `self.quantize`, and nothing ever read it back. The config comment
said "quantize the transformer to 8-bit to fit a smaller card" while the
transformer was loaded at bf16 every time. Same shape as `enhance` having
no caller and split-screen advertising itself LIVE — a control that
describes an outcome it does not produce.

It stopped being cosmetic when LTX-2.5 was registered: 22B at bf16 is
~38 GB of weights against a 24 GB card, so quantization is the difference
between that model running and not running at all.
"""

from __future__ import annotations

import pytest

from clipforge.genvideo.models import REGISTRY, available_models, select_model
from clipforge.genvideo.providers import LocalDiffusersProvider


def _provider(mode: str) -> LocalDiffusersProvider:
    return LocalDiffusersProvider("some/model", quantize=mode)


# ------------------------------------------------------- the knob works

def test_quantization_off_produces_no_config():
    assert _provider("none")._quantization_config() is None


@pytest.mark.parametrize("mode", ["", "off", "false", "NONE"])
def test_every_spelling_of_off_is_off(mode):
    assert _provider(mode)._quantization_config() is None


def test_nf4_produces_a_real_quantization_config():
    """The test that fails against the shipped code, where this method
    did not exist and the knob was inert."""
    cfg = _provider("nf4")._quantization_config()
    assert cfg is not None


def test_int8_produces_a_real_quantization_config():
    assert _provider("int8")._quantization_config() is not None


def test_only_the_transformer_is_quantized():
    """Quantizing the VAE is how the brown-frame bug comes back — it
    already has to run in fp32."""
    cfg = _provider("nf4")._quantization_config()
    mapping = getattr(cfg, "quant_mapping", None)
    assert mapping is not None
    assert set(mapping) == {"transformer"}


def test_nf4_computes_in_bfloat16():
    """The 3090 is Ampere and has no native FP4 tensor cores, so 4-bit is
    a storage format dequantized per layer. Compute must stay bf16 or the
    numerics change under us."""
    import torch

    cfg = _provider("nf4")._quantization_config()
    bnb = cfg.quant_mapping["transformer"]
    assert bnb.bnb_4bit_compute_dtype is torch.bfloat16
    assert bnb.bnb_4bit_quant_type == "nf4"


def test_an_unknown_mode_degrades_loudly_rather_than_guessing():
    assert _provider("fp6_imaginary")._quantization_config() is None


def test_missing_bitsandbytes_is_not_a_crash(monkeypatch):
    """An operator without bitsandbytes gets an unquantized run and a
    warning, not a traceback in the middle of a render."""
    import importlib.util

    real = importlib.util.find_spec

    def fake(name, *a, **kw):
        return None if name == "bitsandbytes" else real(name, *a, **kw)

    monkeypatch.setattr(importlib.util, "find_spec", fake)
    assert _provider("nf4")._quantization_config() is None


# ------------------------------------------------ it reaches the loader

def test_the_config_is_passed_to_from_pretrained():
    """The knob's whole job. Asserted against the SOURCE because the real
    call downloads tens of gigabytes."""
    import inspect

    from clipforge.genvideo import providers

    src = inspect.getsource(providers)
    assert "quantization_config" in src
    assert "load_kw[\"quantization_config\"] = qcfg" in src, (
        "the quantization config must reach from_pretrained; storing it "
        "on self and never reading it is the bug this replaced")


# -------------------------------------------- models that require it

def test_ltx25_is_registered():
    assert "ltx25" in REGISTRY


def test_ltx25_requires_nf4_to_fit_the_card():
    """38 GB of bf16 weights against 24 GB of card. This is a property of
    the model, not an operator preference."""
    assert REGISTRY["ltx25"].requires_quantization == "nf4"


def test_ltx25_uses_the_schedule_that_was_measured_here():
    """This asserted 8 steps at CFG 1.0 - "the distilled schedule from the
    model card" - and it was wrong, in the way the whole registry is
    supposed to protect against: a published number pinned by a test that
    only checked it had been copied correctly.

    Measured 2026-08-20 on one prompt and seed, 8/1.0 against the 30/3.0
    that diffusers' own LTX2 example uses: sharpness (Laplacian variance)
    110.9 -> 134.8, spatial std 41.8 -> 58.5, and by eye the difference is
    blobby fur and a smeared human against real coat texture and legible
    market stalls. The operator's word for the 8-step output was
    "morphing". It costs 2.7x the time.
    """
    spec = REGISTRY["ltx25"]
    assert (spec.steps, spec.guidance_scale) == (30, 3.0)


@pytest.fixture
def one_verified_model(monkeypatch):
    """A registry with one downloaded VERIFIED model beside unverified
    ltx25, independent of what this machine has on disk.

    The three tests below are about selection POLICY — that `verified`
    gates automatic choice, and that a verified model is still reachable.
    They answered that by reading the real HF cache, so on a box where
    wan22 (the only verified entry) is not downloaded they failed for a
    reason that has nothing to do with the policy they assert. What is
    actually downloaded is preflight's question, not a unit test's.

    It also gives `test_ltx25_is_not_auto_selected_while_unverified` its
    teeth back: with `weights_present` true for everything, ltx25 can be
    absent from `available_models` for exactly one reason. Unpatched, that
    test passes on a machine that has never fetched ltx25 at all — a pass
    that proves nothing, which is the trap this repo keeps rediscovering.
    """
    import dataclasses

    from clipforge.genvideo import models as m

    # Costlier than ltx25 ON PURPOSE. select_model breaks ties on cost and
    # then on key, so a cheap fake would be picked over ltx25 by the
    # alphabet even with the `verified` filter deleted, and
    # `test_the_unverified_model_is_not_what_auto_selection_picks` would
    # pass while proving nothing. Priced above it, the only thing keeping
    # selection off ltx25 is the policy under test.
    verified = dataclasses.replace(
        REGISTRY["ltx25"], key="fake_verified", model_id="Fake/Verified",
        label="Fake Verified Model", verified=True, interpreter="",
        requires_quantization="", cost=REGISTRY["ltx25"].cost + 1.0)
    registry = {"fake_verified": verified, "ltx25": REGISTRY["ltx25"]}
    monkeypatch.setattr(m, "REGISTRY", registry)
    monkeypatch.setattr(m, "weights_present", lambda spec: True)
    return registry


def test_ltx25_is_not_auto_selected_while_unverified(one_verified_model):
    """Every number in its envelope came from a model card. Auto-selecting
    on published figures is how blank frames ship silently — this project
    has been burned by exactly that twice."""
    assert REGISTRY["ltx25"].verified is False
    assert "ltx25" not in [m.key for m in available_models(24.0)]


def test_the_unverified_model_is_not_what_auto_selection_picks(
        one_verified_model):
    """The regression guard. LTX 0.9 was retired on 2026-08-13, so this no
    longer names it — what must stay true is that selection never lands on
    a model whose envelope came from a model card."""
    picked = select_model(width=512, height=896)
    assert picked.verified is True
    assert picked.key != "ltx25"


def test_a_verified_model_is_still_selectable(one_verified_model):
    keys = {m.key for m in available_models(24.0)}
    assert keys, "no model is auto-selectable — generation cannot run"
    assert "ltx25" not in keys


# ------------------------------- the reason given for "nothing fits"

def _only_ltx25(monkeypatch):
    """A registry holding just the unverified model, downloaded."""
    from clipforge.genvideo import models as m

    monkeypatch.setattr(m, "REGISTRY", {"ltx25": REGISTRY["ltx25"]})
    monkeypatch.setattr(m, "weights_present", lambda spec: True)


def test_an_unverified_model_is_not_blamed_on_the_pixel_budget(monkeypatch):
    """The message this replaced, verbatim from this machine:

        no installed model can render 512x896. Wan 2.2 TI2V 5B: not
        downloaded; LTX-2.5 22B (distilled): 458,752 px exceeds its
        921,600 px budget

    458,752 is LESS than 921,600. The chain modelled every filter
    `available_models` applies except `verified`, so the fall-through
    branch blamed the budget for a size well inside it — an operator
    reading that shrinks a piece that was never too big, and never learns
    that `--model ltx25` is the one thing that would have worked.
    """
    _only_ltx25(monkeypatch)
    with pytest.raises(ValueError) as exc:
        select_model(width=512, height=896)

    msg = str(exc.value)
    assert "unverified" in msg
    assert "--model ltx25" in msg
    assert "exceeds" not in msg, msg
    assert "px budget" not in msg, msg


def test_a_size_really_past_the_budget_still_says_so(monkeypatch):
    """`verified` is reported LAST because it is the only reason an
    explicit `--model` can overrule. A shot that is genuinely too big is
    refused on that request too, so the budget is what the operator needs
    to hear — naming `verified` here would be the same defect pointing
    the other way."""
    _only_ltx25(monkeypatch)
    with pytest.raises(ValueError) as exc:
        select_model(width=1920, height=1088)

    msg = str(exc.value)
    assert "2,088,960 px exceeds its 921,600 px budget" in msg
    assert "unverified" not in msg, msg


def test_an_off_grid_size_still_says_so(monkeypatch):
    """Same ordering, the other hard blocker: 500 is not a multiple of 32
    and no flag makes it one."""
    _only_ltx25(monkeypatch)
    with pytest.raises(ValueError) as exc:
        select_model(width=500, height=896)

    msg = str(exc.value)
    assert "off the latent grid" in msg
    assert "unverified" not in msg, msg


def test_requires_quantization_overrides_the_operator_preference():
    """config `quantize` is a preference; `requires_quantization` is a
    fact. Loading a model that cannot fit at bf16 must not depend on the
    operator having set a knob."""
    import inspect

    from clipforge import genvideo

    src = inspect.getsource(genvideo.build_router)
    assert "requires_quantization" in src
    assert 'controls["quantize"] = spec.requires_quantization' in src


# ---------------------------------------------------- what "downloaded" means

def _fake_cache(root, model_id, *, files):
    """A HF cache folder for ``model_id`` containing exactly ``files``."""
    snap = (root / ("models--" + model_id.replace("/", "--"))
            / "snapshots" / "abc123")
    for name, body in files.items():
        f = snap / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8") if isinstance(body, str) \
            else f.write_bytes(body)
    return snap


def test_a_config_only_cache_is_not_a_downloaded_model(tmp_path, monkeypatch):
    """The state that cost a pipeline run: 16 MB of JSON, no weights.

    `weights_present` asked whether the snapshot folder had ANY file in
    it, so config.json and the tokenizer reported a 7 GB checkpoint as
    installed. Selection said installed, the run started, and S3 blocked
    mid-pipeline downloading the model it had been told was there.
    """
    import dataclasses

    from clipforge.genvideo import models as m

    spec = dataclasses.replace(REGISTRY["wan22"], model_id="Fake/ConfigOnly")
    _fake_cache(tmp_path, spec.model_id,
                files={"config.json": "{}", "tokenizer.json": "{}",
                       "README.md": "hi"})
    monkeypatch.setattr(m, "hf_cache_dir",
                        lambda mid: tmp_path / ("models--"
                                                + mid.replace("/", "--")))
    assert m.weights_present(spec) is False


def test_a_half_fetched_sharded_model_is_not_downloaded(tmp_path, monkeypatch):
    """1.6 GB of a 16 GB checkpoint - the plain Qwen VL cache's real state.

    The index names every shard, so the missing ones are countable.
    """
    import dataclasses
    import json

    from clipforge.genvideo import models as m

    spec = dataclasses.replace(REGISTRY["wan22"], model_id="Fake/HalfShards")
    index = json.dumps({"weight_map": {"a": "model-00001-of-00002.safetensors",
                                       "b": "model-00002-of-00002.safetensors"}})
    _fake_cache(tmp_path, spec.model_id,
                files={"model.safetensors.index.json": index,
                       "model-00001-of-00002.safetensors": b"\x00" * 16})
    monkeypatch.setattr(m, "hf_cache_dir",
                        lambda mid: tmp_path / ("models--"
                                                + mid.replace("/", "--")))
    assert m.weights_present(spec) is False

    # The missing shard arrives and the answer flips - no hardcoded False
    # to remember to delete.
    (tmp_path / ("models--" + spec.model_id.replace("/", "--"))
     / "snapshots" / "abc123"
     / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * 16)
    assert m.weights_present(spec) is True


def test_a_download_in_flight_is_not_downloaded(tmp_path, monkeypatch):
    """`.incomplete` blobs are what the old comment claimed to check."""
    import dataclasses

    from clipforge.genvideo import models as m

    spec = dataclasses.replace(REGISTRY["wan22"], model_id="Fake/InFlight")
    _fake_cache(tmp_path, spec.model_id,
                files={"model.safetensors": b"\x00" * 16})
    root = tmp_path / ("models--" + spec.model_id.replace("/", "--"))
    (root / "blobs").mkdir(parents=True, exist_ok=True)
    (root / "blobs" / "deadbeef.incomplete").write_bytes(b"\x00")
    monkeypatch.setattr(m, "hf_cache_dir", lambda mid: root)
    assert m.weights_present(spec) is False


def test_the_frame_group_a_model_declares_is_the_one_it_gets():
    """`frame_group` was a registry field with no reader.

    `_latent_frames` hardcoded 8, so a model whose temporal VAE groups by
    4 or 16 could declare it correctly and still be asked for 8n+1 - the
    same dead-knob shape as the `quantize` field fixed on 2026-08-18.
    """
    import dataclasses
    from pathlib import Path

    from clipforge.genvideo.providers import _latent_frames
    from clipforge.genvideo.subproc import SubprocessModelProvider

    assert _latent_frames(2.0, 24, group=8) == 49
    assert _latent_frames(2.0, 24, group=4) == 49
    assert _latent_frames(2.0, 24, group=16) == 49
    # 24 frames rounds DOWN to one group of 16 plus a keyframe,
    # because 17 is nearer to 24 than 33 is.
    assert _latent_frames(1.0, 24, group=16) == 17
    assert _latent_frames(1.0, 24, group=8) == 25

    spec = dataclasses.replace(REGISTRY["ltx25"], frame_group=16)
    provider = SubprocessModelProvider(spec)
    captured = {}
    provider._start = lambda: captured.setdefault("started", True)  # noqa: SLF001

    def _call(request, *, timeout):
        captured["frames"] = request["frames"]
        raise RuntimeError("stop after the request is built")

    provider._call = _call  # noqa: SLF001
    try:
        provider.generate(prompt="x", seconds=1.0, fps=24,
                          out_path=Path("unused.mp4"))
    except RuntimeError:
        pass
    assert captured["frames"] == 17, (
        "the spec's frame_group did not reach the request")
