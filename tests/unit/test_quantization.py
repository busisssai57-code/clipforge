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


def test_ltx25_is_not_auto_selected_while_unverified():
    """Every number in its envelope came from a model card. Auto-selecting
    on published figures is how blank frames ship silently — this project
    has been burned by exactly that twice."""
    assert REGISTRY["ltx25"].verified is False
    assert "ltx25" not in [m.key for m in available_models(24.0)]


def test_the_unverified_model_is_not_what_auto_selection_picks():
    """The regression guard. LTX 0.9 was retired on 2026-08-13, so this no
    longer names it — what must stay true is that selection never lands on
    a model whose envelope came from a model card."""
    picked = select_model(width=512, height=896)
    assert picked.verified is True
    assert picked.key != "ltx25"


def test_a_verified_model_is_still_selectable():
    keys = {m.key for m in available_models(24.0)}
    assert keys, "no model is auto-selectable — generation cannot run"
    assert "ltx25" not in keys


def test_requires_quantization_overrides_the_operator_preference():
    """config `quantize` is a preference; `requires_quantization` is a
    fact. Loading a model that cannot fit at bf16 must not depend on the
    operator having set a knob."""
    import inspect

    from clipforge import genvideo

    src = inspect.getsource(genvideo.build_router)
    assert "requires_quantization" in src
    assert 'controls["quantize"] = spec.requires_quantization' in src
