"""Model registry and automatic selection, pinned.

The registry exists so nobody picks a model by hand per piece. The risk is
that selection quietly picks something that cannot render the request —
which, for video models, does not fail loudly. It returns a blank frame.
"""

from __future__ import annotations

import pytest

from clipforge.genvideo import models
from clipforge.genvideo.models import (ModelSpec, REGISTRY, WAN22_TI2V_5B,
                                       describe_registry, select_model)


def test_the_oversized_wan_variant_is_not_registered():
    """Wan's A14B does not fit 24 GB at video resolutions. A model that
    raises on every call is worse than one that is absent."""
    assert all("A14B" not in m.model_id for m in REGISTRY.values())


def test_every_model_fits_the_card_it_is_registered_for():
    for m in REGISTRY.values():
        assert m.vram_gb <= 24.0, f"{m.label} cannot fit a 24 GB card"


def test_supports_rejects_off_grid_sizes():
    """720 is not a multiple of 32 — the latent patch grid rejects it."""
    assert not WAN22_TI2V_5B.supports(720, 1280)
    assert WAN22_TI2V_5B.supports(704, 1280)


# ------------------------------------------------------- selection

def _pick(needs, w=480, h=896, **kw):
    return select_model(needs=set(needs), width=w, height=h, **kw)


@pytest.mark.skipif(not describe_registry()[0]["installed"],
                    reason="no weights installed")
def test_photoreal_work_prefers_the_higher_fidelity_model():
    assert _pick({"photoreal", "human"}).key == "wan22"


#: Synthetic registry for the selection tests. They used to assert
#: `.key == "ltx"`, which broke the moment LTX 0.9 was retired even though
#: the SELECTION logic had not changed at all. What matters is that needs
#: win, that cost breaks ties, and that a refusal names its reason — none
#: of which is a fact about the shipped registry.
_FAST = ModelSpec(
    key="fake_fast", model_id="test/fast", label="Fast",
    max_pixels=460_000, vram_gb=8.0, steps=20, guidance_scale=3.0, cost=1.0,
    strengths=frozenset({"fast", "atmospheric", "landscape", "draft"}))
_RICH = ModelSpec(
    key="fake_rich", model_id="test/rich", label="Rich",
    max_pixels=720 * 1280, vram_gb=18.0, steps=40, guidance_scale=5.0,
    cost=4.0, strengths=frozenset({"photoreal", "human", "cinematic"}))


@pytest.fixture()
def two_models(monkeypatch):
    """A registry with exactly two installed, verified models."""
    monkeypatch.setattr(models, "REGISTRY",
                        {m.key: m for m in (_FAST, _RICH)})
    monkeypatch.setattr(models, "weights_present", lambda spec: True)
    return _FAST, _RICH


def test_needs_choose_the_model_that_claims_them(two_models):
    fast, rich = two_models
    assert models.select_model(needs={"atmospheric", "landscape"},
                               width=480, height=896).key == fast.key
    assert models.select_model(needs={"photoreal", "human"},
                               width=480, height=896).key == rich.key


def test_no_needs_falls_back_to_the_cheapest_that_fits(two_models):
    """With nothing to match on, cost decides — not alphabetical order or
    dict insertion, both of which would be accidental."""
    fast, _rich = two_models
    assert models.select_model(needs=set(), width=480,
                               height=896).key == fast.key


def test_a_size_outside_every_envelope_is_refused(two_models):
    """And the message must say which constraint failed: "no model fits"
    sends an operator looking for a bigger GPU when the real problem is
    that 704x1280 is past the envelope."""
    with pytest.raises(ValueError) as err:
        models.select_model(needs=set(), width=4096, height=4096)
    assert "px" in str(err.value) or "exceeds" in str(err.value)


def test_forcing_a_model_that_cannot_render_the_size_says_why(two_models):
    fast, _rich = two_models
    with pytest.raises(ValueError) as err:
        models.select_model(needs=set(), width=704, height=1280,
                            prefer=fast.key)
    assert "px" in str(err.value) or "cannot render" in str(err.value)


@pytest.mark.skipif(not describe_registry()[0]["installed"],
                    reason="no weights installed")


@pytest.mark.skipif(not describe_registry()[0]["installed"],
                    reason="no weights installed")
def test_a_size_only_one_model_supports_selects_that_model():
    """704x1280 is over LTX's budget, inside Wan's. Selection must follow
    capability, not preference."""
    assert _pick({"atmospheric"}, w=704, h=1280).key == "wan22"


def test_an_impossible_size_explains_which_constraint_failed():
    """'No model fits' sends someone shopping for a GPU when the real
    problem is that 720 is off the latent grid."""
    with pytest.raises(ValueError) as err:
        _pick({"photoreal"}, w=720, h=1280)
    msg = str(err.value)
    assert "latent grid" in msg or "multiples of" in msg


def test_an_unknown_forced_model_is_rejected():
    with pytest.raises(ValueError) as err:
        _pick(set(), prefer="nope")
    assert "nope" in str(err.value)




def test_registry_description_is_dashboard_shaped():
    rows = describe_registry()
    assert rows
    for row in rows:
        assert {"key", "label", "installed", "fits_gpu", "usable",
                "max_pixels", "vram_gb", "strengths", "notes"} <= set(row)
    # Cheapest first, so a UI listing them in order reads sensibly.
    costs = [r["relative_cost"] for r in rows]
    assert costs == sorted(costs)
