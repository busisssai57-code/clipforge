"""Model registry and automatic selection, pinned.

The registry exists so nobody picks a model by hand per piece. The risk is
that selection quietly picks something that cannot render the request —
which, for video models, does not fail loudly. It returns a blank frame.
"""

from __future__ import annotations

import pytest

from clipforge.genvideo.models import (LTX_VIDEO, REGISTRY, WAN22_TI2V_5B,
                                       describe_registry, select_model)


def test_the_measured_ltx_envelope_is_recorded():
    """460k px is the bisected boundary: 512x896 renders, 704x1280 is
    blank. If this creeps up, blank videos come back."""
    assert LTX_VIDEO.max_pixels == 460_000
    assert LTX_VIDEO.max_pixels < 704 * 1280


def test_the_oversized_wan_variant_is_not_registered():
    """Wan's A14B does not fit 24 GB at video resolutions. A model that
    raises on every call is worse than one that is absent."""
    assert all("A14B" not in m.model_id for m in REGISTRY.values())


def test_every_model_fits_the_card_it_is_registered_for():
    for m in REGISTRY.values():
        assert m.vram_gb <= 24.0, f"{m.label} cannot fit a 24 GB card"


def test_supports_rejects_over_budget_sizes():
    assert LTX_VIDEO.supports(480, 896)
    assert not LTX_VIDEO.supports(704, 1280), (
        "the size that produced blank frames must be refused")


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


@pytest.mark.skipif(not describe_registry()[0]["installed"],
                    reason="no weights installed")
def test_atmospheric_landscape_prefers_the_fast_model():
    assert _pick({"atmospheric", "landscape"}).key == "ltx"


@pytest.mark.skipif(not describe_registry()[0]["installed"],
                    reason="no weights installed")
def test_no_needs_falls_back_to_the_cheapest_that_fits():
    """With nothing to match on, cost decides — not alphabetical order or
    dict insertion, both of which would be accidental."""
    assert _pick(set()).key == "ltx"


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


def test_forcing_a_model_that_cannot_render_the_size_says_why():
    with pytest.raises(ValueError) as err:
        _pick(set(), w=704, h=1280, prefer="ltx")
    assert "px" in str(err.value) or "cannot render" in str(err.value)


def test_registry_description_is_dashboard_shaped():
    rows = describe_registry()
    assert rows
    for row in rows:
        assert {"key", "label", "installed", "fits_gpu", "usable",
                "max_pixels", "vram_gb", "strengths", "notes"} <= set(row)
    # Cheapest first, so a UI listing them in order reads sensibly.
    costs = [r["relative_cost"] for r in rows]
    assert costs == sorted(costs)
