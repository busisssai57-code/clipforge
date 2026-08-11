"""The gate must be able to detect its OWN shrinkage.

CP2 round-1 META-1 (BLOCKER): every verify module reports ``N/N passed`` where
``N = len(_CHECKS)`` — the denominator is produced by the very list it counts.
Deleting two ``@check`` decorators produced ``ai verify: 10/10 passed`` and
``GATE PASSED``, exit 0. VERIFICATION.md's own header says silent scope
reduction is a build failure, and the gate could not enforce it.

These counts are HARDCODED on purpose. When you legitimately add a check, this
test fails and you raise the number in the same commit — which is exactly the
point: the change becomes visible in review instead of invisible in a
self-reported ratio.
"""

from __future__ import annotations

import pytest

from clipforge.verify import ai, all as verify_all, ingestion, skeleton

#: (module, expected number of @check-registered gate checks)
EXPECTED_CHECKS = [
    (skeleton, 7),
    (ingestion, 20),
    # 13 since the S4 geometry invariants became their own check: they used
    # to ride on a full S4 run, which now correctly REFUSES on footage with
    # no subject, so they moved onto the pure helper that computes them.
    (ai, 13),
]


@pytest.mark.parametrize("module,expected",
                         EXPECTED_CHECKS,
                         ids=lambda v: getattr(v, "__name__", v))
def test_gate_module_has_not_silently_shrunk(module, expected):
    got = len(module._CHECKS)
    assert got == expected, (
        f"{module.__name__} registers {got} checks, expected {expected}. "
        "If you ADDED a check, raise the number here in the same commit. "
        "If you did not, a gate check has been deleted and every "
        "'N/N passed' line since is meaningless.")


def test_verify_all_runs_every_gate_module():
    names = [name for name, _mod in verify_all.MODULES]
    assert names == ["skeleton", "ingestion", "ai"], names
    assert len(verify_all.MODULES) == 3


def test_no_gate_check_is_registered_twice():
    """A duplicated name inflates the denominator while covering nothing."""
    for module, _ in EXPECTED_CHECKS:
        names = [name for name, _fn in module._CHECKS]
        assert len(names) == len(set(names)), (
            f"{module.__name__} has duplicate check names: "
            f"{sorted(n for n in names if names.count(n) > 1)}")


def test_gate_check_names_are_pinned():
    """Renaming a check while deleting another keeps the COUNT stable.

    Counting alone is not enough: swap one check for a trivial one and the
    denominator never moves. The names are the identity of the coverage.
    """
    ai_names = {name for name, _ in ai._CHECKS}
    assert "never two models resident (sampled at each load, not from labels)" \
        in ai_names
    assert "the two VRAM-Law layers COMPOSE (orchestrator + in-stage session)" \
        in ai_names
    assert "S2 is byte-deterministic" in ai_names
    assert "spec constants are pinned (they were all silently editable)" \
        in ai_names
