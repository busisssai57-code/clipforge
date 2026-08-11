"""CP2 pre-review sweep findings: S2 boundaries and float determinism.

Both behaviours below were REVERT-SAFE — neutralizing either left the whole
gate green — which under the standing rule since round 4 means they were not
covered at all.
"""

from __future__ import annotations

import math
from itertools import permutations

import pytest

from clipforge.schemas import CandidateWindow
from clipforge.stages.s2_prefilter import iou, nms, score_window, weighted_total


def _cand(start: float, end: float, score: float) -> CandidateWindow:
    return CandidateWindow(start=start, end=end, total_score=score,
                           scores={"total": score}, text="x")


# --------------------------------------------------------------------------
# S2-2: the NMS suppression boundary was unpinned, and float IoU made which
# side a candidate lands on depend on representation.
# --------------------------------------------------------------------------


def test_a_candidate_exactly_on_the_threshold_is_KEPT():
    """The deliberate boundary: `> threshold` suppresses, so exactly-equal
    survives. Flipping to `>=` silently drops a whole class of candidates."""
    # [0,10] and [4,10]: inter = 6, union = 10 -> IoU exactly 0.6.
    kept = nms([_cand(0.0, 10.0, 2.0), _cand(4.0, 10.0, 1.0)],
               iou_threshold=0.6, top_k=10)
    assert len(kept) == 2, (
        "a candidate at exactly the threshold was suppressed; the boundary "
        "is `>`, not `>=`")
    # ...and a hair above it IS suppressed, so the test cannot pass by the
    # threshold simply never biting.
    kept2 = nms([_cand(0.0, 10.0, 2.0), _cand(4.0, 10.0, 1.0)],
                iou_threshold=0.5999, top_k=10)
    assert len(kept2) == 1


def test_iou_is_exact_at_the_boundary_despite_float_input():
    """Second-scale float subtraction leaves error in BOTH terms of the
    ratio, so a pair sitting on the threshold could fall either way. In
    integer milliseconds the quotient is an exact rational."""
    # 0.1 + 0.2 != 0.3 in binary float; these endpoints are chosen so a
    # naive float computation does not land on the exact ratio.
    a = (0.1, 0.3)
    b = (0.2, 0.3)
    assert iou(a, b) == pytest.approx(0.5, abs=0.0), iou(a, b)
    # Exactly representable as a rational in ms: inter=100, union=200.
    assert iou(a, b) == 100 / 200


def test_iou_is_symmetric_and_zero_for_disjoint():
    assert iou((0.0, 5.0), (10.0, 15.0)) == 0.0
    assert iou((0.0, 10.0), (5.0, 15.0)) == iou((5.0, 15.0), (0.0, 10.0))


# --------------------------------------------------------------------------
# S2-6: the weighted total's last bits depended on iteration order, held
# stable only by a `sorted()` nothing tested.
# --------------------------------------------------------------------------


#: Magnitudes chosen so naive left-to-right addition is order-SENSITIVE.
_COMPONENTS = {
    "boundary": 1e16,
    "qa": 1.0,
    "turns": -1e16,
    "energy": 0.5,
    "laughter": 1.0,
    "selfcont": -0.5,
}
_WEIGHTS = dict.fromkeys(_COMPONENTS, 1.0)


def test_the_chosen_components_really_are_order_sensitive():
    """Guards the two tests below from passing vacuously: if plain `sum`
    were order-independent for these values, they would prove nothing."""
    orders = {sum(_WEIGHTS[k] * v for k, v in perm)
              for perm in permutations(_COMPONENTS.items())}
    assert len(orders) > 1, "these values do not exercise non-associativity"


def test_weighted_total_is_bit_identical_under_every_permutation():
    base = weighted_total(_COMPONENTS, _WEIGHTS)
    for perm in permutations(_COMPONENTS.items()):
        got = weighted_total(dict(perm), _WEIGHTS)
        assert got == base, f"{got!r} != {base!r} for {[k for k, _ in perm]}"
    assert base == math.fsum(_WEIGHTS[k] * v
                             for k, v in sorted(_COMPONENTS.items()))


def test_score_window_returns_sorted_keys_with_total_last():
    """The docstring promised byte-stable serialization while returning the
    dict in construction order — the promise rested on nobody reordering
    the literal."""
    scores = score_window([], 30.0, {})
    keys = list(scores)
    assert keys[-1] == "total"
    assert keys[:-1] == sorted(keys[:-1]), keys
