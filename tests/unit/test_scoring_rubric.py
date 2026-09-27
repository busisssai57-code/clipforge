"""What the VL scorer is asked, and how its answers become an order.

The first version asked for three numbers with no definition of what any
number meant. A model asked to "score 0-10" with no anchors returns 7 for
nearly everything: the spread collapses and the ranking is then decided
by noise rather than by the clips.
"""

from __future__ import annotations

import pytest

from pathlib import Path

from clipforge.stages.s3_semantic import (COMPREHENSION_FLOOR, SCORE_WEIGHTS,
                                          SCORING_RUBRIC, weighted_score)

ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------- the rubric

def test_every_axis_is_anchored_at_both_ends():
    """An axis without anchors is a vibe. Each one has to say what a 0, a
    5 and a 10 look like, or the model invents its own scale per clip."""
    for axis in ("visual_action", "hook_strength", "comprehensibility"):
        assert axis in SCORING_RUBRIC
    for anchor in ("\n  0 ", "\n  5 ", "\n  10 "):
        assert SCORING_RUBRIC.count(anchor) == 3, (
            f"expected an anchor {anchor.strip()} on each of the three axes")


def test_the_rubric_pushes_for_spread():
    assert "WHOLE range" in SCORING_RUBRIC
    assert "find the difference" in SCORING_RUBRIC


def test_it_judges_the_first_frame_hardest():
    """A scrolling viewer sees one frame. Scoring the clip's average
    interest answers a question nobody asked."""
    assert "FIRST SECOND" in SCORING_RUBRIC
    assert "first frame hardest" in SCORING_RUBRIC


def test_justification_must_cite_evidence():
    """'Engaging content' is how a model says nothing while sounding
    certain. Naming the moment is checkable; a vibe is not."""
    assert "name the specific thing" in SCORING_RUBRIC
    assert "engaging content" in SCORING_RUBRIC     # named as NOT a reason


def test_the_hook_is_constrained_to_something_readable():
    assert "at most 7 words" in SCORING_RUBRIC
    assert "must be true" in SCORING_RUBRIC


def test_it_forbids_inventing_context():
    assert "Never assume what" in SCORING_RUBRIC
    assert "ONLY a JSON object" in SCORING_RUBRIC


def test_the_prompt_the_model_sees_is_the_rubric():
    import inspect

    from clipforge.stages import s3_semantic

    src = inspect.getsource(s3_semantic)
    assert "f\"{SCORING_RUBRIC}\\n\\n\"" in src, (
        "the local path has drifted back to its own prompt")


# -------------------------------------------------------------- the order

def test_the_hook_carries_the_most_weight():
    """A clip nobody stops for is never watched at all, so the axis that
    decides whether it is watched leads."""
    assert SCORE_WEIGHTS["hook_strength"] > SCORE_WEIGHTS["visual_action"]
    assert SCORE_WEIGHTS["visual_action"] > SCORE_WEIGHTS["comprehensibility"]
    assert sum(SCORE_WEIGHTS.values()) == pytest.approx(1.0)


def test_a_strong_hook_beats_a_prettier_clip_with_no_hook():
    """The case a flat average gets wrong: both average 6.67, and one of
    them is a clip nobody opens."""
    hooky = weighted_score(visual_action=5, hook_strength=9, comprehensibility=6)
    pretty = weighted_score(visual_action=9, hook_strength=5, comprehensibility=6)
    assert hooky > pretty
    assert (5 + 9 + 6) == (9 + 5 + 6), "the old mean could not tell these apart"


def test_a_clip_nobody_can_follow_ranks_below_every_clip_that_can_be():
    """A floor, not a penalty. Halving was not enough: a flashy
    incoherent candidate (10/10/3.9 = 8.78) still outranked a clear
    ordinary one (5/5/7 = 5.40), while the comment claimed it was ranked
    last. Two bands, so the claim is true."""
    from clipforge.schemas.ranking import RankedItem
    from clipforge.stages.s3_semantic import rank_key

    def item(i, va, hs, comp):
        return RankedItem(candidate_index=i, rank=1, visual_action=va,
                          hook_strength=hs, comprehensibility=comp,
                          justification="")

    flashy = item(0, 10, 10, COMPREHENSION_FLOOR - 0.1)
    ordinary = item(1, 5, 5, 7)
    assert [i.candidate_index for i in sorted([flashy, ordinary], key=rank_key)] == [1, 0]
    # ... and it is still visible, with its real score, not deleted.
    assert weighted_score(10, 10, 3.9) > weighted_score(5, 5, 7)


def test_the_scale_is_still_zero_to_ten():
    assert weighted_score(0, 0, 0) == 0.0
    assert weighted_score(10, 10, 10) == pytest.approx(10.0)


def test_ties_break_on_the_hook_then_on_index():
    """Behavioural, not a source grep: the old version of this test would
    have passed against a sort that was never called."""
    from clipforge.schemas.ranking import RankedItem
    from clipforge.stages.s3_semantic import rank_key

    def item(i, hs):
        return RankedItem(candidate_index=i, rank=1, visual_action=6,
                          hook_strength=hs, comprehensibility=6,
                          justification="")

    # Same weighted total, different hook: the hook wins.
    a, b = item(0, 5.0), item(1, 7.0)
    a.visual_action = 8.57                     # makes the totals equal
    assert weighted_score(a.visual_action, a.hook_strength, 6) == pytest.approx(
        weighted_score(6, b.hook_strength, 6), abs=0.01)
    assert [i.candidate_index for i in sorted([a, b], key=rank_key)][0] == 1

    # Identical in every axis: index decides, so the order is stable.
    same = [item(i, 6.0) for i in (3, 1, 2)]
    assert [i.candidate_index for i in sorted(same, key=rank_key)] == [1, 2, 3]


def test_both_judges_are_asked_the_same_question():
    """The cloud judge carried its own copy of the prompt — no anchors,
    no first-frame rule, no 7-word hook limit — so the same clip was
    scored by two different questions depending on which judge ran, and
    the numbers were not comparable."""
    from clipforge import vlrank

    assert SCORING_RUBRIC in vlrank._PROMPT, (
        "the cloud prompt has drifted from the rubric again")


def test_both_judges_order_clips_the_same_way():
    """The cloud path sorted on a flat sum of its own: the same clip
    ranked two different ways depending on which judge was available."""
    import inspect

    from clipforge.stages import s3_semantic

    cloud = inspect.getsource(s3_semantic.S3SemanticRanker._rank_with_cloud)
    assert "rank_key" in cloud, "the cloud path has its own ordering again"


def test_equal_scores_keep_a_stable_order():
    """§3.2: the same input must produce the same clip. Two candidates
    scored identically have to land in a defined order."""
    from clipforge.schemas.ranking import RankedItem

    items = [RankedItem(candidate_index=i, rank=1, visual_action=7.0,
                        hook_strength=7.0, comprehensibility=7.0,
                        justification="") for i in (3, 1, 2)]
    items.sort(key=lambda x: (-weighted_score(x.visual_action, x.hook_strength,
                                              x.comprehensibility),
                              -(x.hook_strength or 0), x.candidate_index))
    assert [i.candidate_index for i in items] == [1, 2, 3]


# ------------------------------------------------- the cache must let it run
#
# Found by the second audit, and confirmed against the live workspace: 13 of
# 21 stored S2 keys still resolved to S3 artifacts scored by the OLD prompt.
# The rubric was rewritten, the weights changed, and a re-run answered from
# cache and reported success. A scorer that cannot run is not a better
# scorer.

def test_the_rubric_is_part_of_what_the_cache_keys_on():
    from clipforge.stages.s3_semantic import S3SemanticRanker, scoring_digest

    stage = S3SemanticRanker(None, ".")
    assert stage.version.endswith(scoring_digest()), (
        "the stage version no longer follows the rubric, so editing the "
        "rubric would silently return old rankings")


def test_editing_the_rubric_or_the_weights_moves_the_digest():
    from clipforge.stages.s3_semantic import scoring_digest

    base = scoring_digest()
    assert scoring_digest(rubric=SCORING_RUBRIC + " and be nice") != base
    assert scoring_digest(weights={**SCORE_WEIGHTS, "hook_strength": 0.9}) != base
    assert scoring_digest(floor=COMPREHENSION_FLOOR + 1) != base


def test_the_digest_is_stable_across_processes():
    """§3.2: it is in a cache key, so it must not depend on set iteration,
    dict order or an address."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-c",
         "from clipforge.stages.s3_semantic import scoring_digest;"
         "print(scoring_digest())"],
        capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert out.returncode == 0, out.stderr[-400:]
    from clipforge.stages.s3_semantic import scoring_digest

    assert out.stdout.strip() == scoring_digest()


def test_an_old_ranking_cannot_answer_a_new_run():
    """The exact regression: same inputs, same params, old version."""
    from clipforge.stages.s3_semantic import S3SemanticRanker

    class Pre(S3SemanticRanker):
        version = "3"

    params = {"model_id": "m", "max_pixels": 451584, "seed": 1234}
    assert (Pre(None, ".").cache_key("a" * 64, params)
            != S3SemanticRanker(None, ".").cache_key("a" * 64, params))


def test_an_unparseable_judgement_is_not_a_middling_clip():
    """It used to write 5/5 and put S2's heuristic total (a 0-8.5 scale
    answering a different question) into hook_strength — the axis that
    now carries the most weight."""
    import inspect

    from clipforge.stages import s3_semantic

    src = inspect.getsource(s3_semantic)
    assert 'hook_strength=float(getattr(cand, "total_score"' not in src, (
        "an S2 heuristic is being written into a VL axis again")
    assert "did not parse" in src


def test_the_output_budget_fits_a_real_judgement():
    """Measured on 137 stored judgements: median 96 tokens, p90 130, max
    196. The cap was 128."""
    from clipforge.stages.s3_semantic import MAX_JUDGEMENT_TOKENS

    assert MAX_JUDGEMENT_TOKENS >= 256
    src = __import__("inspect").getsource(
        __import__("clipforge.stages.s3_semantic", fromlist=["x"]))
    assert "max_new_tokens=128" not in src
    assert "max_new_tokens=MAX_JUDGEMENT_TOKENS" in src
