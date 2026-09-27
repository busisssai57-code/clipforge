"""What the VL scorer is asked, and how its answers become an order.

The first version asked for three numbers with no definition of what any
number meant. A model asked to "score 0-10" with no anchors returns 7 for
nearly everything: the spread collapses and the ranking is then decided
by noise rather than by the clips.
"""

from __future__ import annotations

import pytest

from clipforge.stages.s3_semantic import (COMPREHENSION_FLOOR, SCORE_WEIGHTS,
                                          SCORING_RUBRIC, weighted_score)


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


def test_a_clip_nobody_can_follow_is_pushed_down_hard():
    """Comprehensibility is a floor, not a prize: below it the clip is
    not a good clip with a flaw, it is not a clip."""
    followable = weighted_score(8, 8, COMPREHENSION_FLOOR)
    confusing = weighted_score(8, 8, COMPREHENSION_FLOOR - 0.1)
    assert confusing < followable / 1.9


def test_the_scale_is_still_zero_to_ten():
    assert weighted_score(0, 0, 0) == 0.0
    assert weighted_score(10, 10, 10) == pytest.approx(10.0)


def test_ranking_uses_the_weighted_score_and_breaks_ties_on_the_hook():
    import inspect

    from clipforge.stages import s3_semantic

    src = inspect.getsource(s3_semantic)
    assert "weighted_score(x.visual_action" in src, (
        "the ranking is back to a flat average")
    assert "-(x.hook_strength or 0)" in src, "ties are no longer broken on the hook"
    assert "x.candidate_index" in src, (
        "a tie with no final key makes the order depend on dict iteration, "
        "which breaks the Determinism Law")


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
