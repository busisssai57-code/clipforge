"""Per-clip score breakdown, pinned.

The risk with a score card is not that a number is slightly off — it is
that it looks authoritative while resting on nothing. These tests hold
that every dimension declares what produced it, and that an absent
vision-language pass is visible rather than papered over with heuristics
wearing a model's clothes.
"""

from __future__ import annotations

import pytest

from clipforge.scorecard import ScoreCard, build_scorecard, grade_for

FULL_S2 = {"boundary": 0.9, "turns": 0.8, "energy": 0.7,
           "laughter": 0.6, "selfcont": 0.85, "qa": 0.5}


def test_grades_are_ordered_and_bounded():
    assert grade_for(100) == "A+"
    assert grade_for(0) == "D"
    scores = [95, 87, 82, 77, 72, 67, 60, 20]
    grades = [grade_for(s) for s in scores]
    assert len(set(grades)) > 4, "bands collapse everything into one grade"


def test_all_four_dimensions_are_reported():
    card = build_scorecard(s2_scores=FULL_S2, visual_action=7.0,
                           hook_strength=9.0, comprehensibility=8.0,
                           ranking_source="vl")
    assert [d.name for d in card.dimensions] == ["Hook", "Flow", "Value",
                                                 "Motion"]
    assert all(0 <= d.score <= 100 for d in card.dimensions)
    assert card.grade == grade_for(card.overall)


def test_the_fourth_dimension_is_not_called_trend():
    """A trend score needs platform data this machine never fetches.
    Naming a locally-measured number 'Trend' would be fabricating a
    measurement — the exact failure this codebase keeps finding."""
    card = build_scorecard(s2_scores=FULL_S2, visual_action=7.0)
    names = [d.name.lower() for d in card.dimensions]
    assert "trend" not in names
    motion = [d for d in card.dimensions if d.name == "Motion"][0]
    assert "trend" in motion.note.lower(), (
        "the card should say WHY there is no trend score")


def test_a_dimension_records_what_produced_it():
    card = build_scorecard(s2_scores=FULL_S2, hook_strength=9.0,
                           comprehensibility=8.0, visual_action=7.0,
                           ranking_source="vl")
    hook = card.dimensions[0]
    assert "vl:hook_strength" in hook.inputs
    assert "s2:boundary" in hook.inputs
    assert hook.model_judged is True


def test_without_the_vl_pass_dimensions_say_so():
    """The honesty case: heuristics must not be presented as a model
    judgement. A clip ranked by keyword counting should look different
    from one a model actually watched."""
    card = build_scorecard(s2_scores=FULL_S2, ranking_source="heuristic")
    assert card.source == "heuristic"
    hook = card.dimensions[0]
    assert hook.model_judged is False
    assert "heuristic only" in hook.note
    # Flow is heuristic by construction and should not claim otherwise.
    assert card.dimensions[1].model_judged is False


def test_the_vl_pass_is_detected_even_if_source_is_unset():
    card = build_scorecard(s2_scores=FULL_S2, hook_strength=8.0)
    assert card.source == "vl"


def test_missing_inputs_do_not_crash_or_fabricate():
    card = build_scorecard()
    assert isinstance(card, ScoreCard)
    assert card.overall == 0.0
    assert all(d.score == 0.0 for d in card.dimensions)
    assert all(d.inputs == () for d in card.dimensions)


def test_a_strong_hook_outranks_a_weak_one_all_else_equal():
    strong = build_scorecard(s2_scores=FULL_S2, hook_strength=10.0,
                             comprehensibility=6.0, visual_action=6.0)
    weak = build_scorecard(s2_scores=FULL_S2, hook_strength=2.0,
                           comprehensibility=6.0, visual_action=6.0)
    assert strong.overall > weak.overall
    assert strong.dimensions[0].score > weak.dimensions[0].score


def test_hook_is_weighted_hardest():
    """Short-form is won in the first seconds; every other dimension is
    moot if nobody stays."""
    hook_up = build_scorecard(s2_scores=FULL_S2, hook_strength=10.0,
                              comprehensibility=5.0, visual_action=5.0)
    motion_up = build_scorecard(s2_scores=FULL_S2, hook_strength=5.0,
                                comprehensibility=5.0, visual_action=10.0)
    assert hook_up.overall > motion_up.overall


def test_scores_are_clamped_to_the_reported_range():
    card = build_scorecard(s2_scores={"boundary": 5.0},   # out of range input
                           hook_strength=99.0)
    for d in card.dimensions:
        assert 0.0 <= d.score <= 100.0


def test_serialisation_is_dashboard_shaped():
    card = build_scorecard(s2_scores=FULL_S2, hook_strength=9.0,
                           comprehensibility=8.0, visual_action=7.0)
    blob = card.as_dict()
    assert {"overall", "grade", "source", "dimensions"} <= set(blob)
    for d in blob["dimensions"]:
        assert {"name", "score", "grade", "inputs", "model_judged",
                "note"} <= set(d)
        assert isinstance(d["inputs"], list)
