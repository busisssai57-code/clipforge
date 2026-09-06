"""A beat that contradicts its own niche wins, and does it silently.

The beat is prepended to the shot prompt and lands FIRST, so a stale
screenplay overrides every correction made in the niche. That cost four
rounds: `gen_avoid` carried "a woven mat", "a metal gate" and "a seated
child" while the script's own beat said "a Somali toddler in a bright
patterned shirt SITS ALONE ON A WOVEN MAT ... exterior, COURTYARD GATE".
Every render came back with a seated child on a mat at a gate, and the
prompt work looked ignored when it was being contradicted.

The division being restored: a BEAT says what HAPPENS, a niche says what
it LOOKS LIKE.
"""

from __future__ import annotations

from clipforge.genvideo.presets import beat_conflicts

AVOID = "a woven mat, a metal gate, a seated child, crowd, text, oversaturation"


def test_the_real_collision_that_cost_four_rounds():
    beat = ("exterior, sunlit courtyard, late afternoon. A Somali toddler "
            "in a bright patterned shirt sits alone on a woven mat.")
    assert "a woven mat" in beat_conflicts(beat, AVOID)


def test_a_leading_article_does_not_hide_the_clash():
    """The avoid list says "a woven mat"; a writer types "on a woven mat"
    or "the woven mat". Matching only the exact string misses both."""
    for beat in ("he sits on a woven mat", "the woven mat is dusty"):
        assert beat_conflicts(beat, AVOID), beat


def test_a_clean_beat_reports_nothing():
    assert beat_conflicts("He reaches up and points at the goat.", AVOID) == []


def test_single_words_are_ignored_to_avoid_crying_wolf():
    """"crowd", "text" and "oversaturation" collide with ordinary prose.
    A warning that fires on every beat teaches an operator to ignore it."""
    assert beat_conflicts("a crowd gathers and the text is read", AVOID) == []


def test_longest_match_first():
    """So the most specific conflict is the one an operator reads."""
    got = beat_conflicts("he sits on a woven mat by a metal gate", AVOID)
    assert got[0] == "a seated child" or len(got[0]) >= len(got[-1])


def test_empty_inputs_are_safe():
    assert beat_conflicts("", AVOID) == []
    assert beat_conflicts("anything", "") == []


def test_the_router_actually_warns():
    """A detector nothing calls is a comment."""
    import inspect

    from clipforge.genvideo import router

    src = inspect.getsource(router)
    assert "beat_conflicts(beats[i], preset.avoid)" in src
    assert "genvideo.beat_conflicts_niche" in src
