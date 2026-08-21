"""Screenplay blocks are the beats — no guessing from punctuation.

`split_into_beats` cuts a brief on sentence boundaries, which predicts
where a shot starts only for a one-line idea. The moment a brief contains
dialogue or a scene change, full stops stop meaning anything about shots.

The traps here are the ones that make a parser quietly wrong rather than
loudly broken: an all-caps ACTION line read as a character cue (which then
eats the next line as dialogue), action after a speech still attributed to
the speaker, and dialogue leaking into the video prompt — where it renders
as subtitles and mouth artefacts instead of being spoken.
"""

from __future__ import annotations

import pytest

from clipforge.screenplay import (parse, summary, title_page, to_beats,
                                  to_beats_with_marks, to_shots)


def _types(text):
    return [b.type for b in parse(text)]


# ------------------------------------------------------------ block types

def test_a_slugline_is_a_heading():
    assert _types("INT. KITCHEN - DAY") == ["heading"]


@pytest.mark.parametrize("slug", [
    "INT. BAR - NIGHT", "EXT. STREET - DAY", "EST. CITY - DAWN",
    "I/E. CAR - CONTINUOUS", "int. lowercase also works - day",
])
def test_every_slugline_form_is_recognised(slug):
    assert parse(slug)[0].type == "heading"


def test_prose_is_action():
    assert _types("A man walks in and sits down.") == ["action"]


def test_a_cue_followed_by_a_line_is_a_character():
    assert _types("BOB\nHello there.") == ["character", "dialogue"]


def test_an_all_caps_action_line_is_not_a_character():
    """The trap. 'THE DOOR SLAMS' is upper-case but nobody is speaking —
    reading it as a cue silently turns the NEXT line of action into
    dialogue, and the shot loses a description while gaining a speech."""
    assert _types("THE DOOR SLAMS.\n\nHe turns around.") == ["action", "action"]


def test_a_cue_with_nothing_under_it_is_action():
    assert _types("BOB") == ["action"]


def test_a_parenthetical_belongs_to_the_speaker():
    blocks = parse("BOB\n(quietly)\nI'm leaving.")
    assert [b.type for b in blocks] == ["character", "parenthetical",
                                        "dialogue"]
    assert blocks[1].character == "BOB"


def test_a_blank_line_ends_the_speech():
    """Without this, action after dialogue is attributed to whoever spoke
    last and disappears from the shot description."""
    blocks = parse("BOB\nHello.\n\nHe leaves the room.")
    assert blocks[-1].type == "action"
    assert blocks[-1].character is None


def test_a_transition_is_its_own_type():
    assert parse("CUT TO:")[0].type == "transition"


def test_a_forced_heading_uses_a_leading_dot():
    b = parse(".A CLOSE ON THE GLASS")[0]
    assert b.type == "heading"
    assert b.text == "A CLOSE ON THE GLASS"


def test_a_forced_transition_uses_a_leading_angle():
    b = parse("> SMASH CUT:")[0]
    assert b.type == "transition"


def test_a_character_extension_is_stripped_from_the_name():
    """BOB (V.O.) and BOB are the same person; the voice must not split
    into two speakers because of a parenthetical."""
    blocks = parse("BOB (V.O.)\nI remember it well.")
    assert blocks[1].character == "BOB"


def test_empty_input_yields_nothing():
    assert parse("") == []
    assert parse("   \n\n  ") == []


# ------------------------------------------------------------------ shots

SCRIPT = """INT. TEA SHOP - NIGHT

Smoke hangs under a dead fluorescent tube.

WAXAR
You never pay.

GEEL
(defensive)
I paid last year.

EXT. STREET - CONTINUOUS

He storms out into the rain.
"""


def test_each_heading_starts_a_shot():
    assert len(to_shots(parse(SCRIPT))) == 2


def test_action_lands_on_the_shot_it_sits_under():
    shots = to_shots(parse(SCRIPT))
    assert "Smoke hangs" in shots[0].beat()
    assert "storms out" in shots[1].beat()


def test_dialogue_is_attributed_and_ordered():
    shots = to_shots(parse(SCRIPT))
    assert shots[0].dialogue == [("WAXAR", "You never pay."),
                                 ("GEEL", "I paid last year.")]


def test_dialogue_never_reaches_the_video_prompt():
    """It is what the characters SAY, not what the camera sees. Feeding
    spoken lines to a video model puts subtitles and mouth-shaped
    artefacts in the frame; the words travel to the voice instead."""
    beat = to_shots(parse(SCRIPT))[0].beat()
    assert "You never pay" not in beat
    assert "I paid last year" not in beat


def test_a_script_with_no_heading_is_still_one_shot():
    """Three action lines and no slugline is a scene. Dropping it for
    lack of a heading would silently discard the whole brief."""
    shots = to_shots(parse("A goat stares at the camera.\nIt blinks."))
    assert len(shots) == 1
    assert "stares" in shots[0].beat()


def test_a_transition_is_not_a_shot():
    assert len(to_shots(parse("INT. A - DAY\nX happens.\n\nCUT TO:"))) == 1


def test_a_cue_with_no_picture_does_not_create_an_empty_shot():
    shots = to_shots(parse("BOB\nJust talking."))
    assert len(shots) == 1
    assert shots[0].dialogue


# ------------------------------------------------------------------ beats

def test_beats_follow_the_scenes_by_default():
    assert len(to_beats(SCRIPT)) == 2


def test_fewer_shots_merges_rather_than_truncates():
    """Truncating would silently lose the ending of the script."""
    beats = to_beats(SCRIPT, 1)
    assert len(beats) == 1
    assert "Smoke hangs" in beats[0] and "storms out" in beats[0]


def test_more_shots_repeats_rather_than_inventing():
    beats = to_beats(SCRIPT, 5)
    assert len(beats) == 5
    assert beats[-1] == beats[1]


def test_an_empty_script_produces_no_beats():
    assert to_beats("") == []


def test_beats_never_contain_dialogue_at_any_shot_count():
    for n in (1, 2, 5):
        assert all("You never pay" not in b for b in to_beats(SCRIPT, n))


# ---------------------------------------------------------------- summary

def test_the_summary_counts_what_the_editor_shows():
    s = summary(SCRIPT)
    assert s["shot_count"] == 2
    assert s["speakers"] == ["GEEL", "WAXAR"]
    assert s["counts"]["heading"] == 2
    assert s["word_count"] > 0


def test_the_summary_survives_junk():
    for junk in ("", "   ", "\n\n\n", "()", "...", "123"):
        assert summary(junk)["shot_count"] >= 0


# ---- the title page and the writer's notes (2026-08-18) -----------------

def test_a_title_page_is_not_a_shot():
    """A real screenplay opens with `Title:` / `Credit:` lines. Parsed as
    action they became shot 0 — a generated picture of the file's own
    header, paid for out of the shot budget."""
    text = ("Title: Geel Suuqa Tegey\nCredit: BTA\n\n"
            "EXT. SUUQ - SUBAX\n\nGeel dheer oo khudaar eegaya.\n")
    beats = to_beats(text)
    assert len(beats) == 1
    assert "Title:" not in beats[0]


def test_the_title_page_is_readable_on_its_own():
    text = "Title: X\nHook: INTAAN MIDKEE KAA QOSLIYAY\n\nEXT. A - DAY\n\nGeel.\n"
    assert title_page(text)["hook"] == "INTAAN MIDKEE KAA QOSLIYAY"


def test_a_colon_line_with_no_canonical_key_is_not_a_title_page():
    """`GEEL: waa imisa` at the top of a file is the writer's first line,
    not a header. A made-up key is legal Fountain only ALONGSIDE a real
    one, so a lone unknown key must not swallow the line."""
    assert title_page("GEEL: waa imisa") == {}
    assert to_beats("GEEL: waa imisa", 1) == ["GEEL: waa imisa"]


def test_a_screenplay_that_opens_on_action_has_no_title_page():
    assert title_page("EXT. SUUQ - DAY\n\nGeel.\n") == {}


def test_line_numbers_still_point_at_the_writers_editor():
    """The title page is blanked, not sliced: a block's line_no is what a
    future editor gutter will use, and a shifted one points at the wrong
    line in the file the writer is looking at."""
    text = "Title: X\n\nEXT. SUUQ - DAY\n\nGeel dheer.\n"
    blocks = parse(text)
    assert blocks[0].type == "heading"
    assert blocks[0].line_no == 2


def test_marks_travel_with_their_beat_when_scenes_are_merged():
    """The mark and the beat are redistributed together. Matching them up
    by index afterwards needs a second copy of this distribution, and the
    symptom of the two drifting is a punchline on the wrong shot."""
    text = ("EXT. A - DAY\n\nOne. [[\U0001F602]]\n\n"
            "EXT. B - DAY\n\nTwo.\n\n"
            "EXT. C - DAY\n\nThree. [[\U0001F480]]\n\n"
            "EXT. D - DAY\n\nFour.\n")
    pairs = to_beats_with_marks(text, 2)
    assert len(pairs) == 2
    assert pairs[0][1] == ["\U0001F602"]
    assert pairs[1][1] == ["\U0001F480"]


def test_a_repeated_beat_does_not_repeat_its_punchline():
    """More shots than scenes repeats the last beat. Stamping the same
    emoji on all of them is how one joke becomes four."""
    text = f"EXT. A - DAY\n\nOne. [[\U0001F602]]\n"
    pairs = to_beats_with_marks(text, 3)
    assert [m for _b, m in pairs] == [["\U0001F602"], [], []]
