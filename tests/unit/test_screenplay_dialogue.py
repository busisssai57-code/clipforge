"""Dialogue has to reach the output, not just the parser.

`screenplay.py` excludes dialogue from the video prompt on purpose -- it
is what a character says, not what the camera sees, and feeding spoken
lines to a video model puts subtitles and mouth-shaped artefacts in the
frame. Its docstring says the line "travels separately, to the voice".

There was no voice. Nothing outside screenplay.py read `Shot.spoken()`,
`to_beats_with_marks` returned only (beat, marks), and the generate
manifest carried neither. For a Somali sketch whose punchline IS a line --
a toddler calling a goat "Taksi!" -- the joke reached the audience in no
form at all. These tests pin the road it now travels, which is the same
one the emoji punchline marks already used.
"""

from __future__ import annotations

from clipforge.screenplay import to_beats_with_dialogue, to_beats_with_marks

NL = chr(10)


def _script(*lines: str) -> str:
    return NL.join(lines)


ONE_LINE = _script(
    "EXT. COURTYARD - DAY", "", "A toddler looks up.", "",
    "ILMAHA", "Taksi!", "",
    "EXT. COURTYARD - DAY", "", "The goat chews.", "",
)


def test_dialogue_travels_with_its_beat():
    triples = to_beats_with_dialogue(ONE_LINE)
    assert len(triples) == 2
    assert triples[0][2] == "Taksi!", "the line must ride with its own beat"
    assert triples[1][2] == "", "a scene with no dialogue carries none"


def test_the_line_is_still_absent_from_the_beat():
    """The exclusion that made this worth carrying separately."""
    triples = to_beats_with_dialogue(ONE_LINE)
    assert "Taksi!" not in triples[0][0]
    assert "ILMAHA" not in triples[0][0]


def test_merged_scenes_join_their_lines():
    text = _script(
        "EXT. A - DAY", "", "One.", "", "HOOYO", "Maxaad maqashay?", "",
        "EXT. B - DAY", "", "Two.", "", "ILMAHA", "Taksi!", "",
    )
    merged = to_beats_with_dialogue(text, 1)
    assert merged[0][2] == "Maxaad maqashay? Taksi!"


def test_a_repeated_tail_beat_carries_no_line():
    """One line said once is the joke; four times it is a stutter -- the
    same rule the emoji marks already follow."""
    text = _script("EXT. A - DAY", "", "One.", "", "ILMAHA", "Taksi!", "")
    out = to_beats_with_dialogue(text, 3)
    assert out[0][2] == "Taksi!"
    assert [t for _, _, t in out[1:]] == ["", ""]


def test_the_pair_api_delegates_to_one_distribution():
    """`to_beats_with_marks` now delegates. Two copies of the merge
    arithmetic would drift, and the symptom would be a punchline stamped
    on the wrong shot -- which its own docstring warns about."""
    for n in (1, 2, 5):
        pairs = to_beats_with_marks(ONE_LINE, n)
        triples = to_beats_with_dialogue(ONE_LINE, n)
        assert pairs == [(b, m) for b, m, _ in triples]


def test_the_shot_outcome_carries_the_line():
    """The end of the road: what the post layer would read."""
    from clipforge.genvideo.router import ShotOutcome

    out = ShotOutcome(0, "p", None, "prompt", 1.9, spoken="Taksi!")
    assert out.spoken == "Taksi!"
    assert ShotOutcome(1, "p", None, "prompt", 1.9).spoken == ""
