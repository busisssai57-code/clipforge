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


# ---------------------------------------------- the line reaching a viewer

def _shot(i, spoken="", seconds=1.9, path="x.mp4"):
    from pathlib import Path as _P

    from clipforge.genvideo.router import ShotOutcome

    return ShotOutcome(i, "p", _P(path) if path else None, "prompt",
                       seconds, spoken=spoken)


def test_srt_times_each_line_to_its_own_shot(tmp_path):
    from clipforge.screenplay import write_dialogue_srt

    dest = write_dialogue_srt(
        [_shot(0, "Maxaad maqashay?"), _shot(1), _shot(2, "Taksi!")],
        tmp_path / "s.srt")
    body = dest.read_text(encoding="utf-8")
    assert "00:00:00,000 --> 00:00:01,900" in body
    assert "Maxaad maqashay?" in body
    # shot 2 starts after two 1.9s shots, whether or not shot 1 spoke
    assert "00:00:03,800 --> 00:00:05,700" in body
    assert "Taksi!" in body


def test_cues_renumber_rather_than_skipping(tmp_path):
    """A silent beat must not leave a hole in the numbering."""
    from clipforge.screenplay import write_dialogue_srt

    body = write_dialogue_srt(
        [_shot(0, "one"), _shot(1), _shot(2, "two")],
        tmp_path / "s.srt").read_text(encoding="utf-8")
    numbers = [ln for ln in body.split(chr(10)) if ln.strip().isdigit()]
    assert numbers == ["1", "2"]


def test_no_file_when_nothing_is_said(tmp_path):
    """An empty .srt claims dialogue and shows none -- the same shape of
    lie as an audio stream with silence on it."""
    from clipforge.screenplay import write_dialogue_srt

    dest = tmp_path / "s.srt"
    assert write_dialogue_srt([_shot(0), _shot(1, "   ")], dest) is None
    assert not dest.exists()


def test_a_failed_shot_contributes_no_time_and_no_line(tmp_path):
    """It produced no picture, so the piece is shorter and every later
    cue moves up. Timing off by a missing beat is a subtitle drifting out
    of sync for the rest of the video."""
    from clipforge.screenplay import write_dialogue_srt

    body = write_dialogue_srt(
        [_shot(0, "one"), _shot(1, "lost", path=None), _shot(2, "two")],
        tmp_path / "s.srt").read_text(encoding="utf-8")
    assert "lost" not in body
    assert "00:00:01,900 --> 00:00:03,800" in body, (
        "the surviving third beat must start at 1.9s, not 3.8s")


def test_the_generate_command_writes_it():
    """Wiring, not formatting. The road existed and stopped one step
    short before."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli)
    assert "write_dialogue_srt(result.shots" in src
    assert "sequence.srt" in src
