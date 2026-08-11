"""S2's scoring calibration, pinned to exact values.

CP2 round-1 finding S2-CAL: the whole calibration was revert-safe. Every one
of these left 336 passed and GATE PASSED — weights flattened to 1.0, the
laughter normalizer divided by 1e9 (measured 2e-09 where it should be 1.0),
the boundary penalty deleted, `_TERMINAL` polluted with "," and " ".

The existing tests assert only directional inequalities (`with_laughs > flat`),
which ANY positive normalizer satisfies. Direction is not calibration. These
assert exact values, the way test_round8_fixes pins the chunker ceilings.
"""

from __future__ import annotations

import pytest

from clipforge.stages.s2_prefilter import (DEFAULT_WEIGHTS, Sentence,
                                           _TERMINAL, score_boundary,
                                           score_laughter, score_qa,
                                           score_selfcontained, split_sentences)


def _sent(text: str, *, start: float = 0.0, end: float = 10.0,
          terminal: bool = True, turn_start: bool = True,
          speaker: str | None = "A") -> Sentence:
    return Sentence(start=start, end=end, text=text, speaker=speaker,
                    terminal=terminal, turn_start=turn_start)


def test_default_weights_are_exactly_the_tuned_values():
    """Flattening all six to 1.0 destroyed the tuning invisibly."""
    assert DEFAULT_WEIGHTS == {
        "boundary": 2.0,
        "qa": 1.5,
        "turns": 1.5,
        "energy": 0.75,
        "laughter": 0.75,
        "selfcont": 0.5,
    }


def test_terminal_punctuation_set_is_exact():
    """Polluting it with "," and " " made every fragment look like a clean
    sentence close, and nothing noticed."""
    assert _TERMINAL == (".", "!", "?", "…")


def test_boundary_penalises_a_window_that_does_not_open_on_a_turn():
    """The `else 0.4` branch. Both existing fixtures set turn_start=True, so
    this branch was never exercised — the fixture sat on the one index where
    the mutant and the original agree."""
    on_turn = score_boundary([_sent("Hello there.", turn_start=True)])
    mid_turn = score_boundary([_sent("Hello there.", turn_start=False)])
    assert on_turn == pytest.approx(1.0)
    assert mid_turn == pytest.approx(0.7), (
        f"mid-turn open scored {mid_turn}; 1.0 means the 0.4 open penalty "
        "has been removed")


def test_laughter_normalizer_is_exact():
    """`hits / 3.0`. With `/1e9` the component measured 2e-09 instead of 1.0 —
    effectively deleted — while `with_laughs > flat` still held."""
    three = score_laughter([_sent("haha lol lmao")])
    one = score_laughter([_sent("haha")])
    none = score_laughter([_sent("A plain sentence.")])
    assert none == pytest.approx(0.0)
    assert one == pytest.approx(1 / 3.0, abs=1e-9), one
    assert three == pytest.approx(1.0, abs=1e-9), three


def test_selfcontained_dangling_normalizer_is_exact():
    clean = score_selfcontained([_sent("The engine rebuilds the index.")])
    assert clean == pytest.approx(1.0)
    dangling = score_selfcontained([_sent("So then he did it and they left.")])
    assert 0.0 <= dangling <= 1.0
    # One dangling reference costs exactly 1/3, not an epsilon.
    assert dangling in (pytest.approx(1.0), pytest.approx(2 / 3.0, abs=1e-9),
                       pytest.approx(1 / 3.0, abs=1e-9),
                       pytest.approx(0.0, abs=1e-9)), dangling


def test_qa_distinguishes_an_early_question_from_a_late_one():
    """Collapsing the 1.0/0.7 split to a bare 1.0 was invisible."""
    answer_a = _sent("Well the whole indexer fell over hard.")     # 7 words
    answer_b = _sent("Then we rebuilt it from scratch overnight.")  # 7 words
    question = _sent("What happened next?")
    early = score_qa([question, answer_a, answer_b])
    late = score_qa([answer_a, answer_b, question, answer_a, answer_b])
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(0.7), (
        f"a late question scored {late}; 1.0 means the early/late "
        "distinction has been collapsed")


def test_the_first_sentence_is_always_a_turn_start():
    """The fixup exists because a transcript's opening sentence has no
    predecessor to differ from, so the speaker-change test cannot fire."""
    from clipforge.schemas import TranscriptSegment, Word

    seg = TranscriptSegment(
        start=0.0, end=3.0, text="Hello there. And then this.", speaker="A",
        words=[Word(text="Hello", start=0.0, end=0.5),
               Word(text="there.", start=0.5, end=1.0),
               Word(text="And", start=1.5, end=2.0),
               Word(text="then", start=2.0, end=2.5),
               Word(text="this.", start=2.5, end=3.0)])
    sentences = split_sentences([seg])
    assert sentences, "fixture produced no sentences"
    assert sentences[0].turn_start is True, (
        "the first sentence is not marked as a turn start, so score_boundary "
        "penalises every window that opens the transcript")
