"""Filler-word removal, pinned.

The danger here is not leaving a filler in — it is cutting speech. Every
test below is about NOT mangling the sentence: substring matches that gut
real words, contextual fillers that carry meaning, and micro-cuts that
click without shortening anything.
"""

from __future__ import annotations

import pytest

from clipforge.pacing import (FILLER_PHRASES, FILLER_WORDS, MIN_FILLER_S,
                              find_filler_spans, subtract_spans)


def _words(*spec):
    """(text, start, end) at 0.4s per word unless a gap is given."""
    out, t = [], 0.0
    for item in spec:
        if isinstance(item, (int, float)):      # a pause
            t += float(item)
            continue
        out.append((item, t, t + 0.4))
        t += 0.4
    return out


# ------------------------------------------------------- what it cuts

def test_a_plain_filler_is_cut():
    words = _words("i", "um", "quit")
    spans = find_filler_spans(words)
    assert len(spans) == 1
    assert spans[0] == pytest.approx((0.4, 0.8))


def test_a_filler_phrase_is_cut_as_one_unit():
    """'you know' must go whole. Matching single words first would cut
    'know' and leave a dangling 'you'."""
    words = _words("it", "was", "you", "know", "hard")
    spans = find_filler_spans(words)
    assert len(spans) == 1
    assert spans[0][0] == pytest.approx(0.8)
    assert spans[0][1] == pytest.approx(1.6)


def test_a_contextual_filler_is_cut_mid_sentence():
    words = _words("i", "was", "like", "done")
    assert len(find_filler_spans(words)) == 1


# ---------------------------------------------------- what it leaves

def test_a_real_word_containing_a_filler_is_untouched():
    """Substring matching would gut the transcript: 'umbrella' contains
    'um', 'sold' contains 'so'."""
    words = _words("the", "umbrella", "sold", "well")
    spans = find_filler_spans(words)
    cut_starts = {round(s, 2) for s, _ in spans}
    assert 0.4 not in cut_starts, "'umbrella' was cut as 'um'"
    assert 0.8 not in cut_starts, "'sold' was cut as 'so'"


def test_a_contextual_filler_opening_a_sentence_survives():
    """Cutting the word that opens a sentence leaves a clip starting
    mid-thought — worse than leaving the filler in."""
    words = _words("done", 1.5, "so", "i", "quit")
    spans = find_filler_spans(words)
    assert spans == [], f"cut a sentence-opening word: {spans}"


def test_aggressive_mode_will_cut_it_when_asked():
    words = _words("done", 1.5, "so", "i", "quit")
    assert len(find_filler_spans(words, aggressive=True)) == 1


def test_a_filler_shorter_than_the_noise_floor_is_left_alone():
    """Below alignment precision, cutting produces a click and saves
    nothing a viewer perceives."""
    words = [("um", 0.0, MIN_FILLER_S / 2)]
    assert find_filler_spans(words) == []


def test_an_empty_transcript_is_handled():
    assert find_filler_spans([]) == []


# ------------------------------------------------- subtracting spans

def test_cuts_are_removed_from_the_keep_intervals():
    keeps = [(0.0, 10.0)]
    out = subtract_spans(keeps, [(4.0, 5.0)], pad=0.0)
    assert out == [(0.0, 4.0), (5.0, 10.0)]


def test_slivers_between_adjacent_cuts_are_dropped():
    """A 20ms fragment between two cuts is a click, not speech."""
    keeps = [(0.0, 10.0)]
    out = subtract_spans(keeps, [(4.0, 5.0), (5.02, 6.0)], pad=0.0,
                         min_keep=0.06)
    assert all(b - a >= 0.06 for a, b in out)
    assert not any(abs(a - 5.0) < 0.01 for a, b in out)


def test_a_cut_spanning_a_whole_keep_removes_it():
    out = subtract_spans([(2.0, 3.0)], [(1.0, 5.0)], pad=0.0)
    assert out == []


def test_cuts_outside_the_keeps_change_nothing():
    keeps = [(0.0, 5.0)]
    assert subtract_spans(keeps, [(8.0, 9.0)], pad=0.0) == keeps


def test_padding_shrinks_the_cut_not_the_speech():
    """The edit should land inside the filler, not on the neighbouring
    word's attack."""
    out = subtract_spans([(0.0, 10.0)], [(4.0, 5.0)], pad=0.05)
    assert out[0][1] == pytest.approx(4.05)
    assert out[1][0] == pytest.approx(4.95)


def test_multiple_cuts_across_multiple_keeps():
    keeps = [(0.0, 5.0), (10.0, 15.0)]
    out = subtract_spans(keeps, [(1.0, 2.0), (12.0, 13.0)], pad=0.0)
    assert out == [(0.0, 1.0), (2.0, 5.0), (10.0, 12.0), (13.0, 15.0)]


def test_the_word_lists_are_lowercase_and_alpha():
    """_norm strips to lowercase letters, so an entry with punctuation or
    capitals could never match anything."""
    for w in FILLER_WORDS:
        assert w == w.lower() and w.isalpha(), w
    for phrase in FILLER_PHRASES:
        for token in phrase:
            assert token == token.lower() and token.isalpha(), token
