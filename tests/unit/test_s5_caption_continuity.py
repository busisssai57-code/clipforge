"""Pop captions must be CONTINUOUS across natural speech pauses.

Measured regression this pins: clamping each word event to the word's own
duration left 19 gaps totalling 15.3 s on a 59.5 s clip — 26% of the clip
with nothing on screen, largest hole 2.49 s. Captions that blink out
mid-sentence are the tell of automated captioning.

The assertions are on measured BLANK TIME, not on event count, because
"emits many events" is not the property that matters.
"""

from __future__ import annotations

import pytest

from clipforge.stages.s5_subtitles import _HOLD_MAX_S, _pop_events


def _secs(stamp: str) -> float:
    h, m, rest = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(rest)


def _spans(events: list[str]) -> list[tuple[float, float]]:
    out = []
    for line in events:
        p = line.split(",", 4)
        out.append((_secs(p[1]), _secs(p[2])))
    return sorted(out)


def _blank_time(events: list[str]) -> float:
    """Seconds between the first and last caption with nothing displayed."""
    spans = _spans(events)
    merged: list[list[float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return sum(merged[i + 1][0] - merged[i][1]
               for i in range(len(merged) - 1))


#: Words with deliberate PAUSES between them: 0.4 s, then 0.9 s, then 1.1 s.
#: Every one of these was a blank screen before the fix.
_PAUSED = [[(0.0, 0.5, "ONE"), (0.9, 1.4, "TWO")],
           [(2.3, 2.8, "THREE"), (3.9, 4.4, "FOUR")]]


def test_pauses_between_words_do_not_blank_the_screen():
    events = _pop_events(_PAUSED, 0.0, 4, 2, "&H0000FFFF", "&H007FFF00&")
    blank = _blank_time(events)
    assert blank == pytest.approx(0.0, abs=0.02), (
        f"{blank:.2f}s of blank screen between captions; pauses must be "
        "bridged by holding the line, not by hiding it")


def test_every_word_still_gets_its_own_pop():
    """Continuity must not be bought by merging words into one static line —
    the highlight has to keep moving per word."""
    events = _pop_events(_PAUSED, 0.0, 4, 2, "&H0000FFFF", "&H007FFF00&")
    assert len(events) == 4, f"expected one event per word, got {len(events)}"
    # Each event scales exactly one word.
    for ev in events:
        assert ev.count("\\fscx118") == 1, ev


def test_a_long_silence_does_clear_the_screen():
    """The hold is capped: stale text sitting through a 5 s silence is worse
    than a clean screen."""
    groups = [[(0.0, 0.5, "HELLO")], [(9.0, 9.5, "AGAIN")]]
    events = _pop_events(groups, 0.0, 4, 2, "&H0000FFFF", "&H007FFF00&")
    first_end = _spans(events)[0][1]
    assert first_end <= 0.5 + _HOLD_MAX_S + 0.01, (
        f"first caption held until {first_end}s — a long silence must clear")
    assert _blank_time(events) > 1.0, "an 8.5s silence should show a gap"


def test_captions_never_overlap_each_other():
    """Two events on screen at once would double-render the line."""
    events = _pop_events(_PAUSED, 0.0, 4, 2, "&H0000FFFF", "&H007FFF00&")
    spans = _spans(events)
    for (s1, e1), (s2, _e2) in zip(spans, spans[1:]):
        assert e1 <= s2 + 1e-6, f"event [{s1},{e1}] overlaps next start {s2}"


def test_the_final_word_does_not_vanish_instantly():
    groups = [[(0.0, 0.4, "END")]]
    events = _pop_events(groups, 0.0, 4, 2, "&H0000FFFF", "&H007FFF00&")
    s, e = _spans(events)[0]
    assert e - s >= 0.6, (
        f"last word shown for only {e - s:.2f}s; it needs a readable tail")


# --- Phrase-boundary grouping -------------------------------------------
#
# The sketch sliced the word list every `per_line * max_lines` entries, so a
# caption broke wherever the counter ran out. A real run burned these onto a
# clip: "THEY ASKED MONICA TO SEE IT. ACCORDING TO", "WE BRAINSTORM ALL",
# "THEN KEEP IT A STRAIGHT UP SECRET. [emoji] WHEN". Every one ends mid-clause.

from clipforge.stages.s5_subtitles import (  # noqa: E402
    _group_words, _phrase_break,
)

_DANGLING = {
    "to", "of", "and", "or", "but", "the", "a", "an", "for", "with", "at",
    "in", "on", "from", "by", "as", "that", "when", "if", "all", "is", "was",
}


def _lay_out(sentence: str, *, gap: float = 0.02):
    """Words on a steady cadence, so only punctuation can force a break."""
    words, flags, t = [], [], 0.0
    for tok in sentence.split():
        words.append((t, t + 0.28, tok.upper()))
        flags.append(_phrase_break(tok))
        t += 0.28 + gap
    return words, flags


def test_no_caption_ends_on_a_dangling_function_word():
    words, flags = _lay_out(
        "When they asked Monica to see it, according to August, none of "
        "them know. Then keep it a straight up secret."
    )
    for group in _group_words(words, flags, 4, 2):
        last = group[-1][2].rstrip(".,!?;:").lower()
        assert last not in _DANGLING, (
            f"caption ends on {last!r}: " + " ".join(w[2] for w in group)
        )


def test_a_sentence_end_always_closes_the_caption():
    words, flags = _lay_out("Keep it a secret. When they asked her to see it.")
    groups = _group_words(words, flags, 4, 2)
    first = " ".join(w[2] for w in groups[0])
    assert first.endswith("SECRET."), first


def test_a_breath_breaks_a_caption_that_punctuation_does_not():
    words, flags = _lay_out("we brainstorm all kinds of different outcomes")
    # Widen the gap after "all" — ASR dropped the comma the speaker implied.
    words[2] = (words[2][0], words[2][1] - 0.0, words[2][2])
    words = [w if i != 3 else (w[0] + 0.6, w[1] + 0.6, w[2])
             for i, w in enumerate(words)]
    groups = _group_words(words, flags, 4, 2)
    assert " ".join(w[2] for w in groups[0]) == "WE BRAINSTORM ALL"


def test_the_word_cap_is_still_a_ceiling():
    words, flags = _lay_out(" ".join(["word"] * 40))
    for group in _group_words(words, flags, 4, 2):
        assert len(group) <= 8


def test_a_break_never_strands_a_single_word():
    words, flags = _lay_out("Keep it a straight up secret when they asked.")
    for group in _group_words(words, flags, 4, 2):
        assert len(group) > 1, " ".join(w[2] for w in group)
