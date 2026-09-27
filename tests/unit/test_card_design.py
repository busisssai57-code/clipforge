"""The hook card, measured from the pixels it actually draws.

The operator's verdict on the first version was "weak and not attention
grabbing", and it was right: white caps with a thin outline at 6.2% of
frame height, sitting straight on the picture. An outline survives a dark
background and vanishes into a bright one, and the words were too small
to register in the half-second a thumb gives them while scrolling.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.socialpost import (ACCENT, PostSpec, balanced_wrap, find_font,
                                  render_card)

pytestmark = pytest.mark.skipif(find_font(("arialbd.ttf", "seguibl.ttf",
                                           "segoeuib.ttf", "arial.ttf",
                                           "DejaVuSans-Bold.ttf")) is None,
                                reason="no text font on this machine")


def card(tmp_path, text="she tried it for 6 weeks", *, width=1080,
         size=163, **kw):
    from PIL import Image

    path = render_card(text, width=width, size_px=size,
                       dest=tmp_path / "card.png", **kw)
    return Image.open(path).convert("RGBA")


# ------------------------------------------------------------ the contrast

def test_the_words_sit_on_a_plate_not_bare_on_the_picture(tmp_path):
    """Contrast that does not depend on what the footage is doing. An
    outline alone disappears against a bright frame — which is most UGC,
    shot indoors by a window."""
    im = card(tmp_path)
    opaque_dark = sum(1 for px in im.getdata()
                      if px[3] > 180 and max(px[:3]) < 60)
    assert opaque_dark > 40_000, (
        "no solid backing behind the words; on bright footage this is "
        "white-on-white")


def test_the_last_line_carries_the_accent(tmp_path):
    """The eye lands where the colour is, and the payoff of a hook is its
    last words."""
    im = card(tmp_path)
    accent_px = sum(1 for px in im.getdata()
                    if px[3] > 180 and abs(px[0] - ACCENT[0]) < 30
                    and abs(px[1] - ACCENT[1]) < 40 and px[2] < 80)
    assert accent_px > 2_000, "the payoff line is not picked out"


def test_a_single_line_hook_stays_white(tmp_path):
    """With one line there is no payoff to separate, and a fully coloured
    card reads as a warning label. (A short hook at the default size
    still wraps to two lines, so this pins the one-line case with a size
    that genuinely fits.)"""
    im = card(tmp_path, "wait for it", size=70)
    accent_px = sum(1 for px in im.getdata()
                    if px[3] > 180 and abs(px[0] - ACCENT[0]) < 30
                    and abs(px[1] - ACCENT[1]) < 40 and px[2] < 80)
    assert accent_px == 0


def test_the_plate_can_be_turned_off(tmp_path):
    im = card(tmp_path, plate=False)
    opaque_dark = sum(1 for px in im.getdata()
                      if px[3] > 180 and max(px[:3]) < 60)
    bare = card(tmp_path, plate=True)
    assert opaque_dark < sum(1 for px in bare.getdata()
                             if px[3] > 180 and max(px[:3]) < 60)


# --------------------------------------------------------------- the size

def test_the_default_is_big_enough_to_read_while_scrolling():
    assert PostSpec().hook_size >= 0.08, (
        "6.2% of frame height was measured too small on a phone")


def test_the_card_never_runs_off_the_frame(tmp_path):
    im = card(tmp_path, "an unusually long hook line that will not fit on "
                        "one single line at this size")
    assert im.width == 1080
    assert im.height < 1920 * 0.45, "the card is eating half the frame"


def test_a_long_hook_shrinks_rather_than_stacking_four_lines(tmp_path):
    """Four lines of caps is a paragraph, not a hook."""
    short = card(tmp_path, "wait for it")
    long = card(tmp_path, "she tried this every single night for six whole weeks")
    # Each line adds height; capping at three lines caps the height.
    assert long.height <= short.height * 3.4


# ------------------------------------------------------------ the wrapping

def test_balanced_differs_from_greedy_where_it_matters():
    """A case where greedy leaves a stub: it fills line one to the margin
    and drops the remainder alone underneath."""
    text = "aaaa bb cc dd"
    greedy = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if len(trial) <= 10 or not current:
            current = trial
        else:
            greedy.append(current)
            current = word
    greedy.append(current)
    assert greedy == ["aaaa bb cc", "dd"], greedy

    got = balanced_wrap(text, len, 10.0)
    assert got != greedy, "balancing did nothing"
    assert max(len(x) for x in got) - min(len(x) for x in got) < 8, got


def test_lines_are_balanced_not_greedy():
    """Greedy filling left SHE / TRIED IT / FOR 6 / WEEKS on a real clip:
    four separate thoughts instead of one block."""
    words = "she tried it for 6 weeks"
    # A measure where every character is one unit, and a line fits 12.
    lines = balanced_wrap(words, len, 12.0)
    assert len(lines) <= 3
    widths = [len(line) for line in lines]
    assert max(widths) - min(widths) <= 4, lines


def test_balancing_never_adds_a_line():
    """It moves words between the lines greedy already needed; a taller
    card is a different card."""
    text = "one two three four five six seven eight"
    greedy_lines = 0
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if len(trial) <= 15 or not current:
            current = trial
        else:
            greedy_lines += 1
            current = word
    greedy_lines += 1
    assert len(balanced_wrap(text, len, 15.0)) <= max(greedy_lines, 1)


def test_a_word_longer_than_the_line_still_renders(tmp_path):
    im = card(tmp_path, "Supercalifragilisticexpialidocious")
    assert im.width == 1080 and im.height > 0


def test_an_empty_hook_is_not_a_card(tmp_path):
    assert balanced_wrap("", len, 10.0) == []
