"""Burning the script's dialogue into the picture.

The lines reached `sequence.srt` and stopped there. A sidecar is not a
subtitle anyone watching a TikTok sees -- nothing in that player will ever
load it -- so for a sketch whose punchline IS a line, the joke was still
not being told. These tests pin the burn.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from clipforge.socialpost import (PostSpec, Subtitle, build_overlay,
                                  spec_from_shots)

SHOTS = [1.9, 1.9, 1.9]
NO_MARKS = [[], [], []]


def test_a_line_is_held_for_its_own_shot_and_no_longer():
    spec = spec_from_shots(SHOTS, NO_MARKS,
                           spoken=["Maxaad maqashay?", "", "Taksi!"])
    assert [s.text for s in spec.subtitles] == ["Maxaad maqashay?", "Taksi!"]
    first, second = spec.subtitles
    assert (first.start, round(first.end, 1)) == (0.0, 1.9)
    assert (round(second.start, 1), round(second.end, 1)) == (3.8, 5.7)


def test_a_silent_beat_produces_no_subtitle():
    spec = spec_from_shots(SHOTS, NO_MARKS, spoken=["", "  ", ""])
    assert spec.subtitles == ()


def test_no_spoken_argument_is_the_old_behaviour():
    """Prose briefs have no dialogue and must not gain an empty band."""
    assert spec_from_shots(SHOTS, NO_MARKS).subtitles == ()


def _graph(spec):
    return build_overlay(spec, width=1080, height=1920,
                         work_dir=Path(tempfile.mkdtemp()))[1]


def test_the_subtitle_sits_below_the_sticker_band_and_the_face():
    """Placement is the whole design. The sticker slots occupy 0.34-0.66
    and ari_bridge reserves 750-1050 of a 1920 frame (0.39-0.55) for a
    face, so a subtitle any higher lands on the joke or on the child."""
    spec = spec_from_shots(SHOTS, NO_MARKS, spoken=["Taksi!", "", ""])
    assert "y=H*0.845-h/2" in _graph(spec)


def test_the_handle_is_drawn_after_the_subtitle():
    """A handle that disappears under a subtitle is not a handle."""
    spec = spec_from_shots(SHOTS, NO_MARKS, spoken=["Taksi!", "", ""],
                           watermark="@ari")
    steps = _graph(spec).split(";")
    sub_at = next(i for i, s in enumerate(steps) if "0.845" in s)
    mark_at = next(i for i, s in enumerate(steps) if "H-h-H*0.035" in s)
    assert sub_at < mark_at


def test_each_line_gets_its_own_time_window():
    spec = spec_from_shots(SHOTS, NO_MARKS, spoken=["one", "", "two"])
    graph = _graph(spec)
    assert "enable='between(t,0.000,1.900)'" in graph
    assert "enable='between(t,3.800,5.700)'" in graph


def test_a_spec_with_no_subtitles_draws_none():
    graph = _graph(PostSpec(hook="HOOK"))
    assert "0.845" not in graph


def test_generate_hands_the_lines_to_the_post_layer():
    """Wiring. The burn is worthless if the lines never arrive.

    Read with getattr: the post layer is handed shot-shaped objects from
    more than one place, and it must not refuse to stamp a hook because a
    caller's shot predates the dialogue field.
    """
    import inspect

    from clipforge import cli

    assert 'spoken=[getattr(s, "spoken", "") for s in ok]' in inspect.getsource(cli)
