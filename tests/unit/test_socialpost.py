"""The post layer: hook card, punchline stickers, handle.

These are the parts of a short-form post that no model produces, and they
are all string-and-geometry work — the class of code that looks right in a
preview and is off by a frame or half a screen in the render.
"""

from __future__ import annotations

import pytest

from clipforge.socialpost import (PostError, PostSpec, Sticker, build_overlay,
                                  find_font, render_card, render_emoji,
                                  spec_from_shots, _EMOJI_FONTS, _TEXT_FONTS)

LAUGH = "\U0001F602"
SKULL = "\U0001F480"
CRY = "\U0001F62D"

needs_text_font = pytest.mark.skipif(
    find_font(_TEXT_FONTS) is None, reason="no text font on this machine")
needs_emoji_font = pytest.mark.skipif(
    find_font(_EMOJI_FONTS) is None, reason="no colour emoji font")


# --------------------------------------------------------------- marks ----









# ---------------------------------------------------------------- spec ----

def test_a_sticker_is_held_for_its_own_shot_only():
    spec = spec_from_shots([1.9, 1.9, 1.9], [[], [LAUGH], []])
    (st,) = spec.stickers
    assert (st.start, st.end) == (1.9, 3.8)


def test_sticker_times_follow_uneven_shot_lengths():
    """Shot lengths are not uniform once a provider trims or a beat runs
    long; timing the marks off a nominal length would drift the further
    into the piece you get."""
    spec = spec_from_shots([3.0, 1.2, 2.4], [[], [], [SKULL]])
    (st,) = spec.stickers
    assert (st.start, st.end) == (4.2, 6.6)


def test_marks_beyond_the_shot_count_are_ignored_not_crashed():
    spec = spec_from_shots([1.9], [[], [LAUGH]])
    assert spec.stickers == ()


def test_two_stickers_on_one_shot_do_not_stack_on_each_other():
    spec = spec_from_shots([2.0], [[SKULL, CRY]])
    _, graph = build_overlay(spec, width=720, height=960,
                             work_dir=_tmp("stack"))
    positions = [line.split("overlay=")[1].split(":enable")[0]
                 for line in graph.split(";")]
    assert positions[0] != positions[1], graph


def test_an_empty_spec_is_refused_rather_than_rendering_a_copy(tmp_path):
    with pytest.raises(PostError):
        from clipforge.socialpost import apply_post
        apply_post(tmp_path / "a.mp4", tmp_path / "b.mp4", PostSpec())


# -------------------------------------------------------------- graph ----

def _tmp(name: str):
    import tempfile
    from pathlib import Path
    return Path(tempfile.mkdtemp(prefix=f"post_{name}_"))


@needs_text_font
def test_the_hook_holds_only_over_the_opening():
    spec = PostSpec(hook="INTAAN MIDKEE KAA QOSLIYAY", hook_seconds=2.0)
    _, graph = build_overlay(spec, width=720, height=960, work_dir=_tmp("hook"))
    assert "enable='lt(t,2.000)'" in graph


@needs_text_font
def test_the_handle_is_last_and_never_switched_off():
    """A watermark drawn before a sticker is a watermark a sticker can
    cover, and one with an `enable` is one that blinks."""
    spec = PostSpec(watermark="@bta", stickers=(Sticker(LAUGH, 0.0, 1.0),))
    _, graph = build_overlay(spec, width=720, height=960, work_dir=_tmp("wm"))
    last = graph.split(";")[-1]
    assert last.endswith("[v]")
    assert "enable" not in last


@needs_text_font
def test_the_graph_ends_on_a_single_named_output():
    """ffmpeg maps [v]; a graph whose last label is [v3] fails at runtime
    with 'Output with label v not found', after the render has started."""
    spec = PostSpec(hook="HOOK", watermark="@bta",
                    stickers=(Sticker(LAUGH, 0.0, 1.0),))
    _, graph = build_overlay(spec, width=720, height=960, work_dir=_tmp("out"))
    assert graph.count("[v]") == 1
    assert graph.split(";")[-1].endswith("[v]")


@needs_text_font
def test_every_overlay_input_is_declared():
    """One -i per overlay, or the indices in the graph point at nothing."""
    spec = PostSpec(hook="HOOK", watermark="@bta",
                    stickers=(Sticker(LAUGH, 0.0, 1.0), Sticker(CRY, 1.0, 2.0,
                                                                index=1)))
    inputs, graph = build_overlay(spec, width=720, height=960,
                                  work_dir=_tmp("inputs"))
    assert inputs.count("-i") == len(graph.split(";"))


def test_a_spec_with_nothing_in_it_builds_no_graph():
    inputs, graph = build_overlay(PostSpec(), width=720, height=960,
                                  work_dir=_tmp("empty"))
    assert (inputs, graph) == ([], "")


# ------------------------------------------------------------ raster ----

@needs_text_font
def test_a_long_hook_wraps_inside_the_frame():
    """Measured against the real font rather than estimated from character
    count: a Somali hook is long, and an estimate that is 20% out puts the
    first word off the left edge."""
    png = render_card("HADDII AAD GEEL AHAAN LAHAYD MAXAAD SAMAYN LAHAYD "
                      "MARKA AY KUU IMAANAYSO BOOLIISKA",
                      width=600, size_px=48, dest=_tmp("wrap") / "c.png")
    from PIL import Image
    with Image.open(png) as im:
        assert im.width == 600
        assert im.height > 48 * 2, "a hook this long must be more than one line"


@needs_text_font
def test_the_same_card_renders_the_same_bytes():
    """§3.2. A card that differs run to run makes a byte-identity check on
    the whole piece useless."""
    a = render_card("SAME", width=400, size_px=40, dest=_tmp("d1") / "c.png")
    b = render_card("SAME", width=400, size_px=40, dest=_tmp("d2") / "c.png")
    assert a.read_bytes() == b.read_bytes()


@needs_emoji_font
def test_an_emoji_renders_in_colour_at_the_size_asked_for():
    """`embedded_color=True` is the difference between a sticker and a
    black silhouette, and nothing else in the render would reveal it."""
    png = render_emoji(LAUGH, 96, _tmp("emoji") / "e.png")
    assert png is not None
    from PIL import Image
    with Image.open(png) as im:
        im = im.convert("RGBA")
        assert im.size == (96, 96)
        opaque = {px[:3] for px in im.getdata() if px[3] > 200}
        assert len(opaque) > 8, "a colour emoji has more than a few colours"


@needs_emoji_font
def test_a_codepoint_the_font_has_no_glyph_for_is_reported_not_blank():
    """A blank PNG overlays as an invisible no-op, which reads as 'stickers
    are broken' rather than 'that character is not an emoji'."""
    assert render_emoji("￿", 64, _tmp("noglyph") / "e.png") is None


@needs_emoji_font
def test_a_deliberately_monochrome_emoji_survives_the_notdef_check():
    """The guard above rejects a single-colour render as a substituted
    box. Some real emoji are nearly monochrome, so the guard has to
    separate 'flat fill' from 'few colours' — measured, the black square
    antialiases into 34 tones and .notdef into exactly one."""
    png = render_emoji("⬛", 64, _tmp("mono") / "e.png")
    assert png is not None, "a real emoji was mistaken for a missing glyph"
