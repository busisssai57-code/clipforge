"""Niches — a whole look selected by name, pinned.

The pipeline's defaults encode ONE aesthetic: loud uppercase karaoke
captions in the bottom third, saturated footage, fast cuts. That is wrong
for most formats, and rendering a quiet monochrome channel with it produces
something that looks like a different account entirely.

These tests hold the distinguishing properties of each niche, because the
failure mode is silent: a niche whose captions quietly revert to the house
style still renders a valid video.
"""

from __future__ import annotations

import subprocess

import pytest

from clipforge.niches import (NICHES, get_niche, niche_s5_params,
                              niche_summary)


def test_every_niche_is_retrievable_by_name():
    for name in NICHES:
        assert get_niche(name).name == name


def test_an_unknown_niche_names_the_alternatives():
    with pytest.raises(ValueError) as err:
        get_niche("nope")
    for name in NICHES:
        assert name in str(err.value)


# ------------------------------------------------- the reference format

def test_dark_mindset_captions_are_quiet_and_centred():
    """The reference sets one modest line mid-frame with no outline. The
    restraint IS the aesthetic — loud karaoke reads as another channel."""
    c = get_niche("dark_mindset").caption
    assert c.animation == "quiet", "per-word pop is the wrong register here"
    assert c.uppercase is False, "shouting breaks the tone"
    assert c.alignment == 5, "text sits mid-frame, not in the bottom third"
    assert c.outline == 0.0, "the reference has no caption outline"
    assert c.size <= 64, f"{c.size}px is billboard-sized for this format"


def test_dark_mindset_is_monochrome_and_slow():
    n = get_niche("dark_mindset")
    assert "hue=s=0" in n.grade, "the defining property is no colour"
    assert n.jumpcut is False, (
        "silence removal destroys the pacing of a contemplative narration")
    assert n.shot_seconds >= 5.0
    assert "black and white" in n.gen_style


def test_dark_mindset_forbids_the_things_that_break_it():
    avoid = get_niche("dark_mindset").gen_avoid
    for banned in ("colour", "text", "watermark", "close-up face"):
        assert banned in avoid, f"{banned!r} must be in the negative prompt"


# ------------------------------------------------------- tiktok shop ugc

def test_shop_captions_clear_the_product_anchor():
    """The one element of a shop video that must never be covered is the
    cart, and the house 260 band lands captions on top of it.

    At 1080x1920 the platform's furniture — handle, caption, music ticker,
    and on a Shop video the orange cart pill — owns everything below
    y=1400. margin_v is measured up from the bottom, so clearing it means
    at least 1920-1400."""
    c = get_niche("tiktok_shop_ugc").caption
    assert c.alignment == 2, "shop captions sit in the bottom third"
    assert c.margin_v >= 520, (
        f"margin_v {c.margin_v} puts captions under the product anchor")
    assert c.margin_v > get_niche("viral_clips").caption.margin_v, (
        "the whole point is that the house band is too low for Shop")


def test_shop_captions_are_three_word_karaoke_in_tiktok_yellow():
    c = get_niche("tiktok_shop_ugc").caption
    assert c.animation == "pop" and c.uppercase is True
    assert c.max_words == 3, "three words, competing with the cart for space"
    assert c.primary == "&H0000E8FF", "the active word pops in TikTok yellow"
    assert c.secondary == "&H00FFFFFF", "the rest of the line stays white"
    assert c.shadow > 0 and c.outline > 0, (
        "text over a bright bathroom needs both to stay legible")


def test_shop_pacing_cuts_silence_unlike_the_contemplative_niches():
    """Fast cuts are the e-commerce retention lever, and here the dead air
    between sentences is a scroll. The opposite of dark_mindset, where the
    pauses are the piece."""
    n = get_niche("tiktok_shop_ugc")
    assert n.jumpcut is True
    assert get_niche("dark_mindset").jumpcut is False
    assert n.shot_seconds < get_niche("cinematic_doc").shot_seconds
    assert n.shot_seconds > get_niche("geel_sketch").shot_seconds, (
        "2.5s lets a physical demo complete; 1.9s is comic timing")


def test_shop_niche_protects_the_label_and_the_face():
    """The two things a shop video cannot afford to get wrong."""
    n = get_niche("tiktok_shop_ugc")
    for banned in ("warped label text", "deformed hands", "plastic skin",
                   "identity drift"):
        assert banned in n.gen_avoid, f"{banned!r} must be in the negative"
    assert "photorealistic" in n.gen_style, (
        "README routes photoreal human-subject work to Wan on this word")
    assert n.continuity is True, "unchained shots return a different creator"
    assert n.stickers is False, "emoji on a product claim reads as parody"


def test_the_shop_grade_is_gentler_than_the_loud_niches():
    """ARI_GOAT's note records what a saturated grade does to a face that
    fills the frame. This niche is a face at arm's length for 27 of its 30
    seconds."""
    n = get_niche("tiktok_shop_ugc")
    assert "saturation=1.06" in n.grade
    assert "unsharp" in n.grade, "the label has to survive the transcode"


def test_the_house_style_is_still_available_and_distinct():
    """Adding a niche must not quietly redefine the existing look."""
    viral = get_niche("viral_clips").caption
    dark = get_niche("dark_mindset").caption
    assert viral.animation == "pop" and viral.uppercase is True
    assert viral.alignment == 2, "viral captions sit in the bottom third"
    assert (viral.size, viral.alignment) != (dark.size, dark.alignment)


# --------------------------------------------------------- the grade

@pytest.mark.parametrize("name", sorted(NICHES))
def test_every_grade_is_a_filter_chain_ffmpeg_accepts(name):
    """A malformed grade fails the whole render. The names must exist in
    THIS ffmpeg build, which only running it proves."""
    grade = get_niche(name).grade
    if not grade:
        pytest.skip("no grade")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=128x224:rate=5:duration=0.2",
         "-vf", grade, "-frames:v", "1", "-f", "null", "-"],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert proc.returncode == 0, (
        f"{name} grade rejected by ffmpeg:\n{grade}\n{proc.stderr[-400:]}")


def test_the_monochrome_grade_actually_removes_colour(tmp_path):
    """Measured, not asserted: a saturated source must come out grey."""
    out = tmp_path / "g.png"
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=128x224:rate=5:duration=0.2",
         "-vf", get_niche("dark_mindset").grade, "-frames:v", "1", str(out)],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert proc.returncode == 0, proc.stderr[-300:]
    np = pytest.importorskip("numpy")
    from PIL import Image
    arr = np.asarray(Image.open(out).convert("RGB")).astype(np.float32)
    # In a grey image the channels agree everywhere.
    spread = float(np.abs(arr - arr.mean(axis=2, keepdims=True)).mean())
    assert spread < 3.0, f"channel spread {spread:.2f} — colour survived"


# -------------------------------------------------------- s5 plumbing

@pytest.mark.parametrize("name", sorted(NICHES))
def test_s5_params_carry_the_caption_identity(name):
    n = get_niche(name)
    p = niche_s5_params(n)
    assert p["animation"] == n.caption.animation
    assert p["uppercase"] == n.caption.uppercase
    assert p["font_size"] == n.caption.size
    assert p["alignment"] == n.caption.alignment


@pytest.mark.parametrize("name", sorted(NICHES))
def test_line_length_reaches_the_key_s5_actually_reads(name):
    """S5 reads `max_words_per_line`; this emitted `max_words`.

    `process` builds s5_params from config and then `.update()`s this dict
    over the top, so the niche's line length was not merely dropped — it
    was shadowed by cfg.s5.max_words_per_line for every render ever made.
    dark_mindset's 7-word lines and geel_sketch's 5 never reached libass.
    A niche that silently captions at the house line length still produces
    a valid video, which is why nothing caught it."""
    n = get_niche(name)
    p = niche_s5_params(n)
    assert p["max_words_per_line"] == n.caption.max_words


def test_a_two_colour_niche_keeps_its_base_colour():
    """S5 takes highlight_color and base_color separately; this fed
    `primary` to both, so a niche could only ever be monochrome. The
    karaoke look this platform uses is a white line with the active word
    popping in colour."""
    p = niche_s5_params(get_niche("tiktok_shop_ugc"))
    assert p["highlight_color"] == "&H0000E8FF"
    assert p["base_color"] == "&H00FFFFFF"
    assert p["highlight_color"] != p["base_color"]


@pytest.mark.parametrize("name", sorted(NICHES))
def test_a_single_colour_niche_is_unchanged_by_the_base_colour_field(name):
    """Adding `secondary` must not restyle the niches written before it."""
    n = get_niche(name)
    if n.caption.secondary is not None:
        pytest.skip("two-colour niche")
    p = niche_s5_params(n)
    assert p["base_color"] == n.caption.primary


def test_caption_colours_are_ass_bgr_literals():
    """ASS is &HBBGGRR& — writing RGB here fails SILENTLY, rendering blue
    text where red was intended."""
    for n in NICHES.values():
        for value in (n.caption.primary, n.caption.outline_colour):
            assert value.startswith("&H") and value.endswith(("&", "F", "0")), (
                f"{value!r} is not an ASS colour literal")
            assert len(value) in (10, 11), f"{value!r} has the wrong width"


# ------------------------------------------------- preset resolution
# `bta generate --preset X` and the dashboard both take ONE name. Before
# this, that name was checked only against the four base creative presets
# — every niche name was rejected with "unknown preset", which broke the
# dashboard's Generate button (it sends the niche name) and would have
# broken the swarm's Generator role identically.











def test_summary_is_dashboard_shaped():
    rows = niche_summary()
    assert len(rows) == len(NICHES)
    for row in rows:
        assert {"name", "label", "summary", "aspect", "shots",
                "captions", "graded", "keywords"} <= set(row)
    assert any(r["name"] == "dark_mindset" for r in rows)
