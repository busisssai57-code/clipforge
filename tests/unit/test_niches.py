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

from clipforge.niches import (NICHES, get_niche, niche_as_preset,
                              niche_s5_params, niche_summary, resolve_preset)


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

@pytest.mark.parametrize("name", sorted(NICHES))
def test_every_niche_name_resolves_as_a_preset(name):
    resolved = resolve_preset(name)
    assert resolved.name == name


def test_the_base_presets_still_resolve():
    from clipforge.genvideo.presets import PRESETS

    for name in PRESETS:
        assert resolve_preset(name).name == name


def test_an_unknown_name_lists_both_namespaces():
    with pytest.raises(ValueError) as err:
        resolve_preset("not_a_thing")
    msg = str(err.value)
    assert "dark_mindset" in msg, "niches must be listed as options"
    assert "documentary" in msg, "base presets must be listed as options"


def test_a_niche_preset_carries_the_niches_own_prompt_style():
    """The point of the adapter: generation must use the niche's SPECIFIC
    style text, not fall back to a generic preset that happens to share
    a rough theme."""
    dark = get_niche("dark_mindset")
    resolved = resolve_preset("dark_mindset")
    assert resolved.style == dark.gen_style
    assert resolved.avoid == dark.gen_avoid
    assert resolved.shot_seconds == dark.shot_seconds
    assert resolved.fps == dark.fps
    assert resolved.default_shots == dark.default_shots


def test_niche_as_preset_is_a_real_preset_instance():
    from clipforge.genvideo.presets import Preset

    assert isinstance(niche_as_preset(get_niche("dark_mindset")), Preset)


def test_summary_is_dashboard_shaped():
    rows = niche_summary()
    assert len(rows) == len(NICHES)
    for row in rows:
        assert {"name", "label", "summary", "aspect", "shots",
                "captions", "graded", "keywords"} <= set(row)
    assert any(r["name"] == "dark_mindset" for r in rows)
