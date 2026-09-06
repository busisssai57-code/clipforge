"""The sketch format: a screenplay flag that did nothing, a frame nobody
read, and the pacing that makes the format itself.

Built after taking apart a 65s Somali animal-sketch post (35 shots, 1.86s
mean, hook card on frame one, emoji on the punchlines). The parts that
turned out to be missing were not creative — they were a flag the CLI
parsed and never used, and an aspect ratio every niche declared and
nothing consumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.genvideo.models import _generation_dims_for
from clipforge.genvideo.presets import Preset
from clipforge.genvideo.providers import GenResult, _ASPECT_RATIOS
from clipforge.genvideo.quota import QuotaLedger
from clipforge.genvideo.router import GenerationRouter
from clipforge.niches import (GEEL_SKETCH, NICHES, niche_aspect,
                              niche_as_preset, resolve_aspect, resolve_preset)

LAUGH = "\U0001F602"

SCRIPT = (f"EXT. SUUQ - SUBAX\n\nGeel dheer oo khudaar eegaya. [[{LAUGH}]]\n\n"
          "GEEL\nWaa imisa?\n\n"
          "INT. MAXKAMAD - GALAB\n\nGeel maxkamad taagan.\n\n"
          "XAAKIN\nMaxaad samaysay?\n")


class _FakeProvider:
    """Records what it was asked for; writes nothing."""

    name = "fake"

    def __init__(self):
        self.prompts: list[str] = []

    def available(self):
        return True

    def generate(self, *, prompt, seconds, fps, out_path, negative=None,
                 aspect_ratio="9:16"):
        self.prompts.append(prompt)
        Path(out_path).write_bytes(b"x")
        return GenResult(path=Path(out_path), provider=self.name,
                         seconds=seconds, prompt=prompt, model="fake")


def _router(tmp_path):
    provider = _FakeProvider()
    return provider, GenerationRouter(
        [provider], QuotaLedger.load(tmp_path / "q.json"))


PRESET = Preset(name="t", summary="", style="STYLE", avoid="", shot_seconds=1.9,
                fps=30, narrated=True, default_shots=2, cut_style="fast")


# ------------------------------------------------- the flag that did nothing

def test_screenplay_mode_cuts_on_scenes_not_full_stops(tmp_path):
    """`bta generate --screenplay` was a declared flag nothing read: the
    CLI parsed it and the router always split on sentences. The dashboard
    went through build_storyboard, which DID honour it, which is exactly
    why the CLI half looked implemented."""
    provider, router = _router(tmp_path)
    router.generate_sequence(brief=SCRIPT, preset=PRESET, out_dir=tmp_path,
                             shots=2, screenplay=True)
    # Case-insensitive: a slug is now prosified to lowercase before it
    # reaches the prompt (an all-caps label comes back drawn into the
    # frame). What this test is about is WHERE the cut falls, not casing.
    assert "suuq" in provider.prompts[0].lower()
    assert "maxkamad" in provider.prompts[1].lower()


def test_dialogue_never_reaches_a_generated_prompt(tmp_path):
    """Spoken words in a picture prompt render as subtitles and mouth
    artefacts. This is the property the whole screenplay path exists for,
    asserted at the point the prompt actually leaves for a provider."""
    provider, router = _router(tmp_path)
    router.generate_sequence(brief=SCRIPT, preset=PRESET, out_dir=tmp_path,
                             shots=2, screenplay=True)
    assert not any("Waa imisa" in p for p in provider.prompts)
    assert not any("Maxaad samaysay" in p for p in provider.prompts)


def test_without_the_flag_the_same_script_is_read_as_prose(tmp_path):
    """The counterpart the fix needs: if sentence-splitting also happened
    to produce scene-shaped beats, the test above would pass on the broken
    code too."""
    provider, router = _router(tmp_path)
    router.generate_sequence(brief=SCRIPT, preset=PRESET, out_dir=tmp_path,
                             shots=2, screenplay=False)
    assert "Waa imisa" in " ".join(provider.prompts), (
        "prose mode feeds the whole brief, dialogue included")


def test_punchline_marks_arrive_on_the_shot_that_was_marked(tmp_path):
    _provider, router = _router(tmp_path)
    result = router.generate_sequence(brief=SCRIPT, preset=PRESET,
                                      out_dir=tmp_path, shots=2,
                                      screenplay=True)
    assert [s.marks for s in result.shots] == [[LAUGH], []]


def test_a_failed_shot_still_carries_its_marks(tmp_path):
    """The post layer times stickers off the shots that SURVIVED. A mark
    dropped on failure would silently shift every later sticker."""
    class _Broken(_FakeProvider):
        def generate(self, **kw):
            from clipforge.genvideo.providers import ProviderError
            raise ProviderError("no")

    router = GenerationRouter([_Broken()], QuotaLedger.load(tmp_path / "q.json"))
    result = router.generate_sequence(brief=SCRIPT, preset=PRESET,
                                      out_dir=tmp_path, shots=2,
                                      screenplay=True)
    assert result.shots[0].marks == [LAUGH]


def test_a_brief_with_no_scenes_falls_back_instead_of_generating_nothing(
        tmp_path):
    provider, router = _router(tmp_path)
    router.generate_sequence(brief="", preset=PRESET, out_dir=tmp_path,
                             shots=2, screenplay=True)
    assert len(provider.prompts) == 2


# ------------------------------------------------- the frame nobody read

def test_every_declared_aspect_can_actually_be_generated():
    """`Niche.aspect` reached a dashboard summary and nothing else, so a
    niche could declare a frame the generator had no size for."""
    for niche in NICHES.values():
        assert niche.aspect in _ASPECT_RATIOS, niche.name


def test_a_portrait_aspect_generates_a_portrait_size():
    """The branch this replaces read '9:16 or else 16:9', so 3:4 - the
    sketch format's own frame - came out landscape."""
    w, h = _generation_dims_for("3:4")
    assert h > w
    assert abs((w / h) - 0.75) < 0.03


def test_an_unknown_aspect_falls_back_to_portrait_and_says_so(caplog):
    w, h = _generation_dims_for("banana")
    assert h > w, "a typo must not silently produce a landscape render"


def test_the_niche_frame_beats_the_global_default():
    assert resolve_aspect(None, "geel_sketch", "9:16") == "3:4"


def test_an_explicit_aspect_beats_the_niche():
    assert resolve_aspect("1:1", "geel_sketch", "9:16") == "1:1"


def test_a_preset_with_no_frame_of_its_own_keeps_the_default():
    assert resolve_aspect(None, "documentary", "16:9") == "16:9"
    assert niche_aspect("documentary") is None


# ------------------------------------------------- the format itself

def test_the_sketch_cuts_on_every_reaction():
    """1.9s is the reference's measured median shot. It is the format:
    the same shots at 5s read as an animation reel, not comedy."""
    assert GEEL_SKETCH.shot_seconds == pytest.approx(1.9)
    assert niche_as_preset(GEEL_SKETCH).cut_style == "fast"


def test_cut_style_is_derived_from_pacing_not_asserted():
    """It was hardcoded to "medium" for every niche, including one whose
    shots are six seconds long."""
    from clipforge.niches import ASCENDRO_MIND, VIRAL_CLIPS

    assert niche_as_preset(ASCENDRO_MIND).cut_style == "slow"
    assert niche_as_preset(VIRAL_CLIPS).cut_style == "medium"


def test_only_the_fast_cutting_formats_ask_for_the_post_layer():
    """A hook card on a documentary is a change of format, not of style.

    Keyed on the cutting rhythm rather than on a list of names: the post
    layer belongs to the sketch formats, and a niche added later must not
    be able to acquire one -- or lose one -- quietly. 2.5s is already the
    threshold `niche_as_preset` uses to call a cut "fast".
    """
    assert GEEL_SKETCH.post_layer is True
    for niche in NICHES.values():
        is_sketch = niche.shot_seconds <= 2.5
        assert niche.post_layer is is_sketch, (
            "{}: post_layer={} but shot_seconds={}".format(
                niche.name, niche.post_layer, niche.shot_seconds))


def test_the_sketch_resolves_by_name_like_any_other_preset():
    preset = resolve_preset("geel_sketch")
    assert preset.name == "geel_sketch"
    assert "camel" in preset.style


def test_the_sketch_forbids_burnt_text_in_the_picture():
    """The format's own text is the hook card, drawn by the post layer.
    A model asked for text renders misspelt gibberish over the joke."""
    for banned in ("text", "subtitles", "watermark"):
        assert banned in GEEL_SKETCH.gen_avoid


def test_the_example_script_is_shaped_like_the_format():
    """The shipped example is also the format's documentation: if it
    stops parsing into marked scenes, the feature has no worked example."""
    from clipforge.screenplay import to_beats_with_marks

    text = (Path(__file__).resolve().parents[2] / "examples"
            / "geel_suuq.fountain").read_text(encoding="utf-8")
    pairs = to_beats_with_marks(text)
    assert len(pairs) >= 8
    assert sum(len(m) for _b, m in pairs) >= 3
    assert not any("Title:" in b for b, _m in pairs)


def test_the_storyboard_path_leaks_no_dialogue_either(tmp_path):
    """The dashboard generates through build_storyboard, not the router.
    The leak was in the prompt builder, so it was in BOTH paths - and
    fixing only the one the CLI takes would have left the button that
    most operators press still shipping speech to the model."""
    from clipforge.genvideo.presets import build_storyboard

    board = build_storyboard(SCRIPT, PRESET, 2, screenplay=True)
    assert not any("Waa imisa" in s["prompt"] for s in board)


def test_the_piece_context_is_still_present_after_the_fix(tmp_path):
    """Stripping the leak by deleting the context would cost the model its
    sense of what the piece is. The synopsis keeps the picture half."""
    from clipforge.genvideo.presets import build_storyboard

    board = build_storyboard(SCRIPT, PRESET, 2, screenplay=True)
    assert "maxkamad" in board[0]["prompt"].lower(), board[0]["prompt"]


def test_a_long_script_does_not_bury_the_beat_in_context():
    """Context is context. A whole script pasted into every shot buries
    the one beat the shot is about."""
    from clipforge.screenplay import synopsis

    long_script = "".join(
        f"EXT. PLACE {i} - DAY\n\nSomething happens at length here.\n\n"
        for i in range(40))
    assert len(synopsis(long_script)) <= 330


# ------------------------------------------------- the same bug, twice

@pytest.mark.parametrize("aspect", sorted(_ASPECT_RATIOS))
def test_generation_and_delivery_agree_on_the_shape(aspect):
    """The binary '9:16 or else landscape' branch existed in TWO places:
    the generation size and the delivery size. Fixing only the first left
    a 3:4 render generated portrait and then scaled into a 1920x1080
    landscape frame - measured on a real shot before this test existed.

    So the property is checked end to end: whatever the aspect, the shape
    that comes out of generation must be the shape that is delivered."""
    from clipforge.genvideo.providers import _generation_dims, delivery_dims

    gw, gh = _generation_dims(aspect)
    dw, dh = delivery_dims(aspect)
    assert abs((gw / gh) - (dw / dh)) < 0.05, (
        f"{aspect}: generated {gw}x{gh} but delivered {dw}x{dh}")
    assert abs((dw / dh) - _ASPECT_RATIOS[aspect]) < 0.02


@pytest.mark.parametrize("aspect", sorted(_ASPECT_RATIOS))
def test_every_delivered_size_is_even(aspect):
    """H.264 chroma subsampling rejects odd dimensions - after the render,
    which is the expensive place to find out."""
    from clipforge.genvideo.providers import delivery_dims

    w, h = delivery_dims(aspect)
    assert w % 2 == 0 and h % 2 == 0


def test_the_familiar_short_form_size_is_unchanged():
    """9:16 must still deliver 1080x1920: this refactor is about the
    aspects that were broken, not about moving the one that worked."""
    from clipforge.genvideo.providers import DELIVERY_PORTRAIT, delivery_dims

    assert delivery_dims("9:16") == DELIVERY_PORTRAIT


@pytest.mark.parametrize("aspect", sorted(_ASPECT_RATIOS))
def test_a_generated_frame_spends_the_pixel_budget(aspect):
    """The hole in the test above, found by reading its own output: it
    asserted the SHAPE and said nothing about the SIZE, so a search
    ranked purely on ratio error picked 288x512 for 9:16 - exactly
    correct, 147k of a 460k budget, and upscaled to 1080x1920 from a
    third of the detail. Shape without size is half the property."""
    from clipforge.genvideo.providers import MAX_GEN_PIXELS, _generation_dims

    w, h = _generation_dims(aspect)
    assert w * h >= MAX_GEN_PIXELS * 0.85, (
        f"{aspect}: {w}x{h} leaves {(1 - w * h / MAX_GEN_PIXELS):.0%} of the "
        "envelope unused")


def test_the_measured_good_portrait_size_is_what_portrait_asks_for():
    """512x896 is this module's own measured-good size (test_genvideo_blank
    records the spatial-std measurement). The arithmetic that preceded the
    search returned 480x896 - below the size the measurement blessed."""
    from clipforge.genvideo.providers import _generation_dims

    assert _generation_dims("9:16") == (512, 896)


def test_the_finished_piece_keeps_the_name_the_dashboard_lists(tmp_path,
                                                               monkeypatch):
    """The dashboard enumerates `*/sequence.mp4`. Writing the stamped cut
    beside it as post.mp4 would have shown the operator the version with
    no hook and no stickers and called that the output."""
    import clipforge.cli as cli

    seq = tmp_path / "sequence.mp4"
    seq.write_bytes(b"unstamped")

    class _Shot:
        ok, path, marks, seconds = True, seq, ["\U0001F602"], 1.9

    class _Result:
        shots = [_Shot()]

    def _fake_apply(src, dest, spec, **kw):
        assert spec.hook == "HOOK"
        dest.write_bytes(b"stamped")
        return dest

    monkeypatch.setattr("clipforge.socialpost.apply_post", _fake_apply)
    monkeypatch.setattr("clipforge.ffmpeg.probe",
                        lambda p: type("I", (), {"duration_s": 1.9})())

    out = cli._apply_post_layer(seq, result=_Result(), niche_name="geel_sketch",
                                hook="HOOK", handle="@bta")
    assert out == seq
    assert seq.read_bytes() == b"stamped"
    assert (tmp_path / "sequence.raw.mp4").read_bytes() == b"unstamped", (
        "the un-stamped cut is kept so a new hook needs no regeneration")


def test_a_niche_without_the_post_layer_is_left_alone(tmp_path):
    import clipforge.cli as cli

    seq = tmp_path / "sequence.mp4"
    seq.write_bytes(b"raw")

    class _Result:
        shots = []

    out = cli._apply_post_layer(seq, result=_Result(),
                                niche_name="cinematic_doc", hook="H",
                                handle="@x")
    assert out == seq
    assert not (tmp_path / "sequence.raw.mp4").exists()
