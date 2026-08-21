"""The four defects in the first LTX-2.5 pieces, each pinned.

The operator watched the first output and reported: morphing, weird
textures, blurry, no audio. Three of those were choices this repo made,
not limits of the model:

* **No audio** — the model generates sound with the picture (its own
  config carries `audio_in_channels` and a vocoder; the pipeline returns
  `(video, audio)`), and the worker read only the frames.
* **Blurry** — generation ran at 512x896 and was scaled 2.1x to
  1080x1920, because a 460k-pixel cap MEASURED ON LTX-VIDEO 0.9 was
  applied to every model.
* **Morphing and mush** — 8 steps at CFG 1.0, taken from a model-card
  claim about a "distilled" schedule that nothing here ever ran against
  the alternative. diffusers' own LTX2 example uses 30 at 3.0.

The fourth, temporal drift, is a property of the model and is not
claimed to be fixed here.

The stub worker keeps these honest without loading 22 GB of weights:
it answers the same protocol and returns noise, so what is under test is
this repo's plumbing, which is where all three defects lived.
"""

from __future__ import annotations

import dataclasses
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from clipforge.genvideo import subproc
from clipforge.genvideo.models import LTX_25
from clipforge.genvideo.providers import _generation_dims, _write_audio_wav
from clipforge.genvideo.subproc import SubprocessModelProvider

# `print(..., file=_REPLY)` rather than a write with an escaped newline:
# this source is written into a file by a test, and every backslash in it
# has to survive two levels of quoting to mean what it says.
_STUB = "\n".join([
    "import json, sys",
    "import numpy as np",
    "_REPLY = sys.stdout",
    "sys.stdout = sys.stderr",
    "print('library noise that must not reach the parent')",
    "for line in sys.stdin:",
    "    if not line.strip():",
    "        continue",
    "    req = json.loads(line)",
    "    if req['op'] == 'quit':",
    "        break",
    "    if req['op'] == 'ping':",
    "        print(json.dumps({'ok': True, 'loaded': False}), file=_REPLY,"
    " flush=True)",
    "        continue",
    "    rng = np.random.default_rng(req['seed'])",
    "    frames = int(req['frames'])",
    "    arr = rng.integers(0, 255, size=(frames, 64, 64, 3), dtype=np.uint8)",
    "    np.save(req['out'], arr)",
    "    reply = {'ok': True, 'npy': req['out'], 'frames': frames,",
    "             'echo': req, 'audio_npy': None, 'audio_sample_rate': 0}",
    "    if req.get('audio_out'):",
    "        secs = frames / float(req['fps'])",
    "        n = int(round(secs * 48000))",
    "        t = np.linspace(0, secs, n, endpoint=False)",
    "        tone = (0.25 * np.sin(2 * np.pi * 440 * t)).astype('float32')",
    "        np.save(req['audio_out'], np.stack([tone, tone]))",
    "        reply['audio_npy'] = req['audio_out']",
    "        reply['audio_sample_rate'] = 48000",
    "    print(json.dumps(reply), file=_REPLY, flush=True)",
])

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None
                                  or shutil.which("ffprobe") is None,
                                  reason="needs ffmpeg and ffprobe")


@pytest.fixture()
def provider(tmp_path, monkeypatch):
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB, encoding="utf-8")
    monkeypatch.setattr(subproc, "WORKER", stub)
    # A budget any machine has: nothing here asserts about VRAM.
    spec = dataclasses.replace(LTX_25, vram_gb=0.1)
    p = SubprocessModelProvider(spec, seed=7, interpreter=Path(sys.executable))
    yield p
    p.close()


def _streams(path: Path, kind: str) -> list[str]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", kind[0],
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(path)],
        capture_output=True, text=True, timeout=60)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


# ------------------------------------------------------------- audio

@needs_ffmpeg
def test_generated_sound_reaches_the_delivered_file(provider, tmp_path):
    """The whole of "no audio": the model made sound and we dropped it.

    This asserted only that a stream EXISTED, and that is the same
    mistake as checking a video has frames without looking at a pixel: a
    six-second piece shipped with an AAC stream measuring -inf LUFS and
    passed. It now decodes the audio back and measures it.
    """
    out = tmp_path / "shot.mp4"
    provider.generate(prompt="a camel", seconds=1.0, fps=24, out_path=out)
    assert _streams(out, "audio"), "the delivered shot has no audio stream"
    assert _streams(out, "video"), "the video stream went missing"

    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(out), "-f", "f32le",
         "-ac", "1", "-ar", "48000", "-"],
        capture_output=True, timeout=120)
    samples = np.frombuffer(raw.stdout, dtype="<f4")
    assert samples.size, "the audio stream decoded to nothing"
    rms = float(np.sqrt((samples ** 2).mean()))
    # The stub feeds a 440 Hz tone at 0.25; AAC and the int16 round trip
    # cost a little, and anything near zero is a stream nobody can hear.
    assert rms > 0.05, f"the delivered audio is inaudible (rms {rms:.5f})"


@needs_ffmpeg
def test_no_stray_arrays_survive_the_shot(provider, tmp_path):
    """Frames and audio cross the pipe as files; neither may be left."""
    out = tmp_path / "shot.mp4"
    provider.generate(prompt="a camel", seconds=1.0, fps=24, out_path=out)
    assert not list(out.parent.glob("*.npy"))
    assert not list(out.parent.glob("*.wav"))


def test_audio_is_matched_to_the_picture_not_trimmed_against_it(tmp_path):
    """`-shortest` would have dropped a video frame.

    Measured on the real model: 2.010 s of audio against 2.042 s of
    video. Padding and trimming happen here, in numpy, where the result
    is exact and deterministic.
    """
    import wave

    short = np.zeros((2, 1000), dtype=np.float32)
    dest = tmp_path / "a.wav"
    _write_audio_wav(short, 48000, frames=48, fps=24, dest=dest)
    with wave.open(str(dest)) as w:
        assert w.getnframes() == 96000, "silence was not padded to the frames"
        assert w.getnchannels() == 2

    long = np.zeros((2, 200000), dtype=np.float32)
    _write_audio_wav(long, 48000, frames=48, fps=24, dest=dest)
    with wave.open(str(dest)) as w:
        assert w.getnframes() == 96000, "a long tail was not trimmed"


def test_full_scale_audio_clamps_instead_of_wrapping(tmp_path):
    """This model comes back at peak 1.0; int16 wrap turns that into buzz."""
    import wave

    hot = np.full((1, 48000), 1.5, dtype=np.float32)
    dest = tmp_path / "hot.wav"
    _write_audio_wav(hot, 48000, frames=24, fps=24, dest=dest)
    with wave.open(str(dest)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    assert pcm.max() == 32767 and pcm.min() >= 0, "clipping wrapped"


# -------------------------------------------------------- resolution

def test_generation_uses_the_models_envelope_not_the_global_cap(provider,
                                                                tmp_path):
    """512x896 upscaled 2.1x is most of what "blurry" was.

    `MAX_GEN_PIXELS` is 460k because LTX-Video 0.9 returned blank frames
    above it. That is a real measurement about a DIFFERENT model, and it
    was being applied to every model in the registry.
    """
    out = tmp_path / "shot.mp4"
    provider.generate(prompt="a camel", seconds=1.0, fps=24, out_path=out)
    echo = provider._last_request
    asked = echo["width"] * echo["height"]
    capped = _generation_dims("9:16")
    assert asked > capped[0] * capped[1], (
        f"generated at {echo['width']}x{echo['height']}, the global cap's "
        f"{capped[0]}x{capped[1]} — the model's own envelope was ignored")
    assert asked <= LTX_25.max_pixels


# ---------------------------------------------------------- negative

def test_the_models_own_failures_are_negated_alongside_the_presets(provider):
    """Two different kinds of "avoid", kept apart.

    The preset's list is the operator's editorial choice; "blurry,
    jittery, distorted" is a property of this checkpoint, and the
    vendor's own example negates exactly those.
    """
    got = provider._negative("cartoon, watermark")
    assert got.startswith("cartoon, watermark"), "the operator's list moved"
    for term in ("blurry", "jittery", "distorted"):
        assert term in got


def test_a_term_already_in_the_preset_is_not_repeated(provider):
    got = provider._negative("cartoon, blurry")
    assert got.lower().count("blurry") == 1


def test_a_model_with_no_known_failures_is_left_alone(provider):
    provider.spec = dataclasses.replace(provider.spec, default_negative="")
    assert provider._negative("cartoon") == "cartoon"


# ------------------------------------------------------------- steps

def test_the_spec_says_this_model_makes_sound():
    """The flag the provider warns on when a run comes back silent.

    (The schedule those first pieces were generated with is pinned in
    `test_quantization.py`, next to the rest of the registry's claims.)
    """
    assert LTX_25.generates_audio is True


# --------------------------------------------------------- the cue

def test_an_audio_model_gets_told_there_is_sound(provider):
    """The measured cause of the silent piece.

    All at 2 s, one seed, same settings, measured on the delivered file:

        bare brief                                  -18.7 LUFS
        bare brief + the preset's 50-word style     -52.8 LUFS
        the same, plus one sentence about sound     -12.8 LUFS

    The style block is what `build_shot_prompt` appends to every shot,
    and on this model it silences the track while producing a good
    picture. The cue restores it without touching the style.
    """
    got = provider._prompt("a camel at a market. 35mm anamorphic lens, "
                           "muted filmic colour grade")
    assert got.endswith(LTX_25.audio_prompt_hint)


def test_a_writers_own_audio_direction_is_left_alone(provider):
    """"No dialogue, only wind" must not be argued with by a hint."""
    written = "a camel at a market, no dialogue, only the sound of wind"
    assert provider._prompt(written) == written


def test_a_silent_model_is_not_given_an_audio_cue(provider):
    provider.spec = dataclasses.replace(provider.spec, generates_audio=False)
    assert provider._prompt("a camel") == "a camel"


@needs_ffmpeg
def test_the_cue_reaches_the_worker(provider, tmp_path):
    """Not just computed - actually in the request that crosses the pipe."""
    provider.generate(prompt="a camel at a market, 35mm anamorphic lens",
                      seconds=1.0, fps=24, out_path=tmp_path / "s.mp4")
    assert LTX_25.audio_prompt_hint in provider._last_request["prompt"]


def test_a_word_inside_another_word_does_not_count_as_audio_direction(
        provider):
    """`in` matched "hum" inside "humble" and "score" inside "scoreboard".

    An ordinary brief then looked like it already carried audio
    direction, and lost the cue that is the measured difference between
    -52.8 and -12.8 LUFS - silently, and only for some prompts.
    """
    for brief in ("a humble market trader at dusk",
                  "a scoreboard above a stadium",
                  "a humid afternoon in the souq"):
        got = provider._prompt(brief)
        assert got.endswith(LTX_25.audio_prompt_hint), brief

    # A real mention still suppresses it.
    assert provider._prompt("the hum of the market") == "the hum of the market"


@needs_ffmpeg
def test_a_failed_encode_releases_the_worker(provider, tmp_path, monkeypatch):
    """The router fails over to a provider that needs this card.

    A blank-frame rejection raises out of `_write_video`, and the worker
    was left holding ~13 GB while the next provider tried to load its own
    model into the same 24 GB. The failed-reply path already closed for
    this reason; the encode path did not.
    """
    from clipforge.genvideo import providers as prov

    def _boom(*_a, **_k):
        raise prov.ProviderError("blank frame detected")

    monkeypatch.setattr(subproc, "_write_video", _boom)
    with pytest.raises(prov.ProviderError):
        provider.generate(prompt="a camel", seconds=1.0, fps=24,
                          out_path=tmp_path / "s.mp4")
    assert provider._proc is None, "the worker outlived the failed encode"
