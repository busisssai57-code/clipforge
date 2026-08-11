"""Speech enhancement must actually reach the audio, and be safe when off.

Source-level tests would prove nothing here: the whole feature is a filter
string handed to ffmpeg, and the failure mode that matters is "the filter
name is wrong / not in this build", which only a real render surfaces.
Every filter used is confirmed present by running it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clipforge.ffmpeg import require_binary
from clipforge.stages.s6_render import _SPEECH_CHAINS, _speech_chain

MODES = ["gentle", "strong"]


def _noisy_speechlike(dest: Path) -> Path:
    """Tone + broadband noise + rumble: something the chain can act on."""
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi", "-i", "sine=frequency=220:duration=4",
         "-f", "lavfi", "-i", "anoisesrc=amplitude=0.06:duration=4",
         "-f", "lavfi", "-i", "sine=frequency=35:duration=4",
         "-filter_complex", "[0][1][2]amix=inputs=3:normalize=0[a]",
         "-map", "[a]", "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    assert proc.returncode == 0, proc.stderr[-400:]
    return dest


def _rms_db(path: Path, band: str = "") -> float:
    """Mean volume, optionally after a band filter."""
    af = f"{band},volumedetect" if band else "volumedetect"
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
         "-i", str(path), "-af", af, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    for line in (proc.stderr or "").splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    raise AssertionError(f"no mean_volume in output: {proc.stderr[-300:]}")


def _apply(src: Path, dest: Path, mode: str) -> Path:
    chain = _speech_chain(mode)
    af = f"{chain}anull" if chain else "anull"
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
         "-i", str(src), "-af", af, "-c:a", "pcm_s16le", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=180)
    assert proc.returncode == 0, (
        f"mode={mode!r} filter chain rejected by ffmpeg:\n"
        f"{af}\n{proc.stderr[-600:]}")
    return dest


@pytest.mark.parametrize("mode", MODES)
def test_every_filter_in_the_chain_exists_in_this_ffmpeg(mode, tmp_path):
    """The real risk: a filter that is not compiled into this build. An
    unknown filter name fails the render, not the audio."""
    src = _noisy_speechlike(tmp_path / "src.wav")
    out = _apply(src, tmp_path / f"{mode}.wav", mode)
    assert out.stat().st_size > 1000


@pytest.mark.parametrize("mode", MODES)
def test_enhancement_attenuates_low_frequency_rumble(mode, tmp_path):
    """The highpass is the one change that must be measurable: a 35 Hz
    tone is inaudible on a phone and eats headroom the voice needs.

    Measured as a RATIO of sub-60 Hz energy to full-band energy, not as an
    absolute level. The first version of this test compared absolute dB
    and failed on working filters: `speechnorm` applies makeup gain at the
    end of the chain, which lifts the residual rumble along with the
    voice, so an absolute reading cannot see the highpass at all. The
    ratio is gain-invariant, which is the property actually claimed.
    """
    src = _noisy_speechlike(tmp_path / "src.wav")
    out = _apply(src, tmp_path / f"{mode}.wav", mode)
    before = _rms_db(src, "lowpass=f=60") - _rms_db(src)
    after = _rms_db(out, "lowpass=f=60") - _rms_db(out)
    assert after < before - 6.0, (
        f"{mode}: sub-60Hz share of the signal went {before:.1f} -> "
        f"{after:.1f} dB relative; the highpass is not reaching the audio")


def test_off_is_bit_exactly_a_no_op(tmp_path):
    """Default must not touch a single sample — the Determinism Law reads
    on every clip rendered before this feature existed."""
    src = _noisy_speechlike(tmp_path / "src.wav")
    assert _speech_chain("off") == ""
    out = _apply(src, tmp_path / "off.wav", "off")
    assert out.read_bytes() == src.read_bytes(), (
        "enhance_speech=off altered the audio")


def test_an_unknown_mode_degrades_to_off_rather_than_failing(tmp_path):
    assert _speech_chain("typo") == ""
    assert _speech_chain("") == ""


def test_the_chain_is_applied_before_loudnorm():
    """Order is load-bearing: normalizing first would measure the noise
    floor and set the gain against material the chain then removes."""
    import inspect

    from clipforge.stages import s6_render

    src = inspect.getsource(s6_render.S6Render._execute)
    assert "_speech_chain(" in src
    idx_chain = src.index("enhance = _speech_chain(")
    idx_loud = src.index("loudnorm=I=")
    assert idx_chain < idx_loud
    # And structurally: the prefix is concatenated ahead of loudnorm.
    assert 'f"{enhance}"' in src or "{enhance}" in src


def test_modes_are_the_documented_set():
    assert set(_SPEECH_CHAINS) == {"off", "gentle", "strong"}
