"""The voiceover mix must not truncate the clip or move its level.

Both bugs this file pins were live in `mix_voiceover` from the day it was
written and survived 33 unit tests, because those tests mock `_run` and
assert on the command that WOULD have been executed. The command was
well-formed both times; what ffmpeg did with it was wrong.

They only appeared when the function was first actually called, which was
when the CLI landed — the code had no caller at all, while the capability
tile reported voiceover LIVE.

1. **Truncation.** `sidechaincompress` ends when EITHER input EOFs, so an
   unpadded voice capped the bed at the voice's length: 58.9s of video
   against a 4.6s audio stream. The clip went silent after the hook.
2. **Level.** `amix` divides by its input count unless told otherwise, so
   the entire mix came back 6.0 dB down — uniformly, long after the voice
   had stopped, which breaks the -14 LUFS contract for every voiced clip.

Both are properties of decoded audio, so both are measured from decoded
audio (the pacing round's lesson: measure PCM, not detector events).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from clipforge.enhance import mix_voiceover
from clipforge.ffmpeg import require_binary

#: The bed runs far longer than the voice — that gap is where both bugs
#: live. A voice as long as the clip would hide the truncation entirely.
BED_S = 12.0
VOICE_S = 2.0


def _ff(*args: str, timeout: int = 180) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
         *args], capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout)
    assert proc.returncode == 0, (proc.stderr or "")[-500:]
    return proc


@pytest.fixture
def bed(tmp_path: Path) -> Path:
    """A 12s 'clip': real video stream plus a steady tone to duck."""
    dest = tmp_path / "bed.mp4"
    _ff("-f", "lavfi", "-i", f"testsrc2=size=320x240:rate=15:duration={BED_S}",
        "-f", "lavfi", "-i", f"sine=frequency=220:duration={BED_S}",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(dest))
    return dest


@pytest.fixture
def voice(tmp_path: Path) -> Path:
    """A 2s 'voice' — loud enough to open the sidechain compressor."""
    dest = tmp_path / "voice.wav"
    _ff("-f", "lavfi", "-i", f"sine=frequency=900:duration={VOICE_S}",
        "-af", "volume=0.9", "-c:a", "pcm_s16le", str(dest))
    return dest


def _stream_duration(path: Path, kind: str) -> float:
    proc = subprocess.run(
        [str(require_binary("ffprobe")), "-v", "error", "-select_streams",
         kind, "-show_entries", "stream=duration", "-of", "csv=p=0",
         str(path)], capture_output=True, text=True, timeout=60)
    return float((proc.stdout or "0").strip().splitlines()[0])


def _mean_db(path: Path, start: float, dur: float, band: str = "") -> float:
    """Mean volume over a window, optionally through a band filter.

    The band matters for the ducking test: bed and voice occupy different
    frequencies on purpose, so the bed can be measured inside the mix
    without the voice's own energy masking the very drop being asserted.
    """
    af = f"{band},volumedetect" if band else "volumedetect"
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
         "-ss", str(start), "-t", str(dur), "-i", str(path),
         "-af", af, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=120)
    for line in (proc.stderr or "").splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].split("dB")[0])
    raise AssertionError(
        "no mean_volume — the segment decoded to nothing, which is itself "
        f"the truncation bug: {(proc.stderr or '')[-300:]}")


def test_mix_keeps_the_full_clip_audible(bed, voice, tmp_path):
    """Bug 1: the audio stream must span the VIDEO, not the voice."""
    out = mix_voiceover(bed, voice, tmp_path / "out.mp4")

    audio_s = _stream_duration(out, "a")
    video_s = _stream_duration(out, "v")
    assert audio_s == pytest.approx(video_s, abs=0.15), (
        f"audio {audio_s:.2f}s vs video {video_s:.2f}s — the bed was cut to "
        f"the voice ({VOICE_S}s)")
    # and it is really there, not a silent pad: decode the far end.
    assert _mean_db(out, BED_S - 3, 2) > -50


def test_mix_does_not_attenuate_the_bed(bed, voice, tmp_path):
    """Bug 2: outside the voice, the level must match the source.

    amix's default normalization is a flat -6 dB on two inputs, which is
    far outside this tolerance and shows up in exactly this window.
    """
    out = mix_voiceover(bed, voice, tmp_path / "out.mp4")

    after = VOICE_S + 2.0  # clear of the compressor's 250ms release
    before_db = _mean_db(bed, after, 4.0)
    after_db = _mean_db(out, after, 4.0)
    assert after_db == pytest.approx(before_db, abs=1.0), (
        f"bed sits at {after_db:.1f} dB vs source {before_db:.1f} dB long "
        "after the voice ended")


def test_mix_ducks_while_the_voice_speaks(bed, voice, tmp_path):
    """The feature itself: the bed drops UNDER the voice, not everywhere.

    Paired with the test above this is what distinguishes ducking from a
    blanket attenuation — either one alone is satisfied by a bug.
    """
    out = mix_voiceover(bed, voice, tmp_path / "out.mp4")

    # The bed is a 220 Hz tone, the voice 900 Hz. Measuring the whole mix
    # here reads +0.7 dB in the voice window — the voice's own energy
    # almost exactly offsets the duck, so a broadband assertion sees "no
    # ducking" whether or not ducking happened. Isolating the bed's band is
    # what makes the property observable at all.
    LOW = "lowpass=f=400"
    bed_under_voice = _mean_db(out, 0.2, 1.5, LOW)
    bed_when_quiet = _mean_db(out, VOICE_S + 2.0, 4.0, LOW)
    assert bed_under_voice < bed_when_quiet - 2.0, (
        f"bed not ducked: {bed_under_voice:.1f} dB under the voice vs "
        f"{bed_when_quiet:.1f} dB after it")
