"""A piece keeps the sound its shots came with.

`_concat_shots` re-encodes video and said nothing about audio, which was
correct while every generator was silent. LTX-2.5 is not: it produces
sound with the picture. Two things then break at once — the muxer's
default codec becomes an unstated dependency, and the concat DEMUXER
refuses a set of inputs whose streams do not match, which is exactly what
a failover produces when one shot comes from a model that generates audio
and the next from one that does not.

The fix that suggests itself is to drop audio at the concat. That would
make the mixed case work by throwing away the thing worth keeping, so the
silent shots get silence instead.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from clipforge.cli import _concat_shots, _has_audio, _with_uniform_audio

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None
                                  or shutil.which("ffprobe") is None,
                                  reason="needs ffmpeg and ffprobe")


def _make(path: Path, *, seconds: float = 1.0, audio: bool) -> Path:
    """A tiny real mp4, with or without a track."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-y",
           "-f", "lavfi", "-i", f"testsrc=size=128x224:rate=24:d={seconds}"]
    if audio:
        cmd += ["-f", "lavfi", "-i",
                f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd += [str(path)]
    subprocess.run(cmd, capture_output=True, timeout=120, check=True)
    return path


@needs_ffmpeg
def test_a_sequence_of_sounding_shots_arrives_with_sound(tmp_path):
    shots = [_make(tmp_path / f"s{i}.mp4", audio=True) for i in range(2)]
    dest = tmp_path / "sequence.mp4"
    _concat_shots(shots, dest)
    assert _has_audio(dest), "the piece lost the audio its shots carried"


@needs_ffmpeg
def test_a_mixed_sequence_keeps_the_audio_that_exists(tmp_path):
    """The failover case: one model generates sound, the next does not."""
    shots = [_make(tmp_path / "with.mp4", audio=True),
             _make(tmp_path / "without.mp4", audio=False)]
    dest = tmp_path / "sequence.mp4"
    _concat_shots(shots, dest)
    assert dest.is_file() and dest.stat().st_size > 1024
    assert _has_audio(dest), (
        "a silent shot in the middle took the whole piece's audio with it")


@needs_ffmpeg
def test_padding_only_touches_the_silent_shots(tmp_path):
    with_audio = _make(tmp_path / "with.mp4", audio=True)
    without = _make(tmp_path / "without.mp4", audio=False)
    out = _with_uniform_audio([with_audio, without], tmp_path)
    assert out[0] == with_audio, "a shot that had audio was re-encoded"
    assert out[1] != without and _has_audio(out[1])


@needs_ffmpeg
def test_an_all_silent_sequence_is_left_exactly_as_it_was(tmp_path):
    """Wan 2.2 pieces must not grow a pointless empty track."""
    shots = [_make(tmp_path / f"s{i}.mp4", audio=False) for i in range(2)]
    assert _with_uniform_audio(shots, tmp_path) == shots
    dest = tmp_path / "sequence.mp4"
    _concat_shots(shots, dest)
    assert not _has_audio(dest)


@needs_ffmpeg
def test_the_padding_scaffolding_does_not_stay_in_the_output_folder(tmp_path):
    """Padded copies beside the shots look like shots.

    They doubled a shot's bytes in the piece's own directory and would be
    picked up by anything globbing *.mp4 there - a re-concat, a manual
    re-cut, a listing. They belong to one concat and are removed with it.
    """
    out = tmp_path / "piece"
    out.mkdir()
    shots = [_make(out / "shot_00.mp4", audio=True),
             _make(out / "shot_01.mp4", audio=False)]
    _concat_shots(shots, out / "sequence.mp4")
    assert sorted(p.name for p in out.glob("*.mp4")) == [
        "sequence.mp4", "shot_00.mp4", "shot_01.mp4"]
    assert not [p for p in out.iterdir() if p.is_dir()], (
        "the temp dir outlived the concat")
