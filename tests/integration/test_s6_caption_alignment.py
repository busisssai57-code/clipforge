"""Burned-in captions must reach the picture for windows far from zero.

These render REAL video through REAL libass and compare PIXELS, because the
artifacts cannot tell you this: the .ass can be perfectly formed, the render
can report success, the file can be well-shaped, and the captions can still
be absent from the frame. Only a differential render settles it.

A note on how this file came to exist, because it is the useful part:
a frame sampled from a shipped clip appeared to have no caption, and the
inferred cause was a time-base mismatch (`-ss` as an output option leaving
the filter graph on the media timeline while the .ass is clip-relative). The
sensitivity control written to prove that — render the "broken" shape, expect
no subtitle pixels — FAILED: the supposedly broken shape burned captions in
just fine. The hypothesis was wrong. What survives is this differential test,
which measures the property that actually matters instead of a mechanism that
was guessed at.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from clipforge.ffmpeg import find_binary

#: A window deliberately far from zero — a clip-relative subtitle file has to
#: line up with a picture that starts 40 s into the source.
FAR_START = 40.0
CLIP_DUR = 4.0

#: Peak |difference| above which we call it text. Anti-aliased subtitle edges
#: against any background clear this by a wide margin; codec noise does not.
TEXT_DIFF_FLOOR = 60.0


def _run(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    assert p.returncode == 0, (p.stderr or "")[-500:]
    return p.stderr or ""


def _make_source(dest: Path) -> Path:
    """A moving, textured source: a flat colour would make any difference
    trivially detectable and prove nothing about real footage."""
    _run([find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-y",
          "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30:duration=60",
          "-f", "lavfi", "-i", "sine=frequency=300:duration=60",
          "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-shortest", str(dest)])
    return dest


def _make_ass(dest: Path) -> Path:
    """Clip-relative subtitles spanning the whole clip."""
    dest.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\n"
        "PlayResY: 1920\n\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: Big,Arial,140,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,4,0,2,40,40,300,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
        " Effect, Text\n"
        f"Dialogue: 0,0:00:00.00,0:00:{CLIP_DUR:05.2f},Big,,0,0,0,,"
        "CAPTION HERE\n",
        encoding="utf-8")
    return dest


def _render(src: Path, out: Path, *, ass: Path | None) -> Path:
    """Render the window exactly as S6 does, optionally without subtitles."""
    vf = "setpts=PTS-STARTPTS,scale=1080:1920,setsar=1"
    if ass is not None:
        a = str(ass.resolve()).replace("\\", "/").replace(":", "\\:")
        vf += f",ass='{a}'"
    _run([find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-y",
          "-ss", f"{FAR_START}", "-i", str(src), "-t", f"{CLIP_DUR}",
          "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast",
          "-pix_fmt", "yuv420p", "-an", str(out)])
    return out


def _peak_diff(a: Path, b: Path) -> float:
    """Peak luma difference between two renders of the same window."""
    err = _run([find_binary("ffmpeg"), "-nostdin", "-hide_banner",
                "-i", str(a), "-i", str(b),
                "-lavfi", "blend=all_mode=difference,signalstats,"
                          "metadata=print",
                "-f", "null", "-"])
    peaks = [float(m) for m in re.findall(r"YMAX=([\d.]+)", err)]
    assert peaks, "signalstats produced no YMAX"
    return max(peaks)


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    return _make_source(tmp_path_factory.mktemp("src") / "long.mp4")


def test_captions_reach_the_picture_for_a_far_from_zero_window(source,
                                                               tmp_path):
    """The property that matters: pixels change because of our .ass."""
    ass = _make_ass(tmp_path / "subs.ass")
    with_ass = _render(source, tmp_path / "with.mp4", ass=ass)
    without = _render(source, tmp_path / "without.mp4", ass=None)
    peak = _peak_diff(with_ass, without)
    assert peak > TEXT_DIFF_FLOOR, (
        f"peak |diff| {peak:.0f} between renders with and without the "
        "subtitle filter: the .ass is not reaching the frame")


def test_the_differential_is_sensitive(source, tmp_path):
    """Control: two renders WITHOUT subtitles must be near-identical.

    Without this, the test above could pass on codec noise alone and would
    not be measuring subtitles at all.
    """
    a = _render(source, tmp_path / "a.mp4", ass=None)
    b = _render(source, tmp_path / "b.mp4", ass=None)
    peak = _peak_diff(a, b)
    assert peak < TEXT_DIFF_FLOOR, (
        f"two subtitle-free renders differ by {peak:.0f}; the measurement "
        "cannot distinguish subtitles from encoder noise")


def test_one_time_base_across_the_whole_filter_graph():
    """Structural: every stamp shares the clip-relative origin.

    Not a bug fix — a guard against mixing media-time and clip-time stamps
    in one graph, which is a mistake that renders successfully and looks
    almost right.
    """
    import inspect

    from clipforge.stages import s6_render

    src = inspect.getsource(s6_render.S6Render._execute)
    whole = inspect.getsource(s6_render)
    assert "setpts=PTS-STARTPTS" in src, "video chain must rebase"
    assert "asetpts=PTS-STARTPTS" in src, (
        "audio chain must rebase too, or the fades land in media time")
    assert '"-ss", f"{start:.3f}", "-i"' in src, (
        "-ss belongs on the INPUT: output-side seeking decodes and discards "
        "everything before the window (measured 280 s versus 70 s)")
    assert "start_s +" not in whole, (
        "a media-time stamp survives somewhere in the graph")
