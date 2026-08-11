"""Determinism Law at the RENDER boundary: same inputs, same BYTES.

The architecture blueprint's case for Remotion rests on "completely
deterministic" code-driven rendering. Our render layer is ffmpeg+libass —
and this test holds it to the SAME standard, measured rather than asserted:
two full S6 runs over identical inputs must produce byte-identical mp4s.
(For the record, a headless-Chromium renderer could not pass this bar
across environments; a pure filter graph can.)

Runs the REAL stage twice with separate artifact/state stores so the cache
cannot short-circuit the second render.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from clipforge.ffmpeg import find_binary, probe
from clipforge.schemas.campath import CamPathArtifact, CropFrame
from clipforge.schemas.render import SubtitleArtifact
from clipforge.stages.base import digest_bytes
from clipforge.stages.s6_render import S6Render
from clipforge.state import StateDB

CLIP_S = 8.0


@pytest.fixture(scope="module")
def source(tmp_path_factory) -> Path:
    dest = tmp_path_factory.mktemp("det") / "src.mp4"
    proc = subprocess.run(
        [find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=640x480:rate=30:duration=12",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=12",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-400:]
    return dest


def _artifacts(tmp: Path, source: Path):
    info = probe(source)
    frames = [CropFrame(frame=f, x=180, y=0, w=270, h=480)
              for f in range(int(CLIP_S * 30))]
    campath = CamPathArtifact(
        cache_key="p" * 64, source_ranking="r" * 64,
        clip_start=1.0, clip_end=1.0 + CLIP_S, framing_mode="center",
        frames=frames, assignments=[], src_width=info.width,
        src_height=info.height, src_fps_rational=info.fps_rational)
    ass = tmp / "subs.ass"
    ass.write_text(
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1080\nPlayResY: 1920\n"
        "\n[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
        " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
        " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
        " Alignment, MarginL, MarginR, MarginV, Encoding\n"
        "Style: K,Arial,90,&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,"
        "-1,0,0,0,100,100,0,0,1,3,1,2,60,60,260,1\n\n[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV,"
        " Effect, Text\n"
        "Dialogue: 0,0:00:01.00,0:00:06.00,K,,0,0,0,,DETERMINISM\n",
        encoding="utf-8")
    subs = SubtitleArtifact(
        cache_key="s" * 64, source_campath="p" * 64, ass_path=str(ass),
        clip_start=1.0, clip_end=1.0 + CLIP_S, line_count=1, word_count=1,
        ass_sha256=hashlib.sha256(ass.read_bytes()).hexdigest())
    return campath, subs


def _render_once(workdir: Path, source: Path, campath, subs) -> str:
    workdir.mkdir(parents=True, exist_ok=True)
    db = StateDB(workdir / "s.db")
    try:
        stage = S6Render(db, workdir / "artifacts")
        art = stage.run(
            input_digest=digest_bytes(b"determinism"),
            params={"width": 1080, "height": 1920, "encoder": "libx264",
                    "cq": 23, "audio_bitrate": "128k"},
            campath_artifact=campath, subtitle_artifact=subs,
            video_path=source, clips_dir=workdir / "clips")
        return art.clip_sha256
    finally:
        db.close()


def test_two_renders_of_identical_inputs_are_byte_identical(source, tmp_path):
    campath, subs = _artifacts(tmp_path, source)
    sha1 = _render_once(tmp_path / "run1", source, campath, subs)
    sha2 = _render_once(tmp_path / "run2", source, campath, subs)
    assert sha1 == sha2, (
        "same inputs produced different bytes — the render layer is not "
        "deterministic and the Determinism Law claim for S6 is false")
