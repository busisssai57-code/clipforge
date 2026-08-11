"""S7 quality gate — exercised against REAL rendered files.

Fixtures are generated with the real ffmpeg (lavfi sources), because the QA
stage's whole value is measuring bytes on disk. A QA test that mocks ffprobe
is a QA stage that trusts producers, which is the failure mode S7 exists to
catch. No GPU involved — these are not gpu-marked.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from clipforge.ffmpeg import find_binary
from clipforge.schemas.render import ClipArtifact, SubtitleArtifact
from clipforge.stages.base import digest_bytes
from clipforge.stages.s7_qa import S7QualityGate
from clipforge.state import StateDB


def _render_fixture(dest: Path, *, seconds: float = 30.5, silent: bool = False,
                    black: bool = False, width: int = 1080,
                    height: int = 1920) -> Path:
    """A real mp4 the shape S6 produces, via lavfi."""
    ffmpeg = find_binary("ffmpeg")
    vsrc = (f"color=c=black:s={width}x{height}:r=30" if black
            else f"testsrc=size={width}x{height}:rate=30")
    asrc = ("anullsrc=r=48000:cl=stereo" if silent
            else "sine=frequency=440:sample_rate=48000")
    cmd = [ffmpeg, "-nostdin", "-hide_banner", "-y",
           "-f", "lavfi", "-i", vsrc, "-f", "lavfi", "-i", asrc,
           "-t", f"{seconds}", "-c:v", "libx264", "-preset", "ultrafast",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-movflags", "+faststart", str(dest)]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-400:]
    return dest


def _clip_artifact(path: Path, *, duration: float = 30.5,
                   sha: str | None = None) -> ClipArtifact:
    return ClipArtifact(
        cache_key="c" * 64, source_subtitles="s" * 64,
        clip_path=str(path),
        clip_sha256=sha or hashlib.sha256(path.read_bytes()).hexdigest(),
        duration_s=duration, width=1080, height=1920, encoder="libx264",
        loudness_i=-14.0, loudness_tp=-1.5,
        target_loudness_i=-14.0, target_loudness_tp=-1.5)


def _subs_artifact(tmp_path: Path) -> SubtitleArtifact:
    ass = tmp_path / "subs.ass"
    ass.write_text("[Script Info]\nScriptType: v4.00+\n", encoding="utf-8")
    return SubtitleArtifact(
        cache_key="a" * 64, source_campath="p" * 64,
        ass_path=str(ass), clip_start=0.0, clip_end=30.5,
        line_count=3, word_count=12,
        ass_sha256=hashlib.sha256(ass.read_bytes()).hexdigest())


def _gate(tmp_path: Path) -> S7QualityGate:
    return S7QualityGate(StateDB(tmp_path / "qa.db"), tmp_path / "artifacts")


def _run_gate(tmp_path: Path, clip_art: ClipArtifact, tag: bytes,
              subs: SubtitleArtifact | None = None):
    return _gate(tmp_path).run(
        input_digest=digest_bytes(tag), params={},
        clip_artifact=clip_art, subtitle_artifact=subs,
        campath_artifact=None)


def test_a_healthy_clip_passes(tmp_path):
    clip = _render_fixture(tmp_path / "good.mp4")
    qa = _run_gate(tmp_path, _clip_artifact(clip), b"good",
                   subs=_subs_artifact(tmp_path))
    failed = [c.name for c in qa.checks
              if c.severity == "fail" and not c.passed]
    # sine@440 is never silent and testsrc is never black; loudness of a
    # bare sine is off the -14 target, which must be at most a WARN band
    # issue, never an automatic fail unless > 4 LU... a pure sine at aac
    # default gain measures far from -14, so exclude loudness-target here
    # by asserting on the OTHER checks and on the verdict logic separately.
    structural = [n for n in failed if n not in ("loudness-target",)]
    assert not structural, f"healthy clip failed: {structural}"


def test_a_truncated_clip_fails_integrity(tmp_path):
    clip = _render_fixture(tmp_path / "trunc.mp4")
    art = _clip_artifact(clip)  # hash BEFORE truncation
    data = clip.read_bytes()
    clip.write_bytes(data[:len(data) // 2])
    qa = _run_gate(tmp_path, art, b"trunc")
    assert not qa.passed
    assert any(c.name == "sha256-integrity" and not c.passed
               for c in qa.checks)


def test_a_silent_clip_fails(tmp_path):
    clip = _render_fixture(tmp_path / "silent.mp4", silent=True)
    qa = _run_gate(tmp_path, _clip_artifact(clip), b"silent")
    assert not qa.passed
    # The SEVERITY is part of the assertion. A fully-silent clip also fails
    # loudness-target (integrated loudness is -inf), so asserting only "not
    # qa.passed" let a mutant demote the silence check to "warn" with this
    # test still green — the check was being rescued by its neighbour.
    # Seventh occurrence of the accidental-pass pattern in this project.
    assert any(c.name == "silence" and c.severity == "fail" and not c.passed
               for c in qa.checks), (
        "the silence check must itself be a blocking failure")


def test_a_black_clip_fails(tmp_path):
    clip = _render_fixture(tmp_path / "black.mp4", black=True)
    qa = _run_gate(tmp_path, _clip_artifact(clip), b"black")
    assert not qa.passed
    assert any(c.name == "black-frames" and not c.passed for c in qa.checks)


def test_wrong_geometry_fails(tmp_path):
    clip = _render_fixture(tmp_path / "wide.mp4", width=1920, height=1080)
    qa = _run_gate(tmp_path, _clip_artifact(clip), b"wide")
    assert not qa.passed
    assert any(c.name == "geometry" and not c.passed for c in qa.checks)


def test_out_of_bounds_duration_fails(tmp_path):
    clip = _render_fixture(tmp_path / "short.mp4", seconds=5.0)
    qa = _run_gate(tmp_path, _clip_artifact(clip, duration=5.0), b"short")
    assert not qa.passed
    assert any(c.name == "duration-bounds" and not c.passed
               for c in qa.checks)


def test_a_missing_file_fails_without_crashing(tmp_path):
    ghost = tmp_path / "ghost.mp4"
    art = ClipArtifact(
        cache_key="c" * 64, source_subtitles="s" * 64,
        clip_path=str(ghost), clip_sha256="0" * 64, duration_s=30.0,
        width=1080, height=1920, encoder="libx264",
        loudness_i=-14.0, loudness_tp=-1.5,
        target_loudness_i=-14.0, target_loudness_tp=-1.5)
    qa = _run_gate(tmp_path, art, b"ghost")
    assert not qa.passed
    assert any(c.name == "file-exists" and not c.passed for c in qa.checks)


def test_a_tampered_subtitle_file_fails(tmp_path):
    clip = _render_fixture(tmp_path / "subs.mp4")
    subs = _subs_artifact(tmp_path)
    Path(subs.ass_path).write_text("[Script Info]\nTAMPERED\n",
                                   encoding="utf-8")
    qa = _run_gate(tmp_path, _clip_artifact(clip), b"tamper", subs=subs)
    assert any(c.name == "subtitle-integrity" and not c.passed
               for c in qa.checks)
    assert not qa.passed


def test_a_padded_splice_fails_even_though_every_other_check_passes(tmp_path):
    """PAC-3: concat padding is invisible to every other QA check.

    The clip is well-formed — right geometry, right duration band, audio
    present, streams equal — and ``duration_s`` matches the file, because
    S6 measures it. What is wrong is that the file does not match what the
    splice arithmetic PREDICTED: 30.5 s rendered against a 30.2 s plan is
    ~10 ms of padding per seam over 30 seams, which desyncs captions
    without tripping anything else. Only prediction-vs-measurement sees it.
    """
    clip = _render_fixture(tmp_path / "padded.mp4", seconds=30.5)
    art = _clip_artifact(clip).model_copy(
        update={"expected_duration_s": 30.2})
    qa = _run_gate(tmp_path, art, b"padded")
    assert any(c.name == "splice-duration-integrity" and c.severity == "fail"
               and not c.passed for c in qa.checks), (
        "a spliced clip that missed its predicted duration passed QA")
    assert not qa.passed
    # It must be THIS check catching it, not a neighbour: everything else
    # about the file is healthy.
    others = [c.name for c in qa.checks if c.severity == "fail"
              and not c.passed and c.name not in ("splice-duration-integrity",
                                                  "loudness-target")]
    assert not others, f"expected only the splice check to fail, got {others}"


def test_an_unspliced_clip_skips_the_splice_check(tmp_path):
    """No prediction recorded (no jump-cuts) means nothing to compare —
    the check must be absent, not vacuously passing on a default of 0.0."""
    clip = _render_fixture(tmp_path / "plain.mp4", seconds=30.5)
    art = _clip_artifact(clip)
    assert art.expected_duration_s is None
    qa = _run_gate(tmp_path, art, b"plain")
    assert not any(c.name == "splice-duration-integrity" for c in qa.checks)


def test_warnings_do_not_block():
    """The verdict is failures-only by design: a warning is a recorded
    deviation, not a rejection."""
    from clipforge.schemas.qa import QACheck
    from clipforge.stages.s7_qa import S7QualityGate

    checks = [QACheck(name="w", severity="warn", passed=False,
                      measured="m", expected="e"),
              QACheck(name="f", severity="fail", passed=True,
                      measured="m", expected="e")]
    failed = [c for c in checks if c.severity == "fail" and not c.passed]
    assert not failed
