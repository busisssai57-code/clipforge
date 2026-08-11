"""PAC-2: S6 must actually SPLICE the keep-intervals it is handed.

The panel proved the entire jump-cut path was revert-safe: an S6 that
ignored ``keep_intervals`` passed all 433 tests and QA blessed the desynced
clip. This test closes that hole with measurement, not assertion:

  * a synthetic source where every audio beep is born frame-locked to a
    video flash (same lavfi ``t``), at NTSC 30000/1001 — the rate where
    decimal-seconds splicing demonstrably falls off the pts grid;
  * REAL S6 run with quantized keeps that cut the silences between beeps;
  * flash and beep onsets measured from the DECODED clip must land where
    the TimeMap says, and A/V must still be locked at every seam.

Onsets are measured from raw decoded luma and PCM rather than from
``blackdetect``/``silencedetect``: the first draft of this test used those
filters and they reported phantom events at the leading audio fade and at
EOF, which says nothing about splice arithmetic. Decoded signal has no such
heuristics in it.

Kill-checks this is designed to fail on:
  - keeps ignored          -> container duration ~19 s vs expected ~5.8 s
  - decimal :.3f trims     -> extra frame at off-grid seam ends, video
                              drifts late vs audio, seam lock breaks
  - predicted-not-probed   -> artifact.duration_s stops matching the file
"""

from __future__ import annotations

import hashlib
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from clipforge.ffmpeg import find_binary, probe
from clipforge.pacing import (TimeMap, keeps_seconds_from_frames,
                              quantize_keeps_to_frames)
from clipforge.schemas.campath import CamPathArtifact, CropFrame
from clipforge.schemas.render import SubtitleArtifact
from clipforge.stages.base import digest_bytes
from clipforge.stages.s6_render import S6Render
from clipforge.state import StateDB

_NTSC = "30000/1001"
_FRAME_S = 1001.0 / 30000.0
WINDOW_S = 19.0
BEEP_PERIOD_S = 4.0
BEEP_LEN_S = 0.3

#: Raw (deliberately OFF-grid) keeps. Each contains exactly one flash+beep
#: at t = 4k; the silences between them are cut. 5 segments, 4 seams.
_RAW_KEEPS = [(0.0, 1.0), (3.8, 5.0), (7.9, 9.1),
              (11.85, 13.0), (15.9, 17.2)]

#: Audio analysis hop. Fine enough that hop quantization stays well under
#: one video frame (33 ms).
_HOP_S = 0.005


@pytest.fixture(scope="module")
def beepflash_source(tmp_path_factory) -> Path:
    """Black/silence except a white flash + 440 Hz beep every 4 s, both
    gated on the SAME lavfi clock — born in sync by construction."""
    dest = tmp_path_factory.mktemp("bf") / "beepflash.mp4"
    proc = subprocess.run(
        [find_binary("ffmpeg"), "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi",
         "-i", f"color=c=black:size=640x480:rate={_NTSC}:duration=20,"
               f"drawbox=c=white:t=fill:"
               f"enable='lt(mod(t,{BEEP_PERIOD_S}),{BEEP_LEN_S})'",
         "-f", "lavfi",
         "-i", "sine=frequency=440:sample_rate=48000:duration=20",
         "-af", f"volume='if(lt(mod(t,{BEEP_PERIOD_S}),{BEEP_LEN_S}),1,0)'"
                f":eval=frame",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", str(dest)],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert proc.returncode == 0, proc.stderr[-500:]
    return dest


def _minimal_subs(tmp: Path) -> SubtitleArtifact:
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
        "Dialogue: 0,0:00:00.50,0:00:01.00,K,,0,0,0,,SPLICE\n",
        encoding="utf-8")
    return SubtitleArtifact(
        cache_key="s" * 64, source_campath="p" * 64, ass_path=str(ass),
        clip_start=0.0, clip_end=WINDOW_S, line_count=1, word_count=1,
        ass_sha256=hashlib.sha256(ass.read_bytes()).hexdigest())


def _decode(cmd: list[str]) -> bytes:
    p = subprocess.run(cmd, capture_output=True)
    assert p.returncode == 0, (p.stderr or b"")[-500:].decode(
        "utf-8", "replace")
    return p.stdout


def _run_starts(mask: np.ndarray, times: np.ndarray,
                min_len: int) -> list[float]:
    """Start time of each contiguous True run at least ``min_len`` long."""
    padded = np.concatenate(([False], mask, [False]))
    edges = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return [float(times[s]) for s, e in zip(starts, ends) if e - s >= min_len]


def _flash_onsets(clip: Path, fps: Fraction) -> list[float]:
    """Onset of each white flash, from decoded luma (16x16 mean)."""
    raw = _decode([find_binary("ffmpeg"), "-nostdin", "-hide_banner",
                   "-i", str(clip), "-map", "0:v",
                   "-vf", "scale=16:16,format=gray",
                   "-f", "rawvideo", "-"])
    frames = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 256)
    means = frames.mean(axis=1)
    times = np.arange(len(means)) * float(1 / fps)
    # Flash is a full white frame; the resting frame is black plus a thin
    # progress bar and a caption, both far under half-brightness.
    return _run_starts(means > 128.0, times, min_len=2)


def _beep_onsets(clip: Path) -> list[float]:
    """Onset of each beep, from decoded PCM RMS envelope."""
    raw = _decode([find_binary("ffmpeg"), "-nostdin", "-hide_banner",
                   "-i", str(clip), "-map", "0:a", "-ac", "1",
                   "-ar", "48000", "-f", "s16le", "-"])
    pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32)
    hop = int(48000 * _HOP_S)
    usable = (len(pcm) // hop) * hop
    win = pcm[:usable].reshape(-1, hop)
    rms = np.sqrt((win * win).mean(axis=1))
    times = np.arange(len(rms)) * _HOP_S
    # Adaptive floor: loudnorm sets the absolute level, so threshold on a
    # fraction of the clip's own peak rather than a hardcoded dB.
    thresh = 0.20 * float(rms.max())
    assert thresh > 0, "decoded audio is silent"
    return _run_starts(rms > thresh, times, min_len=int(0.05 / _HOP_S))


def test_s6_splices_keeps_with_av_lock_at_every_seam(
        beepflash_source, tmp_path):
    info = probe(beepflash_source)
    assert info.fps_rational == _NTSC, "fixture must be NTSC to be adversarial"

    frame_pairs = quantize_keeps_to_frames(_RAW_KEEPS, _NTSC)
    keeps = keeps_seconds_from_frames(frame_pairs, _NTSC)
    tmap = TimeMap(keeps)
    expected_dur = tmap.duration()
    assert 5.0 < expected_dur < 7.0, "sanity: the cuts must really cut"

    n_frames = int(round(WINDOW_S * 30000 / 1001))
    campath = CamPathArtifact(
        cache_key="p" * 64, source_ranking="r" * 64,
        clip_start=0.0, clip_end=WINDOW_S, framing_mode="center",
        frames=[CropFrame(frame=f, x=180, y=0, w=270, h=480)
                for f in range(n_frames)],
        assignments=[], src_width=info.width, src_height=info.height,
        src_fps_rational=info.fps_rational)

    db = StateDB(tmp_path / "s.db")
    try:
        stage = S6Render(db, tmp_path / "artifacts")
        art = stage.run(
            input_digest=digest_bytes(b"pac2"),
            params={"width": 1080, "height": 1920, "encoder": "libx264",
                    "cq": 23, "audio_bitrate": "128k",
                    "keep_intervals": [[a, b] for a, b in keeps]},
            campath_artifact=campath,
            subtitle_artifact=_minimal_subs(tmp_path),
            video_path=beepflash_source, clips_dir=tmp_path / "clips")
    finally:
        db.close()

    clip = Path(art.clip_path)
    out_info = probe(clip)
    measured_dur = float(out_info.duration_s)

    # 1. Container duration is the COMPRESSED duration, not the window's.
    #    (keeps-ignored mutant: ~19 s here — off by 13 s, not by millis.)
    assert abs(measured_dur - expected_dur) <= 0.10, (
        f"spliced clip measures {measured_dur:.3f}s but the TimeMap says "
        f"{expected_dur:.3f}s — S6 did not honour keep_intervals")

    # 2. The artifact records the MEASURED duration, with the prediction
    #    beside it rather than in place of it.
    assert art.expected_duration_s == pytest.approx(expected_dur)
    assert art.duration_s == pytest.approx(measured_dur, abs=1e-3)

    # 3. Every flash and beep landed where the map says, and A/V stayed
    #    locked at every seam. One beep+flash per kept segment, so five of
    #    each — a count mismatch means a segment was dropped or doubled.
    fps = Fraction(out_info.fps_rational)
    flashes = _flash_onsets(clip, fps)
    beeps = _beep_onsets(clip)
    expected = [tmap.to_compressed(BEEP_PERIOD_S * k) for k in range(5)]
    assert len(flashes) == 5, f"flash onsets {flashes} (expected {expected})"
    assert len(beeps) == 5, f"beep onsets {beeps} (expected {expected})"

    for i, exp in enumerate(expected):
        assert abs(flashes[i] - exp) <= 3 * _FRAME_S, (
            f"flash {i}: video seam arithmetic is off (measured "
            f"{flashes[i]:.3f}s, map says {exp:.3f}s)")
        assert abs(beeps[i] - exp) <= 3 * _FRAME_S, (
            f"beep {i}: audio seam arithmetic is off (measured "
            f"{beeps[i]:.3f}s, map says {exp:.3f}s)")
        # The lock: video and audio are cut from the SAME frame integers,
        # so their disagreement must not grow with seam count. This is
        # exactly what decimal-second trims break — one swallowed frame
        # per off-grid seam, accumulating.
        assert abs(flashes[i] - beeps[i]) <= 2 * _FRAME_S, (
            f"seam {i}: A/V drifted apart (flash {flashes[i]:.3f}s vs "
            f"beep {beeps[i]:.3f}s)")
