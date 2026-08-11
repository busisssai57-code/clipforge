"""Jump-cut pacing math (blueprint §9.2), pinned.

The property that matters: cuts land only in silences BETWEEN padded word
boundaries, the compressed timeline is monotonic, and the spec's duration
floor is never violated no matter how much dead air the window contains.
"""

from __future__ import annotations

import pytest

from clipforge.pacing import (TimeMap, compute_keep_intervals,
                              enforce_floor_on_frames, frames_to_seconds,
                              keeps_seconds_from_frames,
                              quantize_keeps_to_frames)


def test_a_long_gap_is_cut_with_padding():
    words = [(0.0, 10.0), (12.0, 40.0)]     # 2.0 s gap at 10-12
    keeps = compute_keep_intervals(words, 40.0, gap_threshold=0.6,
                                   pad=0.12, min_duration=29.5)
    assert keeps == [(0.0, pytest.approx(10.12)),
                     (pytest.approx(11.88), 40.0)]


def test_short_breaths_survive():
    words = [(0.0, 10.0), (10.5, 40.0)]     # 0.5 s gap < 0.6 threshold
    keeps = compute_keep_intervals(words, 40.0)
    assert keeps == [(0.0, 40.0)]


def test_duration_floor_restores_smallest_cuts_first():
    """A 36 s window with ~7.5 s of cuts would land at 28.5 s — below the
    floor. Restoring the SMALL cut brings it to 30.2 s; the big one stays.
    (With a window where even that is not enough, ALL cuts are restored —
    the floor is a hard guarantee, verified by the first failing version of
    this very test.)"""
    words = [(0.0, 5.0), (7.0, 20.0), (26.0, 36.0)]  # cuts ~1.8s and ~5.8s
    keeps = compute_keep_intervals(words, 36.0, min_duration=29.5)
    total = sum(b - a for a, b in keeps)
    assert total >= 29.5
    # The 6 s cut survived (compressed), the 2 s one was restored.
    assert len(keeps) == 2
    gap = keeps[1][0] - keeps[0][1]
    assert gap > 4.0, f"the BIG cut should remain, got gap {gap:.2f}s"


def test_no_words_means_no_cuts():
    assert compute_keep_intervals([], 30.0) == [(0.0, 30.0)]


def test_timemap_is_monotonic_and_totals_correctly():
    keeps = [(0.0, 10.0), (12.0, 30.0)]
    tm = TimeMap(keeps)
    assert tm.duration() == pytest.approx(28.0)
    ts = [0.0, 5.0, 9.99, 10.5, 11.9, 12.0, 20.0, 30.0]
    mapped = [tm.to_compressed(t) for t in ts]
    assert mapped == sorted(mapped), "map must never go backwards"
    # A word AFTER the 2 s cut lands exactly 2 s earlier.
    assert tm.to_compressed(15.0) == pytest.approx(13.0)
    # Interior of the cut collapses onto the seam.
    assert tm.to_compressed(11.0) == pytest.approx(10.0)


def test_timemap_is_kept():
    tm = TimeMap([(0.0, 10.0), (12.0, 30.0)])
    assert tm.is_kept(5.0) and tm.is_kept(12.0)
    assert not tm.is_kept(10.5)


def test_s5_retimes_words_through_the_map():
    """A word after a cut must appear EARLIER in the .ass by the cut size."""
    from clipforge.schemas import (TranscriptArtifact, TranscriptSegment,
                                   Word)
    from clipforge.schemas.campath import CamPathArtifact, CropFrame
    from clipforge.stages.base import digest_bytes
    from clipforge.stages.s5_subtitles import S5Subtitles
    from clipforge.state import StateDB
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        db = StateDB(tmp / "s.db")
        try:
            words = [Word(text="EARLY", start=1.0, end=1.4),
                     Word(text="LATE", start=8.0, end=8.4)]
            transcript = TranscriptArtifact(
                cache_key="t" * 64, source_path="x.mp4", abs_offset_s=0.0,
                language="en", diarization_ok=False, turns=[],
                segments=[TranscriptSegment(
                    start=0.0, end=9.0, speaker=None,
                    text="EARLY LATE.", words=words)])
            campath = CamPathArtifact(
                cache_key="p" * 64, source_ranking="r" * 64,
                clip_start=0.0, clip_end=10.0, framing_mode="center",
                frames=[CropFrame(frame=0, x=0, y=0, w=270, h=480)],
                assignments=[], src_width=640, src_height=480,
                src_fps_rational="30/1")
            stage = S5Subtitles(db, tmp / "artifacts")
            # Cut 2-7 (5 s of silence between the words).
            art = stage.run(
                input_digest=digest_bytes(b"jc"),
                params={"keep_intervals": [[0.0, 2.0], [7.0, 10.0]],
                        "animation": "pop"},
                transcript_artifact=transcript, campath_artifact=campath)
            content = Path(art.ass_path).read_text(encoding="utf-8")
            # LATE started at 8.0s originally; compressed = 8.0 - 5.0 = 3.0.
            assert "0:00:03.00" in content, content[-400:]
        finally:
            db.close()


# ------------------------------------------------- frame quantization
# Panel round 2026-07-28 (PAC-1): decimal-seconds quantization at
# 30000/1001 measured ±32 ms seam sawtooth, +200 ms caption drift and a
# +231 ms duration misreport — WORSE than not quantizing. The fix is
# integers on the exact pts grid; these pin its defining properties.

_NTSC = "30000/1001"


def test_frame_quantization_is_idempotent_on_the_ntsc_grid():
    """quantize -> seconds -> quantize must be a fixed point.

    This is exactly the property round(a*float_fps)/float_fps + ':.3f'
    violated: each pass through the decimal pipeline MOVED the boundary.
    """
    keeps = [(0.0, 10.0), (12.34, 29.97), (31.031, 59.5)]
    frames = quantize_keeps_to_frames(keeps, _NTSC)
    secs = keeps_seconds_from_frames(frames, _NTSC)
    assert quantize_keeps_to_frames(secs, _NTSC) == frames
    # And the seconds really are the pts grid: frame 300 at NTSC is
    # 300*1001/30000 = 10.01 exactly-as-a-float-of-the-exact-rational.
    assert frames[0] == (0, 300)
    assert secs[0][1] == frames_to_seconds(300, _NTSC) == 10.01


def test_frame_times_keep_sub_millisecond_precision():
    """S6 stamps audio trims at ``:.6f`` from these values, so anything
    that rounds them to milliseconds re-introduces the sub-frame error the
    whole quantizer exists to remove. Frame 371 at NTSC is
    371371/30000 = 12.3790333... — a value a 3-decimal trim would move.
    """
    from fractions import Fraction

    exact = float(Fraction(371 * 1001, 30000))
    got = frames_to_seconds(371, _NTSC)
    assert got == exact, f"{got!r} is not the exact rational {exact!r}"
    assert abs(got - round(got, 3)) > 1e-5, (
        "frame times have been rounded to milliseconds; the audio trim "
        "strings will no longer agree with the video frame numbers")


def test_subframe_keeps_are_dropped():
    """An interval narrower than one frame cannot be spliced; emitting a
    zero-length trim breaks ffmpeg's concat."""
    assert quantize_keeps_to_frames([(1.0, 1.01)], "30/1") == []
    assert quantize_keeps_to_frames([(1.0, 1.04)], "30/1") == [(30, 31)]


def test_floor_is_reenforced_after_quantization():
    """PAC-4: adversarially phased boundaries lose up to half a frame per
    edge; a 30.10 s pre-quantization plan measured 28.80 s rendered — under
    QA's hard 29.0 floor. The frame-domain floor must merge the SMALLEST
    gap first and clear min_duration."""
    # 30 fps, floor 29.5 s -> min 886 frames. Total here: 880.
    frames = [(0, 300), (310, 590), (700, 1000)]      # gaps: 10 and 110
    out = enforce_floor_on_frames(frames, 1080, "30/1", min_duration=29.5)
    assert out == [(0, 590), (700, 1000)]             # small gap closed
    assert sum(b - a for a, b in out) >= 886          # floor cleared


def test_floor_falls_back_to_full_window_when_merging_is_not_enough():
    frames = [(0, 100), (200, 300)]                   # 200 frames total
    out = enforce_floor_on_frames(frames, 1080, "30/1", min_duration=29.5)
    assert out == [(0, 1080)]
