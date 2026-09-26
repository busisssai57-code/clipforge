"""S3 must seek inside the WINDOW, not on the stream's timeline.

`bta watch` hands the DAG one window at a time. S1 stamps words with
``abs_offset + local``, so S2's candidate spans are absolute stream
seconds — but the file S3 opens starts at 0. Seeking to an absolute time
inside it read the wrong part of the window while the offset was small,
and read nothing at all once the offset passed the window's duration.

Measured on 2026-09-17: with segment_time_s=900, every window after the
first failed S3 with "no frames could be read", so a live broadcast could
only ever produce clips from its first fifteen minutes.
"""

from __future__ import annotations

import pytest

from clipforge.stages.s3_semantic import _local


def test_an_absolute_candidate_time_becomes_a_window_offset():
    # Window 2 of a live capture: the file holds 840..1800 s of stream.
    assert _local(1750.0, 840.0) == 910.0
    assert _local(900.0, 840.0) == 60.0


def test_a_manual_run_is_unchanged():
    """`bta process` on a whole file passes abs_offset 0, and the times
    must survive untouched — this is the path that already worked."""
    for t in (0.0, 12.5, 4000.0):
        assert _local(t, 0.0) == t


def test_a_candidate_starting_before_the_window_clamps_to_its_first_frame():
    """Edge snapping and words that straddle the seam can put a candidate
    a hair before the window. A negative seek index reads nothing."""
    assert _local(839.5, 840.0) == 0.0


@pytest.mark.parametrize("offset", [1740.0, 3600.0, 7200.0])
def test_the_offset_is_always_removed_before_seeking(offset):
    """The regression this file exists for: without subtraction the seek
    target is past the end of a 960 s window, which returns no frames."""
    window_duration = 960.0
    cand_start = offset + 30.0
    assert _local(cand_start, offset) < window_duration
    assert cand_start > window_duration, (
        "this test would prove nothing if the absolute time happened to "
        "land inside the window anyway")


def test_s3_passes_the_offset_through_to_frame_extraction(monkeypatch):
    """Structural guard on the wiring: the helper existing is not the
    fix — S3 calling it with the offset is."""
    import inspect

    from clipforge.stages import s3_semantic

    src = inspect.getsource(s3_semantic)
    assert src.count("_local(cand.start, abs_offset_s)") == 2, (
        "both the local and the cloud ranking paths must convert candidate "
        "times to window offsets")

    from clipforge import cli

    assert "abs_offset_s=abs_offset" in inspect.getsource(cli.process), (
        "process() no longer tells S3 where the window starts")
