"""A sequence whose shots differ wildly in loudness plays broken.

Measured on a delivered three-shot sequence 2026-09-05: -84.3 and -74.4
dBFS for the first two shots and -26.9 for the third -- 3.8 seconds of
silence and then a jump to the edge of clipping, inside 5.7 seconds.

It is reported rather than corrected, and these tests pin that choice as
much as the arithmetic. Normalising cannot fix it: EBU R128 gates silence
out, so the integrated measurement reflects only the shot that HAS audio,
and that shot already sits at the true-peak ceiling. A loudnorm pass was
tried on this exact file and moved it by 0.0 LUFS.
"""

from __future__ import annotations

from clipforge.socialpost import AUDIO_SPREAD_WARN_DB, audio_spread

REAL = {"shot_00.mp4": -84.3, "shot_01.mp4": -74.4, "shot_02.mp4": -26.9}


def test_the_real_sequence_is_flagged():
    spread, quietest, loudest = audio_spread(REAL)
    assert round(spread, 1) == 57.4
    assert quietest == "shot_00.mp4"
    assert loudest == "shot_02.mp4"
    assert spread > AUDIO_SPREAD_WARN_DB


def test_a_consistent_sequence_is_not_flagged():
    spread, _q, _l = audio_spread({"a": -21.0, "b": -24.0, "c": -22.5})
    assert spread < AUDIO_SPREAD_WARN_DB


def test_one_shot_has_no_spread():
    """A single shot cannot be inconsistent with itself."""
    assert audio_spread({"only.mp4": -30.0}) is None


def test_an_unmeasurable_shot_is_not_evidence_of_a_quiet_one():
    """None means "could not measure", not "silent". Treating it as a
    level would invent a spread out of an ffmpeg failure."""
    assert audio_spread({"a.mp4": -20.0, "b.mp4": None}) is None
    assert audio_spread({"a.mp4": None, "b.mp4": None}) is None


def test_three_shots_with_one_unmeasurable_still_compare_the_rest():
    got = audio_spread({"a.mp4": -80.0, "b.mp4": None, "c.mp4": -20.0})
    assert got is not None
    assert round(got[0], 1) == 60.0


def test_generate_reports_it():
    """Wiring. The measurement is worthless if nothing calls it."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli)
    assert "report_audio_spread(list(result.paths))" in src
