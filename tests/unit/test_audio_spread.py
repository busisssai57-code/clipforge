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


# ------------------------------------- dropping the model's scratch audio

def test_dropping_scratch_audio_actually_runs(tmp_path):
    """The SUCCESS path, on a real file.

    It shipped with `log.warning` in a module that has no `log`, so the
    whole command died with NameError AFTER the audio was stripped and
    after GPU-hours had been spent. 1397 tests passed, because every one
    of them exercised the guard clauses and none reached the end of the
    function. A degrade path that is the only tested path is not tested.
    """
    import subprocess

    from clipforge.cli import _drop_scratch_audio
    from clipforge.ffmpeg import require_binary

    src = tmp_path / "seq.mp4"
    subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
         "-loglevel", "error",
         "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=0.5",
         "-f", "lavfi", "-i", "sine=frequency=300:duration=0.5",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(src)],
        check=True, capture_output=True)

    def streams(path):
        out = subprocess.run(
            [str(require_binary("ffprobe")), "-v", "error",
             "-show_entries", "stream=codec_type",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True)
        return set(out.stdout.split())

    assert streams(src) == {"video", "audio"}
    out = _drop_scratch_audio(src)
    assert out == src
    assert streams(src) == {"video"}, "the audio track must be gone"


def test_a_niche_that_wants_its_audio_keeps_it():
    from clipforge.cli import chosen_niche_keeps_audio

    assert chosen_niche_keeps_audio("ari_goat") is False
    assert chosen_niche_keeps_audio("geel_sketch") is True
    assert chosen_niche_keeps_audio("no_such_niche") is True


def test_the_spread_is_only_reported_when_the_audio_survives():
    """Warning about a 58 dB spread on a track just discarded is noise
    about noise, and leaves an operator unsure which line to believe."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli)
    assert "report_audio_spread(list(result.paths)) if keeps_audio else None" in src
