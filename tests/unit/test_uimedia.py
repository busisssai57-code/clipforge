"""uimedia.py — timeline derivatives that must never lie.

Module rules under test here: a failure is reported, never faked (a flat
waveform claims silence), and cache keys include their parameters (a
40-column strip is not an 80-column strip) — plus the cache actually
caching, so the second request costs zero ffmpeg calls. Rule 3, path
confinement, is web.py's to enforce — URL paths are resolved and confined
there before this module ever sees one — so it is covered by the web
tests, not here.
"""

from __future__ import annotations

import json
import os
import struct
import types
from pathlib import Path

import pytest

from clipforge import uimedia
from clipforge.errors import FfmpegError
from clipforge.paths import Workspace


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws").ensure()


@pytest.fixture
def clip(ws):
    path = Path(ws.clips) / "clipA.mp4"
    path.write_bytes(b"\x00" * 1024)
    # a fixed past mtime so freshly written cache entries always win _fresh
    os.utime(path, (1_000_000, 1_000_000))
    return path


class _NeverRun:
    """A run() that fails the test if ffmpeg is invoked at all."""

    def __call__(self, cmd, **kw):
        raise AssertionError(f"ffmpeg must not run here: {cmd}")


class _FakeRun:
    def __init__(self):
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"\xff" * 256)


# ----------------------------------------------------------------- poster

def test_poster_prefers_the_shipped_thumbnail(ws, clip, monkeypatch):
    monkeypatch.setattr(uimedia, "run", _NeverRun())
    shipped = clip.with_suffix(".thumb.jpg")
    shipped.write_bytes(b"\xff")
    assert uimedia.poster(ws, clip) == shipped


def test_poster_generates_then_reuses(ws, clip, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")

    first = uimedia.poster(ws, clip)
    assert first == uimedia.cache_dir(ws) / "clipA.poster.jpg"
    assert first.is_file()
    assert len(fake.cmds) == 1

    monkeypatch.setattr(uimedia, "run", _NeverRun())
    assert uimedia.poster(ws, clip) == first  # cache hit, zero ffmpeg


def test_poster_missing_output_is_an_error(ws, clip, monkeypatch):
    monkeypatch.setattr(uimedia, "run", lambda cmd, **kw: None)  # writes nothing
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    with pytest.raises(FfmpegError, match="poster frame not produced"):
        uimedia.poster(ws, clip)


def test_poster_negative_seek_is_clamped(ws, clip, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    uimedia.poster(ws, clip, at_s=-5.0)
    (cmd,) = fake.cmds
    assert cmd[cmd.index("-ss") + 1] == "0.000"


# -------------------------------------------------------------- filmstrip

def _patch_probe(monkeypatch, width, height):
    import clipforge.ffmpeg as ff

    monkeypatch.setattr(
        ff, "probe",
        lambda p: types.SimpleNamespace(width=width, height=height))


def test_filmstrip_requires_a_positive_duration(ws, clip):
    with pytest.raises(ValueError, match="duration_s must be positive"):
        uimedia.filmstrip(ws, clip, duration_s=0.0)


def test_filmstrip_builds_sprite_and_measures_tiles(ws, clip, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    _patch_probe(monkeypatch, width=40 * 52, height=92)

    strip = uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)
    assert strip.columns == 40
    assert (strip.tile_w, strip.tile_h) == (52, 92)  # measured, not assumed
    (cmd,) = fake.cmds
    vf = cmd[cmd.index("-vf") + 1]
    # fps spaces exactly `columns` frames across the clip: 40/30s
    assert "fps=1.333333" in vf
    assert "tile=40x1" in vf


def test_filmstrip_cache_key_includes_columns(ws, clip, monkeypatch):
    """Rule 2 verbatim: a 40-column strip must not answer for 80."""
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    _patch_probe(monkeypatch, width=40 * 52, height=92)
    uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)

    _patch_probe(monkeypatch, width=80 * 52, height=92)
    strip80 = uimedia.filmstrip(ws, clip, duration_s=30.0, columns=80)
    assert len(fake.cmds) == 2  # second geometry really rendered
    assert strip80.path.name != "clipA.strip40.jpg"
    assert "strip80" in strip80.path.name


def test_filmstrip_cache_hit_is_free(ws, clip, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    _patch_probe(monkeypatch, width=40 * 52, height=92)
    first = uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)

    monkeypatch.setattr(uimedia, "run", _NeverRun())
    again = uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)
    assert again == first


def test_filmstrip_corrupt_meta_regenerates(ws, clip, monkeypatch):
    """A half-written cache entry is regenerated, not trusted."""
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    _patch_probe(monkeypatch, width=40 * 52, height=92)
    uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)

    meta = uimedia.cache_dir(ws) / "clipA.strip40.json"
    meta.write_text("{trunc", encoding="utf-8")
    os.utime(meta, None)  # still "fresh" — corruption, not staleness
    uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)
    assert len(fake.cmds) == 2
    # and the meta was rewritten valid
    assert json.loads(meta.read_text(encoding="utf-8"))["tile_w"] == 52


def test_filmstrip_missing_output_is_an_error(ws, clip, monkeypatch):
    """ffmpeg exiting clean without a sprite is still a failure."""
    monkeypatch.setattr(uimedia, "run", lambda cmd, **kw: None)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    with pytest.raises(FfmpegError, match="filmstrip not produced"):
        uimedia.filmstrip(ws, clip, duration_s=30.0, columns=40)


def test_filmstrip_columns_clamped(ws, clip, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    _patch_probe(monkeypatch, width=120 * 52, height=92)
    strip = uimedia.filmstrip(ws, clip, duration_s=30.0, columns=9999)
    assert strip.columns == 120
    assert "strip120" in strip.path.name


# --------------------------------------------------------------- waveform

def _pcm(*samples: int) -> bytes:
    return struct.pack(f"<{len(samples)}h", *samples)


def _patch_decode(monkeypatch, stdout: bytes, returncode: int = 0,
                  stderr: bytes = b""):
    calls = []

    def fake(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=stdout, returncode=returncode,
                                     stderr=stderr)

    monkeypatch.setattr(uimedia.subprocess, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    return calls


def test_waveform_peaks_are_per_bucket_maxima(ws, clip, monkeypatch):
    # 1s of audio at 8kHz: first half at ±0.5 full-scale, second at -1.0
    samples = [16384, -16384] * 2000 + [-32768] * 4000
    _patch_decode(monkeypatch, _pcm(*samples))

    blob = uimedia.waveform(ws, clip, duration_s=1.0, buckets=50)
    assert blob["buckets"] == 50
    assert blob["decoded_s"] == 1.0
    assert blob["sample_rate"] == 8000
    peaks = blob["peaks"]
    assert peaks[:25] == [0.5] * 25
    assert peaks[25:] == [1.0] * 25


def test_waveform_keeps_transients_it_does_not_average_them(ws, clip,
                                                            monkeypatch):
    """Peak, not RMS — pinned with a signal where the two DIFFER.

    ``test_waveform_peaks_are_per_bucket_maxima`` above cannot see this
    distinction: its windows are constant-magnitude (``[16384, -16384]``,
    then a run of ``-32768``), so the mean of the absolute values equals
    the maximum exactly, and swapping ``max`` for a mean leaves every
    assertion in it true. A mutation sweep caught that.

    Here each bucket is one full-scale sample and three silent ones —
    a speech onset, which is precisely what the timeline is read to find.
    Peak keeps it at 1.0; any averaging buries it at 0.25.
    """
    samples = ([32767] + [0, 0, 0]) * 50  # 200 samples -> 4 per bucket
    _patch_decode(monkeypatch, _pcm(*samples))

    blob = uimedia.waveform(ws, clip, duration_s=1.0, buckets=50)
    assert blob["peaks"] == [1.0] * 50, (
        "a mean over these windows reads 0.25 and the onset vanishes")


def test_waveform_no_audio_is_an_error_not_a_flat_line(ws, clip, monkeypatch):
    """Rule 1 verbatim: a flat line claims the clip is silent."""
    _patch_decode(monkeypatch, b"", returncode=1,
                  stderr=b"Stream map '0:a:0' matches no streams")
    with pytest.raises(FfmpegError, match="no decodable audio") as err:
        uimedia.waveform(ws, clip, duration_s=1.0)
    assert "matches no streams" in err.value.stderr_tail
    # and nothing poisoned the cache for the next request
    assert not list(uimedia.cache_dir(ws).glob("*.json"))


def test_waveform_single_stray_byte_is_empty_audio(ws, clip, monkeypatch):
    _patch_decode(monkeypatch, b"\x7f")  # truncates to zero samples
    with pytest.raises(FfmpegError, match="empty audio stream"):
        uimedia.waveform(ws, clip, duration_s=1.0)


def test_waveform_odd_trailing_byte_is_dropped(ws, clip, monkeypatch):
    _patch_decode(monkeypatch, _pcm(1000, 2000, -3000) + b"\x7f")
    blob = uimedia.waveform(ws, clip, duration_s=1.0, buckets=999)
    # 3 usable samples, one per window — far fewer peaks than the 999
    # requested buckets, and the reported count follows the peaks
    assert blob["peaks"] == [round(1000 / 32768, 4),
                             round(2000 / 32768, 4),
                             round(3000 / 32768, 4)]
    assert blob["buckets"] == len(blob["peaks"])


def test_waveform_ragged_final_window_truncates_to_buckets(ws, clip,
                                                           monkeypatch):
    # 130 samples into 64 buckets: 2-sample windows yield 65 raw peaks;
    # the payload must never exceed the bucket count it reports
    _patch_decode(monkeypatch, _pcm(*([1000] * 130)))
    blob = uimedia.waveform(ws, clip, duration_s=1.0, buckets=64)
    assert len(blob["peaks"]) == 64
    assert blob["buckets"] == 64


def test_waveform_timeout_is_reported(ws, clip, monkeypatch):
    import subprocess as sp

    def fake(cmd, **kw):
        raise sp.TimeoutExpired(cmd, 120.0)

    monkeypatch.setattr(uimedia.subprocess, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    with pytest.raises(FfmpegError, match="timed out"):
        uimedia.waveform(ws, clip, duration_s=1.0)


def test_waveform_unlaunchable_ffmpeg_is_reported(ws, clip, monkeypatch):
    def fake(cmd, **kw):
        raise OSError("exec format error")

    monkeypatch.setattr(uimedia.subprocess, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    with pytest.raises(FfmpegError, match="cannot execute"):
        uimedia.waveform(ws, clip, duration_s=1.0)


def test_waveform_reports_short_decode_honestly(ws, clip, monkeypatch):
    """decoded_s < duration_s tells the caller the decode truncated,
    instead of stretching the envelope across the whole timeline."""
    _patch_decode(monkeypatch, _pcm(*([1000] * 4000)))  # only 0.5s decoded
    blob = uimedia.waveform(ws, clip, duration_s=2.0)
    assert blob["duration_s"] == 2.0
    assert blob["decoded_s"] == 0.5


def test_waveform_cached_by_bucket_count(ws, clip, monkeypatch):
    calls = _patch_decode(monkeypatch, _pcm(*([1000] * 8000)))
    first = uimedia.waveform(ws, clip, duration_s=1.0, buckets=100)
    assert len(calls) == 1

    # same params → served from cache, no decode
    def never(cmd, **kw):
        raise AssertionError("decode must not run on a cache hit")

    monkeypatch.setattr(uimedia.subprocess, "run", never)
    assert uimedia.waveform(ws, clip, duration_s=1.0, buckets=100) == first

    # different bucket count → its own cache entry, decode runs again
    calls2 = _patch_decode(monkeypatch, _pcm(*([1000] * 8000)))
    uimedia.waveform(ws, clip, duration_s=1.0, buckets=200)
    assert len(calls2) == 1


def test_waveform_bucket_bounds(ws, clip, monkeypatch):
    _patch_decode(monkeypatch, _pcm(*([1000] * 8000)))
    blob = uimedia.waveform(ws, clip, duration_s=1.0, buckets=3)
    assert blob["buckets"] == 50  # floor
    _patch_decode(monkeypatch, _pcm(*([1000] * 8000)))
    blob = uimedia.waveform(ws, clip, duration_s=1.0, buckets=10 ** 6)
    assert blob["buckets"] <= 4000  # ceiling


# -------------------------------------------------------------- freshness

def test_stale_cache_is_regenerated_when_the_clip_changes(ws, clip,
                                                          monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(uimedia, "run", fake)
    monkeypatch.setattr(uimedia, "require_binary", lambda n: "ffmpeg")
    uimedia.poster(ws, clip)
    assert len(fake.cmds) == 1

    # the clip is re-rendered: newer than its cached poster
    clip.write_bytes(b"\x11" * 2048)
    os.utime(clip, (2_000_000_000, 2_000_000_000))
    uimedia.poster(ws, clip)
    assert len(fake.cmds) == 2  # derivative rebuilt, not served stale
