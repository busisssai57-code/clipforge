"""Bring-your-own subtitles: an SRT/VTT replaces the ASR (clipforge.ingest.
subtitles + the S1 branch that consumes it).

Two promises are pinned: the parse handles the shapes real caption files
come in, and a run with subtitles produces a normal transcript WITHOUT
touching the speech engine — marked honestly as word-estimated and
speakerless.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.errors import ClipForgeError
from clipforge.ingest.subtitles import (Cue, cues_to_aligned, discover_beside,
                                        from_params, parse_subtitles, to_params)
from clipforge.stages.s1_transcribe import S1Transcribe
from clipforge.state import StateDB


SRT = """1
00:00:01,000 --> 00:00:03,000
Hello there world

2
00:00:04,500 --> 00:00:06,000
Second line here.
"""

VTT = """WEBVTT

NOTE this is a comment

00:00:01.000 --> 00:00:03.000 position:50%
Hello <c>there</c> world

cue-2
00:00:04.500 --> 00:00:06.000
Second line here.
"""


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


# --------------------------------------------------------------- parsing

def test_parses_srt(tmp_path):
    cues = parse_subtitles(_write(tmp_path, "s.srt", SRT))
    assert [(c.start, c.end) for c in cues] == [(1.0, 3.0), (4.5, 6.0)]
    assert cues[0].text == "Hello there world"


def test_parses_vtt_and_strips_tags_and_notes(tmp_path):
    cues = parse_subtitles(_write(tmp_path, "s.vtt", VTT))
    assert [(c.start, c.end) for c in cues] == [(1.0, 3.0), (4.5, 6.0)]
    # inline <c> tags and the NOTE block are gone; words survive.
    assert cues[0].text == "Hello there world"


def test_out_of_order_cues_are_sorted(tmp_path):
    txt = ("1\n00:00:10,000 --> 00:00:12,000\nlater\n\n"
           "2\n00:00:01,000 --> 00:00:02,000\nearlier\n")
    cues = parse_subtitles(_write(tmp_path, "s.srt", txt))
    assert [c.text for c in cues] == ["earlier", "later"]


def test_a_file_with_no_cues_raises(tmp_path):
    with pytest.raises(ClipForgeError):
        parse_subtitles(_write(tmp_path, "junk.srt", "not subtitles at all\n"))


def test_hours_field_is_read(tmp_path):
    txt = "1\n01:02:03,500 --> 01:02:05,000\nlate in the stream\n"
    cues = parse_subtitles(_write(tmp_path, "s.srt", txt))
    assert cues[0].start == pytest.approx(3723.5)


# ------------------------------------------------------- word apportioning

def test_words_span_the_cue_monotonically():
    aligned = cues_to_aligned([Cue(1.0, 3.0, "Hello there world")])
    words = aligned["segments"][0]["words"]
    assert [w["word"] for w in words] == ["Hello", "there", "world"]
    assert words[0]["start"] == pytest.approx(1.0)
    assert words[-1]["end"] == pytest.approx(3.0)
    times = [w["start"] for w in words] + [words[-1]["end"]]
    assert times == sorted(times), "word times are not monotonic"


def test_apportioning_is_deterministic():
    a = cues_to_aligned([Cue(0.0, 5.0, "a slightly longer sentence here now")])
    b = cues_to_aligned([Cue(0.0, 5.0, "a slightly longer sentence here now")])
    assert a == b


# ------------------------------------------------------------ params bridge

def test_params_round_trip():
    cues = [Cue(1.0, 3.0, "one two"), Cue(4.5, 6.0, "three")]
    assert from_params(to_params(cues)) == cues


def test_discovery_matches_exact_stem_only(tmp_path):
    (tmp_path / "video.mp4").write_bytes(b"x")
    assert discover_beside(tmp_path / "video.mp4") is None
    (tmp_path / "video.srt").write_text("1\n00:00:01,000 --> 00:00:02,000\nhi\n")
    assert discover_beside(tmp_path / "video.mp4").name == "video.srt"
    # a different stem must NOT be picked up
    (tmp_path / "video.es.srt").write_text("x")
    assert discover_beside(tmp_path / "other.mp4") is None


# ---------------------------------------------------- S1 consumes subtitles

class _ExplodingEngine:
    """Any use of the speech engine is a bug on the subtitle path."""
    def __getattr__(self, name):
        raise AssertionError(f"S1 touched the ASR engine ({name}) with "
                             "subtitles supplied — it must skip it entirely")


def test_s1_builds_the_transcript_from_subtitles_without_asr(tmp_path):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"\0" * 64)  # must exist; never decoded on this path
    db = StateDB(tmp_path / "s.db")
    try:
        s1 = S1Transcribe(db=db, artifacts_dir=tmp_path / "art",
                          engine_factory=lambda: _ExplodingEngine())
        cues = parse_subtitles(_write(tmp_path, "c.srt", SRT))
        art = s1.run(input_digest="d0", params={"subtitles": to_params(cues)},
                     media_path=media)
    finally:
        db.close()

    assert art.words_aligned is False, "subtitle word times must read as estimated"
    assert art.diarization_ok is False, "a caption file has no speakers"
    assert len(art.segments) == 2
    assert art.segments[0].text == "Hello there world"
    assert art.segments[0].words, "words were not apportioned onto the segment"


def test_subtitle_content_folds_into_the_s1_cache_key(tmp_path):
    media = tmp_path / "clip.mp4"; media.write_bytes(b"\0" * 64)
    db = StateDB(tmp_path / "s.db")
    try:
        s1 = S1Transcribe(db=db, artifacts_dir=tmp_path / "art",
                          engine_factory=lambda: _ExplodingEngine())
        one = parse_subtitles(_write(tmp_path, "a.srt", SRT))
        two = parse_subtitles(_write(tmp_path, "b.srt",
              "1\n00:00:01,000 --> 00:00:03,000\nDifferent words entirely\n"))
        a = s1.run(input_digest="d0", params={"subtitles": to_params(one)},
                   media_path=media)
        b = s1.run(input_digest="d0", params={"subtitles": to_params(two)},
                   media_path=media)
    finally:
        db.close()
    assert a.cache_key != b.cache_key, "different captions must re-transcribe"
