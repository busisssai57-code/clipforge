"""Footage that already carries words gets none of ours on top.

A shipped clip of this project showed why: the source's own caption sat
at the top of frame, our hook card landed on it, and neither could be
read. Short-form video very often arrives already captioned.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipforge import subdetect
from clipforge.config import AppConfig


@pytest.fixture()
def cfg():
    return AppConfig()


def detect(cfg, *, answer, track=False, frames=(b"jpeg",), monkeypatch=None):
    monkeypatch.setattr(subdetect, "has_subtitle_track", lambda v: track)
    return subdetect.detect(Path("clip.mp4"), duration_s=30.0, cfg=cfg,
                            frames=lambda *a, **kw: list(frames),
                            vision=lambda *a, **kw: answer)


# ------------------------------------------------------------ the answers

def test_a_subtitle_track_is_answer_enough(cfg, monkeypatch):
    """ffprobe answers in milliseconds; the vision model costs GPU."""
    asked = []
    monkeypatch.setattr(subdetect, "has_subtitle_track", lambda v: True)
    found = subdetect.detect(Path("c.mp4"), duration_s=30.0, cfg=cfg,
                             frames=lambda *a, **kw: asked.append(1) or [],
                             vision=lambda *a, **kw: None)
    assert found and found.source == "track"
    assert not asked, "the GPU was asked a question ffprobe had answered"


def test_the_vision_model_finds_burned_in_words(cfg, monkeypatch):
    found = detect(cfg, answer=True, monkeypatch=monkeypatch)
    assert found.present and found.source == "vision"


def test_clean_footage_gets_our_captions(cfg, monkeypatch):
    assert not detect(cfg, answer=False, monkeypatch=monkeypatch)


def test_unsure_means_no(cfg, monkeypatch):
    """A clip that needed captions and got none is worse than a captioned
    clip that kept its own: the first is silent for a viewer on mute."""
    found = detect(cfg, answer=None, monkeypatch=monkeypatch)
    assert not found.present and found.source == "unknown"


def test_the_check_can_be_turned_off(cfg, monkeypatch):
    cfg.s5.detect_existing = False
    asked = []
    monkeypatch.setattr(subdetect, "has_subtitle_track", lambda v: False)
    found = subdetect.detect(Path("c.mp4"), duration_s=30.0, cfg=cfg,
                             frames=lambda *a, **kw: asked.append(1) or [],
                             vision=lambda *a, **kw: True)
    assert not found.present and found.source == "flag"
    assert not asked


def test_unreadable_frames_do_not_stop_the_clip(cfg, monkeypatch):
    monkeypatch.setattr(subdetect, "has_subtitle_track", lambda v: False)

    def boom(*a, **kw):
        raise OSError("cannot decode")

    found = subdetect.detect(Path("c.mp4"), duration_s=30.0, cfg=cfg,
                             frames=boom, vision=lambda *a, **kw: True)
    assert not found.present


# ------------------------------------------------- reading the model's prose

@pytest.mark.parametrize("text, expected", [
    ("YES - there are white captions at the bottom", True),
    ("yes, burned-in subtitles", True),
    ("NO. The only text is a shop sign.", False),
    ("no captions, just a logo", False),
    ("It is hard to say from these frames", None),
    ("", None),
])
def test_yes_or_no_is_read_out_of_the_prose(text, expected):
    assert subdetect._verdict_from_text(text) is expected


def test_the_question_excludes_scenery_and_watermarks():
    """A shop sign, a jersey number or our own handle are not captions,
    and treating them as captions would silence our own."""
    ask = subdetect.ASK.lower()
    assert "signs" in ask and "watermark" in ask
    assert "one word" in ask


# ----------------------------------------------------------- what it changes

def test_the_pipeline_asks_once_and_acts_on_it():
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.process)
    assert "subdetect.detect(" in src, "the pipeline never asks"
    assert src.count("subdetect.detect(") == 1, (
        "asked per candidate; the answer is a property of the footage and "
        "the VL model is not cheap")
    assert "subdetect.caption_params(" in src, (
        "the caption decision is back to an unheld branch")
    assert "plain=bool(existing_subs)" in src, (
        "the export pack still carries the hashtag wall")


def test_a_plain_pack_is_a_hook_and_a_caption(tmp_path):
    from clipforge import export_pack

    clip = tmp_path / "c.mp4"
    clip.write_bytes(bytes(16))
    words = "rosemary oil scalp growth routine before after results weeks"
    plain = export_pack.build_pack(clip, title="A title", hook="wait for it",
                                   transcript_text=words, write=False,
                                   plain=True)
    assert plain.hashtags == [], "a hashtag wall marks the account automated"
    assert plain.chapters == []
    assert plain.caption and plain.title == "A title"

    normal = export_pack.build_pack(clip, title="A title", hook="wait for it",
                                    transcript_text=words, write=False)
    assert normal.hashtags, "ordinary clips still get their hashtags"


# ------------------------------------------------- what S5 is asked for

def test_already_captioned_footage_asks_s5_for_nothing():
    from clipforge.subdetect import caption_params

    base = {"theme": "viral", "margin_v": 337}
    got = caption_params(base, existing=True, hook_text="wait for it",
                         keep_intervals=[[0.0, 5.0]], uppercase=True)
    assert got["captions"] == "already_in_picture"
    assert got["hook_text"] == "", (
        "the hook would be burned twice: once in the subtitle track and "
        "once as the overlay")
    assert got["theme"] == "viral", "the rest of the request is untouched"


def test_clean_footage_asks_for_the_hook_and_the_words():
    from clipforge.subdetect import caption_params

    got = caption_params({"theme": "viral"}, existing=False,
                         hook_text="wait for it", keep_intervals=[],
                         uppercase=False)
    assert got["hook_text"] == "wait for it"
    assert "captions" not in got, (
        "the marker must not appear for ordinary footage, or every clip's "
        "S5 cache key changes")


def test_the_marker_is_in_the_cache_key_path():
    """Same window, captioned or not, is two different clips. The marker
    rides in params, which is what S5 hashes."""
    from clipforge.subdetect import caption_params

    a = caption_params({}, existing=True, hook_text="h", keep_intervals=[],
                       uppercase=True)
    b = caption_params({}, existing=False, hook_text="h", keep_intervals=[],
                       uppercase=True)
    assert a != b


def test_s5_really_produces_an_empty_script_for_captioned_footage(tmp_path):
    """The branch that string-matching could not hold: run S5 with the
    marker and look at what comes out. The first version of this built an
    artifact with fields the schema does not have, and nothing noticed
    because no test ran the path."""
    from clipforge.state import StateDB
    from clipforge.stages.s5_subtitles import S5Subtitles

    class FakeCampath:
        cache_key = "c" * 64
        clip_start = 10.0
        clip_end = 40.0

    db = StateDB(tmp_path / "state.sqlite3")
    try:
        s5 = S5Subtitles(db, tmp_path / "artifacts")
        art = s5.run(input_digest="d" * 64, params={"captions": "already_in_picture"},
                     transcript_artifact=object(), campath_artifact=FakeCampath())
    finally:
        db.close()

    assert art.line_count == 0 and art.word_count == 0
    body = Path(art.ass_path).read_text(encoding="utf-8")
    assert "[Events]" in body, "libass logs a parse error on a zero-byte script"
    assert "Dialogue:" not in body, "it would draw words over the footage's own"
    assert art.clip_start == 10.0 and art.clip_end == 40.0
