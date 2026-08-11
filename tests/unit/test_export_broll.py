"""Export pack and B-roll planning, pinned.

The export pack exists INSTEAD of auto-publishing, so its failures are
the ones a person only notices after pasting: a caption cut mid-word, a
caption the platform silently rejects for length, a thumbnail that is a
flat fill. B-roll's failure is subtler — cutting away at the wrong moment
buries the speaker, which is the opposite of what b-roll is for.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

from clipforge.broll import (HOOK_PROTECT_S, MAX_COVERAGE, MIN_INSERT_S,
                             coverage, overlay_filter, plan_broll)
from clipforge.errors import ClipForgeError
from clipforge.export_pack import (PLATFORM_LIMITS, build_hashtags,
                                   build_pack, chapters_from_segments,
                                   fit_caption, grab_thumbnail, keywords)


@dataclass
class Seg:
    start: float
    end: float
    text: str


def _video(dest: Path, *, source="testsrc2", seconds=3.0) -> Path:
    sep = ":" if "=" in source else "="
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-f", "lavfi",
         "-i", f"{source}{sep}size=128x224:rate=10:d={seconds}",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
        capture_output=True, timeout=180)
    assert proc.returncode == 0, (proc.stderr or b"")[-300:]
    return dest


# ------------------------------------------------------ keywords/tags

def test_keywords_skip_filler_and_short_words():
    ks = keywords("The discipline of the morning routine builds discipline")
    assert "discipline" in ks
    assert "the" not in ks and "of" not in ks


def test_hashtags_come_from_this_clip_not_a_fixed_list():
    """Every clip once carried the same five hardcoded tags, which marks
    a whole account as automated."""
    a = build_hashtags("solitude and discipline in the mountains")
    b = build_hashtags("coffee roasting and espresso extraction")
    assert a != b
    assert all(t.startswith("#") for t in a)


def test_hashtags_are_deduped_and_capped():
    tags = build_hashtags("focus focus focus discipline discipline", limit=3)
    assert len(tags) <= 3 and len(set(tags)) == len(tags)


# ------------------------------------------------------- caption fit

def test_a_short_caption_is_untouched():
    text, trimmed = fit_caption("hello world", 280)
    assert text == "hello world" and trimmed is False


def test_a_long_caption_is_cut_at_a_word_boundary():
    """Cutting mid-word looks like a bug to every viewer."""
    text, trimmed = fit_caption("alpha bravo charlie delta echo " * 40, 60)
    assert trimmed is True
    assert len(text) <= 61
    assert "…" in text
    assert not text.rstrip("…").endswith(" ")
    # the last surviving token must be a whole word
    assert text.rstrip("… ").split()[-1] in {"alpha", "bravo", "charlie",
                                             "delta", "echo"}


def test_trimming_is_reported_not_silent():
    """Silently over-length text is rejected by the platform after the
    operator has already pasted it."""
    _t, trimmed = fit_caption("x " * 500, 100)
    assert trimmed is True


# -------------------------------------------------------- chapters

def test_chapters_are_timestamped_and_spaced():
    segs = [Seg(0.0, 2.0, "First thing."), Seg(2.1, 4.0, "Too close."),
            Seg(9.0, 12.0, "Second thing.")]
    chaps = chapters_from_segments(segs, min_gap_s=4.0)
    assert [c["time"] for c in chaps] == ["0:00", "0:09"]


def test_chapters_survive_no_segments():
    assert chapters_from_segments(None) == []


# ------------------------------------------------------- thumbnail

def test_a_blank_thumbnail_is_refused(tmp_path):
    """The thumbnail is the most-seen frame of a clip; a flat fill is the
    worst possible silent failure."""
    blank = _video(tmp_path / "blank.mp4", source="color=c=#5c4a30")
    with pytest.raises(ClipForgeError) as err:
        grab_thumbnail(blank, tmp_path / "t.jpg", at_s=1.0)
    assert "no picture" in str(err.value)
    assert not (tmp_path / "t.jpg").exists()


def test_a_real_thumbnail_is_written(tmp_path):
    real = _video(tmp_path / "real.mp4")
    out = grab_thumbnail(real, tmp_path / "t.jpg", at_s=1.0)
    assert out.is_file() and out.stat().st_size > 500


# ------------------------------------------------------ full pack

def test_the_pack_covers_every_platform_within_its_limit(tmp_path):
    clip = _video(tmp_path / "clip.mp4")
    pack = build_pack(clip, title="Discipline",
                      transcript_text="discipline solitude mountains focus",
                      hook="Stop waiting for motivation",
                      segments=[Seg(0, 2, "One."), Seg(6, 8, "Two.")])
    assert set(pack.platforms) == set(PLATFORM_LIMITS)
    for name, blob in pack.platforms.items():
        assert len(blob["caption"]) <= PLATFORM_LIMITS[name], name
    assert clip.with_suffix(".export.json").is_file()


def test_the_pack_is_marked_draft_and_posts_nothing(tmp_path):
    """The Authorization Law is not relaxed by adding a copy button."""
    clip = _video(tmp_path / "clip.mp4")
    build_pack(clip, transcript_text="focus")
    blob = json.loads(clip.with_suffix(".export.json").read_text("utf-8"))
    assert "DRAFT" in blob["status"]
    assert "nothing was posted" in blob["status"].lower()


def test_a_missing_clip_is_refused(tmp_path):
    with pytest.raises(ClipForgeError):
        build_pack(tmp_path / "nope.mp4")


# --------------------------------------------------------- b-roll

def _talk(n=8, gap=0.6, dur=2.0):
    segs, t = [], 0.0
    for i in range(n):
        segs.append(Seg(t, t + dur, f"Sentence {i} about discipline focus."))
        t += dur + gap
    return segs


def test_broll_never_covers_the_hook():
    """The hook is the whole reason a viewer stays."""
    cues = plan_broll(_talk(), clip_duration_s=40.0)
    assert cues, "expected some cues"
    assert all(c.start_s >= HOOK_PROTECT_S for c in cues)


def test_broll_lands_in_the_pauses_between_sentences():
    segs = _talk()
    cues = plan_broll(segs, clip_duration_s=40.0)
    ends = {round(s.end, 2) for s in segs}
    assert all(round(c.start_s, 2) in ends for c in cues), (
        "an insert started mid-sentence")


def test_no_pauses_means_no_broll():
    """Continuous speech has nowhere to cut away without burying it."""
    assert plan_broll(_talk(gap=0.05), clip_duration_s=40.0) == []


def test_coverage_is_capped():
    cues = plan_broll(_talk(n=30), clip_duration_s=30.0, max_cues=99)
    assert coverage(cues, 30.0) <= MAX_COVERAGE + 1e-6, (
        "b-roll that never returns to the speaker is a slideshow")


def test_inserts_are_long_enough_to_read_as_edits():
    cues = plan_broll(_talk(), clip_duration_s=40.0)
    assert all(c.duration_s >= MIN_INSERT_S for c in cues)


def test_each_cue_describes_what_was_just_said():
    cues = plan_broll(_talk(), clip_duration_s=40.0, style="cinematic")
    assert cues
    for c in cues:
        assert c.subject and c.prompt.endswith(".")
        assert "cinematic" in c.prompt


def test_the_overlay_keeps_the_speakers_audio():
    """Replacing the audio too would just be a different clip."""
    cues = plan_broll(_talk(), clip_duration_s=40.0)
    chain = overlay_filter(cues, width=1080, height=1920)
    assert "[vout]" in chain
    assert ":a]" not in chain and "amix" not in chain


def test_no_cues_means_no_filter():
    assert overlay_filter([], width=1080, height=1920) == ""


# ------------------------------------------ b-roll must never lose a clip

class _Router:
    """Stand-in generator. `fail` makes every shot raise; `blank` writes a
    file too small to be a video."""

    def __init__(self, fail=False, blank=False):
        self.fail, self.blank, self.calls = fail, blank, 0

    def generate_shot(self, *, prompt, seconds, fps, out_path, **kw):
        self.calls += 1
        if self.fail:
            raise RuntimeError("generation exploded")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if self.blank:
            Path(out_path).write_bytes(b"\x00" * 16)     # too small
        else:
            _video(Path(out_path), seconds=max(1.0, seconds))
        return None


def test_a_failed_generation_returns_the_original_clip(tmp_path):
    """B-roll is a garnish. A clip that already passed QA must survive a
    generation failure untouched."""
    from clipforge.broll import BRollCue, render_broll

    clip = _video(tmp_path / "clip.mp4", seconds=6.0)
    before = clip.read_bytes()
    cues = [BRollCue(3.0, 2.0, "harbour", "harbour boats.")]
    out = render_broll(clip, cues, router=_Router(fail=True),
                       width=128, height=224, work_dir=tmp_path / "w")
    assert out == clip
    assert clip.read_bytes() == before, "the original clip was modified"


def test_an_empty_generation_is_not_composited(tmp_path):
    """A blank insert over a good clip is strictly worse than no b-roll."""
    from clipforge.broll import BRollCue, render_broll

    clip = _video(tmp_path / "clip.mp4", seconds=6.0)
    out = render_broll(clip, [BRollCue(3.0, 2.0, "x", "x.")],
                       router=_Router(blank=True), width=128, height=224,
                       work_dir=tmp_path / "w")
    assert out == clip


def test_no_cues_skips_generation_entirely(tmp_path):
    from clipforge.broll import render_broll

    clip = _video(tmp_path / "clip.mp4")
    r = _Router()
    assert render_broll(clip, [], router=r, width=128, height=224,
                        work_dir=tmp_path / "w") == clip
    assert r.calls == 0, "generated b-roll with nothing to insert"


def test_a_successful_composite_keeps_the_audio_stream(tmp_path):
    """The speaker keeps talking under the b-roll — that is the point."""
    from clipforge.broll import BRollCue, render_broll

    src = tmp_path / "clip.mp4"
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi", "-i", "testsrc2=size=128x224:rate=10:d=6",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-shortest", str(src)], capture_output=True, timeout=180)
    assert proc.returncode == 0

    out = render_broll(src, [BRollCue(3.0, 1.5, "x", "x.")],
                       router=_Router(), width=128, height=224,
                       work_dir=tmp_path / "w")
    assert out != src, "composite did not run"
    assert out.is_file()
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)],
        capture_output=True, text=True, timeout=60)
    assert "audio" in (probe.stdout or ""), "b-roll dropped the audio"
