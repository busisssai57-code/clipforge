"""clipmeta.py — the join that ended the dead-markup era.

This module exists because the dashboard twice rendered fields no endpoint
sent. The tests therefore check both directions of honesty: a full
artifact chain populates every field from the artifact that measured it,
and a broken chain yields ABSENT fields plus a provenance map that says
why — never a plausible zero.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from clipforge import clipmeta
from clipforge.paths import Workspace


# ------------------------------------------------------------------ helpers

def _write(path: Path, blob: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")


STEM = "job42_s6cache"


def make_chain(ws: Workspace, *, stem: str = STEM,
               clip_start: float = 110.0, clip_end: float = 140.0,
               cand_start: float = 110.2, cand_end: float = 139.8,
               offset: float = 100.0) -> Path:
    """A complete, mutually consistent S1→S6 chain plus QA and editor."""
    arts = Path(ws.artifacts)
    _write(arts / "s6_render" / f"{stem}.json", {
        "source_subtitles": "s5key", "duration_s": 30.0,
        "width": 1080, "height": 1920, "encoder": "libx264",
        "loudness_i": -14.1, "loudness_tp": -1.6,
    })
    _write(arts / "s5_subtitles" / "s5key.json", {
        "source_campath": "s4key",
        "clip_start": clip_start, "clip_end": clip_end,
    })
    _write(arts / "s4_tracking" / "s4key.json", {
        "source_ranking": "s3key", "framing_mode": "speaker",
        "src_width": 1920, "src_height": 1080,
        "src_fps_rational": "30/1",
        "frames": [{"frame": 0, "x": 420, "y": 0, "w": 607, "h": 1080}],
        "assignments": [],
    })
    _write(arts / "s3_semantic" / "s3key.json", {
        "source_candidates": "s2key", "ranking_source": "vl",
        "items": [{"candidate_index": 0, "title": "The reveal",
                   "hook": "wait for it", "justification": "high action",
                   "visual_action": 8.0, "hook_strength": 7.0,
                   "comprehensibility": 9.0}],
    })
    _write(arts / "s2_prefilter" / "s2key.json", {
        "source_transcript": "s1key",
        "candidates": [{"start": cand_start, "end": cand_end,
                        "scores": {"boundary": 0.9, "qa_structure": 0.5,
                                   "turn_density": 0.7}}],
    })
    _write(arts / "s1_transcribe" / "s1key.json", {
        "source_path": str(Path(ws.root) / "chunks" / "src.mp4"),
        "abs_offset_s": offset, "language": "en", "diarization_ok": True,
        "segments": [{
            "start": 8.0, "end": 35.0, "text": "hello there world",
            "speaker": "SPEAKER_00",
            "words": [
                {"text": "hello", "start": 10.5, "end": 11.0,
                 "score": 0.95, "speaker": "SPEAKER_00"},
                {"text": "there", "start": 12.0, "end": 12.4,
                 "score": 0.4, "speaker": "SPEAKER_00"},
                {"text": "world", "start": 12.5, "end": 12.9,
                 "score": None, "speaker": "SPEAKER_00"},
            ],
        }],
    })
    _write(arts / "s7_qa" / "qacache.json", {
        "source_clip": stem, "passed": True, "failed_count": 0,
        "warned_count": 1, "checks": [{"name": "loudness", "ok": True}],
    })
    _write(arts / "editor" / "edcache.json", {
        "candidate_id": "cand_000", "title": "Editor title",
        "hook_text": "editor hook",
    })

    clip = Path(ws.clips) / f"{stem}.mp4"
    clip.write_bytes(b"\x00" * 2048)
    return clip


@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws").ensure()


# --------------------------------------------------- stem / variant naming

def test_artifact_stem_plain_clip_is_its_own_stem():
    assert clipmeta.artifact_stem("abc123.mp4") == "abc123"


@pytest.mark.parametrize("filename,variant,kind", [
    ("abc.draft.mp4", "draft", "render"),
    ("abc.upscaled.mp4", "upscaled", "render"),
    ("abc.es.mp4", "es", "dub"),
    ("abc.ja.mp4", "ja", "dub"),
    ("abc.mp4", None, None),
])
def test_sidecars_resolve_to_parent_stem(filename, variant, kind):
    assert clipmeta.artifact_stem(filename) == "abc"
    assert clipmeta.variant_of(filename) == variant
    assert clipmeta.variant_kind(clipmeta.variant_of(filename)) == kind


# ------------------------------------------------------------- resolve_clip

def test_chained_sidecars_resolve_to_the_original_key(ws):
    """Sidecars compose, so stripping must not stop after one layer.

    A voiceover over an upscale is `<key>.upscaled.vo.mp4`. Stripping once
    yields `<key>.upscaled`, which is not a cache key and has no artifact
    behind it — the clip would lose its score, transcript and QA and read
    as an unscored orphan. Found when the dashboard's voiceover button ran
    against the newest clip and the newest clip was itself a sidecar.
    """
    assert clipmeta.artifact_stem("abc123.upscaled.vo.mp4") == "abc123"
    assert clipmeta.artifact_stem("abc123.vo.upscaled.mp4") == "abc123"
    assert clipmeta.artifact_stem("abc123.upscaled.es.mp4") == "abc123"
    assert clipmeta.artifact_stem("abc123.draft.upscaled.vo.mp4") == "abc123"
    # the outermost operation is still what names the variant
    assert clipmeta.variant_of("abc123.upscaled.vo.mp4") == "vo"
    # and a plain clip is untouched by the loop
    assert clipmeta.artifact_stem("abc123.mp4") == "abc123"


def test_full_chain_populates_every_field_from_its_artifact(ws):
    make_chain(ws)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta is not None

    # S6 — render facts
    assert meta.duration_s == 30.0
    assert (meta.width, meta.height) == (1080, 1920)
    assert meta.encoder == "libx264"
    assert meta.loudness_i == -14.1
    # S5 — window
    assert (meta.clip_start, meta.clip_end) == (110.0, 140.0)
    # S4 — framing
    assert meta.framing_mode == "speaker"
    assert (meta.source_width, meta.source_height) == (1920, 1080)
    # S3 — judgement text (export pack absent, so ranked item supplies it)
    assert meta.justification == "high action"
    # S1 — source pointer, existence honestly probed
    assert meta.source_path and meta.source_path.endswith("src.mp4")
    assert meta.source_exists is False  # the file was never created
    # QA
    assert meta.qa == {"passed": True, "failed_count": 0, "warned_count": 1,
                       "checks": [{"name": "loudness", "ok": True}]}
    # scorecard is built and marked as VL-ranked
    assert meta.score is not None
    # every hop of the chain is accounted for
    assert meta.provenance == {
        "render": True, "subtitles": True, "tracking": True, "ranking": True,
        "candidates": True, "transcript": True, "qa": True, "editor": True,
        "candidate_matched": True,
    }


def test_editor_pack_fills_title_and_hook(ws):
    make_chain(ws)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    # editor is preferred over ranked-item for title; hook comes from editor
    assert meta.title == "Editor title"
    assert meta.hook == "editor hook"


def test_export_pack_outranks_editor_and_ranking(ws):
    clip = make_chain(ws)
    _write(clip.with_suffix(".export.json"), {
        "title": "Shipped title", "caption": "the caption",
        "hashtags": ["#a", "#b"], "platforms": {"tiktok": {}},
        "chapters": [{"t": 0, "label": "start"}],
    })
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.has_export
    assert meta.title == "Shipped title"  # what the CLI actually shipped
    assert meta.caption == "the caption"
    assert meta.platforms == {"tiktok": {}}
    assert meta.hashtags == ["#a", "#b"]
    assert meta.chapters == [{"t": 0, "label": "start"}]


def test_missing_clip_file_is_none_not_a_ghost_record(ws):
    make_chain(ws)  # artifacts exist, file does not
    assert clipmeta.resolve_clip(ws, "other.mp4") is None


def test_broken_chain_reports_absence_not_zeros(ws):
    """Sever the chain at S5: downstream fields must be absent."""
    make_chain(ws)
    (Path(ws.artifacts) / "s5_subtitles" / "s5key.json").unlink()
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")

    assert meta.duration_s == 30.0  # S6 still stands
    assert meta.clip_start is None  # not 0.0
    assert meta.framing_mode is None
    assert meta.score is None  # no candidate match possible
    assert meta.provenance["render"] is True
    assert meta.provenance["subtitles"] is False
    assert meta.provenance["tracking"] is False  # S4 unreachable without S5
    assert meta.provenance["candidate_matched"] is False


def test_corrupt_artifact_is_treated_as_absent(ws):
    make_chain(ws)
    bad = Path(ws.artifacts) / "s6_render" / f"{STEM}.json"
    bad.write_text("{not json", encoding="utf-8")
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta is not None
    assert meta.duration_s is None
    assert meta.provenance["render"] is False


def test_candidate_match_requires_both_edges(ws):
    """Same start, twice the length — must NOT be claimed."""
    make_chain(ws, cand_start=110.0, cand_end=170.0)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.provenance["candidate_matched"] is False
    assert meta.score is None
    assert meta.provenance["editor"] is False  # editor needs the exact index


def test_candidate_match_tolerates_snapping_drift(ws):
    make_chain(ws, cand_start=110.4, cand_end=139.7)  # 0.7s total delta
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.provenance["candidate_matched"] is True


def test_candidate_tolerance_boundary_just_inside(ws):
    # summed two-edge delta 1.9s — inside the 2 × 1.0s window tolerance
    make_chain(ws, cand_start=110.95, cand_end=139.05)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.provenance["candidate_matched"] is True


def test_candidate_tolerance_boundary_just_outside(ws):
    # summed two-edge delta 2.1s — past the tolerance, must not be claimed
    make_chain(ws, cand_start=111.05, cand_end=138.95)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.provenance["candidate_matched"] is False


def test_thumb_url_only_when_a_thumbnail_really_exists(ws):
    clip = make_chain(ws)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.has_thumb is False
    assert meta.as_dict()["thumb_url"] is None  # no dead image link

    clip.with_suffix(".thumb.jpg").write_bytes(b"\xff")
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.has_thumb is True
    assert meta.as_dict()["thumb_url"] == f"/api/clips/thumb/{STEM}.mp4"


def test_scorecard_source_reflects_vl_ranking(ws):
    make_chain(ws)  # fixture S3 has ranking_source == "vl"
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.score["source"] == "vl"


def test_scorecard_source_heuristic_when_vl_did_not_run(ws):
    make_chain(ws)
    s3 = Path(ws.artifacts) / "s3_semantic" / "s3key.json"
    blob = json.loads(s3.read_text(encoding="utf-8"))
    blob["ranking_source"] = "heuristic"
    # no VL judgements either — a heuristic ranking has none to offer
    blob["items"] = [{"candidate_index": 0, "title": "The reveal",
                      "hook": "wait for it", "justification": "high action"}]
    _write(s3, blob)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.score["source"] == "heuristic"


def test_size_mb_is_the_real_file_size(ws):
    clip = make_chain(ws)
    clip.write_bytes(b"\x00" * (3 * 1024 * 1024 + 512 * 1024))  # 3.5 MB
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.as_dict()["size_mb"] == 3.5


def test_sidecar_inherits_parent_chain(ws):
    make_chain(ws)
    dub = Path(ws.clips) / f"{STEM}.es.mp4"
    dub.write_bytes(b"\x00" * 1024)
    meta = clipmeta.resolve_clip(ws, f"{STEM}.es.mp4")
    assert meta.variant == "es"
    assert meta.stem == STEM
    assert meta.duration_s == 30.0  # parent's chain, not an unscored orphan


def test_subtitle_tracks_found_beside_the_clip(ws):
    clip = make_chain(ws)
    (clip.parent / f"{STEM}.es.srt").write_text("1\n", encoding="utf-8")
    (clip.parent / f"{STEM}.fr.srt").write_text("1\n", encoding="utf-8")
    meta = clipmeta.resolve_clip(ws, f"{STEM}.mp4")
    assert meta.subtitle_tracks == ["es", "fr"]


def test_as_dict_marks_rejected_urls(ws):
    make_chain(ws)
    rej = Path(ws.clips) / "rejected"
    rej.mkdir(exist_ok=True)
    (rej / "bad.mp4").write_bytes(b"\x00")
    meta = clipmeta.resolve_clip(ws, "bad.mp4", rejected=True)
    blob = meta.as_dict()
    assert blob["rejected"] is True
    assert blob["url"].endswith("?rejected=1")
    json.dumps(blob)  # must be shippable as-is


# --------------------------------------------------------------- list_clips

def test_list_clips_newest_first_with_quarantine_flagged(ws):
    make_chain(ws)
    older = Path(ws.clips) / "older.mp4"
    older.write_bytes(b"\x00")
    rej_dir = Path(ws.clips) / "rejected"
    rej_dir.mkdir(exist_ok=True)
    rejected = rej_dir / "quarantined.mp4"
    rejected.write_bytes(b"\x00")

    os.utime(older, (1_000_000, 1_000_000))
    os.utime(Path(ws.clips) / f"{STEM}.mp4", (3_000_000, 3_000_000))
    os.utime(rejected, (2_000_000, 2_000_000))

    out = clipmeta.list_clips(ws)
    assert [m.filename for m in out] == [f"{STEM}.mp4", "quarantined.mp4",
                                         "older.mp4"]
    assert [m.rejected for m in out] == [False, True, False]


# ------------------------------------------------------------ transcript_for

def test_transcript_rebased_to_clip_time(ws):
    """S1 chunk time + abs_offset must land on the S5 window's timeline."""
    make_chain(ws)  # offset=100, window 110..140
    out = clipmeta.transcript_for(ws, f"{STEM}.mp4")
    assert out["available"] is True
    assert out["clip_start"] == 110.0
    assert out["duration_s"] == 30.0
    assert out["language"] == "en"
    assert out["diarized"] is True

    words = out["words"]
    assert [w["text"] for w in words] == ["hello", "there", "world"]
    # 10.5 chunk-relative + 100 offset - 110 window start = 0.5
    assert words[0]["start"] == 0.5
    assert words[0]["end"] == 1.0
    # confidence passes through only when the aligner really reported it
    assert words[0]["score"] == 0.95
    assert words[2]["score"] is None


def test_transcript_gap_pills(ws):
    make_chain(ws)
    words = clipmeta.transcript_for(ws, f"{STEM}.mp4")["words"]
    assert words[0]["gap_before"] is None  # nothing precedes the first word
    assert words[1]["gap_before"] == 1.0   # 12.0 - 11.0
    assert words[2]["gap_before"] is None  # 0.1s is not a pause


def test_transcript_pause_pill_threshold(ws):
    """0.2s is the pill threshold: a 0.25s gap shows, a 0.15s gap does not."""
    make_chain(ws)
    s1 = Path(ws.artifacts) / "s1_transcribe" / "s1key.json"
    blob = json.loads(s1.read_text(encoding="utf-8"))
    blob["segments"][0]["words"] = [
        {"text": "a", "start": 10.5, "end": 11.0},
        {"text": "b", "start": 11.25, "end": 11.5},   # 0.25s after "a"
        {"text": "c", "start": 11.65, "end": 11.9},   # 0.15s after "b"
    ]
    _write(s1, blob)
    words = clipmeta.transcript_for(ws, f"{STEM}.mp4")["words"]
    assert words[1]["gap_before"] == 0.25
    assert words[2]["gap_before"] is None


def test_transcript_segments_carry_rebased_times_and_nested_words(ws):
    make_chain(ws)  # offset=100, window 110..140; segment 8.0..35.0 chunk time
    (seg,) = clipmeta.transcript_for(ws, f"{STEM}.mp4")["segments"]
    # 8.0 + 100 offset - 110 window start = -2.0 (overlaps the window edge)
    assert seg["start"] == -2.0
    assert seg["end"] == 25.0
    assert seg["text"] == "hello there world"
    assert seg["speaker"] == "SPEAKER_00"
    assert [w["text"] for w in seg["words"]] == ["hello", "there", "world"]
    assert seg["words"][0]["start"] == 0.5  # same rebase as the flat list


def test_transcript_words_outside_the_window_are_cut(ws):
    make_chain(ws, clip_start=112.3, clip_end=140.0)
    words = clipmeta.transcript_for(ws, f"{STEM}.mp4")["words"]
    # "hello" (110.5–111.0 abs) ends before the window now
    assert [w["text"] for w in words] == ["there", "world"]


def test_transcript_unavailable_names_the_missing_stage(ws):
    make_chain(ws)
    (Path(ws.artifacts) / "s3_semantic" / "s3key.json").unlink()
    out = clipmeta.transcript_for(ws, f"{STEM}.mp4")
    assert out == {"available": False,
                   "reason": "no ranking artifact for this clip"}


def test_transcript_for_missing_clip(ws):
    assert clipmeta.transcript_for(ws, "nope.mp4") == {
        "available": False, "reason": "clip not found"}


def test_transcript_never_fakes_silence(ws):
    """No S1 → unavailable with a reason; an empty word list would claim
    the clip is silent, which is a different and wrong statement."""
    make_chain(ws)
    (Path(ws.artifacts) / "s1_transcribe" / "s1key.json").unlink()
    out = clipmeta.transcript_for(ws, f"{STEM}.mp4")
    assert out["available"] is False
    assert "words" not in out


# -------------------------------------------------------------- campath_for

def test_campath_normalised_and_timed(ws):
    make_chain(ws)
    out = clipmeta.campath_for(ws, f"{STEM}.mp4")
    assert out["available"] is True
    assert out["timed"] is True
    assert out["fps"] == 30.0
    fr = out["frames"][0]
    assert fr["t"] == 0.0
    assert fr["x"] == round(420 / 1920, 5)
    assert fr["w"] == round(607 / 1920, 5)
    assert fr["h"] == round(1080 / 1080, 5)


def test_campath_without_fps_is_unplaced_not_assumed_30(ws):
    make_chain(ws)
    s4 = Path(ws.artifacts) / "s4_tracking" / "s4key.json"
    blob = json.loads(s4.read_text(encoding="utf-8"))
    del blob["src_fps_rational"]
    _write(s4, blob)
    out = clipmeta.campath_for(ws, f"{STEM}.mp4")
    assert out["available"] is True
    assert out["timed"] is False
    assert out["fps"] is None
    assert "t" not in out["frames"][0]  # unplaced, exactly as documented


def test_campath_passes_shot_analysis_and_assignments_through(ws):
    make_chain(ws)
    s4 = Path(ws.artifacts) / "s4_tracking" / "s4key.json"
    blob = json.loads(s4.read_text(encoding="utf-8"))
    blob["mar_shots"] = 12
    blob["presence_shots"] = 18
    blob["assignments"] = [{"shot": 0, "track": 2, "confidence": 0.9}]
    _write(s4, blob)
    out = clipmeta.campath_for(ws, f"{STEM}.mp4")
    assert out["mar_shots"] == 12
    assert out["presence_shots"] == 18
    assert out["assignments"] == [{"shot": 0, "track": 2, "confidence": 0.9}]


def test_campath_downsamples_long_paths(ws):
    make_chain(ws)
    s4 = Path(ws.artifacts) / "s4_tracking" / "s4key.json"
    blob = json.loads(s4.read_text(encoding="utf-8"))
    blob["frames"] = [{"frame": i, "x": 0, "y": 0, "w": 100, "h": 100}
                      for i in range(1400)]
    _write(s4, blob)
    out = clipmeta.campath_for(ws, f"{STEM}.mp4")
    assert out["frame_count"] == 1400
    assert out["sampled_every"] == 2
    assert len(out["frames"]) == 700


def test_campath_no_tracking_artifact(ws):
    clip = Path(ws.clips) / "lone.mp4"
    clip.write_bytes(b"\x00")
    assert clipmeta.campath_for(ws, "lone.mp4") == {
        "available": False, "reason": "no tracking artifact"}


@pytest.mark.parametrize("raw,expected", [
    ("24000/1001", 24000 / 1001),
    ("30/1", 30.0),
    (30, 30.0),
    (29.97, 29.97),
    ("30/0", None),   # never divide-by-zero into infinity
    ("thirty", None),
    (None, None),
    (0, None),        # zero fps places nothing
])
def test_fps_from_rational(raw, expected):
    got = clipmeta._fps_from(raw)
    if expected is None:
        assert got is None
    else:
        assert got == pytest.approx(expected)
