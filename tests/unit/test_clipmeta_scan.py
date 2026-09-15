"""The gallery listing must stay linear in the number of clips.

`/api/clips` is polled every four seconds by the dashboard, and it was
quadratic in two independent places:

* S7's verdict and the editor pack are keyed by their OWN cache key and
  merely name the clip they describe, so finding one meant reading every
  JSON in the directory — per clip;
* the dub sidecar lookup globbed `clips/<stem>.*.srt`, and a glob lists
  the whole folder — so N clips cost N listings of N entries.

Both are now indexed once per listing. These tests pin the counts, not
the clock: a timing assertion on a busy CI box measures the box.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge import clipmeta
from clipforge.paths import Workspace


def _write(path: Path, blob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(blob), encoding="utf-8")


@pytest.fixture()
def ws(tmp_path):
    """One source video cut into several clips — the shape that hurts.

    Clips from one source share the S3/S2/S1 chain, so a cache that works
    is visible as reads NOT repeated.
    """
    workspace = Workspace(tmp_path / "workspace").ensure()
    arts = Path(workspace.artifacts)
    _write(arts / "s1_transcribe" / "tr1.json",
           {"source_path": "D:/media/long.mp4", "words": []})
    _write(arts / "s2_prefilter" / "cd1.json",
           {"source_transcript": "tr1",
            "candidates": [{"start": i * 60.0, "end": i * 60.0 + 30.0}
                           for i in range(6)]})
    _write(arts / "s3_semantic" / "rk1.json",
           {"source_candidates": "cd1", "items": []})
    for i in range(6):
        _write(arts / "s4_tracking" / f"cm{i}.json", {"source_ranking": "rk1"})
        _write(arts / "s5_subtitles" / f"sb{i}.json",
               {"source_campath": f"cm{i}", "clip_start": i * 60.0,
                "clip_end": i * 60.0 + 30.0})
        _write(arts / "s6_render" / f"rn{i}.json",
               {"source_subtitles": f"sb{i}", "duration_s": 30.0})
        _write(arts / "s7_qa" / f"qa{i}.json",
               {"source_clip": f"rn{i}", "passed": True, "checks": []})
        _write(arts / "editor" / f"ed{i}.json",
               {"candidate_id": f"cand_{i:03d}", "title": f"title {i}"})
        (Path(workspace.clips) / f"rn{i}.mp4").write_bytes(b"\0" * 64)
    return workspace


def test_every_clip_still_resolves_its_own_qa_and_editor_pack(ws):
    """The indices must not smear one clip's artifacts onto another."""
    clips = {m.filename: m for m in clipmeta.list_clips(ws)}
    assert len(clips) == 6
    for i in range(6):
        meta = clips[f"rn{i}.mp4"]
        assert meta.qa is not None, f"rn{i} lost its QA verdict"
        assert meta.qa["passed"] is True
        assert meta.title == f"title {i}", "editor packs got crossed"
        assert meta.source_path == "D:/media/long.mp4"


def test_the_listing_reads_each_artifact_once(ws, monkeypatch):
    """Six clips off one source share S3/S2/S1; each must be parsed once.

    Before the scan this read 6 x (whole s7_qa dir + whole editor dir +
    the shared chain) — the numbers that made a four-second poll cost
    most of a core on a real workspace.
    """
    reads: list[str] = []
    real = clipmeta._read
    monkeypatch.setattr(clipmeta, "_read",
                        lambda p: (reads.append(str(p)), real(p))[1])

    clipmeta.list_clips(ws)

    assert len(reads) == len(set(reads)), "an artifact was parsed twice"
    shared = [r for r in reads if "s1_transcribe" in r]
    assert len(shared) == 1, f"the transcript was read {len(shared)} times"


def test_reading_scales_linearly_not_quadratically(ws, monkeypatch):
    """Adding clips must not multiply the reads each existing clip costs."""
    def count() -> int:
        reads: list[str] = []
        real = clipmeta._read
        monkeypatch.setattr(clipmeta, "_read",
                            lambda p: (reads.append(str(p)), real(p))[1])
        clipmeta.list_clips(ws)
        monkeypatch.undo()
        return len(reads)

    before = count()
    arts = Path(ws.artifacts)
    for i in range(6, 12):
        _write(arts / "s4_tracking" / f"cm{i}.json", {"source_ranking": "rk1"})
        _write(arts / "s5_subtitles" / f"sb{i}.json",
               {"source_campath": f"cm{i}"})
        _write(arts / "s6_render" / f"rn{i}.json",
               {"source_subtitles": f"sb{i}"})
        _write(arts / "s7_qa" / f"qa{i}.json",
               {"source_clip": f"rn{i}", "passed": True, "checks": []})
        (Path(ws.clips) / f"rn{i}.mp4").write_bytes(b"\0" * 64)
    after = count()

    # Doubling the clips roughly doubles the reads. Quadratic growth would
    # be ~4x; the guard is deliberately loose so it fails on the shape of
    # the curve, not on an exact count.
    assert after < before * 2.5, (
        f"{before} reads for 6 clips, {after} for 12 — that is not linear")


def test_a_single_clip_lookup_needs_no_scan_from_the_caller(ws):
    """resolve_clip is public and called on its own by several endpoints."""
    meta = clipmeta.resolve_clip(ws, "rn3.mp4")
    assert meta is not None
    assert meta.title == "title 3"


def test_dub_sidecars_are_still_found(ws):
    clip = Path(ws.clips) / "rn2.mp4"
    (clip.parent / "rn2.es.srt").write_text("1\n", encoding="utf-8")
    (clip.parent / "rn2.fr.srt").write_text("1\n", encoding="utf-8")
    # A different clip's track must not leak into this one's list.
    (clip.parent / "rn3.de.srt").write_text("1\n", encoding="utf-8")

    by_name = {m.filename: m for m in clipmeta.list_clips(ws)}
    assert by_name["rn2.mp4"].subtitle_tracks == ["es", "fr"]
    assert by_name["rn3.mp4"].subtitle_tracks == ["de"]
    assert by_name["rn1.mp4"].subtitle_tracks == []


def test_a_clip_stem_containing_dots_keeps_its_tracks(ws):
    """Sidecars compose, so the stem is not what `.stem` returns."""
    arts = Path(ws.artifacts)
    _write(arts / "s6_render" / "rn1.broll.json", {"duration_s": 30.0})
    clip = Path(ws.clips) / "rn1.broll.mp4"
    clip.write_bytes(b"\0" * 64)
    (clip.parent / "rn1.broll.pt.srt").write_text("1\n", encoding="utf-8")

    by_name = {m.filename: m for m in clipmeta.list_clips(ws)}
    assert by_name["rn1.broll.mp4"].subtitle_tracks == ["pt"]


def test_the_newest_verdict_wins_when_two_name_the_same_clip(ws):
    """A re-run writes a second QA artifact naming the same clip. Which
    one won used to be filesystem order — arbitrary, and able to change
    between two polls with nothing on disk having moved."""
    import os
    import time

    arts = Path(ws.artifacts)
    old = arts / "s7_qa" / "qa0.json"
    new = arts / "s7_qa" / "qa0_rerun.json"
    _write(new, {"source_clip": "rn0", "passed": False, "failed_count": 2,
                 "checks": []})
    now = time.time()
    os.utime(old, (now - 600, now - 600))
    os.utime(new, (now, now))

    meta = {m.filename: m for m in clipmeta.list_clips(ws)}["rn0.mp4"]
    assert meta.qa["passed"] is False
