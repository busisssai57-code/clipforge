"""Holdout measurement: the headline number, the source-set gate, save/diff.

All offline. ``list_clips`` is the only seam to the pipeline, so it is replaced
with fixed clip records and the arithmetic is pinned directly. Two properties
are load-bearing and each had a real bug the docstring in ``holdout.py`` calls
out: the headline reads the scorecard key that EXISTS (``overall``, not
``score`` - the first version reported a grade and a null number over two
graded clips), and two runs over different source sets are declared
incomparable rather than diffed into a meaningless delta.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from clipforge import holdout


def fake_meta(filename, *, source, overall=None, grade=None, duration_s=30.0,
              loudness_i=-14.0, qa_passed=None, qa_checks=0,
              framing_mode="speaker", rejected=False, dimensions=None):
    """One clip's sidecars, shaped exactly as ``_row`` reads them."""
    score = None
    if overall is not None or grade is not None or dimensions is not None:
        score = {"overall": overall, "grade": grade,
                 "dimensions": dimensions or []}
    qa = None
    if qa_passed is not None or qa_checks:
        qa = {"passed": qa_passed, "checks": [{} for _ in range(qa_checks)]}
    return SimpleNamespace(
        filename=filename, source_path=source, duration_s=duration_s,
        score=score, qa=qa, loudness_i=loudness_i, framing_mode=framing_mode,
        rejected=rejected)


def patch_clips(monkeypatch, metas):
    monkeypatch.setattr(holdout, "list_clips", lambda ws: list(metas))


def test_mean_score_reads_overall_not_score(monkeypatch):
    # The regression this whole module exists to stop: the scorecard's key is
    # "overall". Reading "score" returned a grade and a null mean over two
    # graded clips. A scorecard here carries NO "score" key on purpose, so a
    # mutant that reads the wrong key gets None and fails this.
    patch_clips(monkeypatch, [
        fake_meta("a.mp4", source="s1.mp4", overall=80.0, grade="B"),
        fake_meta("b.mp4", source="s1.mp4", overall=90.0, grade="A"),
    ])
    report = holdout.measure(SimpleNamespace())
    assert report["clips"] == 2
    assert report["mean_score"] == 85.0
    assert report["median_score"] == 85.0
    assert report["grades"] == {"A": 1, "B": 1}


def test_sources_filter_restricts_to_named_sources(monkeypatch):
    patch_clips(monkeypatch, [
        fake_meta("a.mp4", source="keep.mp4", overall=70.0),
        fake_meta("b.mp4", source="drop.mp4", overall=10.0),
    ])
    # Filenames, not full paths, are what a clip records and what the holdout
    # set is defined by.
    report = holdout.measure(SimpleNamespace(), sources=["keep.mp4"])
    assert report["clips"] == 1
    assert report["mean_score"] == 70.0
    assert report["sources"] == ["keep.mp4"]
    assert report["requested_sources"] == ["keep.mp4"]


def test_rejected_clips_are_counted_but_excluded_from_metrics(monkeypatch):
    patch_clips(monkeypatch, [
        fake_meta("a.mp4", source="s.mp4", overall=88.0),
        fake_meta("bad.mp4", source="s.mp4", overall=2.0, rejected=True),
    ])
    report = holdout.measure(SimpleNamespace())
    assert report["clips"] == 1          # the accepted one
    assert report["rejected"] == 1
    assert report["mean_score"] == 88.0  # the reject does not drag it down


def test_loudness_in_band_uses_target_and_tolerance(monkeypatch):
    # Target is -14.0 +/- 1.0. -14.5 is in band; -16.0 is not.
    patch_clips(monkeypatch, [
        fake_meta("a.mp4", source="s.mp4", overall=50.0, loudness_i=-14.5),
        fake_meta("b.mp4", source="s.mp4", overall=50.0, loudness_i=-16.0),
    ])
    report = holdout.measure(SimpleNamespace())
    assert report["loudness_in_band"] == 0.5


def test_qa_pass_rate_ignores_unjudged_clips(monkeypatch):
    # Only clips S7 actually judged belong in the denominator; an unrun QA is
    # not a failure.
    patch_clips(monkeypatch, [
        fake_meta("a.mp4", source="s.mp4", overall=50.0, qa_passed=True, qa_checks=23),
        fake_meta("b.mp4", source="s.mp4", overall=50.0, qa_passed=False, qa_checks=23),
        fake_meta("c.mp4", source="s.mp4", overall=50.0, qa_passed=None),
    ])
    report = holdout.measure(SimpleNamespace())
    assert report["qa_pass_rate"] == 0.5   # 1 of 2 judged, the None ignored


def test_empty_workspace_reports_nulls_not_crash(monkeypatch):
    patch_clips(monkeypatch, [])
    report = holdout.measure(SimpleNamespace())
    assert report["clips"] == 0
    assert report["mean_score"] is None
    assert report["qa_pass_rate"] is None
    assert report["loudness_in_band"] is None


def test_compare_flags_different_source_sets_as_incomparable():
    now = {"sources": ["a.mp4", "b.mp4"], "mean_score": 80.0}
    then = {"sources": ["a.mp4"], "mean_score": 60.0}
    delta = holdout.compare(now, then)
    assert delta["comparable"] is False
    # The mean moved 20 points, but only because the inputs changed - the exact
    # number this refuses to report.
    assert "mean_score" not in delta
    assert "source set" in delta["reason"]


def test_compare_computes_deltas_for_same_sources():
    now = {"sources": ["a.mp4"], "mean_score": 82.0, "qa_pass_rate": 1.0,
           "clips": 2}
    then = {"sources": ["a.mp4"], "mean_score": 80.0, "qa_pass_rate": 0.5,
            "clips": 2}
    delta = holdout.compare(now, then)
    assert delta["comparable"] is True
    assert delta["mean_score"] == 2.0
    assert delta["qa_pass_rate"] == 0.5


def test_compare_with_no_previous_is_not_comparable():
    delta = holdout.compare({"sources": ["a.mp4"]}, None)
    assert delta["comparable"] is False
    assert "no earlier" in delta["reason"]


def test_save_then_previous_roundtrips_the_report(tmp_path):
    ws = SimpleNamespace(root=str(tmp_path))
    report = {"generated_at": 1_700_000_000.0, "sources": ["s.mp4"],
              "clips": 3, "mean_score": 77.0}
    path = holdout.save(ws, report)
    assert path.exists()
    got = holdout.previous(ws)
    assert got == report


def test_previous_can_exclude_the_current_file(tmp_path):
    # A run saves itself, then asks for the one before it: passing its own path
    # as `before` must not hand it back its own numbers.
    ws = SimpleNamespace(root=str(tmp_path))
    older = holdout.save(ws, {"generated_at": 1_700_000_000.0,
                              "sources": ["s.mp4"], "mean_score": 70.0})
    newer = holdout.save(ws, {"generated_at": 1_700_000_100.0,
                              "sources": ["s.mp4"], "mean_score": 90.0})
    prior = holdout.previous(ws, before=newer)
    assert prior["mean_score"] == 70.0
    assert older.name in {p.name for p in tmp_path.glob("holdout/*.json")}


def test_previous_is_none_when_no_holdout_dir(tmp_path):
    assert holdout.previous(SimpleNamespace(root=str(tmp_path))) is None
