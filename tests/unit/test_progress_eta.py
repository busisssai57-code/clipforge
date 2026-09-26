"""Progress that is measured, and an ETA that refuses to guess.

The bar this replaces was `style="width:55%"` — a literal constant. A
wedged run and a healthy one rendered identically, which is worse than no
bar at all because it looks like information.

The rule these tests exist to hold: every number shown comes from the
run's own output or from this machine's own history, and when neither
exists the answer is None, not a plausible-looking figure.
"""

from __future__ import annotations

import sqlite3

import pytest

from clipforge.progress import (CLIP_STAGES, RunProgress, StageEstimator,
                                humanize)

# Lines taken verbatim from a real `bta process` run (2026-08-12), because
# a parser tested only against its own idea of the log format is a parser
# tested against nothing.
REAL_LINES = [
    "S1: transcribing chunk_00001.mp4 (window 0.0s)",
    "2026-08-12T18:29:01.100000Z [info     ] stage.done                     "
    "[clipforge.stages.base] cache_key=5efd elapsed_s=12.4 stage=s1_transcribe",
    "2026-08-12T18:29:04.243319Z [info     ] s2.candidates                  "
    "[clipforge.stages.s2_prefilter] generated=9 kept=1",
    "2026-08-12T18:29:04.245997Z [info     ] stage.done                     "
    "[clipforge.stages.base] cache_key=5efd elapsed_s=0.0 stage=s2_prefilter",
    "S3: ranking candidate windows (multimodal VL)",
]


def _fed(kind="process", lines=REAL_LINES, now=1000.0):
    p = RunProgress(kind=kind, started_at=0.0)
    for line in lines:
        p.feed(line, now=now)
    return p


# ------------------------------------------------------------- parsing

def test_finished_stages_are_read_from_the_log():
    assert _fed().done == ["s1_transcribe", "s2_prefilter"]


def test_the_running_stage_is_the_first_unfinished_one():
    assert _fed().current_stage() == "s3_semantic"


def test_the_label_is_the_cli_s_own_wording():
    """Better than a label mapped from a stage id: it carries the detail
    the operator wants ('multimodal VL', the filename)."""
    assert _fed().label == "ranking candidate windows (multimodal VL)"


def test_a_stage_is_never_counted_twice():
    p = _fed(lines=REAL_LINES + [REAL_LINES[1]])
    assert p.done.count("s1_transcribe") == 1


def test_a_cache_hit_counts_as_done():
    """A cached stage produced its artifact; refusing to count it would
    stall the bar at 0% on a re-run that finishes in seconds."""
    p = RunProgress(kind="process", started_at=0.0)
    p.feed("[info] stage.cache_hit [clipforge.stages.base] "
           "stage=s1_transcribe cache_key=abc")
    assert p.done == ["s1_transcribe"]


def test_an_unknown_stage_name_is_ignored():
    p = RunProgress(kind="process", started_at=0.0)
    p.feed("[info] stage.done stage=s99_imaginary")
    assert p.done == []


def test_a_download_percentage_is_surfaced():
    p = RunProgress(kind="grab", started_at=0.0)
    p.feed("[download]  42.7% of 1.20GiB at 5.00MiB/s ETA 00:42")
    assert p.download_pct == pytest.approx(42.7)
    # The label rounds for display; the parsed value keeps its precision.
    assert p.label == "Downloading 43%"


def test_generation_counts_shots():
    p = RunProgress(kind="generate", started_at=0.0)
    p.feed("rendering shot 3/6")
    assert (p.shot, p.shots_total) == (3, 6)


def test_candidate_ranking_progress_is_surfaced():
    p = RunProgress(kind="process", started_at=0.0)
    p.feed("  candidate 3/10: scored (25.0s-45.0s)")
    assert p.label == "Ranking candidate 3 of 10 (multimodal VL)"


def test_garbage_never_raises():
    """This runs on the task's output thread. A parser that throws takes
    the whole log with it."""
    p = RunProgress(kind="process", started_at=0.0)
    for junk in ["", "\x00\x01", "stage=", "S:", "shot /", "[download] %"]:
        p.feed(junk)
    assert p.done == []


# ------------------------------------------------------------ fraction

def test_fraction_is_zero_before_anything_finishes():
    assert RunProgress(kind="process", started_at=0.0).fraction() == 0.0


def test_fraction_grows_as_stages_complete():
    assert _fed().fraction() == pytest.approx(2 / len(CLIP_STAGES))


def test_fraction_is_weighted_by_measured_duration(tmp_path):
    """Unweighted, S1 finishing shows 12.5% whether it was 4 seconds or
    four minutes of the run. Weighted, the bar tracks time."""
    db = _db_with(tmp_path, {"s1_transcribe": [300.0], "s2_prefilter": [1.0],
                             "s3_semantic": [1.0], "editor": [1.0],
                             "s4_tracking": [1.0], "s5_subtitles": [1.0],
                             "s6_render": [1.0], "s7_qa": [1.0]})
    est = StageEstimator(db)
    p = RunProgress(kind="process", started_at=0.0, done=["s1_transcribe"])
    assert p.fraction(est) > 0.9   # S1 was almost the whole run


def test_generation_fraction_uses_shots():
    p = RunProgress(kind="generate", started_at=0.0)
    p.feed("shot 4/8")
    assert p.fraction() == pytest.approx(3 / 8)


def test_generation_without_a_shot_count_has_no_fraction():
    assert RunProgress(kind="generate", started_at=0.0).fraction() is None


# ----------------------------------------------------------------- eta

def _db_with(tmp_path, per_stage):
    path = tmp_path / "state.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE stage_runs (id INTEGER PRIMARY KEY, "
                 "job_id INT, stage TEXT, cache_key TEXT, status TEXT, "
                 "artifact TEXT, error TEXT, started_at REAL, finished_at REAL)")
    for stage, durations in per_stage.items():
        for d in durations:
            conn.execute("INSERT INTO stage_runs (job_id,stage,cache_key,"
                         "status,started_at,finished_at) VALUES (1,?,'k',"
                         "'done',0,?)", (stage, d))
    conn.commit()
    conn.close()
    return path


def test_no_history_means_no_eta(tmp_path):
    """The headline rule. A first run has nothing to estimate from and
    says so, rather than showing a countdown it invented."""
    est = StageEstimator(_db_with(tmp_path, {}))
    assert _fed().eta_s(est) is None


def test_eta_sums_the_remaining_stages(tmp_path):
    est = StageEstimator(_db_with(tmp_path,
                                  {name: [10.0] for name, _ in CLIP_STAGES}))
    p = RunProgress(kind="process", started_at=0.0,
                    done=[n for n, _ in CLIP_STAGES[:6]])
    # Two stages left at 10s each; the running one has not started ticking.
    assert p.eta_s(est, now=0.0) == pytest.approx(20.0)


def test_an_overrunning_stage_does_not_count_backwards(tmp_path):
    """Past its median the remainder floors at zero — an ETA that goes
    negative reads as a broken clock, not as 'any moment now'."""
    est = StageEstimator(_db_with(tmp_path, {"s7_qa": [10.0]}))
    p = RunProgress(kind="process", started_at=0.0,
                    done=[n for n, _ in CLIP_STAGES[:-1]])
    assert p.eta_s(est, now=10_000.0) == 0.0


def test_the_median_ignores_one_pathological_run(tmp_path):
    """A mean would let a single 40-minute render poison every later
    estimate. Nine 10s runs and one 4000s run is still 10s."""
    est = StageEstimator(_db_with(tmp_path,
                                  {"s6_render": [10.0] * 9 + [4000.0]}))
    assert est.median("s6_render") == pytest.approx(10.0)


def test_unfinished_and_failed_runs_are_not_measurements(tmp_path):
    path = _db_with(tmp_path, {"s6_render": [10.0]})
    conn = sqlite3.connect(path)
    conn.execute("INSERT INTO stage_runs (job_id,stage,cache_key,status,"
                 "started_at,finished_at) VALUES (1,'s6_render','k','running',"
                 "0,NULL)")
    conn.execute("INSERT INTO stage_runs (job_id,stage,cache_key,status,"
                 "started_at,finished_at) VALUES (1,'s6_render','k','failed',"
                 "0,9999)")
    conn.commit()
    conn.close()
    assert StageEstimator(path).median("s6_render") == pytest.approx(10.0)


def test_a_missing_database_is_not_a_crash(tmp_path):
    assert StageEstimator(tmp_path / "nope.sqlite3").all_medians() == {}


def test_generation_eta_extrapolates_from_shots_done():
    p = RunProgress(kind="generate", started_at=0.0)
    p.feed("shot 3/9", now=200.0)
    est = StageEstimator("unused")
    # Two shots took 200s => 100s each; seven remain.
    assert p.eta_s(est, now=200.0) == pytest.approx(700.0)


# ------------------------------------------------------------ reporting

def test_the_payload_states_whether_an_eta_exists(tmp_path):
    """`eta_s: null` alone is ambiguous — 'not known' and 'about to
    finish' must not render the same."""
    d = _fed().as_dict(StageEstimator(_db_with(tmp_path, {})))
    assert d["eta_known"] is False and d["eta_s"] is None


@pytest.mark.parametrize("seconds,expected", [
    (0, "0s"), (12.4, "12s"), (44, "44s"), (90, "2 min"),
    (3600, "1.0 h"), (None, ""),
])
def test_durations_read_like_a_person_wrote_them(seconds, expected):
    assert humanize(seconds) == expected
