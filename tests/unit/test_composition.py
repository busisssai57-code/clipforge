"""The seams between features, which is where this round's defects were.

Every one of these was found by an audit tracing a real run, not by the
unit tests — because each feature's tests built their own world from
scratch and never met the others, or the eight-month-old workspace they
have to run against.
"""

from __future__ import annotations

import inspect
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from clipforge import cli
from clipforge.state import SCHEMA_VERSION, StateDB

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- the schema

def test_an_existing_database_gets_tables_added_later(tmp_path):
    """shipped_spans was added to the schema with no migration, and
    _migrate only runs the full schema for a BRAND NEW file. Every
    existing workspace simply had no such table: dedup caught the error,
    logged a warning nobody reads, and answered "not a duplicate" every
    single time. Dead in production, green in every test."""
    old = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(old)
    conn.executescript("""
        CREATE TABLE jobs (id INTEGER PRIMARY KEY);
        PRAGMA user_version=3;
    """)
    conn.commit()
    conn.close()

    db = StateDB(old)
    try:
        assert db.missing_tables() == set(), (
            "a database that predates a feature never gets its table")
        db.record_shipped_span("session:1", 10.0, 20.0, "c.mp4")
        assert len(db.shipped_spans("session:1")) == 1
    finally:
        db.close()


def test_a_version_stamp_is_not_proof_the_schema_is_complete(tmp_path):
    """The version says the migrations RAN. It does not say the tables
    exist — that is exactly the gap this shipped through."""
    db_path = tmp_path / "ws.sqlite3"
    StateDB(db_path).close()
    conn = sqlite3.connect(db_path)
    conn.execute("DROP TABLE shipped_spans")
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
    conn.commit()
    conn.close()

    db = StateDB(db_path)
    try:
        assert db.missing_tables() == set(), "the missing table was not restored"
    finally:
        db.close()


def test_the_live_workspace_has_every_table_the_code_expects():
    """The gate test that would have caught it the day it landed."""
    live = ROOT / "workspace" / "state.sqlite3"
    if not live.is_file():
        pytest.skip("no workspace database on this machine")
    db = StateDB(live)
    try:
        assert db.missing_tables() == set()
    finally:
        db.close()


# ------------------------------------------------- dedup and the candidates

def test_a_duplicate_falls_through_to_the_next_candidate():
    """With clips_per_window=1 the slice happened BEFORE the duplicate
    check: the top candidate was the moment the previous window already
    shipped, it was skipped, and the window produced nothing while the
    next-ranked candidate sat untouched."""
    src = inspect.getsource(cli.process)
    assert "ranked.items[:max(1, clips)]" not in src, (
        "the top-N slice is back in front of the dedup filter")
    assert "for pos, item in enumerate(ranked.items, start=1)" in src
    assert "if rendered >= wanted" in src, "nothing stops at the wanted count"
    assert "attempts_left" in src, (
        "a window of all-duplicates would render its way down the whole list")


# ------------------------------------------------- what reaches the phone

def test_watch_delivers_the_cut_the_hook_was_burned_into():
    """`bta brand --send` has always sent the branded cut; the watch path
    burned the hook into a file and then sent the other one. The hook is
    the one thing already-captioned footage does not have."""
    src = inspect.getsource(cli.process)
    assert "deliver = branded" in src, "the branded cut is not delivered"
    assert "notify.send_clip(\n                        deliver," in src
    assert "caption=notify.caption_for(Path(clip.clip_path))" in src, (
        "the branded cut has no export pack of its own, so its caption "
        "would fall back to a 64-character content hash")


def test_a_failed_overlay_still_delivers_the_original():
    src = inspect.getsource(cli.process)
    head = src[src.index("deliver = Path(clip.clip_path)"):]
    assert head.index("deliver = branded") > 0, "no degradation path"


def test_the_branded_cut_is_a_variant_not_a_second_clip(tmp_path):
    """Without this the gallery shows an untitled unscored tile per clip,
    and holdout reports clips=2 for one clip — the single number it
    exists to make comparable."""
    from clipforge.clipmeta import artifact_stem, variant_of

    assert artifact_stem(Path("abc.branded.mp4")) == "abc"
    assert variant_of(Path("abc.branded.mp4")) == "branded"
    assert variant_of(Path("abc.mp4")) is None


# ------------------------------------------------------------ preemption

def test_a_preempted_job_is_not_recorded_as_a_failure():
    src = inspect.getsource(cli.process)
    assert "except JobPreempted:" in src
    assert 'set_job_status(job_id, "preempted")' in src
    # and it must come BEFORE the catch-all, or it never runs
    assert src.index("except JobPreempted:") < src.index("except BaseException:")


def test_a_preempted_window_is_put_back_in_the_queue(tmp_path):
    """It used to be dropped: an evening of interrupted windows produced
    nothing, however long the machine was idle afterwards.

    Modelled as production behaves — when the operator is back, the gate
    is shut too, so the re-queued window waits there rather than
    hammering the pipeline."""
    from clipforge.dispatch import ClipDispatcher
    from clipforge.idle import JobPreempted

    media = tmp_path / "w.mp4"
    media.write_bytes(bytes(8))
    runs: list[str] = []
    operator_back = threading.Event()
    done = threading.Event()

    def handler(path, abs_start, key=None):
        runs.append("run")
        if len(runs) == 1:
            operator_back.set()          # they sit down mid-job
            raise JobPreempted("operator came back")
        done.set()

    disp = ClipDispatcher(
        handler=handler,
        gate=lambda: "operator active" if operator_back.is_set() else None,
        preempt=lambda: "operator active" if operator_back.is_set() else None,
        gate_poll_s=0.05).start()
    try:
        disp.submit(media, 0.0, ("segment", 1, 0))
        time.sleep(0.4)
        assert len(runs) == 1, "it retried while the operator was still there"
        operator_back.clear()            # they leave again
        assert done.wait(timeout=5), "the window was never retried"
    finally:
        operator_back.clear()
        disp.stop()
    assert len(runs) == 2
    assert disp.stats.processed == 1


def test_a_window_preempted_for_ever_is_eventually_given_up_on(tmp_path):
    from clipforge.dispatch import ClipDispatcher
    from clipforge.idle import JobPreempted

    media = tmp_path / "w.mp4"
    media.write_bytes(bytes(8))
    runs = []

    def handler(path, abs_start, key=None):
        runs.append(1)
        raise JobPreempted("still busy")

    disp = ClipDispatcher(handler=handler, gate=lambda: None,
                          preempt=lambda: None, gate_poll_s=0.01,
                          max_preempts=3).start()
    try:
        disp.submit(media, 0.0, ("segment", 1, 0))
        deadline = time.monotonic() + 5
        while len(runs) < 4 and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.2)
    finally:
        disp.stop()
    assert len(runs) == 4, f"expected 1 try + 3 retries, got {len(runs)}"
    assert disp.stats.dropped >= 1
