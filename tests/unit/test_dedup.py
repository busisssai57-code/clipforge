"""One moment, one clip, across overlapping windows.

`bta watch` gives each window the trailing 60 s of its predecessor so a
highlight on a chunk boundary survives whole (T1). The same highlight is
then a candidate in both windows, each window ranks on its own, and the
two renders get different names because their source windows differ — so
nothing downstream noticed, and the phone got the same moment twice.
"""

from __future__ import annotations

import pytest

from clipforge import dedup
from clipforge.paths import Workspace
from clipforge.state import StateDB


@pytest.fixture()
def db(tmp_path):
    ws = Workspace(tmp_path / "ws").ensure()
    database = StateDB(ws.state_db)
    yield database
    database.close()


@pytest.fixture()
def scope(db):
    token = dedup.set_scope(dedup.Scope(key="session:7", db=db))
    yield db
    dedup.reset_scope(token)


# ------------------------------------------------------------- the measure

def test_overlap_is_measured_against_the_shorter_span():
    """Not IoU: edge snapping can cut one copy short, and IoU reads a
    short-and-long pair as different while a viewer sees a repeat."""
    assert dedup.overlap_fraction(100, 130, 100, 200) == 1.0
    assert dedup.overlap_fraction(100, 130, 115, 200) == 0.5
    assert dedup.overlap_fraction(100, 130, 130, 200) == 0.0
    assert dedup.overlap_fraction(100, 130, 200, 260) == 0.0


# ---------------------------------------------------------------- the rule

def test_the_same_moment_is_not_shipped_twice(scope):
    dedup.record(1000.0, 1030.0, "clips/first.mp4")
    # Window N+1 sees the same highlight, a second later and a touch longer.
    assert dedup.already_shipped(1001.0, 1032.0) == "clips/first.mp4"


def test_a_different_moment_still_ships(scope):
    dedup.record(1000.0, 1030.0, "clips/first.mp4")
    assert dedup.already_shipped(1200.0, 1240.0) is None


def test_a_brief_shared_edge_is_not_a_duplicate(scope):
    """Two clips that merely touch are two clips. Only a mostly-repeated
    span is a repeat."""
    dedup.record(1000.0, 1060.0, "clips/first.mp4")
    assert dedup.already_shipped(1055.0, 1115.0) is None


def test_another_broadcast_is_not_deduped_against_this_one(db):
    token = dedup.set_scope(dedup.Scope(key="session:7", db=db))
    dedup.record(1000.0, 1030.0, "clips/first.mp4")
    dedup.reset_scope(token)

    token = dedup.set_scope(dedup.Scope(key="session:8", db=db))
    try:
        assert dedup.already_shipped(1000.0, 1030.0) is None, (
            "two streams share a clock but not their content")
    finally:
        dedup.reset_scope(token)


def test_without_a_scope_nothing_is_deduped(db):
    """`bta process` on a local file must behave exactly as it always has."""
    assert dedup.already_shipped(1000.0, 1030.0) is None
    dedup.record(1000.0, 1030.0, "clips/x.mp4")   # a no-op, not a crash
    assert db.shipped_spans("session:7") == []


def test_a_broken_database_never_sinks_a_clip(db, monkeypatch):
    """Dedup is a nicety; shipping the clip is the product."""

    class Broken:
        def shipped_spans(self, _scope):
            raise RuntimeError("database is locked")

        def record_shipped_span(self, *a):
            raise RuntimeError("database is locked")

    token = dedup.set_scope(dedup.Scope(key="session:7", db=Broken()))
    try:
        assert dedup.already_shipped(1000.0, 1030.0) is None
        dedup.record(1000.0, 1030.0, "clips/x.mp4")
    finally:
        dedup.reset_scope(token)


def test_spans_survive_a_restart(tmp_path):
    """The queue is memory; this is not. A second `bta watch` on the same
    broadcast must not re-ship what the first one already sent."""
    ws = Workspace(tmp_path / "ws").ensure()
    first = StateDB(ws.state_db)
    token = dedup.set_scope(dedup.Scope(key="session:7", db=first))
    dedup.record(1000.0, 1030.0, "clips/first.mp4")
    dedup.reset_scope(token)
    first.close()

    second = StateDB(ws.state_db)
    token = dedup.set_scope(dedup.Scope(key="session:7", db=second))
    try:
        assert dedup.already_shipped(1002.0, 1031.0) == "clips/first.mp4"
    finally:
        dedup.reset_scope(token)
        second.close()


# ------------------------------------------------------------- the wiring

def test_process_consults_and_records_the_dedup_scope():
    """Structural guard: the module existing is not the fix."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.process)
    assert "dedup.already_shipped(cand.start, cand.end)" in src, (
        "process no longer skips a moment another window already shipped")
    assert "dedup.record(" in src, "accepted clips are not recorded"

    watch_src = inspect.getsource(cli._watch_locked)
    assert "dedup.Scope(" in watch_src and "dedup.set_scope(" in watch_src, (
        "watch no longer scopes dedup to the broadcast being clipped")


def test_a_skipped_candidate_does_not_cost_the_window_its_clip():
    """The window must fall through to the next-ranked candidate rather
    than produce nothing — `continue`, not `break`."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.process)
    head = src[src.index("duplicate = dedup.already_shipped"):]
    body = head[:head.index("win_start = ")]
    assert "continue" in body and "break" not in body
