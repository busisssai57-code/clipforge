"""The watcher's heartbeat: "is it running, and what is it doing?"

`bta watch` is meant to run unattended for days. Before the heartbeat, a
silent phone and a dead watcher looked identical from outside, and the
only way to tell them apart was reading a JSONL log.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from clipforge import watchstatus


def test_a_fresh_heartbeat_reads_as_running(tmp_path):
    watchstatus.write(tmp_path, channels=2, live=["@speed"], queue_pending=3,
                      gate=None, telegram_outbox=0)
    state = watchstatus.read(tmp_path)
    assert state["running"] is True
    assert state["channels"] == 2 and state["live"] == ["@speed"]
    assert state["age_s"] < 5


def test_an_old_heartbeat_reads_as_dead_not_quiet(tmp_path):
    """The failure this file exists for: a watcher that died an hour ago
    must not look like a watcher with nothing to do."""
    watchstatus.write(tmp_path, channels=1)
    stale = json.loads(watchstatus.path_for(tmp_path).read_text(encoding="utf-8"))
    stale["updated_at"] = time.time() - 3600
    watchstatus.path_for(tmp_path).write_text(json.dumps(stale), encoding="utf-8")

    state = watchstatus.read(tmp_path)
    assert state["running"] is False
    assert "not running" in state["note"]


def test_no_heartbeat_at_all_is_reported_honestly(tmp_path):
    state = watchstatus.read(tmp_path)
    assert state["running"] is False and "never wrote" in state["note"]


def test_a_corrupt_heartbeat_does_not_crash_the_reader(tmp_path):
    watchstatus.path_for(tmp_path).write_text("{not json", encoding="utf-8")
    assert watchstatus.read(tmp_path)["running"] is False


def test_a_clean_stop_clears_the_heartbeat(tmp_path):
    watchstatus.write(tmp_path, channels=1)
    watchstatus.clear(tmp_path)
    assert watchstatus.read(tmp_path)["running"] is False
    watchstatus.clear(tmp_path)          # twice is not an error


def test_writing_never_raises_on_a_bad_path(tmp_path):
    blocked = tmp_path / "file"
    blocked.write_text("i am not a directory", encoding="utf-8")
    watchstatus.write(blocked / "nested", channels=1)   # must not raise


def test_the_summary_says_what_an_operator_needs(tmp_path):
    watchstatus.write(tmp_path, channels=1, live=["@speed"], queue_pending=4,
                      gate="operator active (12s since last input)",
                      telegram_outbox=2)
    line = watchstatus.summarize(watchstatus.read(tmp_path))
    assert "LIVE: @speed" in line
    assert "4 window(s) queued" in line
    assert "clipping held" in line
    assert "2 clip(s) owed" in line


def test_the_summary_of_a_dead_watcher_says_so(tmp_path):
    assert "not running" in watchstatus.summarize({"running": False,
                                                   "note": "not running"})


def test_an_idle_but_alive_watcher_reads_as_free(tmp_path):
    watchstatus.write(tmp_path, channels=1, live=[], queue_pending=0,
                      gate=None, telegram_outbox=0)
    line = watchstatus.summarize(watchstatus.read(tmp_path))
    assert "none live" in line and "clipping free to run" in line


# ---------------------------------------------------------------- wiring

def test_watch_publishes_a_heartbeat_and_clears_it_on_exit():
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli._watch_locked)
    assert "watchstatus.write(" in src, "watch no longer publishes a heartbeat"
    assert "watchstatus.clear(" in src, (
        "a clean stop must not leave a heartbeat that reads as running")


def test_the_api_serves_the_same_file(tmp_path, monkeypatch):
    from clipforge import web
    from clipforge.paths import Workspace

    ws = Workspace(tmp_path / "ws").ensure()
    monkeypatch.setattr(web, "_workspace", lambda: ws)
    watchstatus.write(ws.root, channels=1, live=["@speed"], queue_pending=0,
                      gate=None, telegram_outbox=0)
    body = web.watch_status()
    assert body["running"] is True
    assert "LIVE: @speed" in body["summary"]


def test_the_dashboard_shows_the_watcher():
    """The tile is the only place an operator sees the watcher without a
    terminal, and the dashboard is one big file that is easy to edit past.
    """
    html = (Path(__file__).resolve().parents[2] / "clipforge" /
            "dashboard_live.html").read_text(encoding="utf-8")
    assert 'id="watchTile"' in html, "the watcher tile is gone from the home pane"
    assert "/api/watch/status" in html, "the tile no longer reads the heartbeat"
    assert "setInterval(refreshWatch" in html, (
        "the tile must refresh; a one-shot read goes stale the moment the "
        "watcher changes state")
    # Its own catch: a dead watcher must not blank the rest of the page,
    # which is exactly what a missing /api/models once did.
    head = html[html.index("async function refreshWatch"):]
    assert "catch" in head[:head.index("try{")] or "catch(e)" in head[:800]
