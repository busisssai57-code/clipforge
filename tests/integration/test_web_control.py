"""Cancel, bulk delete and live capture — the control endpoints.

Handlers are called as plain functions, matching test_web_api.py: HTTP
path normalisation would mask what the traversal guards themselves do
with a hostile string, and that is the behaviour worth pinning.

The cancel tests are the reason this file exists. The shipped Cancel
button did two wrong things at once, both measured on 2026-08-12:

* ``terminate()`` reaped only the tracked process, so the pipeline's own
  ffmpeg/yt-dlp/torch children kept running and holding the GPU — the
  operator pressed Cancel and watched the render continue;
* the drain thread then overwrote ``cancelled`` with ``failed · exit 1``,
  because TerminateProcess reports 1 on Windows, so the card claimed a
  crash that never happened.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
from fastapi import HTTPException

from clipforge import web
from clipforge.paths import Workspace


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace").ensure()
    monkeypatch.setattr(web, "_workspace", lambda: workspace)
    return workspace


@pytest.fixture(autouse=True)
def clean_tasks():
    web._tasks.clear()
    yield
    web._tasks.clear()


class _FakeProc:
    """A process that records how it was asked to die."""

    def __init__(self, *, dies_on: str = "terminate"):
        self.pid = 4242
        self.dies_on = dies_on
        self.alive = True
        self.calls: list[str] = []

    def poll(self):
        return None if self.alive else 1

    def terminate(self):
        self.calls.append("terminate")
        if self.dies_on == "terminate":
            self.alive = False

    def kill(self):
        self.calls.append("kill")
        self.alive = False

    def wait(self, timeout=None):
        self.calls.append("wait")
        if self.alive:
            raise __import__("subprocess").TimeoutExpired("cmd", timeout or 0)
        return 1


class _FakeGuard:
    """Stands in for the kill-on-close job object."""

    def __init__(self, proc: _FakeProc | None = None, *, works: bool = True):
        self.proc = proc
        self.works = works
        self.closed = False

    def close(self):
        self.closed = True
        if not self.works:
            raise OSError("job object unavailable")
        if self.proc is not None:
            self.proc.alive = False


def _task(**kw):
    defaults = dict(task_id="task-1", kind="process", description="d",
                    started_at=time.time())
    defaults.update(kw)
    t = web.BackgroundTask(**defaults)
    web._tasks[t.task_id] = t
    return t


# ------------------------------------------------------------- killing

def test_the_job_object_is_used_first():
    """It is the only mechanism that catches a grandchild spawned
    microseconds before the kill."""
    proc = _FakeProc(dies_on="never")
    guard = _FakeGuard(proc)
    how = web.kill_process_tree(proc, guard)
    assert guard.closed is True
    assert "job object" in how


def test_a_failed_job_object_falls_through_to_the_other_mechanisms():
    proc = _FakeProc(dies_on="terminate")
    how = web.kill_process_tree(proc, _FakeGuard(proc, works=False))
    assert "terminate" in how or "taskkill" in how
    assert proc.alive is False


def test_a_process_that_ignores_terminate_is_killed():
    proc = _FakeProc(dies_on="never")
    how = web.kill_process_tree(proc, None)
    assert "kill" in how
    assert proc.alive is False


def test_killing_nothing_is_not_an_error():
    assert web.kill_process_tree(None, None) == "no process"


# ------------------------------------------------------------- cancel

def test_cancel_reports_cancelled_not_failed(monkeypatch):
    """The screenshot bug: Cancel produced 'failed · exit 1'."""
    proc = _FakeProc()
    task = _task(process=proc, guard=_FakeGuard(proc))
    out = web.cancel_task("task-1")
    assert out["status"] == "cancelled"
    assert task.status == "cancelled"


def test_cancel_marks_intent_before_killing():
    """`cancelled` must be set BEFORE the process dies, or the drain
    thread — which wakes the instant the pipe closes — reads the exit
    code with no way to know a human asked for it."""
    proc = _FakeProc()
    task = _task(process=proc, guard=_FakeGuard(proc))
    web.cancel_task("task-1")
    assert task.cancelled is True


def test_a_cancelled_task_freezes_its_elapsed_time():
    proc = _FakeProc()
    task = _task(process=proc, guard=_FakeGuard(proc),
                 started_at=time.time() - 30)
    web.cancel_task("task-1")
    first = task.elapsed_s()
    time.sleep(0.05)
    assert task.elapsed_s() == first


def test_cancel_says_how_it_stopped_it():
    """Surfaced in the UI so 'stopped' is a claim with evidence."""
    proc = _FakeProc()
    out = _task(process=proc, guard=_FakeGuard(proc)) and \
        web.cancel_task("task-1")
    assert out["stopped_via"]


def test_cancelling_an_unknown_task_is_a_404():
    with pytest.raises(HTTPException) as exc:
        web.cancel_task("task-nope")
    assert exc.value.status_code == 404


def test_cancelling_a_finished_task_is_a_no_op():
    _task(status="completed")
    out = web.cancel_task("task-1")
    assert out["note"] == "not running"


def test_a_finished_task_keeps_its_exit_code_visible():
    task = _task(status="failed", return_code=2, finished_at=time.time())
    assert task.to_dict()["return_code"] == 2


# -------------------------------------------------------------- clear

def test_clear_removes_only_finished_tasks():
    _task(task_id="a", status="completed")
    _task(task_id="b", status="running")
    out = web.clear_finished_tasks()
    assert out["cleared"] == 1
    assert set(web._tasks) == {"b"}


def test_clear_never_orphans_a_running_process():
    """Dropping a running task from the registry would leave its process
    invisible and uncancellable — the exact failure the registry exists
    to prevent."""
    _task(task_id="b", status="running")
    web.clear_finished_tasks()
    assert "b" in web._tasks


# ------------------------------------------------------- bulk delete

def _clip(ws, name: str, *, rejected: bool = False):
    root = (ws.clips / "rejected") if rejected else ws.clips
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_bytes(b"video")
    return path


def test_bulk_delete_trashes_every_named_clip(ws):
    for n in ("a.mp4", "b.mp4", "c.mp4"):
        _clip(ws, n)
    out = web.delete_clips(web.BulkDeleteRequest(filenames=["a.mp4", "b.mp4"]))
    assert out["count"] == 2
    assert not (ws.clips / "a.mp4").exists()
    assert (ws.clips / "c.mp4").exists()


def test_bulk_delete_reports_failures_instead_of_abandoning_the_rest(ws):
    _clip(ws, "real.mp4")
    out = web.delete_clips(web.BulkDeleteRequest(
        filenames=["real.mp4", "ghost.mp4"]))
    assert out["trashed"] == ["real.mp4"]
    assert out["failed"][0]["filename"] == "ghost.mp4"


def test_bulk_delete_cannot_escape_the_clips_directory(ws):
    outside = ws.root / "secret.mp4"
    outside.write_bytes(b"keep me")
    out = web.delete_clips(web.BulkDeleteRequest(
        filenames=["..\\secret.mp4", "../secret.mp4"]))
    assert out["count"] == 0
    assert outside.exists()


def test_bulk_delete_refuses_an_empty_selection(ws):
    with pytest.raises(HTTPException) as exc:
        web.delete_clips(web.BulkDeleteRequest(filenames=[]))
    assert exc.value.status_code == 400


def test_deleting_is_a_move_to_trash_not_an_unlink(ws):
    _clip(ws, "a.mp4")
    web.delete_clips(web.BulkDeleteRequest(filenames=["a.mp4"]))
    assert (ws.root / "trash" / "a.mp4").exists()


# --------------------------------------------------- generated delete

def test_a_generated_piece_moves_to_trash(ws):
    piece = ws.root / "generated" / "my-piece"
    piece.mkdir(parents=True)
    (piece / "sequence.mp4").write_bytes(b"v")
    (piece / "shot_000.mp4").write_bytes(b"v")
    web.delete_generated(web.GeneratedDeleteRequest(slug="my-piece"))
    assert not piece.exists()
    moved = ws.root / "trash" / "generated" / "my-piece"
    assert (moved / "sequence.mp4").exists()
    assert (moved / "shot_000.mp4").exists()


@pytest.mark.parametrize("slug", ["..", "../..", "..\\..", "a/../..",
                                  "", "."])
def test_generated_delete_cannot_escape(ws, slug):
    (ws.root / "generated").mkdir(parents=True, exist_ok=True)
    victim = ws.root / "clips"
    with pytest.raises(HTTPException):
        web.delete_generated(web.GeneratedDeleteRequest(slug=slug))
    assert victim.exists()


def test_deleting_the_same_slug_twice_does_not_clobber_the_first(ws):
    for _ in range(2):
        piece = ws.root / "generated" / "dup"
        piece.mkdir(parents=True)
        (piece / "sequence.mp4").write_bytes(b"v")
        web.delete_generated(web.GeneratedDeleteRequest(slug="dup"))
    trash = ws.root / "trash" / "generated"
    assert {p.name for p in trash.iterdir()} == {"dup", "dup (1)"}


# ---------------------------------------------------------------- live

def test_live_rejects_an_empty_target(ws):
    with pytest.raises(HTTPException) as exc:
        web.start_live(web.LiveRequest(target="   "))
    assert exc.value.status_code == 400


def test_live_rejects_an_unknown_platform(ws):
    with pytest.raises(HTTPException):
        web.start_live(web.LiveRequest(target="x", platform="vimeo"))


@pytest.mark.parametrize("seconds", [10.0, 5000.0])
def test_live_bounds_the_segment_length(ws, seconds):
    with pytest.raises(HTTPException):
        web.start_live(web.LiveRequest(target="x", segment_s=seconds))


def test_only_one_capture_may_run_at_a_time(ws, monkeypatch):
    """The chunker takes the workspace lock; a second capture would fail
    with a lock error the operator has to decode."""
    _task(task_id="live-1", kind="live", status="running")
    with pytest.raises(HTTPException) as exc:
        web.start_live(web.LiveRequest(target="https://youtu.be/x"))
    assert exc.value.status_code == 409


def test_live_passes_the_segment_length_to_the_cli(ws, monkeypatch):
    seen = {}

    def _spawn(kind, description, args):
        seen["args"] = args
        return "task-x"

    monkeypatch.setattr(web, "_spawn_task", _spawn)
    web.start_live(web.LiveRequest(target="https://youtu.be/x", segment_s=90))
    assert "--segment" in seen["args"]
    assert "90.0" in seen["args"]


# ---------------------------------------------------------------- trim

class _Meta:
    """Enough of a clipmeta result for the trim path."""

    def __init__(self, duration=60.0, source="C:/src.mp4", exists=True):
        self.duration_s = duration
        self.source_path = source
        self.source_exists = exists

    def as_dict(self):
        return {"duration_s": self.duration_s, "source_path": self.source_path,
                "source_exists": self.source_exists}


@pytest.fixture()
def trimmable(ws, monkeypatch):
    """A clip on disk with a known duration and a source to re-run from."""
    _clip(ws, "clip.mp4")
    from clipforge import clipmeta
    monkeypatch.setattr(clipmeta, "resolve_clip",
                        lambda *a, **kw: _Meta())
    spawned = {}

    def _spawn(kind, description, args):
        spawned["args"] = args
        spawned["kind"] = kind
        return "task-trim"

    monkeypatch.setattr(web, "_spawn_task", _spawn)
    return spawned


def test_a_trim_becomes_the_cuts_either_side_of_what_you_kept(ws, trimmable):
    out = web.start_recut(web.RecutRequest(filename="clip.mp4",
                                           trim_start=5.0, trim_end=50.0))
    assert out["cuts"] == 2
    # 0-5 and 50-60 removed => 45s kept out of 60s.
    assert out["removed_s"] == pytest.approx(15.0)


def test_trimming_only_the_head_produces_one_cut(ws, trimmable):
    out = web.start_recut(web.RecutRequest(filename="clip.mp4",
                                           trim_start=4.0, trim_end=60.0))
    assert out["cuts"] == 1
    assert out["removed_s"] == pytest.approx(4.0)


def test_a_trim_needs_no_transcript(ws, trimmable, monkeypatch):
    """Requiring one would block an ordinary trim on any clip whose ASR
    artifact is missing — which, after the live-capture work, includes
    every silent window."""
    from clipforge import clipmeta

    def _boom(*_a, **_kw):
        raise AssertionError("the trim path must not read the transcript")

    monkeypatch.setattr(clipmeta, "transcript_for", _boom)
    out = web.start_recut(web.RecutRequest(filename="clip.mp4",
                                           trim_start=2.0, trim_end=45.0))
    assert out["status"] == "started"


def test_the_trim_reaches_the_cli_as_a_cut_file(ws, trimmable):
    web.start_recut(web.RecutRequest(filename="clip.mp4", trim_start=1.0,
                                     trim_end=41.0))
    args = trimmable["args"]
    assert "--cut-file" in args
    path = Path(args[args.index("--cut-file") + 1])
    assert json.loads(path.read_text(encoding="utf-8")) == [[0.0, 1.0],
                                                            [41.0, 60.0]]


@pytest.mark.parametrize("start,end", [
    (-1.0, 40.0),      # before the beginning
    (40.0, 5.0),       # inverted
    (5.0, 5.0),        # empty
    (0.0, 400.0),      # past the end of a 60s clip
])
def test_an_impossible_trim_is_refused(ws, trimmable, start, end):
    with pytest.raises(HTTPException) as exc:
        web.start_recut(web.RecutRequest(filename="clip.mp4",
                                         trim_start=start, trim_end=end))
    assert exc.value.status_code == 400


def test_a_trim_below_the_renderers_floor_is_refused(ws, trimmable):
    """THE bug this feature shipped with. `enforce_floor_on_frames`
    restores cuts until the clip fits again, so a 15s trim of a 34.7s clip
    logged "editor cuts: 2 span(s), 19.65s removed" and then rendered
    34.688s — a full-length duplicate that passed QA and reported success.
    Measured on a real run, 2026-08-12."""
    from clipforge.pacing import MIN_DURATION_S

    with pytest.raises(HTTPException) as exc:
        web.start_recut(web.RecutRequest(filename="clip.mp4",
                                         trim_start=5.0, trim_end=20.0))
    message = str(exc.value.detail)
    assert f"{MIN_DURATION_S:.1f}" in message
    # The message must explain the consequence, not just say "no".
    assert "full-length" in message


def test_the_floor_comes_from_the_renderer_not_a_copy(ws, trimmable):
    """A trim exactly at the floor is allowed; one just under is not.
    Pinning both sides means the endpoint cannot drift to its own
    hardcoded number."""
    from clipforge.pacing import MIN_DURATION_S

    ok = web.start_recut(web.RecutRequest(
        filename="clip.mp4", trim_start=0.0, trim_end=MIN_DURATION_S))
    assert ok["status"] == "started"
    with pytest.raises(HTTPException):
        web.start_recut(web.RecutRequest(
            filename="clip.mp4", trim_start=0.0, trim_end=MIN_DURATION_S - 0.5))


def test_the_detail_endpoint_publishes_the_floor(ws, monkeypatch):
    """The editor reads it from here rather than mirroring the constant."""
    from clipforge import clipmeta
    from clipforge.pacing import MIN_DURATION_S

    _clip(ws, "clip.mp4")
    monkeypatch.setattr(clipmeta, "resolve_clip", lambda *a, **kw: _Meta())
    monkeypatch.setattr(clipmeta, "transcript_for",
                        lambda *a, **kw: {"available": False})
    monkeypatch.setattr(clipmeta, "campath_for", lambda *a, **kw: {})
    out = web.clip_detail("clip.mp4")
    assert out["limits"]["min_clip_s"] == MIN_DURATION_S


def test_a_trim_that_keeps_everything_is_refused(ws, trimmable):
    """Nothing to re-render, and spending GPU minutes to reproduce the
    same clip is the kind of no-op that reads as a bug."""
    with pytest.raises(HTTPException) as exc:
        web.start_recut(web.RecutRequest(filename="clip.mp4",
                                         trim_start=0.0, trim_end=60.0))
    assert "keeps the whole clip" in str(exc.value.detail)


def test_a_clip_with_no_recorded_duration_cannot_be_trimmed(ws, monkeypatch):
    _clip(ws, "clip.mp4")
    from clipforge import clipmeta
    monkeypatch.setattr(clipmeta, "resolve_clip",
                        lambda *a, **kw: _Meta(duration=0.0))
    with pytest.raises(HTTPException) as exc:
        web.start_recut(web.RecutRequest(filename="clip.mp4", trim_start=1.0,
                                         trim_end=5.0))
    assert exc.value.status_code == 409


def test_a_trim_and_marked_words_survive_together(ws, monkeypatch):
    """The first version rebound `spans` when building word cuts, which
    silently threw the trim away whenever both were used in one edit."""
    _clip(ws, "clip.mp4")
    from clipforge import clipmeta
    monkeypatch.setattr(clipmeta, "resolve_clip", lambda *a, **kw: _Meta())
    monkeypatch.setattr(clipmeta, "transcript_for", lambda *a, **kw: {
        "available": True,
        "words": [{"start": 12.0, "end": 12.4, "word": "um"}]})
    spawned = {}
    monkeypatch.setattr(web, "_spawn_task",
                        lambda k, d, a: spawned.setdefault("args", a) or "t")
    out = web.start_recut(web.RecutRequest(
        filename="clip.mp4", trim_start=5.0, trim_end=50.0,
        cut_words=[12.0]))
    # head + tail + the word
    assert out["cuts"] == 3


def test_nothing_marked_and_no_trim_is_still_refused(ws, trimmable):
    with pytest.raises(HTTPException) as exc:
        web.start_recut(web.RecutRequest(filename="clip.mp4"))
    assert exc.value.status_code == 400
