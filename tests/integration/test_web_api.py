"""Control API, pinned against the bugs that made it useless.

The shipped version opened the wrong database file AND selected columns
that do not exist, with both failures swallowed by `except Exception:
return []`. Every test here fails against that version — which is the
point: an endpoint that cannot distinguish "nothing happened" from "I am
broken" is worse than no endpoint.

The handlers are called as plain functions rather than through
Starlette's TestClient. That client needs an httpx package this venv does
not have, and adding one to run tests is how this project has broken its
CUDA torch build three times. Calling the handlers directly also means
HTTP path normalisation cannot mask what the traversal guard itself does
with a hostile string — which is the behaviour under test.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from clipforge import web
from clipforge.paths import Workspace
from clipforge.state import StateDB


@pytest.fixture()
def ws(tmp_path, monkeypatch):
    workspace = Workspace(tmp_path / "workspace").ensure()
    monkeypatch.setattr(web, "_workspace", lambda: workspace)
    return workspace


def _seed_job(ws) -> int:
    with StateDB(ws.state_db) as db:
        job_id = db.upsert_job("clip", "job-key-1", {"source": "x.mp4"})
        db.set_job_status(job_id, "running")
        run = db.stage_started(job_id, "s6_render", "c" * 64)
        db.stage_finished(run, artifact="clips/abc.mp4")
        return job_id


# ------------------------------------------------------------- jobs

def test_jobs_returns_real_rows(ws):
    """The original query named job_id/chunk_id/stage on the jobs table
    and opened state.db instead of state.sqlite3 — it could only ever
    return []."""
    _seed_job(ws)
    rows = web.list_jobs(limit=20)
    assert len(rows) == 1, rows
    assert rows[0]["key"] == "job-key-1"
    assert rows[0]["kind"] == "clip"
    assert rows[0]["status"] == "running"
    assert set(rows[0]) >= {"id", "kind", "key", "status", "created_at",
                            "updated_at"}


def test_jobs_reads_the_real_database_filename(ws):
    _seed_job(ws)
    assert ws.state_db.name == "state.sqlite3"
    assert not (ws.root / "state.db").exists(), (
        "nothing should be creating the filename the old API read")
    assert web.list_jobs(limit=20), (
        "the API is not reading the database the pipeline writes")
    health = web.health()
    assert health["state_db"].endswith("state.sqlite3")
    assert health["state_db_present"] is True


def test_an_unreadable_database_is_an_error_not_an_empty_list(ws):
    """The failure mode that hid two fatal bugs for the file's lifetime."""
    ws.state_db.write_bytes(b"this is not a sqlite database at all")
    with pytest.raises(HTTPException) as err:
        web.list_jobs(limit=20)
    assert err.value.status_code == 503
    assert "unreadable" in err.value.detail


def test_no_state_yet_is_an_honest_empty_list(ws):
    assert not ws.state_db.exists()
    assert web.list_jobs(limit=20) == []


# ------------------------------------------------------------ stages

def test_stage_runs_are_exposed_and_filterable(ws):
    job_id = _seed_job(ws)
    allruns = web.list_stage_runs(limit=50, job_id=None)
    assert len(allruns) == 1
    assert allruns[0]["stage"] == "s6_render"
    assert allruns[0]["status"] == "done"
    assert allruns[0]["artifact"] == "clips/abc.mp4"
    scoped = web.list_stage_runs(limit=50, job_id=job_id)
    assert [r["id"] for r in scoped] == [r["id"] for r in allruns]


def test_stage_runs_survive_no_database(ws):
    assert web.list_stage_runs(limit=50, job_id=None) == []


# ------------------------------------------------------------- clips

def test_clips_separate_shipped_from_quarantined(ws):
    (ws.clips / "good.mp4").write_bytes(b"\x00" * 32)
    rejected = ws.clips / "rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    (rejected / "bad.mp4").write_bytes(b"\x00" * 16)
    rows = {c["filename"]: c for c in web.list_clips()}
    assert rows["good.mp4"]["rejected"] is False
    assert rows["bad.mp4"]["rejected"] is True, (
        "a quarantined clip must never be listed as shippable")


def test_a_clip_streams(ws):
    (ws.clips / "good.mp4").write_bytes(b"\x00" * 32)
    resp = web.stream_clip("good.mp4")
    assert resp.media_type.startswith("video/")
    assert resp.path.name == "good.mp4"


@pytest.mark.parametrize("attack", [
    "../state.sqlite3",
    "..\\state.sqlite3",              # Windows: backslash IS a separator
    "sub/../../state.sqlite3",
    "..\\..\\config\\config.toml",
    "/etc/passwd",
    "C:\\Windows\\win.ini",
])
def test_the_clip_path_cannot_escape_the_clips_directory(ws, attack):
    """`ws.clips / filename` with an unsanitised segment escaped the
    directory — on Windows especially, where a backslash is legal in a
    URL segment and is also a path separator."""
    ws.state_db.write_bytes(b"secret")
    with pytest.raises(HTTPException) as err:
        web.stream_clip(attack)
    assert err.value.status_code in (400, 404), (
        f"{attack!r} was not rejected: {err.value.detail}")


def test_a_legitimate_name_is_still_served(ws):
    """Control: the traversal guard must not reject ordinary filenames,
    or the test above would pass on a handler that rejects everything."""
    (ws.clips / "b0a31c24.mp4").write_bytes(b"\x00" * 8)
    assert web.stream_clip("b0a31c24.mp4").path.name == "b0a31c24.mp4"


def test_the_rejected_view_is_also_confined(ws):
    (ws.clips / "rejected").mkdir(parents=True, exist_ok=True)
    with pytest.raises(HTTPException):
        web.stream_clip("..\\..\\state.sqlite3", rejected=True)


# ------------------------------------------------------------ delete

def test_deleting_a_clip_moves_it_and_its_sidecars_to_trash(ws):
    """A clip is 20+ minutes of GPU time, so delete is a MOVE. The
    sidecars go with it or the gallery shows metadata for a clip that no
    longer exists."""
    (ws.clips / "good.mp4").write_bytes(b"\x00" * 32)
    (ws.clips / "good.export.json").write_text("{}", encoding="utf-8")
    (ws.clips / "good.thumb.jpg").write_bytes(b"\xff\xd8")

    res = web.delete_clip(web.DeleteRequest(filename="good.mp4"))
    assert res["status"] == "trashed"
    assert len(res["files"]) == 3, res["files"]

    trash = ws.root / "trash"
    assert (trash / "good.mp4").is_file()
    assert (trash / "good.export.json").is_file()
    assert (trash / "good.thumb.jpg").is_file()
    assert not (ws.clips / "good.mp4").exists()
    assert web.list_clips() == []


def test_a_deleted_clip_is_recoverable(ws):
    """The whole reason this moves rather than unlinks."""
    (ws.clips / "good.mp4").write_bytes(b"payload")
    web.delete_clip(web.DeleteRequest(filename="good.mp4"))
    assert (ws.root / "trash" / "good.mp4").read_bytes() == b"payload"


def test_deleting_does_not_touch_unrelated_clips(ws):
    """Sidecars are matched by stem — a prefix match would take
    'good2.mp4' along with 'good.mp4'."""
    (ws.clips / "good.mp4").write_bytes(b"a")
    (ws.clips / "good2.mp4").write_bytes(b"b")
    web.delete_clip(web.DeleteRequest(filename="good.mp4"))
    assert (ws.clips / "good2.mp4").is_file(), (
        "deleting one clip removed another whose name shares a prefix")


def test_deleting_a_clip_spares_its_broll_variant(ws):
    """'abc.broll.mp4' shares the stem AND the dot, but it is its own
    gallery entry — quite likely the better of the two."""
    (ws.clips / "abc.mp4").write_bytes(b"a")
    (ws.clips / "abc.broll.mp4").write_bytes(b"b")
    (ws.clips / "abc.export.json").write_text("{}", encoding="utf-8")
    res = web.delete_clip(web.DeleteRequest(filename="abc.mp4"))
    assert (ws.clips / "abc.broll.mp4").is_file(), (
        "deleting the base clip took the b-roll cut with it")
    assert sorted(res["files"]) == ["abc.export.json", "abc.mp4"]


@pytest.mark.parametrize("attack, victim", [
    ("../state.sqlite3", "state.sqlite3"),
    ("..\\state.sqlite3", "state.sqlite3"),          # Windows separator
    ("..\\..\\config\\config.toml", "../config/config.toml"),
    ("sub/../../state.sqlite3", "state.sqlite3"),
])
def test_delete_cannot_escape_the_clips_directory(ws, attack, victim):
    """The destructive endpoint needs the containment guard most.

    Each case plants a real file at the escape target first. Without one,
    a traversal that got through would still 404 and the test would pass
    on a handler with no guard at all — which is exactly what happened
    when this was first written.
    """
    target = (ws.root / victim).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"important")

    with pytest.raises(HTTPException) as err:
        web.delete_clip(web.DeleteRequest(filename=attack))
    assert err.value.status_code == 400, (
        f"{attack!r} was not rejected by the guard: {err.value.detail}")
    assert target.read_bytes() == b"important", "the guard let it through"
    assert not (ws.root / "trash" / target.name).exists()


def test_deleting_a_missing_clip_is_a_404_not_a_crash(ws):
    with pytest.raises(HTTPException) as err:
        web.delete_clip(web.DeleteRequest(filename="nope.mp4"))
    assert err.value.status_code == 404


def test_a_quarantined_clip_can_be_deleted_too(ws):
    rejected = ws.clips / "rejected"
    rejected.mkdir(parents=True, exist_ok=True)
    (rejected / "bad.mp4").write_bytes(b"\x00" * 8)
    res = web.delete_clip(web.DeleteRequest(filename="bad.mp4", rejected=True))
    assert res["status"] == "trashed"
    assert (ws.root / "trash" / "bad.mp4").is_file()


# ----------------------------------------------------------- channels

def test_a_broken_watchlist_is_reported_not_faked(ws, monkeypatch):
    def _boom(_path):
        raise ValueError("corrupt watchlist")

    monkeypatch.setattr(web, "load_watchlist", _boom)
    with pytest.raises(HTTPException) as err:
        web.get_channels()
    assert err.value.status_code == 503


def test_a_missing_watchlist_is_an_empty_list_with_a_note(ws, monkeypatch):
    def _missing(_path):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(web, "load_watchlist", _missing)
    out = web.get_channels()
    assert out["channels"] == []
    assert "note" in out


# --------------------------------------------------- control endpoints





def test_start_process_spawns_task(ws, monkeypatch):
    spawned = []

    def mock_spawn(kind, desc, args):
        spawned.append((kind, desc, args))
        return "task-456"

    monkeypatch.setattr(web, "_spawn_task", mock_spawn)
    res = web.start_process(web.ProcessRequest(source="video.mp4", clips=3))
    assert res["status"] == "started"
    assert res["task_id"] == "task-456"
    assert len(spawned) == 1
    assert spawned[0][0] == "process"







# ------------------------------------------- voiceover / upscale endpoints
#
# Both features existed as tested library code with NO caller at all while
# their capability tiles reported LIVE. These pin the entry points, because
# "the function works" was already true and was not the problem.

@pytest.fixture()
def spawned(monkeypatch):
    """Capture what would have been run, instead of running it."""
    calls: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        web, "_spawn_task",
        lambda kind, desc, args: calls.append((kind, args)) or "task-test")
    return calls


def _clip(ws, name: str = "abc.mp4"):
    (ws.clips).mkdir(parents=True, exist_ok=True)
    (ws.clips / name).write_bytes(b"\x00")
    return name


def test_voiceover_spawns_with_the_script_in_a_file(ws, spawned):
    """The script must never travel as an argv element: it is operator
    prose with quotes and newlines, on a Windows host."""
    name = _clip(ws)
    out = web.start_voiceover(web.VoiceoverRequest(
        filename=name, script='He said "no" \n then left', gain_db=-3.0))
    assert out["status"] == "started"

    kind, args = spawned[0]
    assert kind == "voiceover"
    assert "--script-file" in args
    assert '"no"' not in " ".join(args)          # prose is NOT in argv
    path = args[args.index("--script-file") + 1]
    from pathlib import Path
    assert Path(path).read_text(encoding="utf-8") == 'He said "no" \n then left'
    assert args[args.index("--gain-db") + 1] == "-3.0"


def test_voiceover_rejects_an_empty_script_before_spawning(ws, spawned):
    _clip(ws)
    with pytest.raises(HTTPException) as err:
        web.start_voiceover(web.VoiceoverRequest(filename="abc.mp4",
                                                 script="   \n  "))
    assert err.value.status_code == 400
    assert not spawned


def test_voiceover_refuses_a_clip_outside_the_workspace(ws, spawned):
    with pytest.raises(HTTPException) as err:
        web.start_voiceover(web.VoiceoverRequest(
            filename=r"..\..\Windows\win.ini", script="hi"))
    assert err.value.status_code == 400
    assert not spawned


def test_upscale_only_accepts_the_offered_targets(ws, spawned):
    """An arbitrary height is refused at the edge. `upscale` itself also
    refuses a non-larger target, but a 401-pixel request should never
    reach a subprocess to find that out."""
    _clip(ws)
    for bad in (0, 720, 9999):
        with pytest.raises(HTTPException) as err:
            web.start_upscale(web.UpscaleRequest(filename="abc.mp4",
                                                 height=bad))
        assert err.value.status_code == 400
    assert not spawned

    web.start_upscale(web.UpscaleRequest(filename="abc.mp4", height=2560))
    kind, args = spawned[0]
    assert kind == "upscale"
    assert args[:2] == ["upscale", "--clip"]
    assert args[args.index("--height") + 1] == "2560"


def test_upscale_missing_clip_is_404_not_a_spawn(ws, spawned):
    with pytest.raises(HTTPException) as err:
        web.start_upscale(web.UpscaleRequest(filename="nope.mp4"))
    assert err.value.status_code == 404
    assert not spawned
