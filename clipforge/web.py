"""Local control API for the BTA dashboard.

Read-write HTTP over the same workspace the CLI writes. It binds to
localhost by default and it never uploads anything — the Authorization Law
holds here exactly as it does in the pipeline: this serves files and state
that already exist on disk and stops.

This module was rewritten after an audit found it had never worked. The
previous version opened ``state.db`` (the real file is ``state.sqlite3``)
and selected ``job_id, chunk_id, stage`` (the ``jobs`` table has ``id,
kind, key, status, payload, created_at, updated_at``) — two independently
fatal bugs, both swallowed by ``except Exception: return []``. The endpoint
therefore reported an idle pipeline unconditionally, forever. Three rules
follow from that:

1. **No bare excepts that fabricate an empty success.** A read that fails
   returns 503 and says why. An empty list must mean "nothing there".
2. **Queries live in StateDB**, next to the schema they depend on, so a
   column rename breaks them loudly rather than silently.
3. **Every path from a URL is resolved and confined** before it touches
   the filesystem.

The control endpoints (POST) spawn CLI commands as subprocesses so that
long-running work (generation, processing, watch) does not block the API.
Each task is tracked in-memory with a unique ID, and the dashboard polls
for progress.
"""

from __future__ import annotations

import json as _json
import mimetypes
import os
import secrets
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                               RedirectResponse, Response)
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

from clipforge import remote
from clipforge.config import load_config, load_watchlist
from clipforge.log import get_logger
from clipforge.paths import Workspace
from clipforge.state import StateDB

log = get_logger(__name__)

app = FastAPI(title="BTA Control API",
              description="Control surface for this machine. It serves and "
                          "acts on files that are already here; reaching it "
                          "from another device needs the access token.")


# ============================================================ access control
#
# This API spawns CLI subprocesses on the host. On loopback that is fine —
# anything that can reach it could already run the CLI directly. Off
# loopback it is not, so `bta web --lan` and `--tunnel` set a token in the
# environment and every request from a non-loopback address must carry it.
# The decision itself lives in clipforge.remote as a pure function; this is
# only the plumbing that feeds it a Request and turns its answer into a
# response.

#: Reachable without a credential, because they are how a credential is
#: obtained in the first place.
_OPEN_PATHS = frozenset({"/login", "/login/redeem", "/favicon.ico"})

#: Rebuilt from the environment at import time. Held in a mutable box so a
#: test (or a future `bta web --rotate`) can swap it without reaching into
#: module globals from three different places.
_policy: remote.AccessPolicy = remote.AccessPolicy.from_env()
_pairing = remote.PairingCodes()


def current_policy() -> remote.AccessPolicy:
    return _policy


def set_policy(policy: remote.AccessPolicy) -> None:
    global _policy
    _policy = policy


def _presented(request: Request) -> remote.Presented:
    """Collect whatever credential material this request carries."""
    header = request.headers.get(remote.TOKEN_HEADER)
    if not header:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            header = auth[7:]
    client = request.client.host if request.client else None
    forwarded = any(h in request.headers for h in remote.FORWARD_HEADERS)
    return remote.Presented(
        client_host=client,
        header_token=header,
        cookie_token=request.cookies.get(remote.COOKIE_NAME),
        query_token=request.query_params.get(remote.TOKEN_QUERY),
        forwarded=forwarded,
        host_header=request.headers.get("host"),
    )


def _wants_html(request: Request) -> bool:
    """A browser navigating, as opposed to a fetch() or a media element.

    Sec-Fetch-Mode is the reliable signal — an <img>/<video> request also
    sends an HTML-ish Accept header, and bouncing those to a login page
    renders a broken-image icon instead of an error anyone can read.
    """
    if request.headers.get("sec-fetch-mode") == "navigate":
        return True
    if request.headers.get("sec-fetch-dest") in ("document", "iframe"):
        return True
    accept = request.headers.get("accept") or ""
    return "text/html" in accept and "sec-fetch-mode" not in request.headers


async def _access_middleware(request: Request, call_next):
    path = request.url.path

    # Host first, and before the OPTIONS/open-path shortcut: a rebinding
    # page reaches a loopback server as same-origin, so CORS never runs
    # and the token is never asked for. The Host header is the one part
    # of that request the attacker's page cannot choose, which makes it
    # the only place the check can live. Applied to every path, because
    # /login is reachable without a credential by design.
    if not remote.host_is_allowed(request.headers.get("host")):
        log.warning("web.host_rejected",
                    host=(request.headers.get("host") or "")[:120],
                    path=path[:120])
        return JSONResponse(
            status_code=421,
            content={"error": "this server does not answer to that host "
                              "name — reach it by address, or add the name "
                              f"to {remote.ENV_ALLOWED_HOSTS}"})

    # Preflight carries no credentials by design and reveals nothing; it
    # must pass or every cross-origin call fails as a CORS error rather
    # than the 401 it actually is.
    if request.method == "OPTIONS" or path in _OPEN_PATHS:
        return await call_next(request)

    decision = remote.decide(_presented(request), _policy, pairing=_pairing)
    if not decision.allowed:
        log.warning("web.access_denied", path=path[:120],
                    client=(request.client.host if request.client else "?"),
                    reason=decision.reason)
        if _wants_html(request):
            nxt = request.url.path
            if request.url.query:
                nxt = f"{nxt}?{request.url.query}"
            return RedirectResponse(
                f"/login?next={_quote(nxt)}", status_code=303)
        return JSONResponse(status_code=401,
                            content={"error": decision.reason,
                                     "pair_at": "/login"})

    response = await call_next(request)
    if decision.set_cookie and decision.cookie_value:
        _set_access_cookie(response, decision.cookie_value)
    return response


def _set_access_cookie(response: Response, value: str) -> None:
    """Persist a proven credential for this browser.

    ``samesite=lax`` rather than ``none``: the dashboard is served by this
    same server, so lax covers it, and ``none`` would require Secure —
    which over plain http on a LAN means the cookie is silently dropped.
    A cross-origin client (the static site on :4321) authenticates with
    the token header or query instead, which needs no cookie at all.
    """
    response.set_cookie(remote.COOKIE_NAME, value, httponly=True,
                        samesite="lax", max_age=60 * 60 * 24 * 30, path="/")


def _quote(value: str) -> str:
    from urllib.parse import quote
    return quote(value, safe="")


def _safe_next(value: str | None) -> str:
    """Confine a post-login destination to a path on this server.

    ``next`` is attacker-choosable — it rides in the URL of a link anyone
    can send. Two things it must not be allowed to become:

    * an absolute URL, which turns /login into an open redirect that
      lends this server's name to a phishing page;
    * a ``javascript:`` URI, which the login page's own
      ``location.replace(next)`` would execute as script in this origin.

    So: one leading slash, never two (``//evil.com`` is protocol-relative
    and absolute), no scheme, no backslashes (browsers normalise those to
    forward slashes, so ``/\evil.com`` is another spelling of the same
    trick). Anything else becomes ``/``.
    """
    raw = (value or "").strip()
    if not raw:
        return "/"
    if not raw.startswith("/"):
        return "/"
    if raw.startswith("//") or raw.startswith("/\\"):
        return "/"
    if "\\" in raw or "\n" in raw or "\r" in raw:
        return "/"
    return raw


# Registration order decides nesting: Starlette wraps the LAST-added
# middleware outermost, so CORS must be added after the access check for
# a 401 to still carry CORS headers.
app.add_middleware(BaseHTTPMiddleware, dispatch=_access_middleware)

# Same-origin needs no CORS at all; this exists for the static site on
# :4321 and for a phone hitting the site's own host. The regex is bounded
# to loopback, RFC1918/CGNAT literals and tailnet names — never a
# wildcard, because credentials are allowed through it.
_ORIGIN_RE = (
    r"^https?://("
    r"localhost|127(\.\d{1,3}){3}|\[::1\]"
    r"|10(\.\d{1,3}){3}"
    r"|192\.168(\.\d{1,3}){2}"
    r"|172\.(1[6-9]|2\d|3[01])(\.\d{1,3}){2}"
    r"|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])(\.\d{1,3}){2}"
    r"|[a-zA-Z0-9-]+\.ts\.net"
    r"|[a-zA-Z0-9-]+\.trycloudflare\.com"
    r")(:\d+)?$"
)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=_ORIGIN_RE,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Strip-Columns", "X-Strip-Tile-W", "X-Strip-Tile-H",
                    "X-Strip-Duration"],
)


# --------------------------------------------------------- helpers

def _workspace() -> Workspace:
    try:
        cfg = load_config(Path("config/config.toml"))
        return Workspace(cfg.workspace.root).ensure()
    except Exception as exc:  # noqa: BLE001 - reported, not hidden
        log.warning("web.config_unavailable", error=str(exc)[:200],
                    note="falling back to ./workspace")
        return Workspace(Path("workspace")).ensure()


def _config():
    try:
        return load_config(Path("config/config.toml"))
    except Exception:  # noqa: BLE001
        from clipforge.config import AppConfig
        return AppConfig()


def _safe_clip_path(ws: Workspace, filename: str) -> Path:
    """Resolve ``filename`` strictly inside the clips directory.

    A URL path segment is attacker-controlled. On Windows a backslash is a
    legal character in a segment AND a path separator, so ``..\\..\\`` in a
    filename escaped the clips directory in the previous version. Resolve
    first, then prove containment — never validate the string.
    """
    clips_root = ws.clips.resolve()
    candidate = (clips_root / filename).resolve()
    if not candidate.is_relative_to(clips_root):
        log.warning("web.path_escape_blocked", requested=filename[:200])
        raise HTTPException(status_code=400, detail="invalid clip name")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="clip not found")
    return candidate


def _confined_clip(ws: Workspace, filename: str, *,
                   rejected: bool = False) -> Path:
    """``_safe_clip_path`` for the endpoints that also serve quarantine.

    Same rule and the same reason — resolve, then prove containment, never
    validate the string — but rooted at ``clips/rejected`` when asked. Kept
    as one function because the three action endpoints (dub, voiceover,
    upscale) all need it, and a security boundary copied per endpoint is a
    boundary that will eventually be copied wrong.
    """
    root = (ws.clips / "rejected") if rejected else ws.clips
    root = root.resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        log.warning("web.path_escape_blocked", requested=filename[:200])
        raise HTTPException(status_code=400, detail="invalid clip name")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="clip not found")
    return candidate


def _safe_generated_path(ws: Workspace, slug: str, filename: str) -> Path:
    """Resolve a generated video path strictly inside generated/."""
    gen_root = (Path(ws.root) / "generated").resolve()
    candidate = (gen_root / slug / filename).resolve()
    if not candidate.is_relative_to(gen_root):
        log.warning("web.path_escape_blocked", requested=f"{slug}/{filename}")
        raise HTTPException(status_code=400, detail="invalid path")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return candidate


# -------------------------------------------------- task tracking

@dataclass
class BackgroundTask:
    task_id: str
    kind: str
    description: str
    started_at: float
    process: subprocess.Popen | None = None
    status: str = "running"
    output_lines: list[str] = field(default_factory=list)
    return_code: int | None = None
    #: Set the instant the run stops, so elapsed freezes. Without it a
    #: finished card kept counting up forever — "196.2s elapsed · exit 1"
    #: on a task that died two minutes ago.
    finished_at: float | None = None
    #: Set BEFORE the kill. The drain thread reads it to decide what the
    #: exit code means: TerminateProcess reports 1 on Windows, so a
    #: cancelled run was being relabelled "failed · exit 1" and the
    #: operator could not tell a cancel from a crash.
    cancelled: bool = False
    #: Windows job object owning the whole child tree. `terminate()` only
    #: reaps the direct child; the pipeline's ffmpeg/yt-dlp/torch children
    #: survived it and kept the GPU busy, which is why Cancel appeared to
    #: do nothing at all.
    guard: Any = None
    progress: Any = None

    def elapsed_s(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    def to_dict(self, estimator: Any = None) -> dict[str, Any]:
        tail = self.output_lines[-30:]
        prog = (self.progress.as_dict(estimator)
                if self.progress is not None and self.status == "running"
                else None)
        return {
            "task_id": self.task_id,
            # The dashboard reads `id` and `log`; the API only ever sent
            # `task_id` and `output`, so the Cancel button posted to
            # /api/tasks/undefined/cancel and the log pane was always
            # empty. Both spellings ship, with `id`/`log` canonical.
            "id": self.task_id,
            "kind": self.kind,
            "description": self.description,
            "status": self.status,
            "elapsed_s": round(self.elapsed_s(), 1),
            "return_code": self.return_code,
            "output": tail,
            "log": "\n".join(tail),
            "progress": prog,
        }


_tasks: dict[str, BackgroundTask] = {}
_tasks_lock = threading.Lock()
_estimator: Any = None


def _stage_estimator():
    """Per-stage medians from this machine's own run history."""
    global _estimator
    from clipforge.progress import StageEstimator

    ws = _workspace()
    if _estimator is None or Path(_estimator.state_db) != Path(ws.state_db):
        _estimator = StageEstimator(ws.state_db)
    return _estimator


def kill_process_tree(proc: subprocess.Popen | None, guard: Any = None) -> str:
    """Kill a spawned run and everything it started. Returns what worked.

    Three mechanisms, tried in order, because each covers a case the
    others miss:

    * the job object, which the kernel applies to the whole tree at once
      and is the only one that catches a grandchild spawned microseconds
      before the kill;
    * ``taskkill /T /F``, which walks the tree by parent id on Windows;
    * ``terminate()``, which is all that exists if the first two are
      unavailable, and which was — measurably — leaving ffmpeg running.
    """
    if proc is None:
        return "no process"
    used: list[str] = []
    if guard is not None:
        try:
            guard.close()
            used.append("job object")
        except Exception as exc:  # noqa: BLE001 - fall through to the rest
            log.warning("web.guard_close_failed", error=str(exc)[:200])

    if sys.platform == "win32" and proc.poll() is None:
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           capture_output=True, timeout=15)
            used.append("taskkill /T")
        except (OSError, subprocess.SubprocessError) as exc:
            log.warning("web.taskkill_failed", error=str(exc)[:200])

    if proc.poll() is None:
        try:
            proc.terminate()
            used.append("terminate")
        except OSError:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            proc.kill()
            used.append("kill")
        except OSError:
            pass
    return ", ".join(used) or "already exited"


def _bta_cmd() -> list[str]:
    """The bta CLI invocation for subprocesses."""
    return [sys.executable, "-m", "clipforge.cli"]


def _cli_value(value: str, field: str) -> str:
    """Pass ``value`` to the CLI as data, or refuse it.

    Nothing here is shell-quoted — commands are spawned as an argv list,
    never through a shell — so this is not about shell metacharacters. It
    is about the OTHER parser: Typer reads argv, and an argument that
    begins with ``-`` is an OPTION, not the filename or brief the caller
    meant. A request naming its source ``--help`` or ``--niche`` does not
    inject a shell command, but it does steer the CLI somewhere the
    endpoint never intended, which is the same class of bug one layer in.

    A ``--`` separator cannot fix this: Click ends option parsing for the
    whole remaining argv, so it would swallow the flags the endpoint
    itself appends. Refusing the leading dash is the honest fix, and it
    costs nothing real — no clip, brief, URL or language legitimately
    starts with one.

    NUL is refused for the same reason every layer below refuses it: it
    truncates the string in the C API and what the CLI then sees is not
    what was validated.
    """
    text = (value or "").strip()
    if text.startswith("-"):
        raise HTTPException(
            400, f"{field} cannot start with '-' — that would read as a "
                 f"command-line option rather than a value")
    if "\x00" in text:
        raise HTTPException(400, f"{field} contains a null byte")
    return text


def _spawn_task(kind: str, description: str, args: list[str]) -> str:
    """Spawn a CLI command as a background subprocess and track it."""
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    cmd = _bta_cmd() + args
    log.info("web.spawn_task", task_id=task_id, kind=kind, cmd=" ".join(cmd))

    from clipforge.ingest.procguard import ProcessGuard
    from clipforge.progress import RunProgress

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        # errors='replace': the pipeline prints filenames and yt-dlp
        # prints stream titles, both of which carry characters that the
        # console codepage cannot decode. A UnicodeDecodeError in here
        # kills the drain thread and the task's log stops mid-run.
        errors="replace",
        cwd=str(Path(".")),
        # Prevent the child from inheriting the server's signal handlers
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32" else 0,
    )

    # Enroll the child in a kill-on-close job object. This is the same
    # mechanism the chunker uses on its streamlink/ffmpeg pair, and for the
    # same reason: killing the process we can see does not kill the ones it
    # started, and those are the ones holding the GPU.
    guard = ProcessGuard()
    guard.assign(proc)

    task = BackgroundTask(
        task_id=task_id,
        kind=kind,
        description=description,
        started_at=time.time(),
        process=proc,
        guard=guard,
        progress=RunProgress(kind=kind, started_at=time.time()),
    )
    with _tasks_lock:
        _tasks[task_id] = task

    def _drain():
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n\r")
            task.output_lines.append(stripped)
            if task.progress is not None:
                task.progress.feed(stripped)
            # Cap stored output at 500 lines
            if len(task.output_lines) > 500:
                task.output_lines = task.output_lines[-300:]
        proc.wait()
        task.return_code = proc.returncode
        task.finished_at = time.time()
        # A cancelled run exits non-zero by construction — TerminateProcess
        # reports 1 — so the exit code alone cannot distinguish it from a
        # crash. The flag set by the cancel endpoint can.
        task.status = ("cancelled" if task.cancelled else
                       "completed" if proc.returncode == 0 else "failed")
        try:
            guard.close()
        except Exception:  # noqa: BLE001 - teardown must not raise
            pass
        log.info("web.task_finished", task_id=task_id, code=proc.returncode,
                 status=task.status)

    t = threading.Thread(target=_drain, daemon=True)
    t.start()
    return task_id


# ================================================= READ ENDPOINTS

def _cli_command() -> str:
    """The exact invocation that runs this install's CLI.

    `bta` is a venv console script and is not on PATH unless the venv is
    activated, so telling an operator to "run bta auth" sends them to a
    CommandNotFoundException. Hand back the resolved path instead — the
    UI shows something that can be pasted and will work.
    """
    exe = Path(sys.executable).with_name(
        "bta.exe" if sys.platform == "win32" else "bta")
    if exe.is_file():
        return str(exe)
    return f'"{sys.executable}" -m clipforge.cli'


@app.get("/api/health")
def health() -> dict[str, Any]:
    ws = _workspace()
    return {
        "status": "ok",
        "workspace": str(ws.root),
        "state_db": str(ws.state_db),
        "state_db_present": ws.state_db.exists(),
        "cli": _cli_command(),
        "cwd": str(Path.cwd()),
        # Absolute, because every CLI command resolves config relative to
        # the CURRENT directory. A command copied out of this UI is run
        # from wherever the operator's shell happens to be, so telling
        # them the path is the difference between it working and a
        # "Config file not found: config\config.toml".
        "config": str(Path("config/config.toml").resolve()),
    }


@app.get("/api/jobs")
def list_jobs(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    """Recent pipeline jobs, newest first."""
    ws = _workspace()
    if not ws.state_db.exists():
        # Honest empty: there is genuinely no state yet.
        return []
    try:
        with StateDB(ws.state_db) as db:
            return [dict(r) for r in db.recent_jobs(limit)]
    except Exception as exc:  # noqa: BLE001
        log.error("web.jobs_query_failed", error=str(exc)[:300])
        raise HTTPException(
            status_code=503,
            detail=f"state database unreadable: {type(exc).__name__}") from exc


@app.get("/api/stages")
def list_stage_runs(
        limit: int = Query(50, ge=1, le=500),
        job_id: int | None = Query(None, ge=1)) -> list[dict[str, Any]]:
    """Stage attempts — the per-stage detail behind each job."""
    ws = _workspace()
    if not ws.state_db.exists():
        return []
    try:
        with StateDB(ws.state_db) as db:
            rows = (db.stage_runs_for(job_id) if job_id is not None
                    else db.recent_stage_runs(limit))
            return [dict(r) for r in rows]
    except Exception as exc:  # noqa: BLE001
        log.error("web.stages_query_failed", error=str(exc)[:300])
        raise HTTPException(
            status_code=503,
            detail=f"state database unreadable: {type(exc).__name__}") from exc


@app.get("/api/clips")
def list_clips() -> list[dict[str, Any]]:
    """Rendered clips on disk, newest first, with everything the pipeline
    recorded about each one.

    This used to return four fields — name, size, mtime, url — while the
    dashboard rendered ``clip.score``, ``clip.title`` and
    ``clip.duration_s``. Those were permanently undefined, so the score
    badge and grade breakdown were dead markup that failed silently. The
    join now lives in :mod:`clipforge.clipmeta`, which reads it back from
    the artifacts each stage already wrote and marks what is missing
    rather than filling gaps in.
    """
    from clipforge import clipmeta

    ws = _workspace()
    return [m.as_dict() for m in clipmeta.list_clips(ws)]


_VIDEO_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v"}


class DeleteRequest(BaseModel):
    filename: str
    rejected: bool = False


@app.post("/api/clips/delete")
def delete_clip(req: DeleteRequest) -> dict[str, Any]:
    """Move a clip and its sidecars to workspace/trash.

    Deliberately a MOVE, not an unlink. A clip is 20+ minutes of GPU time
    and the pipeline already treats destruction as something to avoid —
    QA failures are quarantined rather than deleted, for exactly this
    reason. The button reads as Delete and the clip leaves the gallery;
    recovering it is a drag out of one folder rather than a re-render.

    Sidecars (export pack, thumbnail, draft, b-roll variant) go with it,
    or the gallery is left showing metadata for a clip that no longer
    exists.
    """
    ws = _workspace()
    root = ((Path(ws.clips) / "rejected") if req.rejected
            else Path(ws.clips)).resolve()
    target = (root / req.filename).resolve()
    # Same containment rule as streaming: resolve first, then prove.
    if not target.is_relative_to(root):
        log.warning("web.delete_escape_blocked", requested=req.filename[:200])
        raise HTTPException(status_code=400, detail="invalid clip name")
    if not target.is_file():
        raise HTTPException(status_code=404, detail="clip not found")

    trash = Path(ws.root) / "trash"
    trash.mkdir(parents=True, exist_ok=True)
    moved: list[str] = []
    # Sidecars share the stem, but a bare `stem*` glob also matches a
    # DIFFERENT clip whose name starts the same way ("good2.mp4" when
    # deleting "good.mp4"). Require the dot, and never take another
    # video — "abc.broll.mp4" is its own gallery entry and quite likely
    # the one worth keeping.
    prefix = target.stem + "."
    for path in sorted(target.parent.iterdir()):
        if not path.is_file():
            continue
        if path != target:
            if not path.name.startswith(prefix):
                continue
            if path.suffix.lower() in _VIDEO_SUFFIXES:
                continue
        dest = trash / path.name
        try:
            if dest.exists():
                dest.unlink()
            path.replace(dest)
            moved.append(path.name)
        except OSError as exc:
            log.error("web.delete_failed", path=str(path), error=str(exc))
            raise HTTPException(
                status_code=500,
                detail=f"could not move {path.name}: {exc}") from exc

    log.info("web.clip_trashed", clip=target.name, files=len(moved))
    return {"status": "trashed", "files": moved, "trash": str(trash)}


class BulkDeleteRequest(BaseModel):
    filenames: list[str]
    rejected: bool = False


@app.post("/api/clips/delete-many")
def delete_clips(req: BulkDeleteRequest) -> dict[str, Any]:
    """Trash several clips in one call.

    A loop in the browser would do the same thing, but not atomically from
    the operator's point of view: a failure halfway leaves the gallery
    showing some clips gone and some not, with no statement of which. This
    reports every outcome, and a failure on one clip does not abandon the
    rest.
    """
    if not req.filenames:
        raise HTTPException(400, "nothing selected")
    if len(req.filenames) > 500:
        raise HTTPException(400, "too many at once (limit 500)")
    trashed: list[str] = []
    failed: list[dict[str, str]] = []
    for name in req.filenames:
        try:
            delete_clip(DeleteRequest(filename=name, rejected=req.rejected))
            trashed.append(name)
        except HTTPException as exc:
            failed.append({"filename": name, "error": str(exc.detail)})
    log.info("web.bulk_trashed", ok=len(trashed), failed=len(failed))
    return {"status": "done", "trashed": trashed, "failed": failed,
            "count": len(trashed)}


@app.post("/api/clips/empty-quarantine")
def empty_quarantine() -> dict[str, Any]:
    """Trash everything QA rejected.

    Still a move to trash, not an unlink — a quarantined clip is the
    evidence for why QA failed, and the operator who empties the folder in
    frustration is exactly the one who wants it back an hour later.
    """
    from clipforge import clipmeta

    ws = _workspace()
    names = [m.filename for m in clipmeta.list_clips(ws) if m.rejected]
    if not names:
        return {"status": "done", "trashed": [], "count": 0,
                "note": "quarantine is already empty"}
    return delete_clips(BulkDeleteRequest(filenames=names, rejected=True))


class GeneratedDeleteRequest(BaseModel):
    slug: str


@app.post("/api/generated/delete")
def delete_generated(req: GeneratedDeleteRequest) -> dict[str, Any]:
    """Trash one generated piece — the sequence and all of its shots.

    Generated pieces had no delete path at all: the gallery could show a
    forty-shot experiment and offer no way to remove it, so the folder
    grew until someone went to Explorer. Same rule as clips: it moves to
    workspace/trash rather than being unlinked.
    """
    ws = _workspace()
    gen_root = (Path(ws.root) / "generated").resolve()
    target = (gen_root / req.slug).resolve()
    # Resolve, then prove containment — never validate the string.
    if not target.is_relative_to(gen_root) or target == gen_root:
        log.warning("web.generated_escape_blocked", requested=req.slug[:200])
        raise HTTPException(400, "invalid piece name")
    if not target.is_dir():
        raise HTTPException(404, "no such generated piece")

    trash = Path(ws.root) / "trash" / "generated"
    trash.mkdir(parents=True, exist_ok=True)
    dest = trash / target.name
    # Never clobber a previous trashing of the same slug.
    n = 1
    while dest.exists():
        dest = trash / f"{target.name} ({n})"
        n += 1
    try:
        target.replace(dest)
    except OSError as exc:
        log.error("web.generated_delete_failed", slug=req.slug,
                  error=str(exc)[:200])
        raise HTTPException(500, f"could not move it: {exc}") from exc
    log.info("web.generated_trashed", slug=req.slug, dest=str(dest))
    return {"status": "trashed", "slug": req.slug, "trash": str(dest)}


@app.get("/api/clips/stream/{filename}")
def stream_clip(filename: str, rejected: bool = False) -> FileResponse:
    """Stream one clip.

    FileResponse handles HTTP range requests, which is what lets a phone
    or a seek bar work at all — see tools/serve_dashboard.py for the same
    requirement stated the hard way.
    """
    ws = _workspace()
    root = (ws.clips / "rejected") if rejected else ws.clips
    clips_root = root.resolve()
    candidate = (clips_root / filename).resolve()
    if not candidate.is_relative_to(clips_root):
        log.warning("web.path_escape_blocked", requested=filename[:200])
        raise HTTPException(status_code=400, detail="invalid clip name")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="clip not found")
    media_type = mimetypes.guess_type(candidate.name)[0] or "video/mp4"
    return FileResponse(candidate, media_type=media_type)


# ------------------------------------------------- editor data + derivatives
#
# Route order matters: the literal-prefix routes below must be declared
# before ``/api/clips/{filename}/...`` so that "thumb"/"filmstrip" are
# never captured as a filename.

def _clip_file(ws: Workspace, filename: str, rejected: bool) -> Path:
    """Resolve one clip inside clips/ (or clips/rejected/), or 404."""
    root = ((Path(ws.clips) / "rejected") if rejected
            else Path(ws.clips)).resolve()
    candidate = (root / filename).resolve()
    if not candidate.is_relative_to(root):
        log.warning("web.path_escape_blocked", requested=filename[:200])
        raise HTTPException(status_code=400, detail="invalid clip name")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="clip not found")
    return candidate


@app.get("/api/clips/thumb/{filename}")
def clip_thumb(filename: str, rejected: bool = False) -> FileResponse:
    """Poster frame for a grid card.

    Prefers the export pack's shipped thumbnail — the frame the operator
    would actually post — and only grabs one when that is absent.
    """
    from clipforge import uimedia

    ws = _workspace()
    clip = _clip_file(ws, filename, rejected)
    try:
        path = uimedia.poster(ws, clip)
    except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
        log.warning("web.thumb_failed", clip=filename[:120],
                    error=str(exc)[:300])
        raise HTTPException(503, f"thumbnail unavailable: {exc}") from exc
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/clips/filmstrip/{filename}")
def clip_filmstrip(filename: str, rejected: bool = False,
                   columns: int = Query(40, ge=4, le=120)) -> FileResponse:
    """A tiled sprite of evenly spaced frames for the timeline row.

    Tile geometry rides in response headers so the browser can position
    the sprite without a second round trip.
    """
    from clipforge import clipmeta, uimedia

    ws = _workspace()
    clip = _clip_file(ws, filename, rejected)
    meta = clipmeta.resolve_clip(ws, filename, rejected=rejected)
    duration = (meta.duration_s if meta and meta.duration_s else None)
    if not duration:
        # No render artifact: fall back to probing the file itself rather
        # than guessing a length and spacing the frames wrongly.
        from clipforge.ffmpeg import probe
        try:
            duration = probe(clip).duration_s
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(503, f"clip duration unknown: {exc}") from exc
    if not duration or duration <= 0:
        raise HTTPException(503, "clip duration unknown")

    try:
        strip = uimedia.filmstrip(ws, clip, duration_s=duration,
                                  columns=columns)
    except Exception as exc:  # noqa: BLE001
        log.warning("web.filmstrip_failed", clip=filename[:120],
                    error=str(exc)[:300])
        raise HTTPException(503, f"filmstrip unavailable: {exc}") from exc
    return FileResponse(strip.path, media_type="image/jpeg", headers={
        "X-Strip-Columns": str(strip.columns),
        "X-Strip-Tile-W": str(strip.tile_w),
        "X-Strip-Tile-H": str(strip.tile_h),
        "X-Strip-Duration": f"{duration:.3f}",
        "Access-Control-Expose-Headers":
            "X-Strip-Columns,X-Strip-Tile-W,X-Strip-Tile-H,X-Strip-Duration",
    })


@app.get("/api/clips/waveform/{filename}")
def clip_waveform(filename: str, rejected: bool = False,
                  buckets: int = Query(900, ge=50, le=4000)) -> dict[str, Any]:
    """Peak envelope for the timeline's audio row.

    A clip with no decodable audio returns 503 with the reason. Returning
    zeros would render as a flat line, which is a claim of silence.
    """
    from clipforge import clipmeta, uimedia

    ws = _workspace()
    clip = _clip_file(ws, filename, rejected)
    meta = clipmeta.resolve_clip(ws, filename, rejected=rejected)
    duration = (meta.duration_s if meta and meta.duration_s else 0.0) or 0.0
    try:
        return uimedia.waveform(ws, clip, duration_s=duration,
                                buckets=buckets)
    except Exception as exc:  # noqa: BLE001
        log.warning("web.waveform_failed", clip=filename[:120],
                    error=str(exc)[:300])
        raise HTTPException(503, f"waveform unavailable: {exc}") from exc


@app.get("/api/clips/subtitles/{filename}/{lang}")
def clip_subtitles(filename: str, lang: str,
                   rejected: bool = False) -> FileResponse:
    """A translated subtitle track written by `bta dub`."""
    ws = _workspace()
    clip = _clip_file(ws, filename, rejected)
    if not lang.isalpha() or not 2 <= len(lang) <= 5:
        raise HTTPException(400, "invalid language code")
    srt = clip.with_suffix("").with_suffix(f".{lang}.srt")
    if not srt.is_file():
        # ``with_suffix`` twice loses a stem that itself contains dots, so
        # fall back to building the name directly before reporting 404.
        srt = clip.parent / f"{clip.stem}.{lang}.srt"
    if not srt.is_file():
        raise HTTPException(404, f"no {lang} subtitles for this clip")
    return FileResponse(srt, media_type="text/plain; charset=utf-8",
                        filename=srt.name)


@app.get("/api/clips/{filename}/detail")
def clip_detail(filename: str, rejected: bool = False) -> dict[str, Any]:
    """Everything the detail modal and the editor need, in one call.

    Metadata, the four-dimension scorecard, QA checks, the word-timed
    transcript and the camera path the renderer followed. Sections that
    have no artifact behind them report ``available: false`` with a
    reason instead of an empty structure.
    """
    from clipforge import clipmeta

    from clipforge.pacing import MIN_DURATION_S

    ws = _workspace()
    meta = clipmeta.resolve_clip(ws, filename, rejected=rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")
    return {
        "clip": meta.as_dict(),
        "transcript": clipmeta.transcript_for(ws, filename,
                                              rejected=rejected),
        "campath": clipmeta.campath_for(ws, filename, rejected=rejected),
        # Sent rather than mirrored in the page, so the editor's trim
        # limits cannot drift from the renderer's actual floor.
        "limits": {"min_clip_s": MIN_DURATION_S},
    }


@app.get("/api/channels")
def get_channels() -> dict[str, Any]:
    """Watchlist channels. A malformed watchlist is an error, not an
    empty list — the operator needs to know their config did not load."""
    try:
        wl = load_watchlist(Path("config/channels.toml"))
    except FileNotFoundError:
        return {"channels": [], "note": "no config/channels.toml"}
    except Exception as exc:  # noqa: BLE001
        log.error("web.watchlist_failed", error=str(exc)[:300])
        raise HTTPException(status_code=503,
                            detail=f"watchlist unreadable: {exc}") from exc
    channels = getattr(wl, "channels", [])
    return {"channels": [
        c.model_dump() if hasattr(c, "model_dump") else dict(vars(c))
        for c in channels]}


@app.get("/api/config")
def get_config() -> dict[str, Any]:
    """Current configuration. Secrets are stripped."""
    cfg = _config()
    data = cfg.model_dump() if hasattr(cfg, "model_dump") else {}
    return data


@app.get("/api/niches")
def list_niches() -> list[dict[str, Any]]:
    """Selectable looks. A niche carries the whole format — generation
    style, colour grade, caption styling and pacing — so choosing one
    should mean touching nothing else."""
    from clipforge.niches import niche_summary

    return niche_summary()


@app.get("/api/capabilities")
def list_capabilities() -> list[dict[str, Any]]:
    """What this machine can actually do, probed live.

    The dashboard gates its feature tiles on this rather than on a
    hardcoded list, so a tile can never advertise something that is not
    installed — the failure mode that let blank renders and dead knobs
    sit unnoticed.
    """
    from clipforge.capabilities import summary

    return summary()


@app.get("/api/models")
def list_models() -> list[dict[str, Any]]:
    """Generation models, whether their weights are present, and what each
    is good at. ``usable`` is the only field a UI should gate on."""
    from clipforge.genvideo.models import describe_registry

    return describe_registry()


@app.get("/api/generated")
def list_generated() -> list[dict[str, Any]]:
    """Generated video pieces on disk, newest first."""
    ws = _workspace()
    gen_dir = Path(ws.root) / "generated"
    out: list[dict[str, Any]] = []
    if not gen_dir.is_dir():
        return out
    # stat() once per sequence, not three times. This is polled every four
    # seconds alongside the gallery, and the sort key, the size and the
    # timestamp were each asking the filesystem the same question.
    sequences: list[tuple[float, Path, int]] = []
    for seq in gen_dir.glob("*/sequence.mp4"):
        try:
            st = seq.stat()
        except OSError:
            continue
        sequences.append((st.st_mtime, seq, st.st_size))
    sequences.sort(key=lambda row: row[0], reverse=True)

    for mtime, seq, size in sequences[:20]:
        shots = sorted(seq.parent.glob("shot_*.mp4"))
        out.append({
            "slug": seq.parent.name,
            "shots": len(shots),
            "size_mb": round(size / (1024 * 1024), 2),
            "created_at": mtime,
            # `url` is what the gallery binds to; `sequence_url` is kept
            # because the name says what the file is.
            "url": f"/api/generated/stream/{seq.parent.name}/sequence.mp4",
            "sequence_url": f"/api/generated/stream/{seq.parent.name}/sequence.mp4",
            "shot_urls": [
                f"/api/generated/stream/{seq.parent.name}/{s.name}"
                for s in shots
            ],
        })
    return out


@app.get("/api/generated/stream/{slug}/{filename}")
def stream_generated(slug: str, filename: str) -> FileResponse:
    """Stream a generated video file."""
    ws = _workspace()
    path = _safe_generated_path(ws, slug, filename)
    media_type = mimetypes.guess_type(path.name)[0] or "video/mp4"
    return FileResponse(path, media_type=media_type)


@app.get("/api/generated/status")
def generation_status() -> list[dict[str, Any]]:
    """Provider quota status — reads the same ledger the router writes."""
    ws = _workspace()
    ledger_path = Path(ws.root) / "genvideo_quota.json"
    providers: list[dict[str, Any]] = []
    try:
        blob = _json.loads(ledger_path.read_text(encoding="utf-8"))
        now = time.time()
        for name, st in sorted((blob.get("providers") or {}).items()):
            until = float(st.get("exhausted_until", 0.0) or 0.0)
            calls = int(st.get("calls", 0) or 0)
            secs = float(st.get("seconds_generated", 0.0) or 0.0)
            available = until <= now
            providers.append({
                "name": name,
                "available": available,
                "available_in_s": max(0, until - now) if not available else 0,
                "calls": calls,
                "seconds_generated": round(secs, 1),
            })
    except FileNotFoundError:
        pass
    except Exception:  # noqa: BLE001 - reporting never raises
        providers.append({"name": "ledger", "available": False,
                          "error": "unreadable"})
    return providers


@app.get("/api/tasks")
def list_tasks() -> list[dict[str, Any]]:
    """All tracked background tasks, with live progress for running ones."""
    est = _stage_estimator()
    with _tasks_lock:
        return [t.to_dict(est) for t in _tasks.values()]


@app.get("/api/estimates")
def stage_estimates() -> dict[str, Any]:
    """What each stage has actually taken on this machine.

    Exposed because an ETA with no visible basis is indistinguishable from
    a made-up one. The dashboard shows these under Studio, so the estimate
    on a running job can be checked against the history it came from.
    """
    from clipforge.progress import CLIP_STAGES, humanize

    est = _stage_estimator()
    medians = est.all_medians()
    return {
        "stages": [
            {"stage": name, "label": label,
             "median_s": round(medians[name], 1) if name in medians else None,
             "human": humanize(medians.get(name)),
             "measured": name in medians}
            for name, label in CLIP_STAGES
        ],
        # Only meaningful once every stage has run at least once; said
        # plainly rather than summing the ones that happen to be known.
        "total_s": (round(sum(medians[n] for n, _ in CLIP_STAGES), 1)
                    if all(n in medians for n, _ in CLIP_STAGES) else None),
    }


# ================================================ CONTROL ENDPOINTS

class GenerateRequest(BaseModel):
    brief: str
    preset: str = "documentary"
    shots: int = 6
    aspect_ratio: str = "9:16"
    clip_it: bool = False
    #: The dashboard's niche picker sends the selected look as `niche`
    #: and the toggle as `clip`. Both were silently dropped: unknown
    #: fields are ignored by default, so "also run it through the
    #: clipper" never reached the CLI. Accept both spellings and fold
    #: them in rather than leaving a control that does nothing.
    niche: str | None = None
    clip: bool | None = None
    #: Same flag as the storyboard preview, so what was previewed is what
    #: gets generated.
    screenplay: bool = False
    #: Post-layer text. Both optional: a screenplay carries its own hook
    #: in the title page, and the handle falls back to [genvideo] handle.
    hook: str | None = None
    handle: str | None = None

    def resolved_preset(self) -> str:
        return (self.niche or self.preset or "documentary").strip()

    def resolved_clip(self) -> bool:
        return bool(self.clip_it or self.clip)


class StoryboardRequest(BaseModel):
    brief: str
    preset: str = "documentary"
    shots: int = 6
    #: Read the brief as Fountain: one beat per SCENE, and dialogue kept
    #: out of the picture prompt.
    screenplay: bool = False


class ScreenplayRequest(BaseModel):
    text: str


@app.post("/api/screenplay")
def parse_screenplay(req: ScreenplayRequest) -> dict[str, Any]:
    """Typed blocks, shots and speakers for the editor's live formatting.

    Parsed on the SERVER even though it is only formatting, so the editor
    and the generator cannot disagree about where a shot begins — a second
    parser in JavaScript is a second answer to that question, and the one
    the operator sees would be the one that is wrong.
    """
    from clipforge import screenplay

    return screenplay.summary(req.text or "")


@app.post("/api/storyboard")
def preview_storyboard(req: StoryboardRequest) -> dict[str, Any]:
    """Generate a storyboard breakdown preview without running GPU inference."""
    from clipforge.genvideo.presets import build_storyboard
    from clipforge.niches import resolve_preset
    if not req.brief.strip():
        raise HTTPException(400, "brief is required")
    try:
        preset = resolve_preset(req.preset)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    shots = build_storyboard(req.brief, preset, req.shots,
                             screenplay=req.screenplay)
    return {"brief": req.brief, "preset": req.preset,
            "screenplay": req.screenplay, "shots": shots}


@app.post("/api/generate")
def start_generation(req: GenerateRequest) -> dict[str, Any]:
    """Spawn a generation run as a background task."""
    from clipforge.niches import resolve_preset

    if not req.brief.strip():
        raise HTTPException(400, "brief is required")
    # A niche name (dark_mindset, ...) IS a valid preset — the dashboard's
    # niche picker sends its name straight through as `preset`, and the
    # old hardcoded 4-value check rejected every one of them with a 400.
    preset = req.resolved_preset()
    try:
        resolve_preset(preset)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    if not 1 <= req.shots <= 64:
        raise HTTPException(400, "shots must be 1-64")
    # Validated against the map generation actually uses, not a copy of
    # it: the two-value tuple here 400'd every request from the niche
    # picker the moment a niche declared 3:4, while the dashboard was
    # sending exactly what the niche asked for.
    from clipforge.genvideo.providers import _ASPECT_RATIOS

    if req.aspect_ratio not in _ASPECT_RATIOS:
        raise HTTPException(
            400, f"aspect_ratio must be one of {', '.join(_ASPECT_RATIOS)}")

    clip_it = req.resolved_clip()
    args = [
        "generate", _cli_value(req.brief, "brief"),
        "--preset", preset,
        "--shots", str(req.shots),
        "--aspect", req.aspect_ratio,
        "--clip" if clip_it else "--no-clip",
    ]
    if req.screenplay:
        args.append("--screenplay")
    if (req.hook or "").strip():
        args += ["--hook", _cli_value(req.hook, "hook")]
    if (req.handle or "").strip():
        args += ["--handle", _cli_value(req.handle, "handle")]
    task_id = _spawn_task(
        "generate",
        f"Generate {req.shots} shot(s) · {preset} · {req.brief[:60]}",
        args,
    )
    return {"task_id": task_id, "status": "started"}


class ProcessRequest(BaseModel):
    source: str
    clips: int = 3
    #: Tri-state on purpose, matching the CLI. `bta process` takes
    #: --jumpcut/--no-jumpcut defaulting to None so that "unset" means
    #: "use [pacing] from config". Typing this as `bool = False` made the
    #: API do two wrong things at once: it 422'd when the dashboard sent
    #: null for "niche default", and when it did accept a request it
    #: passed --no-jumpcut, silently overriding the configured default
    #: with off. Same for enhance.
    jumpcut: bool | None = None
    enhance: str | None = None
    niche: str | None = None
    broll: bool = False


_ENHANCE_CHOICES = ("off", "gentle", "strong")


@app.post("/api/process")
def start_process(req: ProcessRequest) -> dict[str, Any]:
    """Spawn a clip pipeline run as a background task."""
    if not req.source.strip():
        raise HTTPException(400, "source is required")
    if not 1 <= req.clips <= 20:
        raise HTTPException(400, "clips must be 1-20")
    if req.enhance is not None and req.enhance not in _ENHANCE_CHOICES:
        raise HTTPException(
            400, f"enhance must be one of {', '.join(_ENHANCE_CHOICES)}")

    args = ["process", _cli_value(req.source, "source"),
            "--clips", str(req.clips)]
    # Only pass the flag when the operator actually chose one; otherwise
    # the CLI applies the configured default, which is the point of the
    # "Niche default" option in the dashboard.
    if req.jumpcut is not None:
        args.append("--jumpcut" if req.jumpcut else "--no-jumpcut")
    if req.enhance is not None:
        args.extend(["--enhance", req.enhance])
    if req.niche:
        args.extend(["--niche", _cli_value(req.niche, "niche")])
    if req.broll:
        args.append("--broll")

    task_id = _spawn_task(
        "process",
        f"Process {req.source[:60]} · {req.clips} clip(s)",
        args,
    )
    return {"task_id": task_id, "status": "started"}


class GrabRequest(BaseModel):
    url: str
    clips: int = 3


@app.post("/api/grab")
def start_grab(req: GrabRequest) -> dict[str, Any]:
    """Download a URL and clip it — the one-click path from the dashboard.

    ``process`` takes a LOCAL file; handing it a URL fails after the
    download step it never runs. ``grab`` is the command that downloads
    first, so the dashboard routes URLs here and paths to /api/process.
    """
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "url is required")
    if not url.lower().startswith(("http://", "https://")):
        raise HTTPException(400, "grab needs an http(s) URL")
    if not 1 <= req.clips <= 20:
        raise HTTPException(400, "clips must be 1-20")
    task_id = _spawn_task("grab", f"Download and clip {url[:60]}",
                          ["grab", _cli_value(url, "url"),
                           "--clips", str(req.clips)])
    return {"task_id": task_id, "status": "started"}


class LiveRequest(BaseModel):
    target: str
    platform: str = "youtube"
    clips: int = 0
    quality: str = ""
    #: Seconds of stream per clipping window. Shorter means the first clip
    #: lands sooner, which is the whole point of watching a capture.
    segment_s: float = 300.0


@app.get("/api/live/check")
def check_live(url: str = Query(..., min_length=3, max_length=500)) -> dict[str, Any]:
    """Is this URL broadcasting right now?

    The dashboard asks before offering a Capture button, because the two
    paths are genuinely different: a live stream is captured as it runs, a
    finished video is downloaded and clipped. Guessing from the URL shape
    gets that wrong for exactly the case people care about — a /watch?v=
    link is both, depending on the minute.
    """
    from clipforge.ingest import youtube

    target = url.strip()
    if not target:
        raise HTTPException(400, "no url given")
    try:
        state = youtube.is_live(target)
    except Exception as exc:  # noqa: BLE001 - reported, never fatal
        log.info("web.live_probe_failed", error=str(exc)[:200])
        return {"live": None, "reason": f"could not check: {exc}"[:200],
                "target": target}
    title = youtube.live_title(target) if state else ""
    return {
        "live": state,
        "title": title,
        "target": target,
        "url": youtube.live_url(target),
        # None is not False, and the UI must not render it as "offline":
        # a probe that could not reach YouTube is a different answer.
        "reason": ("" if state is True else
                   "not broadcasting right now" if state is False else
                   "could not determine — the capture will end on its own "
                   "if there is no stream"),
    }


@app.post("/api/live/start")
def start_live(req: LiveRequest) -> dict[str, Any]:
    """Capture a live stream and clip it while it runs.

    Distinct from /api/grab, which downloads a finished video first. This
    segments the broadcast as it arrives and clips each segment, so the
    gallery fills during the stream instead of after it.
    """
    target = req.target.strip()
    if not target:
        raise HTTPException(400, "target is required")
    if req.platform not in ("youtube", "twitch", "kick"):
        raise HTTPException(400, "platform must be youtube, twitch or kick")
    if not 0 <= req.clips <= 20:
        raise HTTPException(400, "clips must be 0-20 (0 = the configured default)")

    # One capture at a time: the chunker takes the workspace lock, so a
    # second one would fail with a lock error the operator has to decode.
    with _tasks_lock:
        for t in _tasks.values():
            if t.kind in ("live", "watch") and t.status == "running":
                raise HTTPException(
                    409, f"already capturing ({t.description}). Stop that "
                         f"first — one capture owns the workspace.")

    if not 30.0 <= req.segment_s <= 3600.0:
        raise HTTPException(400, "segment_s must be between 30 and 3600")

    args = ["live", _cli_value(target, "target"), "--platform", req.platform,
            "--segment", str(req.segment_s)]
    if req.clips:
        args.extend(["--clips", str(req.clips)])
    if req.quality.strip():
        args.extend(["--quality", _cli_value(req.quality, "quality")])

    task_id = _spawn_task("live", f"Capture live · {target[:70]}", args)
    return {"task_id": task_id, "status": "started",
            "note": f"capturing — the first clip lands about "
                    f"{req.segment_s/60:.0f} min in, then one per window"}


class RerunRequest(BaseModel):
    """Re-run the pipeline on the source a clip came from.

    This is what makes the editor's controls real. The browser cannot cut
    or re-encode; the pipeline can, and it already takes every one of
    these as a flag. Rather than describing the command, the dashboard
    runs it.
    """

    filename: str
    rejected: bool = False
    clips: int = 1
    jumpcut: bool | None = None
    enhance: str | None = None
    niche: str | None = None
    broll: bool = False


@app.post("/api/clips/rerun")
def rerun_clip(req: RerunRequest) -> dict[str, Any]:
    from clipforge import clipmeta

    ws = _workspace()
    meta = clipmeta.resolve_clip(ws, req.filename, rejected=req.rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")
    if not meta.source_path:
        raise HTTPException(
            409, "no source recorded for this clip (the transcribe artifact "
                 "is missing), so it cannot be re-run")
    if not meta.source_exists:
        raise HTTPException(
            409, f"the source file is no longer on disk: {meta.source_path}")
    if req.enhance is not None and req.enhance not in _ENHANCE_CHOICES:
        raise HTTPException(
            400, f"enhance must be one of {', '.join(_ENHANCE_CHOICES)}")
    if not 1 <= req.clips <= 20:
        raise HTTPException(400, "clips must be 1-20")

    args = ["process", _cli_value(meta.source_path, "source"),
            "--clips", str(req.clips)]
    if req.jumpcut is not None:
        args.append("--jumpcut" if req.jumpcut else "--no-jumpcut")
    if req.enhance is not None:
        args.extend(["--enhance", req.enhance])
    if req.niche:
        args.extend(["--niche", req.niche])
    if req.broll:
        args.append("--broll")

    flags = " ".join(args[3:]) or "same settings"
    task_id = _spawn_task(
        "process", f"Re-run {Path(meta.source_path).name} · {flags}", args)
    return {"task_id": task_id, "status": "started",
            "source": meta.source_path, "args": args}


class CamPathRequest(BaseModel):
    """A director-camera move, authored in the editor."""

    filename: str
    rejected: bool = False
    #: [{t, cx, cy, h, easing}] in SOURCE pixels, clip-relative seconds.
    keyframes: list[dict[str, Any]]


@app.get("/api/clips/{filename}/campath")
def get_campath(filename: str, rejected: bool = False) -> dict[str, Any]:
    """The camera the renderer followed, plus a starting path to edit.

    Returns S4's tracked frames so the editor can DRAW what happened, and
    a default keyframe set so the operator starts from a real camera
    rather than an empty canvas.
    """
    from clipforge import clipmeta
    from clipforge.campath_edit import default_keyframes

    ws = _workspace()
    meta = clipmeta.resolve_clip(ws, filename, rejected=rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")
    tracked = clipmeta.campath_for(ws, filename, rejected=rejected)

    # Source geometry comes from the tracked artifact when there is one —
    # authoring against the wrong frame size puts every crop in the wrong
    # place, so this refuses rather than assuming 1920x1080.
    src_w = tracked.get("src_width") or meta.source_width
    src_h = tracked.get("src_height") or meta.source_height
    if not src_w or not src_h:
        raise HTTPException(
            409, "this clip has no recorded source dimensions, so a camera "
                 "path cannot be authored against it")
    duration = float(meta.duration_s or 0.0)
    return {
        "tracked": tracked,
        "src_width": int(src_w),
        "src_height": int(src_h),
        "duration_s": duration,
        "keyframes": [k.as_dict() for k in default_keyframes(
            src_width=int(src_w), src_height=int(src_h),
            duration_s=duration)],
    }


@app.post("/api/clips/recam")
def start_recam(req: CamPathRequest) -> dict[str, Any]:
    """Re-render a clip with an operator-authored camera move.

    Same shape as /api/clips/recut: the browser cannot re-frame video, the
    pipeline can, and it already takes a camera path as an artifact. The
    keyframes are validated HERE, before a GPU minute is spent, because a
    malformed path that fails after S1-S4 costs the whole run.
    """
    from clipforge import clipmeta
    from clipforge.campath_edit import CamPathError, parse_keyframes

    ws = _workspace()
    _confined_clip(ws, req.filename, rejected=req.rejected)
    meta = clipmeta.resolve_clip(ws, req.filename, rejected=req.rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")
    if not meta.source_path or not meta.source_exists:
        raise HTTPException(
            409, "the source this clip came from is no longer on disk, so it "
                 "cannot be re-framed")
    try:
        keys = parse_keyframes(req.keyframes)
    except CamPathError as exc:
        raise HTTPException(400, str(exc)) from exc

    tmp = Path(ws.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    path_file = tmp / f"campath_{uuid.uuid4().hex[:12]}.json"
    path_file.write_text(
        _json.dumps({"keyframes": [k.as_dict() for k in keys]}, indent=1),
        encoding="utf-8")

    args = ["process", _cli_value(meta.source_path, "source"), "--clips", "1",
            "--campath-file", str(path_file)]
    task_id = _spawn_task(
        "recam", f"Re-frame {req.filename[:32]} ({len(keys)} keyframe(s))",
        args)
    return {"task_id": task_id, "status": "started", "keyframes": len(keys)}


class ClipTextRequest(BaseModel):
    filename: str
    rejected: bool = False
    title: str | None = None
    caption: str | None = None
    hashtags: list[str] | None = None


@app.post("/api/clips/text")
def edit_clip_text(req: ClipTextRequest) -> dict[str, Any]:
    """Rewrite the copy in a clip's export pack.

    The pack is the operator-facing draft — title, caption, hashtags. It
    is text beside the video, so editing it is a real, complete action
    that needs no re-render, and the dashboard should just do it.

    Per-platform captions are re-trimmed to each platform's real limit so
    the pack cannot drift into claiming an over-length caption is ready.
    """
    from clipforge.export_pack import fit_caption

    ws = _workspace()
    root = ((Path(ws.clips) / "rejected") if req.rejected
            else Path(ws.clips)).resolve()
    clip = (root / req.filename).resolve()
    if not clip.is_relative_to(root):
        raise HTTPException(400, "invalid clip name")
    if not clip.is_file():
        raise HTTPException(404, "clip not found")

    pack_path = clip.with_suffix(".export.json")
    try:
        pack = _json.loads(pack_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(
            404, "no export pack beside this clip — it is written when the "
                 "publish stage runs") from exc
    except ValueError as exc:
        raise HTTPException(422, f"export pack is not valid JSON: {exc}") from exc

    if req.title is not None:
        pack["title"] = req.title.strip()
    if req.caption is not None:
        pack["caption"] = req.caption.strip()
    if req.hashtags is not None:
        pack["hashtags"] = [h if h.startswith("#") else f"#{h}"
                            for h in req.hashtags if h.strip()]

    body = str(pack.get("caption") or "")
    tags = list(pack.get("hashtags") or [])
    for name, plat in (pack.get("platforms") or {}).items():
        if not isinstance(plat, dict):
            continue
        limit = int(plat.get("limit") or 2200)
        joined = (body + ("\n\n" + " ".join(tags) if tags else "")).strip()
        fitted, trimmed = fit_caption(joined, limit)
        plat["caption"] = fitted
        plat["trimmed"] = bool(trimmed)
        plat["hashtags"] = tags

    from clipforge.paths import atomic_write_json
    try:
        atomic_write_json(pack_path, pack)
    except Exception as exc:  # noqa: BLE001
        log.error("web.pack_write_failed", error=str(exc)[:300])
        raise HTTPException(500, f"could not write export pack: {exc}") from exc

    log.info("web.pack_edited", clip=clip.name)
    return {"status": "saved", "pack": pack}


# Ratings are a UI preference, not pipeline state, so they live in the
# UI cache next to the other derivatives — never in the state database.
def _ratings_path(ws: Workspace) -> Path:
    from clipforge import uimedia
    return uimedia.cache_dir(ws) / "ratings.json"


@app.get("/api/ratings")
def get_ratings() -> dict[str, Any]:
    try:
        return _json.loads(_ratings_path(_workspace()).read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {"liked": [], "hidden": []}


class RatingRequest(BaseModel):
    filename: str
    liked: bool | None = None
    hidden: bool | None = None


@app.post("/api/ratings")
def set_rating(req: RatingRequest) -> dict[str, Any]:
    ws = _workspace()
    path = _ratings_path(ws)
    try:
        blob = _json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        blob = {"liked": [], "hidden": []}
    for key, want in (("liked", req.liked), ("hidden", req.hidden)):
        if want is None:
            continue
        current = set(blob.get(key) or [])
        current.add(req.filename) if want else current.discard(req.filename)
        blob[key] = sorted(current)
    try:
        path.write_text(_json.dumps(blob, indent=1), encoding="utf-8")
    except OSError as exc:
        raise HTTPException(500, f"could not save rating: {exc}") from exc
    return blob


@app.get("/api/auth/status")
def auth_status() -> list[dict[str, Any]]:
    """Which platform sessions `bta auth` has saved.

    Drafting a post needs a logged-in browser session on disk. Reporting
    which exist is what lets the UI offer the action only where it can
    actually complete.
    """
    ws = _workspace()
    auth_dir = Path(ws.root) / "auth"
    out = []
    for platform in ("youtube", "tiktok", "instagram", "x_twitter"):
        session = auth_dir / f"{platform}_session.json"
        out.append({
            "platform": platform,
            "authorized": session.is_file(),
            "saved_at": (session.stat().st_mtime if session.is_file() else None),
        })
    return out


class PostRequest(BaseModel):
    filename: str
    platform: str
    rejected: bool = False


_PLATFORMS = ("youtube", "tiktok", "instagram", "x_twitter")


@app.post("/api/post")
def prepare_draft(req: PostRequest) -> dict[str, Any]:
    """Prepare a DRAFT post. It fills the upload form and stops.

    Nothing is published from here. `bta post` is draft-only by design
    (VERIFICATION.md, 2026-07-27 amendment) — it never presses Publish.
    The dashboard passes --yes because the operator's click on THIS clip
    IS the per-clip approval that flag stands for; it does not widen what
    the command will do.
    """
    from clipforge import clipmeta

    if req.platform not in _PLATFORMS:
        raise HTTPException(400, f"platform must be one of {', '.join(_PLATFORMS)}")
    ws = _workspace()
    meta = clipmeta.resolve_clip(ws, req.filename, rejected=req.rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")

    session = Path(ws.root) / "auth" / f"{req.platform}_session.json"
    if not session.is_file():
        raise HTTPException(
            409, f"no saved {req.platform} session. Run: bta auth {req.platform}")

    root = (Path(ws.clips) / "rejected") if req.rejected else Path(ws.clips)
    clip_path = root / req.filename
    args = ["post", "--clip", str(clip_path), "--platform", req.platform,
            "--yes"]
    if meta.title:
        args.extend(["--title", _cli_value(meta.title, "title")])
    if meta.caption:
        args.extend(["--caption", _cli_value(meta.caption, "caption")])

    label = meta.title or req.filename[:40]
    task_id = _spawn_task("post", f"Draft {req.platform} post · {label}", args)
    return {"task_id": task_id, "status": "started",
            "note": "draft only — nothing is published"}


#: Uploads land in workspace/downloads — the same folder `grab` writes to,
#: so an uploaded file and a downloaded one are handled identically from
#: there on. Bounded because a browser POST holds the whole body in memory.
_MAX_UPLOAD_BYTES = 8 * 1024 * 1024 * 1024
_UPLOAD_SUFFIXES = {".mp4", ".mkv", ".mov", ".webm", ".m4v", ".ts", ".mp3",
                    ".wav", ".m4a", ".flac"}


@app.post("/api/upload")
async def upload_media(request: Request,
                       name: str = Query(..., min_length=1,
                                         max_length=255)) -> dict[str, Any]:
    """Save a dropped file into workspace/downloads and report its path.

    The body is the raw file, with the name in the query string. That
    avoids a multipart parser (and the python-multipart dependency it
    needs) for something a browser can send directly as a fetch body, and
    it streams rather than buffering a multi-gigabyte video in memory.

    The pipeline reads media from a path, so an upload is complete once
    the bytes are on disk — this deliberately does not start a run.
    """
    # A client can claim any name it likes, separators included. Take the
    # bare filename and never trust the string.
    raw = Path(name).name
    if not raw or raw in (".", ".."):
        raise HTTPException(400, "invalid file name")
    suffix = Path(raw).suffix.lower()
    if suffix not in _UPLOAD_SUFFIXES:
        raise HTTPException(
            415, f"{suffix or 'that type'} is not a media file this pipeline "
                 f"reads ({', '.join(sorted(_UPLOAD_SUFFIXES))})")

    ws = _workspace()
    dest_dir = (Path(ws.root) / "downloads")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = (dest_dir / raw).resolve()
    if not dest.is_relative_to(dest_dir.resolve()):
        raise HTTPException(400, "invalid file name")
    # Never overwrite: a second upload of the same name is a new file, and
    # silently replacing one that a job is reading would corrupt that run.
    stem, n = dest.stem, 1
    while dest.exists():
        dest = dest_dir / f"{stem} ({n}){suffix}"
        n += 1

    written = 0
    try:
        with dest.open("wb") as fh:
            # The body streams chunk by chunk; a NameError here previously
            # referenced an ``upload`` variable that never existed, so every
            # upload died with a 500 before writing a byte.
            async for chunk in request.stream():
                written += len(chunk)
                if written > _MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "file is larger than 8 GB")
                fh.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    except OSError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(500, f"could not write upload: {exc}") from exc

    log.info("web.upload", name=dest.name, bytes=written)
    return {"status": "saved", "path": str(dest), "name": dest.name,
            "size_mb": round(written / (1024 * 1024), 2)}


@app.get("/api/dub/languages")
def dub_languages() -> list[dict[str, Any]]:
    """Targets, each saying whether it can be voiced or only subtitled."""
    from clipforge.dubbing import language_options

    return language_options()


class DubRequest(BaseModel):
    filename: str
    lang: str
    rejected: bool = False
    subtitles_only: bool = False
    keep_original: float = 0.0


@app.post("/api/dub")
def start_dub(req: DubRequest) -> dict[str, Any]:
    """Translate a clip and, where a voice exists, dub its audio."""
    from clipforge.dubbing import LANGUAGES

    known = {code for code, _ in LANGUAGES}
    if req.lang not in known:
        raise HTTPException(400, f"unknown language {req.lang!r}")
    if not 0.0 <= req.keep_original <= 1.0:
        raise HTTPException(400, "keep_original must be between 0 and 1")

    ws = _workspace()
    _confined_clip(ws, req.filename, rejected=req.rejected)

    args = ["dub", "--clip", _cli_value(req.filename, "filename"),
            "--lang", req.lang]
    if req.subtitles_only:
        args.append("--subtitles-only")
    if req.keep_original > 0:
        args.extend(["--keep-original", str(req.keep_original)])

    task_id = _spawn_task("dub", f"Dub {req.filename[:36]} to {req.lang}", args)
    return {"task_id": task_id, "status": "started"}


class RecutRequest(BaseModel):
    filename: str
    rejected: bool = False
    #: Word START TIMES, clip-relative — the editor keys its marks by
    #: `w.start`, so that is what comes back. Matched EXACTLY against the
    #: same `transcript_for` output the editor rendered from, not by
    #: nearest-neighbour: both sides are reading one artifact, so a value
    #: that does not match means the clip changed underneath the page, and
    #: cutting the closest word instead would remove the wrong audio.
    cut_words: list[float] = []
    #: Start times of words whose PRECEDING pause was marked.
    cut_gaps: list[float] = []
    #: Keep only [trim_start, trim_end], clip-relative seconds. Expressed
    #: as a KEEP range rather than two cuts because that is what the
    #: operator dragged, and converting it here means the endpoint can
    #: check it against the clip's real duration — a browser that
    #: mis-measures the timeline cannot silently ask for a trim past the
    #: end of the video.
    trim_start: float | None = None
    trim_end: float | None = None


@app.post("/api/clips/recut")
def start_recut(req: RecutRequest) -> dict[str, Any]:
    """Re-render a clip with the editor's marked words and pauses removed.

    The editor used to mark cuts and then tell the operator to go run a CLI
    command themselves — `edSave`, `tlDel` and the notice bar were three
    different toasts describing the same manual step. This turns the plan
    into the actual re-render.

    Word indices become WINDOW-relative spans here, on the server, from the
    same `clipmeta.transcript_for` the editor rendered from — so the two
    cannot disagree about which word index 37 is.
    """
    from clipforge import clipmeta

    trimming = req.trim_start is not None or req.trim_end is not None
    if not req.cut_words and not req.cut_gaps and not trimming:
        raise HTTPException(400, "nothing marked to cut")

    ws = _workspace()
    _confined_clip(ws, req.filename, rejected=req.rejected)

    meta = clipmeta.resolve_clip(ws, req.filename, rejected=req.rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")
    if not meta.source_path or not meta.source_exists:
        raise HTTPException(
            409, "the source this clip came from is no longer on disk, so it "
                 "cannot be re-rendered")

    spans: list[tuple[float, float]] = []

    # ---- trim: a KEEP range becomes the two cuts around it -------------
    if trimming:
        duration = float(meta.duration_s or 0.0)
        if duration <= 0:
            raise HTTPException(
                409, "this clip's duration was never recorded, so a trim "
                     "cannot be checked against it")
        start = float(req.trim_start or 0.0)
        end = float(req.trim_end if req.trim_end is not None else duration)
        if not 0.0 <= start < end <= duration + 0.05:
            raise HTTPException(
                400, f"trim must satisfy 0 <= start < end <= {duration:.2f}s; "
                     f"got {start:.2f}–{end:.2f}")
        end = min(end, duration)
        # The pipeline will not render below its duration floor: the
        # renderer restores the smallest cuts until the clip fits again
        # (pacing.enforce_floor_on_frames), so a trim under the floor
        # silently produces a FULL-LENGTH duplicate that passes QA and
        # looks like success. Measured 2026-08-12: trimming a 34.7s clip
        # to 15s logged "editor cuts: 2 span(s), 19.65s removed" and then
        # rendered 34.688s. Refusing here is the difference between an
        # error and a lie.
        from clipforge.pacing import MIN_DURATION_S

        kept = end - start
        if kept < MIN_DURATION_S:
            raise HTTPException(
                400,
                f"keeping {kept:.1f}s would fall below this pipeline's "
                f"{MIN_DURATION_S:.1f}s minimum, and the renderer restores "
                f"cuts until a clip fits again — so the trim would silently "
                f"produce a full-length copy. Keep at least "
                f"{MIN_DURATION_S:.1f}s, or re-run the source with different "
                f"settings to get a shorter clip.")
        if start > 0.01:
            spans.append((0.0, start))
        if end < duration - 0.01:
            spans.append((end, duration))
        if not spans:
            raise HTTPException(
                400, "that trim keeps the whole clip — nothing to re-render")

    if not req.cut_words and not req.cut_gaps:
        # Trim-only: no transcript needed. Requiring one would block a
        # perfectly ordinary trim on a clip whose ASR artifact is missing.
        return _spawn_recut(ws, meta, req, spans)

    tr = clipmeta.transcript_for(ws, req.filename, rejected=req.rejected)
    if not tr.get("available"):
        raise HTTPException(409, f"no transcript: {tr.get('reason')}")
    words = tr.get("words") or []

    by_start = {round(float(w["start"]), 3): i for i, w in enumerate(words)}

    def _index(start: float, what: str) -> int:
        hit = by_start.get(round(float(start), 3))
        if hit is None:
            raise HTTPException(
                409, f"{what} at {start}s is not in this clip's transcript — "
                     "the page is showing a different version of it; reload "
                     "before re-cutting")
        return hit

    # EXTENDS the trim spans rather than rebinding — a rebind here would
    # silently drop a trim whenever words were marked in the same edit.
    for start in req.cut_words:
        w = words[_index(start, "a marked word")]
        spans.append((float(w["start"]), float(w["end"])))
    for start in req.cut_gaps:
        i = _index(start, "a marked pause")
        if i == 0:
            # The first word has no preceding gap; the UI never draws one
            # there, so this means the page and the artifact disagree.
            raise HTTPException(409, "a pause was marked before the first "
                                     "word; reload the clip")
        spans.append((float(words[i - 1]["end"]), float(words[i]["start"])))

    spans = [(a, b) for a, b in spans if b > a]
    if not spans:
        raise HTTPException(400, "the marked items have no measurable length")

    return _spawn_recut(ws, meta, req, spans)


def _spawn_recut(ws: Workspace, meta: Any, req: RecutRequest,
                 spans: list[tuple[float, float]]) -> dict[str, Any]:
    """Write the cut plan and start the re-render.

    Shared by the trim-only path and the marked-words path so the two
    cannot drift apart in how they spell the command — the trim route
    needs no transcript, which is the only thing that differs.
    """
    spans = [(a, b) for a, b in sorted(spans) if b > a]
    if not spans:
        raise HTTPException(400, "nothing measurable to cut")

    tmp = Path(ws.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    cut_file = tmp / f"recut_{uuid.uuid4().hex[:12]}.json"
    cut_file.write_text(_json.dumps([[a, b] for a, b in spans]),
                        encoding="utf-8")

    args = ["process", _cli_value(meta.source_path, "source"), "--clips", "1",
            "--cut-file", str(cut_file)]
    task_id = _spawn_task(
        "recut", f"Re-cut {req.filename[:32]} ({len(spans)} cut(s))", args)
    return {"task_id": task_id, "status": "started", "cuts": len(spans),
            "removed_s": round(sum(b - a for a, b in spans), 2)}


class VoiceoverRequest(BaseModel):
    filename: str
    script: str
    rejected: bool = False
    voice: str | None = None
    duck_db: float = -12.0
    gain_db: float = 0.0


#: Generous. Kokoro's real limit is tokens against its style pack, not
#: characters, and it refuses over-long text with the exact count — so this
#: only stops a runaway paste from occupying a task slot, and the precise
#: refusal is left to the engine that actually knows it.
_MAX_SCRIPT_CHARS = 4000


@app.post("/api/voiceover")
def start_voiceover(req: VoiceoverRequest) -> dict[str, Any]:
    """Speak a script over a clip, ducking the clip's own audio under it."""
    script = (req.script or "").strip()
    if not script:
        raise HTTPException(400, "script is empty")
    if len(script) > _MAX_SCRIPT_CHARS:
        raise HTTPException(
            400, f"script is {len(script)} characters; the limit is "
                 f"{_MAX_SCRIPT_CHARS}")

    ws = _workspace()
    _confined_clip(ws, req.filename, rejected=req.rejected)

    # The script travels in a FILE, never as an argv element: it is operator
    # prose that will contain quotes, newlines and non-ASCII, and this is a
    # Windows host where argv quoting is the process's own business. `bta
    # voiceover --script-file` exists for exactly this.
    tmp = Path(ws.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    script_file = tmp / f"vo_{uuid.uuid4().hex[:12]}.txt"
    script_file.write_text(script, encoding="utf-8")

    args = ["voiceover", "--clip", _cli_value(req.filename, "filename"),
            "--script-file", str(script_file),
            "--duck-db", str(req.duck_db), "--gain-db", str(req.gain_db)]
    if req.voice:
        args.extend(["--voice", _cli_value(req.voice, "voice")])

    task_id = _spawn_task("voiceover",
                          f"Voiceover on {req.filename[:36]}", args)
    return {"task_id": task_id, "status": "started"}


class UpscaleRequest(BaseModel):
    filename: str
    rejected: bool = False
    height: int = 2560


@app.post("/api/upscale")
def start_upscale(req: UpscaleRequest) -> dict[str, Any]:
    """Resample a clip up. Refuses a target that is not actually larger."""
    from clipforge.enhance import UPSCALE_TARGETS

    allowed = {h for _, h in UPSCALE_TARGETS.values()}
    if req.height not in allowed:
        raise HTTPException(
            400, f"height must be one of {sorted(allowed)}; got {req.height}")

    ws = _workspace()
    _confined_clip(ws, req.filename, rejected=req.rejected)

    task_id = _spawn_task(
        "upscale", f"Upscale {req.filename[:36]} to {req.height}p",
        ["upscale", "--clip", _cli_value(req.filename, "filename"),
         "--height", str(req.height)])
    return {"task_id": task_id, "status": "started"}


@app.post("/api/doctor")
def run_doctor() -> dict[str, Any]:
    """Run the prerequisite check and stream its output to Activity."""
    task_id = _spawn_task("doctor", "Check prerequisites", ["doctor"])
    return {"task_id": task_id, "status": "started"}


@app.post("/api/watch/start")
def start_watch() -> dict[str, Any]:
    """Start watching channels."""
    # Check if a watch task is already running
    with _tasks_lock:
        for t in _tasks.values():
            if t.kind == "watch" and t.status == "running":
                return {"task_id": t.task_id, "status": "already_running"}

    task_id = _spawn_task("watch", "Watch channels", ["watch"])
    return {"task_id": task_id, "status": "started"}


@app.post("/api/watch/stop")
def stop_watch() -> dict[str, Any]:
    """Stop the watch task."""
    with _tasks_lock:
        for t in _tasks.values():
            if t.kind == "watch" and t.status == "running" and t.process:
                t.process.terminate()
                t.status = "stopped"
                return {"task_id": t.task_id, "status": "stopped"}
    return {"status": "no_watch_running"}


@app.post("/api/genquota/reset")
def reset_quota() -> dict[str, Any]:
    """Reset the generation quota ledger."""
    ws = _workspace()
    ledger_path = Path(ws.root) / "genvideo_quota.json"
    try:
        if ledger_path.exists():
            blob = _json.loads(ledger_path.read_text(encoding="utf-8"))
            for name, st in (blob.get("providers") or {}).items():
                st["exhausted_until"] = 0.0
                st["error_streak"] = 0
            ledger_path.write_text(
                _json.dumps(blob, indent=2), encoding="utf-8")
        return {"status": "reset", "note": "all provider quotas cleared"}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, f"quota reset failed: {exc}") from exc


@app.get("/api/tasks/{task_id}/logs")
def get_task_logs(task_id: str) -> dict[str, Any]:
    """Get the full buffered output logs for a task."""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(404, "task not found")
        return {
            "task_id": task_id,
            "status": task.status,
            "lines": task.output_lines
        }


@app.post("/api/tasks/{task_id}/cancel")
def cancel_task(task_id: str) -> dict[str, Any]:
    """Stop a running task and everything it started.

    The previous version called ``terminate()`` on the tracked process and
    declared victory. Two measured problems with that: on Windows the
    child's own children (ffmpeg, yt-dlp, the torch worker) survive it and
    keep running — so the render carried on and the GPU stayed busy — and
    the drain thread then overwrote ``cancelled`` with ``failed · exit 1``,
    because TerminateProcess reports 1. Both are fixed here: the whole tree
    goes, and the intent is recorded before the kill so the exit code
    cannot be misread as a crash.
    """
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(404, "task not found")
        if task.status != "running":
            return {"task_id": task_id, "status": task.status,
                    "note": "not running"}
        task.cancelled = True
        proc, guard = task.process, task.guard

    # Outside the lock: killing a tree takes seconds, and holding the lock
    # would stall every dashboard poll for the duration.
    how = kill_process_tree(proc, guard)
    with _tasks_lock:
        task.status = "cancelled"
        if task.finished_at is None:
            task.finished_at = time.time()
    log.info("web.task_cancelled", task_id=task_id, method=how)
    return {"task_id": task_id, "status": "cancelled", "stopped_via": how}


@app.post("/api/tasks/clear")
def clear_finished_tasks() -> dict[str, Any]:
    """Drop finished tasks from the Activity list.

    Only finished ones: a running task is not a list entry to tidy away,
    and removing it here would orphan the process it tracks — invisible
    and uncancellable, the exact failure the task registry exists to
    prevent.
    """
    with _tasks_lock:
        stale = [tid for tid, t in _tasks.items() if t.status != "running"]
        for tid in stale:
            _tasks.pop(tid, None)
        left = len(_tasks)
    return {"cleared": len(stale), "running": left}


# ============================================== PAIRING A SECOND DEVICE

@app.get("/api/access")
def access_info(request: Request) -> dict[str, Any]:
    """Where this server can be reached, and how exposed it currently is.

    Only an already-authorized caller sees this — it hands back the token,
    which is the entire point: the desktop dashboard renders a link that a
    phone can open, instead of the operator hunting for a file.
    """
    port = request.url.port or int(os.environ.get("BTA_WEB_PORT") or 8765)
    token = _policy.token
    urls = remote.access_urls(port, token)
    public = (os.environ.get(remote.ENV_PUBLIC_URL) or "").strip()
    if public:
        urls["public"] = [f"{public.rstrip('/')}/?{remote.TOKEN_QUERY}={token}"
                          if token else public]
    return {
        "auth_enforced": _policy.enforced,
        "trust_loopback": _policy.trust_loopback,
        "token": token,
        "urls": urls,
        "port": port,
        "pairing_active": _pairing.active(),
        # Said plainly rather than implied, because "it works from my
        # phone" and "it is on the internet" are very different states.
        "exposure": ("public tunnel" if public else
                     "tailnet" if urls.get("tailscale") else
                     "local network" if urls.get("lan") else
                     "this machine only"),
    }


@app.post("/api/pair")
def issue_pairing_code() -> dict[str, Any]:
    """Mint a short code for a second device.

    Callable only by something already trusted (loopback or a paired
    device), because the middleware ran first. That is what makes six
    digits acceptable: an attacker cannot ask for a code, only guess at
    one that an operator deliberately created moments ago.
    """
    if not _policy.enforced:
        return {"pairing": False,
                "note": "auth is disabled on this server — any device on the "
                        "network can already control it, so there is nothing "
                        "to pair"}
    code, ttl = _pairing.issue()
    return {"pairing": True, "code": code, "expires_in_s": ttl,
            "attempts": _pairing.max_attempts}


class RedeemRequest(BaseModel):
    code: str
    next: str = "/"


@app.post("/login/redeem")
def redeem_pairing_code(req: RedeemRequest) -> JSONResponse:
    """Trade a pairing code for the access cookie.

    Deliberately open (the middleware skips /login*): a device with no
    credential is exactly who needs this. The code's own limits — one
    use, minutes of life, five wrong guesses — are the protection.
    """
    if not _policy.enforced:
        return JSONResponse({"status": "ok", "note": "auth disabled"})
    decision = remote.decide(
        remote.Presented(client_host=None, pairing_code=req.code),
        _policy, pairing=_pairing)
    if not decision.allowed:
        return JSONResponse(status_code=401, content={"error": decision.reason})
    resp = JSONResponse({"status": "paired"})
    _set_access_cookie(resp, decision.cookie_value or "")
    return resp


_LOGIN_PAGE = """<!doctype html><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>BTA — pair this device</title>
<style>
:root{color-scheme:dark}
*{box-sizing:border-box}
body{margin:0;min-height:100dvh;display:grid;place-items:center;background:#0a0a0b;
  color:#f2f2f5;font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",system-ui,sans-serif;
  padding:24px calc(24px + env(safe-area-inset-left)) calc(24px + env(safe-area-inset-bottom))}
.card{width:100%;max-width:380px;background:#151517;border:1px solid #26262b;
  border-radius:16px;padding:24px}
h1{font-size:19px;margin:0 0 6px;letter-spacing:-.2px}
p{color:#9a9aa3;font-size:13.5px;margin:0 0 20px}
label{display:block;font-size:10.5px;letter-spacing:.1em;text-transform:uppercase;
  color:#6c6c76;font-weight:650;margin-bottom:8px}
input{width:100%;background:#1c1c1f;border:1px solid #33333a;color:#f2f2f5;
  border-radius:10px;padding:14px;font:inherit;font-size:24px;letter-spacing:.28em;
  text-align:center;outline:none;min-height:56px}
input:focus{border-color:#b4f22e}
button{width:100%;margin-top:14px;background:#fff;color:#0a0a0b;border:0;
  border-radius:10px;padding:15px;font:inherit;font-weight:650;font-size:15px;
  cursor:pointer;min-height:52px}
button:disabled{opacity:.5}
.err{color:#ff6b6b;font-size:13px;margin-top:12px;min-height:19px}
.hint{color:#6c6c76;font-size:12px;margin-top:18px;line-height:1.6}
code{background:#1c1c1f;padding:1px 5px;border-radius:4px;font-size:11.5px}
</style>
<div class="card">
  <h1>Pair this device</h1>
  <p>On the machine running BTA, open <b>Connect a device</b> in the sidebar
     and type the six digits it shows.</p>
  <label for="c">Pairing code</label>
  <input id="c" inputmode="numeric" autocomplete="one-time-code" maxlength="7"
         pattern="[0-9]*" placeholder="000000" autofocus>
  <button id="go">Pair</button>
  <div class="err" id="err"></div>
  <div class="hint">No code? Open the link with the token in it, or run
     <code>bta web --lan</code> again to print one.</div>
</div>
<script>
const q=new URLSearchParams(location.search);
// Same rule the server applies to ?next=: one leading slash and no
// scheme. Without it, location.replace() below happily runs a
// `javascript:` URI as script in this origin, and an absolute URL turns
// the pairing page into an open redirect.
const rawNext=q.get('next')||'/';
// fromCharCode(92) rather than a literal backslash: this page is a
// non-raw Python string, so every backslash here is read twice and
// an escape written the obvious way arrives in the browser broken.
const BS=String.fromCharCode(92);
const next=(rawNext.charAt(0)==='/' && rawNext.charAt(1)!=='/'
            && rawNext.indexOf(BS)<0) ? rawNext : '/';
const inp=document.getElementById('c'), btn=document.getElementById('go'),
      err=document.getElementById('err');
inp.addEventListener('input',()=>{
  inp.value=inp.value.replace(/\\D/g,'').slice(0,6);
  if(inp.value.length===6) submit();
});
inp.addEventListener('keydown',e=>{if(e.key==='Enter')submit()});
btn.onclick=submit;
async function submit(){
  const code=inp.value.replace(/\\D/g,'');
  if(code.length!==6){err.textContent='Six digits.';return}
  btn.disabled=true; err.textContent='';
  try{
    const r=await fetch('/login/redeem',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({code,next})});
    if(!r.ok){const d=await r.json().catch(()=>({}));
      throw new Error(d.error||('HTTP '+r.status))}
    location.replace(next);
  }catch(e){ err.textContent=e.message; btn.disabled=false; inp.select() }
}
</script>
"""


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, next: str = "/") -> Response:
    """The pairing page, or a straight-through if the URL carries a token."""
    if not _policy.enforced:
        return RedirectResponse(_safe_next(next), status_code=303)
    supplied = request.query_params.get(remote.TOKEN_QUERY)
    if supplied and _policy.token and secrets.compare_digest(
            supplied.strip(), _policy.token):
        resp = RedirectResponse(_safe_next(next), status_code=303)
        _set_access_cookie(resp, _policy.token)
        return resp
    return HTMLResponse(_LOGIN_PAGE)


# ========================================== DASHBOARD + ERROR HANDLER

@app.get("/", response_class=HTMLResponse)
def serve_landing() -> HTMLResponse:
    """Serve the landing page — the front door before the dashboard."""
    landing_path = Path(__file__).parent / "landing.html"
    if not landing_path.is_file():
        # No landing page on disk: fall through to the dashboard so the
        # tool stays usable rather than 404ing at its own root.
        return serve_dashboard()
    return HTMLResponse(landing_path.read_text(encoding="utf-8"))


@app.get("/dashboard", response_class=HTMLResponse)
def serve_dashboard() -> HTMLResponse:
    """Serve the live interactive dashboard."""
    dashboard_path = Path(__file__).parent / "dashboard_live.html"
    if not dashboard_path.is_file():
        # Fallback: serve the static workspace dashboard
        ws = _workspace()
        static = Path(ws.root) / "dashboard.html"
        if static.is_file():
            return HTMLResponse(static.read_text(encoding="utf-8"))
        return HTMLResponse("<h1>BTA — no dashboard found</h1>")
    return HTMLResponse(dashboard_path.read_text(encoding="utf-8"))


@app.exception_handler(HTTPException)
def _http_error(_request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code,
                        content={"error": exc.detail})

