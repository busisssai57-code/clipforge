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
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

from clipforge.config import load_config, load_watchlist
from clipforge.log import get_logger
from clipforge.paths import Workspace
from clipforge.state import StateDB

log = get_logger(__name__)

app = FastAPI(title="BTA Control API",
              description="Local-only control surface. Nothing leaves this "
                          "machine.")

# CORS allows the dashboard on the same origin plus common dev ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4321", "http://127.0.0.1:4321",
                   "http://localhost:8770", "http://127.0.0.1:8770",
                   "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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

    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def to_dict(self) -> dict[str, Any]:
        tail = self.output_lines[-30:]
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
        }


_tasks: dict[str, BackgroundTask] = {}
_tasks_lock = threading.Lock()


def _bta_cmd() -> list[str]:
    """The bta CLI invocation for subprocesses."""
    return [sys.executable, "-m", "clipforge.cli"]


def _spawn_task(kind: str, description: str, args: list[str]) -> str:
    """Spawn a CLI command as a background subprocess and track it."""
    task_id = f"task-{uuid.uuid4().hex[:8]}"
    cmd = _bta_cmd() + args
    log.info("web.spawn_task", task_id=task_id, kind=kind, cmd=" ".join(cmd))

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=str(Path(".")),
        # Prevent the child from inheriting the server's signal handlers
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        if sys.platform == "win32" else 0,
    )

    task = BackgroundTask(
        task_id=task_id,
        kind=kind,
        description=description,
        started_at=time.time(),
        process=proc,
    )
    with _tasks_lock:
        _tasks[task_id] = task

    def _drain():
        assert proc.stdout is not None
        for line in proc.stdout:
            stripped = line.rstrip("\n\r")
            task.output_lines.append(stripped)
            # Cap stored output at 500 lines
            if len(task.output_lines) > 500:
                task.output_lines = task.output_lines[-300:]
        proc.wait()
        task.return_code = proc.returncode
        task.status = "completed" if proc.returncode == 0 else "failed"
        log.info("web.task_finished", task_id=task_id, code=proc.returncode)

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

    ws = _workspace()
    meta = clipmeta.resolve_clip(ws, filename, rejected=rejected)
    if meta is None:
        raise HTTPException(404, "clip not found")
    return {
        "clip": meta.as_dict(),
        "transcript": clipmeta.transcript_for(ws, filename,
                                              rejected=rejected),
        "campath": clipmeta.campath_for(ws, filename, rejected=rejected),
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
    for seq in sorted(gen_dir.glob("*/sequence.mp4"),
                      key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
        shots = sorted(seq.parent.glob("shot_*.mp4"))
        try:
            size_mb = round(seq.stat().st_size / (1024 * 1024), 2)
        except OSError:
            continue
        out.append({
            "slug": seq.parent.name,
            "shots": len(shots),
            "size_mb": size_mb,
            "created_at": seq.stat().st_mtime,
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
    """All tracked background tasks."""
    with _tasks_lock:
        return [t.to_dict() for t in _tasks.values()]


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

    def resolved_preset(self) -> str:
        return (self.niche or self.preset or "documentary").strip()

    def resolved_clip(self) -> bool:
        return bool(self.clip_it or self.clip)


class StoryboardRequest(BaseModel):
    brief: str
    preset: str = "documentary"
    shots: int = 6


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

    shots = build_storyboard(req.brief, preset, req.shots)
    return {"brief": req.brief, "preset": req.preset, "shots": shots}


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
    if req.aspect_ratio not in ("9:16", "16:9"):
        raise HTTPException(400, "aspect_ratio must be 9:16 or 16:9")

    clip_it = req.resolved_clip()
    args = [
        "generate", req.brief,
        "--preset", preset,
        "--shots", str(req.shots),
        "--aspect", req.aspect_ratio,
        "--clip" if clip_it else "--no-clip",
    ]
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

    args = ["process", req.source, "--clips", str(req.clips)]
    # Only pass the flag when the operator actually chose one; otherwise
    # the CLI applies the configured default, which is the point of the
    # "Niche default" option in the dashboard.
    if req.jumpcut is not None:
        args.append("--jumpcut" if req.jumpcut else "--no-jumpcut")
    if req.enhance is not None:
        args.extend(["--enhance", req.enhance])
    if req.niche:
        args.extend(["--niche", req.niche])
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
                          ["grab", url, "--clips", str(req.clips)])
    return {"task_id": task_id, "status": "started"}


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

    args = ["process", meta.source_path, "--clips", str(req.clips)]
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
        args.extend(["--title", meta.title])
    if meta.caption:
        args.extend(["--caption", meta.caption])

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

    args = ["dub", "--clip", req.filename, "--lang", req.lang]
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

    if not req.cut_words and not req.cut_gaps:
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

    spans: list[tuple[float, float]] = []
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

    tmp = Path(ws.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    cut_file = tmp / f"recut_{uuid.uuid4().hex[:12]}.json"
    cut_file.write_text(_json.dumps([[a, b] for a, b in sorted(spans)]),
                        encoding="utf-8")

    args = ["process", meta.source_path, "--clips", "1",
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

    args = ["voiceover", "--clip", req.filename,
            "--script-file", str(script_file),
            "--duck-db", str(req.duck_db), "--gain-db", str(req.gain_db)]
    if req.voice:
        args.extend(["--voice", req.voice])

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
        ["upscale", "--clip", req.filename, "--height", str(req.height)])
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
    """Cancel a running task."""
    with _tasks_lock:
        task = _tasks.get(task_id)
        if not task:
            raise HTTPException(404, "task not found")
        if task.status != "running":
            return {"task_id": task_id, "status": task.status,
                    "note": "not running"}
        if task.process:
            task.process.terminate()
        task.status = "cancelled"
        return {"task_id": task_id, "status": "cancelled"}


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

