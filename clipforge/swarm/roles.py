"""The roles that actually run the pipeline.

Each role is small and does one thing, and none of them import each other
— they post follow-up work to the board and the supervisor routes it. That
is what lets the Critic send a piece back for a re-render without knowing
the Generator exists.

The pipeline these wrap is the existing one. Nothing here reimplements
generation, clipping or QA; a role is a thin adapter that turns a task
payload into a call and the result into the next task. When a role starts
doing real work of its own, it belongs in the pipeline, not here.

Task graph:

    plan ──> generate ──> critique ──> package
              (GPU)         (CPU)       (CPU)
    plan ──> clip ────────> critique
              (GPU)

The Critic is the interesting one. It re-derives quality from the FILE
rather than trusting whatever produced it, because this project has twice
shipped output that was structurally perfect and visually empty. A role
that believes the previous role's success report is decoration.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from clipforge.log import get_logger
from clipforge.swarm.board import Task, TaskBoard

log = get_logger(__name__)

#: Frames below this spatial variance carry no picture. Measured: real
#: footage sits in the hundreds to thousands, a flat fill under 10.
BLANK_VARIANCE_FLOOR = 12.0

#: A piece the Critic rejects is re-planned at most this many times before
#: it is left failed. Distinct from the board's attempt ceiling, which
#: counts crashes; this counts QUALITY rejections, which are not errors.
MAX_QUALITY_RETRIES = 1


def _bta() -> list[str]:
    """The CLI entrypoint, as a subprocess argv prefix.

    A subprocess rather than an in-process call on purpose: the pipeline
    loads multi-gigabyte models and the VRAM Law wants that memory
    returned to the driver when the stage ends. Process exit is the only
    teardown that is guaranteed.
    """
    exe = Path(sys.executable).parent / "bta.exe"
    if exe.exists():
        return [str(exe)]
    return [sys.executable, "-m", "clipforge.cli"]


def _run(argv: list[str], *, timeout_s: float, board: TaskBoard | None = None,
         task_id: int | None = None) -> str:
    """Run a pipeline command, heartbeating so the lease does not lapse."""
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace")
    lines: list[str] = []
    deadline = time.monotonic() + timeout_s
    last_beat = time.monotonic()
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip())
        now = time.monotonic()
        if board is not None and task_id is not None and now - last_beat > 60:
            board.heartbeat(task_id)
            last_beat = now
        if now > deadline:
            proc.kill()
            raise TimeoutError(f"{argv[1] if len(argv) > 1 else argv[0]} "
                               f"exceeded {timeout_s:.0f}s")
    code = proc.wait()
    out = "\n".join(lines)
    if code != 0:
        raise RuntimeError(f"{' '.join(argv[1:3])} failed (exit {code}): "
                           f"{out[-600:]}")
    return out


# ------------------------------------------------------------- planner

@dataclass
class Planner:
    """Turns a goal into concrete work.

    Deliberately not a model call. A planner that asks an LLM what to make
    is unpredictable and unauditable; this one expands an explicit goal
    into an explicit task list, which an operator can read before it runs.
    """

    name: str = "planner"
    kinds: tuple[str, ...] = ("plan",)
    gpu: bool = False

    def run(self, task: Task) -> Sequence[tuple[str, dict[str, Any]]]:
        p = task.payload
        out: list[tuple[str, dict[str, Any]]] = []

        source = p.get("source")
        if source:
            out.append(("clip", {"source": source,
                                 "clips": int(p.get("clips", 3)),
                                 "niche": p.get("niche")}))

        briefs = p.get("briefs") or ([p["brief"]] if p.get("brief") else [])
        for brief in briefs:
            out.append(("generate", {
                "brief": brief,
                "niche": p.get("niche"),
                "shots": p.get("shots"),
                "aspect": p.get("aspect"),
                "model": p.get("model"),
                "quality_retries": 0,
            }))

        if not out:
            raise ValueError(
                "a plan task needs 'brief', 'briefs' or 'source'; "
                f"got keys {sorted(p)}")
        log.info("swarm.planned", tasks=len(out), goal=task.goal)
        return out


# ----------------------------------------------------------- generator

@dataclass
class Generator:
    """Renders one piece from a brief. Holds the GPU permit."""

    board: TaskBoard
    timeout_s: float = 5400.0
    name: str = "generator"
    kinds: tuple[str, ...] = ("generate",)
    gpu: bool = True

    def run(self, task: Task) -> Sequence[tuple[str, dict[str, Any]]]:
        p = task.payload
        brief = p.get("brief")
        if not brief:
            raise ValueError("generate task has no brief")
        with _manifest_path() as manifest:
            argv = _bta() + ["generate", brief, "--manifest", str(manifest)]
            if p.get("niche"):
                argv += ["--preset", str(p["niche"])]
            if p.get("shots"):
                argv += ["--shots", str(int(p["shots"]))]
            if p.get("aspect"):
                argv += ["--aspect", str(p["aspect"])]
            if p.get("model"):
                # The board has carried this key from `swarm plan --model`
                # through two payloads since the day it was added, and
                # this is where it stopped: the argv was built without it,
                # so every swarm piece used whatever the registry picked.
                argv += ["--model", str(p["model"])]
            out = _run(argv, timeout_s=self.timeout_s, board=self.board,
                       task_id=task.id)
            outputs = _read_manifest(manifest)

        if not outputs:
            raise RuntimeError(
                "generation reported success but its manifest listed no "
                f"output; tail: {out[-400:]}")
        piece = outputs[0]
        return [("critique", {"path": piece, "kind": "piece",
                              "brief": brief, "niche": p.get("niche"),
                              "shots": p.get("shots"),
                              "aspect": p.get("aspect"),
                              "model": p.get("model"),
                              "quality_retries": int(p.get("quality_retries", 0))})]


# ------------------------------------------------------------- clipper

@dataclass
class Clipper:
    """Runs the clip DAG over a source. Holds the GPU permit."""

    board: TaskBoard
    timeout_s: float = 10800.0
    name: str = "clipper"
    kinds: tuple[str, ...] = ("clip",)
    gpu: bool = True

    def run(self, task: Task) -> Sequence[tuple[str, dict[str, Any]]]:
        p = task.payload
        source = p.get("source")
        if not source:
            raise ValueError("clip task has no source")
        verb = "grab" if str(source).startswith(("http://", "https://")) \
            else "process"
        with _manifest_path() as manifest:
            argv = _bta() + [verb, str(source),
                             "--clips", str(int(p.get("clips", 3))),
                             "--manifest", str(manifest)]
            # The niche carries caption styling, pacing and grade — pass it
            # through or a dark_mindset clip renders in the house style.
            if p.get("niche") and verb == "process":
                argv += ["--niche", str(p["niche"])]
            out = _run(argv, timeout_s=self.timeout_s, board=self.board,
                       task_id=task.id)
            outputs = _read_manifest(manifest)
        if not outputs:
            log.warning("swarm.clip_produced_nothing", source=str(source),
                        tail=out[-300:])
        # S7 already gated these; the Critic re-derives anyway.
        return [("critique", {"path": path, "kind": "clip",
                              "niche": p.get("niche"), "quality_retries": 99})
                for path in outputs]


# -------------------------------------------------------------- critic

@dataclass
class Critic:
    """Judges the FILE, not the report that produced it.

    Twice now this project has shipped output that passed every structural
    check and contained no picture. So this re-measures: the file exists,
    it probes, it has duration, and its frames carry actual variance.

    A rejection is not an error — it posts a fresh generate task with the
    reason attached, up to a quality-retry ceiling, and then gives up
    honestly rather than looping.
    """

    name: str = "critic"
    kinds: tuple[str, ...] = ("critique",)
    gpu: bool = False

    def run(self, task: Task) -> Sequence[tuple[str, dict[str, Any]]]:
        p = task.payload
        path = Path(str(p.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"nothing to critique at {path}")

        verdict = inspect_video(path)
        log.info("swarm.critique", path=str(path), **verdict)

        if verdict["ok"]:
            return [("package", {"path": str(path), "kind": p.get("kind"),
                                 "brief": p.get("brief"),
                                 "niche": p.get("niche"),
                                 **{k: verdict[k] for k in
                                    ("duration_s", "width", "height",
                                     "variance")}})]

        tries = int(p.get("quality_retries", 0))
        if tries >= MAX_QUALITY_RETRIES or not p.get("brief"):
            raise RuntimeError(
                f"rejected and not retrying: {verdict['reason']} "
                f"({path.name})")
        log.warning("swarm.requeue_after_rejection", path=str(path),
                    reason=verdict["reason"], attempt=tries + 1)
        return [("generate", {**{k: p.get(k) for k in
                                 ("brief", "niche", "shots", "aspect", "model")},
                              "quality_retries": tries + 1,
                              "previous_rejection": verdict["reason"]})]


def inspect_video(path: Path) -> dict[str, Any]:
    """Measured verdict on one video file.

    Pure and importable so the same judgement can be used outside the
    swarm — and so it is testable without running a pipeline.
    """
    from clipforge.ffmpeg import probe

    try:
        info = probe(path)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": f"unprobeable: {exc}",
                "duration_s": 0.0, "width": 0, "height": 0, "variance": 0.0}

    duration = float(getattr(info, "duration_s", 0.0) or 0.0)
    width = int(getattr(info, "width", 0) or 0)
    height = int(getattr(info, "height", 0) or 0)
    base = {"duration_s": round(duration, 2), "width": width, "height": height}

    if duration < 0.5:
        return {"ok": False, "reason": f"duration {duration:.2f}s",
                "variance": 0.0, **base}

    variance = _frame_variance(path)
    if variance < BLANK_VARIANCE_FLOOR:
        return {"ok": False,
                "reason": (f"blank picture (variance {variance:.1f} < "
                           f"{BLANK_VARIANCE_FLOOR}) — the file is "
                           "well-formed and empty"),
                "variance": round(variance, 2), **base}
    return {"ok": True, "reason": "", "variance": round(variance, 2), **base}


def _frame_variance(path: Path) -> float:
    """Mean spatial variance over a few sampled frames."""
    import numpy as np

    from clipforge.ffmpeg import require_binary

    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
         "-i", str(path), "-vf", "fps=1,scale=96:96,format=gray",
         "-frames:v", "8", "-f", "rawvideo", "-"],
        capture_output=True, timeout=180)
    raw = proc.stdout or b""
    if len(raw) < 96 * 96:
        return 0.0
    frames = np.frombuffer(raw[:len(raw) // (96 * 96) * 96 * 96],
                           dtype=np.uint8).reshape(-1, 96 * 96)
    return float(frames.astype(np.float32).var(axis=1).mean())


# ------------------------------------------------------------ packager

@dataclass
class Packager:
    """Writes the shipping metadata beside a finished piece.

    Draft-only, by law: this produces files and stops. Nothing here
    uploads, posts, or schedules — the Authorization Law is not relaxed
    because a swarm is doing the work instead of a person.

    It used to also apply the niche colour grade as a second ffmpeg pass.
    That was the wrong altitude twice over: it cost a full extra encode on
    a file S6 had already encoded, and because it ran after the Critic —
    and therefore after S7 — the artifact that actually shipped was never
    QA-measured. A grade that crushed blacks or clipped highlights passed
    every luma check, because those were scored on the pre-grade file. The
    grade now lives in S6's own filter chain, keyed by niche, exactly
    where `enhance_speech` already sits.
    """

    name: str = "packager"
    kinds: tuple[str, ...] = ("package",)
    gpu: bool = False

    def run(self, task: Task) -> Sequence[tuple[str, dict[str, Any]]]:
        import json

        p = task.payload
        path = Path(str(p.get("path", "")))
        if not path.is_file():
            raise FileNotFoundError(f"nothing to package at {path}")

        # The grade is NOT applied here any more. It runs inside S6's own
        # filter chain, keyed by niche — one encode instead of two, and
        # crucially the QA gate then measures the pixels that actually
        # ship. Grading after S7 meant a grade that crushed blacks or
        # clipped highlights sailed past every luma check.
        final = path

        meta = {
            "file": final.name,
            "source_file": path.name,
            "kind": p.get("kind"),
            "niche": p.get("niche"),
            "brief": p.get("brief"),
            "duration_s": p.get("duration_s"),
            "resolution": f"{p.get('width')}x{p.get('height')}",
            "picture_variance": p.get("variance"),
            "graded_in_render": bool(p.get("niche")),
            "status": "DRAFT — reviewed by the swarm, not published",
        }
        dest = final.with_suffix(".draft.json")
        dest.write_text(json.dumps(meta, indent=2, sort_keys=True),
                        encoding="utf-8")
        log.info("swarm.packaged", path=str(dest))
        return []


# ------------------------------------------------------------- factory

def build_swarm(board: TaskBoard, *, cpu_workers: int = 4):
    """Supervisor with every role registered, ready to serve."""
    from clipforge.swarm.supervisor import Supervisor

    sup = Supervisor(board=board, cpu_workers=cpu_workers)
    sup.register(Planner())
    sup.register(Generator(board=board))
    sup.register(Clipper(board=board))
    sup.register(Critic())
    sup.register(Packager())
    return sup


# --------------------------------------------------------- output scrape

@contextmanager
def _manifest_path():
    """A temp path for a pipeline command to write its result manifest to.

    Replaces regexing paths out of console output, which failed two ways
    in production: rich wraps long paths at 80 columns whenever stdout is
    a pipe, splitting a path across lines so no regex can match it; and a
    path containing a space never matched at all. Both look exactly like
    "the render produced nothing".
    """
    import tempfile

    fd, name = tempfile.mkstemp(suffix=".manifest.json")
    os.close(fd)
    path = Path(name)
    path.unlink(missing_ok=True)      # the command creates it
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def _read_manifest(path: Path) -> list[str]:
    """Existing output files a pipeline command reported. Never raises.

    A missing or malformed manifest yields no outputs, and the caller
    turns that into a clear failure — which is honest, because "the
    command did not tell us what it produced" and "it produced nothing"
    should not be silently different.
    """
    import json

    try:
        blob = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        log.warning("swarm.manifest_unreadable", path=str(path),
                    error=str(exc)[:200])
        return []
    outputs = blob.get("outputs") or []
    real = [str(p) for p in outputs if p and Path(p).is_file()]
    if len(real) != len(outputs):
        log.warning("swarm.manifest_listed_missing_files",
                    listed=len(outputs), present=len(real))
    return real
