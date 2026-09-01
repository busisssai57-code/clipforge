"""One comparable number for what the pipeline actually shipped.

Quality work in this project has tended to mean adding a stage — tracking,
active-speaker, jump-cut pacing, a director camera. Each was real, and
none of them could be checked against the run before it, because nothing
ever wrote down how good the output WAS. The only number anyone could
quote was the test count, which says whether the code does what it was
told, never whether the clips are worth posting.

So this measures the clips on disk, and invents nothing to do it. Every
figure here was already computed by the stage that had the evidence:

* the four-dimension scorecard from S2's heuristics and S3's VL pass,
  via :func:`clipforge.clipmeta.resolve_clip`
* S7's QA verdict, pass and fail counts
* the loudness S6 measured back off the rendered file
* duration, frame size and the framing mode S4 chose

A holdout is only a holdout if the inputs stay fixed, so the source set
is recorded with the numbers and two runs over different sets are
reported as incomparable rather than diffed into a meaningless delta.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipforge.clipmeta import list_clips
from clipforge.paths import Workspace

#: Where the runs accumulate. One file per measurement, so a regression
#: can be bisected against the commits between two of them.
HOLDOUT_DIR = "holdout"

#: The delivery target the pipeline renders to. A clip outside this band
#: is not a matter of taste — S6 was asked for -14 LUFS and missed.
TARGET_LUFS = -14.0
LUFS_TOLERANCE = 1.0


@dataclass
class ClipRow:
    """What one clip contributes to the measurement."""

    filename: str
    source: str | None
    duration_s: float | None
    score: float | None
    grade: str | None
    dimensions: dict[str, float] = field(default_factory=dict)
    loudness_i: float | None = None
    qa_passed: bool | None = None
    qa_checks: int = 0
    framing_mode: str | None = None
    rejected: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def _row(meta: Any) -> ClipRow:
    card = meta.score or {}
    dims = {d.get("name", f"dim{i}"): float(d.get("score", 0.0))
            for i, d in enumerate(card.get("dimensions", []) or [])}
    qa = meta.qa or {}
    checks = qa.get("checks") or []
    return ClipRow(
        filename=meta.filename,
        source=Path(meta.source_path).name if meta.source_path else None,
        duration_s=meta.duration_s,
        # "overall", not "score": the first version of this read a key the
        # scorecard does not have, so every clip reported a grade and no
        # number — mean_score came back null over two graded clips, which
        # is exactly the silent-null this file exists to stop.
        score=card.get("overall"),
        grade=card.get("grade"),
        dimensions=dims,
        loudness_i=meta.loudness_i,
        qa_passed=qa.get("passed") if qa else None,
        qa_checks=len(checks),
        framing_mode=meta.framing_mode,
        rejected=meta.rejected,
    )


def measure(ws: Workspace, *, sources: list[str] | None = None) -> dict[str, Any]:
    """Score every clip in the workspace, or only those from ``sources``.

    ``sources`` are source FILENAMES, which is what a clip records. It is
    the holdout definition: leave it out and this measures whatever is on
    disk, which is a snapshot rather than a comparable run.
    """
    rows = [_row(m) for m in list_clips(ws)]
    if sources is not None:
        wanted = {Path(s).name for s in sources}
        rows = [r for r in rows if r.source in wanted]

    accepted = [r for r in rows if not r.rejected]
    scored = [r.score for r in accepted if r.score is not None]
    graded = [r.grade for r in accepted if r.grade]
    loud = [r.loudness_i for r in accepted if r.loudness_i is not None]
    in_band = [v for v in loud if abs(v - TARGET_LUFS) <= LUFS_TOLERANCE]
    judged = [r for r in accepted if r.qa_passed is not None]

    return {
        "generated_at": time.time(),
        "sources": sorted({r.source for r in rows if r.source}),
        "requested_sources": sorted(sources) if sources else None,
        "clips": len(accepted),
        "rejected": sum(1 for r in rows if r.rejected),
        # The headline. Mean of the scorecard the pipeline already
        # produced — not a new opinion about the clips.
        "mean_score": round(statistics.fmean(scored), 1) if scored else None,
        "median_score": round(statistics.median(scored), 1) if scored else None,
        "grades": {g: graded.count(g) for g in sorted(set(graded))},
        "qa_pass_rate": (round(sum(1 for r in judged if r.qa_passed)
                               / len(judged), 3) if judged else None),
        "loudness_in_band": (round(len(in_band) / len(loud), 3)
                             if loud else None),
        "mean_duration_s": (round(statistics.fmean(
            [r.duration_s for r in accepted if r.duration_s]), 1)
            if any(r.duration_s for r in accepted) else None),
        "framing_modes": {m: sum(1 for r in accepted if r.framing_mode == m)
                          for m in sorted({r.framing_mode for r in accepted
                                           if r.framing_mode})},
        "rows": [r.as_dict() for r in rows],
    }


def save(ws: Workspace, report: dict[str, Any]) -> Path:
    """Write one measurement, named for when it was taken."""
    out = Path(ws.root) / HOLDOUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(report["generated_at"]))
    path = out / f"{stamp}.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True),
                    encoding="utf-8")
    return path


def previous(ws: Workspace, *, before: Path | None = None) -> dict[str, Any] | None:
    """The measurement before this one, if there is one."""
    out = Path(ws.root) / HOLDOUT_DIR
    if not out.is_dir():
        return None
    files = sorted(f for f in out.glob("*.json")
                   if before is None or f != before)
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt old run is not this run's problem
        return None


def compare(now: dict[str, Any], then: dict[str, Any] | None) -> dict[str, Any]:
    """The deltas worth reading, or why there are none.

    Two runs over different source sets are NOT comparable, and saying so
    is the whole point of fixing the set: a mean that moved because the
    inputs changed is the kind of number that gets quoted for months.
    """
    if then is None:
        return {"comparable": False, "reason": "no earlier measurement"}
    if set(then.get("sources") or []) != set(now.get("sources") or []):
        return {"comparable": False,
                "reason": "the source set changed; these runs measure "
                          "different material",
                "then_sources": then.get("sources"),
                "now_sources": now.get("sources")}
    deltas: dict[str, Any] = {"comparable": True}
    for key in ("clips", "mean_score", "median_score", "qa_pass_rate",
                "loudness_in_band", "mean_duration_s"):
        a, b = then.get(key), now.get(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            deltas[key] = round(b - a, 3)
    return deltas
