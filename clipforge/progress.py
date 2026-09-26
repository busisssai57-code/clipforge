"""How far along a run is, and how much longer it has — measured, not guessed.

A spinning bar pinned at 55% is a lie with a nice animation. The pipeline
already emits everything needed to do better: every stage logs
``stage.done`` with its own name, and every completed stage has been
timed into ``stage_runs`` since the first run on this machine. So the
progress here is read from the run, and the estimate is read from what
the same stages actually took on the same hardware.

Two rules follow, and both are about not inventing numbers:

1. **No history, no ETA.** The first run of a stage has nothing to
   estimate from. It reports its stage and says the estimate is not
   available yet, rather than showing a plausible-looking countdown.
2. **The estimate is a median of real durations**, not an average — one
   pathological 40-minute render should not permanently poison every
   later prediction, and with a median it cannot.
"""

from __future__ import annotations

import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

#: The clip pipeline in execution order, with the label an operator reads.
#: `editor` (S3.5) is in the list because it runs and it costs time; naming
#: it "S3.5" in the UI would be honest about the architecture and useless
#: to the person watching, so the label says what it does.
CLIP_STAGES: tuple[tuple[str, str], ...] = (
    ("s1_transcribe", "Transcribing"),
    ("s2_prefilter", "Scoring windows"),
    ("s3_semantic", "Ranking with the VL model"),
    ("editor", "Choosing the cut"),
    ("s4_tracking", "Tracking the speaker"),
    ("s5_subtitles", "Building captions"),
    ("s6_render", "Rendering"),
    ("s7_qa", "Quality checks"),
)

#: Generation is not the clip pipeline: it runs a shot loop, not stages.
#: Its progress comes from counting shots, which the CLI prints.
_SHOT_RE = re.compile(r"\bshot\s+(\d+)\s*/\s*(\d+)", re.I)

#: `stage.done stage=s2_prefilter` — structlog's key=value rendering. The
#: value is bounded to word characters so a stray "stage=" inside a longer
#: message cannot be mistaken for a completion.
_STAGE_DONE_RE = re.compile(r"stage\.done\b.*?\bstage=([A-Za-z0-9_]+)")
_STAGE_HIT_RE = re.compile(r"stage\.cache_hit\b.*?\bstage=([A-Za-z0-9_]+)")
#: The console banners the CLI prints as it enters each stage. These are
#: what gives a live label BEFORE the stage finishes; without them the UI
#: would only ever name the stage that just ended.
_BANNER_RE = re.compile(r"\bS(\d)(?:\.\d)?:\s*(.+?)\s*$")

#: Downloading is not a stage and can dominate a run, so it is tracked as
#: its own phase rather than being folded into "before S1".
_DOWNLOAD_RE = re.compile(r"\[download\]\s+(\d{1,3}(?:\.\d)?)%")
_CANDIDATE_RE = re.compile(r"\bcandidate\s+(\d+)\s*/\s*(\d+)", re.I)


@dataclass
class RunProgress:
    """Live progress for one spawned task.

    Fed one output line at a time. Deliberately tolerant: an unrecognised
    line changes nothing, because the pipeline's logging is not a stable
    API and a parser that throws would take the whole task's output
    thread with it.
    """

    kind: str
    started_at: float
    #: Stage names seen finishing, in order, without duplicates.
    done: list[str] = field(default_factory=list)
    #: What the run says it is doing right now.
    label: str = "Starting"
    stage: str | None = None
    #: Generation only.
    shot: int = 0
    shots_total: int = 0
    download_pct: float | None = None
    #: Wall-clock at which each stage was observed finishing, so the
    #: current stage's elapsed time is knowable without a second clock.
    _marks: list[float] = field(default_factory=list)

    @property
    def stages(self) -> tuple[tuple[str, str], ...]:
        return CLIP_STAGES if self.kind != "generate" else ()

    def feed(self, line: str, *, now: float | None = None) -> None:
        now = time.time() if now is None else now
        try:
            self._feed(line, now)
        except Exception:  # noqa: BLE001 - progress must never break a run
            return

    def _feed(self, line: str, now: float) -> None:
        dl = _DOWNLOAD_RE.search(line)
        if dl:
            self.download_pct = float(dl.group(1))
            self.label = f"Downloading {self.download_pct:.0f}%"
            return

        shot = _SHOT_RE.search(line)
        if shot:
            self.shot, self.shots_total = int(shot.group(1)), int(shot.group(2))
            self.label = f"Generating shot {self.shot} of {self.shots_total}"
            return

        cand = _CANDIDATE_RE.search(line)
        if cand:
            c_idx, c_total = int(cand.group(1)), int(cand.group(2))
            self.label = f"Ranking candidate {c_idx} of {c_total} (multimodal VL)"
            return

        for pattern in (_STAGE_DONE_RE, _STAGE_HIT_RE):
            hit = pattern.search(line)
            if hit:
                name = hit.group(1)
                if name not in self.done and name in dict(CLIP_STAGES):
                    self.done.append(name)
                    self._marks.append(now)
                    self.download_pct = None
                return

        banner = _BANNER_RE.search(line)
        if banner and len(line) < 160:
            # The banner text is the CLI's own wording for the stage it is
            # entering — better than a label mapped from a stage id,
            # because it carries the detail ("multimodal VL", the filename).
            self.label = banner.group(2).rstrip(".")
            idx = int(banner.group(1)) - 1
            if 0 <= idx < len(CLIP_STAGES):
                self.stage = CLIP_STAGES[idx][0]
            self.download_pct = None

    # ------------------------------------------------------------ reporting

    def current_stage(self) -> str | None:
        """The stage believed to be running: the first one not yet done."""
        if self.kind == "generate":
            return None
        finished = set(self.done)
        for name, _label in CLIP_STAGES:
            if name not in finished:
                return name
        return None

    def fraction(self, estimator: "StageEstimator | None" = None) -> float | None:
        """Completed fraction in 0..1, or None when it cannot be known.

        Weighted by each stage's expected duration when history exists —
        an unweighted count would show 75% after S6 when the remaining QA
        pass takes seconds, and 12.5% after S1 when transcription was half
        the run.
        """
        if self.kind == "generate":
            if self.shots_total > 0:
                return min(1.0, max(0.0, (self.shot - 1) / self.shots_total))
            return None
        if not CLIP_STAGES:
            return None
        weights = {name: 1.0 for name, _ in CLIP_STAGES}
        if estimator is not None:
            measured = {name: estimator.median(name) for name, _ in CLIP_STAGES}
            if any(v for v in measured.values()):
                fallback = statistics.median(
                    [v for v in measured.values() if v] or [1.0])
                weights = {name: (measured[name] or fallback)
                           for name, _ in CLIP_STAGES}
        total = sum(weights.values()) or 1.0
        done = sum(weights[n] for n in self.done if n in weights)
        return min(1.0, done / total)

    def eta_s(self, estimator: "StageEstimator | None",
              *, now: float | None = None) -> float | None:
        """Seconds remaining, or None when there is nothing to base it on.

        The remaining stages contribute their median duration. The stage
        in flight contributes its median MINUS how long it has already
        been running, floored at zero — so a stage that overruns its
        median stops the estimate at "any moment now" instead of counting
        backwards past zero.
        """
        if estimator is None:
            return None
        now = time.time() if now is None else now

        if self.kind == "generate":
            if self.shots_total <= 0 or self.shot <= 1:
                return None
            elapsed = now - self.started_at
            per_shot = elapsed / max(1, self.shot - 1)
            return max(0.0, per_shot * (self.shots_total - self.shot + 1))

        remaining = 0.0
        known = False
        finished = set(self.done)
        current = self.current_stage()
        for name, _label in CLIP_STAGES:
            if name in finished:
                continue
            median = estimator.median(name)
            if median is None:
                continue
            known = True
            if name == current:
                since = now - (self._marks[-1] if self._marks else self.started_at)
                remaining += max(0.0, median - since)
            else:
                remaining += median
        return remaining if known else None

    def as_dict(self, estimator: "StageEstimator | None" = None,
                *, now: float | None = None) -> dict:
        frac = self.fraction(estimator)
        eta = self.eta_s(estimator, now=now)
        stage = self.current_stage()
        label = self.label
        if stage and label in ("Starting", ""):
            label = dict(CLIP_STAGES).get(stage, stage)
        return {
            "label": label,
            "stage": stage,
            "stages_done": list(self.done),
            "stages_total": len(CLIP_STAGES) if self.kind != "generate" else 0,
            "fraction": None if frac is None else round(frac, 4),
            "eta_s": None if eta is None else round(eta, 1),
            # Said out loud so the UI never has to infer it from a null:
            # "no estimate yet" and "about to finish" look identical
            # otherwise, and only one of them is worth showing a bar for.
            "eta_known": eta is not None,
            "shot": self.shot,
            "shots_total": self.shots_total,
            "download_pct": self.download_pct,
        }


class StageEstimator:
    """Median duration per stage, read from this machine's own history.

    Loaded once and cached for a short window: the numbers move only when
    a stage finishes, and re-querying SQLite for every dashboard poll (a
    poll per second, per task) is a lot of I/O for a number that changes
    every few minutes.
    """

    #: How long a loaded snapshot is reused before the DB is read again.
    TTL_S = 30.0
    #: Runs older than this are ignored — a median dragged from six months
    #: of a different GPU is not this machine's behaviour.
    MAX_SAMPLES = 40

    def __init__(self, state_db: Path, *, ttl_s: float | None = None) -> None:
        self.state_db = Path(state_db)
        self.ttl_s = self.TTL_S if ttl_s is None else float(ttl_s)
        self._medians: dict[str, float] = {}
        self._loaded_at = 0.0

    def median(self, stage: str) -> float | None:
        self._ensure()
        return self._medians.get(stage)

    def all_medians(self) -> dict[str, float]:
        self._ensure()
        return dict(self._medians)

    def _ensure(self) -> None:
        now = time.monotonic()
        if self._medians and now - self._loaded_at < self.ttl_s:
            return
        self._medians = self._load()
        self._loaded_at = now

    def _load(self) -> dict[str, float]:
        import sqlite3

        if not self.state_db.exists():
            return {}
        try:
            conn = sqlite3.connect(f"file:{self.state_db}?mode=ro", uri=True,
                                   timeout=2.0)
        except sqlite3.Error:
            return {}
        try:
            rows = conn.execute(
                "SELECT stage, finished_at - started_at AS d FROM stage_runs "
                "WHERE status='done' AND finished_at IS NOT NULL "
                "AND finished_at > started_at ORDER BY id DESC LIMIT ?",
                (self.MAX_SAMPLES * len(CLIP_STAGES),)).fetchall()
        except sqlite3.Error:
            return {}
        finally:
            conn.close()

        buckets: dict[str, list[float]] = {}
        for stage, duration in rows:
            if duration is None or duration <= 0:
                continue
            buckets.setdefault(str(stage), []).append(float(duration))
        return {stage: statistics.median(vals[:self.MAX_SAMPLES])
                for stage, vals in buckets.items() if vals}


def humanize(seconds: float | None) -> str:
    """A duration a person reads at a glance, or an honest blank."""
    if seconds is None:
        return ""
    seconds = max(0.0, float(seconds))
    if seconds < 45:
        return f"{int(seconds)}s"
    minutes = seconds / 60.0
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes/60:.1f} h"


