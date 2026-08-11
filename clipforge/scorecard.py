"""Per-clip score breakdown, from numbers the pipeline already measured.

S2 scores every candidate window on six heuristic components and S3's
vision-language pass scores the survivors on three more. All nine were
computed, stored in artifacts, and shown nowhere — the dashboard displayed
a single blended number. This turns them into the dimension breakdown an
editor can actually act on: a clip that scores badly should say WHY.

**On the fourth dimension.** Commercial tools show a "Trend" score. We do
not have one and will not invent one: trend means "how this resembles what
is performing on a platform right now", which requires platform data this
machine deliberately never fetches. Claiming it would be fabricating a
measurement — the exact failure this project keeps finding in its own
code. The fourth dimension here is MOTION, which is genuinely measured
(visual action from the VL pass, laughter/reaction density from S2), and
it is named for what it is.

Every dimension records its own provenance. When the VL stage did not run,
a dimension that depends on it says so rather than quietly reporting a
heuristic number as if a model had judged it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

#: Letter bands. Deliberately coarse: the underlying signals are not
#: precise enough to justify distinguishing 87 from 88, and a fine-grained
#: grade would imply accuracy the inputs do not have.
_BANDS: tuple[tuple[float, str], ...] = (
    (90.0, "A+"), (85.0, "A"), (80.0, "A-"),
    (75.0, "B+"), (70.0, "B"), (65.0, "B-"),
    (55.0, "C"), (0.0, "D"),
)


def grade_for(score: float) -> str:
    for floor, letter in _BANDS:
        if score >= floor:
            return letter
    return "D"


@dataclass(frozen=True)
class Dimension:
    name: str
    score: float               # 0-100
    grade: str
    #: Which measurements produced this, so a number can be traced back.
    inputs: tuple[str, ...] = ()
    #: True when a VL judgement was available; False = heuristics only.
    model_judged: bool = False
    note: str = ""


@dataclass(frozen=True)
class ScoreCard:
    overall: float
    grade: str
    dimensions: tuple[Dimension, ...] = ()
    #: "vl" when the vision-language pass ran, "heuristic" when the
    #: ranking fell back to S2 ordering.
    source: str = "heuristic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "overall": round(self.overall, 1),
            "grade": self.grade,
            "source": self.source,
            "dimensions": [
                {"name": d.name, "score": round(d.score, 1), "grade": d.grade,
                 "inputs": list(d.inputs), "model_judged": d.model_judged,
                 "note": d.note}
                for d in self.dimensions
            ],
        }


def _pct(value: float | None, *, scale: float) -> float | None:
    """Normalise a raw score to 0-100, or None when it is absent."""
    if value is None:
        return None
    return max(0.0, min(100.0, (float(value) / scale) * 100.0))


def _blend(parts: list[tuple[float, float]]) -> float:
    """Weighted mean of (value, weight), ignoring absent parts."""
    live = [(v, w) for v, w in parts if v is not None and w > 0]
    if not live:
        return 0.0
    total_w = sum(w for _, w in live)
    return sum(v * w for v, w in live) / total_w


def build_scorecard(*, s2_scores: dict[str, float] | None = None,
                    visual_action: float | None = None,
                    hook_strength: float | None = None,
                    comprehensibility: float | None = None,
                    ranking_source: str = "heuristic") -> ScoreCard:
    """Combine what S2 and S3 measured into four actionable dimensions.

    ``s2_scores`` is the CandidateWindow.scores mapping (components in
    [0,1]); the three VL fields are in [0,10] and are None when the model
    did not run. Nothing here re-derives a signal — it only re-presents
    measurements that already exist.
    """
    s2 = {k: float(v) for k, v in (s2_scores or {}).items()
          if isinstance(v, (int, float))}

    def s2_pct(key: str) -> float | None:
        return _pct(s2.get(key), scale=1.0) if key in s2 else None

    vl_ran = ranking_source == "vl" or any(
        x is not None for x in (visual_action, hook_strength, comprehensibility))

    hook_vl = _pct(hook_strength, scale=10.0)
    hook = Dimension(
        name="Hook",
        score=_blend([(hook_vl, 3.0), (s2_pct("boundary"), 1.0)]),
        grade="",
        inputs=tuple(x for x in ("vl:hook_strength" if hook_vl is not None
                                 else None, "s2:boundary" if "boundary" in s2
                                 else None) if x),
        model_judged=hook_vl is not None,
        note=("how strongly the opening earns the next three seconds"
              if hook_vl is not None else
              "heuristic only: measures a clean sentence start, not "
              "whether the opening is compelling"),
    )

    flow = Dimension(
        name="Flow",
        score=_blend([(s2_pct("turns"), 1.0), (s2_pct("energy"), 1.0)]),
        grade="",
        inputs=tuple(x for x in ("s2:turns" if "turns" in s2 else None,
                                 "s2:energy" if "energy" in s2 else None) if x),
        model_judged=False,
        note="speaker-turn density and words per second against a plateau",
    )

    value_vl = _pct(comprehensibility, scale=10.0)
    value = Dimension(
        name="Value",
        score=_blend([(value_vl, 2.0), (s2_pct("selfcont"), 1.5)]),
        grade="",
        inputs=tuple(x for x in ("vl:comprehensibility" if value_vl is not None
                                 else None, "s2:selfcont" if "selfcont" in s2
                                 else None) if x),
        model_judged=value_vl is not None,
        note="whether the clip stands alone without the surrounding video",
    )

    motion_vl = _pct(visual_action, scale=10.0)
    motion = Dimension(
        name="Motion",
        score=_blend([(motion_vl, 2.0), (s2_pct("laughter"), 1.0)]),
        grade="",
        inputs=tuple(x for x in ("vl:visual_action" if motion_vl is not None
                                 else None, "s2:laughter" if "laughter" in s2
                                 else None) if x),
        model_judged=motion_vl is not None,
        note=("visible action and reaction density. NOT a trend score — "
              "trend needs platform data this machine never fetches"),
    )

    dims = tuple(
        Dimension(d.name, d.score, grade_for(d.score), d.inputs,
                  d.model_judged, d.note)
        for d in (hook, flow, value, motion))

    # Hook weighted hardest: short-form is won or lost in the first
    # seconds, and every other dimension is moot if nobody stays.
    overall = _blend([(dims[0].score, 3.0), (dims[1].score, 2.0),
                      (dims[2].score, 2.0), (dims[3].score, 1.0)])
    return ScoreCard(overall=overall, grade=grade_for(overall),
                     dimensions=dims,
                     source="vl" if vl_ran else "heuristic")
