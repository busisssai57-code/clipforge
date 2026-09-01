"""Self-correcting repair: turn a QA rejection into a bounded, honest retry.

S7 rejects a clip and quarantines it. That is correct but it is not
finished: most rejections have a mechanical cause the pipeline can fix
itself — a window that landed 2 s long, a loudness target the source could
not reach in one pass, a splice whose arithmetic disagreed with the file.
This module decides WHAT to change and records WHY, so a second attempt is
a reasoned edit rather than a re-roll.

Three rules hold it honest:

1. **Only mechanical faults are repairable.** A geometry mismatch or a
   sha256 failure means the code is wrong, not the parameters. Retrying
   those burns minutes of GPU time and, worse, can eventually "pass" by
   luck and bury the bug. Those return ``repairable=False`` with the
   reason named, and the clip stays rejected.
2. **Bounded attempts.** ``MAX_ATTEMPTS`` re-renders, then it stops. A
   repair loop with no ceiling is an infinite loop with extra steps.
3. **Deterministic.** ``plan_repair`` is a pure function of the checks,
   the window and the attempt number. The same rejection always produces
   the same plan, and because the plan changes S6's params it changes the
   cache key — so a repaired clip is a NEW artifact, never an in-place
   overwrite of one the ledger already describes.

The repair itself never touches the picture. It moves window edges and
audio targets; it cannot invent content, and it does not try to.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from clipforge.stages.s7_qa import (LOUDNESS_FAIL_LU, MAX_CLIP_S, MIN_CLIP_S,
                                    SHIP_TP_CEILING)

#: Re-renders allowed per clip. Two: one to fix the obvious fault, one to
#: fix what the first repair disturbed. A third has never helped in
#: practice and costs a full render.
MAX_ATTEMPTS = 2

#: Fallback for callers that do not say which TP target the failed render
#: used. Kept in step with ``s6_render.LOUDNESS_TP``; imported lazily-by-value
#: rather than from s6 to keep this module free of the render stack.
LOUDNESS_TP_DEFAULT = -2.0

#: Loudness correction is clamped: loudnorm cannot exceed the source's
#: headroom, and chasing a target past this just trades LUFS for clipping.
MAX_GAIN_CORRECTION_LU = 3.0

#: How far a window edge may be nudged in one attempt, and in total.
EDGE_NUDGE_S = 0.75
MAX_TOTAL_NUDGE_S = 2.5

#: Checks whose failure means the CODE is wrong. Never retried — a repair
#: loop that "fixes" these is laundering a bug into a green tick.
STRUCTURAL_CHECKS = frozenset({
    "file-exists", "sha256-integrity", "probe", "video-stream",
    "audio-stream", "video-codec", "pixel-format", "geometry",
    "audio-codec", "duration-matches-artifact", "faststart",
    "subtitle-file", "subtitle-integrity", "campath-bounds",
    "framing-recorded", "loudness-measurable",
})


@dataclass(frozen=True)
class Remedy:
    """One change, and the check that demanded it."""

    action: str          # stable kebab-case identifier
    driven_by: str       # QA check name
    detail: str          # human-readable, carries the measured number


@dataclass(frozen=True)
class RepairPlan:
    repairable: bool
    reason: str = ""
    remedies: tuple[Remedy, ...] = ()
    window_start: float = 0.0
    window_end: float = 0.0
    param_overrides: dict[str, float] = field(default_factory=dict)
    drop_jumpcuts: bool = False

    def summary(self) -> str:
        if not self.repairable:
            return f"not repairable: {self.reason}"
        return "; ".join(f"{r.action} ({r.driven_by})" for r in self.remedies)


def _number(text: str) -> float | None:
    """First signed decimal in a QA ``measured`` string, or None.

    The measured strings are formatted for humans ("-16.55 LUFS",
    "31.72s"), so this is deliberately tolerant: a caller that cannot
    parse falls back to a fixed conservative step rather than guessing.
    """
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else None


def plan_repair(
    failed: list,
    *,
    window_start: float,
    window_end: float,
    source_duration: float,
    attempt: int,
    target_i: float,
    target_tp: float = LOUDNESS_TP_DEFAULT,
    original_start: float | None = None,
    original_end: float | None = None,
) -> RepairPlan:
    """Decide how to re-render a rejected clip.

    ``failed`` is the list of QACheck objects with severity="fail" that did
    not pass. Returns a plan whose ``repairable`` flag is the caller's
    gate — a False plan must NOT be retried.
    """
    if attempt >= MAX_ATTEMPTS:
        return RepairPlan(False, f"attempt budget spent ({MAX_ATTEMPTS})",
                          window_start=window_start, window_end=window_end)
    if not failed:
        return RepairPlan(False, "nothing failed",
                          window_start=window_start, window_end=window_end)

    names = {c.name for c in failed}
    blocking = names & STRUCTURAL_CHECKS
    if blocking:
        return RepairPlan(
            False,
            "structural failure means the code is wrong, not the "
            f"parameters: {', '.join(sorted(blocking))}",
            window_start=window_start, window_end=window_end)

    by_name = {c.name: c for c in failed}
    remedies: list[Remedy] = []
    overrides: dict[str, float] = {}
    start, end = float(window_start), float(window_end)
    drop_jumpcuts = False

    o_start = original_start if original_start is not None else window_start
    o_end = original_end if original_end is not None else window_end

    # --- duration out of the spec band -------------------------------
    if "duration-bounds" in by_name:
        measured = _number(by_name["duration-bounds"].measured)
        if measured is None:
            return RepairPlan(False, "duration-bounds reported no number",
                              window_start=start, window_end=end)
        if measured <= 0:
            return RepairPlan(False, "duration-bounds reported no duration",
                              window_start=start, window_end=end)
        # Scale the WINDOW by the ratio between what we wanted and what the
        # file actually measured. Subtracting the overshoot directly would
        # be wrong whenever rendered duration != window length, which is
        # exactly what jump-cut pacing makes true.
        if measured > MAX_CLIP_S:
            aim = MAX_CLIP_S - 1.0
            # Trim from the END: the hook lives at the start, and the
            # ranking chose this window for what it opens with.
            end = start + (end - start) * (aim / measured)
            remedies.append(Remedy(
                "shrink-window", "duration-bounds",
                f"{measured:.2f}s > {MAX_CLIP_S}s, trimming the tail"))
        elif measured < MIN_CLIP_S:
            aim = MIN_CLIP_S + 1.0
            # Rendered seconds per window second — below 1.0 when pacing
            # cut silence out of this window.
            yield_ratio = measured / max(1e-6, end - start)
            need = aim / yield_ratio          # window length that would fit
            if need > source_duration:
                return RepairPlan(
                    False,
                    f"source has only {source_duration:.1f}s; cannot reach "
                    f"the {MIN_CLIP_S}s floor",
                    window_start=start, window_end=end)
            end = start + need
            if end > source_duration:         # slide back rather than clip
                end = source_duration
                start = max(0.0, end - need)
            remedies.append(Remedy(
                "grow-window", "duration-bounds",
                f"{measured:.2f}s < {MIN_CLIP_S}s, extending the tail"))

    # --- the splice disagreed with the rendered file -------------------
    if "splice-duration-integrity" in by_name:
        drop_jumpcuts = True
        remedies.append(Remedy(
            "drop-jumpcuts", "splice-duration-integrity",
            "re-rendering unspliced; pacing is editorial, correctness is not"))

    # --- audio -------------------------------------------------------
    if "true-peak-ceiling" in by_name:
        measured = _number(by_name["true-peak-ceiling"].measured)
        over = (measured - SHIP_TP_CEILING) if measured is not None else 1.0
        # Derived from the ceiling AND from what this render actually asked
        # for, then whichever is lower wins. Ceiling-only was a silent no-op:
        # a clip measuring -0.9 gives over=0.1, the max(0.5, ...) floor turns
        # that into -1.0-0.5 = -1.5 — which WAS the default target, so the
        # retry re-rendered with identical params, hit the stage cache, and
        # returned the same rejected file. A repair must always ask for
        # something strictly quieter than the attempt that just failed.
        from_ceiling = SHIP_TP_CEILING - max(0.5, min(2.0, over))
        overrides["loudness_tp"] = round(min(from_ceiling, target_tp - 0.5), 2)
        remedies.append(Remedy(
            "lower-peak-ceiling", "true-peak-ceiling",
            f"{measured if measured is not None else '?'} dBTP over "
            f"{SHIP_TP_CEILING}; asking for more headroom"))

    if "loudness-target" in by_name:
        measured = _number(by_name["loudness-target"].measured)
        if measured is None:
            correction = 0.0
        else:
            # Ask for as much as it fell short by, clamped. loudnorm cannot
            # beat the source's headroom, so this converges or it does not
            # — either way it stops after MAX_ATTEMPTS.
            correction = max(-MAX_GAIN_CORRECTION_LU,
                             min(MAX_GAIN_CORRECTION_LU, target_i - measured))
        if abs(correction) < 0.1:
            return RepairPlan(
                False,
                f"loudness is {measured} LUFS against a {target_i} target "
                "and no correction would move it; the source is "
                "headroom-limited",
                window_start=start, window_end=end)
        overrides["loudness_i"] = round(target_i + correction, 2)
        remedies.append(Remedy(
            "nudge-loudness-target", "loudness-target",
            f"{measured:.2f} LUFS vs {target_i}; "
            f"retargeting {overrides['loudness_i']:+.2f}"))

    # --- dead content at the edges ------------------------------------
    if "black-frames" in by_name or "silence" in by_name:
        driver = "black-frames" if "black-frames" in by_name else "silence"
        nudged_start = start + EDGE_NUDGE_S
        nudged_end = end + EDGE_NUDGE_S
        # Keep the duration intact while stepping off the dead region, and
        # refuse to wander far from what the ranking actually chose. The
        # source check is on the UNCLAMPED end: clamping it first made this
        # guard unreachable, so a window at the tail of the file shifted
        # into nothing instead of giving up.
        if (abs(nudged_start - o_start) > MAX_TOTAL_NUDGE_S
                or nudged_end > source_duration):
            return RepairPlan(
                False,
                f"{driver}: window cannot move further without leaving the "
                "moment the ranking selected",
                window_start=start, window_end=end)
        start, end = nudged_start, nudged_end
        remedies.append(Remedy(
            "shift-window", driver,
            f"stepping {EDGE_NUDGE_S}s past dead content"))

    # --- a transient mux disagreement: just render it again ------------
    if "av-duration-match" in by_name and not remedies:
        remedies.append(Remedy(
            "re-render", "av-duration-match",
            "stream durations disagreed; re-muxing unchanged"))

    if not remedies:
        return RepairPlan(
            False,
            "no remedy is defined for: " + ", ".join(sorted(names)),
            window_start=start, window_end=end)

    if end <= start:
        return RepairPlan(False, "repair produced an empty window",
                          window_start=start, window_end=end)

    return RepairPlan(True, "", tuple(remedies), round(start, 3),
                      round(end, 3), overrides, drop_jumpcuts)
