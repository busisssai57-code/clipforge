"""One moment, one clip — across the windows that overlap it.

`bta watch` hands the DAG overlapping windows on purpose: each carries the
trailing ``overlap_s`` (60 s by default) of its predecessor, so a highlight
that straddles a chunk boundary is whole in at least one of them (T1). The
cost is that a highlight in that shared minute is whole in BOTH, and each
window picks its own top candidate without knowing what the previous one
shipped. Two renders, two different file names — because the name is the
cache key of a different source window — and two near-identical clips on
the operator's phone.

So accepted clips record the span they cover, in absolute stream seconds,
against the broadcast they came from. The next window skips a candidate
that mostly repeats one already shipped and takes the next-ranked one
instead, which is the difference between "no duplicate" and "no clip".

Scoped, not global: two different streams, and the same stream on another
day, are different broadcasts. `bta process` on a local file sets no scope
at all and behaves exactly as it always has.
"""

from __future__ import annotations

import contextvars
from dataclasses import dataclass
from typing import Any

from clipforge.log import get_logger

log = get_logger(__name__)

#: Fraction of the SHORTER span that must be shared to count as the same
#: moment. Deliberately not IoU: edge snapping and a window boundary can
#: clip one copy short, and IoU reads a short-and-long pair as different
#: while a viewer sees the same thing twice.
DEFAULT_OVERLAP = 0.5


@dataclass(frozen=True)
class Scope:
    """Which broadcast is being clipped, and where its spans are kept."""

    key: str
    db: Any


_SCOPE: contextvars.ContextVar[Scope | None] = contextvars.ContextVar(
    "clipforge_dedup_scope", default=None)


def set_scope(scope: Scope | None):
    return _SCOPE.set(scope)


def reset_scope(token) -> None:
    _SCOPE.reset(token)


def overlap_fraction(a_start: float, a_end: float,
                     b_start: float, b_end: float) -> float:
    """Shared seconds as a fraction of the shorter span (0.0 if disjoint)."""
    shared = min(a_end, b_end) - max(a_start, b_start)
    if shared <= 0:
        return 0.0
    shortest = min(a_end - a_start, b_end - b_start)
    return shared / shortest if shortest > 0 else 0.0


def already_shipped(start_s: float, end_s: float, *,
                    threshold: float = DEFAULT_OVERLAP) -> str | None:
    """The clip that already covers this moment, or None.

    Returns the earlier clip's path so the caller can say WHICH clip it is
    repeating rather than just refusing.
    """
    scope = _SCOPE.get()
    if scope is None:
        return None
    try:
        rows = scope.db.shipped_spans(scope.key)
    except Exception as exc:  # noqa: BLE001 - dedup never sinks a clip
        log.warning("dedup.unavailable", error=f"{type(exc).__name__}: {exc}")
        return None
    for row in rows:
        if overlap_fraction(start_s, end_s,
                            float(row["abs_start_s"]),
                            float(row["abs_end_s"])) >= threshold:
            return str(row["clip_path"])
    return None


def record(start_s: float, end_s: float, clip_path: str) -> None:
    """Remember that this span shipped. Never raises."""
    scope = _SCOPE.get()
    if scope is None:
        return
    try:
        scope.db.record_shipped_span(scope.key, float(start_s), float(end_s),
                                     str(clip_path))
    except Exception as exc:  # noqa: BLE001
        log.warning("dedup.record_failed", error=f"{type(exc).__name__}: {exc}")
