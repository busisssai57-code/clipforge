"""Live chat as an engagement signal — the audience voting, per second.

Transcript ranking (S2) and the VL pass (S3) both look at the *content* of
a clip. Neither sees what the room did while it happened. On a livestream
the chat is exactly that record: a laugh, a play, a callout shows up as a
burst of messages a second or two later, and the transcript often has no
words for it at all. This module turns a chat log into a per-second
engagement curve that S2 scores its candidate windows against, so a
window the audience reacted to ranks above one it sat quiet through.

Three things make this honest rather than a message counter:

* **One person is not a crowd.** A single user spamming the same emote
  fifty times is fifty messages and one opinion. Each user's contribution
  within a one-second bin is capped, so a flood counts once, not fifty
  times. This is the difference between "the room reacted" and "one
  account is loud".
* **Gifts and superchats weigh more**, when the format exposes them,
  because paying to be seen is a stronger vote than typing. Never lower
  than a plain message, never allowed to swamp the crowd either.
* **Deterministic.** Times bin to integer seconds and every sum is over a
  sorted set, so the same log scores the same everywhere — the Law the
  ranking stages hold themselves to, held here too.

Nothing here reaches the network or a model. A missing or unreadable log
is not an error: the curve is simply empty and every window scores 0 for
chat, which is exactly today's behaviour.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from clipforge.log import get_logger

log = get_logger(__name__)

#: A single user's messages inside one one-second bin count at most this
#: many times. The whole anti-flood mechanism is this cap: sustained
#: spam from one account cannot manufacture a peak.
PER_USER_PER_BIN_CAP = 3

#: Weight a paid message (superchat, bits, gift) carries relative to a
#: plain one. High enough to matter, low enough that a single gift cannot
#: outweigh a genuine crowd reaction.
PAID_WEIGHT = 5.0

#: Per-second engagement at which a window scores half of the maximum.
#: The score saturates rather than dividing by the curve's own peak: a big
#: reaction must outscore a lone message on an ABSOLUTE scale, so twelve
#: people beat one person whether or not either is the busiest second in
#: the whole stream. Roughly "a lively chat" — tuned to be reachable, not
#: to require a viral moment.
SATURATION_MSGS_PER_S = 2.0

#: Words in a paid-message type/label across the formats we read.
_PAID_HINT = re.compile(
    r"paid|superchat|super_chat|supersticker|bits|cheer|gift|membership",
    re.IGNORECASE)


@dataclass(frozen=True)
class ChatEvent:
    """One message, placed on the media timeline.

    ``t_s`` is seconds from the START OF THE MEDIA, not wall-clock: that is
    what lines a message up with the moment it reacted to. ``weight`` is 1
    for a plain message and higher for a paid one.
    """

    t_s: float
    user: str
    weight: float = 1.0


# --------------------------------------------------------------- parsing

def _num(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _yt_offset_s(obj: dict[str, Any]) -> float | None:
    """YouTube replay chat carries ``videoOffsetTimeMsec`` (a string)."""
    raw = obj.get("videoOffsetTimeMsec")
    ms = _num(raw)
    return ms / 1000.0 if ms is not None else None


def _walk_json(obj: Any) -> Iterable[dict[str, Any]]:
    """Yield every dict in a nested JSON structure, parents before children.

    Chat exporters bury the renderer that holds the author and the paid
    marker at varying depths; rather than hard-code one exporter's shape,
    we find the offset at the top of a replay action and the author/paid
    hints anywhere beneath it.
    """
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _author_and_paid(scope: dict[str, Any]) -> tuple[str, bool]:
    author = ""
    paid = False
    for node in _walk_json(scope):
        for key, val in node.items():
            lk = key.lower()
            if not author and isinstance(val, str) and lk in (
                    "authorname", "author", "commenter", "name",
                    "displayname", "user", "username", "nickname"):
                author = val.strip()
            if not author and lk == "simpletext" and "name" in str(
                    node).lower():
                pass
            if isinstance(val, str) and _PAID_HINT.search(lk):
                paid = True
            if lk in ("_type", "type", "renderer") and isinstance(val, str) \
                    and _PAID_HINT.search(val):
                paid = True
    return author, paid


def _parse_jsonl(text: str) -> list[ChatEvent]:
    """One JSON object per line (YouTube live_chat.json and friends)."""
    out: list[ChatEvent] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        t = _yt_offset_s(obj)
        if t is None:
            for node in _walk_json(obj):
                t = _yt_offset_s(node)
                if t is not None:
                    break
        if t is None or t < 0:
            continue
        author, paid = _author_and_paid(obj)
        out.append(ChatEvent(t_s=t, user=author or "?",
                             weight=PAID_WEIGHT if paid else 1.0))
    return out


def _parse_json_array(text: str) -> list[ChatEvent]:
    """Twitch VOD chat exports: ``{"comments": [{content_offset_seconds,
    commenter:{display_name}}, ...]}`` (twitch-dl / TDT and similar)."""
    try:
        doc = json.loads(text)
    except ValueError:
        return []
    comments = doc.get("comments") if isinstance(doc, dict) else (
        doc if isinstance(doc, list) else None)
    if not isinstance(comments, list):
        return []
    out: list[ChatEvent] = []
    for c in comments:
        if not isinstance(c, dict):
            continue
        t = _num(c.get("content_offset_seconds"))
        if t is None:
            t = _yt_offset_s(c)
        if t is None or t < 0:
            continue
        author, paid = _author_and_paid(c)
        out.append(ChatEvent(t_s=t, user=author or "?",
                             weight=PAID_WEIGHT if paid else 1.0))
    return out


_IRC_LINE = re.compile(
    r"^\[?(?P<h>\d{1,2}):(?P<m>\d{2}):(?P<s>\d{2})(?:\.\d+)?\]?"
    r"\s+<?(?P<user>[^>:]+?)>?\s*[:>]\s*(?P<msg>.*)$")


def _parse_irc(text: str) -> list[ChatEvent]:
    """Plain logs: ``[00:12:34] user: message`` (the offset is media time)."""
    out: list[ChatEvent] = []
    for line in text.splitlines():
        m = _IRC_LINE.match(line.strip())
        if not m:
            continue
        t = int(m["h"]) * 3600 + int(m["m"]) * 60 + int(m["s"])
        out.append(ChatEvent(t_s=float(t), user=m["user"].strip() or "?"))
    return out


def _parse_csv(text: str) -> list[ChatEvent]:
    """``offset_seconds,user[,weight]`` or a header naming those columns."""
    rows = list(csv.reader(text.splitlines()))
    if not rows:
        return []
    header = [h.strip().lower() for h in rows[0]]
    has_header = any(h in ("offset_seconds", "offset", "time", "t", "seconds",
                           "user", "commenter", "author", "weight")
                     for h in header)
    ti, ui, wi = 0, 1, None
    if has_header:
        def _idx(*names: str) -> int | None:
            for n in names:
                if n in header:
                    return header.index(n)
            return None
        ti = _idx("offset_seconds", "offset", "seconds", "time", "t") or 0
        ui = _idx("user", "commenter", "author", "name")
        wi = _idx("weight", "amount")
        body = rows[1:]
    else:
        body = rows
    out: list[ChatEvent] = []
    for row in body:
        if len(row) <= ti:
            continue
        t = _num(row[ti])
        if t is None or t < 0:
            continue
        user = (row[ui].strip() if ui is not None and len(row) > ui else "?") or "?"
        w = 1.0
        if wi is not None and len(row) > wi:
            wv = _num(row[wi])
            if wv is not None and wv > 0:
                w = wv
        out.append(ChatEvent(t_s=t, user=user, weight=w))
    return out


def parse_chat_log(path: Path | str) -> list[ChatEvent]:
    """Read a chat log in whatever format it is, or return [] if unreadable.

    Format is chosen by content, not just extension: a ``.json`` may be an
    object with ``comments`` (Twitch) or newline-delimited replay actions
    (YouTube), and both are common. Order of attempts is widest-first, and
    the first parser that yields events wins. Never raises — a log that
    cannot be read is the same as no log, which the caller treats as "no
    chat signal", never as a failure.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        log.warning("chat.unreadable", path=str(p), error=str(exc)[:200])
        return []
    if not text.strip():
        return []

    stripped = text.lstrip()
    attempts = []
    if stripped[:1] in "{[":
        attempts = [_parse_json_array, _parse_jsonl]
    else:
        attempts = [_parse_jsonl]
    attempts += [_parse_irc, _parse_csv]

    for parse in attempts:
        events = parse(text)
        if events:
            events.sort(key=lambda e: (e.t_s, e.user))
            log.info("chat.parsed", path=str(p), events=len(events),
                     via=parse.__name__)
            return events
    log.warning("chat.no_events", path=str(p),
                note="recognised no chat format")
    return []


# ------------------------------------------------------------ engagement

@dataclass(frozen=True)
class ChatCurve:
    """Per-second engagement, anti-spam applied, ready for scoring.

    ``bins[k]`` is the (capped, weighted) engagement in the k-th second of
    media. Empty when there was no usable chat. ``peak`` is the largest
    single-second value, cached because every window score divides by it.
    """

    bins: dict[int, float]
    peak: float

    def __bool__(self) -> bool:
        return bool(self.bins) and self.peak > 0.0

    def total(self) -> float:
        return math.fsum(self.bins.values())

    def to_params(self) -> list[list[float]]:
        """A canonical ``[[second, value], ...]`` for a stage's params dict.

        This is how the audience signal enters S2's cache key: the CONTENT
        of the curve, never the path it came from, so the same chat scores
        the same clip identically no matter where the log file lives, and a
        different chat correctly invalidates the cache. Values round to 3
        decimals so the JSON digest is byte-identical across machines —
        the same reason S2 compares window bounds in integer milliseconds.
        Sorted by second, so serialization is stable.
        """
        return [[b, round(v, 3)] for b, v in sorted(self.bins.items())]

    @classmethod
    def from_params(cls, rows: Any) -> "ChatCurve":
        """Rebuild a curve from :meth:`to_params` output (or None → empty)."""
        bins: dict[int, float] = {}
        for row in rows or []:
            try:
                b, v = int(row[0]), float(row[1])
            except (TypeError, ValueError, IndexError):
                continue
            if math.isfinite(v) and v > 0.0:
                bins[b] = v
        return cls(bins=bins, peak=max(bins.values()) if bins else 0.0)


def build_curve(events: Iterable[ChatEvent], *,
                per_user_cap: int = PER_USER_PER_BIN_CAP) -> ChatCurve:
    """Collapse messages into a per-second engagement curve.

    The cap is the whole point: within each one-second bin a single user's
    messages count at most ``per_user_cap`` times, so a flood from one
    account cannot invent a peak. Paid messages carry their weight on top
    of the cap, because a superchat is a distinct, stronger signal — but
    the count of plain messages it rides with is still capped.
    """
    # bin -> user -> [plain_count, paid_weight_sum]
    grid: dict[int, dict[str, list[float]]] = {}
    for e in events:
        if e.t_s < 0 or not math.isfinite(e.t_s):
            continue
        b = int(e.t_s)  # floor to the second — deterministic, no float sum
        cell = grid.setdefault(b, {}).setdefault(e.user, [0.0, 0.0])
        if e.weight > 1.0:
            cell[1] += e.weight
        else:
            cell[0] += 1.0

    bins: dict[int, float] = {}
    for b in sorted(grid):
        total = 0.0
        for _user, (plain, paid) in sorted(grid[b].items()):
            total += min(plain, float(per_user_cap)) + paid
        if total > 0.0:
            bins[b] = total
    peak = max(bins.values()) if bins else 0.0
    return ChatCurve(bins=bins, peak=peak)


def score_window(start_s: float, end_s: float, curve: ChatCurve) -> float:
    """How hard the audience reacted across ``[start_s, end_s]``, in 0..1.

    The window's average per-second engagement, passed through a
    saturating curve so the score rises with ABSOLUTE reaction and is
    bounded in [0, 1). Average rather than sum, so a long window is not
    rewarded for its length — that is another heuristic's concern.

    Absolute, deliberately: an earlier version divided by the curve's own
    busiest second, which made a lone message and a twelve-person reaction
    score identically whenever each was all its curve contained. The vote
    that matters is how many people reacted, not how that compares to the
    curve's self-maximum, so a lively second counts the same in a quiet
    stream as in a loud one. A window overlapping no chat scores 0.
    """
    if not curve or end_s <= start_s:
        return 0.0
    lo = int(math.floor(start_s))
    hi = int(math.ceil(end_s))
    span = max(1, hi - lo)
    got = math.fsum(curve.bins.get(b, 0.0) for b in range(lo, hi))
    mean = got / span
    return 1.0 - 0.5 ** (mean / SATURATION_MSGS_PER_S)


def peaks(curve: ChatCurve, *, min_fraction: float = 0.5) -> list[tuple[int, float]]:
    """Seconds whose engagement is a strong fraction of the busiest second.

    Not consumed by S2's scoring path (which scores existing windows); this
    is the hook for later work that would seed candidates from reactions
    that have no transcript behind them. Returned sorted by time.
    """
    if not curve:
        return []
    floor = curve.peak * max(0.0, min(1.0, min_fraction))
    return [(b, v) for b, v in sorted(curve.bins.items()) if v >= floor]


#: Sidecar names a downloader leaves beside a source video, in preference
#: order. yt-dlp writes ``<name>.live_chat.json``; VOD chat tools vary.
_CHAT_SIDECARS = (".live_chat.json", ".chat.json", ".rechat.json",
                  ".chat.jsonl", ".chat.csv", ".chat.log")


def discover_beside(source: Path | str) -> Path | None:
    """A chat log sitting next to ``source``, or None. Never guesses across
    directories — only the exact stem plus a known chat suffix."""
    src = Path(source)
    stem = src.with_suffix("")
    for suffix in _CHAT_SIDECARS:
        cand = stem.with_name(stem.name + suffix)
        if cand.is_file():
            return cand
    return None


def curve_for(source: Path | str | None, explicit: Path | str | None) -> ChatCurve:
    """The engagement curve for a run: the explicit log if given, else one
    discovered beside the source, else empty. The single entry point the
    pipeline calls; empty curve == today's behaviour."""
    path = None
    if explicit:
        path = Path(explicit)
    elif source is not None:
        path = discover_beside(source)
    if path is None:
        return ChatCurve(bins={}, peak=0.0)
    return build_curve(parse_chat_log(path))
