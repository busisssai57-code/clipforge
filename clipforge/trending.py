"""Trend discovery — find a currently-trending YouTube video to clip.

This is the front half of ``bta trending``: given nothing but a search seed,
return the videos that are trending *this week*, ranked the way YouTube ranks
them, already filtered to what this pipeline can actually use.

Why a search URL and not the Trending feed: YouTube retired the global
``/feed/trending`` page and the ``/charts`` endpoints in 2025 — yt-dlp's tab
extractor now 404s or redirects to the home page for all of them (verified
2026-08). The durable signal that remains is a normal search with the
"this week + sort by view count" filter applied, which is exactly what the
``sp`` token below encodes. It returns real, freshly-uploaded, high-view
videos every time.

All yt-dlp interaction goes through the injectable ``run_tool`` seam (same as
the ingest modules), so selection logic is unit-tested offline; the live
network path is covered by ``bta trending`` itself.
"""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Callable

from clipforge.ingest.runner import run_tool
from clipforge.log import get_logger

log = get_logger(__name__)

RunTool = Callable[..., object]  # (name, args, **kw) -> CompletedProcess[str]

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")

#: yt-dlp emits one record per line in this order, tab-separated. A tab never
#: occurs inside a YouTube title (it is stripped on upload), so it is a safe
#: field separator even though titles contain every other punctuation mark.
_PRINT_TEMPLATE = "%(id)s\t%(duration)s\t%(view_count)s\t%(is_live)s\t%(title)s"

#: The ``sp`` filter token for "Upload date: this week" + "Sort by: view
#: count". It is a base64url-encoded protobuf; YouTube has kept it stable for
#: years because the filter UI serialises to exactly these bytes. Decoded it is
#: ``{1: {1: 3 (this week)}, 2: {1: 1 (view count)}}``.
SEARCH_SP = "CAMSBAgDEAE%3D"


@dataclass(frozen=True)
class Candidate:
    """One trending video, as much as a flat (no-download) listing knows."""

    video_id: str
    title: str
    duration_s: int | None
    view_count: int | None
    is_live: bool

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"

    @property
    def duration_hms(self) -> str:
        if self.duration_s is None:
            return "??:??"
        m, s = divmod(int(self.duration_s), 60)
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def build_search_url(query: str, *, region: str = "US") -> str:
    """The view-sorted, this-week search URL for ``query``.

    ``gl`` biases results toward a region's audience the same way opening
    YouTube in that country would; it does not hard-filter, so a globally
    viral video still surfaces regardless of region.
    """
    q = urllib.parse.quote_plus((query or "").strip() or "news")
    gl = urllib.parse.quote_plus((region or "US").strip().upper())
    return (f"https://www.youtube.com/results?search_query={q}"
            f"&sp={SEARCH_SP}&gl={gl}")


def _to_int(raw: str) -> int | None:
    raw = (raw or "").strip()
    if not raw or raw.upper() == "NA":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def parse_candidates(stdout: str) -> list[Candidate]:
    """Turn yt-dlp's ``--print`` block into typed candidates.

    Defensive by construction: yt-dlp occasionally lands a warning on stdout,
    and a search's first "entry" is sometimes a channel or a shelf header with
    an unusable id. Anything whose first field is not an 11-char video id is
    dropped rather than trusted.
    """
    out: list[Candidate] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        vid, dur, views, live, title = parts[0], parts[1], parts[2], parts[3], "\t".join(parts[4:])
        vid = vid.strip()
        if not _VIDEO_ID_RE.fullmatch(vid):
            continue
        out.append(Candidate(
            video_id=vid,
            title=title.strip(),
            duration_s=_to_int(dur),
            view_count=_to_int(views),
            is_live=live.strip().lower() in ("true", "1"),
        ))
    return out


def filter_candidates(cands: list[Candidate], *,
                      min_minutes: float = 3.0,
                      max_minutes: float = 90.0) -> list[Candidate]:
    """Keep only what the clip DAG can turn into good shorts.

    - Live and upcoming items are skipped: they have no fixed duration and the
      VOD does not exist yet.
    - Unknown-duration items are skipped: a search shelf/channel row, never a
      real video, is the usual cause.
    - Too-short sources cannot yield a distinct 9:16 moment; too-long ones tie
      up the GPU for a first pass (the caller can raise the ceiling).

    Input order is preserved, which is view-count order from the search — so
    the first survivor is the most-viewed usable video.
    """
    lo, hi = min_minutes * 60.0, max_minutes * 60.0
    kept: list[Candidate] = []
    for c in cands:
        if c.is_live or c.duration_s is None:
            continue
        if c.duration_s < lo or c.duration_s > hi:
            continue
        kept.append(c)
    return kept


def probe_language(video_id: str, *, run: RunTool = run_tool) -> str | None:
    """Best-effort ISO language for one video (an extra extraction call).

    Only used when the caller asks to language-filter, because a flat listing
    does not carry language and probing every candidate would defeat the point
    of a flat (cheap) listing.
    """
    if not _VIDEO_ID_RE.fullmatch(video_id):
        return None
    try:
        proc = run("yt-dlp", [
            "--skip-download", "--no-warnings",
            "--print", "%(language)s",
            f"https://www.youtube.com/watch?v={video_id}",
        ], timeout_s=60.0)
    except Exception:  # noqa: BLE001 - language is a soft preference, never fatal
        return None
    line = (proc.stdout or "").strip().splitlines()
    val = (line[-1].strip() if line else "")
    return None if not val or val.upper() == "NA" else val.lower()


#: How many search results to pull before filtering. Deliberately much larger
#: than the number anyone wants back: duration filtering is aggressive (a
#: trending page is mostly Shorts and livestreams), and it is ONE flat-playlist
#: call whatever the number, so fetching wide costs nothing. Measured with
#: limit=15 and a 3-12 minute window, 15 raw candidates left exactly 1 usable —
#: which then made `--lang` fail outright with nothing to match against.
DEFAULT_LIMIT = 60


def discover(query: str, *, region: str = "US",
             min_minutes: float = 3.0, max_minutes: float = 90.0,
             limit: int = DEFAULT_LIMIT,
             stats: dict | None = None,
             run: RunTool = run_tool) -> list[Candidate]:
    """Trending, clip-ready videos for ``query`` — most-viewed first.

    ``--flat-playlist`` never downloads media; it lists the search results and
    their metadata in one cheap call.

    ``stats``, when given, is filled in with why candidates were dropped
    (``raw``/``usable``/``too_long``/``too_short``/``live``). Passing a dict
    rather than changing the return type keeps every existing caller working,
    and lets the CLI explain a thin result instead of blaming the wrong
    filter: searching "podcast" with a 4-10 minute window dropped 58 of 59 as
    too long, and the failure that surfaced was about --lang.
    """
    url = build_search_url(query, region=region)
    proc = run("yt-dlp", [
        "--flat-playlist",
        "--playlist-end", str(max(1, int(limit))),
        "--print", _PRINT_TEMPLATE,
        "--no-warnings",
        url,
    ], timeout_s=90.0)
    cands = parse_candidates(proc.stdout or "")
    kept = filter_candidates(cands, min_minutes=min_minutes, max_minutes=max_minutes)
    counts = {
        "raw": len(cands), "usable": len(kept),
        "too_short": sum(1 for c in cands if c.duration_s is not None
                         and not c.is_live and c.duration_s < min_minutes * 60),
        "too_long": sum(1 for c in cands if c.duration_s is not None
                        and not c.is_live and c.duration_s > max_minutes * 60),
        "live": sum(1 for c in cands if c.is_live),
    }
    if stats is not None:
        stats.update(counts)
    log.info("trending.discovered", query=query, region=region, **counts)
    return kept
