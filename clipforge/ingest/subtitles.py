"""Bring-your-own subtitles — an accurate SRT/VTT stands in for the ASR.

S1 runs WhisperX to get a transcript. When an operator already has an
accurate caption file — a broadcaster's SRT, a hand-corrected VTT — that
work is done, and re-transcribing is slower and often worse. This module
parses SRT and WebVTT into the same segment shape S1 emits, so a run with
``--subtitles`` skips ASR, alignment and diarization entirely and every
downstream stage sees a normal transcript.

Two honesty markers travel with it:

* **Word times are ESTIMATED, not aligned.** SRT/VTT carry one time span
  per caption line, not per word. Words are spread across their line's
  span in proportion to their length, and the artifact is marked
  ``words_aligned=False`` — captions still land, but their per-word timing
  is apportioned, not measured, and the pipeline is told so rather than
  left to infer it.
* **No speakers.** A caption file has no diarization, so the artifact is
  ``diarization_ok=False`` and downstream degrades exactly as it does when
  pyannote is unavailable — the turn heuristic stands down, the rest runs.

Nothing here reaches the network or a model. A malformed file raises a
typed error the caller turns into a clear message; it never silently
yields a half-parsed transcript.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipforge.errors import ClipForgeError
from clipforge.log import get_logger

log = get_logger(__name__)

#: SRT uses a comma before milliseconds, VTT a dot; both allow an optional
#: hours field. One pattern reads either.
_TS = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE_RANGE = re.compile(rf"^\s*{_TS}\s*-->\s*{_TS}")


@dataclass(frozen=True)
class Cue:
    """One caption line: a text span with a start and end in media seconds."""

    start: float
    end: float
    text: str


def _ts_to_seconds(h: str | None, m: str, s: str, frac: str) -> float:
    hours = int(h) if h else 0
    ms = int(frac.ljust(3, "0")[:3])  # "5" -> 500 ms, "50" -> 500, "500" -> 500
    return hours * 3600 + int(m) * 60 + int(s) + ms / 1000.0


_VTT_TAGS = re.compile(r"</?[cvbiu.][^>]*>|<\d{2}:\d{2}:\d{2}[.,]\d{1,3}>")


def _clean_text(lines: list[str]) -> str:
    """Join a cue's text lines and strip VTT inline tags and cue-position
    junk. A caption's job here is the words; styling and karaoke timing
    tags are noise to the transcript."""
    joined = " ".join(ln.strip() for ln in lines if ln.strip())
    joined = _VTT_TAGS.sub("", joined)
    return re.sub(r"\s+", " ", joined).strip()


def parse_subtitles(path: Path | str) -> list[Cue]:
    """Parse an SRT or WebVTT file into time-ordered cues.

    Format is detected from content, not the extension: both are
    blank-line-delimited blocks whose second-or-first line is a
    ``start --> end`` range. Blocks with no valid range (SRT indices, VTT
    ``NOTE``/``STYLE``/``WEBVTT`` headers, cue identifiers) are skipped.
    Overlapping or out-of-order cues are kept and sorted — a downstream
    concern, not a parse error. Raises :class:`ClipForgeError` only when the
    file cannot be read or contains no cue at all.
    """
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        raise ClipForgeError(f"could not read subtitles {p}: {exc}") from exc

    cues: list[Cue] = []
    block: list[str] = []

    def _flush() -> None:
        if not block:
            return
        # The range line is the first line that IS one — SRT puts an index
        # line above it, VTT may put a cue identifier there.
        range_idx = next((i for i, ln in enumerate(block)
                          if _CUE_RANGE.match(ln)), None)
        if range_idx is None:
            return
        m = _CUE_RANGE.match(block[range_idx])
        assert m is not None
        g = m.groups()
        start = _ts_to_seconds(g[0], g[1], g[2], g[3])
        end = _ts_to_seconds(g[4], g[5], g[6], g[7])
        body = _clean_text(block[range_idx + 1:])
        if end > start and body:
            cues.append(Cue(start=start, end=end, text=body))

    for raw in text.splitlines():
        if raw.strip() == "":
            _flush()
            block = []
        else:
            block.append(raw)
    _flush()

    if not cues:
        raise ClipForgeError(
            f"{p.name} parsed to no captions — is it an SRT or VTT file?")
    cues.sort(key=lambda c: (c.start, c.end))
    log.info("subtitles.parsed", path=str(p), cues=len(cues))
    return cues


# --------------------------------------------------------- word apportioning

def _apportion_words(cue: Cue) -> list[dict[str, Any]]:
    """Spread a cue's words across its span in proportion to their length.

    Word-level times are what captions and the editor's word marks key
    off. A caption line gives only its own span, so each word is given a
    slice proportional to its character length (plus one, so a single-
    character word is never zero-width). Boundaries are computed in integer
    milliseconds and are monotonic, so the same cue always yields the same
    word times — the Determinism Law, one function down.
    """
    tokens = cue.text.split()
    if not tokens:
        return []
    weights = [len(t) + 1 for t in tokens]
    total = sum(weights)
    start_ms = round(cue.start * 1000.0)
    end_ms = round(cue.end * 1000.0)
    span_ms = max(1, end_ms - start_ms)

    words: list[dict[str, Any]] = []
    acc = 0
    for tok, w in zip(tokens, weights):
        w_start = start_ms + round(span_ms * acc / total)
        acc += w
        w_end = start_ms + round(span_ms * acc / total)
        if w_end <= w_start:
            w_end = w_start + 1
        words.append({"word": tok, "start": w_start / 1000.0,
                      "end": w_end / 1000.0, "score": None})
    return words


def cues_to_aligned(cues: list[Cue]) -> dict[str, Any]:
    """Shape cues into the ``aligned`` dict S1's assembly step consumes,
    with per-cue words apportioned. One segment per cue."""
    return {"segments": [
        {"start": c.start, "end": c.end, "text": c.text,
         "words": _apportion_words(c)}
        for c in cues]}


# ------------------------------------------------------------ params bridge

def to_params(cues: list[Cue]) -> list[list[Any]]:
    """Canonical ``[[start, end, text], ...]`` for S1's cache-keyed params.

    The subtitle CONTENT enters S1's cache key this way — never the file
    path — so the same captions transcribe the same clip identically, a
    moved file does not bust the cache, and different captions re-run.
    Times round to 3 decimals for a byte-stable digest across machines.
    """
    return [[round(c.start, 3), round(c.end, 3), c.text] for c in cues]


def from_params(rows: Any) -> list[Cue]:
    """Rebuild cues from :meth:`to_params` output (or None → [])."""
    out: list[Cue] = []
    for row in rows or []:
        try:
            start, end, txt = float(row[0]), float(row[1]), str(row[2])
        except (TypeError, ValueError, IndexError):
            continue
        if end > start and txt:
            out.append(Cue(start=start, end=end, text=txt))
    return out


_SUB_SIDECARS = (".srt", ".vtt")


def discover_beside(source: Path | str) -> Path | None:
    """An ``.srt``/``.vtt`` with the source's exact stem, or None. Never a
    different stem — a translated or partial sidecar is not silently used."""
    src = Path(source)
    stem = src.with_suffix("")
    for suffix in _SUB_SIDECARS:
        cand = stem.with_name(stem.name + suffix)
        if cand.is_file():
            return cand
    return None
