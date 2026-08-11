"""T1 — virtual windows: chunk + trailing overlap of the previous chunk.

A 15-minute hard cut severs the best moment of the stream, so the DAG never
processes a bare chunk: it processes ``tail(prev, overlap_s) + chunk``,
joined with ffmpeg's **concat demuxer** (stream copy, no re-encode), and
every downstream timestamp is ABSOLUTE stream time so candidates that
appear in two adjacent windows dedup naturally.

Precision note, recorded honestly: ``-ss`` with ``-c copy`` cuts on the
previous keyframe, so the actual tail is usually a little LONGER than
requested. We probe the produced tail and report the REAL overlap; the
window's ``abs_start_s`` uses the probed value, keeping absolute time exact
even though the requested overlap is approximate.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from clipforge.errors import FfmpegError, IngestError
from clipforge.ffmpeg import (COPY_TIMEOUT_S, MediaInfo, probe, require_binary,
                              run)
from clipforge.log import get_logger
from clipforge.paths import _replace_with_retry

log = get_logger(__name__)


@dataclass(frozen=True)
class VirtualWindow:
    """What the DAG actually consumes (spec T1)."""

    path: Path
    #: Absolute stream time of this file's t=0 (= chunk start − real overlap).
    abs_start_s: float
    #: The overlap actually achieved (keyframe-quantized), not the requested.
    overlap_s: float
    duration_s: float


def _cut_tail(src: Path, overlap_s: float, src_duration_s: float,
              dest: Path, *, runner: Callable = run) -> Path:
    """Copy the last ~overlap_s of ``src`` into ``dest`` (TS, stream copy)."""
    ffmpeg = require_binary("ffmpeg")
    seek = max(0.0, src_duration_s - overlap_s)
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        runner([str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                "-ss", f"{seek:.3f}", "-i", str(src),
                "-map", "0", "-c", "copy", "-f", "mpegts", str(tmp)],
               timeout=COPY_TIMEOUT_S)
        _replace_with_retry(tmp, dest)
    finally:
        # An unattended process may never restart, so the startup sweep is
        # not a sufficient backstop for these temps (ffmpeg.py documents the
        # same reasoning for remux).
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return dest


def _concat_ts(parts: list[Path], dest: Path, *, runner: Callable = run) -> Path:
    """Concat demuxer join (no re-encode). The list file uses forward
    slashes — ffmpeg's concat parser chokes on backslashes on Windows."""
    ffmpeg = require_binary("ffmpeg")
    list_file = dest.with_suffix(".txt")
    list_file.write_text(
        "".join(f"file '{p.resolve().as_posix()}'\n" for p in parts),
        encoding="utf-8")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    try:
        runner([str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(list_file),
                "-map", "0", "-c", "copy", "-f", "mpegts", str(tmp)],
               timeout=COPY_TIMEOUT_S)
        _replace_with_retry(tmp, dest)
    finally:
        for leftover in (list_file, tmp):
            if leftover.exists():
                try:
                    leftover.unlink()
                except OSError:
                    pass
    return dest


def build_virtual_window(*, chunk: Path, chunk_abs_start_s: float,
                         prev_chunk: Path | None, overlap_s: float,
                         out_dir: Path,
                         prober: Callable[[Path], MediaInfo] = probe,
                         runner: Callable = run) -> VirtualWindow:
    """Produce the T1 virtual window for ``chunk``.

    First chunk of a session (``prev_chunk=None``) or ``overlap_s == 0``:
    the window IS the chunk (no copy — content-addressing makes reuse safe).
    Otherwise: cut prev's tail, concat, probe, and report exact numbers.
    """
    if prev_chunk is None or overlap_s <= 0:
        info = prober(chunk)
        return VirtualWindow(path=chunk, abs_start_s=chunk_abs_start_s,
                             overlap_s=0.0, duration_s=info.duration_s)

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        prev_info = prober(prev_chunk)
        tail = _cut_tail(prev_chunk, overlap_s, prev_info.duration_s,
                         out_dir / f"{chunk.stem}.tail.ts", runner=runner)
        real_overlap = prober(tail).duration_s
        window = _concat_ts([tail, chunk],
                            out_dir / f"{chunk.stem}.window.ts", runner=runner)
        duration = prober(window).duration_s
    except FfmpegError as exc:
        # Degrade, never die: a broken prev tail must not block THIS chunk.
        log.warning("overlap.window_failed", chunk=str(chunk), error=str(exc),
                    action="fall back to bare chunk (overlap lost)")
        info = prober(chunk)
        return VirtualWindow(path=chunk, abs_start_s=chunk_abs_start_s,
                             overlap_s=0.0, duration_s=info.duration_s)

    return VirtualWindow(
        path=window,
        abs_start_s=chunk_abs_start_s - real_overlap,
        overlap_s=real_overlap,
        duration_s=duration,
    )


def dedup_by_absolute_time(candidates: list[tuple[float, float]],
                           *, iou_threshold: float = 0.5) -> list[tuple[float, float]]:
    """T1's second half: windows overlap, so the SAME moment surfaces from
    two adjacent virtual windows at (nearly) the same ABSOLUTE times.
    Greedy keep-first dedup over (abs_start, abs_end) pairs, sorted for
    determinism. Shared helper for S2's cross-chunk merge (CP2 consumes it).

    Degenerate inputs are normalized rather than silently mis-handled:
    inverted pairs are swapped, non-finite ones dropped, and exact duplicates
    (including zero-length ones, whose IoU is 0/0) collapse. Each candidate
    is then compared against EVERY kept window — see the note in the body on
    why the obvious sorted-order short-circuit is unsound.
    """
    import math

    norm: list[tuple[float, float]] = []
    seen: set[tuple[float, float]] = set()
    for start, end in candidates:
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        if end < start:
            start, end = end, start
        if (start, end) in seen:
            continue  # exact duplicate, incl. zero-length
        seen.add((start, end))
        norm.append((start, end))

    kept: list[tuple[float, float]] = []
    for start, end in sorted(norm):
        dup = False
        # Full scan of `kept`, deliberately. An earlier "sorted ⇒ everything
        # earlier is disjoint" short-circuit was UNSOUND: `kept` is ordered by
        # START, not by END, so one short early-ending window hid a longer
        # earlier one that still overlapped. Example it got wrong:
        # [(0,100),(30,35),(40,100)] kept all three, though IoU((0,100),
        # (40,100)) = 0.6. Candidate counts here are small (top-K per window),
        # so correctness beats the micro-optimization.
        for ks, ke in kept:
            inter = min(end, ke) - max(start, ks)
            union = max(end, ke) - min(start, ks)
            if inter > 0 and union > 0 and inter / union > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append((start, end))
    return kept
