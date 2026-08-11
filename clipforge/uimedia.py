"""Derived media the editor UI needs: poster, filmstrip, waveform peaks.

None of these are pipeline outputs — they exist only so a browser can
draw a timeline. They are therefore cached under ``workspace/uicache``
and are safe to delete at any time; a missing cache entry costs one
ffmpeg call, not a re-render.

Three rules this module keeps, all of them learned from the audit that
found the dashboard reporting an idle pipeline forever:

1. **A failure is reported, never faked.** If ffmpeg cannot read the
   audio, the waveform endpoint says so. It does not return a flat line,
   because a flat line is a claim that the clip is silent.
2. **Cache keys include the parameters.** A filmstrip cached at 40
   columns must not be served when 80 are asked for.
3. **Nothing is written outside the cache directory.** Paths come from
   URLs; every one is resolved and confined before use.
"""

from __future__ import annotations

import array
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipforge.errors import FfmpegError
from clipforge.ffmpeg import require_binary, run
from clipforge.log import get_logger
from clipforge.paths import Workspace

log = get_logger(__name__)

#: Derivatives are cheap but not free; a 60s clip's filmstrip is ~1s of
#: ffmpeg. Bounded so a wedged decode cannot pin a request thread.
_STRIP_TIMEOUT_S = 120.0
_PEAKS_TIMEOUT_S = 120.0

#: Thumbnail height in the filmstrip. The timeline row is ~46px tall in
#: the UI; 2× that keeps it crisp on a HiDPI display.
_STRIP_TILE_H = 92

#: Waveform peaks are computed from mono PCM at this rate. 8 kHz is far
#: below speech bandwidth but peak envelope only needs amplitude, and it
#: keeps a 60s decode in the tens of milliseconds.
_PEAK_SAMPLE_RATE = 8000


def cache_dir(ws: Workspace) -> Path:
    """``workspace/uicache``, created on demand."""
    path = Path(ws.root) / "uicache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _win_flags() -> dict:
    if sys.platform == "win32":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0)}
    return {}


def _fresh(cached: Path, source: Path) -> bool:
    """True when the cached derivative is newer than the clip it came from."""
    try:
        return (cached.is_file()
                and cached.stat().st_mtime >= source.stat().st_mtime)
    except OSError:
        return False


# ------------------------------------------------------------------- poster

def poster(ws: Workspace, clip: Path, *, at_s: float = 1.0) -> Path:
    """A single frame for the grid card.

    The export pack already writes ``<clip>.thumb.jpg`` for clips that
    reached the publish stage. That one is preferred — it is the frame
    the operator would actually post. Only when it is absent is one
    grabbed here.
    """
    shipped = clip.with_suffix(".thumb.jpg")
    if shipped.is_file():
        return shipped

    dest = cache_dir(ws) / f"{clip.stem}.poster.jpg"
    if _fresh(dest, clip):
        return dest

    ffmpeg = require_binary("ffmpeg")
    run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
         "-ss", f"{max(0.0, at_s):.3f}", "-i", str(clip),
         "-frames:v", "1", "-q:v", "4", str(dest)],
        timeout=_STRIP_TIMEOUT_S)
    if not dest.is_file():
        raise FfmpegError(f"poster frame not produced for {clip.name}")
    return dest


# ---------------------------------------------------------------- filmstrip

@dataclass(frozen=True)
class Filmstrip:
    path: Path
    columns: int
    tile_w: int
    tile_h: int

    def as_dict(self) -> dict[str, Any]:
        return {"columns": self.columns, "tile_w": self.tile_w,
                "tile_h": self.tile_h}


def filmstrip(ws: Workspace, clip: Path, *, duration_s: float,
              columns: int = 40) -> Filmstrip:
    """One wide JPEG of ``columns`` evenly spaced frames.

    A tiled sprite rather than N files: the timeline shows every frame at
    once, and forty separate requests to paint one row is forty times the
    overhead for the same pixels.
    """
    columns = max(4, min(120, int(columns)))
    if duration_s <= 0:
        raise ValueError("duration_s must be positive to space frames")

    meta_path = cache_dir(ws) / f"{clip.stem}.strip{columns}.json"
    dest = cache_dir(ws) / f"{clip.stem}.strip{columns}.jpg"
    if _fresh(dest, clip) and _fresh(meta_path, clip):
        try:
            blob = json.loads(meta_path.read_text(encoding="utf-8"))
            return Filmstrip(dest, columns, int(blob["tile_w"]),
                             int(blob["tile_h"]))
        except (OSError, ValueError, KeyError):
            pass  # regenerate rather than trust a half-written cache

    # fps chosen so exactly `columns` frames fall inside the clip. The
    # tile filter pads a short final row, which is why frames:v 1 is safe.
    fps = columns / duration_s
    ffmpeg = require_binary("ffmpeg")
    run([str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(clip),
         "-vf", (f"fps={fps:.6f},scale=-2:{_STRIP_TILE_H},"
                 f"tile={columns}x1"),
         "-frames:v", "1", "-q:v", "5", str(dest)],
        timeout=_STRIP_TIMEOUT_S)
    if not dest.is_file():
        raise FfmpegError(f"filmstrip not produced for {clip.name}")

    # Tile width is the source aspect at the fixed height; measure it from
    # the produced sprite rather than assuming 9:16, because a 16:9 clip
    # tiles just as legitimately.
    from clipforge.ffmpeg import probe  # local: keeps import graph flat

    info = probe(dest)
    tile_w = int((info.width or (_STRIP_TILE_H * columns * 9 // 16))
                 / columns)
    tile_h = int(info.height or _STRIP_TILE_H)
    meta_path.write_text(json.dumps({"tile_w": tile_w, "tile_h": tile_h}),
                         encoding="utf-8")
    return Filmstrip(dest, columns, tile_w, tile_h)


# ----------------------------------------------------------------- waveform

def waveform(ws: Workspace, clip: Path, *, duration_s: float,
             buckets: int = 900) -> dict[str, Any]:
    """Peak envelope as ``buckets`` values in [0,1].

    Decodes mono 16-bit PCM and takes the maximum absolute sample per
    bucket. Peak, not RMS: the timeline is read to find where speech
    starts and stops, and RMS smooths exactly the transients that answer
    that question.
    """
    buckets = max(50, min(4000, int(buckets)))
    dest = cache_dir(ws) / f"{clip.stem}.peaks{buckets}.json"
    if _fresh(dest, clip):
        try:
            return json.loads(dest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass

    ffmpeg = require_binary("ffmpeg")
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error",
           "-i", str(clip), "-map", "0:a:0", "-ac", "1",
           "-ar", str(_PEAK_SAMPLE_RATE), "-f", "s16le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True,
                              timeout=_PEAKS_TIMEOUT_S, **_win_flags())
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"waveform decode timed out for {clip.name}") from exc
    except OSError as exc:
        raise FfmpegError(f"cannot execute ffmpeg: {exc}") from exc

    if proc.returncode != 0 or not proc.stdout:
        tail = (proc.stderr or b"")[-400:].decode("utf-8", "replace")
        # No audio stream is a real, reportable state — not zero peaks.
        raise FfmpegError(
            f"no decodable audio in {clip.name}", stderr_tail=tail)

    pcm = array.array("h")
    usable = len(proc.stdout) - (len(proc.stdout) % 2)
    pcm.frombytes(proc.stdout[:usable])
    if sys.byteorder == "big":
        pcm.byteswap()
    if not pcm:
        raise FfmpegError(f"empty audio stream in {clip.name}")

    total = len(pcm)
    size = max(1, total // buckets)
    peaks: list[float] = []
    for i in range(0, total, size):
        window = pcm[i:i + size]
        if not window:
            continue
        peaks.append(round(max(abs(min(window)), abs(max(window))) / 32768.0, 4))
    peaks = peaks[:buckets]

    blob = {
        "peaks": peaks,
        "buckets": len(peaks),
        "duration_s": round(duration_s, 3),
        "sample_rate": _PEAK_SAMPLE_RATE,
        # Decoded seconds, so a caller can tell a truncated decode from a
        # short clip instead of stretching the envelope over the timeline.
        "decoded_s": round(total / _PEAK_SAMPLE_RATE, 3),
    }
    try:
        dest.write_text(json.dumps(blob), encoding="utf-8")
    except OSError as exc:
        log.warning("uimedia.peaks_cache_failed", error=str(exc)[:200])
    return blob
