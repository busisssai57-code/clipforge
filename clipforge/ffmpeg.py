"""ffmpeg/ffprobe subprocess wrappers.

Everything media-shaped funnels through here so that:
  * binary discovery happens once (PATH → env override → known local installs),
  * every subprocess is spawned with ``CREATE_NO_WINDOW`` on Windows,
  * stderr is always captured and attached to typed :class:`FfmpegError`s,
  * the two-pass loudnorm dance (T8) has exactly one implementation.

No stage shells out to ffmpeg directly.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from clipforge.errors import FfmpegError, PreflightError
from clipforge.log import get_logger

log = get_logger(__name__)

# Windows: never flash a console window from a background service.
CREATE_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0

_STDERR_TAIL_CHARS = 4000


def _win_flags() -> dict:
    return {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def find_binary(name: str) -> Path | None:
    """Locate ffmpeg/ffprobe: env override → PATH → imageio-ffmpeg (ffmpeg only).

    Env override: ``CLIPFORGE_FFMPEG_DIR`` pointing at a directory that
    contains both binaries — this is the documented fix on machines where
    ffmpeg is not on PATH (doctor tells the operator exactly this).
    """
    override = os.environ.get("CLIPFORGE_FFMPEG_DIR")
    if override:
        cand = Path(override) / (f"{name}.exe" if sys.platform == "win32" else name)
        if cand.exists():
            return cand
    which = shutil.which(name)
    if which:
        return Path(which)
    if name == "ffmpeg":
        try:  # imageio-ffmpeg ships a static ffmpeg (no ffprobe) — last resort
            import imageio_ffmpeg  # noqa: PLC0415

            return Path(imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            return None
    return None


def require_binary(name: str) -> Path:
    path = find_binary(name)
    if path is None:
        raise PreflightError(
            f"{name} not found. Install ffmpeg (winget install Gyan.FFmpeg) or set "
            "CLIPFORGE_FFMPEG_DIR to a directory containing ffmpeg.exe and ffprobe.exe."
        )
    return path


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


def run(cmd: list[str], *, timeout: float | None = None,
        check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a media subprocess to completion, capturing stderr for diagnostics."""
    log.debug("ffmpeg.run", cmd=cmd[:8])
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout, **_win_flags())
    except subprocess.TimeoutExpired as exc:
        raise FfmpegError(f"{cmd[0]} timed out after {timeout}s", cmd=cmd,
                          stderr_tail=(exc.stderr or "")[-_STDERR_TAIL_CHARS:] if isinstance(exc.stderr, str) else "") from exc
    except OSError as exc:
        raise FfmpegError(f"Cannot execute {cmd[0]}: {exc}", cmd=cmd) from exc
    if check and proc.returncode != 0:
        raise FfmpegError(
            f"{Path(cmd[0]).name} exited {proc.returncode}", cmd=cmd,
            returncode=proc.returncode,
            stderr_tail=proc.stderr[-_STDERR_TAIL_CHARS:])
    return proc


# --------------------------------------------------------------------------
# ffprobe
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MediaInfo:
    """The subset of probe data the pipeline actually consumes."""

    duration_s: float
    width: int | None
    height: int | None
    fps: float | None          # exact rational evaluated (T10: pin the true fps)
    fps_rational: str | None   # e.g. "60000/1001" — pass THIS to -framerate
    v_codec: str | None
    a_codec: str | None


#: ffprobe must never block forever: it runs on the chunker's worker thread,
#: which is not cancellable, so a wedged probe on a torn TS would also block
#: process exit. Generous but bounded.
PROBE_TIMEOUT_S = 120.0

#: Same reasoning for stream-copy work (remux, T1 tail cut + concat): it runs
#: on that same worker thread, so an unbounded call makes `clipforge watch`
#: unkillable and stalls the disk guard. A 15-minute 1080p60 stream copy is
#: I/O-bound and finishes in well under this.
COPY_TIMEOUT_S = 900.0


def ffprobe_json(path: Path | str) -> dict:
    ffprobe = require_binary("ffprobe")
    proc = run([str(ffprobe), "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)],
               timeout=PROBE_TIMEOUT_S)
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise FfmpegError(f"ffprobe produced non-JSON output for {path}",
                          stderr_tail=proc.stderr[-_STDERR_TAIL_CHARS:]) from exc


def parse_media_info(probe: dict) -> MediaInfo:
    """Pure parser (unit-testable without binaries)."""
    fmt = probe.get("format", {})
    duration = float(fmt.get("duration", 0.0) or 0.0)
    width = height = None
    fps = None
    fps_rational = None
    v_codec = a_codec = None
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video" and v_codec is None:
            v_codec = stream.get("codec_name")
            width = stream.get("width")
            height = stream.get("height")
            # avg_frame_rate is the honest rate for VFR-ish HLS captures;
            # r_frame_rate is the container tick rate and can be double.
            rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or ""
            m = re.fullmatch(r"(\d+)/(\d+)", rate)
            if m and int(m.group(2)) != 0:
                fps_rational = rate
                fps = int(m.group(1)) / int(m.group(2))
            if duration == 0.0 and stream.get("duration"):
                duration = float(stream["duration"])
        elif stream.get("codec_type") == "audio" and a_codec is None:
            a_codec = stream.get("codec_name")
    return MediaInfo(duration_s=duration, width=width, height=height, fps=fps,
                     fps_rational=fps_rational, v_codec=v_codec, a_codec=a_codec)


def probe(path: Path | str) -> MediaInfo:
    return parse_media_info(ffprobe_json(path))


# --------------------------------------------------------------------------
# remux (T2: .ts → .mp4 after the segment is provably closed)
# --------------------------------------------------------------------------


def remux_ts_to_mp4(src: Path, dest: Path) -> Path:
    """Stream-copy remux with faststart. Writes to a temp name then renames,
    so a killed remux never leaves a plausible-looking dest file.

    The temp uses the ``.partial`` suffix so the startup sweep collects it
    after a hard crash; on an ordinary failure it is removed HERE — an
    unattended process that never restarts must not accumulate GB-scale
    debris until the disk floor pauses ingestion.
    """
    from clipforge.paths import _replace_with_retry  # local: avoid cycle

    ffmpeg = require_binary("ffmpeg")
    tmp = dest.with_suffix(dest.suffix + ".remux.partial")
    try:
        # Bounded: this runs on the chunker's non-cancellable worker thread,
        # so an unbounded call would make shutdown hang (see COPY_TIMEOUT_S).
        run([str(ffmpeg), "-y", "-hide_banner", "-loglevel", "error",
             "-i", str(src), "-map", "0", "-c", "copy",
             "-movflags", "+faststart", "-f", "mp4", str(tmp)],
            timeout=COPY_TIMEOUT_S)
        # Same Windows hold-hazard as artifact commits: retried + typed
        # (AtomicWriteError) instead of a raw PermissionError. Fresh multi-GB
        # files are exactly what Defender scans.
        _replace_with_retry(tmp, dest)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass  # locked; the startup sweep is the backstop
    return dest


# --------------------------------------------------------------------------
# two-pass loudnorm (T8)
# --------------------------------------------------------------------------

# ffmpeg prints the measurement JSON to stderr after a banner line; there may
# be other braces in the log, so scan for the LAST well-formed JSON object.
_LOUDNORM_KEYS = {"input_i", "input_tp", "input_lra", "input_thresh"}


@dataclass(frozen=True)
class LoudnormMeasurement:
    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    target_offset: float


def parse_loudnorm_json(stderr: str) -> LoudnormMeasurement:
    """Extract the loudnorm measurement block from ffmpeg stderr (pure)."""
    candidates = re.findall(r"\{[^{}]*\}", stderr, flags=re.DOTALL)
    for blob in reversed(candidates):
        try:
            data = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if _LOUDNORM_KEYS.issubset(data.keys()):
            return LoudnormMeasurement(
                input_i=float(data["input_i"]),
                input_tp=float(data["input_tp"]),
                input_lra=float(data["input_lra"]),
                input_thresh=float(data["input_thresh"]),
                target_offset=float(data.get("target_offset", 0.0)),
            )
    raise FfmpegError("No loudnorm measurement JSON found in ffmpeg stderr",
                      stderr_tail=stderr[-_STDERR_TAIL_CHARS:])


def measure_loudness(path: Path, *, i: float = -14.0, tp: float = -1.5,
                     lra: float = 11.0) -> LoudnormMeasurement:
    """Pass 1: loudnorm in analysis mode (print_format=json, output discarded)."""
    ffmpeg = require_binary("ffmpeg")
    proc = run([str(ffmpeg), "-hide_banner", "-nostats", "-i", str(path),
                "-af", f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json",
                "-f", "null", os.devnull], check=True)
    return parse_loudnorm_json(proc.stderr)


def loudnorm_filter(measured: LoudnormMeasurement, *, i: float = -14.0,
                    tp: float = -1.5, lra: float = 11.0) -> str:
    """Pass 2 filter string: linear mode with pass-1 measurements (spec §S6)."""
    return (
        f"loudnorm=I={i}:TP={tp}:LRA={lra}"
        f":measured_I={measured.input_i}:measured_TP={measured.input_tp}"
        f":measured_LRA={measured.input_lra}:measured_thresh={measured.input_thresh}"
        f":offset={measured.target_offset}:linear=true"
    )


# --------------------------------------------------------------------------
# capability probes (doctor)
# --------------------------------------------------------------------------


def list_encoders() -> str:
    ffmpeg = require_binary("ffmpeg")
    return run([str(ffmpeg), "-hide_banner", "-encoders"]).stdout


def list_filters() -> str:
    ffmpeg = require_binary("ffmpeg")
    return run([str(ffmpeg), "-hide_banner", "-filters"]).stdout
