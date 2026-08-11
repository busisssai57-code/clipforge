"""Subprocess runner for external ingest tools (yt-dlp, streamlink).

One narrow seam so every platform module is unit-testable offline: tests
inject a fake ``Runner`` returning canned stdout; production uses
:func:`run_tool`, which locates the tool (venv Scripts, then PATH), applies
``CREATE_NO_WINDOW``, captures output, and converts every failure mode into
typed :class:`IngestError` — network flakiness is NORMAL here (§2), so
callers must be able to catch one exception type and back off.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from clipforge.errors import IngestError
from clipforge.ffmpeg import CREATE_NO_WINDOW
from clipforge.log import get_logger

log = get_logger(__name__)


class ToolResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


Runner = Callable[..., "subprocess.CompletedProcess[str]"]


def find_tool(name: str) -> Path | None:
    """Locate a tool: this venv's Scripts dir first (pip-installed
    streamlink/yt-dlp), then PATH. Venv-first means the doctor-verified
    versions are the ones actually used."""
    exe = f"{name}.exe" if sys.platform == "win32" else name
    venv_scripts = Path(sys.executable).parent
    cand = venv_scripts / exe
    if cand.exists():
        return cand
    which = shutil.which(name)
    return Path(which) if which else None


def run_tool(name: str, args: list[str], *, timeout_s: float = 120.0,
             ok_codes: tuple[int, ...] = (0,),
             stop: "threading.Event | None" = None,
             poll_interval_s: float = 0.5) -> subprocess.CompletedProcess[str]:
    """Run an external tool to completion. Raises IngestError for: tool
    missing, timeout, spawn failure, cancellation, or exit code outside
    ``ok_codes`` (streamlink exits nonzero for 'no stream' — pass
    ok_codes=(0,1) and inspect output when that is expected, see twitch.py).

    ``stop`` makes long downloads INTERRUPTIBLE. Without it, a Ctrl+C during
    an hour-long yt-dlp run cannot take effect until the subprocess finishes
    — review measured a full-timeout hang, because ``asyncio.to_thread``
    workers are not cancellable and the interpreter joins them on exit.
    """
    tool = find_tool(name)
    if tool is None:
        raise IngestError(f"{name} not found - pip install {name} (doctor checks this)")
    cmd = [str(tool), *args]
    flags = CREATE_NO_WINDOW if sys.platform == "win32" else 0

    if stop is None:
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=timeout_s, creationflags=flags)
        except subprocess.TimeoutExpired as exc:
            raise IngestError(f"{name} timed out after {timeout_s}s: {args[:4]}") from exc
        except OSError as exc:
            raise IngestError(f"cannot execute {name}: {exc}") from exc
    else:
        proc = _run_interruptible(name, cmd, timeout_s=timeout_s, stop=stop,
                                  poll_interval_s=poll_interval_s, flags=flags)

    if proc.returncode not in ok_codes:
        raise IngestError(
            f"{name} exited {proc.returncode}: {(proc.stderr or '')[-500:].strip()}")
    return proc


def _run_interruptible(name: str, cmd: list[str], *, timeout_s: float,
                       stop: "threading.Event", poll_interval_s: float,
                       flags: int) -> subprocess.CompletedProcess[str]:
    """Popen + poll loop that kills the child when ``stop`` fires."""
    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", creationflags=flags)
    except OSError as exc:
        raise IngestError(f"cannot execute {name}: {exc}") from exc

    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=poll_interval_s)
                return subprocess.CompletedProcess(cmd, proc.returncode, out, err)
            except subprocess.TimeoutExpired:
                pass
            if stop.is_set():
                proc.kill()
                proc.communicate()
                raise IngestError(f"{name} cancelled by shutdown")
            if time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                raise IngestError(f"{name} timed out after {timeout_s}s")
    finally:
        if proc.poll() is None:  # pragma: no cover - defensive
            proc.kill()
