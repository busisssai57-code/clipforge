"""Windows Job Objects — children die when the parent dies (trap T3).

The problem this solves, demonstrated in review: `TerminateProcess` on the
ClipForge process (SIGKILL, Task Manager, a crash) does NOT reap the
streamlink/ffmpeg pair. The orphans keep encoding into a session directory
that the restarted process never rescans, stranding media and burning CPU
indefinitely. Python's `atexit`/`finally` cannot help — a hard kill runs no
Python code at all.

The OS-level answer is a Job Object with
``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``: when the last handle to the job
closes — which the kernel does automatically when the owning process dies,
however it dies — every process in the job is terminated.

This module is deliberately total: every failure degrades to "no job
object" plus a warning. A missing safety net must never take down the
recorder that the net was meant to protect.
"""

from __future__ import annotations

import sys
from typing import Any

from clipforge.log import get_logger

log = get_logger(__name__)

# winbase.h / jobapi2.h constants
_JobObjectExtendedLimitInformation = 9
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class ProcessGuard:
    """Kill-on-close job object. ``assign(proc)`` enrolls a child.

    Usage: one guard per chunker connect; keep it alive for the connect's
    duration. Dropping the last reference closes the job handle, which
    terminates any child still running — a belt-and-braces backstop for the
    explicit teardown, and the ONLY protection against a hard parent kill.
    """

    def __init__(self) -> None:
        self._handle: Any = None
        self._kernel32: Any = None
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

            class IO_COUNTERS(ctypes.Structure):
                _fields_ = [(n, ctypes.c_ulonglong) for n in
                            ("ReadOperationCount", "WriteOperationCount",
                             "OtherOperationCount", "ReadTransferCount",
                             "WriteTransferCount", "OtherTransferCount")]

            class JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD),
                ]

            class JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
                _fields_ = [
                    ("BasicLimitInformation", JOBOBJECT_BASIC_LIMIT_INFORMATION),
                    ("IoInfo", IO_COUNTERS),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t),
                ]

            handle = kernel32.CreateJobObjectW(None, None)
            if not handle:
                raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")

            info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
            info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
            if not kernel32.SetInformationJobObject(
                    handle, _JobObjectExtendedLimitInformation,
                    ctypes.byref(info), ctypes.sizeof(info)):
                err = ctypes.get_last_error()
                kernel32.CloseHandle(handle)
                raise OSError(err, "SetInformationJobObject failed")

            self._handle = handle
            self._kernel32 = kernel32
            self._ctypes = ctypes
        except Exception as exc:  # total: no job object is survivable
            log.warning("procguard.unavailable", error=f"{type(exc).__name__}: {exc}",
                        note="children will NOT be auto-reaped if the parent is killed")
            self._handle = None

    @property
    def active(self) -> bool:
        return self._handle is not None

    def assign(self, proc: Any) -> bool:
        """Enroll ``proc`` (a Popen) in the job. Returns success; never raises."""
        if self._handle is None or not hasattr(proc, "pid"):
            return False
        try:
            ctypes = self._ctypes
            hproc = self._kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, int(proc.pid))
            if not hproc:
                return False
            try:
                ok = bool(self._kernel32.AssignProcessToJobObject(self._handle, hproc))
                if not ok:
                    log.debug("procguard.assign_failed", pid=proc.pid,
                              winerr=ctypes.get_last_error())
                return ok
            finally:
                self._kernel32.CloseHandle(hproc)
        except Exception as exc:
            log.debug("procguard.assign_error", error=str(exc))
            return False

    def close(self) -> None:
        """Close the job handle. With KILL_ON_JOB_CLOSE this terminates any
        process still in the job — the last-resort teardown."""
        if self._handle is not None and self._kernel32 is not None:
            try:
                self._kernel32.CloseHandle(self._handle)
            except Exception:
                pass
            self._handle = None

    def __enter__(self) -> "ProcessGuard":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
