"""Is the operator using this machine right now?

`bta watch` records a live stream the moment it starts, because a live
stream is not replayable. Clipping is different: it saturates the GPU for
minutes at a time, and on this box (one RTX 3090, shared with whatever the
operator is doing) that means a stuttering game or a frozen editor. So
clipping waits until nobody is at the machine.

"Nobody is at the machine" is four measurements, because each one alone
lies:

* **keyboard/mouse idle time** (``GetLastInputInfo``). The obvious signal,
  and blind to anything played with a gamepad or watched without touching
  the mouse: XInput does not reset it. It is also blind from the wrong
  Windows session — a scheduled task running as "whether user is logged
  on or not" lives in session 0, where nobody ever types, so it reads as
  permanently idle while the operator works in session 1.
* **fullscreen / presentation state** (``SHQueryUserNotificationState``).
  Catches the film, the stream and the fullscreen game that touch no
  input at all.
* **GPU utilisation and decoder load by someone else** (``nvidia-smi``).
  Catches the controller game the first signal misses, and, through the
  decoder counter, video playback that barely touches the 3D engine. It
  is read only while the pipeline itself is NOT rendering — the gate is
  checked between stages — so the number is everyone else's load.
* **free VRAM.** A game sitting at 0% in a menu, or a local LLM someone
  left loaded, still holds the memory this pipeline needs. Starting a
  window that will die in the VRAM guard helps nobody.
* **CPU load that is not ours, and free RAM.** Heavy work is not always
  on the GPU: a training run, a local model on the CPU, a compile, a
  game's simulation thread. Our own process tree is subtracted, because
  a checkpoint runs while the previous stage is still winding down.

Everything is injected so the gate is testable without a desktop session
or a GPU.
"""

from __future__ import annotations

import contextvars
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from clipforge.log import get_logger

log = get_logger(__name__)

#: Returned by ``seconds_since_input`` when this process cannot see the
#: operator's input at all (wrong session). Distinct from None ("no way to
#: measure here", e.g. not Windows) because it means the opposite: there IS
#: someone whose input we are blind to.
BLIND = "blind"

#: SHQueryUserNotificationState values that mean "do not interrupt".
#: 2 = QUNS_BUSY, 3 = QUNS_RUNNING_D3D_FULL_SCREEN, 4 = QUNS_PRESENTATION_MODE.
#: QUNS_APP (7) is deliberately NOT here: on Windows 10/11 it is the
#: ordinary "an app is in the foreground" state and would read as busy
#: forever.
_BUSY_NOTIFICATION_STATES = {2: "a fullscreen app", 3: "a fullscreen game",
                             4: "presentation mode"}


def _session_can_see_input() -> bool | None:
    """Does THIS process share the session the operator types into?

    None when it cannot be determined. False means GetLastInputInfo is
    answering for a session nobody uses, which is the dangerous case: it
    reports hours of idle while the operator works elsewhere.
    """
    import ctypes

    try:
        kernel32 = ctypes.windll.kernel32
        pid = kernel32.GetCurrentProcessId()
        session = ctypes.c_ulong()
        if not kernel32.ProcessIdToSessionId(pid, ctypes.byref(session)):
            return None
        if session.value == 0:  # services / "run whether logged on or not"
            return False
        kernel32.WTSGetActiveConsoleSessionId.restype = ctypes.c_ulong
        console = kernel32.WTSGetActiveConsoleSessionId()
        if console == 0xFFFFFFFF:  # a session switch is in progress
            return None
        return console == session.value
    except (AttributeError, OSError):
        return None


def seconds_since_input() -> float | str | None:
    """Seconds since the last keyboard or mouse input in this session.

    ``BLIND`` when this process is in a session whose input is not the
    operator's, None when it cannot be measured at all (not Windows).
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    if _session_can_see_input() is False:
        return BLIND

    class LASTINPUTINFO(ctypes.Structure):
        _fields_ = [("cbSize", wintypes.UINT), ("dwTime", wintypes.DWORD)]

    info = LASTINPUTINFO()
    info.cbSize = ctypes.sizeof(LASTINPUTINFO)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
        return None
    ctypes.windll.kernel32.GetTickCount.restype = wintypes.DWORD
    now = int(ctypes.windll.kernel32.GetTickCount()) & 0xFFFFFFFF
    delta = (now - int(info.dwTime)) & 0xFFFFFFFF
    # A timestamp a few ticks AHEAD of the counter (they are sampled
    # separately) wraps to ~49.7 days of idle and would open the gate
    # while the operator types. Anything in the top half of the range is
    # that race, not six weeks away from the desk.
    if delta > 0x7FFFFFFF:
        return 0.0
    return delta / 1000.0


def _nvidia_smi(query: str) -> list[list[str]] | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    rows = [[c.strip() for c in line.split(",")]
            for line in proc.stdout.splitlines() if line.strip()]
    return rows or None


def gpu_state() -> dict[str, float] | None:
    """Busiest GPU's compute %, decoder % and free VRAM in GB, or None."""
    rows = _nvidia_smi("utilization.gpu,utilization.decoder,memory.free")
    if not rows:
        return None
    best: dict[str, float] | None = None
    for row in rows:
        try:
            util, dec, free_mb = (float(row[0]), float(row[1]), float(row[2]))
        except (IndexError, ValueError):
            continue
        if best is None or util > best["util_pct"]:
            best = {"util_pct": util, "decoder_pct": dec,
                    "free_vram_gb": free_mb / 1024.0}
    return best


def system_load(interval_s: float = 0.6, ps: Any = None) -> dict[str, float] | None:
    """CPU load that is NOT ours, and free RAM.

    The GPU signals miss a whole class of heavy work: a training run or a
    local LLM on the CPU, a compile, a game's simulation thread. They also
    cannot be attributed per process on Windows — nvidia-smi reports
    "[N/A]" for per-process VRAM under WDDM — so total free VRAM stands in
    for "someone else is holding the card", and this stands in for
    "someone else is holding the machine".

    Our own process tree is subtracted. A checkpoint runs between stages
    with the previous stage still winding down, and a gate that counted
    our own tail as the operator's load would never open again.
    """
    if ps is None:
        try:
            import psutil as ps
        except ImportError:
            return None
    psutil = ps
    try:
        me = psutil.Process()
        tree = [me, *me.children(recursive=True)]
        for proc in tree:
            try:
                proc.cpu_percent(None)       # prime the per-process counter
            except psutil.Error:
                pass
        total = psutil.cpu_percent(interval=interval_s)
        cores = psutil.cpu_count() or 1
        ours = 0.0
        for proc in tree:
            try:
                ours += proc.cpu_percent(None) / cores
            except psutil.Error:
                pass
        mem = psutil.virtual_memory()
        return {"cpu_pct": max(0.0, total - ours),
                "free_ram_gb": mem.available / 1024 ** 3}
    except Exception:  # noqa: BLE001 - a probe never breaks the gate
        return None


def notification_state() -> int | None:
    """SHQueryUserNotificationState, or None where it cannot be read."""
    if sys.platform != "win32":
        return None
    import ctypes

    try:
        state = ctypes.c_int()
        if ctypes.windll.shell32.SHQueryUserNotificationState(
                ctypes.byref(state)) != 0:  # S_OK == 0
            return None
        return int(state.value)
    except (AttributeError, OSError):
        return None


@dataclass
class IdleGate:
    """Open when the operator is away and the GPU is theirs to take.

    Unknown readings do not block — a machine where nothing can be
    measured has nobody to protect, and a missing nvidia-smi should not
    park the clip queue for ever. A BLIND input reading is the exception:
    that one means there IS an operator this process cannot see, so it
    fails closed.
    """

    idle_after_s: float = 300.0
    gpu_busy_pct: int = 40
    #: Sustained video decode means something is being watched.
    decoder_busy_pct: int = 10
    #: A window needs this much VRAM to survive the stage guards.
    min_free_vram_gb: float = 8.0
    #: Someone else using this much of the CPU is doing something.
    cpu_busy_pct: float = 35.0
    #: Headroom this pipeline needs, and a proxy for a big job in memory.
    min_free_ram_gb: float = 4.0
    #: Consecutive clear readings before the gate opens. One sample catches
    #: a game on a loading screen or between frames; two, a poll apart, do
    #: not.
    quiet_checks: int = 2
    input_idle: Callable[[], float | str | None] = seconds_since_input
    gpu: Callable[[], dict[str, float] | None] = gpu_state
    notification: Callable[[], int | None] = notification_state
    load: Callable[[], dict[str, float] | None] = system_load
    _warned: set = field(default_factory=set, init=False)
    _clear_streak: int = field(default=0, init=False)

    def reason_busy(self) -> str | None:
        """Why clipping must wait, or None when the machine is free."""
        reason = self._reason()
        if reason is not None:
            self._clear_streak = 0
            return reason
        self._clear_streak += 1
        if self._clear_streak < self.quiet_checks:
            return (f"waiting for a second quiet reading "
                    f"({self._clear_streak}/{self.quiet_checks})")
        return None

    def _reason(self) -> str | None:
        idle = self.input_idle()
        if idle == BLIND:
            return ("this process cannot see the operator's input (it is "
                    "not in the console session) — register `bta watch` as "
                    "a task that runs only when you are logged on")
        if idle is None:
            self._warn_once("input", "keyboard/mouse idle time unreadable; "
                                     "not gating on it")
        elif float(idle) < self.idle_after_s:
            return (f"operator active ({float(idle):.0f}s since last input, "
                    f"needs {self.idle_after_s:.0f}s)")

        state = self.notification()
        if state in _BUSY_NOTIFICATION_STATES:
            return f"{_BUSY_NOTIFICATION_STATES[state]} is on screen"

        gpu = self.gpu()
        if gpu is None:
            self._warn_once("gpu", "GPU state unreadable; not gating on it")
            return None
        if gpu["util_pct"] >= self.gpu_busy_pct:
            return f"GPU busy with something else ({gpu['util_pct']:.0f}%)"
        if gpu["decoder_pct"] >= self.decoder_busy_pct:
            return (f"something is playing video "
                    f"({gpu['decoder_pct']:.0f}% decoder)")
        if gpu["free_vram_gb"] < self.min_free_vram_gb:
            return (f"only {gpu['free_vram_gb']:.1f} GB VRAM free, "
                    f"need {self.min_free_vram_gb:.1f}")

        load = self.load()
        if load is None:
            self._warn_once("load", "CPU/RAM load unreadable; not gating on it")
            return None
        if load["cpu_pct"] >= self.cpu_busy_pct:
            return (f"something else is using the CPU "
                    f"({load['cpu_pct']:.0f}%)")
        if load["free_ram_gb"] < self.min_free_ram_gb:
            return (f"only {load['free_ram_gb']:.1f} GB RAM free, "
                    f"need {self.min_free_ram_gb:.1f}")
        return None

    def is_idle(self) -> bool:
        return self.reason_busy() is None

    def _warn_once(self, key: str, note: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            log.warning("idle.unmeasurable", signal=key, note=note)


@dataclass
class PreemptCheck:
    """Mid-job test: has the operator just come back?

    Deliberately NOT the full gate. A checkpoint runs between stages, when
    this pipeline's own models have unloaded but the GPU may still be
    settling from them, so reading utilisation there would pause the job
    on its own load. Fresh keyboard or mouse input is the signal that
    means "someone is here NOW".
    """

    #: Input within this many seconds means the operator is back.
    within_s: float = 60.0
    #: A job in flight also yields when someone else starts something
    #: heavy — a game launched from another machine's remote play, a
    #: training run kicked off by a scheduled task, an LLM loading.
    cpu_busy_pct: float = 55.0
    min_free_vram_gb: float = 4.0
    input_idle: Callable[[], float | str | None] = seconds_since_input
    notification: Callable[[], int | None] = notification_state
    load: Callable[[], dict[str, float] | None] = system_load
    gpu: Callable[[], dict[str, float] | None] = gpu_state

    def reason_busy(self) -> str | None:
        idle = self.input_idle()
        if idle == BLIND:
            return "operator input not visible from this session"
        if idle is not None and float(idle) < self.within_s:
            return f"operator came back ({float(idle):.0f}s since input)"
        state = self.notification()
        if state in _BUSY_NOTIFICATION_STATES:
            return f"{_BUSY_NOTIFICATION_STATES[state]} is on screen"
        load = self.load()
        if load is not None and load["cpu_pct"] >= self.cpu_busy_pct:
            return f"another program wants the CPU ({load['cpu_pct']:.0f}%)"
        gpu = self.gpu()
        # Deliberately NOT utilisation: between our stages the card may
        # still be settling from OUR last one. Memory someone else is
        # holding is theirs, and it is what stops a game from starting.
        if gpu is not None and gpu["free_vram_gb"] < self.min_free_vram_gb:
            return (f"another program is holding the GPU's memory "
                    f"({gpu['free_vram_gb']:.1f} GB free)")
        return None


class JobPreempted(BaseException):
    """Raised inside a running job when it must stop for the operator.

    BaseException, not Exception: the pipeline catches Exception in a
    dozen places to degrade gracefully, and "the owner came back and this
    job is halting" must not be swallowed by a stage's fallback.
    """


#: Set by whoever owns the GPU policy (the watch dispatcher) around the
#: call into the pipeline. A context variable rather than a parameter so a
#: manual `bta process` is byte-for-byte the command it always was.
_CHECKPOINT: contextvars.ContextVar[Callable[[str], None] | None] =     contextvars.ContextVar("clipforge_idle_checkpoint", default=None)


def set_checkpoint(fn: "Callable[[str], None] | None"):
    """Install the checkpoint for this context; returns the reset token."""
    return _CHECKPOINT.set(fn)


def reset_checkpoint(token) -> None:
    _CHECKPOINT.reset(token)


def idle_checkpoint(stage: str) -> None:
    """A place the pipeline may be paused, named for the log line.

    Call ONLY between stages, where the previous stage has unloaded its
    model: a pause inside a ``gpu_session`` would hand the operator back
    the compute while still holding their VRAM, which is the half that
    actually stops a game from starting.

    Does nothing at all unless a checkpoint is installed, so every manual
    command behaves exactly as before.
    """
    fn = _CHECKPOINT.get()
    if fn is not None:
        fn(stage)


def wait_until_idle(gate: IdleGate, stop: "Callable[[], bool]", *,
                    poll_s: float = 30.0,
                    sleep: Callable[[float], None] = time.sleep,
                    on_wait: Callable[[str], None] | None = None) -> bool:
    """Block until the gate opens. Returns False if ``stop()`` fired first.

    ``on_wait`` is told the reason each time the gate is found closed, so
    the caller can log a transition without this module owning the policy.
    """
    while True:
        if stop():
            return False
        reason = gate.reason_busy()
        if reason is None:
            return True
        if on_wait is not None:
            on_wait(reason)
        sleep(poll_s)
