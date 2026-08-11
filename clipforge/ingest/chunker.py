"""Live chunker — streamlink→ffmpeg segment pipeline (spec §S0, traps T2/T3).

The pipe is wired IN PYTHON (never ``shell=True``):

    Popen(streamlink --stdout <url> best, stdout=PIPE)
      → Popen(ffmpeg -i pipe:0 -c copy -f segment ... chunk_%05d.ts,
              stdin=<streamlink.stdout>)

with the parent closing its copy of the pipe handle immediately, so that
killing streamlink delivers EOF to ffmpeg's stdin and ffmpeg finalizes the
tail segment on its own (T3: the only clean shutdown mechanism on Windows,
where terminate == TerminateProcess and nothing graceful exists).

Lifecycle guarantees, each of which review demonstrated was missing before:

  * **Nothing escapes without teardown.** Everything after the spawn runs
    under ``try/finally``; a StateError from the DB, an ffprobe blowup, or
    a disk error can no longer orphan the pair. A half-spawned pipe (ffmpeg
    fails to start) kills the streamlink half on the way out.
  * **A hard parent kill still reaps the children**, via a Windows Job
    Object with KILL_ON_JOB_CLOSE (:mod:`clipforge.ingest.procguard`).
  * **Absolute media time is durable and gap-free.** Every closed segment
    is recorded *and* banks the session offset in ONE transaction
    (``bank_segment_and_offset``), so a crash mid-connect resumes on the
    correct timeline instead of restarting at zero. Quarantined and torn
    segments still ADVANCE the clock — dropping their duration silently
    shifted every later timestamp in the session.
  * **Segments are MPEG-TS while being written** (T2: byte-stream
    resilient) and are remuxed to faststart MP4 only once provably closed.

A segment is READY when the next segment appears (primary rule) or when its
size has been stable for ``ready_stable_s`` (secondary rule). A stalled tail
means the stream is dead upstream, so we kill the connection and let the
reconnect loop take over — but a stall-kill is reported back to ``run()``
so it does not escalate the reconnect backoff like a real failure would.

Everything external is injectable (popen, prober, clock, tool resolution) —
unit tests run the full state machine offline; integration tests run the
real pipe against a scripted fake streamlink and real ffmpeg.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from clipforge.errors import ClipForgeError, IngestError
from clipforge.ffmpeg import (CREATE_NO_WINDOW, MediaInfo, probe,
                              remux_ts_to_mp4, require_binary)
from clipforge.ingest.backoff import backoff_delay
from clipforge.ingest.procguard import ProcessGuard
from clipforge.ingest.runner import find_tool
from clipforge.log import get_logger
from clipforge.state import StateDB

log = get_logger(__name__)

#: A tail smaller than this is torn beyond use — discarded, loudly.
MIN_TAIL_BYTES = 188 * 64  # 64 TS packets

#: Grace given to ffmpeg to finalize the tail after streamlink dies (T3).
FFMPEG_FINALIZE_GRACE_S = 20.0

#: Media seconds a connect must capture to count as "healthy" and reset the
#: reconnect backoff. Deliberately small: review showed that requiring a
#: full 900 s segment meant a flaky-but-productive stream escalated its
#: backoff monotonically to the 300 s cap while still delivering media.
HEALTHY_CONNECT_S = 30.0


#: A connect that has produced NO segment at all by this point is not going
#: to. Without this the capture loop can spin forever with both halves
#: "alive" but no data flowing (streamlink negotiating a stream that never
#: arrives), and because no ConnectResult is ever returned, EVERY
#: termination guard — the three-strike rule, the monitor's is_live re-poll —
#: is bypassed. The channel is then wedged for the life of the process.
FIRST_SEGMENT_TIMEOUT_S = 120.0

#: Absolute ceiling on one connect regardless of progress. A capture that
#: outlives this is re-established rather than trusted indefinitely; the
#: session (and therefore absolute time) survives across the reconnect.
MAX_CONNECT_S = 6 * 3600.0

#: Hard ceiling on how often we may re-establish the pipe, PER CHANNEL.
#:
#: This is a STRUCTURAL invariant, deliberately replacing the health
#: thresholds that reviewers relocated three times running (0.04 s, 3 s,
#: 5-30 s, then above-healthy). Round 7 then proved a per-run() window was
#: itself a relocation: the three-strike rule ended the session, the
#: monitor re-entered run() with a FRESH window, and the storm continued
#: ACROSS sessions at ~61-82 spawns/hour. The rolling window therefore
#: lives in a ConnectLedger owned by the CALLER (the monitor keys one per
#: channel), surviving session hand-backs; run() creates a private one only
#: when standalone. Whatever the media numbers say, the process may not
#: spawn streamlink for a channel more often than this.
MAX_CONNECTS_PER_HOUR = 20


class ConnectLedger:
    """Rolling one-hour record of pipe spawns for ONE channel.

    Deliberately dumb: a list of clock readings plus a prune. The whole
    point is that it is keyed per channel and OUTLIVES any single session,
    so no session-lifecycle trick can reset it.
    """

    def __init__(self) -> None:
        self.times: list[float] = []

    def record(self, now: float) -> None:
        self.times.append(now)
        self.prune(now)

    def prune(self, now: float) -> None:
        self.times = [t for t in self.times if now - t < 3600.0]

    def count(self, now: float) -> int:
        self.prune(now)
        return len(self.times)

    #: Added to a throttled cooldown so the wait provably CLEARS the window.
    #: Waiting exactly `3600 - (now - oldest)` lands on the boundary, and in
    #: float that is 3599.999999999999x — still inside `now - t < 3600`. The
    #: old code recorded anyway, admitting a 21st spawn per sliding hour;
    #: re-checking instead live-locks on ~1e-13 s waits. The margin makes the
    #: boundary resolve in one wait, deterministically.
    COOLDOWN_MARGIN_S = 0.05

    def cooldown_s(self, now: float) -> float:
        """Seconds until the oldest in-window spawn ages out (0 if under)."""
        if self.count(now) < MAX_CONNECTS_PER_HOUR:
            return 0.0
        return (max(0.0, 3600.0 - (now - min(self.times)))
                + self.COOLDOWN_MARGIN_S)

#: A session is handed back to the monitor after this long no matter how it
#: is doing, so is_live is re-polled and channel-level policy applies. The
#: three-strike rule alone could hold a channel for 3 x MAX_CONNECT_S.
MAX_SESSION_S = 6 * 3600.0

#: ffmpeg input flags for HLS discontinuity tolerance (§S0). Twitch stitches
#: ads server-side; streamlink drops those segments, which leaves timestamp
#: discontinuities in the byte stream we copy. Without these, ffmpeg aborts
#: or emits non-monotonic-DTS garbage at every ad break.
HLS_TOLERANCE_FLAGS = ["-fflags", "+discardcorrupt+genpts", "-err_detect", "ignore_err"]


@dataclass(frozen=True)
class SegmentEvent:
    """Emitted exactly once per READY segment — the chunker's only output."""

    session_id: int
    seg_index: int
    path: Path
    abs_start_s: float
    duration_s: float
    #: The segment that immediately precedes this one ON THE TIMELINE, or
    #: None when the predecessor is missing/quarantined/unusable. T1's
    #: overlap MUST come from the true neighbour: reusing the last
    #: successfully-emitted segment across a hole would splice
    #: non-adjacent media together and mistime the window.
    prev_path: Path | None = None


@dataclass(frozen=True)
class ConnectResult:
    """Outcome of one connect, so ``run()`` can distinguish causes."""

    #: REAL, probed media seconds — the health/liveness signal. Estimated
    #: time for unprobeable segments must NEVER inflate this: doing so let a
    #: stream that died after 2 s score as a healthy 15-minute capture, which
    #: reset the backoff and the empty-connect counter forever (the loop
    #: never exited and the channel was never re-polled).
    captured_s: float
    #: Seconds the session clock advanced, INCLUDING estimates for media we
    #: could not probe. Used for absolute-time bookkeeping only.
    timeline_s: float = 0.0
    #: True when WE killed a healthy-looking pipe because its tail stalled.
    #: A stall is an upstream hiccup, not our failure — it must not escalate
    #: the reconnect backoff the way a spawn/network error does.
    stalled: bool = False
    #: True when any segment's duration had to be estimated.
    estimated: bool = False


@dataclass
class ChunkerConfig:
    segment_time_s: int = 900
    ready_stable_s: float = 20.0
    backoff_base_s: float = 5.0
    backoff_max_s: float = 300.0
    poll_interval_s: float = 1.0  # segment-dir scan cadence
    #: Remux closed .ts segments to faststart .mp4 (§S0/T2). The .ts is
    #: deleted on success — keeping both would double the footprint of
    #: 15-minute 1080p60 chunks for no benefit.
    remux_to_mp4: bool = True


def _resolve_tools() -> tuple[str, str]:
    """(streamlink, ffmpeg) absolute paths, or a typed IngestError.

    ``require_binary`` raises PreflightError, which is a sibling of
    IngestError, not a subclass — converting here keeps the monitor's
    ``except IngestError`` backoff path the single containment point for
    everything ingestion can throw.
    """
    streamlink = find_tool("streamlink")
    if streamlink is None:
        raise IngestError("streamlink not found - pip install streamlink")
    try:
        ffmpeg = require_binary("ffmpeg")
    except ClipForgeError as exc:
        raise IngestError(f"ffmpeg unavailable for the chunker: {exc}") from exc
    return str(streamlink), str(ffmpeg)


class ChunkerSession:
    """One live-capture session for one channel: reconnect loop + segment
    state machine. ``run()`` blocks — the monitor drives it in a thread."""

    def __init__(self, *, db: StateDB, chunks_root: Path, quarantine_dir: Path,
                 platform: str, handle: str, streamlink_args: list[str],
                 cfg: ChunkerConfig | None = None,
                 on_segment_ready: Callable[[SegmentEvent], None] | None = None,
                 disk_ok: Callable[[], bool] | None = None,
                 log_dir: Path | None = None,
                 popen: Callable[..., Any] = subprocess.Popen,
                 prober: Callable[[Path], MediaInfo] = probe,
                 clock: Callable[[], float] = time.monotonic,
                 resolve_tools: Callable[[], tuple[str, str]] | None = None) -> None:
        self.db = db
        self.chunks_root = Path(chunks_root)
        self.quarantine_dir = Path(quarantine_dir)
        self.platform = platform
        self.handle = handle
        self.streamlink_args = streamlink_args
        self.cfg = cfg or ChunkerConfig()
        self.on_segment_ready = on_segment_ready
        #: Checked inside the capture loop — §6 requires ingestion to pause
        #: on a low disk floor, and a broadcast lasts hours, so a per-tick
        #: check in the monitor alone never fires mid-capture.
        self.disk_ok = disk_ok
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self._popen = popen
        self._prober = prober
        self._clock = clock
        # Injectable so unit tests never need the real binaries installed.
        self._resolve_tools = resolve_tools or _resolve_tools

    # ------------------------------------------------------------------ api

    def run(self, stop: threading.Event, *, session_id: int | None = None,
            ledger: "ConnectLedger | None" = None) -> None:
        """Blocking reconnect loop; returns when ``stop`` is set or the
        stream is over (several CONSECUTIVE empty connects).

        ``session_id`` resumes an existing session (reconnecting to the same
        broadcast keeps one timeline); omit it to open a new one.
        ``ledger`` is the CHANNEL's connect ledger — pass the same object on
        every run() for a channel or the rate cap resets with the session.
        """
        sid = (session_id if session_id is not None
               else self.db.open_stream_session(self.platform, self.handle))
        log.info("chunker.session_open", platform=self.platform,
                 handle=self.handle, session_id=sid)
        attempt = 0
        empty_connects = 0
        session_started = self._clock()
        if ledger is None:
            ledger = ConnectLedger()  # standalone use only
        try:
            while not stop.is_set():
                # ADMISSION GATE. Both ceilings must hold at the INSTANT the
                # spawn is recorded, so they are re-checked in a loop after
                # every cooldown wait. A single wait-then-record leaked past
                # both: the rate cap admitted 21 spawns per sliding hour
                # (nothing re-read the ledger after the wait), and because
                # the cooldown was inserted AFTER the age check, a session
                # could sleep up to one full cooldown past MAX_SESSION_S and
                # then issue one guaranteed-sterile spawn (round-8 measured
                # overshoot: +3209.6 s, 11.7x the intended bound).
                expired = False
                while not stop.is_set():
                    if self._clock() - session_started >= MAX_SESSION_S:
                        # Hand back to the monitor so is_live is re-polled and
                        # channel-level policy (backoff, disk, config) applies.
                        # Without this the three-strike rule could hold a
                        # channel for 3 x MAX_CONNECT_S, monitor blind to it.
                        log.info("chunker.session_max_age", handle=self.handle,
                                 session_id=sid)
                        expired = True
                        break
                    # Cap BEFORE spawning, not merely between reconnect waits:
                    # the ledger outlives this session, so a hand-back/re-enter
                    # cycle cannot launder spawns past the cap.
                    pre_cooldown = ledger.cooldown_s(self._clock())
                    if pre_cooldown <= 0:
                        break
                    log.warning("chunker.reconnect_rate_limited",
                                handle=self.handle,
                                connects_last_hour=ledger.count(self._clock()),
                                cap=MAX_CONNECTS_PER_HOUR,
                                cooldown_s=round(pre_cooldown, 1))
                    stop.wait(pre_cooldown)
                if expired or stop.is_set():
                    break
                ledger.record(self._clock())
                try:
                    result = self._connect_once(
                        stop, sid,
                        deadline=session_started + MAX_SESSION_S)
                except IngestError as exc:
                    log.warning("chunker.connect_failed", handle=self.handle,
                                error=str(exc))
                    result = ConnectResult(captured_s=0.0)
                if stop.is_set():
                    break

                # NOTE: captured_s is REAL probed media only. Estimated
                # seconds (unprobeable segments) advance the timeline but
                # must never make a dying stream look productive — that is
                # what turned a 2-second failure into an endless loop.
                if result.captured_s >= HEALTHY_CONNECT_S:
                    # Productive connect: forget past failures entirely, and
                    # a stall that killed a productive pipe is an upstream
                    # hiccup rather than our failure — reconnect promptly.
                    attempt = 0
                    empty_connects = 0
                else:
                    # ANY sub-healthy connect is a strike. There is
                    # deliberately no middle band: every threshold I have
                    # tried between "zero" and "healthy" simply MOVED the
                    # permanent loop (0.04 s reset it, then 3 s did, then
                    # 5-30 s did). Three consecutive strikes end the session
                    # and hand control back to the monitor, whose is_live
                    # poll restarts a capture immediately if the channel is
                    # genuinely still live — so a merely-modest stream costs
                    # one poll interval, not a wedged channel.
                    empty_connects += 1
                    if empty_connects >= 3:
                        log.info("chunker.stream_over", handle=self.handle,
                                 session_id=sid,
                                 last_captured_s=round(result.captured_s, 2))
                        break
                delay = backoff_delay(attempt, base_s=self.cfg.backoff_base_s,
                                      cap_s=self.cfg.backoff_max_s,
                                      seed_key=f"{self.platform}:{self.handle}")
                attempt += 1
                log.info("chunker.reconnect_wait", handle=self.handle,
                         attempt=attempt, delay_s=round(delay, 1),
                         stalled=result.stalled,
                         connects_last_hour=ledger.count(self._clock()))
                stop.wait(delay)
        finally:
            self.db.close_stream_session(sid)
            log.info("chunker.session_closed", session_id=sid)

    # ------------------------------------------------------------- internals

    def _spawn_pipe(self, seg_dir: Path, start_index: int,
                    guard: ProcessGuard) -> tuple[Any, Any]:
        """Spawn streamlink→ffmpeg with the pipe wired per T3.

        If the ffmpeg half fails to start, the streamlink half is killed and
        its pipe closed here — review demonstrated that the previous code
        leaked a streamlink that then blocked forever on a full pipe.
        """
        streamlink, ffmpeg = self._resolve_tools()
        flags = {"creationflags": CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        sl_err = self._open_stderr_log()

        sl = self._popen([str(streamlink), *self.streamlink_args],
                         stdout=subprocess.PIPE, stderr=sl_err,
                         stdin=subprocess.DEVNULL, **flags)
        guard.assign(sl)
        try:
            ff = self._popen([
                str(ffmpeg), "-hide_banner", "-loglevel", "error",
                *HLS_TOLERANCE_FLAGS,
                "-i", "pipe:0", "-map", "0", "-c", "copy",
                "-f", "segment", "-segment_format", "mpegts",
                "-segment_time", str(self.cfg.segment_time_s),
                "-reset_timestamps", "1",
                "-segment_start_number", str(start_index),
                str(seg_dir / "chunk_%05d.ts"),
            ], stdin=sl.stdout, stderr=sl_err, stdout=subprocess.DEVNULL, **flags)
            guard.assign(ff)
        except BaseException:
            # Half-spawned pipe: never leave the first half running.
            try:
                sl.kill()
                sl.wait(timeout=5.0)
            except Exception:
                pass
            raise
        finally:
            # Parent must drop ITS copy of the read end: with it open, ffmpeg
            # would never see EOF after streamlink dies (T3 core mechanism).
            if sl.stdout is not None:
                try:
                    sl.stdout.close()
                except OSError:
                    pass
            if sl_err not in (None, subprocess.DEVNULL):
                try:
                    sl_err.close()
                except Exception:
                    pass
        return sl, ff

    def _open_stderr_log(self):
        """Both halves' diagnostics go to a per-channel log file.

        Discarding them (the previous DEVNULL) made a multi-day failure
        undebuggable and hid streamlink's own deprecation/plugin warnings.
        """
        if self.log_dir is None:
            return subprocess.DEVNULL
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in self.handle)
            path = self.log_dir / f"capture_{self.platform}_{safe}.log"
            # Append-forever leaks on multi-day runs (retention sweeps
            # chunks/tmp/quarantine, never logs/): one-deep rotation at
            # 32 MB keeps diagnostics without unbounded growth.
            try:
                if path.exists() and path.stat().st_size > 32 * 1024 * 1024:
                    os_replace_target = path.with_suffix(".log.1")
                    path.replace(os_replace_target)
            except OSError:
                pass  # rotation is best-effort; logging must not fail spawn
            return open(path, "ab", buffering=0)
        except OSError as exc:
            log.warning("chunker.stderr_log_unavailable", error=str(exc))
            return subprocess.DEVNULL

    def _connect_once(self, stop: threading.Event, session_id: int,
                      deadline: float | None = None) -> ConnectResult:
        """One connect: spawn the pipe, track segments until the stream drops
        or ``stop`` fires, finalize the tail, bank the media time.

        ``deadline`` (a clock reading) bounds this connect by the SESSION's
        remaining budget: checking session age only between connects made
        MAX_SESSION_S a soft bound of up to MAX_SESSION_S + MAX_CONNECT_S.
        """
        sess = self.db.get_session(session_id)
        if sess is None:
            raise IngestError(f"session {session_id} vanished from the DB")
        start_index = int(sess["next_segment"])
        base_offset = float(sess["base_offset_s"])
        seg_dir = (self.chunks_root / f"{self.platform}_{self.handle}"
                   / f"s{session_id:05d}")
        try:
            seg_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise IngestError(f"cannot create segment dir {seg_dir}: {exc}") from exc

        state = _ConnectState(base_offset=base_offset, start_index=start_index)
        guard = ProcessGuard()
        sl = ff = None
        stalled = False
        try:
            sl, ff = self._spawn_pipe(seg_dir, start_index, guard)
            stalled = self._capture_loop(stop, session_id, seg_dir, state,
                                         sl, ff, deadline=deadline)
        finally:
            # Runs on EVERY exit path — normal end, stall, stop, or any
            # exception from the DB/prober/filesystem. This is the T3
            # guarantee that review proved was missing.
            try:
                if sl is not None or ff is not None:
                    self._shutdown_pipe(sl, ff)
                    # Everything on disk is closed now: finalize all, tail too.
                    self._finalize_all(session_id, seg_dir, state)
                    # Persist the index cursor even for segments that banked
                    # no media (a discarded torn tail still consumed an
                    # index — reusing it would collide on reconnect).
                    try:
                        self.db.set_next_segment(session_id, state.next_index)
                    except ClipForgeError as exc:
                        log.error("chunker.next_segment_persist_failed",
                                  error=str(exc))
            finally:
                guard.close()  # kills anything still alive, always
        log.info("chunker.connect_done",
                 captured_s=round(state.real_media_s, 1),
                 timeline_s=round(state.timeline_s, 1),
                 next_segment=state.next_index, stalled=stalled,
                 estimated=state.estimated)
        return ConnectResult(captured_s=state.real_media_s,
                             timeline_s=state.timeline_s, stalled=stalled,
                             estimated=state.estimated)

    def _capture_loop(self, stop: threading.Event, session_id: int,
                      seg_dir: Path, state: "_ConnectState",
                      sl: Any, ff: Any, deadline: float | None = None) -> bool:
        """Poll the segment dir until the connection ends. Returns ``stalled``."""
        tail_size: int | None = None
        tail_stable_since: float | None = None
        registered_tail: int | None = None
        first_seen: dict[int, float] = {}
        connect_started = self._clock()

        while True:
            if stop.is_set():
                return False
            if sl.poll() is not None and ff.poll() is not None:
                return False  # both halves gone: connection over

            elapsed = self._clock() - connect_started
            if not first_seen and elapsed >= FIRST_SEGMENT_TIMEOUT_S:
                # No data has EVER arrived on this connect. Returning lets
                # the three-strike rule and the monitor's liveness poll do
                # their jobs; spinning here bypasses both entirely.
                log.warning("chunker.no_data", handle=self.handle,
                            waited_s=round(elapsed, 1))
                return False
            if elapsed >= MAX_CONNECT_S:
                log.info("chunker.connect_max_age", handle=self.handle,
                         elapsed_s=round(elapsed, 1), action="re-establish")
                return False
            if deadline is not None and self._clock() >= deadline:
                # The SESSION's budget is up mid-connect: end the connect so
                # run() sees the age check now, keeping the monitor-handback
                # latency at MAX_SESSION_S rather than that plus a connect.
                log.info("chunker.session_deadline_mid_connect",
                         handle=self.handle)
                return False
            if self.disk_ok is not None and not self.disk_ok():
                # §6: pause ingestion rather than fill the drive. Stopping
                # the capture is the only way to stop writing — the monitor
                # will not restart it until space is free again.
                log.warning("chunker.disk_floor_mid_capture", handle=self.handle,
                            action="ending capture")
                return False

            segs = _scan(seg_dir, state.start_index)
            if segs:
                max_idx = segs[-1][0]
                # Primary rule: every non-max segment is provably closed —
                # and closed BY THE MUXER, so its nominal length is exact.
                for idx, path in segs[:-1]:
                    self._finalize(session_id, idx, path, state, is_tail=False)
                if registered_tail != max_idx:
                    registered_tail = max_idx
                    tail_size, tail_stable_since = None, None
                # Remember when EACH segment first appeared. The LIFETIME is
                # computed at finalize time, not here: recording it in the
                # loop stored 0.0 on first sight and went stale by however
                # long the shutdown took, under-stating the bound.
                now_wall = self._clock()
                for idx, _path in segs:
                    if idx not in first_seen:
                        # It already existed when we first saw it, so credit
                        # one poll interval — the window in which it could
                        # have been created unobserved.
                        first_seen[idx] = now_wall - self.cfg.poll_interval_s
                state.first_seen_at.update(first_seen)
                # Secondary rule: stalled tail while the processes live.
                try:
                    size: int | None = segs[-1][1].stat().st_size
                except OSError:
                    size = None  # transient stat failure ≠ "stable"
                now = self._clock()
                if size is not None and size == tail_size:
                    if tail_stable_since is None:
                        tail_stable_since = now
                    elif now - tail_stable_since >= self._stall_kill_s():
                        log.warning("chunker.tail_stalled", handle=self.handle,
                                    stable_s=round(now - tail_stable_since, 1))
                        return True
                else:
                    tail_size, tail_stable_since = size, now
            stop.wait(self.cfg.poll_interval_s)

    def _stall_kill_s(self) -> float:
        """How long a frozen tail must stay frozen before we KILL the pipe.

        Deliberately longer than ``ready_stable_s``. The spec's 20-second
        size-stable rule answers "is this segment safe to hand to the DAG";
        this answers the different and more destructive question "is the
        stream dead". They are not the same threshold: ffmpeg's segment
        muxer flushes in ~256 KB steps, so a perfectly healthy stream below
        roughly 100 kbps leaves ``st_size`` unchanged for more than 20
        seconds at a time, and killing on that would sever a working
        broadcast every few minutes.

        Recorded deviation from a literal reading of §S0: the readiness rule
        keeps the specified 20 s; only the kill decision is more patient.
        """
        return max(60.0, self.cfg.ready_stable_s * 3.0)

    def _shutdown_pipe(self, sl: Any, ff: Any) -> None:
        """T3 teardown: kill streamlink; its EOF lets ffmpeg finalize."""
        if sl is not None and sl.poll() is None:
            sl.kill()  # hard kill is fine — its death IS the signal (EOF)
            try:
                sl.wait(timeout=5.0)
            except Exception:  # pragma: no cover
                log.error("chunker.zombie", proc="streamlink",
                          pid=getattr(sl, "pid", None))
        if ff is not None and ff.poll() is None:
            # Do NOT kill ffmpeg yet: post-EOF it writes the tail segment's
            # trailer. Grace period first; kill only if it wedges (T3).
            try:
                ff.wait(timeout=FFMPEG_FINALIZE_GRACE_S)
            except Exception:
                log.warning("chunker.ffmpeg_wedged", action="kill")
                try:
                    ff.kill()
                    ff.wait(timeout=5.0)
                except Exception:  # pragma: no cover
                    log.error("chunker.zombie", proc="ffmpeg",
                              pid=getattr(ff, "pid", None))

    def _finalize_all(self, session_id: int, seg_dir: Path,
                      state: "_ConnectState") -> None:
        """Post-shutdown sweep: everything on disk is closed now. The LAST
        segment is the tail (arbitrarily short); the rest were closed by the
        muxer at the configured segment length."""
        segs = _scan(seg_dir, state.start_index)
        for pos, (idx, path) in enumerate(segs):
            self._finalize(session_id, idx, path, state,
                           is_tail=(pos == len(segs) - 1))

    # ------------------------------------------------------------- finalize

    def _finalize(self, session_id: int, idx: int, path: Path,
                  state: "_ConnectState", *, is_tail: bool = False) -> None:
        """Close out one segment: probe, remux, record, emit.

        Every exit path advances the session clock, but ONLY probed media
        counts toward the connect's health. ``is_tail`` distinguishes a
        muxer-closed segment (nominal duration is exact) from the arbitrarily
        short final one (must be estimated from observed bitrate).
        """
        if idx in state.finalized:
            return
        state.finalized.add(idx)
        abs_start = state.abs_cursor

        try:
            size = path.stat().st_size
        except OSError:
            size = 0

        if size < MIN_TAIL_BYTES:
            # Torn beyond use: too small to contain meaningful media, so the
            # clock advance is genuinely ~0. It consumes its index though.
            log.warning("chunker.tail_discarded", seg_index=idx, size=size,
                        note="torn beyond use")
            try:
                path.unlink()
            except OSError:
                pass
            state.advance(idx, 0.0, real=False)
            state.last_good = None  # a hole: T1 must not bridge it
            return

        try:
            duration = float(self._prober(path).duration_s)
        except ClipForgeError as exc:
            duration = 0.0
            log.error("chunker.probe_failed", seg_index=idx, error=str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            duration = 0.0
            log.error("chunker.probe_error", seg_index=idx,
                      error=f"{type(exc).__name__}: {exc}")

        if duration <= 0.0:
            self._quarantine(session_id, idx, path, abs_start, state,
                             size=size, is_tail=is_tail)
            return

        state.note_probed(size, duration)
        media = self._remux_if_configured(path)
        state.advance(idx, duration, real=True)
        prev = state.last_good
        state.last_good = media
        try:
            self.db.bank_segment_and_offset(
                session_id, idx, media, abs_start, duration,
                next_segment=state.next_index)
        except ClipForgeError as exc:
            # The media is on disk and the clock advanced in memory; a DB
            # hiccup must not orphan the pipe (we are inside the caller's
            # try/finally) nor stop the capture.
            log.error("chunker.segment_record_failed", seg_index=idx,
                      error=str(exc))
            return

        log.info("chunker.segment_ready", seg_index=idx,
                 abs_start_s=round(abs_start, 2), duration_s=round(duration, 2),
                 path=str(media))
        if self.on_segment_ready is not None:
            try:
                self.on_segment_ready(SegmentEvent(
                    session_id=session_id, seg_index=idx, path=media,
                    abs_start_s=abs_start, duration_s=duration,
                    prev_path=prev))
            except Exception as exc:
                log.error("chunker.on_segment_failed", seg_index=idx,
                          error=f"{type(exc).__name__}: {exc}")

    def _remux_if_configured(self, path: Path) -> Path:
        """T2/§S0: remux the CLOSED .ts to faststart .mp4, then drop the .ts.

        Failure degrades to the .ts (still perfectly playable) — a remux
        problem must never cost us the recording.
        """
        if not self.cfg.remux_to_mp4 or path.suffix.lower() != ".ts":
            return path
        dest = path.with_suffix(".mp4")
        try:
            remux_ts_to_mp4(path, dest)
        except ClipForgeError as exc:
            log.warning("chunker.remux_failed", path=str(path), error=str(exc),
                        action="keeping .ts")
            return path
        try:
            path.unlink()
        except OSError:
            pass  # retention sweep will get it
        return dest

    def _quarantine(self, session_id: int, idx: int, path: Path,
                    abs_start: float, state: "_ConnectState", *,
                    size: int, is_tail: bool) -> None:
        """Move an unprobeable segment aside under a COLLISION-FREE name.

        Every session produces a ``chunk_00000.ts``; flat naming meant
        session 2 silently overwrote session 1's evidence, and the DB
        recorded the pre-move path, which pointed at nothing.

        Duration is ESTIMATED, never assumed: a muxer-closed segment really
        does hold ``segment_time_s``, but a tail holds an arbitrary amount,
        so it is estimated from this connect's observed bitrate. Either way
        the estimate advances the timeline WITHOUT counting as real captured
        media, and the session is flagged in the DB so consumers can see
        that its absolute times are approximate from here on.
        """
        dest = path
        try:
            self.quarantine_dir.mkdir(parents=True, exist_ok=True)
            dest = self.quarantine_dir / f"s{session_id:05d}_{path.name}"
            path.replace(dest)
        except OSError as exc:
            log.warning("chunker.quarantine_move_failed", path=str(path),
                        error=str(exc))
            dest = path

        nominal = float(self.cfg.segment_time_s)
        if is_tail:
            duration, estimated = state.estimate_duration(
                size, nominal, wall_s=state.wall_age(idx, self._clock()))
        else:
            duration, estimated = nominal, True  # muxer-closed: exact length
        state.advance(idx, duration, real=False, estimated=estimated)
        state.last_good = None  # a hole: T1 must not bridge it
        log.warning("chunker.segment_quarantined", seg_index=idx,
                    path=str(dest), estimated_duration_s=round(duration, 2),
                    is_tail=is_tail,
                    note="absolute time for later segments is ESTIMATED")
        try:
            if estimated:
                self.db.mark_timeline_estimated(session_id)
            self.db.bank_segment_and_offset(
                session_id, idx, dest, abs_start, duration,
                next_segment=state.next_index, status="quarantined")
        except ClipForgeError as exc:
            log.error("chunker.quarantine_record_failed", seg_index=idx,
                      error=str(exc))


@dataclass
class _ConnectState:
    """Mutable bookkeeping for one connect.

    TWO clocks, deliberately separate:
      * ``timeline_s`` moves the absolute-time cursor and may include
        ESTIMATES for media we could not probe;
      * ``real_media_s`` counts only media we actually measured, and is the
        sole health signal. Conflating them let phantom estimated seconds
        masquerade as a productive capture.
    """

    base_offset: float
    start_index: int
    timeline_s: float = 0.0
    real_media_s: float = 0.0
    estimated: bool = False

    def __post_init__(self) -> None:
        self.finalized: set[int] = set()
        self.next_index: int = self.start_index
        #: Bytes/seconds of PROBED media, for estimating unprobeable tails.
        self.probed_bytes: int = 0
        self.probed_seconds: float = 0.0
        #: Path of the last usable segment, for T1's true predecessor.
        self.last_good: Path | None = None
        #: Per-index CLOCK READING at which each segment was first observed.
        #: The lifetime is derived at finalize time (see ``wall_age``), which
        #: matters twice: one shared field held whatever index the loop last
        #: saw, and a lifetime snapshotted during the loop is 0.0 on first
        #: sight and stale by the length of the shutdown afterwards.
        self.first_seen_at: dict[int, float] = {}

    @property
    def abs_cursor(self) -> float:
        return self.base_offset + self.timeline_s

    def advance(self, idx: int, duration: float, *, real: bool,
                estimated: bool = False) -> None:
        self.timeline_s += duration
        if real:
            self.real_media_s += duration
        self.next_index = max(self.next_index, idx + 1)
        self.estimated = self.estimated or estimated

    def wall_age(self, idx: int, now: float) -> float | None:
        """Seconds since segment ``idx`` was first observed, evaluated NOW."""
        first = self.first_seen_at.get(idx)
        return None if first is None else max(0.0, now - first)

    def note_probed(self, size: int, duration: float) -> None:
        self.probed_bytes += size
        self.probed_seconds += duration

    def estimate_duration(self, size: int, nominal: float, *,
                          wall_s: float | None = None) -> tuple[float, bool]:
        """Best-effort duration for an UNPROBEABLE TAIL segment.

        A non-tail segment was closed by the muxer at ``segment_time``, so
        the nominal value is exact and this is not used. A TAIL is
        arbitrarily short — assuming nominal there was the inverted bug: a
        stream dying after 2 s banked 15 minutes of phantom media.

        Two independent bounds, and we take the smaller:
          * bitrate: ``size / observed_bytes_per_second`` from THIS connect;
          * wall clock: a live tail cannot contain more media than the real
            time it spent being written. On variable-bitrate media the
            bitrate estimate alone is unreliable (a high-motion tail
            estimated 689 s for 4 s of media, and the nominal clamp is far
            too loose to catch it at a 900 s segment length), so the
            wall-clock bound is what actually keeps this honest.

        With nothing to learn from, refuse to guess and return 0.
        """
        bounds = [nominal]
        if wall_s is not None and wall_s >= 0:
            bounds.append(wall_s)
        if self.probed_bytes > 0 and self.probed_seconds > 0:
            bytes_per_s = self.probed_bytes / self.probed_seconds
            if bytes_per_s > 0:
                bounds.append(size / bytes_per_s)
        else:
            if wall_s is None:
                return 0.0, True
        return max(0.0, min(bounds)), True


def _scan(seg_dir: Path, start_index: int) -> list[tuple[int, Path]]:
    """This connect's segment files, sorted by index. Only ``.ts`` — an
    already-remuxed ``.mp4`` is finished business."""
    out: list[tuple[int, Path]] = []
    try:
        candidates = sorted(seg_dir.glob("chunk_*.ts"))
    except OSError:
        return out
    for p in candidates:
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            continue
        if idx >= start_index:
            out.append((idx, p))
    return out
