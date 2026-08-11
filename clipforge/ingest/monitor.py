"""Channel monitor — async poll loop, per-platform backoff, disk guard.

One asyncio task per enabled channel (spec §6: ingest pool is
network-bound, high concurrency). Each tick:

  * housekeeping first — retention sweep, then the disk guard: below the
    free-space floor ingestion PAUSES (loudly) rather than filling the
    drive (§6);
  * twitch/kick: ``is_live`` poll → on live, start a ChunkerSession and
    babysit it until it returns;
  * youtube: pending VOD ids (newly discovered + previously-failed
    requeues) → download each → hand the file to the DAG seam.

**Everything blocking runs off the event loop.** ``is_live`` shells out to
streamlink with a 30 s timeout; review demonstrated that calling it inline
serialized every channel behind the slowest probe. Blocking work goes to a
dedicated ThreadPoolExecutor sized from ``orchestration.ingest_concurrency``
plus the live-channel count, so a long-running capture (which holds its
thread for the whole broadcast) can never starve VOD ingestion.

Failure containment: any IngestError backs off THAT channel with the same
deterministic-jitter schedule the chunker uses; Kick returning ``None``
("cannot determine", T4) is treated as offline and never raises. A crash of
one channel's task never touches another's.

**T1 wiring.** The DAG never sees a bare chunk: each ready segment is joined
with the trailing ``overlap_s`` of its predecessor via the concat demuxer,
and the resulting virtual window (with absolute start corrected by the real,
keyframe-quantized overlap) is what reaches ``on_media``.

CP1 scope note: ``on_media`` is the seam the CP5 orchestrator plugs the DAG
into; at CP1 it defaults to a log line, so ``clipforge watch`` records,
chunks, and windows — producing files and DB rows — without processing them.
"""

from __future__ import annotations

import asyncio
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from clipforge.config import AppConfig, ChannelSpec, Watchlist
from clipforge.errors import ClipForgeError, IngestError
from clipforge.ingest import kick, twitch, youtube
from clipforge.ingest.backoff import backoff_delay
from clipforge.ingest.chunker import (ChunkerConfig, ChunkerSession,
                                      ConnectLedger, SegmentEvent)
from clipforge.ingest.overlap import build_virtual_window
from clipforge.ingest.retention import sweep_retention
from clipforge.log import get_logger
from clipforge.paths import Workspace
from clipforge.state import StateDB

log = get_logger(__name__)

#: Seconds between retention sweeps. The sweep is cheap but hits the
#: filesystem; once a minute keeps well ahead of 15-minute chunks.
RETENTION_INTERVAL_S = 60.0

#: A channel that goes live again within this window is treated as the SAME
#: broadcast and RESUMES its session, so a brief CDN outage does not restart
#: absolute media time at zero in a fresh session (which would break T1's
#: cross-chunk dedup at the seam).
SESSION_RESUME_WINDOW_S = 300.0

#: A capture session shorter than this achieved nothing: the channel claims
#: to be live but yields no usable media. Without backing such a channel off,
#: the three-strike rule inside the chunker plus a 60 s poll interval produce
#: a sustained reconnect storm (roughly 150 connects/hour, forever).
UNPRODUCTIVE_SESSION_S = 60.0


def disk_allows(root: Path, floor_gb: float) -> bool:
    """The §6 disk guard: False ⇒ pause ingestion this tick."""
    try:
        free_gb = shutil.disk_usage(root).free / 1024 ** 3
    except OSError as exc:
        log.error("monitor.disk_probe_failed", error=str(exc))
        return False  # cannot verify ⇒ err on the side of not writing
    if free_gb < floor_gb:
        log.warning("monitor.disk_floor", free_gb=round(free_gb, 1),
                    floor_gb=floor_gb, action="ingestion paused")
        return False
    return True


@dataclass
class ChannelMonitor:
    """Owns the poll loops for every enabled channel."""

    cfg: AppConfig
    db: StateDB
    ws: Workspace
    watchlist: Watchlist
    #: DAG seam: called with (path, abs_start_s) for every ready media file.
    on_media: Callable[[Path, float], None] | None = None
    #: Injectable platform probes (tests): defaults are the real modules.
    twitch_is_live: Callable[..., bool] = twitch.is_live
    kick_is_live: Callable[..., bool | None] = kick.is_live
    discover: Callable[..., list[str]] = youtube.pending_vod_ids
    download: Callable[..., Path] = youtube.download_vod
    make_chunker: Callable[..., ChunkerSession] | None = None

    _stop: threading.Event = field(default_factory=threading.Event)
    _executor: ThreadPoolExecutor | None = field(default=None, init=False)
    #: Housekeeping gets its OWN thread. Sharing the ingest pool meant the
    #: retention sweep queued behind live captures that hold their threads
    #: for the whole broadcast — the exact starvation the separate task was
    #: supposed to fix.
    _housekeeper: ThreadPoolExecutor | None = field(default=None, init=False)
    #: Consecutive unproductive capture sessions per channel — drives the
    #: anti-reconnect-storm backoff.
    _barren: dict[tuple[str, str], int] = field(default_factory=dict, init=False)
    #: Per-channel connect ledgers for the STRUCTURAL rate cap. Owned here —
    #: not inside run() — precisely so the window survives session
    #: hand-backs; a per-run() window let the storm continue across
    #: sessions at 3x the cap (round-7 finding).
    _ledgers: dict[tuple[str, str], "ConnectLedger"] = field(
        default_factory=dict, init=False)

    # ------------------------------------------------------------------ api

    async def run(self) -> None:
        """Run all channel loops until :meth:`stop`. Deterministic startup
        order (Watchlist.enabled_sorted); one task per channel."""
        channels = self._authorized_channels()
        if not channels:
            log.warning("monitor.no_channels",
                        note="channels.toml has no enabled entries")
            return

        # A live capture occupies its thread for the entire broadcast, so the
        # pool must fit every live channel PLUS the network-bound VOD work;
        # sharing the default executor let captures starve VOD ingestion.
        live_channels = sum(1 for c in channels if c.platform != "youtube")
        workers = max(2, live_channels + max(1, self.cfg.orchestration.ingest_concurrency))
        self._executor = ThreadPoolExecutor(max_workers=workers,
                                            thread_name_prefix="cf-ingest")
        self._housekeeper = ThreadPoolExecutor(max_workers=1,
                                               thread_name_prefix="cf-keep")
        log.info("monitor.start", channels=len(channels), workers=workers)
        tasks = [asyncio.create_task(self._channel_loop(ch),
                                     name=f"monitor:{ch.platform}:{ch.handle}")
                 for ch in channels]
        # Housekeeping is its OWN task: hanging it off a channel loop meant
        # it stopped for the entire duration of every live capture — i.e.
        # exactly when chunks accumulate and reclamation matters most.
        tasks.append(asyncio.create_task(self._housekeeping_loop(),
                                         name="monitor:housekeeping"))
        try:
            await asyncio.gather(*tasks)
        finally:
            self._stop.set()
            # Do not block shutdown on in-flight work: the stop event has
            # already told every worker to wind down, and the chunker's job
            # object reaps any child that outlives us.
            for pool in (self._executor, self._housekeeper):
                if pool is not None:
                    pool.shutdown(wait=False, cancel_futures=True)

    def stop(self) -> None:
        self._stop.set()

    async def _to_thread(self, fn: Callable, *args):
        """Run blocking work on OUR executor, not the loop's default one."""
        loop = asyncio.get_running_loop()
        if self._executor is None:  # direct-call paths in tests
            return await asyncio.to_thread(fn, *args)
        return await loop.run_in_executor(self._executor, fn, *args)

    def _authorized_channels(self) -> list[ChannelSpec]:
        """Authorization Law boundary: ONLY operator-listed, enabled
        channels; Kick additionally requires its feature flag (T4)."""
        chans = self.watchlist.enabled_sorted()
        if not self.cfg.ingest.kick_enabled:
            for c in [c for c in chans if c.platform == "kick"]:
                log.info("monitor.kick_disabled", handle=c.handle,
                         note="set ingest.kick_enabled=true to opt in (T4)")
            chans = [c for c in chans if c.platform != "kick"]
        return chans

    # ---------------------------------------------------------- channel loop

    async def _channel_loop(self, ch: ChannelSpec) -> None:
        attempt = 0
        while not self._stop.is_set():
            try:
                if disk_allows(self.ws.root, self.cfg.disk.free_floor_gb):
                    if ch.platform == "youtube":
                        await self._to_thread(self._tick_youtube, ch)
                    else:
                        await self._tick_live(ch)
                attempt = 0
            except IngestError as exc:
                attempt += 1
                delay = backoff_delay(attempt,
                                      base_s=self.cfg.ingest.backoff_base_s,
                                      cap_s=self.cfg.ingest.backoff_max_s,
                                      seed_key=f"monitor:{ch.platform}:{ch.handle}")
                log.warning("monitor.channel_error", platform=ch.platform,
                            handle=ch.handle, error=str(exc), attempt=attempt,
                            retry_in_s=round(delay, 1))
                await self._sleep(delay)
                continue
            except Exception as exc:  # containment: never kill sibling loops
                attempt += 1
                delay = backoff_delay(attempt,
                                      base_s=self.cfg.ingest.backoff_base_s,
                                      cap_s=self.cfg.ingest.backoff_max_s,
                                      seed_key=f"monitor:{ch.platform}:{ch.handle}")
                log.error("monitor.channel_crashed", platform=ch.platform,
                          handle=ch.handle,
                          error=f"{type(exc).__name__}: {exc}", attempt=attempt,
                          retry_in_s=round(delay, 1))
                # Back off unexpected failures too: review showed an untyped
                # error produced a fixed 60 s hot loop for days.
                await self._sleep(delay)
                continue
            log.debug("monitor.tick", platform=ch.platform, handle=ch.handle)
            await self._sleep(self.cfg.ingest.poll_interval_s)

    async def _housekeeping_loop(self) -> None:
        """Retention sweep on its own cadence — the disk guard's other half.

        Independent of the channel loops precisely because those block for
        the whole of a live broadcast: reclamation has to keep running while
        a capture is writing, which is when the drive actually fills.
        """
        loop = asyncio.get_running_loop()
        while not self._stop.is_set():
            try:
                # Dedicated pool: the ingest pool's threads are held for the
                # duration of live captures, so scheduling here would put
                # reclamation behind exactly the work that fills the disk.
                await loop.run_in_executor(
                    self._housekeeper,
                    lambda: sweep_retention(
                        self.db, self.ws,
                        retention_hours=self.cfg.disk.retention_hours))
            except Exception as exc:  # housekeeping must never kill the run
                log.error("monitor.retention_failed",
                          error=f"{type(exc).__name__}: {exc}")
            await self._sleep(RETENTION_INTERVAL_S)

    async def _sleep(self, seconds: float) -> None:
        """Interruptible sleep: wakes early when stop() fires."""
        step = 0.5
        waited = 0.0
        while waited < seconds and not self._stop.is_set():
            await asyncio.sleep(min(step, seconds - waited))
            waited += step

    # -------------------------------------------------------------- youtube

    def _tick_youtube(self, ch: ChannelSpec) -> None:
        """Blocking; runs on the ingest executor.

        Everything here is scoped to THIS channel's handle — a platform-wide
        pending query made every channel return the union of all channels'
        ids, so concurrent loops downloaded the same VOD twice into
        different folders.
        """
        pending = self.discover(self.db, ch.handle, self.cfg.ingest.playlist_end)
        for vid in pending:
            if self._stop.is_set():
                return
            if not disk_allows(self.ws.root, self.cfg.disk.free_floor_gb):
                # Status stays 'seen'/'failed': the requeue sweep offers this
                # id again on a later tick once space is free.
                break
            dest = self.ws.chunks / "youtube" / ch.handle.lstrip("@")
            self.db.bump_video_attempt("youtube", ch.handle, vid)
            try:
                path = self.download(vid, dest, stop=self._stop)
            except IngestError as exc:
                self.db.set_video_status("youtube", ch.handle, vid, "failed")
                log.warning("monitor.vod_download_failed", video_id=vid,
                            handle=ch.handle, error=str(exc),
                            note="will be retried")
                continue
            self.db.set_video_status("youtube", ch.handle, vid, "downloaded")
            # A VOD is a complete recording: no predecessor, no overlap.
            self._emit_media(path, abs_start_s=0.0)

    # ----------------------------------------------------------------- live

    async def _tick_live(self, ch: ChannelSpec) -> None:
        if ch.platform == "twitch":
            live = await self._to_thread(self.twitch_is_live, ch.handle)
        else:  # kick — T4: None means "cannot determine" ⇒ offline
            live = bool(await self._to_thread(self.kick_is_live, ch.handle))
        if not live:
            return

        log.info("monitor.channel_live", platform=ch.platform, handle=ch.handle)
        key = (ch.platform, ch.handle)
        session = self._build_chunker(ch)
        # Resume the recent session for this channel if there is one: a
        # broadcast that briefly dropped is ONE timeline, and starting a new
        # session would restart absolute time at zero mid-broadcast.
        resume_id = self.db.resumable_session(ch.platform, ch.handle,
                                              SESSION_RESUME_WINDOW_S)
        if resume_id is not None:
            self.db.reopen_session(resume_id)
            log.info("monitor.session_resumed", handle=ch.handle,
                     session_id=resume_id)
        # Babysit the blocking session on the ingest executor; this channel's
        # loop resumes polling only after the session ends (stream over).
        # The connect ledger is THIS CHANNEL's, shared across every session.
        ledger = self._ledgers.setdefault(key, ConnectLedger())
        started = time.monotonic()
        await self._to_thread(lambda: session.run(self._stop,
                                                  session_id=resume_id,
                                                  ledger=ledger))
        # A session that ends almost immediately means the channel reports
        # live but yields nothing usable. Re-polling at the normal cadence
        # would reconnect ~3 times per poll interval indefinitely — a
        # reconnect storm against the platform. Back off THIS channel
        # instead, on the same deterministic schedule the chunker uses.
        if time.monotonic() - started < UNPRODUCTIVE_SESSION_S:
            self._barren[key] = self._barren.get(key, 0) + 1
            delay = backoff_delay(self._barren[key],
                                  base_s=self.cfg.ingest.backoff_base_s,
                                  cap_s=self.cfg.ingest.backoff_max_s,
                                  seed_key=f"barren:{ch.platform}:{ch.handle}")
            log.warning("monitor.barren_session", handle=ch.handle,
                        consecutive=self._barren[key],
                        backoff_s=round(delay, 1))
            await self._sleep(delay)
        else:
            self._barren.pop(key, None)

    def _build_chunker(self, ch: ChannelSpec) -> ChunkerSession:
        if self.make_chunker is not None:  # test seam
            return self.make_chunker(ch)
        ing = self.cfg.ingest
        if ch.platform == "twitch":
            sl_args = twitch.chunker_args(ch.handle, ing.quality,
                                          disable_ads=ing.twitch_disable_ads)
        else:
            sl_args = kick.chunker_args(ch.handle, ing.quality)
        return ChunkerSession(
            db=self.db, chunks_root=self.ws.chunks,
            quarantine_dir=self.ws.quarantine, platform=ch.platform,
            handle=ch.handle, streamlink_args=sl_args,
            cfg=ChunkerConfig(
                segment_time_s=ing.segment_time_s,
                ready_stable_s=ing.segment_ready_stable_s,
                backoff_base_s=ing.backoff_base_s,
                backoff_max_s=ing.backoff_max_s,
            ),
            on_segment_ready=self._on_segment,
            disk_ok=lambda: disk_allows(self.ws.root, self.cfg.disk.free_floor_gb),
            log_dir=self.ws.logs,
        )

    # ---------------------------------------------------------------- output

    def _on_segment(self, ev: SegmentEvent) -> None:
        """T1: hand the DAG a virtual window, never a bare chunk.

        The predecessor comes from the EVENT, not from a "last emitted"
        cache: a quarantined or unrecorded segment leaves a hole, and
        splicing across it would join non-adjacent media and mistime the
        window. The chunker reports ``prev_path=None`` across a hole, which
        degrades this window to the bare chunk — correct, not lossy.
        """
        prev = ev.prev_path
        overlap_s = float(self.cfg.ingest.overlap_s)
        try:
            window = build_virtual_window(
                chunk=ev.path, chunk_abs_start_s=ev.abs_start_s,
                prev_chunk=prev, overlap_s=overlap_s,
                out_dir=self.ws.tmp / f"windows_s{ev.session_id:05d}")
        except ClipForgeError as exc:
            # build_virtual_window degrades internally; this catches the
            # residual (probe of the chunk itself failing). Emit the bare
            # chunk rather than losing the segment.
            log.warning("monitor.window_failed", path=str(ev.path),
                        error=str(exc), action="emitting bare chunk")
            self._emit_media(ev.path, ev.abs_start_s)
            return
        log.info("monitor.window_ready", path=str(window.path),
                 abs_start_s=round(window.abs_start_s, 2),
                 overlap_s=round(window.overlap_s, 2))
        self._emit_media(window.path, window.abs_start_s)

    def emit_recovered(self, path: Path, abs_start_s: float) -> None:
        """Emit a segment recovered by startup reconciliation.

        Recovered media goes through the SAME T1 path as live segments —
        emitting it bare would silently give the DAG a different input shape
        depending on whether a crash happened. The predecessor is unknown
        after a crash, so these windows carry no overlap; that is recorded
        rather than faked.
        """
        self._on_segment(SegmentEvent(
            session_id=-1, seg_index=-1, path=path, abs_start_s=abs_start_s,
            duration_s=0.0, prev_path=None))

    def _emit_media(self, path: Path, abs_start_s: float) -> None:
        if self.on_media is not None:
            try:
                self.on_media(path, abs_start_s)
            except Exception as exc:  # DAG seam bug ≠ ingestion death
                log.error("monitor.on_media_failed", path=str(path),
                          error=f"{type(exc).__name__}: {exc}")
        else:
            log.info("monitor.media_ready", path=str(path),
                     abs_start_s=round(abs_start_s, 2),
                     note="DAG processing lands at CP2+")
