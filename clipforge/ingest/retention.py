"""Retention + startup reconciliation — the disk guard's other half (§6).

Why this is CP1 and not CP5: the free-space floor "pauses ingestion rather
than filling the drive", but a pause with nothing ever freeing space is a
PERMANENT stop. A guard without retention fails quietly, days later, in an
unattended process.

Three jobs:

  * :func:`sweep_retention` — reclaim aged media EVERYWHERE it is produced:
    tracked chunk segments, T1 virtual windows and their tails under
    ``tmp/``, quarantine, and **stranded** media that has no DB row at all
    (a crash or a failed DB write leaves files nothing else can find).
    Rows whose file is gone are actually deleted, not merely counted.
  * :func:`reconcile_sessions` — the ingestion analogue of CP0's
    ``.partial`` discard: sessions left open by a crash are rescanned, their
    unregistered media recovered onto the correct absolute timeline, and the
    session closed. Media still being written is left alone (a killed
    process's orphan may still be growing), and both ``.ts`` and remuxed
    ``.mp4`` are recognized.
  * :func:`workspace_lock` — single-instance guard. Reconciliation decides a
    session is dead purely from ``ended_at IS NULL``, so a second process
    starting while the first is still recording would "recover" (and close)
    a LIVE session. One writer per workspace, enforced by an OS-level
    exclusive file lock that the OS releases even on a hard kill.
"""

from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from clipforge.errors import ClipForgeError, IngestError, StateError
from clipforge.ffmpeg import MediaInfo, probe
from clipforge.log import get_logger
from clipforge.paths import Workspace
from clipforge.state import StateDB

log = get_logger(__name__)

#: Media that is safe to delete once old enough. 'ready' is included with a
#: caveat — see the module note in VERIFICATION.md: until the DAG marks
#: segments 'processed' (CP5), a CP1-only deployment ages out media that was
#: never consumed. Operators running CP1 alone should raise retention_hours.
_DELETABLE_STATUSES = ("processed", "ready", "quarantined", "deleted")

#: Segment forms the chunker produces. Both must be recognized: a closed
#: segment is remuxed to .mp4 and its .ts deleted, so a crash between those
#: steps can leave either form on disk.
MEDIA_GLOBS = ("chunk_*.ts", "chunk_*.mp4")

#: Everything else that lands under chunks/ and is NOT a tracked segment:
#: downloaded VODs and yt-dlp's own partial/format files. Without these, a
#: YouTube-only deployment had no reclamation path at all — the disk guard
#: would eventually pause ingestion permanently.
UNTRACKED_MEDIA_GLOBS = ("yt_*.*", "*.part", "*.ytdl", "*.f[0-9]*.*")

#: A file is "settled" (safe to recover/delete) only if it has not been
#: modified for this long. Guards against an orphaned writer still appending.
SETTLE_SECONDS = 30.0

#: Below this a segment cannot hold meaningful media (64 TS packets), so it
#: is credited ZERO timeline even when it "should" have been muxer-closed.
#: Mirrors the chunker's MIN_TAIL_BYTES; without it a 0-byte crash artifact
#: was credited a full segment_time_s.
MIN_SEGMENT_BYTES = 188 * 64


@dataclass(frozen=True)
class RetentionReport:
    files_deleted: int = 0
    bytes_freed: int = 0
    rows_pruned: int = 0


# --------------------------------------------------------------------------
# single-instance lock
# --------------------------------------------------------------------------


@contextmanager
def workspace_lock(ws: Workspace) -> Iterator[bool]:
    """Exclusive lock on the workspace. Yields True if acquired.

    Uses an OS advisory lock (msvcrt on Windows, fcntl elsewhere) rather than
    a PID file: the OS drops the lock when the process dies **however** it
    dies, so a hard kill cannot leave a stale lock that blocks restart.
    """
    ws.root.mkdir(parents=True, exist_ok=True)
    lock_path = ws.root / "clipforge.lock"
    handle = None
    acquired = False
    try:
        # Opening can itself fail when another holder has the file locked
        # (Windows raises PermissionError from open in some sharing modes),
        # so acquisition covers the open AND the lock: any OSError simply
        # means "someone else has it".
        try:
            handle = open(lock_path, "a+b")
            # Seek to 0 BEFORE locking: msvcrt.locking() locks a byte range
            # at the CURRENT position, and "a+b" starts at end-of-file — so
            # each process would lock a different byte (after the previous
            # one wrote its PID) and every instance would "acquire" it.
            handle.seek(0)
            if sys.platform == "win32":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - POSIX
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            acquired = True
            # The handle is in APPEND mode, so this lands at EOF regardless
            # of position — which is what we want: byte 0 (the locked byte)
            # is written once when the file is created and never rewritten,
            # and each run appends a human-readable owner line.
            handle.write(f"pid={os.getpid()}\n".encode("ascii"))
            handle.flush()
        except OSError as exc:
            log.debug("workspace_lock.unavailable", path=str(lock_path),
                      error=str(exc))
            acquired = False
        yield acquired
    finally:
        if handle is not None:
            if acquired:
                try:
                    if sys.platform == "win32":
                        import msvcrt

                        handle.seek(0)  # unlock the same byte we locked
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:  # pragma: no cover - POSIX
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()


# --------------------------------------------------------------------------
# retention
# --------------------------------------------------------------------------


def _settled(path: Path, now: float, settle_s: float | None = None) -> bool:
    """True if nothing has written to ``path`` recently.

    ``settle_s=None`` resolves SETTLE_SECONDS at CALL time. A def-time
    default binds the value at import, which silently makes the constant
    unadjustable — the same late-binding trap that made a CP0 test's probe
    stub ineffective.
    """
    threshold = SETTLE_SECONDS if settle_s is None else settle_s
    try:
        return (now - path.stat().st_mtime) >= threshold
    except OSError:
        return False


def _await_settled(path: Path, *, settle_s: float | None,
                   sleep: Callable[[float], None]) -> bool:
    """Bounded wait for ``path`` to stop growing. True if it went stable.

    Reconciliation runs at boot while ``watch`` holds the EXCLUSIVE workspace
    lock, so no ClipForge process can be writing this file: an "unsettled"
    file here is nearly always the crashed run's final segment, whose mtime
    is merely younger than SETTLE_SECONDS. Skipping it left the session OPEN,
    and R7-3 made an open session unresumable — so the FRESHEST crash, the
    exact case the resume window exists for, was the one that could not
    resume. R7-4 and R7-3 cancelled each other out.

    Waiting is bounded by poll COUNT, not by a clock deadline: reconcile's
    ``now`` is injectable and tests pin it, so a deadline loop would spin
    forever under a frozen clock. A genuinely growing file never goes
    size-stable, so the stop-the-walk path still fires for it.
    """
    threshold = SETTLE_SECONDS if settle_s is None else settle_s
    if threshold <= 0:
        return False
    step = min(0.5, threshold)
    try:
        last = path.stat().st_size
    except OSError:
        return False
    stable = 0
    for _ in range(max(1, int(threshold / step))):
        sleep(step)
        try:
            size = path.stat().st_size
        except OSError:
            return False
        if size != last:
            stable, last = 0, size
            continue
        stable += 1
        if stable >= 2:
            log.info("reconcile.settled_by_wait", path=str(path), size=size)
            return True
    return False


def _unlink(path: Path) -> int:
    """Delete, returning bytes freed (0 if it could not be removed)."""
    try:
        size = path.stat().st_size
        path.unlink()
        return size
    except OSError as exc:
        log.debug("retention.delete_skipped", path=str(path), error=str(exc))
        return 0


def sweep_retention(db: StateDB, ws: Workspace, *, retention_hours: float,
                    now: Callable[[], float] = time.time) -> RetentionReport:
    """Reclaim aged media and prune dead rows.

    Deletion is by FILE MTIME, not DB timestamps: mtime is the ground truth
    for "how old is this media" and survives DB loss.
    """
    t_now = now()
    cutoff = t_now - retention_hours * 3600.0
    deleted = freed = pruned = 0

    # ---- 1. tracked segments -------------------------------------------
    for status in _DELETABLE_STATUSES:
        try:
            rows = list(db.segments_with_status(status))
        except StateError as exc:
            log.error("retention.db_scan_failed", status=status, error=str(exc))
            continue
        for row in rows:
            path = Path(row["path"])
            if not path.exists():
                # The row's media is gone: actually DELETE the row. Counting
                # it as "pruned" while leaving it in place (the previous
                # behaviour) meant every later sweep re-counted it forever.
                try:
                    db.delete_segment_row(int(row["session_id"]),
                                          int(row["seg_index"]))
                    pruned += 1
                except StateError as exc:
                    log.warning("retention.row_delete_failed", error=str(exc))
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            size = _unlink(path)
            if size:
                deleted += 1
                freed += size
                try:
                    db.set_segment_status(int(row["session_id"]),
                                          int(row["seg_index"]), "deleted")
                except StateError as exc:
                    log.warning("retention.status_update_failed", error=str(exc))

    # ---- 2. T1 virtual windows + their tails ----------------------------
    # These are DERIVED media (chunk + previous tail) written to tmp/. They
    # are never tracked in `segments`, so nothing else can reclaim them —
    # and they grow at roughly the chunk rate, doubling the footprint.
    if ws.tmp.exists():
        for path in sorted(ws.tmp.rglob("*")):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            size = _unlink(path)
            if size:
                deleted += 1
                freed += size

    # ---- 3. quarantine --------------------------------------------------
    if ws.quarantine.exists():
        for path in sorted(ws.quarantine.glob("*")):
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime >= cutoff:
                    continue
            except OSError:
                continue
            size = _unlink(path)
            if size:
                deleted += 1
                freed += size

    # ---- 4. STRANDED media (no DB row at all) ---------------------------
    # A crash, or a segment whose DB write failed while the capture
    # deliberately continued, leaves media that no row references. Without
    # this pass it is an unbounded, unreclaimable leak.
    try:
        tracked = {os.path.normcase(p) for p in db.all_segment_paths()}
    except StateError as exc:
        log.error("retention.tracked_scan_failed", error=str(exc))
        tracked = set()
    if ws.chunks.exists():
        for pattern in MEDIA_GLOBS + UNTRACKED_MEDIA_GLOBS:
            for path in sorted(ws.chunks.rglob(pattern)):
                if not path.is_file():
                    continue
                if os.path.normcase(str(path)) in tracked:
                    continue
                try:
                    if path.stat().st_mtime >= cutoff:
                        continue
                except OSError:
                    continue
                if not _settled(path, t_now):
                    continue  # someone may still be writing it
                size = _unlink(path)
                if size:
                    deleted += 1
                    freed += size
                    log.info("retention.stranded_media_removed", path=str(path))

    if deleted or pruned:
        log.info("retention.sweep", files_deleted=deleted,
                 gb_freed=round(freed / 1024 ** 3, 3), rows_pruned=pruned,
                 retention_hours=retention_hours)
    return RetentionReport(files_deleted=deleted, bytes_freed=freed,
                           rows_pruned=pruned)


# --------------------------------------------------------------------------
# reconciliation
# --------------------------------------------------------------------------


def reconcile_sessions(db: StateDB, ws: Workspace, *,
                       on_segment: Callable[[Path, float], None] | None = None,
                       prober: Callable[[Path], MediaInfo] = probe,
                       settle_s: float | None = None,
                       now: Callable[[], float] = time.time,
                       segment_time_s: float = 900.0,
                       sleep: Callable[[float], None] = time.sleep) -> int:
    """Close sessions left open by a crash, recovering their media.

    For each session with ``ended_at IS NULL``: rescan its segment directory,
    register + emit any media the DB does not already know, then close the
    session. Absolute start times continue from the session's banked offset
    plus the durations recovered here.

    Safety rules learned the hard way:
      * a file that is still being written is SKIPPED (an orphaned recorder
        may outlive its parent) — registering it would hand the DAG a
        truncated clip and freeze its duration at the partial value;
      * a segment that cannot be probed still ADVANCES the cursor, so the
        segments after it are not silently re-timed;
      * both ``.ts`` and remuxed ``.mp4`` are recognized.

    Total by design: a failure on one session never blocks startup.
    """
    recovered = 0
    try:
        open_sessions = db.open_sessions()
    except StateError as exc:
        log.error("reconcile.scan_failed", error=str(exc))
        return 0

    for sess in open_sessions:
        sid = int(sess["id"])
        seg_dir = (ws.chunks / f"{sess['platform']}_{sess['handle']}"
                   / f"s{sid:05d}")
        try:
            rows = list(db.segments_for_session(sid))
        except StateError as exc:
            log.warning("reconcile.rows_failed", session_id=sid, error=str(exc))
            continue

        log.info("reconcile.session", session_id=sid, dir=str(seg_dir),
                 known_rows=len(rows))
        result = _recover_dir(db, sid, seg_dir, rows,
                              on_segment=on_segment, prober=prober,
                              settle_s=settle_s, now=now,
                              segment_time_s=segment_time_s,
                              banked_base=float(sess["base_offset_s"]),
                              banked_next=int(sess["next_segment"]),
                              sleep=sleep)
        recovered += result.registered

        # Progress writeback FROM DURABLE STATE ONLY: re-read the rows and
        # take the maxima. Earlier versions carried an in-memory cursor into
        # this write, so a boot whose row writes failed still advanced
        # base_offset_s — and every failing boot COMPOUNDED it (45→90→…).
        # Recomputing from rows makes reconcile idempotent: same rows in,
        # same progress out, however many times a broken boot repeats.
        try:
            fresh_rows = db.segments_for_session(sid)
            if fresh_rows:
                base = max(float(r["abs_start_s"]) + float(r["duration_s"] or 0.0)
                           for r in fresh_rows)
                nxt = max(int(r["seg_index"]) for r in fresh_rows) + 1
                db.set_session_progress(sid, base, nxt)
        except StateError as exc:
            log.error("reconcile.progress_writeback_failed",
                      session_id=sid, error=str(exc))

        if result.newest_media_mtime > 0:
            # Freshness for the resume window comes from real media age.
            # The segment-close stamp alone is too coarse: it lands once per
            # segment (900 s), coarser than the 300 s resume window, so a
            # crash mid-segment looked stale when it was seconds old.
            try:
                db.freshen_last_media(sid, result.newest_media_mtime)
            except StateError as exc:
                log.warning("reconcile.freshen_failed", session_id=sid,
                            error=str(exc))

        if result.unsettled or result.failed_register:
            # NOT fully recovered — leave the session OPEN so the next boot
            # retries. This also prevents overwrite: an open session is
            # never resumed by the monitor, so a new broadcast gets a fresh
            # session directory instead of ffmpeg restarting its numbering
            # on top of the unregistered files.
            log.warning("reconcile.session_left_open", session_id=sid,
                        unsettled=result.unsettled,
                        failed_register=result.failed_register,
                        note="will retry on the next boot")
            continue
        try:
            # Marked so resume judges it on last_media_at, not on the
            # ended_at stamp reconciliation writes "now".
            db.close_stream_session(sid, by_reconcile=True)
        except StateError as exc:
            log.warning("reconcile.close_failed", session_id=sid, error=str(exc))
    return recovered


def _scan_media(seg_dir: Path) -> list[tuple[int, Path]]:
    """(index, path) for every segment file, .ts and .mp4, sorted by index.
    If both forms exist for one index (crash mid-remux) prefer the .mp4."""
    found: dict[int, Path] = {}
    for pattern in MEDIA_GLOBS:
        for path in seg_dir.glob(pattern):
            try:
                idx = int(path.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
            if idx not in found or path.suffix == ".mp4":
                found[idx] = path
    return sorted(found.items())


@dataclass
class _RecoverResult:
    #: Rows written, INCLUDING estimated/quarantined ones. The operator
    #: banner keys off this: reporting only probeable recoveries told the
    #: operator "nothing happened" while rows were written.
    registered: int = 0
    #: Segments actually emitted to the DAG (probeable ones only).
    emitted: int = 0
    #: Settled files whose row write FAILED — the session must stay open.
    failed_register: int = 0
    #: Files still being written — the session must stay open.
    unsettled: int = 0
    #: Newest media mtime seen — feeds the session's last_media_at, which is
    #: what the resume window judges (segment-close stamps are too coarse:
    #: they land once per 900 s, coarser than the 300 s resume window).
    newest_media_mtime: float = 0.0


def _recover_dir(db: StateDB, sid: int, seg_dir: Path,
                 rows: list[Any], *,
                 on_segment: Callable[[Path, float], None] | None,
                 prober: Callable[[Path], MediaInfo],
                 settle_s: float | None, now: Callable[[], float],
                 segment_time_s: float, banked_base: float = 0.0,
                 banked_next: int = 0,
                 sleep: Callable[[float], None] = time.sleep) -> _RecoverResult:
    """Idempotent timeline reconstruction over ROWS ∪ FILES.

    Design rules, each the scar of a refuted round:

      * The walk is over the UNION of DB rows and disk files, in index
        order. A ROW is authoritative for its own position wherever its
        media now lives — keying the walk on disk files alone meant a
        quarantined segment (file moved away) never re-anchored the cursor
        and its timeline slot was silently reused.
      * The cursor seeds at 0.0 — a session's timeline starts at zero by
        construction. Seeding with ``base_offset_s`` (the timeline END)
        stamped a recovered index-0 segment at the end of the broadcast.
      * An UNSETTLED file STOPS the walk: everything after it depends on
        its duration, so placing later files this boot would mistime them.
        The session stays open; the next boot resumes from the same rows.
      * Nothing here writes session progress — the caller recomputes it
        from ROWS after the walk. That is what makes a boot with failing
        row writes repeatable instead of compounding the cursor.
    """
    result = _RecoverResult()
    files = dict(_scan_media(seg_dir)) if seg_dir.is_dir() else {}
    row_by_idx: dict[int, Any] = {int(r["seg_index"]): r for r in rows
                                  if r["status"] != "recording"}
    # The session's banked progress is a VIRTUAL anchor: "everything below
    # banked_next ends at banked_base". It matters when retention has pruned
    # the rows a walk would otherwise anchor on — without it, a later
    # recovered file was placed at 0.0 instead of after the pruned media.
    # It applies whether or not the anchor index's MEDIA still exists.
    # Requiring the file to be absent suppressed the anchor in the ordinary
    # case — retention prunes ROWS and FILES independently, so a surviving
    # file whose row was pruned is common — and the whole surviving suffix
    # was then re-timed from 0.0: the exact mistiming the anchor prevents.
    # The anchor's own file is dropped from the walk so it is not placed
    # twice; banked progress already states where it ends.
    anchor_idx = banked_next - 1
    if anchor_idx >= 0 and anchor_idx not in row_by_idx:
        row_by_idx[anchor_idx] = {"seg_index": anchor_idx,
                                  "abs_start_s": banked_base,
                                  "duration_s": 0.0,
                                  "status": "ready"}
        files.pop(anchor_idx, None)
    indices = sorted(set(files) | set(row_by_idx))
    if not indices:
        return result
    t_now = now()

    # Bitrate + nominal learned from probeable siblings, and their typical
    # SIZE: an unprobeable file far smaller than its probeable siblings is
    # a connect's TAIL. Reconnected sessions have one tail per connect —
    # not only at the end of the directory — so "last file" alone cannot
    # identify tails, and crediting a mid-session tail a full nominal
    # fabricated minutes of timeline (round-7 measurement of R6-7).
    probed_bytes = probed_seconds = 0.0
    observed_durs: list[float] = []
    probed_sizes: list[int] = []
    for idx in sorted(files):
        path = files[idx]
        try:
            dur = float(prober(path).duration_s)
            if dur > 0:
                size = path.stat().st_size
                probed_bytes += size
                probed_seconds += dur
                observed_durs.append(dur)
                probed_sizes.append(size)
        except (ClipForgeError, OSError):
            continue
    bytes_per_s = (probed_bytes / probed_seconds) if probed_seconds > 0 else 0.0
    # What the muxer ACTUALLY produced here beats the current config's
    # segment_time_s: recovered media may predate a config change.
    observed_nominal = max(observed_durs) if len(observed_durs) > 1 else 0.0
    typical_size = (sorted(probed_sizes)[len(probed_sizes) // 2]
                    if probed_sizes else 0)

    cursor = 0.0
    for idx in indices:
        row = row_by_idx.get(idx)
        if row is not None:
            # Authoritative re-anchor, wherever the media lives now.
            cursor = float(row["abs_start_s"]) + float(row["duration_s"] or 0.0)
            continue
        path = files[idx]
        try:
            result.newest_media_mtime = max(result.newest_media_mtime,
                                            path.stat().st_mtime)
        except OSError:
            pass
        if not _settled(path, t_now, settle_s) and not _await_settled(
                path, settle_s=settle_s, sleep=sleep):
            log.warning("reconcile.skipped_unsettled", path=str(path),
                        note="file still changing; stopping this session's "
                             "walk - later placements would depend on it")
            result.unsettled += 1
            break

        try:
            duration = float(prober(path).duration_s)
        except ClipForgeError as exc:
            log.warning("reconcile.unprobeable", path=str(path), error=str(exc))
            duration = 0.0

        estimated = False
        if duration <= 0.0:
            try:
                size = path.stat().st_size
            except OSError:
                size = 0
            if size < MIN_SEGMENT_BYTES:
                # Too small to hold meaningful media (0-byte post-power-loss
                # artifact): occupied ~no time. This size floor is
                # load-bearing — without it such a file was credited a full
                # segment_time_s, fabricating 15 minutes PER FILE.
                duration = 0.0
            elif bytes_per_s > 0:
                # MEASURED bitrate beats a nominal constant for every
                # unprobeable file, tail or not: a full-length segment's
                # size/bitrate already lands at nominal, while a partial
                # one lands at its real length. Gating this on a 0.6x size
                # threshold meant any tail in the upper 40% of sizes was
                # still credited a whole segment_time_s — the fabrication
                # the size test was added to remove, just moved uphill.
                duration = size / bytes_per_s
            elif idx == indices[-1]:
                # A trailing file with NO bitrate reference anywhere in the
                # directory: credit nothing rather than invent 15 minutes.
                duration = 0.0
            else:
                # Muxer-closed mid-directory segment and not one probeable
                # sibling to learn a bitrate from: nominal is the only
                # estimate available, and a gap here would mistime the rest.
                duration = observed_nominal or float(segment_time_s)
            estimated = True
            try:
                db.mark_timeline_estimated(sid)
            except StateError:
                pass

        status = "quarantined" if estimated else "ready"
        try:
            db.record_closed_segment(sid, idx, path, cursor, duration,
                                     status=status)
        except StateError as exc:
            # Not durable => not emitted, and the caller keeps the session
            # OPEN (so no resume can overwrite this file). The in-memory
            # advance keeps LATER placements this boot consistent; the
            # durable progress is recomputed from rows either way.
            log.warning("reconcile.register_failed", path=str(path),
                        error=str(exc))
            result.failed_register += 1
            cursor += duration
            continue
        log.info("reconcile.recovered_segment", session_id=sid, seg_index=idx,
                 abs_start_s=round(cursor, 2), duration_s=round(duration, 2),
                 estimated=estimated)
        result.registered += 1
        if not estimated and on_segment is not None:
            try:
                on_segment(path, cursor)
                result.emitted += 1
            except Exception as exc:
                log.error("reconcile.emit_failed", path=str(path),
                          error=f"{type(exc).__name__}: {exc}")
        cursor += duration
    return result
