"""SQLite (WAL) job store — jobs, stages, artifacts, ingest bookkeeping.

Why SQLite/WAL:
  * Single-process asyncio orchestrator + WAL = readers never block the writer.
  * Survives SIGKILL: WAL replay restores the last committed transaction, which
    pairs with the Resumability Law ("lose at most one in-flight stage").

Concurrency model: connections are cheap; this class opens ONE connection and
serializes writes behind a threading.Lock. All methods are synchronous — the
asyncio layer calls them directly (they are sub-millisecond) or via
``asyncio.to_thread`` for scans. ``busy_timeout`` guards the rare case of an
external inspector holding the file.

Schema versioning: ``PRAGMA user_version`` gates migrations. Never ALTER in
place without bumping.
"""

from __future__ import annotations

import functools
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, TypeVar

from clipforge.errors import StateError

#: v2: seen_videos is keyed per CHANNEL, not per platform — a platform-wide
#: pending query made every YouTube channel return the union of all channels'
#: ids, so concurrent loops downloaded the same VOD into different folders.
#: v2 also adds stream_sessions.timeline_estimated.
SCHEMA_VERSION = 3

_F = TypeVar("_F", bound=Callable[..., Any])


def _guarded(fn: _F) -> _F:
    """Wrap a StateDB method so every sqlite failure surfaces TYPED.

    busy_timeout exhaustion, disk-I/O errors, and use-after-close are the
    sqlite errors a multi-day unattended run actually hits; orchestration
    policy matches on :class:`StateError`, never on raw ``sqlite3.*``.
    """

    @functools.wraps(fn)
    def wrapper(self: "StateDB", *args: Any, **kwargs: Any) -> Any:
        try:
            return fn(self, *args, **kwargs)
        except sqlite3.Error as exc:
            raise StateError(
                f"StateDB.{fn.__name__} failed on {self._path.name}: {exc}"
            ) from exc

    return wrapper  # type: ignore[return-value]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,              -- 'vod' | 'live_chunk' | 'clip'
    key         TEXT NOT NULL UNIQUE,       -- stable natural key (dedup anchor)
    status      TEXT NOT NULL DEFAULT 'pending',
    payload     TEXT NOT NULL DEFAULT '{}', -- JSON, small; big data lives in artifacts
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS stage_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      INTEGER NOT NULL REFERENCES jobs(id),
    stage       TEXT NOT NULL,              -- 's1_transcribe' ...
    cache_key   TEXT NOT NULL,
    status      TEXT NOT NULL,              -- 'running' | 'done' | 'failed'
    artifact    TEXT,                       -- path of the produced artifact
    error       TEXT,
    started_at  REAL NOT NULL,
    finished_at REAL
);
CREATE INDEX IF NOT EXISTS ix_stage_runs_job ON stage_runs(job_id, stage);

-- Content-addressed artifact registry: THE resume lookup.
CREATE TABLE IF NOT EXISTS artifacts (
    cache_key   TEXT PRIMARY KEY,
    stage       TEXT NOT NULL,
    stage_version TEXT NOT NULL,
    path        TEXT NOT NULL,
    created_at  REAL NOT NULL
);

-- YouTube VOD dedup: never re-download a known id (spec §S0).
-- Keyed by CHANNEL: two channels may legitimately both list a video, and a
-- channel's tick must only ever return its OWN pending ids.
CREATE TABLE IF NOT EXISTS seen_videos (
    platform    TEXT NOT NULL,
    handle      TEXT NOT NULL,
    video_id    TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'seen',  -- 'seen'|'downloaded'|'processed'|'failed'
    first_seen  REAL NOT NULL,
    -- Download attempts so far: the requeue sweep retries 'seen'/'failed'
    -- ids until this hits its cap (a permanently-broken VOD must not pin
    -- the ingest loop forever).
    attempts    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (platform, handle, video_id)
);

-- Live-session bookkeeping: absolute media time survives reconnects (spec §S0).
CREATE TABLE IF NOT EXISTS stream_sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    platform      TEXT NOT NULL,
    handle        TEXT NOT NULL,
    started_at    REAL NOT NULL,
    ended_at      REAL,
    -- Media seconds already recorded before the current connect (sum of
    -- closed-segment durations from prior connects in this session).
    base_offset_s REAL NOT NULL DEFAULT 0,
    next_segment  INTEGER NOT NULL DEFAULT 0,
    -- 1 once any segment's duration had to be ESTIMATED (unprobeable
    -- media). Downstream must treat this session's absolute times as
    -- approximate from that point on — the flag is persisted precisely so
    -- consumers can see it, rather than living in a process-local variable.
    timeline_estimated INTEGER NOT NULL DEFAULT 0,
    -- 1 when a crash left this session open and STARTUP RECONCILIATION
    -- closed it. Such a session must never be resumed: reconciliation
    -- stamps ended_at with "now", so an ancient crashed broadcast would
    -- otherwise fall inside the resume window and today's unrelated stream
    -- would be appended to it, sharing one session id and one timeline.
    closed_by_reconcile INTEGER NOT NULL DEFAULT 0,
    -- Wall-clock time of the most recent segment write. This, NOT ended_at,
    -- is what says how old a crash-recovered broadcast really is:
    -- reconciliation stamps ended_at with "now" long after the fact.
    last_media_at REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS segments (
    session_id    INTEGER NOT NULL REFERENCES stream_sessions(id),
    seg_index     INTEGER NOT NULL,
    path          TEXT NOT NULL,
    abs_start_s   REAL NOT NULL,     -- absolute stream-relative start time
    duration_s    REAL,              -- filled when the segment closes
    status        TEXT NOT NULL DEFAULT 'recording',  -- 'recording'|'ready'|'processed'|'quarantined'
    PRIMARY KEY (session_id, seg_index)
);
"""


class StateDB:
    """Thread-safe wrapper over one SQLite connection in WAL mode."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        try:
            self._conn = sqlite3.connect(self._path, check_same_thread=False)
        except sqlite3.Error as exc:
            raise StateError(f"Cannot open state DB {self._path}: {exc}") from exc
        try:
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._migrate()
        except StateError:
            # Schema mismatch: close BEFORE raising, or the leaked handle
            # (pinned by the traceback) blocks the "move the DB aside" fix
            # the error message itself recommends (Windows file locking).
            self._conn.close()
            raise
        except sqlite3.Error as exc:
            self._conn.close()
            raise StateError(f"Cannot open state DB {self._path}: {exc}") from exc

    # ------------------------------------------------------------- lifecycle

    #: Forward migrations, applied in order. Additive-only: each entry is a
    #: list of statements that upgrade FROM the keyed version to key+1.
    #: Without these an existing workspace DB became permanently unopenable
    #: on any schema bump, discarding all session and VOD history.
    _MIGRATIONS: dict[int, list[str]] = {
        2: ["ALTER TABLE stream_sessions ADD COLUMN "
            "closed_by_reconcile INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE stream_sessions ADD COLUMN "
            "last_media_at REAL NOT NULL DEFAULT 0"],
    }

    def _migrate(self) -> None:
        with self._lock, self._conn:
            (ver,) = self._conn.execute("PRAGMA user_version").fetchone()
            if ver == 0:
                self._conn.executescript(_SCHEMA)
                self._conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                return
            while ver < SCHEMA_VERSION and ver in self._MIGRATIONS:
                for stmt in self._MIGRATIONS[ver]:
                    try:
                        self._conn.execute(stmt)
                    except sqlite3.OperationalError as exc:
                        # Intermediate dev builds stamped version numbers
                        # with part of the next schema already present; an
                        # additive ALTER that finds its column simply skips.
                        # Anything else is a real migration failure.
                        if "duplicate column" not in str(exc).lower():
                            raise
                ver += 1
                self._conn.execute(f"PRAGMA user_version={ver}")
            if ver != SCHEMA_VERSION:
                raise StateError(
                    f"State DB schema v{ver} != code v{SCHEMA_VERSION}; "
                    "no migration path - move the DB aside or upgrade code."
                )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "StateDB":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------- jobs

    @_guarded
    def upsert_job(self, kind: str, key: str, payload: dict[str, Any] | None = None) -> int:
        """Insert if new, return existing id otherwise. Idempotent by ``key``."""
        now = time.time()
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT INTO jobs(kind, key, payload, created_at, updated_at) "
                "VALUES(?,?,?,?,?) ON CONFLICT(key) DO NOTHING",
                (kind, key, json.dumps(payload or {}, sort_keys=True), now, now),
            )
            row = self._conn.execute("SELECT id FROM jobs WHERE key=?", (key,)).fetchone()
        return int(row["id"])

    @_guarded
    def set_job_status(self, job_id: int, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute("UPDATE jobs SET status=?, updated_at=? WHERE id=?",
                               (status, time.time(), job_id))

    @_guarded
    def get_job(self, key: str) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute("SELECT * FROM jobs WHERE key=?", (key,)).fetchone()

    # ------------------------------------------------------------- stage runs

    @_guarded
    def stage_started(self, job_id: int, stage: str, cache_key: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO stage_runs(job_id, stage, cache_key, status, started_at) "
                "VALUES(?,?,?,'running',?)", (job_id, stage, cache_key, time.time()))
        return int(cur.lastrowid)

    @_guarded
    def stage_finished(self, run_id: int, *, artifact: str | None = None,
                       error: str | None = None) -> None:
        status = "done" if error is None else "failed"
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stage_runs SET status=?, artifact=?, error=?, finished_at=? WHERE id=?",
                (status, artifact, error, time.time(), run_id))

    @_guarded
    def recent_jobs(self, limit: int = 20) -> list[sqlite3.Row]:
        """Most recently touched jobs, newest first.

        Read side of ``upsert_job``. The control API had been issuing its
        own SQL for columns that do not exist (``job_id``, ``chunk_id``,
        ``stage``) against a database file that does not exist
        (``state.db``; the real one is ``state.sqlite3``), inside a bare
        ``except`` that returned ``[]`` — so it reported an idle pipeline
        unconditionally. Queries belong here, next to the schema.
        """
        with self._lock:
            return list(self._conn.execute(
                "SELECT id, kind, key, status, payload, created_at, updated_at "
                "FROM jobs ORDER BY updated_at DESC LIMIT ?", (limit,)))

    @_guarded
    def stage_runs_for(self, job_id: int) -> list[sqlite3.Row]:
        """Every stage attempt for one job, oldest first (execution order)."""
        with self._lock:
            return list(self._conn.execute(
                "SELECT id, job_id, stage, cache_key, status, artifact, error, "
                "started_at, finished_at FROM stage_runs WHERE job_id=? "
                "ORDER BY started_at ASC, id ASC", (job_id,)))

    @_guarded
    def recent_stage_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        """Latest stage attempts across all jobs, newest first."""
        with self._lock:
            return list(self._conn.execute(
                "SELECT id, job_id, stage, cache_key, status, artifact, error, "
                "started_at, finished_at FROM stage_runs "
                "ORDER BY started_at DESC, id DESC LIMIT ?", (limit,)))

    # ------------------------------------------------------------- artifacts

    @_guarded
    def record_artifact(self, cache_key: str, stage: str, stage_version: str,
                        path: Path | str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts(cache_key, stage, stage_version, path, created_at) "
                "VALUES(?,?,?,?,?)", (cache_key, stage, stage_version, str(path), time.time()))

    @_guarded
    def lookup_artifact(self, cache_key: str, stage: str) -> Path | None:
        """The resume fast-path: cache_key hit ⇒ stage is a no-op.

        ``stage`` is part of the lookup contract (DAG Law): even though the
        key format embeds the stage name, filtering here guarantees a stage
        can never be handed another stage's artifact — belt AND suspenders.
        The registry row is only trusted if the file still exists — retention
        may have deleted it, in which case the stage legitimately re-runs.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT path FROM artifacts WHERE cache_key=? AND stage=?",
                (cache_key, stage)).fetchone()
        if row is None:
            return None
        p = Path(row["path"])
        return p if p.exists() else None

    # ------------------------------------------------------------- VOD dedup

    @_guarded
    def mark_video_seen(self, platform: str, handle: str, video_id: str) -> bool:
        """Returns True if this id was NEW for this CHANNEL."""
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO seen_videos(platform, handle, video_id, first_seen) "
                "VALUES(?,?,?,?) ON CONFLICT(platform, handle, video_id) DO NOTHING",
                (platform, handle, video_id, time.time()))
        return cur.rowcount == 1

    @_guarded
    def set_video_status(self, platform: str, handle: str, video_id: str,
                         status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE seen_videos SET status=? "
                "WHERE platform=? AND handle=? AND video_id=?",
                (status, platform, handle, video_id))

    @_guarded
    def videos_needing_download(self, platform: str, handle: str,
                                max_attempts: int = 5) -> list[sqlite3.Row]:
        """The requeue sweep's query: ids this CHANNEL saw but never landed.

        'seen' = discovered, download not attempted or interrupted (disk
        guard, SIGINT). 'failed' = attempted and errored — retried until
        ``attempts`` hits the cap, then left alone so one permanently-broken
        VOD cannot pin the loop forever. Scoped by handle so two channels
        never race the same id. Deterministic order (§3.2).
        """
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM seen_videos WHERE platform=? AND handle=? "
                "AND status IN ('seen','failed') AND attempts < ? "
                "ORDER BY first_seen, video_id",
                (platform, handle, max_attempts)).fetchall()

    @_guarded
    def bump_video_attempt(self, platform: str, handle: str,
                           video_id: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE seen_videos SET attempts = attempts + 1 "
                "WHERE platform=? AND handle=? AND video_id=?",
                (platform, handle, video_id))

    # ------------------------------------------------------------- live sessions

    @_guarded
    def open_stream_session(self, platform: str, handle: str) -> int:
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO stream_sessions(platform, handle, started_at) VALUES(?,?,?)",
                (platform, handle, time.time()))
        return int(cur.lastrowid)

    @_guarded
    def close_stream_session(self, session_id: int, *,
                             by_reconcile: bool = False) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET ended_at=?, closed_by_reconcile=? "
                "WHERE id=?",
                (time.time(), 1 if by_reconcile else 0, session_id))

    @_guarded
    def bump_session_offset(self, session_id: int, *, add_media_s: float,
                            next_segment: int) -> None:
        """After a disconnect: bank recorded media time so absolute timestamps
        survive the reconnect (spec §S0 reconnect requirement)."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET base_offset_s = base_offset_s + ?, "
                "next_segment=? WHERE id=?", (add_media_s, next_segment, session_id))

    @_guarded
    def set_next_segment(self, session_id: int, next_segment: int) -> None:
        """Persist the next segment index without touching the offset.

        Needed for segments that consume an index but bank no media time
        (a torn tail that was discarded): without this the reconnect would
        reuse the index and collide with the discarded file's slot.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET next_segment=MAX(next_segment,?) "
                "WHERE id=?", (next_segment, session_id))

    @_guarded
    def get_session(self, session_id: int) -> sqlite3.Row | None:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM stream_sessions WHERE id=?", (session_id,)).fetchone()

    @_guarded
    def open_sessions(self) -> list[sqlite3.Row]:
        """Sessions a crash left open — the startup reconciliation input."""
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM stream_sessions WHERE ended_at IS NULL "
                "ORDER BY id").fetchall()

    @_guarded
    def resumable_session(self, platform: str, handle: str,
                          within_s: float) -> int | None:
        """The most recent session for this channel that ended less than
        ``within_s`` ago — so a brief outage resumes ONE broadcast timeline
        instead of restarting absolute time at zero in a new session."""
        cutoff = time.time() - within_s
        with self._lock:
            # A normally-closed session is judged on ended_at. A
            # RECONCILE-closed one is judged on when its media was last
            # WRITTEN: reconciliation stamps ended_at with "now", so trusting
            # that would make a years-old crashed broadcast resumable, while
            # excluding such sessions outright would stop a crash seconds
            # into a live stream from resuming its timeline — restarting
            # absolute media time at zero mid-broadcast.
            # Ranking uses each session's REAL recency, not ended_at:
            # reconciliation stamps ended_at with "now", so ordering by it
            # made a crash-recovered session always outrank a normally-closed
            # one — the monitor would resume the STALER broadcast and orphan
            # the fresher. Reconcile-closed rows rank on last_media_at (when
            # media was actually written); normally-closed rows on ended_at.
            #
            # OPEN sessions stay excluded on purpose: reconcile leaves a
            # session open only when unregistered media remains, and resuming
            # one would stamp new segments over a timeline range that media
            # already occupies. The freshest-crash case R7-4 targets is fixed
            # upstream instead — retention._await_settled lets that crash
            # RECOVER and close, so it reaches this query by the normal arm.
            row = self._conn.execute(
                "SELECT id, "
                "  CASE WHEN closed_by_reconcile = 1 "
                "       THEN last_media_at ELSE ended_at END AS recency "
                "FROM stream_sessions "
                "WHERE platform=? AND handle=? AND ended_at IS NOT NULL AND ("
                "  (closed_by_reconcile = 0 AND ended_at >= ?) OR "
                "  (closed_by_reconcile = 1 AND last_media_at >= ?)) "
                "ORDER BY recency DESC, id DESC LIMIT 1",
                (platform, handle, cutoff, cutoff)).fetchone()
        return int(row["id"]) if row else None

    @_guarded
    def reopen_session(self, session_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET ended_at=NULL WHERE id=?", (session_id,))

    @_guarded
    def set_session_progress(self, session_id: int, base_offset_s: float,
                             next_segment: int) -> None:
        """Write back a session's timeline cursor after out-of-band recovery.

        Reconciliation registers crash-stranded media directly into
        ``segments``; without also advancing the SESSION row, a later resume
        reads stale bookkeeping, restarts ffmpeg's ``-segment_start_number``
        on top of the recovered files, and destroys them. MAX() so a
        concurrent banking write is never rolled backwards.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET base_offset_s=MAX(base_offset_s,?), "
                "next_segment=MAX(next_segment,?) WHERE id=?",
                (base_offset_s, next_segment, session_id))

    @_guarded
    def freshen_last_media(self, session_id: int, mtime: float) -> None:
        """Raise (never lower) the session's media-recency stamp.

        Reconciliation calls this with the newest media FILE mtime: the
        per-segment-close stamp alone is coarser (once per 900 s) than the
        300 s resume window, so a crash mid-segment looked stale when its
        media was seconds old.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET last_media_at=MAX(last_media_at,?) "
                "WHERE id=?", (mtime, session_id))

    @_guarded
    def mark_timeline_estimated(self, session_id: int) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE stream_sessions SET timeline_estimated=1 WHERE id=?",
                (session_id,))

    @_guarded
    def delete_segment_row(self, session_id: int, seg_index: int) -> None:
        """Actually remove a row whose media is gone (retention's 'prune')."""
        with self._lock, self._conn:
            self._conn.execute(
                "DELETE FROM segments WHERE session_id=? AND seg_index=?",
                (session_id, seg_index))

    @_guarded
    def all_segment_paths(self) -> set[str]:
        """Every path the DB knows about — retention uses this to tell a
        tracked file from a stranded one."""
        with self._lock:
            return {r["path"] for r in
                    self._conn.execute("SELECT path FROM segments").fetchall()}

    @_guarded
    def segments_for_session(self, session_id: int) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM segments WHERE session_id=? ORDER BY seg_index",
                (session_id,)).fetchall()

    @_guarded
    def add_segment(self, session_id: int, seg_index: int, path: Path | str,
                    abs_start_s: float) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments(session_id, seg_index, path, abs_start_s) "
                "VALUES(?,?,?,?)", (session_id, seg_index, str(path), abs_start_s))

    @_guarded
    def segment_ready(self, session_id: int, seg_index: int, duration_s: float,
                      path: Path | str | None = None) -> None:
        """Mark a segment closed. ONE transaction with add_segment's row, so
        a crash cannot leave a 'recording' row that reconciliation must guess
        about; ``path`` updates the location when the file moved (quarantine)."""
        with self._lock, self._conn:
            if path is None:
                self._conn.execute(
                    "UPDATE segments SET status='ready', duration_s=? "
                    "WHERE session_id=? AND seg_index=?",
                    (duration_s, session_id, seg_index))
            else:
                self._conn.execute(
                    "UPDATE segments SET status='ready', duration_s=?, path=? "
                    "WHERE session_id=? AND seg_index=?",
                    (duration_s, str(path), session_id, seg_index))

    @_guarded
    def record_closed_segment(self, session_id: int, seg_index: int,
                              path: Path | str, abs_start_s: float,
                              duration_s: float, status: str = "ready") -> None:
        """Insert a fully-closed segment in ONE transaction.

        Replaces the add_segment→segment_ready pair on the chunker's hot
        path: review demonstrated that a crash between the two left rows
        stuck at 'recording' forever with no reconciliation.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments"
                "(session_id, seg_index, path, abs_start_s, duration_s, status) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, seg_index, str(path), abs_start_s, duration_s, status))

    @_guarded
    def bank_segment_and_offset(self, session_id: int, seg_index: int,
                                path: Path | str, abs_start_s: float,
                                duration_s: float, next_segment: int,
                                status: str = "ready") -> None:
        """Record a closed segment AND advance the session offset atomically.

        This is the crash-window fix: review showed that banking the offset
        only at connect-end meant a crash mid-connect reset absolute time to
        zero on restart. Now every closed segment moves the durable timeline
        forward in the same transaction that records it.
        """
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO segments"
                "(session_id, seg_index, path, abs_start_s, duration_s, status) "
                "VALUES(?,?,?,?,?,?)",
                (session_id, seg_index, str(path), abs_start_s, duration_s, status))
            self._conn.execute(
                "UPDATE stream_sessions SET base_offset_s=?, next_segment=?, "
                "last_media_at=? WHERE id=?",
                (abs_start_s + duration_s, next_segment, time.time(), session_id))

    @_guarded
    def segments_with_status(self, status: str) -> Iterable[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(
                "SELECT * FROM segments WHERE status=? ORDER BY session_id, seg_index",
                (status,)).fetchall()

    @_guarded
    def set_segment_status(self, session_id: int, seg_index: int, status: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE segments SET status=? WHERE session_id=? AND seg_index=?",
                (status, session_id, seg_index))
