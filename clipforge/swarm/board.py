"""Durable task board — the shared memory of the swarm.

Agents do not call each other. They post work here and claim work from
here, which is what makes the system a swarm rather than a call graph: an
agent can be added, removed or crash without any other agent knowing.

Three properties do the real work:

* **Durable.** The board lives in the same SQLite file as everything else,
  so a crash mid-run loses at most the in-flight lease. An in-memory queue
  would silently discard a night's planned work on a power cut.
* **Leased, not locked.** A claimed task carries an expiry. If the worker
  dies, the lease lapses and the task returns to the pool automatically —
  no reaper process, no stuck rows.
* **Attempt-bounded.** Every task counts its attempts and dies permanently
  at a ceiling. A retry loop with no ceiling is an infinite loop that
  costs GPU time.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from clipforge.log import get_logger

log = get_logger(__name__)

Clock = Callable[[], float]

#: Attempts before a task is parked as permanently failed.
MAX_ATTEMPTS = 3

#: How long a claim is held before it lapses back to the pool.
DEFAULT_LEASE_S = 900.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS swarm_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    payload     TEXT NOT NULL DEFAULT '{}',
    priority    INTEGER NOT NULL DEFAULT 50,
    status      TEXT NOT NULL DEFAULT 'pending',
    attempts    INTEGER NOT NULL DEFAULT 0,
    parent_id   INTEGER,
    goal        TEXT,
    claimed_by  TEXT,
    lease_until REAL NOT NULL DEFAULT 0,
    error       TEXT,
    result      TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_swarm_ready
    ON swarm_tasks(status, priority, id);
"""


@dataclass
class Task:
    id: int
    kind: str
    payload: dict[str, Any]
    priority: int = 50
    status: str = "pending"
    attempts: int = 0
    parent_id: int | None = None
    goal: str | None = None
    error: str = ""

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Task":
        try:
            payload = json.loads(row["payload"])
        except (ValueError, TypeError):
            payload = {}
        return cls(id=int(row["id"]), kind=str(row["kind"]), payload=payload,
                   priority=int(row["priority"]), status=str(row["status"]),
                   attempts=int(row["attempts"]),
                   parent_id=row["parent_id"], goal=row["goal"],
                   error=row["error"] or "")


@dataclass
class TaskBoard:
    path: Path
    clock: Clock = time.time
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False,
                                     timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # ----------------------------------------------------------- submit

    def submit(self, kind: str, payload: dict[str, Any] | None = None, *,
               priority: int = 50, parent_id: int | None = None,
               goal: str | None = None) -> int:
        now = self.clock()
        with self._lock, self._conn:
            cur = self._conn.execute(
                "INSERT INTO swarm_tasks(kind,payload,priority,parent_id,goal,"
                "created_at,updated_at) VALUES(?,?,?,?,?,?,?)",
                (kind, json.dumps(payload or {}, sort_keys=True), priority,
                 parent_id, goal, now, now))
        return int(cur.lastrowid)

    def submit_many(self, tasks: Iterable[tuple[str, dict[str, Any]]], *,
                    priority: int = 50, parent_id: int | None = None,
                    goal: str | None = None) -> list[int]:
        return [self.submit(k, p, priority=priority, parent_id=parent_id,
                            goal=goal) for k, p in tasks]

    # ------------------------------------------------------------ claim

    def claim(self, kinds: Iterable[str], *, worker: str,
              lease_s: float = DEFAULT_LEASE_S) -> Task | None:
        """Highest-priority ready task of any of ``kinds``, or None.

        Reclaims tasks whose lease has lapsed in the same statement, so a
        dead worker's task returns to the pool with no separate reaper.
        """
        kinds = list(kinds)
        if not kinds:
            return None
        now = self.clock()
        placeholders = ",".join("?" * len(kinds))
        with self._lock, self._conn:
            row = self._conn.execute(
                f"SELECT * FROM swarm_tasks "
                f"WHERE kind IN ({placeholders}) "
                f"  AND (status='pending' "
                f"       OR (status='running' AND lease_until < ?)) "
                f"ORDER BY priority ASC, id ASC LIMIT 1",
                (*kinds, now)).fetchone()
            if row is None:
                return None
            if row["status"] == "running":
                log.warning("swarm.lease_lapsed", task=row["id"],
                            kind=row["kind"], previous_worker=row["claimed_by"])
            self._conn.execute(
                "UPDATE swarm_tasks SET status='running', claimed_by=?, "
                "attempts=attempts+1, lease_until=?, updated_at=? WHERE id=?",
                (worker, now + lease_s, now, row["id"]))
            row = self._conn.execute(
                "SELECT * FROM swarm_tasks WHERE id=?", (row["id"],)).fetchone()
        return Task.from_row(row)

    def heartbeat(self, task_id: int, *, lease_s: float = DEFAULT_LEASE_S) -> None:
        """Extend a lease. Long GPU work must call this or it is reclaimed
        mid-render and run twice."""
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE swarm_tasks SET lease_until=?, updated_at=? WHERE id=?",
                (self.clock() + lease_s, self.clock(), task_id))

    # --------------------------------------------------------- complete

    def complete(self, task_id: int, result: dict[str, Any] | None = None) -> None:
        now = self.clock()
        with self._lock, self._conn:
            self._conn.execute(
                "UPDATE swarm_tasks SET status='done', result=?, error='', "
                "lease_until=0, updated_at=? WHERE id=?",
                (json.dumps(result or {}, sort_keys=True), now, task_id))

    def fail(self, task_id: int, error: str, *,
             max_attempts: int = MAX_ATTEMPTS) -> str:
        """Record a failure. Returns the resulting status.

        Below the ceiling the task returns to 'pending' and will be tried
        again. At the ceiling it is parked as 'failed' — visible, not
        retried, and not silently dropped.
        """
        now = self.clock()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT attempts FROM swarm_tasks WHERE id=?",
                (task_id,)).fetchone()
            attempts = int(row["attempts"]) if row else max_attempts
            status = "failed" if attempts >= max_attempts else "pending"
            self._conn.execute(
                "UPDATE swarm_tasks SET status=?, error=?, lease_until=0, "
                "claimed_by=NULL, updated_at=? WHERE id=?",
                (status, error[:500], now, task_id))
        if status == "failed":
            log.error("swarm.task_exhausted", task=task_id,
                      attempts=attempts, error=error[:200])
        return status

    # ---------------------------------------------------------- inspect

    def get(self, task_id: int) -> Task | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM swarm_tasks WHERE id=?", (task_id,)).fetchone()
        return Task.from_row(row) if row else None

    def pending_count(self, kinds: Iterable[str] | None = None) -> int:
        now = self.clock()
        sql = ("SELECT COUNT(*) c FROM swarm_tasks WHERE "
               "(status='pending' OR (status='running' AND lease_until < ?))")
        args: list[Any] = [now]
        if kinds:
            kinds = list(kinds)
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            args += kinds
        with self._lock:
            return int(self._conn.execute(sql, args).fetchone()["c"])

    def stats(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) c FROM swarm_tasks GROUP BY status"
            ).fetchall()
        return {str(r["status"]): int(r["c"]) for r in rows}

    def recent(self, limit: int = 50) -> list[Task]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM swarm_tasks ORDER BY updated_at DESC LIMIT ?",
                (limit,)).fetchall()
        return [Task.from_row(r) for r in rows]
