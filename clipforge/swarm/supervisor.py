"""Supervisor — runs the swarm.

Agents declare which task kinds they handle and whether they need the GPU.
The supervisor claims work on their behalf, enforces the resource laws,
and folds any follow-up tasks an agent emits back onto the board.

The honest constraint, stated once here so nobody has to infer it: **the
VRAM Law allows exactly one GPU stage at a time.** Parallelism in this
swarm is therefore real for discovery, transcription queuing, quality
checking and packaging, and strictly serial for generation and rendering.
A design that ran four renderers on one card would not be four times
faster; it would raise, or thrash, or return blank frames. So GPU agents
share a single permit and CPU agents run free up to their own cap.

Agents never call each other. An agent returns follow-up tasks and the
supervisor posts them, which is what lets a role be added or removed
without touching any other role.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, Sequence

from clipforge.log import get_logger
from clipforge.swarm.board import Task, TaskBoard

log = get_logger(__name__)


class Agent(Protocol):
    """One role in the swarm."""

    name: str
    #: Task kinds this agent can handle.
    kinds: tuple[str, ...]
    #: Whether running this agent occupies the single GPU permit.
    gpu: bool

    def run(self, task: Task) -> Sequence[tuple[str, dict[str, Any]]]:
        """Do the work; return follow-up tasks as (kind, payload) pairs."""


@dataclass
class SwarmStats:
    claimed: int = 0
    done: int = 0
    failed: int = 0
    retried: int = 0
    spawned: int = 0

    def snapshot(self) -> dict[str, int]:
        return {"claimed": self.claimed, "done": self.done,
                "failed": self.failed, "retried": self.retried,
                "spawned": self.spawned}


@dataclass
class Supervisor:
    board: TaskBoard
    agents: list[Agent] = field(default_factory=list)
    #: Concurrent NON-GPU workers. GPU work is serialised regardless.
    cpu_workers: int = 4
    poll_s: float = 0.5
    lease_s: float = 900.0
    stats: SwarmStats = field(default_factory=SwarmStats)

    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _gpu: threading.Semaphore = field(
        default_factory=lambda: threading.Semaphore(1), init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    #: Tasks claimed and handed to the pool but not yet finished. The board
    #: cannot see these: a task being worked is 'running' with a LIVE
    #: lease, which `pending_count` deliberately excludes (it counts work
    #: that is claimable, not work that exists). Draining on the board
    #: alone therefore exits while a render is still going.
    _inflight: int = field(default=0, init=False)

    def register(self, agent: Agent) -> "Supervisor":
        overlap = {k for a in self.agents for k in a.kinds} & set(agent.kinds)
        if overlap:
            # Two agents claiming the same kind is not sharing, it is a
            # race whose winner depends on thread scheduling.
            raise ValueError(
                f"{agent.name} handles {sorted(overlap)}, already handled by "
                "another agent; task kinds must map to exactly one agent")
        self.agents.append(agent)
        return self

    def stop(self) -> None:
        self._stop.set()

    # ------------------------------------------------------------- work

    def _handle(self, agent: Agent, task: Task) -> None:
        permit = self._gpu if agent.gpu else None
        if permit is not None:
            permit.acquire()
        started = time.monotonic()
        try:
            follow = agent.run(task) or []
            ids = [self.board.submit(kind, payload, parent_id=task.id,
                                     goal=task.goal)
                   for kind, payload in follow]
            self.board.complete(task.id, {"spawned": ids})
            with self._lock:
                self.stats.done += 1
                self.stats.spawned += len(ids)
            log.info("swarm.task_done", task=task.id, kind=task.kind,
                     agent=agent.name, spawned=len(ids),
                     elapsed_s=round(time.monotonic() - started, 1))
        except BaseException as exc:  # noqa: BLE001
            # A failing agent must never take the supervisor down: other
            # roles are still working and the board is still durable.
            status = self.board.fail(task.id, f"{type(exc).__name__}: {exc}")
            with self._lock:
                if status == "failed":
                    self.stats.failed += 1
                else:
                    self.stats.retried += 1
            log.error("swarm.task_failed", task=task.id, kind=task.kind,
                      agent=agent.name, status=status,
                      error=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            if permit is not None:
                permit.release()
            with self._lock:
                self._inflight -= 1

    def _claim_one(self, pool: ThreadPoolExecutor) -> bool:
        """Claim at most one task for one agent. True if something ran."""
        for agent in self.agents:
            task = self.board.claim(agent.kinds, worker=agent.name,
                                    lease_s=self.lease_s)
            if task is None:
                continue
            # Count it in flight BEFORE submitting: the drain check runs on
            # this same thread between iterations, and a gap here is a
            # window where the task is invisible to both the board and the
            # counter.
            with self._lock:
                self.stats.claimed += 1
                self._inflight += 1
            try:
                pool.submit(self._handle, agent, task)
            except BaseException:
                with self._lock:
                    self._inflight -= 1
                self.board.fail(task.id, "could not be scheduled")
                raise
            return True
        return False

    @property
    def inflight(self) -> int:
        with self._lock:
            return self._inflight

    def run_until_drained(self, *, timeout_s: float = 3600.0) -> SwarmStats:
        """Work until the board has nothing ready, then return.

        The finite mode: used for a single goal, and by the tests. The
        always-on mode is `serve`, which is the same loop without the
        drain exit.
        """
        deadline = time.monotonic() + timeout_s
        with ThreadPoolExecutor(max_workers=max(1, self.cpu_workers)) as pool:
            idle_rounds = 0
            while not self._stop.is_set():
                if time.monotonic() > deadline:
                    log.warning("swarm.timeout", timeout_s=timeout_s,
                                **self.stats.snapshot())
                    break
                if self._claim_one(pool):
                    idle_rounds = 0
                    continue
                # Nothing ready. Give in-flight work a chance to post
                # follow-ups before declaring the board drained — exiting
                # on the first empty poll would stop mid-pipeline.
                idle_rounds += 1
                # Drained means: nothing claimable AND nothing being worked.
                # Checking the board alone exited 1.5s into a 5-minute
                # render, because an in-flight task holds a live lease and
                # is deliberately not "pending". The pool then blocked on
                # __exit__ for work the loop had already stopped watching,
                # so its retry was never re-claimed and its follow-up tasks
                # never ran.
                if (idle_rounds >= 3 and self.inflight == 0
                        and self.board.pending_count() == 0):
                    break
                self._stop.wait(self.poll_s)
        return self.stats

    def serve(self) -> SwarmStats:
        """Always-on mode: never exits on an empty board."""
        with ThreadPoolExecutor(max_workers=max(1, self.cpu_workers)) as pool:
            while not self._stop.is_set():
                if not self._claim_one(pool):
                    self._stop.wait(self.poll_s)
        return self.stats

    # ----------------------------------------------------------- report

    def describe(self) -> dict[str, Any]:
        return {
            "agents": [{"name": a.name, "kinds": list(a.kinds), "gpu": a.gpu}
                       for a in self.agents],
            "cpu_workers": self.cpu_workers,
            "gpu_permits": 1,
            "board": self.board.stats(),
            "stats": self.stats.snapshot(),
        }


def make_agent(name: str, kinds: Sequence[str], fn: Callable[[Task], Any], *,
               gpu: bool = False) -> Agent:
    """Wrap a plain function as an agent — for small roles and for tests."""

    @dataclass
    class _Fn:
        name: str
        kinds: tuple[str, ...]
        gpu: bool

        def run(self, task: Task):
            return fn(task) or []

    return _Fn(name, tuple(kinds), gpu)
