"""Swarm board and supervisor, pinned.

The properties that make a swarm trustworthy are all failure properties:
work survives a crash, a dead worker's task comes back, a failing agent
does not take the system down, and retries stop. Those are what these
tests hold. The happy path is the easy part.
"""

from __future__ import annotations

import threading
import time

import pytest

from clipforge.swarm import MAX_ATTEMPTS, Supervisor, TaskBoard, make_agent


class FakeClock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, s):
        self.t += s


@pytest.fixture()
def board(tmp_path):
    b = TaskBoard(tmp_path / "swarm.sqlite3")
    yield b
    b.close()


# --------------------------------------------------------------- board

def test_work_survives_a_restart(tmp_path):
    """Durability is the whole reason this is not an in-memory queue: a
    crash must not discard a night of planned work."""
    path = tmp_path / "s.sqlite3"
    first = TaskBoard(path)
    tid = first.submit("render", {"brief": "x"}, priority=10)
    first.close()

    reborn = TaskBoard(path)            # new "process"
    got = reborn.claim(["render"], worker="w1")
    assert got is not None and got.id == tid
    assert got.payload == {"brief": "x"}
    reborn.close()


def test_priority_then_age_decides_order(board):
    low = board.submit("render", priority=90)
    high = board.submit("render", priority=10)
    older_high = board.submit("render", priority=10)
    assert board.claim(["render"], worker="w").id == high
    assert board.claim(["render"], worker="w").id == older_high
    assert board.claim(["render"], worker="w").id == low


def test_a_claimed_task_is_not_handed_to_a_second_worker(board):
    board.submit("render")
    assert board.claim(["render"], worker="a") is not None
    assert board.claim(["render"], worker="b") is None, (
        "the same task was claimed twice")


def test_a_dead_workers_task_returns_when_its_lease_lapses(tmp_path):
    """No reaper process: the lease expiring IS the recovery."""
    clock = FakeClock()
    b = TaskBoard(tmp_path / "s.sqlite3", clock=clock)
    tid = b.submit("render")
    assert b.claim(["render"], worker="dies", lease_s=60).id == tid
    assert b.claim(["render"], worker="other", lease_s=60) is None
    clock.advance(61)
    again = b.claim(["render"], worker="other", lease_s=60)
    assert again is not None and again.id == tid
    assert again.attempts == 2, "the reclaim must count as another attempt"
    b.close()


def test_a_heartbeat_keeps_a_long_render_from_being_reclaimed(tmp_path):
    clock = FakeClock()
    b = TaskBoard(tmp_path / "s.sqlite3", clock=clock)
    b.submit("render")
    t = b.claim(["render"], worker="slow", lease_s=60)
    clock.advance(50)
    b.heartbeat(t.id, lease_s=60)
    clock.advance(50)
    assert b.claim(["render"], worker="thief", lease_s=60) is None, (
        "a heartbeating worker had its task stolen mid-render")
    b.close()


def test_retries_are_bounded(board):
    tid = board.submit("render")
    for _ in range(MAX_ATTEMPTS - 1):
        board.claim(["render"], worker="w")
        assert board.fail(tid, "boom") == "pending"
    board.claim(["render"], worker="w")
    assert board.fail(tid, "boom") == "failed"
    assert board.claim(["render"], worker="w") is None, (
        "an exhausted task must stop being retried")


def test_an_exhausted_task_is_parked_visibly_not_deleted(board):
    tid = board.submit("render")
    for _ in range(MAX_ATTEMPTS):
        board.claim(["render"], worker="w")
        board.fail(tid, "boom")
    task = board.get(tid)
    assert task.status == "failed" and "boom" in task.error


def test_claiming_an_unhandled_kind_returns_nothing(board):
    board.submit("render")
    assert board.claim(["transcribe"], worker="w") is None


# ---------------------------------------------------------- supervisor

def test_a_pipeline_runs_through_several_roles(board):
    seen: list[str] = []

    def scout(_t):
        seen.append("scout")
        return [("render", {"n": 1}), ("render", {"n": 2})]

    def render(t):
        seen.append(f"render{t.payload['n']}")
        return [("qa", {})]

    def qa(_t):
        seen.append("qa")
        return []

    sup = Supervisor(board=board, cpu_workers=2)
    sup.register(make_agent("scout", ["scout"], scout))
    sup.register(make_agent("render", ["render"], render, gpu=True))
    sup.register(make_agent("qa", ["qa"], qa))
    board.submit("scout")
    stats = sup.run_until_drained(timeout_s=30)

    assert seen.count("scout") == 1
    assert {"render1", "render2"} <= set(seen)
    assert seen.count("qa") == 2, "each render should have been checked"
    assert stats.done == 5 and stats.failed == 0


def test_gpu_agents_never_run_concurrently(board):
    """The VRAM Law in one assertion. Two renders at once on a single
    card do not go faster; they raise or return garbage."""
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def render(_t):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.05)
        with lock:
            live["now"] -= 1
        return []

    sup = Supervisor(board=board, cpu_workers=6)
    sup.register(make_agent("render", ["render"], render, gpu=True))
    for _ in range(6):
        board.submit("render")
    sup.run_until_drained(timeout_s=30)
    assert live["peak"] == 1, (
        f"{live['peak']} GPU agents ran at once; the permit is not held")


def test_cpu_agents_do_run_concurrently(board):
    """Control: if everything serialised, the GPU test above would pass
    for the wrong reason and the swarm would be a queue with extra steps.
    """
    live = {"now": 0, "peak": 0}
    lock = threading.Lock()

    def work(_t):
        with lock:
            live["now"] += 1
            live["peak"] = max(live["peak"], live["now"])
        time.sleep(0.15)
        with lock:
            live["now"] -= 1
        return []

    sup = Supervisor(board=board, cpu_workers=4)
    sup.register(make_agent("qa", ["qa"], work))
    for _ in range(6):
        board.submit("qa")
    sup.run_until_drained(timeout_s=30)
    assert live["peak"] > 1, "CPU work is not actually parallel"


def test_one_failing_agent_does_not_stop_the_swarm(board):
    done: list[int] = []

    def flaky(t):
        if t.payload.get("bad"):
            raise RuntimeError("this one explodes")
        done.append(t.id)
        return []

    sup = Supervisor(board=board, cpu_workers=2)
    sup.register(make_agent("w", ["w"], flaky))
    board.submit("w", {"bad": True})
    good = [board.submit("w", {}) for _ in range(3)]
    sup.run_until_drained(timeout_s=30)

    assert sorted(done) == sorted(good), "good work was lost to a bad task"
    assert sup.stats.failed == 1


def test_two_agents_cannot_claim_the_same_kind(board):
    sup = Supervisor(board=board)
    sup.register(make_agent("a", ["render"], lambda t: []))
    with pytest.raises(ValueError) as err:
        sup.register(make_agent("b", ["render", "qa"], lambda t: []))
    assert "render" in str(err.value)


def test_the_supervisor_describes_its_own_limits(board):
    sup = Supervisor(board=board, cpu_workers=3)
    sup.register(make_agent("r", ["render"], lambda t: [], gpu=True))
    d = sup.describe()
    assert d["gpu_permits"] == 1
    assert d["cpu_workers"] == 3
    assert d["agents"][0]["gpu"] is True
