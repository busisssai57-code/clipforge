"""Stable-file debounce: never hand a growing file to the DAG."""

from pathlib import Path

from clipforge.watch.watcher import StableFileTracker


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_growing_file_is_not_emitted(tmp_path: Path):
    clock = Clock()
    tracker = StableFileTracker(stable_s=20.0, clock=clock)
    f = tmp_path / "a.mp4"

    f.write_bytes(b"x" * 100)
    tracker.observe(f)
    clock.advance(15.0)
    assert tracker.harvest() == []

    f.write_bytes(b"x" * 200)  # still growing: timer resets
    tracker.observe(f)
    clock.advance(15.0)
    assert tracker.harvest() == [], "reset on size change"

    clock.advance(10.0)  # now 25s at a stable size
    assert tracker.harvest() == [f]


def test_each_file_emitted_once(tmp_path: Path):
    clock = Clock()
    tracker = StableFileTracker(stable_s=1.0, clock=clock)
    f = tmp_path / "a.ts"
    f.write_bytes(b"x")
    tracker.observe(f)
    clock.advance(5.0)
    assert tracker.harvest() == [f]
    tracker.observe(f)
    clock.advance(5.0)
    assert tracker.harvest() == [], "already emitted"


def test_non_media_and_empty_files_ignored(tmp_path: Path):
    clock = Clock()
    tracker = StableFileTracker(stable_s=1.0, clock=clock)
    txt = tmp_path / "notes.txt"
    txt.write_text("x", encoding="utf-8")
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    tracker.observe(txt)
    tracker.observe(empty)
    clock.advance(5.0)
    assert tracker.harvest() == []


def test_vanished_file_is_dropped(tmp_path: Path):
    clock = Clock()
    tracker = StableFileTracker(stable_s=1.0, clock=clock)
    f = tmp_path / "a.mp4"
    f.write_bytes(b"x")
    tracker.observe(f)
    f.unlink()
    tracker.observe(f)  # must not raise
    clock.advance(5.0)
    assert tracker.harvest() == []


def test_harvest_order_is_deterministic(tmp_path: Path):
    clock = Clock()
    tracker = StableFileTracker(stable_s=1.0, clock=clock)
    names = ["c.mp4", "a.mp4", "b.mp4"]
    for n in names:
        p = tmp_path / n
        p.write_bytes(b"x")
        tracker.observe(p)
    clock.advance(5.0)
    assert [p.name for p in tracker.harvest()] == ["a.mp4", "b.mp4", "c.mp4"]
