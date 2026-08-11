"""Atomic write + crash-debris semantics (Resumability Law plumbing)."""

import json
import os
import sys
from pathlib import Path

import pytest

from clipforge import paths
from clipforge.errors import AtomicWriteError


def test_atomic_write_creates_parents_and_content(tmp_path: Path) -> None:
    dest = tmp_path / "deep" / "nested" / "a.json"
    paths.atomic_write_json(dest, {"k": "v"})
    assert json.loads(dest.read_text(encoding="utf-8")) == {"k": "v"}


def test_atomic_write_replaces_existing(tmp_path: Path) -> None:
    dest = tmp_path / "a.txt"
    paths.atomic_write_text(dest, "old")
    paths.atomic_write_text(dest, "new")
    assert dest.read_text(encoding="utf-8") == "new"


def test_no_partial_left_after_success(tmp_path: Path) -> None:
    paths.atomic_write_text(tmp_path / "a.txt", "x")
    assert list(paths.iter_partials(tmp_path)) == []


def test_json_bytes_are_order_independent(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    paths.atomic_write_json(a, {"x": 1, "y": {"b": 2, "a": 3}})
    paths.atomic_write_json(b, {"y": {"a": 3, "b": 2}, "x": 1})
    assert a.read_bytes() == b.read_bytes()


def test_discard_partials_removes_only_partials(tmp_path: Path) -> None:
    keep = tmp_path / "artifact.json"
    keep.write_text("{}", encoding="utf-8")
    debris1 = tmp_path / f"artifact.json.abcd1234{paths.PARTIAL_SUFFIX}"
    debris2 = tmp_path / "sub" / f"other.ts.ffff0000{paths.PARTIAL_SUFFIX}"
    debris2.parent.mkdir()
    debris1.write_bytes(b"torn")
    debris2.write_bytes(b"torn")

    removed = paths.discard_partials(tmp_path)

    assert sorted(removed) == sorted([debris1, debris2])
    assert keep.exists()
    assert not debris1.exists() and not debris2.exists()


def test_failed_write_leaves_destination_untouched(tmp_path: Path,
                                                   monkeypatch) -> None:
    """Fault injection at the fsync boundary: a failure mid-write must leave
    the old destination bytes intact and raise TYPED."""
    dest = tmp_path / "a.json"
    paths.atomic_write_json(dest, {"v": 1})
    original = dest.read_bytes()

    def boom(fd):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(AtomicWriteError, match="atomic write"):
        paths.atomic_write_json(dest, {"v": 2})
    assert dest.read_bytes() == original
    assert list(paths.iter_partials(tmp_path)) == []  # eager cleanup worked


@pytest.mark.skipif(sys.platform != "win32",
                    reason="share-mode replace blocking is Windows-specific")
def test_replace_blocked_by_open_handle_raises_typed(tmp_path: Path) -> None:
    """The WinError-5 case from review: destination held open by a reader.
    Must surface as AtomicWriteError after bounded retries, never as a raw
    PermissionError."""
    dest = tmp_path / "a.json"
    paths.atomic_write_json(dest, {"v": 1})
    with open(dest, "r", encoding="utf-8"):
        with pytest.raises(AtomicWriteError, match="held open"):
            paths._replace_with_retry(  # exercise the primitive directly,
                _make_tmp(tmp_path),    # with fast test-sized retries
                dest, attempts=2, delay_s=0.01)
    # After the handle is released the same write succeeds:
    paths.atomic_write_json(dest, {"v": 2})
    assert json.loads(dest.read_text(encoding="utf-8")) == {"v": 2}


def _make_tmp(root: Path) -> Path:
    tmp = root / f"x.{paths.PARTIAL_SUFFIX}"
    tmp.write_bytes(b"data")
    return tmp


def test_workspace_layout(tmp_path: Path) -> None:
    ws = paths.Workspace(tmp_path / "w").ensure()
    for d in ws.all_dirs():
        assert d.is_dir()
    assert ws.state_db.parent == ws.root
    # frozen: layout cannot be mutated after construction
    import dataclasses
    import pytest

    with pytest.raises(dataclasses.FrozenInstanceError):
        ws.root = tmp_path  # type: ignore[misc]
