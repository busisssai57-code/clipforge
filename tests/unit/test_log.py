"""Logging survivability: rotation must never drop records (Windows)."""

import logging
import sys
from pathlib import Path

import pytest

from clipforge.log import SafeRotatingFileHandler


def _make_handler(path: Path, max_bytes: int) -> SafeRotatingFileHandler:
    h = SafeRotatingFileHandler(path, maxBytes=max_bytes, backupCount=2,
                                encoding="utf-8", delay=True)
    h.setFormatter(logging.Formatter("%(message)s"))
    return h


def _emit(handler: logging.Handler, msg: str) -> None:
    handler.emit(logging.LogRecord("t", logging.INFO, __file__, 1, msg,
                                   None, None))


def test_normal_rotation_still_works(tmp_path: Path) -> None:
    log_file = tmp_path / "log.jsonl"
    h = _make_handler(log_file, max_bytes=64)
    for i in range(10):
        _emit(h, f"record-{i:03d} " + "x" * 60)
    h.close()
    assert log_file.exists()
    assert (tmp_path / "log.jsonl.1").exists(), "rotation never happened"


@pytest.mark.skipif(sys.platform != "win32",
                    reason="rename-while-held blocking is Windows-specific")
def test_blocked_rotation_keeps_all_records(tmp_path: Path) -> None:
    """The WinError-32 case from review: another handle (AV/indexer/tail)
    holds the log during rollover. Stock behavior drops every record for
    the duration; ours must keep appending and lose NOTHING."""
    log_file = tmp_path / "log.jsonl"
    h = _make_handler(log_file, max_bytes=64)
    _emit(h, "before-hold")

    with open(log_file, "r", encoding="utf-8"):  # simulate the holder
        for i in range(5):
            _emit(h, f"during-hold-{i} " + "x" * 80)  # every emit wants to rotate

    _emit(h, "after-release")
    h.close()

    all_text = log_file.read_text(encoding="utf-8")
    for backup in tmp_path.glob("log.jsonl.*"):
        all_text += backup.read_text(encoding="utf-8")
    assert "before-hold" in all_text
    for i in range(5):
        assert f"during-hold-{i}" in all_text, "record lost during blocked rotation"
    assert "after-release" in all_text
