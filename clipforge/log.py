"""Structured logging: structlog → JSONL file + rich console.

Design:
  * Machine stream — one JSON object per line appended to
    ``workspace/logs/clipforge.jsonl`` (observability, post-mortems).
  * Human stream — rich-colored console lines.
  * Timestamps appear ONLY in logs, never in decision paths
    (Determinism Law: logging is observation, not computation).

Call :func:`setup_logging` exactly once, early in every entrypoint.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

import structlog

_CONFIGURED = False
_CONFIGURED_DIR: Path | None = None


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """RotatingFileHandler that survives Windows rollover failures.

    Stock rollover renames ``clipforge.jsonl`` → ``.1``; on Windows that
    rename fails with WinError 32 whenever ANY other handle holds the file
    (Defender, Search indexer, an operator tailing the log) — and stock
    behavior then DROPS the record via handleError. During a multi-day
    unattended run, rollover instants × background scanners make that a
    guaranteed silent-loss channel for exactly the records a post-mortem
    needs. Here a failed rollover is skipped: we reopen the current file,
    keep appending (file temporarily exceeds maxBytes), and retry rotation
    at the next emit. Records are never sacrificed to housekeeping.
    """

    def doRollover(self) -> None:  # noqa: N802 (stdlib naming)
        try:
            super().doRollover()
        except OSError:
            # Rotation blocked — reopen the un-rotated file and carry on.
            if self.stream is None:
                self.stream = self._open()


def setup_logging(log_dir: Path | None = None, *, level: str = "INFO",
                  console: bool = True) -> None:
    """Configure stdlib + structlog.

    Idempotent for the SAME target: re-calls are no-ops. A re-call with a
    DIFFERENT ``log_dir`` is almost certainly a bug (logs would silently
    keep flowing to the old file), so it logs a loud warning instead of
    silently ignoring the new destination.
    """
    global _CONFIGURED, _CONFIGURED_DIR
    if _CONFIGURED:
        if log_dir is not None and _CONFIGURED_DIR is not None \
                and Path(log_dir).resolve() != _CONFIGURED_DIR:
            logging.getLogger(__name__).warning(
                "setup_logging called again with a different log_dir (%s); "
                "logging continues to the original %s", log_dir, _CONFIGURED_DIR)
        return

    handlers: list[logging.Handler] = []

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        # Rotating so a week-long unattended run cannot fill the disk with
        # logs; Safe* variant so rotation can never lose records (Windows).
        file_handler = SafeRotatingFileHandler(
            log_dir / "clipforge.jsonl",
            maxBytes=64 * 1024 * 1024,
            backupCount=8,
            encoding="utf-8",
            delay=True,  # don't hold the file open before the first record
        )
        file_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=structlog.processors.JSONRenderer(sort_keys=True),
                foreign_pre_chain=_shared_processors(),
            )
        )
        handlers.append(file_handler)

    if console:
        console_handler = logging.StreamHandler()
        try:
            from rich.console import Console

            renderer = structlog.dev.ConsoleRenderer(colors=Console().is_terminal)
        except ImportError:  # pragma: no cover — rich is a hard dependency
            renderer = structlog.dev.ConsoleRenderer(colors=False)
        console_handler.setFormatter(
            structlog.stdlib.ProcessorFormatter(
                processor=renderer,
                foreign_pre_chain=_shared_processors(),
            )
        )
        handlers.append(console_handler)

    logging.basicConfig(level=level.upper(), handlers=handlers, force=True)

    structlog.configure(
        processors=[
            *_shared_processors(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True
    _CONFIGURED_DIR = Path(log_dir).resolve() if log_dir is not None else None


def _shared_processors() -> list:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Project-wide logger accessor. ``name`` should be the module path."""
    return structlog.get_logger(name)
