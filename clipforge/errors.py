"""Typed exception hierarchy for ClipForge.

Every subsystem raises from this tree so orchestration can make policy
decisions (retry / fallback / quarantine / abort) on exception *type*,
never by string-matching messages.
"""

from __future__ import annotations


class ClipForgeError(Exception):
    """Root of all ClipForge-raised errors."""


# ---------------------------------------------------------------- config


class ConfigError(ClipForgeError):
    """Invalid or missing configuration. Fail-fast at startup, never at runtime."""


# ---------------------------------------------------------------- preflight


class PreflightError(ClipForgeError):
    """A `doctor` prerequisite is missing. Message must be actionable."""


# ---------------------------------------------------------------- GPU / VRAM


class GpuError(ClipForgeError):
    """Base for GPU management failures."""


class CoResidencyError(GpuError):
    """Attempted to register a model class while another is resident.

    Raising here (instead of loading and hoping) is the VRAM Law's teeth:
    a deliberate co-load attempt must raise, not OOM later.
    """


class VramBudgetError(GpuError):
    """Free VRAM below the stage's declared budget at acquisition time."""


# ---------------------------------------------------------------- filesystem


class AtomicWriteError(ClipForgeError):
    """An atomic artifact commit failed after retries (Windows: destination
    held open by AV/indexer/reader; or disk full). Typed so orchestration
    can retry/quarantine instead of dying on a raw OSError."""


# ---------------------------------------------------------------- stages


class StageError(ClipForgeError):
    """Base for stage execution failures."""

    def __init__(self, message: str, *, stage: str | None = None) -> None:
        super().__init__(message)
        self.stage = stage


class RetryableStageError(StageError):
    """Transient failure — orchestrator may retry per the stage's policy."""


class FatalStageError(StageError):
    """Deterministic stage failed: this is a bug, not bad luck. Do not retry."""


class QuarantineError(StageError):
    """Input chunk is unprocessable after retries — move aside, keep going."""


class FallbackUsed(StageError):
    """Not an error to propagate — a marker recorded when a stage degraded
    to its documented fallback (e.g. S3 → heuristic order). Stages catch
    their own failure, record this in the artifact, and return success."""


# ---------------------------------------------------------------- media / ffmpeg


class FfmpegError(ClipForgeError):
    """ffmpeg/ffprobe subprocess failed. Carries the tail of stderr."""

    def __init__(self, message: str, *, cmd: list[str] | None = None,
                 returncode: int | None = None, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.cmd = cmd or []
        self.returncode = returncode
        self.stderr_tail = stderr_tail


# ---------------------------------------------------------------- ingest


class IngestError(ClipForgeError):
    """Base for ingestion failures (network, platform, chunker)."""


class StreamGoneError(IngestError):
    """Live stream ended or became unreachable — normal, triggers reconnect loop."""


# ---------------------------------------------------------------- state


class StateError(ClipForgeError):
    """Job-store (SQLite) failure."""
