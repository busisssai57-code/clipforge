"""Workspace layout and atomic write helpers.

Resumability Law (§3.3) plumbing lives here:

  * All artifact writes go through :func:`atomic_write_*` — content is
    written to a ``*.partial`` temp file in the *same directory* (same
    volume, so ``os.replace`` is atomic on NTFS), fsync'd, then renamed
    over the destination. A crash at any instant leaves either the old
    file, no file, or a ``.partial`` — never a torn destination file.
  * On startup :func:`discard_partials` sweeps ``*.partial`` files, which
    by construction are the only debris a crash can leave.

Nothing in this module may depend on config — config depends on us.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from clipforge.errors import AtomicWriteError

PARTIAL_SUFFIX = ".partial"

# Windows: os.replace fails with PermissionError while ANY other handle holds
# the destination without FILE_SHARE_DELETE (Defender, Search indexer, a
# reader). These holds are transient (ms–s), so a short bounded retry turns
# a spurious crash into a non-event. Exhausting retries raises TYPED.
_REPLACE_ATTEMPTS = 6
_REPLACE_DELAY_S = 0.05  # exponential: 0.05, 0.1, 0.2, ... (~1.6 s total)


@dataclass(frozen=True)
class Workspace:
    """Canonical on-disk layout (spec §4). All pipeline paths derive from here."""

    root: Path
    chunks: Path = field(init=False)
    artifacts: Path = field(init=False)
    clips: Path = field(init=False)
    logs: Path = field(init=False)
    tmp: Path = field(init=False)
    quarantine: Path = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        object.__setattr__(self, "root", root)
        object.__setattr__(self, "chunks", root / "chunks")
        object.__setattr__(self, "artifacts", root / "artifacts")
        object.__setattr__(self, "clips", root / "clips")
        object.__setattr__(self, "logs", root / "logs")
        object.__setattr__(self, "tmp", root / "tmp")
        object.__setattr__(self, "quarantine", root / "quarantine")

    @property
    def state_db(self) -> Path:
        return self.root / "state.sqlite3"

    def all_dirs(self) -> tuple[Path, ...]:
        return (self.root, self.chunks, self.artifacts, self.clips,
                self.logs, self.tmp, self.quarantine)

    def ensure(self) -> "Workspace":
        """Create the full directory tree. Idempotent."""
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self


def _partial_path(dest: Path) -> Path:
    """Unique sibling temp path. Same directory ⇒ same volume ⇒ atomic replace.

    A uuid component means two writers racing on the same destination never
    collide on the temp file. Racing replaces are content-safe because
    artifacts are content-addressed (same key ⇒ same bytes) — though on
    Windows a concurrent reader can still block the rename itself, which
    :func:`_replace_with_retry` absorbs.
    """
    return dest.with_name(f"{dest.name}.{uuid.uuid4().hex[:8]}{PARTIAL_SUFFIX}")


def _replace_with_retry(tmp: Path, dest: Path, *,
                        attempts: int = _REPLACE_ATTEMPTS,
                        delay_s: float = _REPLACE_DELAY_S) -> None:
    """``os.replace`` with bounded backoff for Windows share-mode holds.

    The rename itself is atomic on NTFS within one volume; what is NOT
    guaranteed on Windows is that it *succeeds* while another handle holds
    ``dest``. Readers of completed artifacts are transient, so retrying a
    few times absorbs almost all real holds; a persistent hold surfaces as
    a typed :class:`AtomicWriteError` the orchestrator can act on.
    """
    for attempt in range(attempts):
        try:
            os.replace(tmp, dest)
            return
        except PermissionError as exc:
            if attempt == attempts - 1:
                raise AtomicWriteError(
                    f"os.replace({tmp.name!r} -> {dest}) still blocked after "
                    f"{attempts} attempts - destination held open by another "
                    f"process (AV scanner / indexer / reader): {exc}") from exc
            time.sleep(delay_s * (2 ** attempt))
        except OSError as exc:
            # Non-hold failure (dest dir removed, invalid path): no retry
            # would help — surface immediately, but still TYPED.
            raise AtomicWriteError(
                f"os.replace({tmp.name!r} -> {dest}) failed: {exc}") from exc


def atomic_write_bytes(dest: Path, data: bytes) -> Path:
    """Write ``data`` to ``dest`` atomically (temp + fsync + os.replace).

    Failure contract: raises :class:`AtomicWriteError` (typed) for every
    I/O failure mode; on ANY failure ``dest`` is untouched (old content or
    absence preserved) and the only possible debris is a ``.partial``.
    """
    dest = Path(dest)
    tmp: Path | None = None
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = _partial_path(dest)
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        _replace_with_retry(tmp, dest)
    except AtomicWriteError:
        raise
    except OSError as exc:  # disk full, path invalid, fsync failure, ...
        raise AtomicWriteError(f"atomic write to {dest} failed: {exc}") from exc
    finally:
        # If replace succeeded tmp is gone; if write/replace failed, remove
        # eagerly so debris stays rare (the startup sweep is the backstop).
        if tmp is not None and tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass  # locked/AV-scanned; startup sweep will retry
    return dest


def atomic_write_text(dest: Path, text: str, encoding: str = "utf-8") -> Path:
    return atomic_write_bytes(dest, text.encode(encoding))


def atomic_write_json(dest: Path, obj: Any, *, indent: int = 2) -> Path:
    """Deterministic JSON: sorted keys, fixed separators, LF, trailing newline.

    Determinism Law: identical object ⇒ identical bytes, byte-for-byte.
    """
    text = json.dumps(obj, indent=indent, sort_keys=True, ensure_ascii=False,
                      separators=(",", ": ")) + "\n"
    return atomic_write_bytes(dest, text.encode("utf-8"))


def iter_partials(root: Path) -> Iterator[Path]:
    """Yield all crash debris under ``root`` (recursive)."""
    yield from Path(root).rglob(f"*{PARTIAL_SUFFIX}")


def discard_partials(root: Path) -> list[Path]:
    """Delete all ``*.partial`` files under ``root``. Returns what was removed.

    Called once at startup (Resumability Law): a crash mid-stage leaves only
    a ``.partial``, which is discarded here; the stage re-runs from its cache
    key as if it never started.
    """
    removed: list[Path] = []
    for p in sorted(iter_partials(root)):  # sorted: deterministic log order
        try:
            p.unlink()
            removed.append(p)
        except OSError:
            # File locked (AV scan, straggler handle). Leave it; it is inert —
            # nothing ever reads *.partial as an artifact.
            continue
    return removed




def hf_cache_dir(model_id: str) -> Path:
    """Where this model's blobs live, without importing torch or touching
    the network.

    Lived in genvideo/models.py until the generation half was removed, but
    it was never about generation: preflight asks the same question about
    S3's vision-language weights, and two copies of this path arithmetic
    would drift the moment HF changes its layout.
    """
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # noqa: PLC0415
        root = Path(HF_HUB_CACHE)
    except Exception:  # noqa: BLE001
        root = Path.home() / ".cache" / "huggingface" / "hub"
    return root / ("models--" + model_id.replace("/", "--"))
