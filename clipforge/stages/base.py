"""Stage ABC — the contract every pipeline stage obeys (spec §5, laws §3).

A stage declares:
  * ``name`` / ``version``     — both participate in the cache key, so
                                 changing stage logic MUST bump ``version``,
  * ``vram_budget_gb``         — asserted by the GPULock before model load,
  * ``wall_budget_s``          — observability: overruns are logged, not killed,
  * its failure mode           — via which typed exception ``_execute`` raises.

Resumability (§3.3), mechanically::

    cache_key = sha256(stage_name | input_digest | stage_version | params_digest)

(§3.3 gives ``sha256(input_digest + stage_version + params_digest)``; the
stage name and ``|`` separators are a deliberate strengthening — without
them two stages at the same version with the same input collide and can be
handed each other's artifacts, and unseparated concatenation is ambiguous.)

  1. ``run()`` looks the key up in the artifact registry (scoped by stage)
     → hit = validate + return, no recompute.
  2. Miss: ``_execute()`` produces the artifact model; it is written via
     temp-file + ``os.replace`` (never a torn file), then registered.
  3. A crash between execute and register leaves a complete file but no
     registry row → the re-run recomputes and overwrites identical bytes
     (Determinism Law makes this safe).
  4. A cache-hit whose file is unreadable/corrupt is quarantined (renamed
     ``*.corrupt``) and the stage recomputes — a bad artifact must never
     become a permanent poison pill on the resume path.

Failure bookkeeping is TOTAL: every exit from ``run()`` — typed stage
errors, foreign exceptions, even bugs in the stage itself — records the
``stage_runs`` outcome before propagating, and foreign exceptions are
re-raised wrapped in :class:`StageError` so orchestration policy can always
match on type.

Determinism note: ``run()`` measures wall time for LOGS only. Nothing
time-derived flows into the artifact or the cache key.
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Generic, TypeVar

from pydantic import ValidationError

from clipforge.errors import FatalStageError, StageError, StateError
from clipforge.log import get_logger
from clipforge.schemas.base import ArtifactModel
from clipforge.state import StateDB

log = get_logger(__name__)

ArtifactT = TypeVar("ArtifactT", bound=ArtifactModel)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path, *, chunk: int = 1 << 20) -> str:
    """Content digest of an input file (streamed; chunks can be GBs)."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _reject_noncanonical(obj: Any, path: str = "params") -> None:
    """Recursive canonicality check for cache-key params.

    json.dumps silently coerces int dict keys to strings, so {1: x} and
    {"1": x} would alias to one cache key — reject non-str keys outright,
    in the same strict spirit as rejecting sets/objects/NaN.
    """
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not isinstance(k, str):
                raise FatalStageError(
                    f"{path}: dict key {k!r} is {type(k).__name__}, not str - "
                    "json coercion would alias distinct params to one cache key")
            _reject_noncanonical(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _reject_noncanonical(v, f"{path}[{i}]")


def digest_params(params: dict[str, Any]) -> str:
    """Canonical digest of the stage-relevant config subset.

    Only tunables that CHANGE THE OUTPUT belong in ``params``. The digest
    REJECTS anything that is not canonically JSON-serializable — a
    ``default=str`` escape hatch here would let sets (hash-randomized
    iteration) or objects (memory addresses) produce keys that differ
    across process restarts, silently breaking resumability. Callers
    convert: sets → sorted lists, Paths → strings, int keys → str keys,
    before passing.
    """
    _reject_noncanonical(params)
    try:
        canon = json.dumps(params, sort_keys=True, separators=(",", ":"),
                           allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise FatalStageError(
            f"stage params are not canonically JSON-serializable ({exc}); "
            "convert sets/objects/NaN before passing - cache keys must be "
            "identical across process restarts") from exc
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


class Stage(ABC, Generic[ArtifactT]):
    """Base class. Subclasses implement ``_execute`` and declare the class
    attributes below. GPU-using stages additionally wrap their model work in
    ``GPU_LOCK.acquire(...)`` and free models via ``hard_unload`` (dict
    discipline — see clipforge.gpu) inside ``_execute``'s own ``finally``."""

    #: Stable stage identifier — appears in the cache key, artifact
    #: filenames, and the DB.
    name: str = "base"
    #: Bump on ANY behavioral change; participates in the cache key.
    version: str = "0"
    #: Declared VRAM need (GB); 0 for CPU stages. Asserted by GPULock.
    vram_budget_gb: float = 0.0
    #: Soft wall-clock budget (seconds) — overrun logs a warning.
    wall_budget_s: float = 600.0
    #: The artifact model this stage produces.
    artifact_type: type[ArtifactT]

    def __init__(self, db: StateDB, artifacts_dir: Path, *,
                 vram_budget_gb: float | None = None) -> None:
        self.db = db
        self.artifacts_dir = Path(artifacts_dir)
        # The class attribute is this stage's MEASURED need; the config
        # knob of the same name existed in config.toml (and in the
        # example, documented as tunable) while every stage read its own
        # hardcoded number — the same shape of lie as the `quantize` knob
        # and the hardcoded x264 preset this project has found before.
        # An explicit override wins; None leaves the measured default.
        if vram_budget_gb is not None:
            self.vram_budget_gb = float(vram_budget_gb)

    # ------------------------------------------------------------------ keys

    def cache_key(self, input_digest: str, params: dict[str, Any]) -> str:
        """§3.3 strengthened: stage name + separators (module docstring)."""
        payload = f"{self.name}|{input_digest}|{self.version}|{digest_params(params)}"
        return hashlib.sha256(payload.encode("ascii")).hexdigest()

    def artifact_path(self, cache_key: str) -> Path:
        """Content-addressed location: <artifacts>/<stage>/<key>.json."""
        return self.artifacts_dir / self.name / f"{cache_key}.json"

    # ------------------------------------------------------------------- run

    def run(self, *, input_digest: str, params: dict[str, Any],
            job_id: int | None = None, **inputs: Any) -> ArtifactT:
        """Cache-or-compute. Synchronous: worker pools decide threading.

        ``inputs`` carries the actual input artifacts/paths for ``_execute``;
        ``input_digest`` must already summarize them (the caller computes it
        from the input artifact's own cache_key or the input file's digest —
        see each stage's docstring).
        """
        key = self.cache_key(input_digest, params)

        cached = self._try_cached(key)
        if cached is not None:
            return cached

        run_id = None
        if job_id is not None:
            run_id = self.db.stage_started(job_id, self.name, key)
        started = time.monotonic()
        try:
            artifact = self._execute(cache_key=key, params=params, **inputs)
            if artifact.cache_key != key or artifact.stage != self.name:
                # A wrong stamp would poison the registry now and force a
                # silent recompute-on-every-resume later — fail loudly here.
                raise FatalStageError(
                    f"{self.name} produced artifact stamped "
                    f"stage={artifact.stage!r}/key={artifact.cache_key!r}, "
                    f"expected {self.name!r}/{key!r}", stage=self.name)
            dest = self.artifact_path(key)
            artifact.write(dest)  # atomic: temp + fsync + os.replace (typed)
            self.db.record_artifact(key, self.name, self.version, dest)
        except StageError as exc:
            exc.stage = exc.stage or self.name
            self._record_failure(run_id, key, exc)
            raise
        except Exception as exc:
            # Foreign exception (library bug, OSError, sqlite...): bookkeeping
            # must still happen, and orchestration must get a TYPED error.
            wrapped = StageError(
                f"unexpected {type(exc).__name__} in {self.name}: {exc}",
                stage=self.name)
            self._record_failure(run_id, key, wrapped)
            raise wrapped from exc

        elapsed = time.monotonic() - started
        if run_id is not None:
            # Same degrade-to-log treatment as _record_failure: the artifact
            # is committed and registered — a bookkeeping hiccup must not
            # convert a SUCCESS into an escaping exception.
            try:
                self.db.stage_finished(run_id, artifact=str(dest))
            except StateError as db_exc:
                log.error("stage.success_bookkeeping_failed", stage=self.name,
                          run_id=run_id, error=str(db_exc))
        if elapsed > self.wall_budget_s:
            log.warning("stage.wall_budget_exceeded", stage=self.name,
                        elapsed_s=round(elapsed, 1), budget_s=self.wall_budget_s)
        log.info("stage.done", stage=self.name, cache_key=key,
                 elapsed_s=round(elapsed, 1))
        return artifact

    # ----------------------------------------------------------------- cache

    def _try_cached(self, key: str) -> ArtifactT | None:
        """Resume fast-path with poison-pill protection.

        A registry hit whose file is missing → miss (retention deleted it).
        A hit whose file is unreadable, fails validation, or carries the
        wrong stage/key → quarantine the file and recompute. Every failure
        here degrades to a recompute — the resume path must never be the
        thing that kills the pipeline.
        """
        cached = self.db.lookup_artifact(key, self.name)
        if cached is None:
            return None
        try:
            artifact = self.artifact_type.read(cached)
            if artifact.stage != self.name or artifact.cache_key != key:
                raise StateError(
                    f"artifact at {cached} claims stage={artifact.stage!r} "
                    f"key={artifact.cache_key!r}, expected {self.name!r}/{key!r}")
            log.info("stage.cache_hit", stage=self.name, cache_key=key)
            return artifact
        # ValueError covers JSONDecodeError, UnicodeDecodeError, AND pydantic
        # ValidationError (all subclasses) — every corrupt-bytes shape lands
        # here; read() additionally pre-types most of them as StateError.
        except (OSError, ValueError, ValidationError, StateError) as exc:
            log.warning("stage.cache_corrupt", stage=self.name, cache_key=key,
                        path=str(cached), error=str(exc),
                        action="quarantine + recompute")
            self._quarantine_corrupt(cached)
            return None

    @staticmethod
    def _quarantine_corrupt(path: Path) -> None:
        """Move a bad artifact aside (never delete evidence). Best-effort:
        if even the rename fails (file locked), fall through — the recompute
        will atomically overwrite it anyway."""
        try:
            path.replace(path.with_suffix(path.suffix + ".corrupt"))
        except OSError:
            pass

    # ------------------------------------------------------------ bookkeeping

    def _record_failure(self, run_id: int | None, key: str, exc: StageError) -> None:
        """Total: stage_runs must never be left 'running' after a failure.
        If even the DB write fails, log it — never mask the original error."""
        log.error("stage.failed", stage=self.name, cache_key=key,
                  error=str(exc), error_type=type(exc).__name__)
        if run_id is None:
            return
        try:
            self.db.stage_finished(run_id, error=f"{type(exc).__name__}: {exc}")
        except StateError as db_exc:
            log.error("stage.failure_bookkeeping_failed", stage=self.name,
                      run_id=run_id, error=str(db_exc))

    # ----------------------------------------------------------------- hooks

    @abstractmethod
    def _execute(self, *, cache_key: str, params: dict[str, Any],
                 **inputs: Any) -> ArtifactT:
        """Produce the artifact. MUST be deterministic w.r.t. inputs+params.
        MUST stamp ``cache_key`` into the artifact. Raises a typed
        StageError subclass matching the stage's declared failure mode."""
