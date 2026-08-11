"""GPU governance — the VRAM Law lives here (spec §3.1).

Three mechanisms, layered:

  1. :class:`GPULock` — a global asyncio semaphore of count **1**. Every
     stage that touches CUDA acquires it. Acquiring runs :func:`vram_guard`
     (free VRAM ≥ the stage's declared budget) *before* any model load.
  2. :class:`ResidencyRegistry` — bookkeeping of which model *class* is
     resident. Registering a second class while one is held raises
     :class:`CoResidencyError` immediately — a deliberate co-load attempt
     must raise, not OOM twenty minutes later.
  3. :func:`hard_unload` — the mandated ``del`` → ``gc.collect()`` →
     ``torch.cuda.empty_cache()`` ritual, designed to be called from
     ``finally:`` so it survives exceptions.

The unload discipline (load-bearing — read this before writing a stage):
a Python function CANNOT free objects its caller still references, so
stages must keep every loaded model in a dict and NEVER in bare locals::

    models: dict[str, Any] = {}
    try:
        models["asr"] = whisperx.load_model(...)
        models["align"] = whisperx.load_align_model(...)
        ... use models["asr"] ...
    finally:
        hard_unload(models, "align", "asr")   # in the stage's unload ORDER

``hard_unload`` pops the dict entries (dropping the only strong reference),
then ``gc.collect()`` + ``empty_cache()`` — the true del→gc→empty_cache
order. Holding a model in a local while calling this defeats it; the
weakref-based unit test pins the working pattern.

torch is imported lazily: unit tests (no GPU, no torch) exercise the
registry and lock logic with the CUDA probes stubbed out.
"""

from __future__ import annotations

import asyncio
import gc
import sys
import threading
import time
from contextlib import asynccontextmanager, contextmanager
from enum import Enum
from pathlib import Path
from typing import Any, AsyncIterator, Callable, MutableMapping

from clipforge.errors import CoResidencyError, VramBudgetError
from clipforge.log import get_logger

log = get_logger(__name__)

GB = 1024 ** 3


class ModelClass(str, Enum):
    """One resident class at a time; a class is a STAGE's whole model suite.

    S1's window covers WhisperX ASR + alignment + pyannote diarization —
    they load/unload sequentially inside ONE ``GPULock.acquire(ASR, ...)``
    window under one 8 GB budget (spec §5). Giving diarization its own class
    would either trip the registry or deadlock the non-reentrant semaphore,
    so it is deliberately NOT a separate class. The law's hard pair — VL
    and ASR never co-resident — is exactly what distinct classes enforce.
    """

    ASR = "asr"    # S1 suite: WhisperX ASR + alignment + pyannote diarization
    VL = "vl"      # S3: Qwen2.5-VL
    POSE = "pose"  # S4: YOLO11-pose + ByteTrack


def _torch() -> Any | None:
    """Lazy torch import. Returns None when torch/CUDA is unavailable so the
    pure-logic paths (registry, lock) stay testable on CPU-only machines."""
    try:
        import torch  # noqa: PLC0415

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def free_vram_bytes() -> int | None:
    """Free bytes on cuda:0, or None when CUDA is absent (probe disabled)."""
    torch = _torch()
    if torch is None:
        return None
    free, _total = torch.cuda.mem_get_info(0)
    return int(free)


def vram_guard(budget_gb: float, model_class: ModelClass, *,
               probe: Callable[[], int | None] = free_vram_bytes) -> None:
    """Assert free VRAM ≥ the declared stage budget BEFORE loading anything.

    Raises :class:`VramBudgetError` on violation. When the probe is
    unavailable (no CUDA — CI, unit tests) the assertion cannot run; that
    is logged LOUDLY rather than silently skipped, because on a production
    box a dead probe means the law is unenforced and the operator must know.
    """
    free = probe()
    if free is None:
        log.warning("gpu.vram_probe_unavailable",
                    model_class=model_class.value, budget_gb=budget_gb,
                    note="CUDA probe absent - VRAM budget NOT asserted")
        return
    if free < budget_gb * GB:
        raise VramBudgetError(
            f"Free VRAM {free / GB:.1f} GB < declared budget "
            f"{budget_gb:.1f} GB for {model_class.value!r}. "
            "A previous stage leaked memory - refusing to load.")


class ResidencyRegistry:
    """Tracks the single resident model class. Synchronous and tiny on purpose —
    it is consulted under the GPULock, so no extra locking is needed, but it
    still hard-fails on misuse from any code path."""

    def __init__(self,
                 probe: Callable[[], int | None] = free_vram_bytes) -> None:
        #: The VRAM probe both law layers must consult. It lives HERE because
        #: the registry is the one object the two layers share: GPULock took a
        #: `vram_probe` for injection but `gpu_session` called
        #: `vram_guard(budget, cls)` with no probe, so the inner layer always
        #: used the real CUDA probe. Composition checks that injected a stub
        #: (verify/ai.py's `GPULock(vram_probe=lambda: 24 * GB)` then
        #: `gpu_session(..., registry=lock.registry)`) were silently
        #: exercising real hardware instead of the stub they declared.
        self.probe = probe
        self._resident: ModelClass | None = None
        #: Nesting depth for the SAME class. The two law-enforcing layers
        #: are designed to compose — the orchestrator takes `GPULock.acquire`
        #: before dispatching a stage, and the stage itself opens a
        #: `gpu_session` on the worker thread — so the same class is
        #: legitimately registered twice. Without counting, the inner exit
        #: cleared residency and the OUTER exit then raised
        #: `unregister('asr') but resident is None`, deterministically, on
        #: every run of the documented CP5 deployment (and, being a raise
        #: from a `finally`, it would have masked any real stage exception).
        self._depth = 0

    @property
    def resident(self) -> ModelClass | None:
        return self._resident

    @property
    def depth(self) -> int:
        """How many nested holders currently claim the resident class."""
        return self._depth

    def register(self, cls: ModelClass) -> None:
        if self._resident is not None and self._resident is not cls:
            raise CoResidencyError(
                f"VRAM Law violation: attempted to load {cls.value!r} while "
                f"{self._resident.value!r} is resident. Unload first."
            )
        self._resident = cls
        self._depth += 1

    def unregister(self, cls: ModelClass) -> None:
        # Unregistering a non-resident class is a bug worth surfacing loudly.
        if self._resident is not cls:
            raise CoResidencyError(
                f"unregister({cls.value!r}) but resident is "
                f"{self._resident.value if self._resident else None!r}"
            )
        self._depth -= 1
        if self._depth <= 0:
            self._resident = None
            self._depth = 0


class GPULock:
    """Global GPU gate: semaphore(1) + registry + VRAM budget assertion."""

    def __init__(self, *, vram_probe: Callable[[], int | None] = free_vram_bytes) -> None:
        self._sem = asyncio.Semaphore(1)
        # The probe rides on the registry so BOTH layers see it (see
        # ResidencyRegistry.probe).
        self.registry = ResidencyRegistry(probe=vram_probe)
        self._vram_probe = vram_probe

    @asynccontextmanager
    async def acquire(self, model_class: ModelClass, budget_gb: float) -> AsyncIterator[None]:
        """Serialize GPU work. Order of operations is load-bearing:

        semaphore → vram_guard → residency register → [caller loads models]
        … caller work …
        finally: last-resort empty_cache + peak log → residency unregister
                 → semaphore release

        The caller MUST free its own models via :func:`hard_unload` (dict
        discipline — see module docstring) inside its own ``finally`` before
        this context exits: we cannot drop references we never held. This
        context's job is lock discipline, the budget gate, peak logging, and
        a last-resort cache flush.
        """
        async with self._sem:
            vram_guard(budget_gb, model_class, probe=self._vram_probe)
            self.registry.register(model_class)
            torch = _torch()
            try:
                # INSIDE the try — see gpu_session for why: a raise here (a
                # sticky CUDA context) between register() and the try leaked
                # the registration permanently, with no finally to undo it.
                if torch is not None:
                    torch.cuda.reset_peak_memory_stats()
                yield
            finally:
                # Survives exceptions: registry always released, cache always
                # flushed, peak always logged (observability requirement §S1).
                try:
                    if torch is not None:
                        peak = torch.cuda.max_memory_allocated()
                        log.info("gpu.stage_peak", model_class=model_class.value,
                                 peak_gb=round(peak / GB, 3), budget_gb=budget_gb)
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                finally:
                    self.registry.unregister(model_class)


def hard_unload(models: MutableMapping[str, Any], *names: str) -> None:
    """The mandated unload ritual: drop refs → gc → empty_cache.

    ``models`` is the stage's model dict — the ONLY strong reference holder
    under the store-in-dict discipline (module docstring). ``names`` gives
    the unload order (spec §S1: alignment, diarization, ASR); empty names =
    everything, sorted for determinism. Popping the dict entry drops the
    reference *before* ``gc.collect()`` runs, so ``empty_cache()`` can
    actually return the tensors' blocks — this ordering is the entire point
    and is pinned by a weakref unit test.

    Missing names are ignored (partially-constructed stage state after an
    exception is the normal caller context — this runs in ``finally:``).
    """
    keys = list(names) if names else sorted(models.keys())
    for key in keys:
        models.pop(key, None)
    gc.collect()
    torch = _torch()
    if torch is not None:
        torch.cuda.empty_cache()


@contextmanager
def gpu_session(model_class: ModelClass, budget_gb: float, *,
                registry: ResidencyRegistry | None = None):
    """SYNCHRONOUS residency window for a stage's model suite.

    The concurrency story has two layers, split on purpose:

      * ``GPULock`` (asyncio, semaphore of 1) serializes GPU work BETWEEN
        tasks — the orchestrator acquires it before dispatching a GPU stage
        to a worker thread (CP5).
      * ``gpu_session`` (this) enforces the VRAM Law INSIDE the stage,
        which runs synchronously on that worker thread where an asyncio
        semaphore cannot be awaited: budget assert before any load,
        residency registration (a co-load attempt raises), and the
        peak-log + ``empty_cache`` + unregister ritual in ``finally``.

    Both layers consult the SAME registry, so a stage invoked standalone
    (``clipforge process``, tests) gets the law's teeth without the
    orchestrator, and a co-load under the orchestrator still raises.

    The caller MUST free its own models via :func:`hard_unload` (dict
    discipline, module docstring) before this context exits.
    """
    reg = registry if registry is not None else GPU_LOCK.registry
    # MUTUAL EXCLUSION. The spec's law says "semaphore(1)", and GPULock
    # provides that for asyncio — but gpu_session is the layer the STAGES
    # actually use, on the worker thread, where an asyncio semaphore cannot
    # be awaited. It had no lock at all. Depth counting alone cannot tell
    # "the same holder re-entering" from "a second concurrent holder", so two
    # threads each opened an ASR session and each loaded a full model suite
    # with no error raised (measured: 2 co-resident suites, 512 MB, registry
    # ending clean so nothing downstream ever learned). The budget assert was
    # racy for the same reason: both threads checked before either allocated.
    #
    # Re-entry by the SAME thread must not deadlock — that is the composition
    # the depth counter exists for — so this is a reentrant lock, and the
    # budget assert happens under it so check and load are atomic.
    with _SESSION_LOCK, _process_gpu_lock():
        # Budget FIRST, then registration. Reversing them leaks residency
        # permanently: VramBudgetError propagates before the try/finally is
        # entered, so unregister never runs and the module-level singleton
        # wedges every later stage of a different class.
        vram_guard(budget_gb, model_class, probe=reg.probe)
        reg.register(model_class)
        torch = _torch()
        try:
            # INSIDE the try. This call can raise on a sticky CUDA context
            # (after an illegal access or a prior OOM) — exactly the state
            # this code exists to survive — and outside the try that raise
            # leaked the registration with no finally to undo it.
            if torch is not None:
                torch.cuda.reset_peak_memory_stats()
            yield
        finally:
            try:
                if torch is not None:
                    peak = torch.cuda.max_memory_allocated()
                    log.info("gpu.stage_peak", model_class=model_class.value,
                             peak_gb=round(peak / GB, 3), budget_gb=budget_gb)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            finally:
                reg.unregister(model_class)


#: Path of the cross-PROCESS GPU lock, set at boot once the workspace is
#: known. None = single-process mode (tests, library use), where the thread
#: lock alone is the whole story.
_PROCESS_LOCK_PATH: Path | None = None

#: Re-entry depth for the process lock on THIS thread. gpu_session composes
#: (an orchestrator session wrapping a stage session), and a file lock is
#: not reentrant, so the second acquire would deadlock against itself.
_PROCESS_LOCK_DEPTH = threading.local()


def configure_process_gpu_lock(path: Path | str | None) -> None:
    """Point the cross-process GPU lock at a workspace file.

    The VRAM Law says one GPU stage at a time, and until now that was
    enforced only WITHIN a process — by an asyncio semaphore, a thread
    lock, and a single dispatcher worker, none of which a subprocess
    inherits. The dashboard spawns `bta generate` as its own process, so a
    dashboard render and a `bta swarm serve` render each held "the" permit
    and both loaded a model onto the same card.
    """
    global _PROCESS_LOCK_PATH
    _PROCESS_LOCK_PATH = Path(path) if path is not None else None


@contextmanager
def _process_gpu_lock(poll_s: float = 1.0):
    """Block until this machine's GPU is free, then hold it.

    An OS advisory lock, matching `ingest.retention.workspace_lock`: the
    kernel drops it however the process dies, so a hard kill cannot strand
    a lock file and wedge every later run. BLOCKING rather than fail-fast
    — a queued render is correct behaviour, a refused one is not.
    """
    path = _PROCESS_LOCK_PATH
    if path is None:
        yield
        return

    depth = getattr(_PROCESS_LOCK_DEPTH, "value", 0)
    if depth:                       # already ours on this thread
        _PROCESS_LOCK_DEPTH.value = depth + 1
        try:
            yield
        finally:
            _PROCESS_LOCK_DEPTH.value = depth
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "a+b")
    waited = 0.0
    try:
        while True:
            try:
                handle.seek(0)
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:  # pragma: no cover - POSIX
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if waited == 0.0:
                    log.info("gpu.waiting_for_another_process",
                             lock=str(path),
                             note="another BTA process holds the GPU")
                time.sleep(poll_s)
                waited += poll_s
        if waited:
            log.info("gpu.acquired_after_wait", waited_s=round(waited, 1))
        _PROCESS_LOCK_DEPTH.value = 1
        try:
            yield
        finally:
            _PROCESS_LOCK_DEPTH.value = 0
            try:
                handle.seek(0)
                if sys.platform == "win32":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:  # pragma: no cover - POSIX
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass
    finally:
        handle.close()


#: Serializes gpu_session bodies across THREADS. Reentrant so that the same
#: thread re-entering (the orchestrator/stage composition the depth counter
#: exists for) does not deadlock, while a genuinely concurrent holder blocks.
_SESSION_LOCK = threading.RLock()

# Module-level singleton: one process, one GPU, one lock (spec §6).
GPU_LOCK = GPULock()
