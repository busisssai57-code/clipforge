"""VRAM Law mechanics — no torch, no GPU: probes are injected."""

import asyncio
import weakref

import pytest

from clipforge.errors import CoResidencyError, VramBudgetError
from clipforge.gpu import (GB, GPULock, ModelClass, ResidencyRegistry,
                           hard_unload, vram_guard)


def test_registry_single_residency():
    reg = ResidencyRegistry()
    reg.register(ModelClass.ASR)
    assert reg.resident is ModelClass.ASR
    with pytest.raises(CoResidencyError):
        reg.register(ModelClass.VL)  # the spec's "deliberate co-load must raise"
    reg.unregister(ModelClass.ASR)
    reg.register(ModelClass.VL)  # legal after unload
    reg.unregister(ModelClass.VL)


def test_registry_same_class_reregister_is_tolerated():
    reg = ResidencyRegistry()
    reg.register(ModelClass.ASR)
    reg.register(ModelClass.ASR)  # idempotent, not a violation
    reg.unregister(ModelClass.ASR)


def test_registry_unregister_wrong_class_raises():
    reg = ResidencyRegistry()
    reg.register(ModelClass.ASR)
    with pytest.raises(CoResidencyError):
        reg.unregister(ModelClass.VL)


async def test_lock_asserts_budget_before_load():
    lock = GPULock(vram_probe=lambda: 4 * GB)
    with pytest.raises(VramBudgetError):
        async with lock.acquire(ModelClass.ASR, budget_gb=8.0):
            raise AssertionError("must not enter with insufficient VRAM")
    # Failure path must not leak the semaphore or residency:
    assert lock.registry.resident is None
    async with lock.acquire(ModelClass.ASR, budget_gb=1.0):
        assert lock.registry.resident is ModelClass.ASR
    assert lock.registry.resident is None


async def test_lock_serializes_gpu_users():
    lock = GPULock(vram_probe=lambda: None)  # probe disabled (no CUDA machine)
    active = 0
    max_active = 0

    async def user() -> None:
        nonlocal active, max_active
        async with lock.acquire(ModelClass.VL, budget_gb=1.0):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.005)
            active -= 1

    await asyncio.gather(*(user() for _ in range(5)))
    assert max_active == 1, "GPULock must serialize all GPU work"


async def test_lock_releases_residency_on_exception():
    lock = GPULock(vram_probe=lambda: None)
    with pytest.raises(RuntimeError):
        async with lock.acquire(ModelClass.ASR, budget_gb=1.0):
            raise RuntimeError("stage blew up")
    assert lock.registry.resident is None
    # And the semaphore is free again:
    async with lock.acquire(ModelClass.VL, budget_gb=1.0):
        pass


class _FakeModel:
    """Weakref-able stand-in for a loaded model."""


def test_hard_unload_actually_drops_references():
    """THE unload guarantee: after hard_unload, the model object is dead —
    not merely unbound in some scope. This is what makes empty_cache able
    to return the tensors' blocks on a real GPU."""
    model = _FakeModel()
    ref = weakref.ref(model)
    models = {"asr": model}
    del model  # the dict is now the only strong reference
    hard_unload(models, "asr")
    assert ref() is None, "hard_unload must drop the last strong reference"
    assert "asr" not in models


def test_hard_unload_respects_order_and_missing_names():
    """Unload order is the caller's (spec §S1: align → diarize → asr);
    missing names are tolerated (finally-block after partial construction)."""
    popped: list[str] = []

    class Tracking(dict):
        def pop(self, key, *default):
            popped.append(key)
            return super().pop(key, *default)

    models = Tracking(align=_FakeModel(), asr=_FakeModel())
    hard_unload(models, "align", "diar", "asr")  # 'diar' never got loaded
    assert popped == ["align", "diar", "asr"]
    assert not models


def test_hard_unload_no_names_unloads_everything_sorted():
    models = {"b": _FakeModel(), "a": _FakeModel()}
    refs = {k: weakref.ref(v) for k, v in models.items()}
    hard_unload(models)
    assert not models
    assert all(r() is None for r in refs.values())


def test_vram_guard_raises_under_budget_and_passes_over():
    with pytest.raises(VramBudgetError):
        vram_guard(8.0, ModelClass.ASR, probe=lambda: 4 * GB)
    vram_guard(2.0, ModelClass.ASR, probe=lambda: 4 * GB)  # no raise
    vram_guard(99.0, ModelClass.ASR, probe=lambda: None)   # probe absent: warn-only
