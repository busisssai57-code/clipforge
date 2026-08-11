"""CP0 deterministic gate — proves the skeleton's laws mechanically.

Checks (all real, all offline):
  1. Atomic writes: a failure injected mid-write leaves the destination
     untouched; crash debris (.partial) is swept without touching artifacts.
  2. Determinism: atomic_write_json + ArtifactModel.to_json_bytes are
     byte-stable across runs and dict orderings; cache_key matches a
     HARDCODED golden value (a format change fails this gate loudly).
  3. Resumability: Stage.run() computes once, then no-ops from cache;
     a corrupted cached artifact is quarantined and recomputed, not fatal.
  4. VRAM Law: co-residency registration raises; budget asserted; lock
     serializes; unload actually drops references (weakref-proven).
  5. State DB: WAL round-trip, dedup semantics, stage-scoped artifact registry.
  6. Config: example TOML parses; unknown keys rejected.

Run: python -m clipforge.verify.skeleton
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import traceback
import weakref
from pathlib import Path
from typing import Callable

from clipforge import paths
from clipforge.config import load_config, load_watchlist
from clipforge.errors import (AtomicWriteError, CoResidencyError, ConfigError,
                              VramBudgetError)
from clipforge.gpu import (GB, GPULock, ModelClass, ResidencyRegistry,
                           hard_unload)
from clipforge.schemas import TranscriptArtifact
from clipforge.stages.base import Stage, digest_bytes
from clipforge.state import StateDB

_CHECKS: list[tuple[str, Callable[[], None]]] = []

# Recompute ONLY on a deliberate cache-format bump, and say so in the diff:
#   sha256("s_golden|" + sha256(b"input") + "|1|" + digest_params({"p": 1}))
GOLDEN_CACHE_KEY = "7b137fedb3639b85a208811bc4ed4e02610bfbaad50b07fd938cb59308dfb956"


def check(name: str):
    def deco(fn: Callable[[], None]):
        _CHECKS.append((name, fn))
        return fn
    return deco


# ---------------------------------------------------------------- 1. atomicity


@check("atomic write: injected failure leaves dest untouched; sweep is safe")
def _atomic() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        dest = root / "artifact.json"
        paths.atomic_write_json(dest, {"b": 2, "a": 1})
        original = dest.read_bytes()

        # Inject a failure at the fsync boundary — mid-"write", pre-replace.
        real_fsync = os.fsync
        os.fsync = lambda fd: (_ for _ in ()).throw(OSError(28, "No space left"))
        try:
            try:
                paths.atomic_write_json(dest, {"clobbered": True})
                raise AssertionError("injected fsync failure must raise")
            except AtomicWriteError:
                pass
        finally:
            os.fsync = real_fsync
        assert dest.read_bytes() == original, "failed write must not touch dest"

        # Crash debris next to a good artifact: swept, artifact untouched.
        debris = root / f"artifact.json.deadbeef{paths.PARTIAL_SUFFIX}"
        debris.write_bytes(b"torn")
        removed = paths.discard_partials(root)
        assert debris in removed and not debris.exists()
        assert dest.read_bytes() == original


# ---------------------------------------------------------------- 2. determinism


@check("byte-determinism of serialization")
def _determinism() -> None:
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a.json", Path(td) / "b.json"
        paths.atomic_write_json(a, {"z": 1, "a": {"y": 2, "x": 3}})
        paths.atomic_write_json(b, {"a": {"x": 3, "y": 2}, "z": 1})  # same content, other order
        assert a.read_bytes() == b.read_bytes(), "dict order leaked into bytes"

    art1 = TranscriptArtifact(cache_key="k", source_path="p", segments=[], turns=[])
    art2 = TranscriptArtifact(turns=[], segments=[], source_path="p", cache_key="k")
    assert art1.to_json_bytes() == art2.to_json_bytes()


@check("cache_key matches hardcoded golden")
def _cache_key_golden() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = StateDB(Path(td) / "s.db")
        try:
            class S(_NullStage):
                name, version = "s_golden", "1"

            s = S(db, Path(td))
            key = s.cache_key(digest_bytes(b"input"), {"p": 1})
            # THE golden pin: a silent cache-format change fails here.
            assert key == GOLDEN_CACHE_KEY, (
                f"cache key format changed: {key} != {GOLDEN_CACHE_KEY}. If "
                "deliberate, bump stage versions and update the golden.")
            assert key != s.cache_key(digest_bytes(b"input2"), {"p": 1})
            assert key != s.cache_key(digest_bytes(b"input"), {"p": 2})
        finally:
            db.close()  # Windows: an open handle blocks temp-dir cleanup


# ---------------------------------------------------------------- 3. resumability


class _NullStage(Stage[TranscriptArtifact]):
    """Minimal concrete stage for exercising the ABC machinery."""

    name = "s_null"
    version = "1"
    artifact_type = TranscriptArtifact
    calls = 0

    def _execute(self, *, cache_key: str, params: dict, **inputs) -> TranscriptArtifact:
        type(self).calls += 1
        # Stamp OUR name: run() enforces artifact.stage == stage.name.
        return TranscriptArtifact(cache_key=cache_key, source_path="x",
                                  stage=self.name)


@check("stage cache: compute once, no-op after, corrupt hit recovers")
def _resume() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = StateDB(Path(td) / "state.db")
        try:
            stage = _NullStage(db, Path(td) / "artifacts")
            _NullStage.calls = 0
            a1 = stage.run(input_digest=digest_bytes(b"in"), params={})
            a2 = stage.run(input_digest=digest_bytes(b"in"), params={})
            assert _NullStage.calls == 1, "second run must be a cache no-op"
            assert a1.cache_key == a2.cache_key
            art_path = stage.artifact_path(a1.cache_key)
            assert art_path.exists()
            assert list(paths.iter_partials(Path(td))) == []

            # Poison-pill protection: corrupt the cached artifact on disk.
            art_path.write_text("{ not json", encoding="utf-8")
            a3 = stage.run(input_digest=digest_bytes(b"in"), params={})
            assert _NullStage.calls == 2, "corrupt cache must recompute"
            assert a3.cache_key == a1.cache_key
            assert art_path.exists() and TranscriptArtifact.read(art_path)
        finally:
            db.close()


# ---------------------------------------------------------------- 4. VRAM law


@check("co-residency raises; budget asserted; lock serializes; unload frees")
def _vram_law() -> None:
    reg = ResidencyRegistry()
    reg.register(ModelClass.ASR)
    try:
        reg.register(ModelClass.VL)
        raise AssertionError("VL over ASR must raise CoResidencyError")
    except CoResidencyError:
        pass
    reg.unregister(ModelClass.ASR)

    # hard_unload must actually drop the reference (the whole point):
    class _FakeModel:
        pass

    model = _FakeModel()
    ref = weakref.ref(model)
    models = {"m": model}
    del model  # dict now holds the only strong reference
    hard_unload(models, "m")
    assert ref() is None, "hard_unload left the model alive"

    async def scenario() -> None:
        # Probe reports 4 GB free → an 8 GB budget must be refused.
        lock = GPULock(vram_probe=lambda: 4 * GB)
        try:
            async with lock.acquire(ModelClass.ASR, budget_gb=8.0):
                raise AssertionError("unreachable")
        except VramBudgetError:
            pass

        # Serialization: second acquire waits until the first exits.
        lock2 = GPULock(vram_probe=lambda: None)  # probe disabled (no CUDA)
        order: list[str] = []

        async def user(tag: str) -> None:
            async with lock2.acquire(ModelClass.ASR, budget_gb=1.0):
                order.append(f"{tag}+")
                await asyncio.sleep(0.01)
                order.append(f"{tag}-")

        await asyncio.gather(user("a"), user("b"))
        assert order in (["a+", "a-", "b+", "b-"], ["b+", "b-", "a+", "a-"]), order
        assert lock2.registry.resident is None

    asyncio.run(scenario())


# ---------------------------------------------------------------- 5. state DB


@check("state DB: WAL, dedup, stage-scoped artifact registry")
def _state_db() -> None:
    with tempfile.TemporaryDirectory() as td:
        db = StateDB(Path(td) / "state.db")
        try:
            # dedup: first sighting is new, second is not (§S0 "never
            # re-download"), scoped per CHANNEL so two channels listing the
            # same video never race each other's downloads.
            assert db.mark_video_seen("youtube", "@c", "abc") is True
            assert db.mark_video_seen("youtube", "@c", "abc") is False
            assert db.mark_video_seen("youtube", "@other", "abc") is True
            # job idempotency by natural key
            j1 = db.upsert_job("vod", "youtube:abc")
            j2 = db.upsert_job("vod", "youtube:abc")
            assert j1 == j2
            # artifact registry: stage-scoped, trusts only existing files
            art = Path(td) / "a.json"
            art.write_text("{}", encoding="utf-8")
            db.record_artifact("k1", "s1", "1", art)
            assert db.lookup_artifact("k1", "s1") == art
            assert db.lookup_artifact("k1", "s2") is None, \
                "stage mismatch must not resolve (DAG Law)"
            art.unlink()
            assert db.lookup_artifact("k1", "s1") is None, \
                "deleted artifact must not resolve"
        finally:
            db.close()


# ---------------------------------------------------------------- 6. config


@check("example config + watchlist parse; unknown keys rejected")
def _config() -> None:
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "config" / "config.example.toml")
    assert cfg.orchestration.gpu_concurrency == 1, "GPU concurrency is LAW: 1"
    wl = load_watchlist(repo / "config" / "channels.toml")
    assert isinstance(wl.enabled_sorted(), list)

    with tempfile.TemporaryDirectory() as td:
        bad = Path(td) / "bad.toml"
        bad.write_text("[workspace]\nroot = 'w'\ntypod_key = 1\n", encoding="utf-8")
        try:
            load_config(bad)
            raise AssertionError("unknown key must be rejected")
        except ConfigError:
            pass


# ---------------------------------------------------------------- runner


def main() -> int:
    failures = 0
    for name, fn in _CHECKS:
        try:
            fn()
            print(f"[PASS] {name}")
        except Exception:
            failures += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    total = len(_CHECKS)
    print(f"skeleton verify: {total - failures}/{total} passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
