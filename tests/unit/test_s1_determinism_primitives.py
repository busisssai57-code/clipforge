"""Spec §3.2 determinism primitives in S1, pinned.

CP2 round-1 finding DET-1: `torch.manual_seed(0)` before diarizer
construction, `segments.sort(key=(start, end))` in `_build_artifact`, and the
`sorted()` in `turns_of` were each deletable with 336 passed and GATE PASSED.
No fixture anywhere fed S1 out-of-order segments or turns, so the sorts were
never exercised — they were present but untested, which is indistinguishable
from absent.

Also C2-1-ORDER: only the PRESENCE of gc.collect()/empty_cache() was pinned,
never their order, and never that the caller's `del` precedes them.
"""

from __future__ import annotations

import gc
import sys
import types
import weakref
from typing import Any

import pytest

from clipforge.stages.base import digest_bytes
from clipforge.stages.s1_transcribe import S1Transcribe

from tests.unit.test_s1_transcribe import FakeEngine, env, make_stage  # noqa: F401


def test_segments_are_emitted_in_sorted_order(env):
    """Feed _build_artifact deliberately out-of-order aligned segments."""
    db, media, tmp = env

    class OutOfOrder(FakeEngine):
        def align(self, aligner, segments, media_path):
            self.calls.append("align")
            def _seg(start, end, text):
                return {"start": start, "end": end, "text": text,
                        "words": [{"word": text, "start": start,
                                   "end": end, "score": 0.9}]}
            return {"segments": [_seg(9.0, 10.0, "third."),
                                 _seg(1.0, 2.0, "first."),
                                 _seg(5.0, 6.0, "second.")]}

    art = make_stage(db, tmp, OutOfOrder([])).run(
        input_digest=digest_bytes(b"m"), params={}, media_path=media)
    starts = [s.start for s in art.segments]
    assert starts == sorted(starts), (
        f"segments emitted out of order: {starts} — the §3.2 sort is gone")
    assert [s.text for s in art.segments] == ["first.", "second.", "third."]


def test_diarization_turns_are_emitted_in_sorted_order(env):
    db, media, tmp = env

    class OutOfOrderTurns(FakeEngine):
        def turns_of(self, diarization):
            return [("B", 9.0, 10.0), ("A", 1.0, 2.0), ("A", 5.0, 6.0)]

    art = make_stage(db, tmp, OutOfOrderTurns([])).run(
        input_digest=digest_bytes(b"m"), params={}, media_path=media)
    keys = [(t.start, t.speaker) for t in art.turns]
    assert keys == sorted(keys), f"turns emitted out of order: {keys}"


def test_the_seed_is_pinned_before_the_diarizer_is_constructed(env):
    """§3.2 pins seeds ahead of clustering. The real engine's load_diarizer
    calls torch.manual_seed(0); nothing asserted it, so deleting the line was
    invisible."""
    import inspect

    from clipforge.stages.s1_transcribe import WhisperXEngine

    src = inspect.getsource(WhisperXEngine.load_diarizer)
    seed_at = src.find("manual_seed(0)")
    build_at = max(src.find("pipeline_cls("), src.find("DiarizationPipeline("))
    assert seed_at != -1, (
        "torch.manual_seed(0) is gone from load_diarizer; pyannote's "
        "clustering is seeded, so without it S1 is not reproducible")
    assert build_at != -1 and seed_at < build_at, (
        "the seed is pinned AFTER the pipeline is constructed, which is too "
        "late to affect its initialization")


def test_the_teardown_ritual_runs_in_the_documented_order(env):
    """del -> gc.collect() -> empty_cache(), per phase.

    Only the presence of the two calls was pinned. Reordering `_unload` to
    empty_cache-then-gc.collect left the whole gate green — yet flushing the
    allocator before collecting means the blocks the collection frees are
    still held by the caching allocator when the flush happens.
    """
    db, media, tmp = env
    order: list[str] = []

    real_collect = gc.collect

    def tracking_collect(*a, **kw):
        order.append("gc")
        return real_collect(*a, **kw)

    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(  # type: ignore[attr-defined]
        is_available=lambda: True,
        device_count=lambda: 1,
        mem_get_info=lambda *a: (20 * 1024 ** 3, 24 * 1024 ** 3),
        memory_allocated=lambda *a: 0,
        max_memory_allocated=lambda *a: 0,
        reset_peak_memory_stats=lambda *a: None,
        empty_cache=lambda: order.append("flush"))

    real_torch = sys.modules.get("torch")
    sys.modules["torch"] = fake
    gc.collect = tracking_collect  # type: ignore[assignment]
    try:
        make_stage(db, tmp, FakeEngine([])).run(
            input_digest=digest_bytes(b"m"), params={}, media_path=media)
    finally:
        gc.collect = real_collect  # type: ignore[assignment]
        if real_torch is not None:
            sys.modules["torch"] = real_torch
        else:
            del sys.modules["torch"]

    # Each phase contributes exactly gc-then-flush, in that order.
    phases = [order[i:i + 2] for i in range(0, 6, 2)]
    assert phases == [["gc", "flush"]] * 3, (
        f"ritual order violated: {order[:6]} — collect must precede the "
        "allocator flush, or the flush cannot return what collect frees")


def test_the_last_phases_handle_is_dead_the_moment_its_unload_returns(env):
    """C2-1-ORDER. The earlier cyclic test used the ASR model, whose cycle is
    broken by the NEXT phase's gc.collect() — so swapping `del` and
    `_unload()` in every phase left the gate green. The DIARIZER is the last
    phase: nothing runs after it, so only its own teardown can free it."""
    db, media, tmp = env
    seen: dict[str, Any] = {}

    class Cyclic:
        def __init__(self) -> None:
            self.loop = self

    class CyclicDiarizer(FakeEngine):
        def load_diarizer(self, hf_token):
            obj = Cyclic()
            seen["ref"] = weakref.ref(obj)
            self.refs["diar"] = seen["ref"]
            return obj

    gc.disable()
    try:
        make_stage(db, tmp, CyclicDiarizer([])).run(
            input_digest=digest_bytes(b"m"), params={}, media_path=media)
        assert seen["ref"]() is None, (
            "the diarizer survived its own phase teardown; no later phase "
            "exists to rescue it, so this is the real per-phase guarantee")
    finally:
        gc.enable()
