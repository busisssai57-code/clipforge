"""CP2 adversarial round 1 findings â€” each test fails if its fix is reverted.

Every finding here was measured by an independent reviewer against the frozen
tree. Assertions compare against HARDCODED values, never against the constant
under test: that self-referential pattern has now shipped four times in this
project.
"""

from __future__ import annotations

import gc
import threading
import weakref

import pytest

from clipforge import gpu
from clipforge.errors import CoResidencyError, VramBudgetError
from clipforge.gpu import GB, ModelClass, ResidencyRegistry, gpu_session, hard_unload
from clipforge.schemas import CandidateWindow
from clipforge.stages.s2_prefilter import Sentence, generate_windows, nms


class Cyclic:
    """Shaped like a torch Module: unreachable by refcounting alone."""

    def __init__(self) -> None:
        self.loop = self


# --------------------------------------------------------------------------
# GPU-1: hard_unload's gc.collect() was pinned by nothing, because both the
# unit test and verify/skeleton.py used a refcount-freeable stand-in. It is
# live in s3_semantic.py and s4_tracking.py, where the objects ARE Modules.
# --------------------------------------------------------------------------


def test_hard_unload_reclaims_a_cyclic_model():
    obj = Cyclic()
    ref = weakref.ref(obj)
    handles = {"model": obj}
    del obj
    gc.disable()
    try:
        hard_unload(handles)
        assert ref() is None, (
            "hard_unload left a cyclic (torch-Module-shaped) model alive; "
            "`del` drops a refcount but only gc.collect() breaks the cycle")
    finally:
        gc.enable()


# --------------------------------------------------------------------------
# GPU-2 / VRAM-04: budget assert BEFORE registration, and nothing between
# registration and the try that could leak it.
# --------------------------------------------------------------------------


def test_budget_failure_leaves_no_residency_behind():
    reg = ResidencyRegistry(probe=lambda: int(4 * GB))
    with pytest.raises(VramBudgetError):
        with gpu_session(ModelClass.ASR, 8.0, registry=reg):
            pytest.fail("body must not run when the budget assertion fires")
    assert reg.resident is None, (
        "a failed budget assertion registered residency anyway; the module "
        "level registry is a singleton, so this wedges the process")
    assert reg.depth == 0
    # ...and the registry is still usable for a DIFFERENT class.
    reg2 = ResidencyRegistry(probe=lambda: int(24 * GB))
    with gpu_session(ModelClass.VL, 8.0, registry=reg2):
        assert reg2.resident is ModelClass.VL


def test_a_raise_between_register_and_yield_still_unregisters(monkeypatch):
    """A sticky CUDA context makes reset_peak_memory_stats raise â€” exactly
    the state this code exists to survive. Outside the try, that raise leaked
    the registration permanently."""
    reg = ResidencyRegistry(probe=lambda: int(24 * GB))

    class _Boom:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def reset_peak_memory_stats(*_a):
            raise RuntimeError("CUDA error: an illegal memory access")

        @staticmethod
        def max_memory_allocated(*_a) -> int:
            return 0

        @staticmethod
        def empty_cache() -> None:
            return None

    class _FakeTorch:
        cuda = _Boom()

    monkeypatch.setattr(gpu, "_torch", lambda: _FakeTorch())

    with pytest.raises(RuntimeError):
        with gpu_session(ModelClass.ASR, 8.0, registry=reg):
            pytest.fail("body must not run")

    assert reg.resident is None and reg.depth == 0, (
        f"registry wedged at resident={reg.resident} depth={reg.depth}")


# --------------------------------------------------------------------------
# VRAM-05: gpu_session had NO mutual exclusion. Two threads each opened an
# ASR session and each loaded a suite, no error raised.
# --------------------------------------------------------------------------


def test_two_threads_cannot_hold_a_gpu_session_at_once():
    reg = ResidencyRegistry(probe=lambda: int(24 * GB))
    barrier = threading.Barrier(2, timeout=5.0)
    overlapped: list[bool] = []
    inside = threading.Semaphore(0)
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            with gpu_session(ModelClass.ASR, 8.0, registry=reg):
                # If the lock works, only one thread is ever here, so the
                # barrier must TIME OUT rather than trip.
                try:
                    barrier.wait()
                    overlapped.append(True)
                except threading.BrokenBarrierError:
                    pass
                inside.release()
        except BaseException as exc:  # noqa: BLE001 - reported, not swallowed
            errors.append(exc)
            inside.release()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, errors
    assert not overlapped, (
        "two threads were inside a gpu_session simultaneously; the VRAM Law "
        "says semaphore(1) and depth counting cannot tell a second concurrent "
        "holder from the same holder re-entering")
    assert reg.resident is None and reg.depth == 0


def test_the_same_thread_may_still_re_enter():
    """The composition the depth counter exists for must not deadlock."""
    reg = ResidencyRegistry(probe=lambda: int(24 * GB))
    with gpu_session(ModelClass.ASR, 8.0, registry=reg):
        with gpu_session(ModelClass.ASR, 8.0, registry=reg):
            assert reg.depth == 2
        assert reg.depth == 1
    assert reg.resident is None and reg.depth == 0


# --------------------------------------------------------------------------
# VRAM-06: gpu_session ignored the injected probe, so composition checks that
# declared a stub were silently exercising real hardware.
# --------------------------------------------------------------------------


def test_gpu_session_consults_the_registrys_probe():
    calls: list[str] = []

    def probe() -> int:
        calls.append("probe")
        return int(24 * GB)

    reg = ResidencyRegistry(probe=probe)
    with gpu_session(ModelClass.ASR, 8.0, registry=reg):
        pass
    assert calls == ["probe"], (
        "gpu_session did not use the injected probe; any test that injects "
        "one is testing the real CUDA probe instead")


# --------------------------------------------------------------------------
# VRAM-01: S3/S4 must use gpu_session and REGISTER residency. A programming
# error must never be absorbed into a quality fallback.
# --------------------------------------------------------------------------


def test_s3_and_s4_use_a_real_residency_window():
    """Checked on the AST, not on source text â€” the surrounding comments
    legitimately mention the broken form, and a substring match cannot tell
    a comment from a call."""
    import ast
    import inspect

    from clipforge.stages import s3_semantic, s4_tracking

    def _with_callees(module) -> set[str]:
        tree = ast.parse(inspect.getsource(module))
        out: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, (ast.With, ast.AsyncWith)):
                continue
            for item in node.items:
                call = item.context_expr
                if isinstance(call, ast.Call):
                    out.add(ast.unparse(call.func))
        return out

    for module, member in ((s3_semantic, "VL"), (s4_tracking, "POSE")):
        callees = _with_callees(module)
        assert "gpu_session" in callees, (
            f"{module.__name__} must open a residency window; it uses "
            f"{sorted(callees)}")
        assert "vram_guard" not in callees, (
            f"{module.__name__} uses vram_guard as a context manager â€” it is "
            "a plain function, so this raises TypeError before the guarded "
            "block runs and the blanket except turns it into a silent "
            "fallback")
        assert f"ModelClass.{member}" in inspect.getsource(module), (
            f"{module.__name__} must register ModelClass.{member}")


def test_model_class_has_no_phantom_members():
    """S4 named ModelClass.DETECTION, which does not exist."""
    assert {m.name for m in ModelClass} == {"ASR", "VL", "POSE"}


def test_programming_errors_are_not_absorbed_into_fallbacks():
    """Pins the constant AND its use.

    Asserting only `_PROGRAMMING_ERRORS == (...)` left the `except` clause
    free to ignore it: changing `except _PROGRAMMING_ERRORS:` to `except ():`
    kept the whole gate green. Pinning an ingredient is not pinning the
    wiring — the same distinction that made C2-3 revert-safe.
    """
    import ast
    import inspect

    from clipforge.stages import s3_semantic, s4_tracking

    for module in (s3_semantic, s4_tracking):
        assert module._PROGRAMMING_ERRORS == (TypeError, AttributeError,
                                              NameError), module.__name__
        handlers = [
            ast.unparse(h.type) if h.type is not None else "bare"
            for node in ast.walk(ast.parse(inspect.getsource(module)))
            if isinstance(node, ast.Try) for h in node.handlers]
        assert "_PROGRAMMING_ERRORS" in handlers, (
            f"{module.__name__} never catches _PROGRAMMING_ERRORS to re-raise "
            f"them; its handlers are {handlers}. A misused API would be "
            "absorbed into a quality fallback again.")


@pytest.mark.parametrize("stage_mod,fallback_marker", [
    ("s3_semantic", "_heuristic_fallback"),
    ("s4_tracking", "_build_center_crop"),
])
def test_the_reraise_precedes_the_blanket_handler(stage_mod, fallback_marker):
    """Order matters: `except Exception` first would shadow the re-raise."""
    import ast
    import inspect
    import importlib

    module = importlib.import_module(f"clipforge.stages.{stage_mod}")
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if not isinstance(node, ast.Try):
            continue
        names = [ast.unparse(h.type) if h.type is not None else "bare"
                 for h in node.handlers]
        if "_PROGRAMMING_ERRORS" in names and "Exception" in names:
            assert names.index("_PROGRAMMING_ERRORS") < names.index("Exception"), (
                f"{stage_mod}: the blanket handler precedes the re-raise, so "
                "programming errors are still swallowed")
            return
    pytest.fail(f"{stage_mod} has no try that both re-raises and falls back")


# --------------------------------------------------------------------------
# S2-D1: `break` assumed monotone sentence ends and dropped legal windows.
# --------------------------------------------------------------------------


def _sent(start: float, end: float, text: str = "x") -> Sentence:
    return Sentence(start=start, end=end, text=text, speaker=None,
                    terminal=True, turn_start=True)


def test_non_monotone_sentence_ends_do_not_drop_legal_windows():
    """One stray word timestamp (a routine WhisperX alignment failure) makes
    a later sentence end EARLIER than an earlier one."""
    sentences = [_sent(0.0, 1.0), _sent(2.0, 3.0), _sent(60.5, 61.0),
                 _sent(44.0, 45.0), _sent(50.0, 50.5)]
    got = generate_windows(sentences, min_s=30.0, max_s=60.0)

    brute = [(i, j)
             for i in range(len(sentences))
             for j in range(i, len(sentences))
             if 30.0 <= sentences[j].end - sentences[i].start <= 60.0]
    assert sorted(got) == sorted(brute), (
        f"missed {sorted(set(brute) - set(got))} â€” the scan broke early on a "
        "non-monotone sentence end")


# --------------------------------------------------------------------------
# S2-D2: the 30/60 s bounds were tested on raw float seconds while iou() one
# function away was quantized for exactly this reason.
# --------------------------------------------------------------------------


#: Offsets where the raw float subtraction ``end - start`` does NOT return the
#: exact length: 2.3 gives 29.999999999999996 (window dropped) and 2.2 gives
#: 30.000000000000004 (kept). Found by search, not by guessing — my first
#: choice of offsets was exactly representable, so the test passed against its
#: own mutant and the sweep reported the fix REVERT-SAFE.
_FRAGILE_OFFSETS = [0.0, 2.2, 2.3, 2.7, 2.8, 3.3]


@pytest.mark.parametrize("offset", _FRAGILE_OFFSETS)
def test_an_exactly_minimum_length_window_is_kept_at_every_offset(offset):
    sentences = [_sent(offset, offset + 1.0), _sent(offset + 29.0, offset + 30.0)]
    got = generate_windows(sentences, min_s=30.0, max_s=60.0)
    assert (0, 1) in got, (
        f"offset {offset}: a window of exactly 30.000 s was dropped "
        f"(raw float length {sentences[1].end - sentences[0].start!r})")


@pytest.mark.parametrize("offset", _FRAGILE_OFFSETS)
def test_an_exactly_maximum_length_window_is_kept_at_every_offset(offset):
    sentences = [_sent(offset, offset + 1.0), _sent(offset + 59.0, offset + 60.0)]
    got = generate_windows(sentences, min_s=30.0, max_s=60.0)
    assert (0, 1) in got, f"offset {offset}: exactly 60.000 s was dropped"


# --------------------------------------------------------------------------
# S2-D3: the NMS sort key was not a total order.
# --------------------------------------------------------------------------


def _cand(start: float, end: float, score: float, text: str) -> CandidateWindow:
    return CandidateWindow(start=start, end=end, total_score=score,
                           scores={"total": score}, text=text)


def test_a_failing_load_asr_still_runs_its_phase_teardown(tmp_path):
    """VRAM-03. Phases 1 and 2 put the load OUTSIDE the try with no `None`
    pre-bind, so a raise from load_asr/load_align — after it had already
    allocated — skipped the ritual entirely. Measured `_last_unload_order`
    was [] for a load_asr failure and ['asr'] for a load_align failure."""
    from tests.unit.test_s1_transcribe import FakeEngine, make_stage
    from clipforge.errors import RetryableStageError
    from clipforge.stages.base import digest_bytes
    from clipforge.state import StateDB

    media = tmp_path / "m.mp4"
    media.write_bytes(b"\x00" * 64)
    db = StateDB(tmp_path / "s.db")

    for failing, expected in (("load_asr", ["asr"]),
                              ("load_align", ["asr", "align"])):
        class Failing(FakeEngine):
            pass

        def boom(*_a, **_kw):
            raise RuntimeError(f"{failing} died after allocating")

        engine = Failing([])
        setattr(engine, failing, boom)
        stage = make_stage(db, tmp_path, engine)
        with pytest.raises(RetryableStageError):
            stage.run(input_digest=digest_bytes(failing.encode()), params={},
                      media_path=media)
        assert stage._last_unload_order == expected, (
            f"{failing} raised and the ritual recorded "
            f"{stage._last_unload_order}, expected {expected} — a load that "
            "allocates then dies must still be torn down")


def test_a_failed_phase_does_not_pin_its_model_via_the_traceback(tmp_path):
    """VRAM-02. `del` drops only the stage frame's reference; the raising
    engine method's frame still holds the model in a parameter, and that
    frame is kept alive by exc.__traceback__, which `raise ... from exc`
    carries out of the stage. Measured: a full model retained, 3/3 runs, on
    every failure path — for as long as any caller holds the exception."""
    from tests.unit.test_s1_transcribe import FakeEngine, make_stage
    from clipforge.errors import RetryableStageError
    from clipforge.stages.base import digest_bytes
    from clipforge.state import StateDB

    media = tmp_path / "m.mp4"
    media.write_bytes(b"\x00" * 64)
    db = StateDB(tmp_path / "s.db")
    seen: dict[str, Any] = {}

    class Weighty:
        pass

    class Engine(FakeEngine):
        def load_asr(self, model_name, compute_type):
            obj = Weighty()
            seen["ref"] = weakref.ref(obj)
            self.refs["asr"] = seen["ref"]
            return obj

        def transcribe(self, asr, media_path, *, batch_size, language):
            # `asr` is bound in THIS frame; the traceback retains it.
            raise RuntimeError("CUDA error: out of memory")

    stage = make_stage(db, tmp_path, Engine([]))
    try:
        stage.run(input_digest=digest_bytes(b"tb"), params={},
                  media_path=media)
    except RetryableStageError as exc:
        # Hold the exception, exactly as a retry loop would.
        held = exc
        gc.collect()
        assert seen["ref"]() is None, (
            "the ASR model is still reachable through the retained "
            "exception's traceback frames; a retry loop would load a second "
            "full suite on top of it")
        assert held.__cause__ is not None, (
            "clearing frames must not destroy the cause chain the operator "
            "needs to diagnose the failure")
    else:
        pytest.fail("the stage should have raised")


def test_a_non_finite_weight_is_rejected():
    """S2-D5. `float(v)` on an arbitrary param defeated digest_params'
    allow_nan guard: a NaN float is rejected there, but the JSON STRING
    'nan' hashes cleanly and became a non-finite weight inside the stage."""
    from clipforge.errors import FatalStageError
    from clipforge.stages.s2_prefilter import _finite_weight

    for bad in ("nan", "inf", "-inf", float("nan"), float("inf"), 1e12):
        with pytest.raises(FatalStageError):
            _finite_weight("qa", bad)
    assert _finite_weight("qa", "1.5") == 1.5
    assert _finite_weight("qa", 2) == 2.0


def test_the_stage_itself_rejects_a_non_finite_weight(tmp_path):
    """The helper alone is not the fix — the STAGE has to call it.

    Testing `_finite_weight` directly left `_execute` free to go back to
    `float(v)`, and the whole gate stayed green. `digest_params` rejects a
    NaN float but the JSON string 'nan' hashes cleanly, so this is the only
    barrier on that path.
    """
    from clipforge.errors import FatalStageError
    from clipforge.stages.base import digest_bytes
    from clipforge.stages.s2_prefilter import S2Prefilter
    from clipforge.state import StateDB

    from tests.unit.test_s2_prefilter import _transcript

    db = StateDB(tmp_path / "s.db")
    stage = S2Prefilter(db, tmp_path)
    transcript = _transcript()

    with pytest.raises(FatalStageError):
        stage.run(input_digest=digest_bytes(b"nanw"),
                  params={"weights": {"qa": "nan"}},
                  transcript=transcript)


def test_artifacts_cannot_serialize_bare_nan_tokens():
    """Bare NaN/Infinity are not valid JSON (RFC 8259). Python round-trips
    them only because its own loader is lenient; a conforming parser in any
    downstream tool rejects the file."""
    from clipforge.schemas import CandidatesArtifact

    art = CandidatesArtifact(
        cache_key="k" * 64, source_transcript="s" * 64,
        candidates=[CandidateWindow(start=0.0, end=30.0,
                                    total_score=float("nan"),
                                    scores={"total": float("nan")},
                                    text="t")])
    with pytest.raises(ValueError):
        art.to_json_bytes()


def test_the_gpu_test_count_is_pinned():
    """GATE-GPU. Two CP2 neutralizations were caught ONLY by gpu-marked
    tests, so on a CPU-only machine those fixes silently become revert-safe
    while the gate still reports PASSED."""
    from tests.integration import conftest as integration_conftest

    # 5 -> 8 on 2026-08-18 (real-weights NF4 checks). Equality, not >=:
    # the ratchet only works if RAISING it is also a deliberate edit, made
    # in the same change that adds the tests it claims to count.
    # 8 -> 5 on 2026-09-09: the generation half left the product and its
    # three NF4 real-weights checks left with it. Lowering is legitimate
    # only when the covered surface shrank in the same change.
    assert integration_conftest.EXPECTED_GPU_TESTS == 5, (
        "the expected real-engine test count changed; lowering it hides "
        "deleted GPU coverage behind an unchanged headline number")


def test_align_handle_owns_both_halves():
    """C2-3, restated honestly. The single handle does NOT reclaim the
    369 MB library cache â€” that claim is retracted. What it does guarantee is
    that model and metadata share one lifetime: dropping the stage's handle
    makes BOTH unreachable, with neither half's lifetime left implicit in
    engine state. That is what this pins."""
    class _Weighty:
        pass

    class _Engine:
        def load_align(self, language):
            return (_Weighty(), {"language": language, "blob": _Weighty()})

    handle = _Engine().load_align("en")
    model_ref = weakref.ref(handle[0])
    meta_ref = weakref.ref(handle[1]["blob"])
    del handle
    gc.collect()
    assert model_ref() is None and meta_ref() is None, (
        "one half outlived the handle â€” its lifetime is held somewhere other "
        "than the caller's reference")


def test_nms_is_invariant_to_input_order_under_an_exact_key_collision():
    a = _cand(0.0, 30.0, 1.0, "AAA")
    b = _cand(0.0, 30.0, 1.0, "BBB")   # ties on score, start AND length
    forward = [c.text for c in nms([a, b], iou_threshold=0.4, top_k=10)]
    reverse = [c.text for c in nms([b, a], iou_threshold=0.4, top_k=10)]
    assert forward == reverse, (
        f"input order decided the winner ({forward} vs {reverse}); `text` is "
        "S3's prompt input, so this changes downstream ranking")
    assert len(forward) == 1, "the pair overlaps completely; one must survive"

