"""The VRAM Law's unload ritual, pinned at the CALL SITES.

`hard_unload(models, *names)` pops entries from the dict the stage owns so
the refcount drops BEFORE gc.collect() and empty_cache() run. That only
works if the dict is the ONLY strong reference. Both GPU stages were
calling it as::

    hard_unload({"model": model})

which builds a throwaway dict, pops from that, and leaves the frame's own
`model` local holding the weights. Measured with a weakref: the object
survives the call and only dies when the local goes. The visible symptom
is the NEXT stage refusing to load with "a previous stage leaked memory" —
a true statement whose cause was two lines away.

`gpu.py` documents this at length and a weakref test pins the function.
Nothing pinned the callers, so both of them drifted.
"""

from __future__ import annotations

import gc
import inspect
import weakref

import pytest

from clipforge.gpu import hard_unload
from clipforge.stages.s3_semantic import S3SemanticRanker
from clipforge.stages.s4_tracking import S4Tracking

GPU_STAGES = [("s3_semantic", S3SemanticRanker), ("s4_tracking", S4Tracking)]


@pytest.mark.parametrize("name,cls", GPU_STAGES)
def test_no_stage_unloads_a_throwaway_dict(name, cls):
    src = inspect.getsource(cls._execute)
    offenders = [ln.strip() for ln in src.splitlines()
                 if "hard_unload({" in ln]
    assert not offenders, (
        f"{name} unloads a dict literal, so its own local still pins the "
        f"weights: {offenders}")


@pytest.mark.parametrize("name,cls", GPU_STAGES)
def test_models_are_reached_through_the_owning_dict(name, cls):
    """Aliasing the dict entry into a local re-creates the bug even when
    the unload call itself looks right."""
    src = inspect.getsource(cls._execute)
    aliases = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("model = models[",
                                         "processor = models["))]
    assert not aliases, (
        f"{name} aliases a model out of the dict into a local: {aliases}")
    assert 'models["model"]' in src, (
        f"{name} does not reach its model through the owning dict")


@pytest.mark.parametrize("name,cls", GPU_STAGES)
def test_the_unload_runs_in_a_finally(name, cls):
    src = inspect.getsource(cls._execute)
    assert "hard_unload(models)" in src, f"{name} never unloads its dict"
    # The LAST unload is the teardown one. Earlier calls are the
    # load-failure cleanups on the paths that return before the try block
    # exists, which are correct but are not the guarantee being pinned.
    idx = src.rindex("hard_unload(models)")
    assert "finally:" in src[:idx], (
        f"{name} unloads outside a finally: an exception mid-inference "
        "would strand the weights on the card")
    tail = src[idx:]
    assert "finally:" not in tail, (
        f"{name}'s last unload is not the teardown one")


def test_the_dict_discipline_actually_frees_and_a_local_does_not():
    """The behavioural claim behind all of the above, measured."""
    class Weights:
        pass

    owned = Weights()
    ref = weakref.ref(owned)
    models = {"model": owned}
    del owned                      # the dict is now the ONLY reference
    hard_unload(models)
    gc.collect()
    assert ref() is None, "popping the owning dict must free the object"

    # And the shape that was shipped: a local alongside the dict.
    leaked = Weights()
    ref2 = weakref.ref(leaked)
    hard_unload({"model": leaked})  # throwaway dict, exactly as before
    gc.collect()
    assert ref2() is not None, (
        "control: a bare local must still pin the object — if this ever "
        "fails, the bug this file guards against is no longer possible "
        "and the guard can go")
