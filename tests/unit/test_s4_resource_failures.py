"""Resource failures must escape S4, never masquerade as a framing choice.

History: S4's tracking body was once entirely fake and every run reported
success with a centred crop. The blanket `except Exception → centre crop`
was the same failure mode waiting to recur — a CUDA OOM, a dying disk or a
decoder crash would "frame the centre" and the run would say it worked.
`_is_resource_failure` is the classifier that decides which exceptions are
allowed to degrade and which must abort; these tests pin the boundary.
"""

from __future__ import annotations

import pytest

from clipforge.stages.s4_tracking import _is_resource_failure


class OutOfMemoryError(RuntimeError):  # noqa: A001 - torch's class name
    """Stands in for torch.cuda.OutOfMemoryError (matched by NAME, since
    importing torch in a unit test costs seconds and a GPU)."""


@pytest.mark.parametrize("exc", [
    MemoryError("host allocation failed"),
    OSError(28, "No space left on device"),
    OSError(5, "Input/output error"),
    OutOfMemoryError("CUDA out of memory. Tried to allocate 2.50 GiB"),
    RuntimeError("CUDA error: device-side assert triggered"),
    RuntimeError("cuDNN error: CUDNN_STATUS_INTERNAL_ERROR"),
    RuntimeError("CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate"),
    RuntimeError("HIP out of memory"),
])
def test_resource_failures_are_classified_as_such(exc):
    assert _is_resource_failure(exc), (
        f"{type(exc).__name__}: {exc} would have been masked as a centre "
        "crop while the run reported success")


@pytest.mark.parametrize("exc", [
    ValueError("could not parse box"),
    RuntimeError("shape mismatch between tensors"),   # a bug, not the GPU
    KeyError("track_id"),
    ZeroDivisionError(),
])
def test_ordinary_failures_still_degrade_to_the_documented_fallback(exc):
    """The centre crop remains the right answer for a degraded ENVIRONMENT
    — the classifier must not turn every exception into an abort, or a
    machine without ultralytics could never produce a clip at all."""
    assert not _is_resource_failure(exc)


def test_the_stage_wiring_uses_the_classifier():
    """The classifier is only a fix if the catch-all consults it."""
    import inspect

    from clipforge.stages.s4_tracking import S4Tracking

    src = inspect.getsource(S4Tracking._execute)
    idx = src.rindex("except Exception as exc:")
    tail = src[idx:]
    assert "_is_resource_failure(exc)" in tail, (
        "the blanket except no longer classifies resource failures; CUDA "
        "OOM degrades to a silent centre crop again")
    assert "RetryableStageError" in tail
