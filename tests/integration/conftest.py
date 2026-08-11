"""Integration tests — everything here requires real hardware or binaries.

Layering (spec §4):
  * ``tests/unit``        — NO GPU, no network, no external binaries. Always green.
  * ``tests/integration`` — gated on the ``gpu`` marker (CUDA) and/or real
    ffmpeg. Populated as stages land (CP2+); the GPU smoke path is part of
    the Definition of Done (§9).

Run only the GPU-gated set:   pytest tests/integration -m gpu
Skip GPU on a CPU machine:    pytest tests/integration -m "not gpu"
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


#: How many gpu-marked tests must exist. The pytest half's headline count is
#: machine-dependent — `pytest -m "not gpu"` reports 331 passed and exits 0 —
#: and two CP2 neutralizations were caught ONLY by gpu-marked tests, so on a
#: CPU-only machine those fixes silently become revert-safe while the gate
#: still says PASSED. Pinning the count makes the shrinkage visible.
EXPECTED_GPU_TESTS = 5


def pytest_collection_modifyitems(config: pytest.Config,
                                  items: list[pytest.Item]) -> None:
    """Auto-skip gpu-marked tests when CUDA is absent, with a clear reason.

    Set ``CLIPFORGE_GATE=full`` to make their absence a FAILURE instead — the
    mode a release gate should run in, where "we did not check" must not be
    reported as "it passed".
    """
    gpu_items = [i for i in items if "gpu" in i.keywords]
    # Enforce the count only on FULL-SUITE runs (directory args, no -k/-m).
    # A developer running one file collects zero gpu tests legitimately;
    # failing that run is a false positive, and a guard that cries wolf on
    # every focused run gets deleted within a week.
    full_suite = all(Path(a.split("::")[0]).is_dir()
                     for a in config.args) if config.args else True
    if (full_suite and len(gpu_items) < EXPECTED_GPU_TESTS
            and not config.option.keyword and not config.option.markexpr):
        raise pytest.UsageError(
            f"expected at least {EXPECTED_GPU_TESTS} gpu-marked tests, "
            f"collected {len(gpu_items)}. Real-engine coverage has been "
            "deleted or unmarked; the gate's count would not have moved.")
    if _cuda_available():
        return
    if os.environ.get("CLIPFORGE_GATE") == "full":
        raise pytest.UsageError(
            "CLIPFORGE_GATE=full but CUDA is unavailable: the real-engine "
            f"half of the gate ({len(gpu_items)} tests) cannot run, so a "
            "PASS here would mean 'not checked', not 'checked and fine'.")
    skip_gpu = pytest.mark.skip(reason="CUDA GPU not available (gpu marker)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
