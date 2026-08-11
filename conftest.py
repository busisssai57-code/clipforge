"""Repo-root conftest: guard the `tests` package against shadowing.

ultralytics 8.4.108 ships a top-level ``tests`` package INTO site-packages
(their own test_cuda.py / test_engine.py). A real package beats this repo's
namespace-package ``tests`` on import, so every cross-module test import
(``from tests.unit.test_s1_transcribe import ...``) broke with
``ModuleNotFoundError: No module named 'tests.unit'`` the moment ultralytics
was installed. The stray copy was deleted from the venv; this guard makes the
failure LOUD and diagnosable if a future upgrade reinstates it, instead of
surfacing as a baffling import error in an unrelated test file.
"""

from __future__ import annotations

from pathlib import Path


def pytest_configure(config) -> None:
    import tests  # noqa: PLC0415 - the point is to see which one wins

    got = Path(next(iter(tests.__path__))).resolve()
    want = Path(__file__).resolve().parent / "tests"
    if got != want:
        raise RuntimeError(
            f"the `tests` package resolves to {got}, not this repo's {want}. "
            "A third-party wheel (ultralytics has done this) installed its "
            "own top-level `tests` into site-packages and is shadowing ours. "
            "Delete it: Remove-Item <venv>/Lib/site-packages/tests -Recurse")
