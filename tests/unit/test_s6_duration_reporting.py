"""S6 must not log a duration it did not measure.

The artifact has recorded ``probed_duration`` since the splice work — the
comment above that block explicitly forbids substituting the prediction,
because S7's duration-matches-artifact check exists to catch fabricated
numbers. The LOG line was still printing the pre-splice window length.

Consequence, measured 2026-08-12: a trim that worked (34.7s window, 31.1s
file) logged ``s6.render_complete duration_s=34.688``, so the operator —
and the person debugging the feature — read a working trim as a trim that
had silently done nothing. Twenty minutes went into chasing a bug in the
pacing math that was correct all along.

Structural rather than behavioural on purpose: reproducing it needs a real
render, and the property worth protecting is "this call site reports the
probed value", which is exactly what a revert would change.
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

from clipforge.stages import s6_render


def _render_complete_call() -> ast.Call:
    """The `log.info("s6.render_complete", ...)` node."""
    tree = ast.parse(Path(inspect.getfile(s6_render)).read_text(
        encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value == "s6.render_complete":
            return node
    raise AssertionError("s6.render_complete is no longer logged at all")


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def test_the_logged_duration_is_the_probed_one():
    call = _render_complete_call()
    value = _kwarg(call, "duration_s")
    assert value is not None, "render_complete stopped reporting a duration"
    names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
    assert "probed_duration" in names, (
        "s6.render_complete must log the duration probed back from the "
        f"file; it currently logs {ast.unparse(value)}")


def test_the_logged_duration_is_not_the_pre_splice_window():
    """The exact revert this guards against."""
    call = _render_complete_call()
    value = _kwarg(call, "duration_s")
    names = {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}
    assert "duration" not in names, (
        "`duration` is the window length before splicing — logging it "
        "makes a working jump-cut or trim look like one that did nothing")


def test_the_prediction_is_still_reported_beside_it():
    """Both numbers are useful; the point is that they are LABELLED. A
    divergence between them is what S7's splice-duration check measures,
    so hiding the prediction would make that failure unreadable."""
    call = _render_complete_call()
    assert _kwarg(call, "planned_duration_s") is not None


def test_the_artifact_still_records_the_measured_duration():
    """The half that was already right, pinned so a 'consistency' cleanup
    cannot fix the log by breaking the artifact instead."""
    src = Path(inspect.getfile(s6_render)).read_text(encoding="utf-8")
    assert "duration_s=probed_duration," in src
