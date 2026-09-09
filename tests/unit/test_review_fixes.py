"""The seven review findings, pinned so none can silently return.

Two of these were confirmed by a real swarm run rather than by reading
code, and both are the kind that report success while doing nothing: a
supervisor that declared itself drained mid-render, and an output-path
scrape that reported "produced nothing" after two shots rendered fine.
"""

from __future__ import annotations

import io
import json
import threading
import time
from pathlib import Path

import pytest





# ---------------------------------------------- 1. drain vs in-flight







# ------------------------------------------- 2. manifest, not scraping









def test_the_manifest_writer_is_atomic_and_optional(tmp_path):
    from clipforge.cli import _write_manifest

    _write_manifest(None, kind="generate", outputs=["x"])   # no-op, no crash

    dest = tmp_path / "deep" / "m.json"
    _write_manifest(dest, kind="process", outputs=["a.mp4"], extra={"job_id": 7})
    blob = json.loads(dest.read_text(encoding="utf-8"))
    assert blob["outputs"] == ["a.mp4"] and blob["count"] == 1
    assert blob["kind"] == "process" and blob["job_id"] == 7
    assert not dest.with_suffix(dest.suffix + ".partial").exists()




# --------------------------------------- 3. machine-wide GPU lock

def test_the_gpu_lock_is_reentrant_on_one_thread(tmp_path):
    """gpu_session composes (orchestrator wrapping stage). A file lock is
    not reentrant, so without depth tracking the second acquire would
    deadlock against itself."""
    from clipforge import gpu

    gpu.configure_process_gpu_lock(tmp_path / "gpu.lock")
    try:
        with gpu._process_gpu_lock():
            with gpu._process_gpu_lock():
                pass
    finally:
        gpu.configure_process_gpu_lock(None)


def test_an_unconfigured_gpu_lock_is_a_no_op():
    from clipforge import gpu

    gpu.configure_process_gpu_lock(None)
    with gpu._process_gpu_lock():
        pass


def test_boot_points_the_gpu_lock_at_the_workspace():
    """The invariant only holds across processes if EVERY entry path sets
    it, and they all boot through here."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli._boot)
    assert "configure_process_gpu_lock" in src


# ------------------------------------- 4/7. niche reaches the render

def test_a_niche_supplies_caption_style_and_grade_to_the_stages():
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.process)
    assert "niche_s5_params" in src, (
        "niche caption styling never reaches S5 — a dark_mindset clip "
        "renders in the default 96pt uppercase karaoke")
    assert '"grade"' in src, "the niche grade never reaches S6"


def test_s6_applies_the_grade_before_captions_and_progress():
    """Order matters: grading over the overlays would desaturate white
    captions and the progress bar along with the picture."""
    import inspect

    from clipforge.stages import s6_render

    src = inspect.getsource(s6_render.S6Render._execute)
    assert "grade_part" in src
    i_grade = src.index("{grade_part}")
    assert i_grade < src.index("{progress}") < src.index("ass='{ass_arg}'")




# ------------------------------- 5. auto-selection reaches generation





# --------------------------------------- 6. no reload orphaning work

def test_the_web_server_does_not_auto_reload():
    """A reload empties the in-memory task registry while the spawned
    render keeps holding the GPU — orphaned and invisible."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.web)
    assert "reload=False" in src
    assert "reload=True" not in src
