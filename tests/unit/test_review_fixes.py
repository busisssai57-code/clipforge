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

from clipforge.swarm import Supervisor, TaskBoard, make_agent


@pytest.fixture()
def board(tmp_path):
    b = TaskBoard(tmp_path / "s.sqlite3")
    yield b
    b.close()


# ---------------------------------------------- 1. drain vs in-flight

def test_drain_waits_for_work_that_is_already_running(board):
    """LIVE FAILURE: the loop exited 1.5s into a 5-minute render.

    `pending_count()` counts claimable work — 'pending', or 'running' with
    an EXPIRED lease. A task being worked right now is 'running' with a
    fresh lease, so it counted as zero and the drain check fired. The pool
    then blocked on __exit__ for work the loop had stopped watching, so
    its retry was never re-claimed and its follow-ups never ran.
    """
    started = threading.Event()
    order: list[str] = []

    def slow(_t):
        started.set()
        time.sleep(1.2)              # longer than 3 idle polls (1.5s @0.5)
        order.append("slow-finished")
        return [("after", {})]

    def after(_t):
        order.append("follow-up-ran")
        return []

    sup = Supervisor(board=board, cpu_workers=2, poll_s=0.1)
    sup.register(make_agent("slow", ["slow"], slow, gpu=True))
    sup.register(make_agent("after", ["after"], after))
    board.submit("slow")
    stats = sup.run_until_drained(timeout_s=60)

    assert started.is_set()
    assert order == ["slow-finished", "follow-up-ran"], (
        f"drain abandoned in-flight work: {order}")
    assert stats.done == 2 and stats.failed == 0


def test_a_retry_after_an_in_flight_failure_is_not_abandoned(board):
    """The exact live shape: the task fails, returns to pending, and must
    be picked up again rather than left behind by an early exit."""
    attempts = {"n": 0}

    def flaky(_t):
        attempts["n"] += 1
        time.sleep(0.4)
        if attempts["n"] == 1:
            raise RuntimeError("first attempt fails")
        return []

    sup = Supervisor(board=board, cpu_workers=2, poll_s=0.1)
    sup.register(make_agent("w", ["w"], flaky, gpu=True))
    board.submit("w")
    sup.run_until_drained(timeout_s=60)
    assert attempts["n"] >= 2, "the retry was never re-claimed"


def test_inflight_is_observable(board):
    sup = Supervisor(board=board)
    assert sup.inflight == 0


# ------------------------------------------- 2. manifest, not scraping

def test_console_wrapping_would_defeat_a_path_scrape(tmp_path):
    """Why the manifest exists, stated as a measurement.

    rich wraps at 80 columns when stdout is not a terminal, splitting a
    long path across lines. Any regex that cannot cross whitespace — which
    is every path regex — then finds nothing, and the caller cannot tell
    that from "the render produced nothing".
    """
    from rich.console import Console

    deep = tmp_path / "workspace" / "generated" / ("a-lone-figure-walks-a-"
                                                   "fog-covered-shoreline")
    deep.mkdir(parents=True)
    real = deep / "sequence.mp4"
    real.write_bytes(b"\x00" * 16)

    buf = io.StringIO()
    Console(file=buf, force_terminal=False).print(f"[bold green]{real}[/] (2 shots)")
    assert len(buf.getvalue().strip().splitlines()) > 1, (
        "control: this path must actually wrap, or the test proves nothing")


def test_the_manifest_survives_what_the_scrape_could_not(tmp_path):
    from clipforge.swarm.roles import _read_manifest

    weird = tmp_path / "My Videos" / "out"
    weird.mkdir(parents=True)
    piece = weird / "sequence.mp4"          # a SPACE in the path
    piece.write_bytes(b"\x00" * 16)

    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"outputs": [str(piece)]}), encoding="utf-8")
    assert _read_manifest(manifest) == [str(piece)]


def test_a_manifest_listing_a_missing_file_reports_only_real_ones(tmp_path):
    from clipforge.swarm.roles import _read_manifest

    real = tmp_path / "there.mp4"
    real.write_bytes(b"\x00" * 8)
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps(
        {"outputs": [str(real), str(tmp_path / "gone.mp4")]}), encoding="utf-8")
    assert _read_manifest(manifest) == [str(real)]


def test_a_missing_or_corrupt_manifest_yields_nothing_rather_than_raising(tmp_path):
    from clipforge.swarm.roles import _read_manifest

    assert _read_manifest(tmp_path / "never-written.json") == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert _read_manifest(bad) == []


def test_the_manifest_writer_is_atomic_and_optional(tmp_path):
    from clipforge.cli import _write_manifest

    _write_manifest(None, kind="generate", outputs=["x"])   # no-op, no crash

    dest = tmp_path / "deep" / "m.json"
    _write_manifest(dest, kind="process", outputs=["a.mp4"], extra={"job_id": 7})
    blob = json.loads(dest.read_text(encoding="utf-8"))
    assert blob["outputs"] == ["a.mp4"] and blob["count"] == 1
    assert blob["kind"] == "process" and blob["job_id"] == 7
    assert not dest.with_suffix(dest.suffix + ".partial").exists()


def test_roles_no_longer_scrape_console_output():
    """Guard the altitude fix itself: reintroducing the regex brings the
    live failure back."""
    import inspect

    from clipforge.swarm import roles

    src = inspect.getsource(roles)
    assert "_all_paths" not in src, "the stdout path scrape is back"
    assert "--manifest" in src, "roles must ask for a machine-readable result"


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


def test_the_packager_no_longer_re_encodes():
    """The grade moved into S6 so QA measures the shipped pixels; a second
    pass here would cost a generation of quality AND bypass the gate."""
    import inspect

    from clipforge.swarm import roles

    src = inspect.getsource(roles.Packager)
    assert "_apply_grade" not in src
    assert "libx264" not in src


# ------------------------------- 5. auto-selection reaches generation

def test_build_router_consults_the_model_registry():
    import inspect

    from clipforge import genvideo

    src = inspect.getsource(genvideo.build_router)
    assert "select_model" in src, (
        "the registry is still bypassed; auto-selection cannot take effect")


def test_selection_scores_against_the_generation_size_not_delivery():
    from clipforge.genvideo.models import _generation_dims_for
    from clipforge.genvideo.providers import _generation_dims

    assert _generation_dims_for("9:16") == _generation_dims("9:16")


# --------------------------------------- 6. no reload orphaning work

def test_the_web_server_does_not_auto_reload():
    """A reload empties the in-memory task registry while the spawned
    render keeps holding the GPU — orphaned and invisible."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.web)
    assert "reload=False" in src
    assert "reload=True" not in src
