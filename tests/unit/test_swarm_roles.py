"""The swarm's actual roles, pinned.

The Critic carries most of the weight here. This project has twice shipped
video that passed every structural check and contained no picture, so a
role that trusts the previous role's success report is decoration. These
tests hold that it re-derives quality from the FILE, and that a rejection
loops back exactly once rather than forever.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from clipforge.ffmpeg import require_binary
from clipforge.swarm import Supervisor, TaskBoard
from clipforge.swarm.board import Task
from clipforge.swarm.roles import (BLANK_VARIANCE_FLOOR, MAX_QUALITY_RETRIES,
                                   Critic, Packager, Planner, build_swarm,
                                   inspect_video)


def _task(kind: str, payload: dict) -> Task:
    return Task(id=1, kind=kind, payload=payload)


def _video(dest: Path, *, source: str, seconds: float = 1.0) -> Path:
    # lavfi wants `name=opt=v:opt=v`. A bare filter name takes `=` for its
    # first option and `:` thereafter — `testsrc2:size=...` is a parse
    # error, while `color=c=x:size=...` happens to be valid, which is why
    # the blank cases passed and the real ones did not.
    sep = ":" if "=" in source else "="
    spec = f"{source}{sep}size=128x224:rate=10:d={seconds}"
    proc = subprocess.run(
        [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
         "-f", "lavfi", "-i", spec,
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(dest)],
        capture_output=True, timeout=180)
    assert proc.returncode == 0, (
        f"fixture failed for {spec!r}: "
        f"{(proc.stderr or b'')[-400:].decode('utf-8', 'replace')}")
    return dest


@pytest.fixture()
def board(tmp_path):
    b = TaskBoard(tmp_path / "s.sqlite3")
    yield b
    b.close()


# ------------------------------------------------------------- planner

def test_a_brief_becomes_a_generate_task():
    out = Planner().run(_task("plan", {"brief": "a lone road", "shots": 2,
                                       "niche": "dark_mindset"}))
    assert [k for k, _ in out] == ["generate"]
    assert out[0][1]["brief"] == "a lone road"
    assert out[0][1]["niche"] == "dark_mindset"


def test_many_briefs_become_many_tasks():
    out = Planner().run(_task("plan", {"briefs": ["a", "b", "c"]}))
    assert len(out) == 3 and all(k == "generate" for k, _ in out)


def test_a_source_becomes_a_clip_task():
    out = Planner().run(_task("plan", {"source": "D:/v.mp4", "clips": 5}))
    assert out[0][0] == "clip" and out[0][1]["clips"] == 5


def test_an_empty_plan_is_refused_not_silently_dropped():
    """A plan that expands to nothing would look like success and produce
    no work at all."""
    with pytest.raises(ValueError) as err:
        Planner().run(_task("plan", {"niche": "dark_mindset"}))
    assert "brief" in str(err.value)


# -------------------------------------------------------------- critic

def test_a_blank_video_is_rejected(tmp_path):
    """The exact artefact that shipped twice: valid mp4, no picture."""
    blank = _video(tmp_path / "blank.mp4", source="color=c=#5c4a30")
    verdict = inspect_video(blank)
    assert verdict["ok"] is False
    assert "blank" in verdict["reason"]
    assert verdict["variance"] < BLANK_VARIANCE_FLOOR


def test_real_content_passes(tmp_path):
    real = _video(tmp_path / "real.mp4", source="testsrc2")
    verdict = inspect_video(real)
    assert verdict["ok"] is True, verdict["reason"]
    assert verdict["variance"] > BLANK_VARIANCE_FLOOR


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        Critic().run(_task("critique", {"path": str(tmp_path / "nope.mp4")}))


def test_an_unprobeable_file_is_rejected_not_crashed(tmp_path):
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"not a video at all")
    verdict = inspect_video(junk)
    assert verdict["ok"] is False


def test_a_good_piece_goes_to_packaging(tmp_path):
    real = _video(tmp_path / "real.mp4", source="testsrc2")
    out = Critic().run(_task("critique", {"path": str(real), "kind": "piece",
                                          "brief": "x"}))
    assert [k for k, _ in out] == ["package"]
    assert out[0][1]["variance"] > 0


def test_a_rejected_piece_is_regenerated_once_then_given_up_on(tmp_path):
    """A quality rejection must retry — but a loop with no ceiling burns
    the GPU forever on a prompt the model cannot render."""
    blank = _video(tmp_path / "blank.mp4", source="color=c=#5c4a30")

    first = Critic().run(_task("critique", {
        "path": str(blank), "brief": "x", "quality_retries": 0}))
    assert [k for k, _ in first] == ["generate"]
    assert first[0][1]["quality_retries"] == 1
    assert "previous_rejection" in first[0][1]

    with pytest.raises(RuntimeError) as err:
        Critic().run(_task("critique", {
            "path": str(blank), "brief": "x",
            "quality_retries": MAX_QUALITY_RETRIES}))
    assert "not retrying" in str(err.value)


def test_a_rejected_clip_is_not_regenerated(tmp_path):
    """Clips come from a source, not a brief — there is nothing to
    re-render, so it must fail rather than loop."""
    blank = _video(tmp_path / "blank.mp4", source="color=c=#5c4a30")
    with pytest.raises(RuntimeError):
        Critic().run(_task("critique", {"path": str(blank), "kind": "clip",
                                        "quality_retries": 99}))


# ------------------------------------------------------------ packager

def test_packaging_writes_a_draft_beside_the_piece(tmp_path):
    real = _video(tmp_path / "real.mp4", source="testsrc2")
    out = Packager().run(_task("package", {
        "path": str(real), "kind": "piece", "niche": None,
        "brief": "a lone road", "duration_s": 1.0, "width": 128,
        "height": 224, "variance": 900.0}))
    assert out == []
    draft = real.with_suffix(".draft.json")
    assert draft.is_file()
    meta = json.loads(draft.read_text(encoding="utf-8"))
    assert "DRAFT" in meta["status"], (
        "packaged output must be marked draft — the Authorization Law is "
        "not relaxed because a swarm did the work")
    assert meta["file"] == real.name, "the piece ships as rendered"
    assert meta["graded_in_render"] is False


def test_packaging_does_not_re_encode_the_piece(tmp_path):
    """The grade moved into S6's filter chain. Packaging must now be a
    metadata step only: a second encode cost a generation of quality and,
    because it ran after the Critic and therefore after S7, produced a
    shipped file that the QA gate had never measured.
    """
    real = _video(tmp_path / "real.mp4", source="testsrc2")
    before = real.read_bytes()

    out = Packager().run(_task("package", {
        "path": str(real), "kind": "piece", "niche": "dark_mindset",
        "brief": "a lone road", "duration_s": 1.0, "width": 128,
        "height": 224, "variance": 900.0}))
    assert out == []

    assert real.read_bytes() == before, "packaging altered the video"
    assert not real.with_name(real.stem + ".graded" + real.suffix).exists(), (
        "packaging produced a second encode; the grade belongs in S6")

    meta = json.loads(real.with_suffix(".draft.json").read_text(encoding="utf-8"))
    assert meta["graded_in_render"] is True, (
        "the record must say the niche grade was applied during the render")
    assert meta["file"] == real.name


def test_an_unknown_niche_name_does_not_break_packaging(tmp_path):
    """A typo'd or stale niche must still package — the render already
    happened, and losing its metadata helps nobody."""
    real = _video(tmp_path / "real.mp4", source="testsrc2")
    out = Packager().run(_task("package", {
        "path": str(real), "niche": "not_a_real_niche"}))
    assert out == []
    assert real.with_suffix(".draft.json").is_file()


# --------------------------------------------------------------- wiring

def test_every_role_is_registered_without_overlap(board):
    sup = build_swarm(board)
    kinds = [k for a in sup.agents for k in a.kinds]
    assert len(kinds) == len(set(kinds)), "two roles claim the same kind"
    assert set(kinds) == {"plan", "generate", "clip", "critique", "package"}


def test_only_the_pipeline_roles_hold_the_gpu(board):
    sup = build_swarm(board)
    gpu = {a.name for a in sup.agents if a.gpu}
    assert gpu == {"generator", "clipper"}, (
        "planning, critiquing and packaging must not occupy the single "
        "GPU permit — they would serialise the whole swarm behind a "
        "json write")


def test_a_full_graph_runs_end_to_end(board, tmp_path, monkeypatch):
    """Plan -> generate -> critique -> package, with the GPU stage faked
    so this stays hermetic. Proves the ROUTING, which is the part the
    roles own."""
    real = _video(tmp_path / "piece.mp4", source="testsrc2")

    from clipforge.swarm import roles as R

    class FakeGen:
        name, kinds, gpu = "generator", ("generate",), True

        def run(self, task):
            return [("critique", {"path": str(real), "kind": "piece",
                                  "brief": task.payload.get("brief"),
                                  "niche": "dark_mindset",
                                  "quality_retries": 0})]

    sup = Supervisor(board=board, cpu_workers=2)
    sup.register(R.Planner())
    sup.register(FakeGen())
    sup.register(R.Critic())
    sup.register(R.Packager())

    board.submit("plan", {"brief": "a lone road", "niche": "dark_mindset"})
    stats = sup.run_until_drained(timeout_s=60)

    assert stats.failed == 0, "the graph did not complete cleanly"
    # The grade is applied during the render now, so packaging leaves the
    # file alone and the draft lands beside the piece itself.
    draft = real.with_suffix(".draft.json")
    assert draft.is_file(), "the piece never reached packaging"
    assert json.loads(draft.read_text(encoding="utf-8"))["niche"] == "dark_mindset"
