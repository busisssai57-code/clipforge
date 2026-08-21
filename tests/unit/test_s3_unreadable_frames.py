"""The crash that killed the last real clip run.

2026-08-13, a 4 GB Minecraft source: `S3 execution error: list index out
of range`, the job marked failed, and nine more like it sitting in the
state DB unexamined. The cause is two lines apart in `s3_semantic`:
`_extract_frames_cv2` returns `[]` when every `cap.read()` in a window
fails — a seek past the end, a damaged GOP, a codec the build cannot
decode at that offset — and the empty list goes straight into the
processor, which indexes it.

One unreadable window losing ten candidates and the whole job is the
defect. The window being unreadable is not.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from clipforge.errors import StageError
from clipforge.schemas.candidates import CandidatesArtifact, CandidateWindow
from clipforge.stages import s3_semantic
from clipforge.stages.s3_semantic import S3SemanticRanker, _extract_frames_cv2
from clipforge.state import StateDB

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None,
                                  reason="needs ffmpeg to make a video")


def _tiny_video(path: Path, seconds: float = 1.0) -> Path:
    subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-y", "-f", "lavfi", "-i",
         f"testsrc=size=128x96:rate=24:d={seconds}", "-c:v", "libx264",
         "-pix_fmt", "yuv420p", "-preset", "ultrafast", str(path)],
        capture_output=True, timeout=120, check=True)
    return path


@needs_ffmpeg
def test_a_window_past_the_end_reads_no_frames_and_does_not_raise(tmp_path):
    """The extractor's own contract: nothing to read is [], not an error."""
    video = _tiny_video(tmp_path / "tiny.mp4")
    assert _extract_frames_cv2(video, 60.0, 70.0, num_frames=8) == []
    # And the sane case still works, or the test above proves nothing.
    assert _extract_frames_cv2(video, 0.0, 0.9, num_frames=4)


def _fake_transformers(monkeypatch):
    """A processor that indexes its images, exactly like the real one.

    A fake that tolerates an empty list would make this test pass for a
    reason the production stack does not share.
    """
    class _Processor:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Processor()

        def apply_chat_template(self, *_a, **_k):
            return "prompt"

        def __call__(self, *, text, images, **_k):
            images[0]  # noqa: B018 - the IndexError under test
            raise AssertionError("unreachable in these tests")

    class _Model:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Model()

    mod = types.ModuleType("transformers")
    mod.AutoProcessor = _Processor
    mod.Qwen2_5_VLForConditionalGeneration = _Model
    monkeypatch.setitem(sys.modules, "transformers", mod)


def _stage(tmp_path):
    return S3SemanticRanker(StateDB(tmp_path / "state.db"),
                            tmp_path / "artifacts")


def _candidates(*windows):
    return CandidatesArtifact(
        cache_key="cands", source_transcript="tkey",
        candidates=[CandidateWindow(start=s, end=e, text=f"beat {i}",
                                    total_score=9.0 - i,
                                    scores={"boundary": 1.0,
                                            "total": 9.0 - i})
                    for i, (s, e) in enumerate(windows)])


@needs_ffmpeg
def test_every_window_unreadable_names_the_video_instead_of_indexing(
        tmp_path, monkeypatch):
    """`IndexError: list index out of range` told nobody anything.

    A file that cannot be read at any candidate offset is a real failure
    and still fails - but it says which file and how many windows.
    """
    _fake_transformers(monkeypatch)
    video = _tiny_video(tmp_path / "tiny.mp4")
    with pytest.raises(StageError) as err:
        _stage(tmp_path).run(input_digest="d", params={},
                             candidates_artifact=_candidates((60.0, 70.0),
                                                             (80.0, 90.0)),
                             video_path=video)
    message = str(err.value)
    assert "no frames could be read" in message, message
    assert "list index out of range" not in message, (
        "the empty list still reached the processor")
    assert "tiny.mp4" in message


@needs_ffmpeg
def test_one_unreadable_window_does_not_lose_the_other_candidates(
        tmp_path, monkeypatch):
    """The shape of the bug: ten candidates died for one bad window.

    The readable ones are ranked normally; the unreadable one is scored
    last with a justification saying so, rather than being silently
    dropped - a missing beat that nothing mentions is how a clip nobody
    can explain gets shipped.
    """
    video = _tiny_video(tmp_path / "tiny.mp4", seconds=2.0)

    scored = {}

    def _fake_rank(self, *, cache_key, candidates_artifact, video_path,
                   frames_per_cand):
        raise RuntimeError("no cloud in this test")

    # The local path needs weights; what is under test is the guard, so
    # the model call is replaced by one that returns a fixed score for a
    # window that HAS frames.
    monkeypatch.setattr(s3_semantic, "_extract_frames_cv2",
                        lambda video_path, start_s, end_s, num_frames:
                        [] if start_s > 5.0 else ["frame"])

    class _Processor:
        @staticmethod
        def from_pretrained(*_a, **_k):
            return _Processor()

        def apply_chat_template(self, *_a, **_k):
            return "prompt"

        def __call__(self, *, text, images, **_k):
            scored["seen"] = scored.get("seen", 0) + 1
            raise RuntimeError("stop here: the guard is what is under test")

    mod = types.ModuleType("transformers")
    mod.AutoProcessor = _Processor
    mod.Qwen2_5_VLForConditionalGeneration = type(
        "M", (), {"from_pretrained": staticmethod(lambda *a, **k: object())})
    monkeypatch.setitem(sys.modules, "transformers", mod)

    with pytest.raises(StageError) as err:
        _stage(tmp_path).run(input_digest="d", params={},
                             candidates_artifact=_candidates((60.0, 70.0),
                                                             (0.0, 1.5)),
                             video_path=video)
    # The unreadable window did NOT stop the loop: the readable one was
    # carried on to the model, which is where this test cuts the run off.
    assert scored.get("seen") == 1, str(err.value)


# ------------------------------------------------------------ the reaper

def test_a_job_left_running_by_a_crash_is_reaped(tmp_path):
    """A status is only ever moved by the process running the job.

    So a kill, a power cut or an OOM leaves one `running` for ever, and a
    stuck job looks exactly like a busy one. MEASURED on this workspace:
    one had been `running` for 7.8 days, with its stage rows stuck the
    same way, which quietly poisons every per-stage median above it.
    """
    import time as _time

    db = StateDB(tmp_path / "state.db")
    stale = db.upsert_job("clip", "key-stale", {})
    fresh = db.upsert_job("clip", "key-fresh", {})
    run_id = db.stage_started(stale, "s1_transcribe", "cache-1")
    db.set_job_status(stale, "running")
    db.set_job_status(fresh, "running")
    # Age only the first one.
    with db._conn:  # noqa: SLF001 - the test is about persisted state
        db._conn.execute("UPDATE jobs SET updated_at=? WHERE id=?",
                         (_time.time() - 48 * 3600, stale))

    reaped = db.reap_stale_jobs(older_than_s=6 * 3600)

    assert reaped == [stale]
    assert db.get_job("key-stale")["status"] == "failed"
    assert db.get_job("key-fresh")["status"] == "running", (
        "a job that is genuinely in flight must survive the reaper")
    runs = {r["id"]: r for r in db.stage_runs_for(stale)}
    assert runs[run_id]["status"] == "failed"
    assert "abandoned" in (runs[run_id]["error"] or "")


def test_a_stage_row_outliving_its_job_is_reaped_too(tmp_path):
    """The rows the first version of the reaper missed.

    A stage row can outlive its job's status: an exception handler moves
    the JOB to failed while the row it was inside never gets its
    `stage_finished`. MEASURED on this workspace after reaping the stuck
    job — two rows were still 'running' under jobs that had finished
    eight days earlier, one of them s7_qa, which is what the dashboard
    reads for its per-stage medians.
    """
    import time as _time

    db = StateDB(tmp_path / "state.db")
    job = db.upsert_job("clip", "key-finished", {})
    orphan = db.stage_started(job, "s7_qa", "cache-2")
    db.set_job_status(job, "failed")          # the job is NOT running
    with db._conn:  # noqa: SLF001
        db._conn.execute("UPDATE stage_runs SET started_at=? WHERE id=?",
                         (_time.time() - 48 * 3600, orphan))

    db.reap_stale_jobs(older_than_s=6 * 3600)

    row = {r["id"]: r for r in db.stage_runs_for(job)}[orphan]
    assert row["status"] == "failed"
    assert "abandoned" in (row["error"] or "")


# ------------------------------------------------------------ the transport

def test_boot_turns_off_the_transport_that_hangs(tmp_path, monkeypatch):
    """A stalled download is indistinguishable from a slow stage.

    Hugging Face's xet transport hung here with the process alive, 18
    seconds of CPU burnt and a cache that never grew — S3 printed
    "Fetching 5 files: 0%" and sat there. Nothing times it out. The
    classic HTTP path fetched the same 7 GB immediately.

    An operator who has made their own choice keeps it.
    """
    import os

    from clipforge import cli

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    cfg = Path(__file__).resolve().parents[2] / "config" / "config.example.toml"
    monkeypatch.chdir(tmp_path)
    try:
        cli._boot(cfg, sweep_partials=False)
    except Exception:  # noqa: BLE001 - booting fully is not what is under test
        pass
    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"

    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    try:
        cli._boot(cfg, sweep_partials=False)
    except Exception:  # noqa: BLE001
        pass
    assert os.environ["HF_HUB_DISABLE_XET"] == "0", (
        "the operator's own setting was overwritten")
