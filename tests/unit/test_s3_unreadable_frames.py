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

import importlib.util
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

from clipforge.errors import StageError

#: S3 refuses to fall back when its model stack is missing — that refusal
#: is deliberate (ranking silently degrading to heuristics is how a bad
#: clip ships looking scored). It also means these tests, which are about
#: what S3 does with an UNREADABLE WINDOW, never reach that code without
#: the stack installed: the run dies earlier with "transformers not
#: installed" and the assertion reports a frame-reading bug that is not
#: there.
needs_vl_stack = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("transformers") is None,
    reason="S3 ranking needs torch + transformers (the GPU stack, CP2+)")
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
@needs_vl_stack
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
@needs_vl_stack
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
@needs_vl_stack
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

    `_boot` is allowed to RAISE here rather than being wrapped in a bare
    `except`: setdefault runs on its first line, so swallowing everything
    after it would leave this test green on a completely broken boot.
    """
    import os

    from clipforge import cli

    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    cfg = tmp_path / "config.toml"
    base = (Path(__file__).resolve().parents[2] / "config"
            / "config.example.toml").read_text(encoding="utf-8")
    cfg.write_text(base.replace('root = "workspace"',
                                f'root = {str(tmp_path / "ws")!r}'),
                   encoding="utf-8")

    cfg_obj, ws = cli._boot(cfg, sweep_partials=False)

    assert os.environ.get("HF_HUB_DISABLE_XET") == "1"
    # And the boot itself did its job, so the assertion above is about
    # ordering inside a working boot rather than about one env line.
    assert ws.root.is_dir() and cfg_obj.workspace.root


def test_boot_leaves_the_operators_own_transport_choice_alone(tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    from clipforge import cli

    cfg = tmp_path / "config.toml"
    base = (Path(__file__).resolve().parents[2] / "config"
            / "config.example.toml").read_text(encoding="utf-8")
    cfg.write_text(base.replace('root = "workspace"',
                                f'root = {str(tmp_path / "ws")!r}'),
                   encoding="utf-8")

    cli._boot(cfg, sweep_partials=False)

    import os

    assert os.environ["HF_HUB_DISABLE_XET"] == "0"


# ------------------------------------------------------------- the encoder

def test_nvenc_is_reported_on_whether_it_encodes_not_on_being_listed(
        monkeypatch):
    """`doctor` said PASS while every render fell back to libx264.

    The check asked ffmpeg which encoders it knows about. This machine
    lists h264_nvenc and cannot run it — driver 596.49 offers nvenc API
    13.0 where ffmpeg 8.x wants 13.1 — so the fast path was reported
    available while the pipeline quietly used the slow one on every clip.
    """
    from clipforge import preflight

    class _Failed:
        returncode = 1
        stderr = ("[h264_nvenc @ 0000] Driver does not support the required "
                  "nvenc API version. Required: 13.1 Found: 13.0\n"
                  "Conversion failed!\n")

    monkeypatch.setattr(preflight.subprocess if hasattr(preflight, "subprocess")
                        else __import__("subprocess"), "run",
                        lambda *a, **k: _Failed())
    ok, why = preflight._nvenc_encodes_a_frame()
    assert ok is False
    # The last line ffmpeg writes is "Conversion failed!", which names
    # nothing; the reason has to be the line that does.
    assert "13.1" in why and "13.0" in why, why
    assert "Conversion failed" not in why


def test_the_nvenc_probe_frame_is_large_enough_for_nvenc():
    """A 128x128 probe frame is below NVENC's minimum frame dimension.

    It failed with "Frame Dimension less than the minimum supported value"
    on a machine where every real 1080x1920 render used h264_nvenc, and the
    doctor blamed a driver that had already been updated. Measured on
    2026-09-13: 128x128 fails, 256x256 / 640x360 / 1080x1920 all encode.
    """
    from clipforge import preflight

    w, h = (int(x) for x in preflight.NVENC_PROBE_SIZE.split("x"))
    assert w >= 256 and h >= 200, preflight.NVENC_PROBE_SIZE


def test_the_nvenc_reason_is_the_rejection_not_the_stream_mapping(monkeypatch):
    """ffmpeg prints "Stream #0:0 -> #0:0 (... h264_nvenc)" on every run.

    It contains "nvenc", and "nvenc" was checked before "InitializeEncoder",
    so the doctor and the GPU-encoding tile both quoted the stream mapping as
    the reason. This is the real stderr from the failing probe.
    """
    from clipforge import preflight

    class _Failed:
        returncode = 1
        stderr = (
            "Input #0, lavfi, from 'testsrc=size=128x128:rate=1:d=1':\n"
            "  Stream #0:0: Video: wrapped_avframe, rgb24, 128x128\n"
            "Stream mapping:\n"
            "  Stream #0:0 -> #0:0 (wrapped_avframe (native) -> h264 (h264_nvenc))\n"
            "[h264_nvenc @ 0000] InitializeEncoder failed: invalid param (8): "
            "Frame Dimension less than the minimum supported value.\n"
            "[vf#0:0 @ 0000] Task finished with error code: -22 (Invalid argument)\n"
            "Conversion failed!\n")

    monkeypatch.setattr(__import__("subprocess"), "run", lambda *a, **k: _Failed())
    ok, why = preflight._nvenc_encodes_a_frame()
    assert ok is False
    assert "InitializeEncoder" in why, why
    assert "Stream #" not in why, why


def test_driver_advice_is_only_given_when_the_driver_is_the_reason(monkeypatch):
    """Telling the operator to update a current driver sends them the wrong way."""
    from clipforge import preflight

    monkeypatch.setattr(preflight.ff, "list_encoders", lambda: "h264_nvenc")
    monkeypatch.setattr(preflight.ff, "list_filters", lambda: " ass ")
    monkeypatch.setattr(preflight, "_nvenc_encodes_a_frame",
                        lambda: (False, "InitializeEncoder failed: invalid param"))
    nv = next(r for r in preflight.check_ffmpeg_capabilities()
              if r.name == "h264_nvenc")
    assert "Update the NVIDIA driver" not in nv.fix, nv.fix

    monkeypatch.setattr(preflight, "_nvenc_encodes_a_frame",
                        lambda: (False, "Driver does not support the required "
                                        "nvenc API version"))
    nv = next(r for r in preflight.check_ffmpeg_capabilities()
              if r.name == "h264_nvenc")
    assert "Update the NVIDIA driver" in nv.fix, nv.fix


# ------------------------------------------------------------- the weights

def test_a_half_fetched_model_does_not_read_as_present(tmp_path, monkeypatch):
    """2.9 GB of a 7 GB model is a download, not a model.

    The first clip run of the day printed "S3: ranking candidate windows"
    and sat there for an hour: an empty cache, 7 GB fetched from inside
    the stage, no progress anywhere. A check that only asked whether the
    folder existed would have said yes — 16 MB of config files make a
    folder.
    """
    from clipforge import preflight

    cache = tmp_path / "models--Qwen--Qwen2.5-VL-7B-Instruct-AWQ"
    (cache / "blobs").mkdir(parents=True)
    monkeypatch.setattr("clipforge.paths.hf_cache_dir",
                        lambda _model_id: cache)

    # nothing but configs
    (cache / "blobs" / "config").write_bytes(b"x" * 4096)
    assert preflight.check_ranking_weights().ok is False

    # a fetch in flight: big, and unfinished
    (cache / "blobs" / "shard.incomplete").write_bytes(b"x" * int(2e9))
    result = preflight.check_ranking_weights()
    assert result.ok is False
    assert "unfinished" in result.message

    # finished
    (cache / "blobs" / "shard.incomplete").rename(cache / "blobs" / "shard")
    assert preflight.check_ranking_weights().ok is True


def test_a_slow_job_is_not_reaped_while_its_stages_report_in(tmp_path):
    """The reaper must tell "stuck" from "slow", and could not.

    `set_job_status` runs exactly twice in a job's life, so `updated_at`
    meant "when this started", and six hours of honest work — a two-hour
    source through S1, or an LTX-2.5 brief at ~25 minutes a shot — read
    as silence. Every stage boundary is now a heartbeat.
    """
    import time as _time

    db = StateDB(tmp_path / "state.db")
    job = db.upsert_job("clip", "key-slow", {})
    db.set_job_status(job, "running")
    with db._conn:  # noqa: SLF001 - simulate eight hours of elapsed work
        db._conn.execute("UPDATE jobs SET updated_at=? WHERE id=?",
                         (_time.time() - 8 * 3600, job))

    # ... and then a stage finishes, exactly as a live job does.
    run = db.stage_started(job, "s1_transcribe", "cache-slow")

    assert db.reap_stale_jobs(older_than_s=6 * 3600) == []
    assert db.get_job("key-slow")["status"] == "running"
    assert {r["id"]: r for r in db.stage_runs_for(job)}[run]["status"] == \
        "running", "the row of a live job was marked failed under it"

    db.stage_finished(run, artifact="a.json")
    assert db.reap_stale_jobs(older_than_s=6 * 3600) == [], (
        "finishing a stage must also refresh the job")


def test_an_open_row_under_a_live_job_survives_the_orphan_sweep(tmp_path):
    """The orphan sweep swept by AGE alone, which catches slow work.

    A stage row older than the cutoff is only abandoned if the job above
    it has stopped; under a job still marked running it is a long stage.
    """
    import time as _time

    db = StateDB(tmp_path / "state.db")
    live = db.upsert_job("clip", "key-live", {})
    dead = db.upsert_job("clip", "key-dead", {})
    live_run = db.stage_started(live, "s3_semantic", "c-live")
    dead_run = db.stage_started(dead, "s3_semantic", "c-dead")
    db.set_job_status(live, "running")
    db.set_job_status(dead, "failed")
    old = _time.time() - 48 * 3600
    with db._conn:  # noqa: SLF001
        db._conn.execute("UPDATE stage_runs SET started_at=?", (old,))
        db._conn.execute("UPDATE jobs SET updated_at=? WHERE id=?", (old, dead))

    db.reap_stale_jobs(older_than_s=6 * 3600)

    assert {r["id"]: r for r in db.stage_runs_for(dead)}[dead_run]["status"] \
        == "failed"
    assert {r["id"]: r for r in db.stage_runs_for(live)}[live_run]["status"] \
        == "running", "a long stage under a live job was reaped"
