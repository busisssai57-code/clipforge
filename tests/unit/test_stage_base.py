"""Stage ABC: cache keys, resume no-op, artifact writing, error typing."""

from pathlib import Path

import pytest

from clipforge.errors import FatalStageError, StageError
from clipforge.schemas import CandidatesArtifact, TranscriptArtifact
from clipforge.stages.base import Stage, digest_bytes, digest_file, digest_params
from clipforge.state import StateDB


class CountingStage(Stage[TranscriptArtifact]):
    name = "s_counting"
    version = "1"
    artifact_type = TranscriptArtifact

    def __init__(self, db, artifacts_dir):
        super().__init__(db, artifacts_dir)
        self.calls = 0

    def _execute(self, *, cache_key, params, **inputs):
        self.calls += 1
        # Test stages reuse TranscriptArtifact but must stamp THEIR name —
        # run() enforces artifact.stage == stage.name (anti-poison check).
        return TranscriptArtifact(cache_key=cache_key, source_path="src",
                                  stage=self.name)


class FailingStage(CountingStage):
    name = "s_failing"

    def _execute(self, *, cache_key, params, **inputs):
        raise FatalStageError("deterministic bug")


class WrongKeyStage(CountingStage):
    name = "s_wrongkey"

    def _execute(self, *, cache_key, params, **inputs):
        return TranscriptArtifact(cache_key="not-the-key", source_path="src")


@pytest.fixture()
def db(tmp_path: Path):
    d = StateDB(tmp_path / "state.db")
    yield d
    d.close()


def test_cache_key_is_sha256_of_inputs(db, tmp_path):
    s = CountingStage(db, tmp_path)
    k = s.cache_key("d" * 64, {"a": 1})
    assert len(k) == 64
    # Sensitive to every component:
    assert k != s.cache_key("e" * 64, {"a": 1})
    assert k != s.cache_key("d" * 64, {"a": 2})
    s.version = "2"
    assert k != s.cache_key("d" * 64, {"a": 1})


def test_digest_params_canonicalizes_order():
    assert digest_params({"a": 1, "b": 2}) == digest_params({"b": 2, "a": 1})
    assert digest_params({"a": 1}) != digest_params({"a": 2})


def test_digest_params_rejects_noncanonical_types():
    """Sets iterate in hash-randomized order and objects stringify to memory
    addresses — either would silently break resumability across restarts.
    The digest must REFUSE them, typed."""
    with pytest.raises(FatalStageError):
        digest_params({"langs": {"en", "de"}})
    with pytest.raises(FatalStageError):
        digest_params({"obj": object()})
    with pytest.raises(FatalStageError):
        digest_params({"nan": float("nan")})


def test_cache_key_differs_across_stages_same_inputs():
    """The cross-stage collision from review round 1: two stages, same
    version, same input, same params must NOT share a cache key."""
    class A(CountingStage):
        name = "s_a"

    class B(CountingStage):
        name = "s_b"

    # cache_key doesn't touch the DB, so None deps are fine here.
    a = A(None, Path("x"))  # type: ignore[arg-type]
    b = B(None, Path("x"))  # type: ignore[arg-type]
    assert a.cache_key("d" * 64, {}) != b.cache_key("d" * 64, {})


def test_digest_file_streams(tmp_path):
    f = tmp_path / "f.bin"
    f.write_bytes(b"hello world")
    assert digest_file(f) == digest_bytes(b"hello world")


def test_run_computes_once_then_cache_hits(db, tmp_path):
    s = CountingStage(db, tmp_path / "artifacts")
    a1 = s.run(input_digest=digest_bytes(b"x"), params={"p": 1})
    a2 = s.run(input_digest=digest_bytes(b"x"), params={"p": 1})
    assert s.calls == 1
    assert a1.cache_key == a2.cache_key
    assert s.artifact_path(a1.cache_key).exists()


def test_run_recomputes_when_artifact_file_deleted(db, tmp_path):
    """Registry row without a file (retention deleted it) => legitimate re-run."""
    s = CountingStage(db, tmp_path / "artifacts")
    a1 = s.run(input_digest=digest_bytes(b"x"), params={})
    s.artifact_path(a1.cache_key).unlink()
    s.run(input_digest=digest_bytes(b"x"), params={})
    assert s.calls == 2


def test_run_distinct_params_distinct_artifacts(db, tmp_path):
    s = CountingStage(db, tmp_path / "artifacts")
    a1 = s.run(input_digest=digest_bytes(b"x"), params={"p": 1})
    a2 = s.run(input_digest=digest_bytes(b"x"), params={"p": 2})
    assert a1.cache_key != a2.cache_key
    assert s.calls == 2


def _stage_run_rows(db, job_id):
    with db._lock:  # test-only peek
        return db._conn.execute(
            "SELECT stage, status, error FROM stage_runs WHERE job_id=? "
            "ORDER BY id", (job_id,)).fetchall()


def test_failure_propagates_typed_and_records(db, tmp_path):
    s = FailingStage(db, tmp_path / "artifacts")
    job = db.upsert_job("clip", "j1")
    with pytest.raises(FatalStageError):
        s.run(input_digest=digest_bytes(b"x"), params={}, job_id=job)
    # No artifact was registered for the failed key.
    assert db.lookup_artifact(s.cache_key(digest_bytes(b"x"), {}), s.name) is None
    # Bookkeeping is total: the run row is 'failed', never stuck 'running'.
    rows = _stage_run_rows(db, job)
    assert rows and rows[-1]["status"] == "failed"
    assert "FatalStageError" in rows[-1]["error"]


class ForeignExceptionStage(CountingStage):
    name = "s_foreign"

    def _execute(self, *, cache_key, params, **inputs):
        raise ValueError("some library blew up")


def test_foreign_exception_is_wrapped_typed_and_recorded(db, tmp_path):
    """Non-StageError from _execute: must surface as StageError (typed, so
    orchestration can act) and must still record the failure."""
    s = ForeignExceptionStage(db, tmp_path / "artifacts")
    job = db.upsert_job("clip", "j2")
    with pytest.raises(StageError, match="ValueError"):
        s.run(input_digest=digest_bytes(b"x"), params={}, job_id=job)
    rows = _stage_run_rows(db, job)
    assert rows and rows[-1]["status"] == "failed"


def test_wrong_cache_key_is_rejected_and_recorded(db, tmp_path):
    s = WrongKeyStage(db, tmp_path / "artifacts")
    job = db.upsert_job("clip", "j3")
    with pytest.raises(FatalStageError, match="stamped"):
        s.run(input_digest=digest_bytes(b"x"), params={}, job_id=job)
    rows = _stage_run_rows(db, job)
    assert rows and rows[-1]["status"] == "failed"


@pytest.mark.parametrize("corrupt_bytes", [
    b"{ definitely not json",       # invalid JSON
    b"null\n",                      # valid JSON, wrong top-level type
    b"[1, 2, 3]\n",                 # valid JSON array
    b'"hello"\n',                   # valid JSON string
    b"\xff\xfe\x01garbage\x80\x81",  # invalid UTF-8 (disk corruption shape)
])
def test_every_corrupt_shape_recovers_not_poisons(db, tmp_path, corrupt_bytes):
    """Round-2 finding: JSONDecodeError alone was under-inclusive. EVERY
    corrupt-bytes class must quarantine + recompute — never escape untyped,
    never permanently brick the resume path."""
    s = CountingStage(db, tmp_path / "artifacts")
    a1 = s.run(input_digest=digest_bytes(b"x"), params={})
    path = s.artifact_path(a1.cache_key)
    path.write_bytes(corrupt_bytes)

    a2 = s.run(input_digest=digest_bytes(b"x"), params={})  # must not raise
    assert s.calls == 2
    assert a2.cache_key == a1.cache_key
    # And a THIRD run cache-hits the freshly rewritten artifact (no loop):
    s.run(input_digest=digest_bytes(b"x"), params={})
    assert s.calls == 2


def test_success_bookkeeping_failure_does_not_mask_success(db, tmp_path,
                                                           monkeypatch):
    """Round-2 finding: a DB hiccup on the success-path stage_finished must
    not convert a committed success into an escaping exception."""
    from clipforge.errors import StateError

    s = CountingStage(db, tmp_path / "artifacts")
    job = db.upsert_job("clip", "j_success_bk")
    real = db.stage_finished

    def flaky(run_id, *, artifact=None, error=None):
        if artifact is not None:  # fail ONLY the success call
            raise StateError("busy_timeout exhausted")
        return real(run_id, artifact=artifact, error=error)

    monkeypatch.setattr(db, "stage_finished", flaky)
    art = s.run(input_digest=digest_bytes(b"x"), params={}, job_id=job)
    assert art.cache_key  # returned normally — no escaping StateError


def test_digest_params_rejects_nonstring_dict_keys():
    """Round-2 finding: json coerces int keys to str, aliasing {1:..} and
    {'1':..} to one cache key. Reject at any nesting depth."""
    with pytest.raises(FatalStageError, match="not str"):
        digest_params({1: "a"})
    with pytest.raises(FatalStageError, match="not str"):
        digest_params({"outer": [{"ok": 1}, {2: "bad"}]})
    digest_params({"1": "a", "nested": [{"x": 1}]})  # str keys fine


def test_corrupt_cached_artifact_recovers_by_recompute(db, tmp_path):
    """A bad cached file must be quarantined + recomputed — never a
    permanent poison pill on the resume path."""
    s = CountingStage(db, tmp_path / "artifacts")
    a1 = s.run(input_digest=digest_bytes(b"x"), params={})
    path = s.artifact_path(a1.cache_key)
    path.write_text("{ definitely not json", encoding="utf-8")

    a2 = s.run(input_digest=digest_bytes(b"x"), params={})
    assert s.calls == 2
    assert a2.cache_key == a1.cache_key
    assert TranscriptArtifact.read(path)  # rewritten valid
    assert path.with_suffix(".json.corrupt").exists()  # evidence kept


def test_cached_artifact_from_wrong_stage_recovers(db, tmp_path):
    """Even if the registry somehow returned a foreign artifact (defense in
    depth), the stage validates artifact.stage and recomputes."""
    s = CountingStage(db, tmp_path / "artifacts")
    key = s.cache_key(digest_bytes(b"x"), {})
    # Forge a registry row pointing at an artifact claiming another stage.
    foreign = CandidatesArtifact(cache_key=key, source_transcript="t")
    dest = s.artifact_path(key)
    foreign.write(dest)
    db.record_artifact(key, s.name, s.version, dest)

    art = s.run(input_digest=digest_bytes(b"x"), params={})
    assert s.calls == 1  # recomputed (forged file quarantined)
    assert art.stage == s.name
