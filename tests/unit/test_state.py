"""State DB: WAL survival semantics, dedup, sessions, artifact registry."""

import sqlite3
from pathlib import Path

import pytest

from clipforge.errors import StateError
from clipforge.state import SCHEMA_VERSION, StateDB


@pytest.fixture()
def db(tmp_path: Path):
    d = StateDB(tmp_path / "state.db")
    yield d
    d.close()


def test_wal_mode_enabled(db, tmp_path):
    raw = sqlite3.connect(tmp_path / "state.db")
    (mode,) = raw.execute("PRAGMA journal_mode").fetchone()
    raw.close()
    assert mode.lower() == "wal"


def test_schema_version_mismatch_refuses(tmp_path):
    p = tmp_path / "state.db"
    StateDB(p).close()
    raw = sqlite3.connect(p)
    raw.execute(f"PRAGMA user_version={SCHEMA_VERSION + 1}")
    raw.close()
    with pytest.raises(StateError, match="schema"):
        StateDB(p)


def test_video_dedup_is_per_channel(db):
    assert db.mark_video_seen("youtube", "@a", "id1") is True
    assert db.mark_video_seen("youtube", "@a", "id1") is False
    assert db.mark_video_seen("youtube", "@b", "id1") is True  # other channel
    assert db.mark_video_seen("twitch", "@a", "id1") is True   # other platform


def test_pending_videos_are_channel_scoped(db):
    db.mark_video_seen("youtube", "@a", "id1")
    db.mark_video_seen("youtube", "@b", "id2")
    a = [r["video_id"] for r in db.videos_needing_download("youtube", "@a")]
    b = [r["video_id"] for r in db.videos_needing_download("youtube", "@b")]
    assert a == ["id1"] and b == ["id2"]


def test_session_resume_within_window(db):
    sid = db.open_stream_session("twitch", "t")
    db.close_stream_session(sid)
    assert db.resumable_session("twitch", "t", within_s=300) == sid
    assert db.resumable_session("twitch", "other", within_s=300) is None
    # A session that ended before the window opened is NOT resumable
    # (negative window puts the cutoff in the future).
    assert db.resumable_session("twitch", "t", within_s=-60) is None
    db.reopen_session(sid)
    assert [s["id"] for s in db.open_sessions()] == [sid]


def test_timeline_estimated_flag_is_persisted(db):
    """A fabricated timeline must be visible to consumers, not held in a
    process-local variable that nothing ever reads."""
    sid = db.open_stream_session("twitch", "t")
    assert db.get_session(sid)["timeline_estimated"] == 0
    db.mark_timeline_estimated(sid)
    assert db.get_session(sid)["timeline_estimated"] == 1


def test_delete_segment_row_actually_removes_it(db, tmp_path):
    sid = db.open_stream_session("twitch", "t")
    db.record_closed_segment(sid, 0, tmp_path / "c.ts", 0.0, 900.0)
    assert len(db.segments_for_session(sid)) == 1
    db.delete_segment_row(sid, 0)
    assert db.segments_for_session(sid) == []


def test_job_upsert_idempotent(db):
    a = db.upsert_job("vod", "youtube:x", {"n": 1})
    b = db.upsert_job("vod", "youtube:x", {"n": 999})  # payload NOT overwritten
    assert a == b
    row = db.get_job("youtube:x")
    assert row["status"] == "pending"


def test_stage_run_lifecycle(db):
    job = db.upsert_job("clip", "k")
    run = db.stage_started(job, "s1_transcribe", "cachekey")
    db.stage_finished(run, artifact="/x/a.json")
    # failed path
    run2 = db.stage_started(job, "s1_transcribe", "cachekey")
    db.stage_finished(run2, error="RetryableStageError: boom")


def test_artifact_registry_requires_existing_file(db, tmp_path):
    f = tmp_path / "art.json"
    f.write_text("{}", encoding="utf-8")
    db.record_artifact("key", "s2_prefilter", "1", f)
    assert db.lookup_artifact("key", "s2_prefilter") == f
    f.unlink()
    assert db.lookup_artifact("key", "s2_prefilter") is None


def test_artifact_registry_is_stage_scoped(db, tmp_path):
    """DAG Law: a stage must never resolve another stage's artifact, even
    on a (hypothetical) cache-key collision."""
    f = tmp_path / "art.json"
    f.write_text("{}", encoding="utf-8")
    db.record_artifact("key", "s1_transcribe", "1", f)
    assert db.lookup_artifact("key", "s1_transcribe") == f
    assert db.lookup_artifact("key", "s2_prefilter") is None


def test_sqlite_errors_surface_typed(tmp_path):
    """Every StateDB method must raise StateError, never raw sqlite3.*."""
    d = StateDB(tmp_path / "typed.db")
    d.close()
    with pytest.raises(StateError, match="upsert_job"):
        d.upsert_job("vod", "k")  # use-after-close: ProgrammingError → StateError
    with pytest.raises(StateError, match="lookup_artifact"):
        d.lookup_artifact("k", "s1")


def test_stream_session_offset_survives_reconnect(db, tmp_path):
    """The §S0 requirement: absolute media time is DB-tracked across drops."""
    sid = db.open_stream_session("twitch", "somestreamer")
    db.add_segment(sid, 0, tmp_path / "c0.ts", abs_start_s=0.0)
    db.segment_ready(sid, 0, duration_s=900.0)
    # Stream drops after one full segment: bank 900s, next index is 1.
    db.bump_session_offset(sid, add_media_s=900.0, next_segment=1)
    sess = db.get_session(sid)
    assert sess["base_offset_s"] == 900.0
    assert sess["next_segment"] == 1
    # Reconnect writes the next segment at the banked absolute offset.
    db.add_segment(sid, 1, tmp_path / "c1.ts", abs_start_s=900.0)
    ready = list(db.segments_with_status("ready"))
    assert len(ready) == 1 and ready[0]["seg_index"] == 0


def test_segment_status_transitions(db, tmp_path):
    sid = db.open_stream_session("twitch", "s")
    db.add_segment(sid, 0, tmp_path / "c0.ts", 0.0)
    db.segment_ready(sid, 0, 899.5)
    db.set_segment_status(sid, 0, "processed")
    assert list(db.segments_with_status("ready")) == []
