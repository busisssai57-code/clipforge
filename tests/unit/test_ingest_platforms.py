"""Platform modules: yt-dlp/streamlink parsing, dedup, T4 Kick containment.

Everything offline — the tool runner is injected.
"""

import subprocess
from pathlib import Path

import pytest

from clipforge.errors import IngestError
from clipforge.ingest import kick, twitch, youtube
from clipforge.ingest.backoff import backoff_delay
from clipforge.state import StateDB


def fake_run(stdout: str = "", *, returncode: int = 0, stderr: str = ""):
    def _run(name, args, **kw):
        return subprocess.CompletedProcess([name, *args], returncode,
                                           stdout=stdout, stderr=stderr)
    return _run


def raising_run(exc: Exception):
    def _run(name, args, **kw):
        raise exc
    return _run


# ------------------------------------------------------------------ backoff


def test_backoff_is_deterministic_and_bounded():
    a = [backoff_delay(i, base_s=5, cap_s=300, seed_key="twitch:x") for i in range(8)]
    b = [backoff_delay(i, base_s=5, cap_s=300, seed_key="twitch:x") for i in range(8)]
    assert a == b, "same channel+attempt must give the same delay (Determinism Law)"
    assert all(0 < d <= 300 for d in a)
    assert a[3] > a[0], "delay must grow with attempts"
    # Different channels decorrelate (that is what jitter is for):
    other = [backoff_delay(i, base_s=5, cap_s=300, seed_key="twitch:y") for i in range(8)]
    assert a != other


def test_backoff_never_overflows():
    assert backoff_delay(9999, base_s=5, cap_s=300, seed_key="k") <= 300


# ------------------------------------------------------------------ youtube


def test_channel_videos_url_normalizes_handle():
    assert youtube.channel_videos_url("streamer") == \
        "https://www.youtube.com/@streamer/videos"
    assert youtube.channel_videos_url("@streamer") == \
        "https://www.youtube.com/@streamer/videos"


def test_discover_filters_junk_lines():
    out = "dQw4w9WgXcQ\nWARNING: something\n\nabcdefghijk\ntoo_short\n"
    ids = youtube.discover_vod_ids("@x", 5, run=fake_run(out))
    assert ids == ["dQw4w9WgXcQ", "abcdefghijk"]


def test_downloaded_ids_are_never_offered_again(tmp_path: Path):
    """§S0: never re-download a known id — once it actually landed."""
    db = StateDB(tmp_path / "s.db")
    try:
        run = fake_run("dQw4w9WgXcQ\nabcdefghijk\n")
        first = youtube.pending_vod_ids(db, "@x", 5, run=run)
        assert sorted(first) == ["abcdefghijk", "dQw4w9WgXcQ"]
        # Order is (first_seen, video_id): the exact sequence depends on
        # discovery time, but it must be STABLE across polls (§3.2).
        assert youtube.pending_vod_ids(db, "@x", 5, run=run) == first
        assert youtube.pending_vod_ids(db, "@x", 5, run=run) == first
        for vid in first:
            db.set_video_status("youtube", "@x", vid, "downloaded")
        assert youtube.pending_vod_ids(db, "@x", 5, run=run) == []
    finally:
        db.close()


def test_undownloaded_ids_are_requeued(tmp_path: Path):
    """The data-loss fix: an id discovered but never downloaded (disk guard
    skipped it, or the download failed) MUST be offered again — previously
    it was marked seen once and lost forever."""
    db = StateDB(tmp_path / "s.db")
    try:
        run = fake_run("dQw4w9WgXcQ\nabcdefghijk\n")
        assert len(youtube.pending_vod_ids(db, "@x", 5, run=run)) == 2
        # One failed, one was never attempted: both come back.
        db.set_video_status("youtube", "@x", "dQw4w9WgXcQ", "failed")
        again = youtube.pending_vod_ids(db, "@x", 5, run=run)
        assert sorted(again) == ["abcdefghijk", "dQw4w9WgXcQ"]
    finally:
        db.close()


def test_requeue_gives_up_after_max_attempts(tmp_path: Path):
    """A permanently-broken VOD must not pin the ingest loop forever."""
    db = StateDB(tmp_path / "s.db")
    try:
        run = fake_run("dQw4w9WgXcQ\n")
        assert youtube.pending_vod_ids(db, "@x", 5, max_attempts=2, run=run)
        for _ in range(2):
            db.bump_video_attempt("youtube", "@x", "dQw4w9WgXcQ")
            db.set_video_status("youtube", "@x", "dQw4w9WgXcQ", "failed")
        assert youtube.pending_vod_ids(db, "@x", 5, max_attempts=2, run=run) == []
    finally:
        db.close()


def test_pending_ids_do_not_leak_across_channels(tmp_path: Path):
    """Round-2 finding: a platform-wide pending query made every channel
    return the UNION of all channels' ids, so two loops downloaded the same
    video into different folders simultaneously."""
    db = StateDB(tmp_path / "s.db")
    try:
        alice = youtube.pending_vod_ids(
            db, "@alice", 5, run=fake_run("aaaaaaaaaaa\nbbbbbbbbbbb\n"))
        bob = youtube.pending_vod_ids(
            db, "@bob", 5, run=fake_run("ccccccccccc\n"))
        assert sorted(alice) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
        assert bob == ["ccccccccccc"], "bob must not see alice's pending ids"
        # And alice does not inherit bob's either.
        assert sorted(youtube.pending_vod_ids(
            db, "@alice", 5, run=fake_run(""))) == ["aaaaaaaaaaa", "bbbbbbbbbbb"]
    finally:
        db.close()


def test_download_vod_returns_printed_path(tmp_path: Path):
    media = tmp_path / "yt_dQw4w9WgXcQ.mp4"
    media.write_bytes(b"x")
    got = youtube.download_vod("dQw4w9WgXcQ", tmp_path,
                               run=fake_run(f"junk line\n{media}\n"))
    assert got == media


def test_download_vod_missing_output_raises(tmp_path: Path):
    with pytest.raises(IngestError, match="no output file"):
        youtube.download_vod("dQw4w9WgXcQ", tmp_path,
                             run=fake_run("C:/nope/missing.mp4\n"))


def test_download_vod_rejects_malformed_id(tmp_path: Path):
    with pytest.raises(IngestError, match="malformed"):
        youtube.download_vod("../../etc/passwd", tmp_path, run=fake_run(""))


# ------------------------------------------------------------------- twitch


def test_twitch_live_true():
    out = '{"streams": {"best": {"type": "hls"}}}'
    assert twitch.is_live("someone", run=fake_run(out)) is True


def test_twitch_offline_is_not_an_error():
    out = '{"error": "No playable streams found on this URL"}'
    assert twitch.is_live("someone", run=fake_run(out, returncode=1)) is False


def test_twitch_non_json_raises_typed():
    with pytest.raises(IngestError, match="non-JSON"):
        twitch.is_live("someone", run=fake_run("<html>cloudflare</html>"))


def test_twitch_chunker_args_omit_the_dead_ad_flag():
    """streamlink >=6 filters Twitch ads unconditionally and 8.x warns that
    --twitch-disable-ads 'has been disabled'. Passing it would emit a
    deprecation warning while doing nothing, so it must NOT be sent."""
    args = twitch.chunker_args("someone", "best", disable_ads=True)
    assert "--twitch-disable-ads" not in args
    assert args[-2:] == ["https://www.twitch.tv/someone", "best"]
    assert "--stdout" in args


def test_twitch_chunker_args_carry_resilience_flags():
    """§2: drops and discontinuities are normal — segment-level retries keep
    a transient CDN hiccup from ending the capture."""
    args = twitch.chunker_args("someone", "best")
    for flag in ("--stream-timeout", "--stream-segment-attempts",
                 "--stream-segment-timeout"):
        assert flag in args, flag


def test_chunker_args_have_no_infinite_stream_retry():
    """--retry-streams without --retry-max makes streamlink retry the stream
    lookup FOREVER, so an offline channel never exits and our capture loop
    is pinned. OUR reconnect loop is the retry mechanism."""
    for args in (twitch.chunker_args("x", "best"), kick.chunker_args("x", "best")):
        assert "--retry-streams" not in args, args


def test_chunker_ffmpeg_flags_tolerate_discontinuities():
    """§S0: ad filtering leaves timestamp discontinuities in the copied
    byte stream; ffmpeg must be told to tolerate them."""
    from clipforge.ingest.chunker import HLS_TOLERANCE_FLAGS

    joined = " ".join(HLS_TOLERANCE_FLAGS)
    assert "discardcorrupt" in joined and "genpts" in joined
    assert "-err_detect" in HLS_TOLERANCE_FLAGS


def test_twitch_non_object_json_raises_typed():
    """A Cloudflare interstitial or bare array must surface as IngestError
    (backoff path), not AttributeError, and never as a silent 'offline'."""
    for payload in ("[]", "null", "123", '"str"'):
        with pytest.raises(IngestError):
            twitch.is_live("someone", run=fake_run(payload))


# --------------------------------------------------------------------- kick


def test_kick_live_via_streamlink():
    assert kick.is_live("x", run=fake_run('{"streams": {"best": {}}}')) is True


def test_kick_offline_confident():
    out = '{"error": "No playable streams found on this URL"}'
    assert kick.is_live("x", run=fake_run(out, returncode=1)) is False


def test_kick_cloudflare_block_is_indeterminate_not_fatal():
    """T4: Kick failures must NEVER raise — containment is the contract."""
    out = '{"error": "Unable to open URL: 403 Forbidden"}'
    assert kick.is_live("x", run=fake_run(out, returncode=1)) is None


def test_kick_total_under_any_exception():
    """Even an unexpected exception type must be swallowed (T4)."""
    assert kick.is_live("x", run=raising_run(RuntimeError("plugin exploded"))) is None
    assert kick.is_live("x", run=raising_run(IngestError("tool missing"))) is None


def test_kick_falls_back_to_ytdlp():
    calls = []

    def run(name, args, **kw):
        calls.append(name)
        if name == "streamlink":
            raise IngestError("no plugin for kick.com")
        return subprocess.CompletedProcess([name], 0, stdout="True\n", stderr="")

    assert kick.is_live("x", run=run) is True
    assert calls == ["streamlink", "yt-dlp"]
