"""dubbing.py — refusals must be loud, placement must not drift.

Two properties carry this module: the honest split (subtitles for any
language, audio only where a voice exists) and cue PLACEMENT rather than
concatenation. Both are cheap to break silently, so both are pinned here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge import dubbing
from clipforge.dubbing import (CLOUD_OFF_NOTE, LANGUAGES, NO_KEY_NOTE,
                               Cue, cues_from_transcript,
                               installed_voice_languages, language_options,
                               translator_blocker, write_srt)
from clipforge.errors import ClipForgeError
from clipforge.paths import Workspace


# ---------------------------------------------------------------- srt time

@pytest.mark.parametrize("seconds,expected", [
    (0.0, "00:00:00,000"),
    (3661.5, "01:01:01,500"),
    (0.9996, "00:00:01,000"),    # ms carry into seconds, no ",1000"
    (-3.0, "00:00:00,000"),      # clamped, never negative
    (0.0014, "00:00:00,001"),
])
def test_srt_time(seconds, expected):
    assert dubbing._srt_time(seconds) == expected


def test_write_srt_format(tmp_path):
    dest = tmp_path / "out.srt"
    write_srt([Cue(0.5, 2.0, "hello"), Cue(2.5, 4.0, "world")], dest)
    assert dest.read_text(encoding="utf-8") == (
        "1\n00:00:00,500 --> 00:00:02,000\nhello\n\n"
        "2\n00:00:02,500 --> 00:00:04,000\nworld\n")


# ------------------------------------------------------------------- cues

def test_cues_from_transcript_drops_junk():
    out = cues_from_transcript({"segments": [
        {"start": 1.0, "end": 3.0, "text": " keep me "},
        {"start": 4.0, "end": 5.0, "text": "   "},      # blank
        {"start": 6.0, "end": 6.0, "text": "zero"},     # zero-length
        {"start": 8.0, "end": 7.0, "text": "inverted"},
        {"start": -2.0, "end": 1.5, "text": "clamped"},
    ]})
    assert [(c.start, c.end, c.text) for c in out] == [
        (1.0, 3.0, "keep me"), (0.0, 1.5, "clamped")]


def test_cues_from_empty_transcript():
    assert cues_from_transcript({}) == []


# ------------------------------------------------------------------ voices

def _install_voices(root: Path, *names: str) -> Path:
    (root / "voices").mkdir(parents=True, exist_ok=True)
    for n in names:
        (root / "voices" / f"{n}.bin").write_bytes(b"\x00")
    return root


def test_installed_voice_languages_maps_prefix_to_language(tmp_path):
    root = _install_voices(tmp_path, "af_heart", "ef_dora", "zf_xiaobei")
    assert installed_voice_languages(root) == {
        "en": "af_heart", "es": "ef_dora", "zh": "zf_xiaobei"}


def test_first_voice_per_language_wins_deterministically(tmp_path):
    root = _install_voices(tmp_path, "bm_george", "af_heart")
    # both are English; sorted order makes af_heart the stable pick
    assert installed_voice_languages(root) == {"en": "af_heart"}


def test_unknown_prefix_is_not_a_language(tmp_path):
    root = _install_voices(tmp_path, "qf_mystery")
    assert installed_voice_languages(root) == {}


def test_no_voices_dir_is_empty_not_an_error(tmp_path):
    assert installed_voice_languages(tmp_path) == {}


def test_language_options_split_subtitles_from_audio(tmp_path):
    root = _install_voices(tmp_path, "ef_dora")
    opts = {o["code"]: o for o in language_options(root)}
    assert len(opts) == len(LANGUAGES)
    assert opts["es"]["audio"] is True and opts["es"]["blocker"] == ""
    assert opts["es"]["voice"] == "ef_dora"
    # German: offered as subtitles, refused as audio, with the reason
    assert opts["de"]["subtitles"] is True
    assert opts["de"]["audio"] is False
    assert "no installed Kokoro voice" in opts["de"]["blocker"]


# --------------------------------------------------------- blocker prose

def test_translator_blocker_names_the_local_route_as_the_cause(monkeypatch):
    """Since the local translator landed, cloud-off alone blocks nothing.

    Reaching `translator_blocker` now means BOTH routes are out, which
    requires the local model to be uncached AND barred from downloading.
    Naming only the cloud state here would advise a fix that does not
    help — the defect the previous version of this function had, in the
    opposite direction.
    """
    import clipforge.cloud as cloud

    monkeypatch.setattr(cloud, "cloud_enabled", lambda cfg, feat: False)
    msg = translator_blocker(object())
    assert "off by decision" in msg          # cloud state as CONTEXT
    assert "local translator cannot run" in msg   # ...and the real cause
    assert "allow_model_download" in msg          # ...with the actual fix


def test_translator_blocker_on_but_keyless_does_not_claim_a_decision(
        monkeypatch):
    """use_cloud=true + no key must NOT print 'off by decision'."""
    import clipforge.cloud as cloud

    monkeypatch.setattr(cloud, "cloud_enabled", lambda cfg, feat: True)
    msg = translator_blocker(object())
    assert "no usable Gemini key" in msg
    assert "off by decision" not in msg


# ------------------------------------------------------------ dub track

class _FakeRun:
    """Records the ffmpeg command and fabricates its output file."""

    def __init__(self):
        self.cmds: list[list[str]] = []

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"\x00" * 64)


def test_build_dub_track_places_cues_at_their_start(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(dubbing, "run", fake)
    monkeypatch.setattr(dubbing, "require_binary", lambda n: "ffmpeg")

    def synth(text, dest, voice):
        Path(dest).write_bytes(b"\x00")
        return 1.0  # each cue speaks for exactly 1s

    monkeypatch.setattr(dubbing, "_synth_cue", synth)

    report: list[str] = []
    dest = tmp_path / "dub.wav"
    dubbing.build_dub_track(
        [Cue(0.5, 2.0, "a"), Cue(10.25, 12.0, "b")], dest, voice="ef_dora",
        total_s=30.0, work_dir=tmp_path / "work", report=report)

    assert dest.is_file()
    (cmd,) = fake.cmds
    graph = cmd[cmd.index("-filter_complex") + 1]
    # placed, not concatenated: each cue delayed to its own start in ms
    assert "adelay=500|500" in graph
    assert "adelay=10250|10250" in graph
    # summed without per-input attenuation, cut at the bed's end, limited
    assert "amix=inputs=3:normalize=0" in graph
    assert "duration=first" in graph
    assert "alimiter" in graph
    # the silent bed spans the whole clip, not just the last cue's end
    assert "anullsrc" in " ".join(cmd)
    assert cmd[cmd.index("-t") + 1] == "30.000"
    assert report == []  # nothing skipped, nothing overrunning


def test_build_dub_track_reports_overruns_and_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(dubbing, "run", _FakeRun())
    monkeypatch.setattr(dubbing, "require_binary", lambda n: "ffmpeg")

    def synth(text, dest, voice):
        if text == "fails":
            raise RuntimeError("synth exploded")
        Path(dest).write_bytes(b"\x00")
        return 5.0  # spoken 5s into a 1s slot → overrun

    monkeypatch.setattr(dubbing, "_synth_cue", synth)

    report: list[str] = []
    dubbing.build_dub_track(
        [Cue(0.0, 1.0, "long line"), Cue(2.0, 3.0, "fails")],
        tmp_path / "dub.wav", voice="v", total_s=10.0,
        work_dir=tmp_path / "work", report=report)

    # one bad cue is not fatal, but neither problem is hidden
    assert any("could not be synthesised" in r for r in report)
    assert any("overlap the next line" in r for r in report)


def test_build_dub_track_all_cues_failing_is_fatal(tmp_path, monkeypatch):
    monkeypatch.setattr(dubbing, "run", _FakeRun())
    monkeypatch.setattr(dubbing, "require_binary", lambda n: "ffmpeg")
    monkeypatch.setattr(dubbing, "_synth_cue",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("dead")))
    with pytest.raises(ClipForgeError, match="no cue could be synthesised"):
        dubbing.build_dub_track([Cue(0.0, 1.0, "x")], tmp_path / "d.wav",
                                voice="v", total_s=5.0,
                                work_dir=tmp_path / "work")


# ---------------------------------------------------------------- mux_dub

def test_mux_dub_replaces_audio_and_copies_video(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(dubbing, "run", fake)
    monkeypatch.setattr(dubbing, "require_binary", lambda n: "ffmpeg")
    dubbing.mux_dub(tmp_path / "clip.mp4", tmp_path / "dub.wav",
                    tmp_path / "out.mp4")
    (cmd,) = fake.cmds
    assert "-filter_complex" not in cmd  # straight replacement
    assert cmd[cmd.index("-c:v") + 1] == "copy"  # QA geometry survives
    assert "1:a:0" in cmd


def test_mux_dub_can_duck_the_original_bed(tmp_path, monkeypatch):
    fake = _FakeRun()
    monkeypatch.setattr(dubbing, "run", fake)
    monkeypatch.setattr(dubbing, "require_binary", lambda n: "ffmpeg")
    dubbing.mux_dub(tmp_path / "clip.mp4", tmp_path / "dub.wav",
                    tmp_path / "out.mp4", keep_original_at=0.25)
    (cmd,) = fake.cmds
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "volume=0.250" in graph
    assert "amix=inputs=2:normalize=0" in graph


# --------------------------------------------------------------- dub_clip

@pytest.fixture
def ws(tmp_path):
    return Workspace(tmp_path / "ws").ensure()


def test_dub_clip_refuses_unknown_language(ws):
    with pytest.raises(ClipForgeError, match="unknown language 'xx'"):
        dubbing.dub_clip(ws, "clip.mp4", target="xx")


def test_dub_clip_refuses_missing_clip(ws):
    with pytest.raises(ClipForgeError, match="clip not found"):
        dubbing.dub_clip(ws, "ghost.mp4", target="es")


def test_dub_clip_refuses_without_transcript(ws, monkeypatch):
    from clipforge import clipmeta

    (Path(ws.clips) / "clip.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(
        clipmeta, "transcript_for",
        lambda *a, **k: {"available": False, "reason": "no render artifact"})
    with pytest.raises(ClipForgeError, match="no transcript for this clip"):
        dubbing.dub_clip(ws, "clip.mp4", target="es")


def test_dub_clip_no_translator_quotes_the_diagnosed_blocker(ws, monkeypatch):
    """No route at all must surface translator_blocker's prose, which is
    keyed to the ACTUAL config state rather than asserting a decision.

    Both routes are taken away here: `build_translator` returning None is
    now a two-condition state, so cloud-off alone no longer produces it.
    """
    import clipforge.config as config
    import clipforge.translate as translate
    from clipforge import clipmeta

    (Path(ws.clips) / "clip.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(clipmeta, "transcript_for", lambda *a, **k: {
        "available": True, "duration_s": 10.0, "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hi"}]})
    monkeypatch.setattr(config, "load_config", lambda p: object())
    monkeypatch.setattr(translate, "build_translator", lambda cfg: None)
    monkeypatch.setattr(dubbing, "translator_blocker",
                        lambda cfg: "NOTHING CAN TRANSLATE")

    with pytest.raises(ClipForgeError) as err:
        dubbing.dub_clip(ws, "clip.mp4", target="es")
    assert "NOTHING CAN TRANSLATE" in str(err.value)


def test_dub_clip_subtitles_only_when_no_voice_speaks_target(ws, monkeypatch):
    import clipforge.config as config
    import clipforge.vlrank as vlrank
    from clipforge import clipmeta

    (Path(ws.clips) / "clip.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(clipmeta, "transcript_for", lambda *a, **k: {
        "available": True, "duration_s": 10.0, "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"},
                     {"start": 3.0, "end": 5.0, "text": "world"}]})
    monkeypatch.setattr(config, "load_config", lambda p: object())
    monkeypatch.setattr(vlrank, "build_ranker", lambda cfg: object())
    monkeypatch.setattr(
        vlrank, "translate_lines",
        lambda ranker, lines, target_language, source_language=None:
        [f"[de] {t}" for t in lines])
    monkeypatch.setattr(dubbing, "installed_voice_languages",
                        lambda root=None: {})  # no German voice

    result = dubbing.dub_clip(ws, "clip.mp4", target="de")

    srt = Path(ws.clips) / "clip.de.srt"
    assert result.subtitles_path == srt and srt.is_file()
    assert "[de] hello" in srt.read_text(encoding="utf-8")
    # audio REFUSED, not faked through an English phonemiser
    assert result.audio_path is None and result.video_path is None
    assert any("subtitles only" in n for n in result.notes)


def test_dub_clip_voices_the_audio_when_a_voice_speaks_target(ws, monkeypatch):
    """The positive path: track built from the transcript's duration,
    muxed with the requested bed level, result fully populated."""
    import clipforge.config as config
    import clipforge.vlrank as vlrank
    from clipforge import clipmeta

    (Path(ws.clips) / "clip.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(clipmeta, "transcript_for", lambda *a, **k: {
        "available": True, "duration_s": 30.0, "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": "hello"}]})
    monkeypatch.setattr(config, "load_config", lambda p: object())
    monkeypatch.setattr(vlrank, "build_ranker", lambda cfg: object())
    monkeypatch.setattr(
        vlrank, "translate_lines",
        lambda ranker, lines, target_language, source_language=None:
        [f"[es] {t}" for t in lines])
    monkeypatch.setattr(dubbing, "installed_voice_languages",
                        lambda root=None: {"es": "ef_dora"})

    build_calls: list[dict] = []

    def fake_build(cues, dest, *, voice, total_s, work_dir, report=None):
        build_calls.append({"voice": voice, "total_s": total_s})
        Path(dest).write_bytes(b"\x00")
        return Path(dest)

    mux_calls: list[dict] = []

    def fake_mux(clip, dub_audio, dest, *, keep_original_at=0.0):
        mux_calls.append({"keep_original_at": keep_original_at})
        Path(dest).write_bytes(b"\x00")
        return Path(dest)

    monkeypatch.setattr(dubbing, "build_dub_track", fake_build)
    monkeypatch.setattr(dubbing, "mux_dub", fake_mux)

    result = dubbing.dub_clip(ws, "clip.mp4", target="es",
                              keep_original_at=0.3)

    assert result.voice == "ef_dora"
    assert result.audio_path == Path(ws.clips) / "clip.es.wav"
    assert result.video_path == Path(ws.clips) / "clip.es.mp4"
    (bc,) = build_calls
    # the bed spans the transcript's duration, not the last cue's end
    assert bc["total_s"] == 30.0
    assert bc["voice"] == "ef_dora"
    (mc,) = mux_calls
    assert mc["keep_original_at"] == 0.3


def test_dub_clip_same_language_passes_lines_through(ws, monkeypatch):
    import clipforge.config as config
    import clipforge.vlrank as vlrank
    from clipforge import clipmeta

    (Path(ws.clips) / "clip.mp4").write_bytes(b"\x00")
    monkeypatch.setattr(clipmeta, "transcript_for", lambda *a, **k: {
        "available": True, "duration_s": 10.0, "language": "en",
        "segments": [{"start": 0.0, "end": 2.0, "text": "as spoken"}]})
    monkeypatch.setattr(config, "load_config", lambda p: object())
    monkeypatch.setattr(vlrank, "build_ranker", lambda cfg: object())

    def never(*a, **k):
        raise AssertionError("translator must not run for same-language")

    monkeypatch.setattr(vlrank, "translate_lines", never)

    result = dubbing.dub_clip(ws, "clip.mp4", target="en", audio=False)
    assert any("pass through" in n for n in result.notes)
    text = Path(ws.clips).joinpath("clip.en.srt").read_text(encoding="utf-8")
    assert "as spoken" in text
