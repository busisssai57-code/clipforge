"""enhance.py — the interesting judgement is what it REFUSES to do.

A silent voiceover that ships, an "upscale" that downscales, a duck that
flattens the whole bed without saying so — each of these fails loud here,
because each has a quiet failure mode that looks like success.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipforge import enhance
from clipforge.errors import ClipForgeError


class _FakeRun:
    """Stands in for enhance._run: records commands, fabricates output."""

    def __init__(self, out_bytes: int = 4096, fail_with: str | None = None):
        self.cmds: list[list[str]] = []
        self.out_bytes = out_bytes
        self.fail_with = fail_with

    def __call__(self, cmd, *, what, timeout_s=900.0):
        self.cmds.append(list(cmd))
        if self.fail_with:
            raise ClipForgeError(f"{what} failed: {self.fail_with}")
        Path(cmd[-1]).write_bytes(b"\x00" * self.out_bytes)


def _filters(monkeypatch, *names: str) -> None:
    have = frozenset(names)
    monkeypatch.setattr(enhance, "has_filter", lambda n: n in have)


# ----------------------------------------------------------------- _run

def test_run_raises_with_the_stderr_tail(monkeypatch):
    import types

    monkeypatch.setattr(
        enhance.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(
            returncode=1, stderr="x" * 600 + "the real reason"))
    with pytest.raises(ClipForgeError) as err:
        enhance._run(["ffmpeg"], what="probe")
    # the message carries the diagnosis, not just "failed"
    assert "probe failed" in str(err.value)
    assert "the real reason" in str(err.value)


def test_run_success_is_silent(monkeypatch):
    import types

    monkeypatch.setattr(
        enhance.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(returncode=0, stderr=""))
    enhance._run(["ffmpeg"], what="probe")  # no raise


# --------------------------------------------------------- kokoro presence

def test_kokoro_available_requires_model_and_a_voice(tmp_path):
    assert not enhance.kokoro_available(tmp_path)
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"\x00")
    assert not enhance.kokoro_available(tmp_path)  # model without a voice
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "af_heart.bin").write_bytes(b"\x00")
    assert enhance.kokoro_available(tmp_path)


def test_synthesize_kokoro_refuses_missing_model(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    with pytest.raises(ClipForgeError, match="Kokoro model missing"):
        enhance.synthesize_kokoro("hi", tmp_path / "o.wav", root=tmp_path)


def test_synthesize_kokoro_names_installed_voices_on_bad_voice(tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    (tmp_path / "onnx").mkdir()
    (tmp_path / "onnx" / "model.onnx").write_bytes(b"\x00")
    (tmp_path / "voices").mkdir()
    (tmp_path / "voices" / "af_heart.bin").write_bytes(b"\x00")
    with pytest.raises(ClipForgeError, match="installed: af_heart"):
        enhance.synthesize_kokoro("hi", tmp_path / "o.wav",
                                  voice="zf_ghost", root=tmp_path)


def _kokoro_world(monkeypatch, tmp_path, audio, *, tokens=(1, 2, 3),
                  style_frames=8, feeds=None):
    """Real weights layout with the inference boundary mocked out: the
    session returns ``audio`` and the tokenizer returns fixed tokens.

    ``tokens`` and ``style_frames`` are adjustable because the token count
    relative to the style pack is load-bearing — it decides both whether
    the text fits in one pass and WHICH style vector is used. With the
    original fixed 3-tokens-into-8-frames world neither could be observed.

    Style frame ``i`` is filled with the value ``i``, so a captured style
    vector names the index it came from. Zeros (the first version) make
    every frame identical and the selection invisible.
    """
    import sys
    import types

    np = pytest.importorskip("numpy")
    ort = pytest.importorskip("onnxruntime")

    root = tmp_path / "kokoro"
    (root / "onnx").mkdir(parents=True)
    (root / "onnx" / "model.onnx").write_bytes(b"\x00")
    (root / "voices").mkdir()
    np.repeat(np.arange(style_frames, dtype=np.float32), 256).reshape(
        style_frames, 1, 256).tofile(root / "voices" / "af_heart.bin")

    class _Session:
        def __init__(self, path, providers=None):
            pass

        def get_inputs(self):
            return [
                types.SimpleNamespace(name="input_ids", type="tensor(int64)"),
                types.SimpleNamespace(name="style", type="tensor(float)"),
                types.SimpleNamespace(name="speed", type="tensor(float)"),
            ]

        def run(self, outputs, feed):
            if feeds is not None:
                feeds.append(feed)
            return [audio]

    monkeypatch.setattr(ort, "InferenceSession", _Session)

    class _Tok:
        def phonemize(self, text, lang="en-us"):
            return text

        def tokenize(self, phonemes):
            return list(tokens)

    tok_mod = types.ModuleType("kokoro_onnx.tokenizer")
    tok_mod.Tokenizer = _Tok
    pkg = types.ModuleType("kokoro_onnx")
    pkg.tokenizer = tok_mod
    monkeypatch.setitem(sys.modules, "kokoro_onnx", pkg)
    monkeypatch.setitem(sys.modules, "kokoro_onnx.tokenizer", tok_mod)
    return root


def test_synthesize_kokoro_renders_audio_to_completion(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    root = _kokoro_world(monkeypatch, tmp_path,
                         np.full(12000, 0.5, dtype=np.float32))  # 0.5s spoken
    dest = tmp_path / "voice.wav"
    out = enhance.synthesize_kokoro("hello world", dest, root=root)

    assert out == dest and dest.is_file()
    assert not list(tmp_path.glob("*.partial"))  # promoted, not left
    import wave

    with wave.open(str(dest), "rb") as w:
        assert w.getframerate() == enhance.KOKORO_SR
        assert w.getnchannels() == 1
        assert w.getnframes() == 12000


def test_synthesize_kokoro_refuses_text_too_long_for_one_pass(monkeypatch,
                                                              tmp_path):
    """Tokens must index INSIDE the style pack.

    The pack has one style vector per supported length, so a text with as
    many tokens as the pack has frames indexes one past the end. Without
    this guard that is an IndexError from numpy deep inside synthesis —
    or, worse, a silent wrap on a differently-shaped export.
    """
    np = pytest.importorskip("numpy")
    root = _kokoro_world(monkeypatch, tmp_path,
                         np.full(12000, 0.5, dtype=np.float32),
                         tokens=tuple(range(1, 9)), style_frames=8)
    dest = tmp_path / "voice.wav"
    with pytest.raises(ClipForgeError, match="too long for one pass"):
        enhance.synthesize_kokoro("a very long line", dest, root=root)
    assert not dest.exists()
    assert not list(tmp_path.glob("**/*.partial"))


def test_synthesize_kokoro_accepts_the_longest_legal_text(monkeypatch,
                                                          tmp_path):
    """The boundary the refusal message states: limit is ``frames - 1``.

    Pinned from the other side so the guard cannot be "fixed" by tightening
    it until it rejects legitimate text.
    """
    np = pytest.importorskip("numpy")
    root = _kokoro_world(monkeypatch, tmp_path,
                         np.full(12000, 0.5, dtype=np.float32),
                         tokens=tuple(range(1, 8)), style_frames=8)
    dest = tmp_path / "voice.wav"
    assert enhance.synthesize_kokoro("right at the limit", dest,
                                     root=root).is_file()


def test_synthesize_kokoro_styles_by_token_length(monkeypatch, tmp_path):
    """The style vector is chosen BY the token count, not fixed.

    Kokoro ships one style frame per length; feeding frame 0 for every
    input is a real quality regression that produces perfectly valid,
    correctly-timed audio — so nothing downstream can catch it. Frame i
    is filled with i, so the fed vector names its own index.
    """
    np = pytest.importorskip("numpy")
    feeds: list[dict] = []
    root = _kokoro_world(monkeypatch, tmp_path,
                         np.full(12000, 0.5, dtype=np.float32),
                         tokens=(1, 2, 3, 4, 5), style_frames=8, feeds=feeds)
    enhance.synthesize_kokoro("hello", tmp_path / "voice.wav", root=root)

    assert len(feeds) == 1
    assert float(np.asarray(feeds[0]["style"]).flat[0]) == 5.0, (
        "expected style_pack[len(tokens)]; frame 0 means the length was "
        "ignored")


def test_synthesize_kokoro_refuses_silence(monkeypatch, tmp_path):
    """The guard the module states: a structurally valid, silent file is
    the failure that ships — it must raise, not write."""
    np = pytest.importorskip("numpy")
    root = _kokoro_world(monkeypatch, tmp_path,
                         np.zeros(12000, dtype=np.float32))
    dest = tmp_path / "voice.wav"
    with pytest.raises(ClipForgeError, match="Kokoro produced silence"):
        enhance.synthesize_kokoro("hello world", dest, root=root)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_synthesize_kokoro_wave_failure_leaves_no_partial(monkeypatch,
                                                          tmp_path):
    np = pytest.importorskip("numpy")
    root = _kokoro_world(monkeypatch, tmp_path,
                         np.full(12000, 0.5, dtype=np.float32))
    import wave

    def broken_open(path, mode):
        Path(path).write_bytes(b"\x00")  # the partial exists, then dies
        raise OSError("disk full")

    monkeypatch.setattr(wave, "open", broken_open)
    dest = tmp_path / "voice.wav"
    with pytest.raises(OSError, match="disk full"):
        enhance.synthesize_kokoro("hello world", dest, root=root)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.partial"))


# -------------------------------------------------------- flite voiceover

def test_voiceover_refuses_without_flite(monkeypatch, tmp_path):
    _filters(monkeypatch)  # no filters at all
    with pytest.raises(ClipForgeError, match="no local TTS"):
        enhance.synthesize_voiceover("hello", tmp_path / "v.wav")


def test_voiceover_refuses_empty_text(monkeypatch, tmp_path):
    _filters(monkeypatch, "flite")
    with pytest.raises(ClipForgeError, match="text is empty"):
        enhance.synthesize_voiceover("   ", tmp_path / "v.wav")


def test_voiceover_refuses_unknown_voice(monkeypatch, tmp_path):
    _filters(monkeypatch, "flite")
    with pytest.raises(ClipForgeError, match="unknown flite voice"):
        enhance.synthesize_voiceover("hello", tmp_path / "v.wav",
                                     voice="hal9000")


def test_voiceover_text_travels_in_a_file_with_escaped_path(monkeypatch,
                                                            tmp_path):
    """Text with ':' or '\\' must never touch the filter graph directly."""
    fake = _FakeRun()
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch, "flite")

    dest = tmp_path / "v.wav"
    text = "meet at 10:30 on C:\\drive"
    out = enhance.synthesize_voiceover(text, dest)

    assert out == dest and dest.is_file()
    (cmd,) = fake.cmds
    spec = next(a for a in cmd if a.startswith("flite="))
    assert "textfile=" in spec
    assert text not in spec  # the prose is not inline in the graph
    # A colon in the PATH would end the filter's option list, so it is
    # escaped. Only a Windows path has one to escape (`C:\\...`), and
    # asserting unconditionally made this fail on POSIX for the one
    # reason that is not a bug — so assert on the escaping rule itself.
    raw_path = str((tmp_path / "x").parent.resolve()).replace("\\", "/")
    if ":" in raw_path:
        assert "\\:" in spec
    assert ":" not in spec.split("textfile='", 1)[1].split("'", 1)[0] \
        .replace("\\:", "")
    # the temp text file is cleaned up whether or not synthesis succeeded
    assert not list(tmp_path.glob("*.txt"))


def test_voiceover_tiny_output_is_a_failure_not_a_file(monkeypatch, tmp_path):
    """<1 KB of wav is silence in a valid container — the failure that
    ships. It must raise and leave nothing behind."""
    fake = _FakeRun(out_bytes=10)
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch, "flite")

    dest = tmp_path / "v.wav"
    with pytest.raises(ClipForgeError, match="produced no audio"):
        enhance.synthesize_voiceover("hello", dest)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.partial"))


def test_voiceover_failed_run_cleans_the_partial(monkeypatch, tmp_path):
    monkeypatch.setattr(enhance, "_run", _FakeRun(fail_with="boom"))
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch, "flite")
    with pytest.raises(ClipForgeError):
        enhance.synthesize_voiceover("hello", tmp_path / "v.wav")
    assert not list(tmp_path.iterdir())  # no partial, no txt, no wav


# -------------------------------------------------------------- mix

def test_mix_requires_both_inputs(monkeypatch, tmp_path):
    _filters(monkeypatch, "sidechaincompress")
    video = tmp_path / "v.mp4"
    with pytest.raises(ClipForgeError, match="no video"):
        enhance.mix_voiceover(video, tmp_path / "vo.wav", tmp_path / "o.mp4")
    video.write_bytes(b"\x00")
    with pytest.raises(ClipForgeError, match="no voiceover"):
        enhance.mix_voiceover(video, tmp_path / "vo.wav", tmp_path / "o.mp4")


def test_mix_prefers_sidechain_ducking(monkeypatch, tmp_path):
    fake = _FakeRun()
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch, "sidechaincompress")
    (tmp_path / "v.mp4").write_bytes(b"\x00")
    (tmp_path / "vo.wav").write_bytes(b"\x00")

    enhance.mix_voiceover(tmp_path / "v.mp4", tmp_path / "vo.wav",
                          tmp_path / "o.mp4")
    (cmd,) = fake.cmds
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" in graph  # bed drops only under speech
    assert cmd[cmd.index("-c:v") + 1] == "copy"


def test_mix_flat_duck_fallback_when_no_sidechain(monkeypatch, tmp_path):
    fake = _FakeRun()
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch)  # build lacks sidechaincompress
    (tmp_path / "v.mp4").write_bytes(b"\x00")
    (tmp_path / "vo.wav").write_bytes(b"\x00")

    enhance.mix_voiceover(tmp_path / "v.mp4", tmp_path / "vo.wav",
                          tmp_path / "o.mp4", duck_db=-9.0)
    (cmd,) = fake.cmds
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "sidechaincompress" not in graph
    assert "volume=-9.0dB" in graph  # worse but honest, and applied


def test_mix_writes_a_partial_and_promotes_it(monkeypatch, tmp_path):
    fake = _FakeRun()
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch, "sidechaincompress")
    (tmp_path / "v.mp4").write_bytes(b"\x00")
    (tmp_path / "vo.wav").write_bytes(b"\x00")

    dest = tmp_path / "o.mp4"
    out = enhance.mix_voiceover(tmp_path / "v.mp4", tmp_path / "vo.wav", dest)
    assert out == dest and dest.is_file()
    (cmd,) = fake.cmds
    assert cmd[-1].endswith(".partial")  # ffmpeg wrote the temp, not dest
    assert not list(tmp_path.glob("*.partial"))


def test_mix_failed_run_leaves_no_partial(monkeypatch, tmp_path):
    def broken_run(cmd, *, what, timeout_s=900.0):
        Path(cmd[-1]).write_bytes(b"\x00" * 64)  # partial written, then dies
        raise ClipForgeError(f"{what} failed: boom")

    monkeypatch.setattr(enhance, "_run", broken_run)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    _filters(monkeypatch, "sidechaincompress")
    (tmp_path / "v.mp4").write_bytes(b"\x00")
    (tmp_path / "vo.wav").write_bytes(b"\x00")

    with pytest.raises(ClipForgeError):
        enhance.mix_voiceover(tmp_path / "v.mp4", tmp_path / "vo.wav",
                              tmp_path / "o.mp4")
    assert not (tmp_path / "o.mp4").exists()
    assert not list(tmp_path.glob("*.partial"))


# ------------------------------------------------------------- upscale

def test_upscale_filter_prefers_libplacebo_with_cas(monkeypatch):
    _filters(monkeypatch, "libplacebo", "cas")
    vf = enhance.upscale_filter(2160, 3840)
    assert "libplacebo=w=2160:h=3840" in vf
    assert "cas=0.35" in vf
    assert vf.endswith("setsar=1")


def test_upscale_filter_lanczos_fallback(monkeypatch):
    _filters(monkeypatch, "unsharp")
    vf = enhance.upscale_filter(2160, 3840)
    assert "scale=2160:3840:flags=lanczos" in vf
    assert "unsharp" in vf


def test_upscale_filter_sharpen_clamped_and_optional(monkeypatch):
    _filters(monkeypatch, "libplacebo", "cas")
    assert "cas=1.0" in enhance.upscale_filter(100, 100, sharpen=7.0)
    assert "cas" not in enhance.upscale_filter(100, 100, sharpen=0.0)


def test_upscale_filter_unsharp_amount_is_clamped_too(monkeypatch):
    _filters(monkeypatch, "unsharp")  # no libplacebo, no cas
    vf = enhance.upscale_filter(100, 100, sharpen=7.0)
    assert "unsharp=5:5:1.50" in vf  # 7.0 * 1.5 capped at 1.5, not 10.5


class _Info:
    def __init__(self, w, h):
        self.width, self.height = w, h


def test_upscale_refuses_to_downscale(monkeypatch, tmp_path):
    import clipforge.ffmpeg as ff

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")
    monkeypatch.setattr(ff, "probe", lambda p: _Info(1920, 1080))
    with pytest.raises(ClipForgeError, match="would degrade it"):
        enhance.upscale(src, tmp_path / "o.mp4", width=1280, height=720)


def test_upscale_refuses_equal_dimensions(monkeypatch, tmp_path):
    """target == source is still not an upscale."""
    import clipforge.ffmpeg as ff

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")
    monkeypatch.setattr(ff, "probe", lambda p: _Info(1080, 1920))
    with pytest.raises(ClipForgeError, match="not larger than the source"):
        enhance.upscale(src, tmp_path / "o.mp4", width=1080, height=1920)


def test_upscale_one_axis_larger_is_a_real_upscale(monkeypatch, tmp_path):
    """Width equal, height larger: the refusal needs BOTH axes to be
    non-larger, so this proceeds."""
    import clipforge.ffmpeg as ff

    fake = _FakeRun()
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ff, "probe", lambda p: _Info(1080, 1920))
    _filters(monkeypatch, "libplacebo", "cas")

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")
    dest = tmp_path / "o.mp4"
    out = enhance.upscale(src, dest, width=1080, height=3840)
    assert out == dest and dest.is_file()
    assert fake.cmds  # it really rendered


def test_upscale_refuses_odd_dimensions(monkeypatch, tmp_path):
    import clipforge.ffmpeg as ff

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")
    monkeypatch.setattr(ff, "probe", lambda p: _Info(1280, 720))
    with pytest.raises(ClipForgeError, match="even dimensions"):
        enhance.upscale(src, tmp_path / "o.mp4", width=1921, height=1080)


def test_upscale_refuses_missing_source(tmp_path):
    with pytest.raises(ClipForgeError, match="no source"):
        enhance.upscale(tmp_path / "ghost.mp4", tmp_path / "o.mp4",
                        width=2160, height=3840)


def test_upscale_runs_and_finalizes_atomically(monkeypatch, tmp_path):
    import clipforge.ffmpeg as ff

    fake = _FakeRun()
    monkeypatch.setattr(enhance, "_run", fake)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ff, "probe", lambda p: _Info(1080, 1920))
    _filters(monkeypatch, "libplacebo", "cas")

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")
    dest = tmp_path / "up.mp4"
    out = enhance.upscale(src, dest, width=2160, height=3840)

    assert out == dest and dest.is_file()
    assert not list(tmp_path.glob("*.partial"))  # temp promoted, not left
    (cmd,) = fake.cmds
    assert cmd[-1].endswith(".partial")  # ffmpeg wrote the temp, not dest
    assert "libx264" in cmd


def test_upscale_failed_run_leaves_no_partial(monkeypatch, tmp_path):
    import clipforge.ffmpeg as ff

    def broken_run(cmd, *, what, timeout_s=900.0):
        Path(cmd[-1]).write_bytes(b"\x00" * 64)  # partial written, then dies
        raise ClipForgeError(f"{what} failed: boom")

    monkeypatch.setattr(enhance, "_run", broken_run)
    monkeypatch.setattr(enhance, "_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(ff, "probe", lambda p: _Info(1080, 1920))
    _filters(monkeypatch, "libplacebo", "cas")

    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00")
    with pytest.raises(ClipForgeError):
        enhance.upscale(src, tmp_path / "up.mp4", width=2160, height=3840)
    assert not (tmp_path / "up.mp4").exists()
    assert not list(tmp_path.glob("*.partial"))
