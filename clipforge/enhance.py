"""Voiceover and upscaling — both entirely local, no downloads.

These are thin, honest wrappers over capabilities this ffmpeg build
already has. They are deliberately small: the interesting judgement is in
what they REFUSE to claim.

* **Voiceover** uses libflite. It is intelligible and robotic. It is not
  a neural voice and this module never suggests otherwise; the upgrade
  path (a Piper/Kokoro ONNX voice on the already-installed onnxruntime)
  is named in the capability note rather than implied by silence.
* **Upscaling** is high-quality RESAMPLING — libplacebo on the GPU when
  available, Lanczos otherwise, plus contrast-adaptive sharpening. It
  cannot invent detail that is not in the source. Calling that
  "super-resolution" would be the same category of overclaim as a blank
  video reported as a successful render.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.capabilities import has_filter
from clipforge.errors import ClipForgeError
from clipforge.log import get_logger

log = get_logger(__name__)

#: flite voices that ship with the library.
FLITE_VOICES = ("slt", "kal", "awb", "rms", "kal16")
DEFAULT_VOICE = "slt"

#: Sharpening strength after upscaling. Conservative: over-sharpening an
#: upscale produces halos that read as lower quality, not higher.
DEFAULT_SHARPEN = 0.35


def _ffmpeg() -> str:
    from clipforge.ffmpeg import require_binary

    return str(require_binary("ffmpeg"))


def _run(cmd: list[str], *, what: str, timeout_s: float = 900.0) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=timeout_s)
    if proc.returncode != 0:
        raise ClipForgeError(
            f"{what} failed: {(proc.stderr or '')[-500:]}")


# ------------------------------------------------------------ voiceover

#: Kokoro-82M (Apache-2.0) lives here once fetched. Neural, natural, and
#: 3.8x realtime on CPU — measured on this machine.
KOKORO_DIR = Path("workspace/models/kokoro")
KOKORO_SR = 24000


def kokoro_available(root: Path | None = None) -> bool:
    base = Path(root) if root else KOKORO_DIR
    return ((base / "onnx" / "model.onnx").is_file()
            and any((base / "voices").glob("*.bin")))


def synthesize_kokoro(text: str, dest: Path, *, voice: str = "af_heart",
                      speed: float = 1.0,
                      root: Path | None = None) -> Path:
    """Neural TTS via Kokoro-82M, run directly against the ONNX graph.

    Deliberately NOT routed through the `kokoro-onnx` convenience wrapper:
    version 0.5.0 feeds one input as int32 where this export declares
    float, and the session rejects it. Its tokeniser is fine, so that part
    is reused and the graph is fed with the dtypes it actually declares —
    read off the session rather than assumed, so a future export that
    changes them still works.
    """
    import numpy as np
    import onnxruntime as ort

    base = Path(root) if root else KOKORO_DIR
    model = base / "onnx" / "model.onnx"
    style_file = base / "voices" / f"{voice}.bin"
    if not model.is_file():
        raise ClipForgeError(f"Kokoro model missing at {model}")
    if not style_file.is_file():
        have = sorted(p.stem for p in (base / "voices").glob("*.bin"))
        raise ClipForgeError(
            f"no Kokoro voice {voice!r}; installed: {', '.join(have) or 'none'}")

    text = (text or "").strip()
    if not text:
        raise ClipForgeError("voiceover text is empty")

    from kokoro_onnx import tokenizer as ktok

    tk = ktok.Tokenizer()
    tokens = tk.tokenize(tk.phonemize(text, lang="en-us"))
    if not tokens:
        raise ClipForgeError("text produced no phonemes")

    style_pack = np.fromfile(style_file, dtype=np.float32).reshape(-1, 1, 256)
    if len(tokens) >= len(style_pack):
        raise ClipForgeError(
            f"text is too long for one pass ({len(tokens)} tokens, limit "
            f"{len(style_pack) - 1}); split it into sentences")

    sess = ort.InferenceSession(str(model),
                                providers=["CPUExecutionProvider"])
    declared = {i.name: i.type for i in sess.get_inputs()}
    ids = np.asarray([[0, *tokens, 0]],
                     dtype=np.int64 if "int64" in declared.get(
                         "input_ids", "int64") else np.int32)
    audio = sess.run(None, {
        "input_ids": ids,
        "style": style_pack[len(tokens)].astype(np.float32),
        "speed": np.asarray([float(speed)], dtype=np.float32),
    })[0]

    arr = np.asarray(audio, dtype=np.float32).flatten()
    # Same guard as everywhere else in this project: a structurally valid,
    # silent file is the failure that ships.
    if arr.size < KOKORO_SR * 0.2 or float(np.abs(arr).max()) < 0.02:
        raise ClipForgeError("Kokoro produced silence")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    import wave

    # Same partial-file protocol as every other writer here: a failed or
    # interrupted write must not leave a stray .partial behind.
    try:
        with wave.open(str(partial), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(KOKORO_SR)
            w.writeframes(
                (np.clip(arr, -1, 1) * 32767).astype("<i2").tobytes())
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    log.info("enhance.voiceover_kokoro", path=str(dest), voice=voice,
             seconds=round(arr.size / KOKORO_SR, 2))
    return dest


def synthesize_voiceover(text: str, dest: Path, *,
                         voice: str = DEFAULT_VOICE,
                         timeout_s: float = 300.0) -> Path:
    """Render ``text`` to a wav with the local flite voice.

    Prefer `synthesize_kokoro` when its weights are present — flite is
    intelligible but plainly synthetic, and Kokoro is neural and natural
    at a fraction of realtime on CPU. This remains as the zero-download
    fallback, since flite is compiled into this ffmpeg.

    Raises rather than returning a silent file: a voiceover that is
    silently empty is worse than one that failed, because it ships.
    """
    if not has_filter("flite"):
        raise ClipForgeError(
            "no local TTS: this ffmpeg was built without libflite. A build "
            "with --enable-libflite, or a Piper/Kokoro ONNX voice, enables "
            "voiceover.")
    text = (text or "").strip()
    if not text:
        raise ClipForgeError("voiceover text is empty")
    if voice not in FLITE_VOICES:
        raise ClipForgeError(
            f"unknown flite voice {voice!r}; available: "
            f"{', '.join(FLITE_VOICES)}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The text goes in a FILE, not on the command line: flite's filter
    # syntax treats ':' and '\' as separators, so any prompt containing a
    # path, a time, or an apostrophe would corrupt the filter graph.
    txt = dest.with_suffix(".txt")
    txt.write_text(text, encoding="utf-8")
    spec = str(txt.resolve()).replace("\\", "/").replace(":", "\\:")
    partial = dest.with_suffix(dest.suffix + ".partial")
    try:
        _run([_ffmpeg(), "-nostdin", "-hide_banner", "-y",
              "-f", "lavfi", "-i", f"flite=textfile='{spec}':voice={voice}",
              "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(partial)],
             what="voiceover synthesis", timeout_s=timeout_s)
        if partial.stat().st_size < 1024:
            raise ClipForgeError("voiceover produced no audio")
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        txt.unlink(missing_ok=True)
    log.info("enhance.voiceover", path=str(dest), voice=voice,
             chars=len(text))
    return dest


#: `normalize=0` is the whole point. amix's default divides by the input
#: count, so mixing a voice into a clip attenuates the ENTIRE result by
#: 6 dB — measured, uniformly, on the first real run: quiet regions came
#: back 6.0 dB under the source long after the voice had stopped. That
#: silently breaks the -14 LUFS contract for every voiced sidecar.
#: `dubbing.build_dub_track` already knew this and says so in its own
#: comment; this function was written without it. Summing instead of
#: averaging can exceed full scale, so the sum is limited, exactly as the
#: dub track does.
_MIX = ("amix=inputs=2:duration=first:dropout_transition=0:normalize=0,"
        "alimiter=limit=0.95")


def mix_voiceover(video: Path, voice: Path, dest: Path, *,
                  duck_db: float = -12.0, voice_gain_db: float = 0.0,
                  timeout_s: float = 900.0) -> Path:
    """Lay a voiceover over a video, ducking the original audio under it.

    Ducking is sidechained off the VOICE, so the bed drops only while the
    voice is actually speaking rather than for the whole clip.
    """
    video, voice, dest = Path(video), Path(voice), Path(dest)
    if not video.is_file():
        raise ClipForgeError(f"no video at {video}")
    if not voice.is_file():
        raise ClipForgeError(f"no voiceover at {voice}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")

    if has_filter("sidechaincompress"):
        # `apad` is load-bearing, not tidiness. sidechaincompress ends as
        # soon as EITHER input EOFs, so an unpadded voice caps the BED at
        # the voice's length: a 59s clip whose audio stops after a 5s hook,
        # with full-length video and a 4.6s audio stream. Measured on the
        # first real run of this function — it had never been called.
        # Padding the sidechain to infinity makes [0:a] the only thing that
        # can end the compressor, and `duration=first` then ends the mix on
        # the bed rather than on the padding.
        chain = (
            f"[1:a]volume={voice_gain_db}dB,apad,asplit=2[vo][sc];"
            f"[0:a][sc]sidechaincompress=threshold=0.05:ratio=8:attack=5:"
            f"release=250[bed];"
            f"[bed][vo]{_MIX}[aout]")
    else:
        # No sidechain in this build: a flat duck is worse but honest, and
        # it is logged so the difference is not silently absorbed.
        log.warning("enhance.no_sidechain",
                    note="ducking flat; sidechaincompress unavailable")
        chain = (f"[0:a]volume={duck_db}dB[bed];"
                 f"[1:a]volume={voice_gain_db}dB[vo];"
                 f"[bed][vo]{_MIX}[aout]")
    try:
        _run([_ffmpeg(), "-nostdin", "-hide_banner", "-y",
              "-i", str(video), "-i", str(voice),
              "-filter_complex", chain, "-map", "0:v", "-map", "[aout]",
              "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
              "-f", "mp4", str(partial)],
             what="voiceover mix", timeout_s=timeout_s)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    log.info("enhance.voiceover_mixed", path=str(dest))
    return dest


# ------------------------------------------------------------- upscale

def upscale_filter(width: int, height: int, *,
                   sharpen: float = DEFAULT_SHARPEN) -> str:
    """Filter chain that resamples to (width, height) and sharpens.

    Prefers libplacebo (GPU, better kernel) and falls back to Lanczos.
    Both are RESAMPLERS — neither reconstructs detail the source lacks.
    """
    if has_filter("libplacebo"):
        chain = f"libplacebo=w={width}:h={height}:upscaler=ewa_lanczos"
    else:
        chain = f"scale={width}:{height}:flags=lanczos"
    if sharpen > 0 and has_filter("cas"):
        chain += f",cas={min(1.0, max(0.0, sharpen))}"
    elif sharpen > 0 and has_filter("unsharp"):
        chain += f",unsharp=5:5:{min(1.5, sharpen * 1.5):.2f}:5:5:0.0"
    return f"{chain},setsar=1"


def upscale(src: Path, dest: Path, *, width: int, height: int,
            sharpen: float = DEFAULT_SHARPEN, crf: int = 18,
            timeout_s: float = 1800.0) -> Path:
    """Resample a video up to (width, height).

    Refuses to "upscale" downward: silently accepting a smaller target
    would make the button a quality-destroying no-op that still reports
    success.
    """
    src, dest = Path(src), Path(dest)
    if not src.is_file():
        raise ClipForgeError(f"no source at {src}")
    from clipforge.ffmpeg import probe

    info = probe(src)
    if width <= int(info.width) and height <= int(info.height):
        raise ClipForgeError(
            f"target {width}x{height} is not larger than the source "
            f"{info.width}x{info.height}; this would degrade it, not "
            "upscale it")
    if width % 2 or height % 2:
        raise ClipForgeError("h264 requires even dimensions")

    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    vf = upscale_filter(width, height, sharpen=sharpen)
    try:
        _run([_ffmpeg(), "-nostdin", "-hide_banner", "-y", "-i", str(src),
              "-vf", vf, "-c:v", "libx264", "-preset", "slow",
              "-crf", str(crf), "-pix_fmt", "yuv420p",
              "-c:a", "copy", "-f", "mp4", str(partial)],
             what="upscale", timeout_s=timeout_s)
        partial.replace(dest)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    log.info("enhance.upscaled", path=str(dest),
             frm=f"{info.width}x{info.height}", to=f"{width}x{height}")
    return dest


# ------------------------------------------------------- clip entry points
#
# Everything above operates on paths. These two resolve a clip by NAME
# inside the workspace and write a sidecar beside it, which is the shape
# `bta` commands and the dashboard both need.
#
# Both were missing, and their absence was invisible in the worst way: the
# capability tiles reported voiceover and upscale LIVE (correctly — this
# ffmpeg and this Kokoro install really can do both) while nothing in the
# CLI, the API or the pipeline ever called them. The dashboard shipped a
# full voiceover panel whose button raised a toast explaining the feature
# instead of running it. A tile that says LIVE for something with no caller
# is the same defect as a tile that lies about ffmpeg, one level up.

#: Voiceover and upscale results are sidecars: a NEW file beside the clip,
#: leaving the original untouched, exactly like a dub. `clipmeta` strips
#: these suffixes so a sidecar inherits its parent's score, transcript and
#: QA rather than appearing as an unscored orphan in the gallery.
VOICEOVER_SUFFIX = ".vo"
UPSCALE_SUFFIX = ".upscaled"

#: 9:16 clips render at 1080x1920. Doubling that is the only target worth
#: offering: 4K vertical is 2160x3840, and anything between is a resample
#: to a size no platform asks for.
UPSCALE_TARGETS = {"1440": (1440, 2560), "2160": (2160, 3840)}


@dataclass
class VoiceoverResult:
    """What a voiceover run produced, including what it declined to do."""

    script_chars: int
    voice: str
    engine: str
    audio_path: Path | None = None
    video_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "script_chars": self.script_chars,
            "voice": self.voice,
            "engine": self.engine,
            "audio_path": str(self.audio_path) if self.audio_path else None,
            "video_path": str(self.video_path) if self.video_path else None,
            "notes": list(self.notes),
        }


def _clip_path(ws, filename: str, *, rejected: bool):
    """Resolve one clip inside the workspace, refusing to escape it."""
    root = (Path(ws.clips) / "rejected") if rejected else Path(ws.clips)
    clip = (root / filename).resolve()
    # The filename reaches here from an HTTP request. web.py confines it too,
    # but a second caller (the CLI, a future one) must not have to remember.
    if not clip.is_relative_to(root.resolve()):
        raise ClipForgeError(f"invalid clip name: {filename!r}")
    if not clip.is_file():
        raise ClipForgeError(f"clip not found: {filename}")
    return clip


def voiceover_clip(ws, filename: str, *, script: str, rejected: bool = False,
                   voice: str | None = None, duck_db: float = -12.0,
                   voice_gain_db: float = 0.0) -> VoiceoverResult:
    """Speak ``script`` over one clip, ducking the clip's own audio under it.

    Kokoro is preferred and flite is the fallback, in that order, because
    the difference is audible and the operator asked for a voiceover, not
    for a specific synthesiser. Which one actually ran is recorded in
    ``engine`` rather than left to be inferred from how it sounds.

    The original clip is never modified — the result is ``<stem>.vo.mp4``.
    """
    script = (script or "").strip()
    if not script:
        raise ClipForgeError("voiceover script is empty")

    clip = _clip_path(ws, filename, rejected=rejected)
    stem = clip.with_suffix("")
    wav = Path(f"{stem}{VOICEOVER_SUFFIX}.wav")
    notes: list[str] = []

    if kokoro_available():
        engine, chosen = "kokoro", (voice or "af_heart")
        synthesize_kokoro(script, wav, voice=chosen)
    elif has_filter("flite"):
        engine, chosen = "flite", (voice or DEFAULT_VOICE)
        notes.append(
            "flite voice: intelligible but plainly synthetic. Fetching "
            "Kokoro-82M (~330 MB) into workspace/models/kokoro upgrades this "
            "to a neural voice.")
        synthesize_voiceover(script, wav, voice=chosen)
    else:
        raise ClipForgeError(
            "no local TTS available: this ffmpeg lacks libflite and Kokoro "
            "weights are not in workspace/models/kokoro")

    result = VoiceoverResult(script_chars=len(script), voice=chosen,
                             engine=engine, audio_path=wav, notes=notes)
    if not has_filter("sidechaincompress"):
        # mix_voiceover already logs this; surfacing it is what makes the
        # difference visible to the person listening to the result.
        result.notes.append(
            "this ffmpeg lacks sidechaincompress, so the original audio is "
            "ducked flat for the whole clip instead of only under the voice")

    result.video_path = mix_voiceover(
        clip, wav, Path(f"{stem}{VOICEOVER_SUFFIX}.mp4"),
        duck_db=duck_db, voice_gain_db=voice_gain_db)
    log.info("enhance.voiceover_clip", clip=filename, engine=engine,
             voice=chosen, chars=len(script))
    return result


def upscale_clip(ws, filename: str, *, height: int = 2560,
                 rejected: bool = False,
                 sharpen: float = DEFAULT_SHARPEN) -> Path:
    """Resample one clip up, preserving its aspect ratio.

    ``height`` names the LONG edge because these are 9:16 verticals; the
    width follows from the source aspect and is rounded to an even number,
    which h264 requires. Deriving it rather than taking both avoids the
    caller silently changing the aspect ratio while asking for a resize.
    """
    clip = _clip_path(ws, filename, rejected=rejected)
    from clipforge.ffmpeg import probe

    info = probe(clip)
    src_w, src_h = int(info.width or 0), int(info.height or 0)
    if not src_w or not src_h:
        raise ClipForgeError(f"could not measure {filename}; refusing to guess")

    width = int(round(src_w * (height / src_h)))
    width += width % 2
    height += height % 2

    dest = Path(f"{clip.with_suffix('')}{UPSCALE_SUFFIX}.mp4")
    # `upscale` refuses a non-larger target itself; letting it do so keeps
    # one place that decides what counts as an upscale.
    return upscale(clip, dest, width=width, height=height, sharpen=sharpen)
