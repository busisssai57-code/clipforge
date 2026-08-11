"""Dubbing: translated subtitles for any language, dubbed audio where a
voice exists.

The capability probe has reported dubbing as unavailable because the
translation model it looked for (NLLB/M2M100, ~2.5 GB) was never
installed. A translator is now wired in for ranking, so the missing half
is present — and the honest split is worth stating plainly, because the
two halves have different requirements:

* **Translated subtitles** need a translator and nothing else. They work
  for every language the model handles, and they are what most short-form
  "dubbing" is actually consumed as.
* **Dubbed audio** additionally needs a *voice in the target language*.
  Kokoro ships one voice per language file, and this machine has
  ``af_heart`` — American English. So dubbed audio is offered for English
  targets and refused for the rest, rather than run through an English
  phonemiser and shipped as a Spanish dub.

Refusing loudly is the point. A dub voiced by the wrong phonemiser is not
a worse dub; it is a different, wrong artifact that sounds like a bug in
the pipeline rather than a missing model.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipforge.errors import ClipForgeError
from clipforge.ffmpeg import require_binary, run
from clipforge.log import get_logger
from clipforge.paths import Workspace

log = get_logger(__name__)

#: Why no translator exists — the single source both surfaces quote.
#: ``bta dub``'s refusal and the capability probe's blocker used to carry
#: independently hand-written copies of this prose; when the local
#: translator lands, one copy would have been updated and the other would
#: have kept announcing it as "not built yet".
#: Retired as a BLOCKER on 2026-08-05 when the local route landed. Cloud
#: being off no longer stops translation, so this is a note about which
#: engine you get, not a reason the feature is unavailable. Its old text
#: ended "the local route ... is not built yet", which became false the
#: moment `clipforge.translate` existed — a stale claim in prose reads as
#: evidence that the thing it describes is still true.
CLOUD_OFF_NOTE = (
    "cloud inference is off by decision ([s3] use_cloud = false; spec "
    "section 2 says 'Cloud: None'), so translation runs on the local "
    "NLLB-200 model. Quality is below Gemini's on idiomatic speech.")

NO_KEY_NOTE = (
    "cloud translation is enabled but no usable Gemini key was found, so "
    "translation falls back to the local NLLB-200 model. Set "
    "CLIPFORGE_GEMINI_API_KEY in .env to use the cloud translator (this "
    "sends transcript text to Google).")


def translator_blocker(cfg: Any) -> str:
    """Diagnose why ``build_translator`` returned None, from real state.

    Reaching this means NEITHER route can run, which since the local
    translator landed requires the local model to be both uncached and
    barred from downloading. The cloud state is reported as context rather
    than as the cause, because it is no longer sufficient on its own — the
    earlier version of this function asserted "off by decision" for every
    None and so advised flipping a switch that would not have helped.
    """
    from clipforge.cloud import cloud_enabled  # noqa: PLC0415
    from clipforge.translate import _local_model_id  # noqa: PLC0415

    cloud = ("cloud translation is on but no usable Gemini key was found"
             if cloud_enabled(cfg, "translation")
             else "cloud inference is off by decision ([s3] use_cloud "
                  "= false)")
    return (
        f"{cloud}, and the local translator cannot run: "
        f"{_local_model_id(cfg)} is not cached and [dubbing] "
        "allow_model_download is false. Allow that one-time ~2.4 GB "
        "download, pre-fetch the model, or set a Gemini key.")


#: Kokoro voice files are per-language. The prefix encodes the language
#: and gender: ``af_`` = American English female, ``bm_`` = British male.
#: Only languages with an installed voice can be dubbed as AUDIO.
_VOICE_LANG_PREFIX = {
    "a": "en", "b": "en", "e": "es", "f": "fr", "h": "hi",
    "i": "it", "j": "ja", "p": "pt", "z": "zh",
}

#: Offered as subtitle targets. Not a capability claim — the translator
#: decides what it can do; this is the menu the UI shows.
LANGUAGES: tuple[tuple[str, str], ...] = (
    ("en", "English"), ("es", "Spanish"), ("pt", "Portuguese"),
    ("fr", "French"), ("de", "German"), ("it", "Italian"),
    ("nl", "Dutch"), ("pl", "Polish"), ("tr", "Turkish"),
    ("ru", "Russian"), ("ar", "Arabic"), ("hi", "Hindi"),
    ("ja", "Japanese"), ("ko", "Korean"), ("zh", "Chinese"),
    ("id", "Indonesian"), ("vi", "Vietnamese"),
)
_LANG_NAME = dict(LANGUAGES)


def installed_voice_languages(root: Path | None = None) -> dict[str, str]:
    """Map language code → an installed Kokoro voice that speaks it."""
    from clipforge.enhance import KOKORO_DIR

    base = Path(root) if root else KOKORO_DIR
    out: dict[str, str] = {}
    voices_dir = base / "voices"
    if not voices_dir.is_dir():
        return out
    for path in sorted(voices_dir.glob("*.bin")):
        lang = _VOICE_LANG_PREFIX.get(path.stem[:1])
        if lang and lang not in out:
            out[lang] = path.stem
    return out


def language_options(root: Path | None = None) -> list[dict[str, Any]]:
    """Every offered language, with whether audio dubbing is possible."""
    voices = installed_voice_languages(root)
    return [{
        "code": code,
        "label": label,
        "subtitles": True,
        "audio": code in voices,
        "voice": voices.get(code),
        "blocker": ("" if code in voices else
                    f"no installed Kokoro voice speaks {label}; subtitles "
                    f"only"),
    } for code, label in LANGUAGES]


# --------------------------------------------------------------- subtitles

def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


@dataclass
class Cue:
    start: float
    end: float
    text: str


def cues_from_transcript(transcript: dict[str, Any]) -> list[Cue]:
    """Timed cues from a clipmeta transcript, clamped to the clip."""
    out: list[Cue] = []
    for seg in transcript.get("segments") or []:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = max(0.0, float(seg.get("start") or 0.0))
        end = float(seg.get("end") or start)
        if end <= start:
            continue
        out.append(Cue(start, end, text))
    return out


def write_srt(cues: list[Cue], dest: Path) -> Path:
    body = "\n".join(
        f"{i}\n{_srt_time(c.start)} --> {_srt_time(c.end)}\n{c.text}\n"
        for i, c in enumerate(cues, 1))
    dest.write_text(body, encoding="utf-8")
    log.info("dubbing.srt_written", path=str(dest), cues=len(cues))
    return dest


# ------------------------------------------------------------------- audio

@dataclass
class DubResult:
    language: str
    subtitles_path: Path | None = None
    audio_path: Path | None = None
    video_path: Path | None = None
    voice: str | None = None
    cues: int = 0
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "language": self.language,
            "language_label": _LANG_NAME.get(self.language, self.language),
            "subtitles": str(self.subtitles_path) if self.subtitles_path else None,
            "audio": str(self.audio_path) if self.audio_path else None,
            "video": str(self.video_path) if self.video_path else None,
            "voice": self.voice,
            "cues": self.cues,
            "notes": self.notes,
        }


def _synth_cue(text: str, dest: Path, voice: str) -> float:
    """Speak one cue. Returns its rendered duration in seconds."""
    from clipforge.enhance import synthesize_kokoro
    from clipforge.ffmpeg import probe

    synthesize_kokoro(text, dest, voice=voice)
    return float(probe(dest).duration_s or 0.0)


def build_dub_track(cues: list[Cue], dest: Path, *, voice: str,
                    total_s: float, work_dir: Path,
                    report: list[str] | None = None) -> Path:
    """Lay each spoken cue at its own start time on a silent bed.

    Cues are placed, not concatenated. Concatenation would drift: a
    translated line is rarely the same length as the original, and after
    thirty cues the voice is talking about something that happened ten
    seconds ago. Placing each at its recorded start keeps the dub locked
    to the picture, and a cue that runs long simply overlaps the next —
    reported, not hidden.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_binary("ffmpeg")

    pieces: list[tuple[float, Path]] = []
    overruns = 0
    for i, cue in enumerate(cues):
        piece = work_dir / f"cue_{i:04d}.wav"
        try:
            spoken = _synth_cue(cue.text, piece, voice)
        except Exception as exc:  # noqa: BLE001 - one bad cue is not fatal
            log.warning("dubbing.cue_failed", index=i, error=str(exc)[:200])
            continue
        if spoken > (cue.end - cue.start) + 0.35:
            overruns += 1
        pieces.append((cue.start, piece))

    if not pieces:
        raise ClipForgeError("no cue could be synthesised")

    # One filter graph: a silent bed of the clip's length, with every cue
    # delayed to its start and mixed in. `amix` would attenuate by input
    # count, so the sum is taken and limited instead.
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
           "-f", "lavfi", "-t", f"{max(0.1, total_s):.3f}",
           "-i", "anullsrc=r=24000:cl=mono"]
    for _, piece in pieces:
        cmd.extend(["-i", str(piece)])

    parts = []
    labels = ["[0:a]"]
    for idx, (start, _) in enumerate(pieces, start=1):
        parts.append(f"[{idx}:a]adelay={int(start * 1000)}|"
                     f"{int(start * 1000)}[d{idx}]")
        labels.append(f"[d{idx}]")
    graph = ";".join(parts) + ";" + "".join(labels) + \
        f"amix=inputs={len(labels)}:normalize=0:duration=first," \
        f"alimiter=limit=0.95[out]"

    cmd.extend(["-filter_complex", graph, "-map", "[out]",
                "-ar", "48000", "-ac", "1", "-c:a", "pcm_s16le", str(dest)])
    run(cmd, timeout=900.0)
    if not dest.is_file():
        raise ClipForgeError("dub track was not produced")
    log.info("dubbing.track_built", path=str(dest), cues=len(pieces),
             overruns=overruns)
    if report is not None:
        skipped = len(cues) - len(pieces)
        if skipped:
            report.append(f"{skipped} cue(s) could not be synthesised and "
                          f"are silent in the dub")
        if overruns:
            # Synthesised speech is routinely longer than the original
            # line, and this is the number that decides whether a dub is
            # usable. Logging it and not reporting it is how a dub that
            # talks over itself ships looking fine.
            report.append(
                f"{overruns} of {len(pieces)} lines are longer than the "
                f"original speech and overlap the next line. Listen before "
                f"publishing — a slower source dubs more cleanly.")
    return dest


def mux_dub(clip: Path, dub_audio: Path, dest: Path, *,
            keep_original_at: float = 0.0) -> Path:
    """Replace the clip's audio with the dub, video stream-copied.

    Video is copied, never re-encoded: the dub changes the audio track and
    nothing else, and a re-encode would silently invalidate the QA stage's
    measured geometry and integrity hash for no benefit.
    """
    ffmpeg = require_binary("ffmpeg")
    cmd = [str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(clip), "-i", str(dub_audio)]
    if keep_original_at > 0:
        # Duck the original under the dub rather than dropping it, when
        # asked — some clips need the crowd, the music or the laughter.
        cmd.extend([
            "-filter_complex",
            f"[0:a]volume={keep_original_at:.3f}[bed];"
            f"[bed][1:a]amix=inputs=2:normalize=0:duration=first,"
            f"alimiter=limit=0.95[out]",
            "-map", "0:v:0", "-map", "[out]"])
    else:
        cmd.extend(["-map", "0:v:0", "-map", "1:a:0"])
    cmd.extend(["-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", str(dest)])
    run(cmd, timeout=900.0)
    if not dest.is_file():
        raise ClipForgeError("dubbed video was not produced")
    log.info("dubbing.muxed", path=str(dest))
    return dest


# -------------------------------------------------------------- entry point

def dub_clip(ws: Workspace, filename: str, *, target: str,
             rejected: bool = False, audio: bool = True,
             keep_original_at: float = 0.0) -> DubResult:
    """Translate one clip's transcript and, where possible, voice it."""
    from clipforge import clipmeta
    from clipforge.config import load_config
    from clipforge.translate import build_translator

    if target not in _LANG_NAME:
        raise ClipForgeError(
            f"unknown language {target!r}; known: "
            f"{', '.join(c for c, _ in LANGUAGES)}")

    root = (Path(ws.clips) / "rejected") if rejected else Path(ws.clips)
    clip = root / filename
    if not clip.is_file():
        raise ClipForgeError(f"clip not found: {filename}")

    transcript = clipmeta.transcript_for(ws, filename, rejected=rejected)
    if not transcript.get("available"):
        raise ClipForgeError(
            f"no transcript for this clip: {transcript.get('reason')}")
    cues = cues_from_transcript(transcript)
    if not cues:
        raise ClipForgeError("transcript has no timed lines to translate")

    cfg = load_config(Path("config/config.toml"))
    translator = build_translator(cfg)
    if translator is None:
        # translator_blocker distinguishes the reasons — cloud off, cloud on
        # but unkeyed, or no local route either. A message that asserts one
        # of those on the wrong path contradicts the config it prints.
        raise ClipForgeError(
            f"translation is unavailable: {translator_blocker(cfg)}")

    result = DubResult(language=target, cues=len(cues))
    source_lang = transcript.get("language")
    if source_lang and source_lang == target:
        result.notes.append(
            f"source is already {_LANG_NAME[target]}; lines pass through")
        translated = [c.text for c in cues]
    else:
        translated = translator.translate(
            [c.text for c in cues], target=target, source=source_lang)
        result.notes.append(f"translated by {translator.name}")
    out_cues = [Cue(c.start, c.end, t) for c, t in zip(cues, translated)]

    stem = clip.with_suffix("")
    result.subtitles_path = write_srt(
        out_cues, Path(f"{stem}.{target}.srt"))

    if not audio:
        return result

    voices = installed_voice_languages()
    voice = voices.get(target)
    if not voice:
        result.notes.append(
            f"subtitles only: no installed Kokoro voice speaks "
            f"{_LANG_NAME[target]}. Fetch one into workspace/models/kokoro/"
            f"voices to dub the audio.")
        return result

    total = float(transcript.get("duration_s") or 0.0) or out_cues[-1].end
    work = Path(ws.tmp) / f"dub_{clip.stem[:16]}_{target}"
    track = build_dub_track(out_cues, Path(f"{stem}.{target}.wav"),
                            voice=voice, total_s=total, work_dir=work,
                            report=result.notes)
    result.audio_path = track
    result.voice = voice
    result.video_path = mux_dub(
        clip, track, Path(f"{stem}.{target}.mp4"),
        keep_original_at=keep_original_at)

    # The dub is a sibling render, so it gets the same sidecar treatment
    # the gallery expects: it shows up as its own card.
    for junk in work.glob("cue_*.wav"):
        junk.unlink(missing_ok=True)
    try:
        work.rmdir()
    except OSError:
        pass
    return result
