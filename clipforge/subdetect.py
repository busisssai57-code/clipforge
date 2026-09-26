"""Does this footage already carry captions?

A great deal of short-form video arrives with its words already burned
into the picture — UGC shot on a phone, a repost, anything cut by another
tool. Adding karaoke captions on top of those produces two sets of words
fighting over the same frame, which is what a shipped clip of this
project looked like: the hook card landed on the source's own caption and
neither could be read.

Two questions, cheapest first:

* **Is there a subtitle TRACK?** ffprobe answers in milliseconds. A track
  is not burned in, but it says the source was captioned and the words
  are already available.
* **Is there text IN the picture?** Only the vision model can answer
  that, and it is the same Qwen VL weights S3 ranks with, under the same
  one-VL-at-a-time lease. Three frames, one question, a few GPU seconds.

The answer is deliberately conservative: unsure means "no", because
adding captions to a clip that has none is a worse failure than leaving
a captioned clip uncaptioned — the first ships a clip with no words at
all for a viewer watching on mute.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from clipforge.log import get_logger

log = get_logger(__name__)

#: Asked of the vision model. Deliberately about BURNED-IN words, and
#: deliberately not about any text: a shop sign in the background, a
#: jersey number or a phone UI are not captions, and treating them as
#: captions would silence our own.
ASK = (
    "Look at these frames from one short video. Are there SUBTITLES or "
    "CAPTIONS burned into the picture - words added on top of the footage "
    "that transcribe or paraphrase what someone is saying? Ignore text "
    "that belongs to the scene itself (signs, clothing, screens, logos) "
    "and ignore a channel handle or watermark. Answer with one word, YES "
    "or NO, then a short reason.")


@dataclass(frozen=True)
class SubtitleFinding:
    present: bool
    source: str          # "track" | "vision" | "flag" | "unknown"
    detail: str = ""

    def __bool__(self) -> bool:
        return self.present


def has_subtitle_track(video: Path) -> bool:
    """True when the container carries a subtitle stream."""
    try:
        from clipforge.ffmpeg import ffprobe_json

        blob = ffprobe_json(video)
    except Exception:  # noqa: BLE001 - a probe never decides the pipeline
        return False
    for stream in (blob.get("streams") or []):
        if str(stream.get("codec_type", "")).lower() == "subtitle":
            return True
    return False


def _verdict_from_text(text: str) -> bool | None:
    """YES/NO out of the model's prose, or None if it said neither."""
    head = (text or "").strip().lower()
    if not head:
        return None
    first = re.sub(r"[^a-z]", " ", head).split()
    for word in first[:6]:
        if word in ("yes", "yeah", "true"):
            return True
        if word in ("no", "none", "false"):
            return False
    return None


def ask_vision(frames: list[bytes], *, model_id: str,
               vram_budget_gb: float) -> bool | None:
    """Put the question to the local VL model. None = it did not answer."""
    if not frames:
        return None
    import io

    from PIL import Image

    from clipforge.gpu import ModelClass, gpu_session, hard_unload

    images = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    models: dict[str, Any] = {}
    try:
        with gpu_session(ModelClass.VL, vram_budget_gb):
            import torch
            from transformers import (AutoProcessor,
                                      Qwen2_5_VLForConditionalGeneration)

            models["processor"] = AutoProcessor.from_pretrained(model_id)
            models["model"] = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, dtype="auto", device_map="auto")
            messages = [{"role": "user", "content":
                         [*({"type": "image", "image": im} for im in images),
                          {"type": "text", "text": ASK}]}]
            prompt = models["processor"].apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = models["processor"](text=[prompt], images=images,
                                         return_tensors="pt",
                                         padding=True).to("cuda")
            with torch.no_grad():
                out = models["model"].generate(**inputs, max_new_tokens=60,
                                               do_sample=False)
            text = models["processor"].batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
        log.info("subdetect.vision_said", answer=(text or "")[:120])
        return _verdict_from_text(text)
    except Exception as exc:  # noqa: BLE001 - unsure is an answer, not a crash
        log.warning("subdetect.vision_unavailable",
                    error=f"{type(exc).__name__}: {exc}"[:200])
        return None
    finally:
        hard_unload(models)


def detect(video: Path, *, duration_s: float, cfg: Any,
           frames: Callable[..., list[bytes]] | None = None,
           vision: Callable[..., bool | None] | None = None,
           ) -> SubtitleFinding:
    """Does ``video`` already carry captions? Cheapest question first."""
    video = Path(video)
    if has_subtitle_track(video):
        return SubtitleFinding(True, "track", "the container has a subtitle stream")

    if not getattr(cfg.s5, "detect_existing", True):
        return SubtitleFinding(False, "flag", "detection is off in [s5]")

    if frames is None:
        from clipforge.vlqa import sample_frames as frames  # noqa: PLC0415
    if vision is None:
        vision = ask_vision
    try:
        shots = frames(video, duration_s, 3)
    except Exception as exc:  # noqa: BLE001
        log.warning("subdetect.frames_failed", error=str(exc)[:200])
        return SubtitleFinding(False, "unknown", "frames could not be read")

    answer = vision(shots, model_id=cfg.s3.model_id,
                    vram_budget_gb=cfg.s3.vram_budget_gb)
    if answer is None:
        # Unsure means no: a clip that needed captions and got none is
        # worse than one that kept the captions it already had.
        return SubtitleFinding(False, "unknown", "the model did not answer")
    return SubtitleFinding(bool(answer), "vision",
                           "words are already burned into the picture"
                           if answer else "no captions in the picture")


def caption_params(base: dict, *, existing: bool, hook_text: str,
                   keep_intervals: Any, uppercase: bool) -> dict:
    """What to ask S5 for, given whether the footage is already captioned.

    A function rather than a branch at the call site so it can be held by
    a test: the branch was string-matched, and a mutant that flipped it
    went unnoticed.

    ``captions`` rides in params, so it is in S5's cache key — the same
    window with and without existing captions is two different clips, and
    they must not share an artifact.
    """
    params = {**base, "keep_intervals": keep_intervals, "uppercase": uppercase}
    if existing:
        # No words of ours on top of theirs, and no hook line in the
        # subtitle track either: the hook goes on as an overlay instead.
        return {**params, "hook_text": "", "captions": "already_in_picture"}
    return {**params, "hook_text": hook_text}
