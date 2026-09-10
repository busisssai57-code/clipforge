"""A vision-language second opinion on a finished clip.

S7's deterministic checks measure the container: codec, geometry, loudness,
true peak, black frames, caption phrasing, whether a head is visible. They
are cheap, exact, and replayable, and they are the reason a clip with 96 kHz
audio or a caption cut mid-clause can no longer ship.

They still cannot see whether a clip is *good*. A holdout run scored eleven
clips at a mean of 70.5 with a 100% QA pass rate, and the operator deleted
every one on sight. That gap is what this module is for: a judge that looks
at the picture and says what is wrong with it in words.

The chain, in order:

    anthropic -> openai -> gemini -> local

Each link is skipped when it has no credential, and the verdict always
records which one actually ran. That matters more than it sounds: a chain
that silently degrades to its weakest link, and reports a verdict either
way, is how a hosted judge appears to be working for a month after its key
expired.

The local link is last and is never skipped - it needs no key and no
network, so there is always a verdict. It runs the same Qwen VL weights S3
ranks with, under the same `gpu_session(ModelClass.VL)` lease, because two
VL models resident at once is exactly what the VRAM Law forbids.

Nothing here reads a credential. `clipforge.cloud.provider_key` is the only
code that may, and it hands one out only when the operator's `[s7] use_cloud`
authorizes the registered `vl_qa` feature.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipforge.ffmpeg import require_binary
from clipforge.log import get_logger

log = get_logger(__name__)

#: Try them in this order. Local is last and always answers.
CHAIN: tuple[str, ...] = ("anthropic", "openai", "gemini", "local")

#: Fixed sampling positions, as fractions of duration. Fixed rather than
#: random because this stage is under the Determinism Law like any other:
#: a verdict that looked at different frames each run cannot be replayed
#: against the run that produced it.
def taps(n: int) -> tuple[float, ...]:
    """``n`` evenly spaced sample points, clear of the very first and last
    frame (both are often a fade and tell you nothing)."""
    n = max(1, n)
    return tuple((i + 1) / (n + 1) for i in range(n))


ASK = (
    "You are grading one short-form vertical video clip for a creator who "
    "will post it. You are shown frames sampled evenly through it.\n\n"
    "Report only defects a viewer would notice in the first two seconds or "
    "that would make them scroll past. Be specific and concrete.\n\n"
    "Look for: no face or the back of a head in the opening frame; a face "
    "cut by the frame edge; captions that collide with text already burned "
    "into the source footage; captions that are unreadable against the "
    "background; a subject who is off-centre with dead space beside them; "
    "a frame that is black, frozen, or a transition.\n\n"
    "Do NOT comment on the content, the speaker's opinions, or production "
    "value you cannot change by re-cutting.\n\n"
    'Return ONLY a JSON object: {"ok": bool, "findings": [str], '
    '"opening_frame_ok": bool, "worst": str}'
)


@dataclass
class Verdict:
    """What the judge said, and which judge said it."""

    provider: str
    ok: bool = True
    findings: list[str] = field(default_factory=list)
    opening_frame_ok: bool = True
    worst: str = ""
    note: str = ""
    frames_seen: int = 0
    #: Links that were tried and could not run, with why. Kept so a report
    #: can say "Claude was skipped, no key" rather than implying it passed.
    skipped: list[str] = field(default_factory=list)

    @property
    def ran(self) -> bool:
        return self.provider != "none"


def sample_frames(clip: Path, duration_s: float, count: int) -> list[bytes]:
    """JPEG bytes at fixed fractions of the clip, smallest useful size."""
    ffmpeg = require_binary("ffmpeg")
    out: list[bytes] = []
    for frac in taps(count):
        ts = max(0.0, duration_s * frac)
        r = subprocess.run(
            [ffmpeg, "-v", "error", "-ss", f"{ts:.3f}", "-i", str(clip),
             "-frames:v", "1", "-vf", "scale=512:-2", "-f", "image2pipe",
             "-vcodec", "mjpeg", "-"],
            capture_output=True, timeout=60)
        if r.returncode == 0 and r.stdout:
            out.append(r.stdout)
    return out


def _parse(text: str, provider: str, seen: int) -> Verdict:
    """Read the model's JSON, tolerating the fence it often wraps it in."""
    blob = text.strip()
    m = re.search(r"\{.*\}", blob, re.S)
    if not m:
        return Verdict(provider=provider, ok=True, frames_seen=seen,
                       note="judge returned no JSON; treated as no finding")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return Verdict(provider=provider, ok=True, frames_seen=seen,
                       note="judge returned unparseable JSON")
    findings = [str(f) for f in (data.get("findings") or []) if str(f).strip()]
    return Verdict(
        provider=provider,
        ok=bool(data.get("ok", not findings)),
        findings=findings,
        opening_frame_ok=bool(data.get("opening_frame_ok", True)),
        worst=str(data.get("worst", "") or "")[:300],
        frames_seen=seen,
    )


# --- the hosted links ----------------------------------------------------
# Each takes an already-granted key. None of them reads one.

def _post(url: str, headers: dict, body: dict, timeout: int = 120) -> Any:
    import requests  # noqa: PLC0415

    r = requests.post(url, headers=headers, json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
    return r.json()


def judge_anthropic(frames: list[bytes], key: str, *, model: str) -> Verdict:
    content: list[dict] = [
        {"type": "image", "source": {"type": "base64",
                                     "media_type": "image/jpeg",
                                     "data": base64.b64encode(f).decode()}}
        for f in frames]
    content.append({"type": "text", "text": ASK})
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {"x-api-key": key, "anthropic-version": "2023-06-01",
         "content-type": "application/json"},
        {"model": model, "max_tokens": 700,
         "messages": [{"role": "user", "content": content}]})
    text = "".join(p.get("text", "") for p in data.get("content", []))
    return _parse(text, "anthropic", len(frames))


def judge_openai(frames: list[bytes], key: str, *, model: str) -> Verdict:
    content: list[dict] = [{"type": "text", "text": ASK}]
    content += [
        {"type": "image_url",
         "image_url": {"url": "data:image/jpeg;base64,"
                              + base64.b64encode(f).decode()}}
        for f in frames]
    data = _post(
        "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        {"model": model, "max_tokens": 700,
         "messages": [{"role": "user", "content": content}]})
    text = data["choices"][0]["message"]["content"]
    return _parse(text, "openai", len(frames))


def judge_gemini(frames: list[bytes], key: str, *, model: str) -> Verdict:
    parts: list[dict] = [
        {"inline_data": {"mime_type": "image/jpeg",
                         "data": base64.b64encode(f).decode()}}
        for f in frames]
    parts.append({"text": ASK})
    data = _post(
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent",
        {"x-goog-api-key": key, "Content-Type": "application/json"},
        {"contents": [{"parts": parts}]})
    text = "".join(
        p.get("text", "")
        for p in data["candidates"][0]["content"].get("parts", []))
    return _parse(text, "gemini", len(frames))


# --- the local link ------------------------------------------------------

def judge_local(frames: list[bytes], *, model_id: str,
                vram_budget_gb: float) -> Verdict:
    """The same Qwen VL weights S3 ranks with, under the same lease.

    Held inside `gpu_session(ModelClass.VL)` because the VRAM Law allows one
    VL model resident at a time, and S7 runs while nothing else should be.
    """
    import io  # noqa: PLC0415

    from PIL import Image  # noqa: PLC0415

    from clipforge.gpu import ModelClass, gpu_session, hard_unload  # noqa: PLC0415

    images = [Image.open(io.BytesIO(f)).convert("RGB") for f in frames]
    models: dict[str, Any] = {}
    try:
        with gpu_session(ModelClass.VL, vram_budget_gb):
            import torch  # noqa: PLC0415
            from transformers import (  # noqa: PLC0415
                AutoProcessor, Qwen2_5_VLForConditionalGeneration)

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
                out = models["model"].generate(**inputs, max_new_tokens=400,
                                               do_sample=False)
            text = models["processor"].batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0]
        return _parse(text, "local", len(frames))
    finally:
        hard_unload(models)


# --- the chain -----------------------------------------------------------

def judge(clip: Path, *, cfg: Any, duration_s: float) -> Verdict:
    """Walk the chain and return the first verdict anything could give.

    Never raises. A judge that takes the pipeline down when a key expires is
    worse than no judge, so every failure becomes a skip with a reason and
    the chain moves on.
    """
    from clipforge import cloud  # noqa: PLC0415

    s7 = getattr(cfg, "s7", None)
    if not getattr(s7, "vl_qa", False):
        return Verdict(provider="none", note="[s7] vl_qa is off")

    frames = sample_frames(clip, duration_s, getattr(s7, "vl_frames", 6))
    if not frames:
        return Verdict(provider="none", note="no frame could be decoded")

    skipped: list[str] = []
    for name in CHAIN:
        try:
            if name == "local":
                s3 = getattr(cfg, "s3", None)
                v = judge_local(
                    frames,
                    model_id=getattr(s3, "local_model_id", None)
                    or "Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
                    vram_budget_gb=float(getattr(s3, "vram_budget_gb", 8.0)))
            else:
                key = cloud.provider_key(cfg, feature="vl_qa", provider=name)
                if not key:
                    skipped.append(f"{name}: no key, or [s7] use_cloud is off")
                    continue
                model = getattr(s7, f"{name}_model", None) or _DEFAULT_MODEL[name]
                v = _HOSTED[name](frames, key, model=model)
            v.skipped = skipped
            log.info("vlqa.verdict", provider=v.provider, ok=v.ok,
                     findings=len(v.findings), frames=v.frames_seen)
            return v
        except Exception as exc:  # noqa: BLE001 - a dead link is not a dead pipeline
            skipped.append(f"{name}: {type(exc).__name__}: {exc}"[:180])
            log.warning("vlqa.link_failed", provider=name,
                        error=str(exc)[:200])
    return Verdict(provider="none", skipped=skipped,
                   note="every link in the chain declined or failed")


_HOSTED = {
    "anthropic": judge_anthropic,
    "openai": judge_openai,
    "gemini": judge_gemini,
}

_DEFAULT_MODEL = {
    "anthropic": "claude-sonnet-5",
    "openai": "gpt-5",
    "gemini": "gemini-2.5-flash",
}
