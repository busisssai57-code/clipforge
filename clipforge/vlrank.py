"""Cloud vision-language ranking for S3, with the local model as fallback.

S3 judges each candidate window on hook strength, comprehensibility and
visual action, and writes the title and on-screen hook. Until now that was
Qwen2.5-VL-7B loaded locally in INT4 — 10 GB of VRAM, a hard pairing
constraint against ASR, and a model small enough that its judgements are
the weakest link in the score.

This module puts a stronger model in front of it. Gemini Pro reads the
same frames and the same transcript and returns the same JSON, so nothing
downstream changes: :mod:`clipforge.scorecard` still consumes
``hook_strength`` / ``comprehensibility`` / ``visual_action`` on the same
0–10 scale.

Three properties this keeps, because the pipeline's laws do not bend for
a faster model:

1. **Cloud is opt-in and reported.** ``s3.use_cloud`` decides whether this
   machine talks to Google at all. With it off the provider is never
   constructed, so no frame can leave. The artifact records which ranker
   produced it either way — a score whose provenance is unknown is a score
   that cannot be trusted.
2. **Fallback is for unavailability, never for quality.** No key, quota
   exhausted, network down, malformed response: fall back to the local
   model. A *programming* error propagates, because swallowing it is how
   this codebase previously ended up with stages that silently never ran.
3. **Nothing is invented.** A candidate the model would not score comes
   back unscored, and S3 keeps the local path's behaviour for it, rather
   than being handed a plausible 5.0.
"""

from __future__ import annotations

import base64
import io
import json
import time
from dataclasses import dataclass
from typing import Any

from clipforge.log import get_logger

log = get_logger(__name__)

#: Google AI Studio generateContent. Overridable so a model or endpoint
#: revision does not require a code change.
GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta"

#: Frames are re-encoded to JPEG before upload. The VL processor's local
#: cap is ~451k pixels per frame; matching it here keeps the two paths
#: looking at comparable detail and keeps a 10-candidate run small.
_MAX_EDGE = 768
_JPEG_QUALITY = 82

#: A ranking call is on the critical path of a run the operator is
#: watching. Bounded so a hung request cannot stall the DAG.
_TIMEOUT_S = 90.0
_MAX_ATTEMPTS = 3


class RankerUnavailable(RuntimeError):
    """No key, cloud disabled, or the dependency is missing."""


class RankerQuotaExhausted(RuntimeError):
    """The provider metered us out. Retry later; fall back for now."""

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class RankerError(RuntimeError):
    """The provider answered, but not with something usable."""


@dataclass(frozen=True)
class Judgement:
    """One candidate's scores, on the same 0–10 scale as the local path."""

    visual_action: float
    hook_strength: float
    comprehensibility: float
    justification: str
    title: str
    hook: str

    def as_kwargs(self) -> dict[str, Any]:
        return {
            "visual_action": self.visual_action,
            "hook_strength": self.hook_strength,
            "comprehensibility": self.comprehensibility,
            "justification": self.justification[:500],
            "title": self.title[:120],
            "hook": self.hook[:160],
        }


def _clamp10(value: Any, *, field: str) -> float:
    """0–10 or a hard error.

    A model that returns 85 for a 0–10 field has misunderstood the scale,
    and quietly clamping it to 10 would turn a misread into a top score.
    Values just outside the range are snapped; anything wilder is refused.
    """
    try:
        num = float(value)
    except (TypeError, ValueError) as exc:
        raise RankerError(f"{field} is not a number: {value!r}") from exc
    if num != num or num in (float("inf"), float("-inf")):
        raise RankerError(f"{field} is not finite")
    if num < -0.5 or num > 10.5:
        raise RankerError(f"{field}={num} is outside the 0-10 scale")
    return max(0.0, min(10.0, num))


def _encode_frame(image: Any) -> str:
    """PIL image → base64 JPEG, downscaled to the per-frame cap."""
    img = image
    if getattr(img, "mode", "RGB") != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    longest = max(w, h)
    if longest > _MAX_EDGE:
        scale = _MAX_EDGE / float(longest)
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_PROMPT = (
    "You are a short-form video editor picking and packaging clips for "
    "TikTok, Reels and Shorts. You are shown frames sampled evenly across "
    "one candidate clip, plus its transcript.\n\n"
    "Judge the clip on three axes, each 0-10:\n"
    "  visual_action     - how much visibly happens on screen\n"
    "  hook_strength     - how strongly the opening earns the next three "
    "seconds\n"
    "  comprehensibility - whether it stands alone without the surrounding "
    "video\n\n"
    "Then write the packaging. The title must describe what actually "
    "happens in these frames: specific and concrete, never generic, at "
    "most 8 words, no hashtags. The hook is the single line to put on "
    "screen in the first seconds.\n\n"
    "Score honestly. Most clips are not exceptional, and a run where "
    "everything scores 8+ is useless for ranking.\n\n"
    "Return ONLY a JSON object with exactly these keys: visual_action, "
    "hook_strength, comprehensibility, justification, title, hook."
)

#: Ask for JSON and mean it, so the response does not arrive wrapped in
#: prose or a markdown fence that then has to be guessed at.
_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "visual_action": {"type": "number"},
        "hook_strength": {"type": "number"},
        "comprehensibility": {"type": "number"},
        "justification": {"type": "string"},
        "title": {"type": "string"},
        "hook": {"type": "string"},
    },
    "required": ["visual_action", "hook_strength", "comprehensibility",
                 "justification", "title", "hook"],
}


class GeminiRanker:
    """Rank candidates with Gemini Pro over the Google AI Studio API."""

    #: Reported into the artifact so a score can be traced to its judge.
    kind = "gemini"

    def __init__(self, api_key: str | None, *, model: str = "gemini-2.5-pro",
                 endpoint: str = GEMINI_ENDPOINT,
                 timeout_s: float = _TIMEOUT_S,
                 max_attempts: int = _MAX_ATTEMPTS,
                 models: list[str] | None = None) -> None:
        self._key = (api_key or "").strip()
        # An ordered preference, not a single name. Gemini Pro is not on
        # the free API tier at all — it answers 429 with "limit: 0" rather
        # than 403 — so a single-model client silently drops all the way
        # to the local 7B on an unbilled key. Walking the chain uses Pro
        # the moment billing is enabled and Flash until then, which is
        # still a much stronger judge than the local model.
        self.models = [m for m in (models or [model]) if m]
        if not self.models:
            self.models = ["gemini-2.5-pro"]
        #: The model that last answered. Pinned after the first success so
        #: one run's candidates are all judged by the SAME model — ranking
        #: is a comparison, and scores from two judges do not compare.
        self.model = self.models[0]
        self._pinned: str | None = None
        self.endpoint = endpoint.rstrip("/")
        self.timeout_s = float(timeout_s)
        self.max_attempts = max(1, int(max_attempts))

    def available(self) -> bool:
        if not self._key:
            return False
        try:
            import requests  # noqa: F401,PLC0415
        except ImportError:
            return False
        return True

    def describe(self) -> str:
        return f"{self.kind}:{self._pinned or self.model}"

    # ---------------------------------------------------------------- call

    def judge(self, *, transcript: str, frames: list[Any],
              start_s: float, end_s: float) -> Judgement:
        """Score one candidate. Raises rather than returning a default."""
        if not self.available():
            raise RankerUnavailable(
                "no Gemini API key. Set CLIPFORGE_GEMINI_API_KEY in .env, or "
                "set s3.use_cloud = false to rank on the local model.")
        if not frames:
            raise RankerError("no frames extracted for this candidate")

        import requests

        parts: list[dict[str, Any]] = [{
            "text": (f"{_PROMPT}\n\nCandidate window: {start_s:.1f}s-"
                     f"{end_s:.1f}s\nTranscript: {transcript}")
        }]
        for frame in frames:
            parts.append({"inline_data": {"mime_type": "image/jpeg",
                                          "data": _encode_frame(frame)}})

        # Once a model has answered, stay on it for the rest of the run.
        chain = [self._pinned] if self._pinned else list(self.models)
        payload = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                # Deterministic: the Determinism Law applies to a judgement
                # that decides which clip ships as much as to a render.
                "temperature": 0.0,
                "topP": 1.0,
                "candidateCount": 1,
                "maxOutputTokens": 8192,
                "responseMimeType": "application/json",
                "responseSchema": _RESPONSE_SCHEMA,
            },
        }
        exhausted: Exception | None = None
        for model in chain:
            url = f"{self.endpoint}/models/{model}:generateContent"
            last: Exception | None = None
            for attempt in range(1, self.max_attempts + 1):
                try:
                    resp = requests.post(
                        url, params={"key": self._key},
                        headers={"Content-Type": "application/json"},
                        json=payload, timeout=self.timeout_s)
                except Exception as exc:  # noqa: BLE001 - network, not logic
                    last = RankerError(f"request failed: {exc}")
                    log.warning("vlrank.request_failed", model=model,
                                attempt=attempt, error=str(exc)[:200])
                    if attempt < self.max_attempts:
                        time.sleep(min(8.0, 1.5 ** attempt))
                        continue
                    raise last from exc

                # Quota and "not available on this tier" both arrive as
                # 429; a 404 means the name is retired. Either way this
                # model cannot serve the run, so try the next one down.
                if resp.status_code in (404, 429) or (
                        resp.status_code == 403 and "quota" in resp.text.lower()):
                    exhausted = RankerQuotaExhausted(
                        f"{model} unavailable ({resp.status_code}): "
                        f"{_error_message(resp.text)}",
                        retry_after_s=_retry_after(resp.headers))
                    log.info("vlrank.model_unavailable", model=model,
                             status=resp.status_code)
                    break
                if resp.status_code in (500, 502, 503, 504):
                    last = RankerError(f"provider {resp.status_code}")
                    log.warning("vlrank.server_error", model=model,
                                attempt=attempt, status=resp.status_code)
                    if attempt < self.max_attempts:
                        time.sleep(min(8.0, 1.5 ** attempt))
                        continue
                    exhausted = last
                    break
                if resp.status_code != 200:
                    raise RankerError(
                        f"Gemini {resp.status_code}: {resp.text[:300]}")

                judgement = _parse(resp.text)
                if self._pinned != model:
                    self._pinned = model
                    log.info("vlrank.model_pinned", model=model)
                return judgement

        raise exhausted or RankerError(
            f"no Gemini model in {self.models} could serve this request")


#: Translation reuses the ranker's key, model chain and error handling —
#: it is the same provider answering the same way, and a second client
#: would be a second place for the quota rules to drift.
_TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["lines"],
}


def translate_lines(ranker: GeminiRanker, lines: list[str], *,
                    target_language: str,
                    source_language: str | None = None) -> list[str]:
    """Translate subtitle lines, preserving the one-to-one line mapping.

    Each input line is a timed subtitle cue, so the output must have
    exactly as many lines as the input or the timings no longer describe
    the words. A count mismatch is an error, never a best-effort re-align:
    silently shifting captions is worse than not producing them.
    """
    if not lines:
        return []
    src = f" from {source_language}" if source_language else ""
    prompt = (
        f"Translate these {len(lines)} subtitle lines{src} into "
        f"{target_language}.\n"
        "Rules:\n"
        "- Return EXACTLY one translated line per input line, in order.\n"
        "- Never merge, split, reorder or drop a line; they are timed cues.\n"
        "- Keep it spoken and natural, not literal.\n"
        "- A line that is already in the target language passes through.\n"
        "- Preserve an empty line as an empty line.\n\n"
        + "\n".join(f"{i + 1}. {line}" for i, line in enumerate(lines))
    )
    body = _post_chain(
        ranker, [{"text": prompt}],
        schema=_TRANSLATE_SCHEMA, label="translate")
    try:
        blob = json.loads(body)
        text = "".join(
            part.get("text", "")
            for part in (((blob.get("candidates") or [{}])[0].get("content")
                          or {}).get("parts") or []))
        out = json.loads(text)["lines"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RankerError(f"translation response unusable: {exc}") from exc

    out = [str(x) for x in out]
    if len(out) != len(lines):
        raise RankerError(
            f"translator returned {len(out)} lines for {len(lines)} cues; "
            "timings would no longer match the words")
    return out


def _post_chain(ranker: GeminiRanker, parts: list[dict[str, Any]], *,
                schema: dict[str, Any], label: str) -> str:
    """POST to the first model in the chain that will serve us."""
    if not ranker.available():
        raise RankerUnavailable("no Gemini API key")
    import requests

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": 0.0, "topP": 1.0, "candidateCount": 1,
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": schema,
        },
    }
    chain = [ranker._pinned] if ranker._pinned else list(ranker.models)
    last: Exception | None = None
    for model in chain:
        url = f"{ranker.endpoint}/models/{model}:generateContent"
        try:
            resp = requests.post(url, params={"key": ranker._key},
                                 headers={"Content-Type": "application/json"},
                                 json=payload, timeout=ranker.timeout_s)
        except Exception as exc:  # noqa: BLE001
            last = RankerError(f"{label} request failed: {exc}")
            continue
        if resp.status_code in (404, 429):
            last = RankerQuotaExhausted(
                f"{model} unavailable ({resp.status_code}): "
                f"{_error_message(resp.text)}")
            log.info("vlrank.model_unavailable", model=model, task=label,
                     status=resp.status_code)
            continue
        if resp.status_code != 200:
            raise RankerError(f"{label} {resp.status_code}: {resp.text[:300]}")
        ranker._pinned = model
        return resp.text
    raise last or RankerError(f"no model could serve {label}")


def _error_message(body: str) -> str:
    """The human part of a Google API error, for the log and the artifact."""
    try:
        return str(json.loads(body).get("error", {}).get("message", ""))[:200]
    except ValueError:
        return body[:200]


def _retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("Retry-After")
    except Exception:  # noqa: BLE001
        return None
    if not value:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse(body: str) -> Judgement:
    """Pull the judgement out of a generateContent response."""
    try:
        blob = json.loads(body)
    except ValueError as exc:
        raise RankerError("response was not JSON") from exc

    candidates = blob.get("candidates") or []
    if not candidates:
        # A prompt-level block reports here, and it is worth naming: it
        # means the frames were refused, not that the clip scored badly.
        reason = ((blob.get("promptFeedback") or {}).get("blockReason")
                  or "no candidates in response")
        raise RankerError(f"Gemini returned nothing: {reason}")

    first = candidates[0]
    finish = first.get("finishReason")
    if finish and finish not in ("STOP", "MAX_TOKENS"):
        raise RankerError(f"Gemini stopped early: {finish}")

    text = "".join(
        part.get("text", "")
        for part in ((first.get("content") or {}).get("parts") or []))
    if not text.strip():
        raise RankerError("Gemini returned an empty judgement")

    # responseMimeType=application/json means this should already be bare
    # JSON, but a fenced block is cheap to survive.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise RankerError("no JSON object in the judgement")
    try:
        data = json.loads(cleaned[start:end + 1])
    except ValueError as exc:
        raise RankerError(f"judgement was not valid JSON: {exc}") from exc

    return Judgement(
        visual_action=_clamp10(data.get("visual_action"), field="visual_action"),
        hook_strength=_clamp10(data.get("hook_strength"), field="hook_strength"),
        comprehensibility=_clamp10(data.get("comprehensibility"),
                                   field="comprehensibility"),
        justification=str(data.get("justification") or "").strip(),
        title=str(data.get("title") or "").strip(),
        hook=str(data.get("hook") or "").strip(),
    )


def build_ranker(cfg: Any) -> GeminiRanker | None:
    """Construct the cloud ranker, or None when it must not exist.

    Returning None rather than a disabled object is deliberate: with the
    cloud off there is no object holding a key and no code path that could
    send a frame.
    """
    from clipforge.cloud import cloud_enabled, gemini_key  # noqa: PLC0415

    if not cloud_enabled(cfg, "s3_ranking"):
        return None
    # The credential comes through the §2 chokepoint (clipforge.cloud) —
    # the only module allowed to read it. A structural test fails any
    # other read, so this cannot regress back to an inline Secrets() call.
    key = gemini_key(cfg, feature="s3_ranking")
    s3 = getattr(cfg, "s3", None)
    ranker = GeminiRanker(
        key,
        model=getattr(s3, "cloud_model", "gemini-2.5-pro"),
        models=list(getattr(s3, "cloud_models", []) or []) or None,
        timeout_s=getattr(s3, "cloud_timeout_s", _TIMEOUT_S),
        max_attempts=getattr(s3, "cloud_max_attempts", _MAX_ATTEMPTS),
    )
    return ranker if ranker.available() else None
