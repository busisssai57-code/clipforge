"""Generation backends: Google Veo (metered, cloud) and a local model.

**Read this before enabling the cloud provider.** Everything else in BTA
runs on this machine and the Authorization Law says the pipeline produces
files and stops. Veo is a hosted API: enabling it means your prompts — and
any brief text you pass — are sent to Google. That is a deliberate,
operator-made exception, so it is OFF by default, it must be switched on in
config, and every call logs that content left the machine. The local
provider has no such caveat.

Both providers implement the same tiny protocol so the router can swap them
without knowing which is which:

    available() -> bool          # can this run at all (creds/model present)
    generate(...) -> GenResult   # produce one shot, or raise

Failure is typed, because the router has to tell three cases apart:
QuotaExhausted (fall back, retry later), ProviderUnavailable (not
configured — skip permanently this run), and ProviderError (transient).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from clipforge.log import get_logger

log = get_logger(__name__)


class ProviderUnavailable(RuntimeError):
    """Not configured on this machine (no key, no weights). Not an error
    in the run — just a provider that cannot participate."""


class QuotaExhausted(RuntimeError):
    """Metered capacity is gone. Carries the reset hint when the provider
    gave one, so the ledger does not have to guess a window."""

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ProviderError(RuntimeError):
    """Transient/unexpected failure. Worth a retry or a fallback, but not
    evidence that quota ran out."""


@dataclass(frozen=True)
class GenResult:
    path: Path
    provider: str
    seconds: float
    prompt: str
    model: str


class Provider(Protocol):
    name: str

    def available(self) -> bool: ...

    def generate(self, *, prompt: str, seconds: float, fps: int,
                 out_path: Path, negative: str = "",
                 aspect_ratio: str = "9:16") -> GenResult: ...


# --------------------------------------------------------------- Veo

#: Overridable so a model/endpoint revision does not require a code change.
VEO_ENDPOINT = os.environ.get(
    "CLIPFORGE_VEO_ENDPOINT",
    "https://generativelanguage.googleapis.com/v1beta")
VEO_MODEL = os.environ.get("CLIPFORGE_VEO_MODEL", "veo-3.1-generate-preview")

#: HTTP statuses that mean "metered out" rather than "broken".
_QUOTA_STATUSES = {429}
_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "quota", "QUOTA", "rate limit",
                  "billing", "exceeded")


def _looks_like_quota(status: int, body: str) -> bool:
    if status in _QUOTA_STATUSES:
        return True
    if status == 403 and any(m in body for m in _QUOTA_MARKERS):
        return True
    return False


def _dig(blob: Any, *paths: tuple[str, ...]) -> Any:
    """First value found at any of the given key paths.

    Response envelopes move between API revisions. Rather than hard-code
    one shape and silently return nothing when it changes, try the known
    shapes and let the caller raise with the raw body if all miss.
    """
    for path in paths:
        cur = blob
        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            elif isinstance(cur, list) and key.isdigit() and int(key) < len(cur):
                cur = cur[int(key)]
            else:
                cur = None
                break
        if cur is not None:
            return cur
    return None


class VeoProvider:
    """Google Veo via the Gemini REST API.

    Uses ``requests`` (already a dependency) rather than an SDK, so
    enabling this adds no package to an environment where pip has broken
    the CUDA torch build three times.
    """

    name = "veo"

    def __init__(self, api_key: str | None, *, model: str = VEO_MODEL,
                 endpoint: str = VEO_ENDPOINT, poll_interval_s: float = 10.0,
                 timeout_s: float = 900.0) -> None:
        self._key = (api_key or "").strip()
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s

    def available(self) -> bool:
        return bool(self._key)

    def _headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self._key,
                "Content-Type": "application/json"}

    def generate(self, *, prompt: str, seconds: float, fps: int,
                 out_path: Path, negative: str = "",
                 aspect_ratio: str = "9:16") -> GenResult:
        if not self.available():
            raise ProviderUnavailable(
                "no Veo API key. Set CLIPFORGE_GEMINI_API_KEY in .env "
                "(this sends prompts to Google — see the module docstring).")
        import requests  # noqa: PLC0415

        params: dict[str, Any] = {
            "aspectRatio": aspect_ratio,
            "durationSeconds": int(round(seconds)),
        }
        if negative:
            params["negativePrompt"] = negative
        body = {"instances": [{"prompt": prompt}], "parameters": params}

        # This is the line where content leaves the machine. It is logged
        # every time, deliberately.
        log.warning("genvideo.cloud_call", provider=self.name,
                    model=self.model, chars=len(prompt),
                    note="prompt sent to Google (operator-enabled)")

        url = f"{self.endpoint}/models/{self.model}:predictLongRunning"
        try:
            resp = requests.post(url, headers=self._headers(),
                                 json=body, timeout=120)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Veo request failed: {exc}") from exc

        text = resp.text or ""
        if _looks_like_quota(resp.status_code, text):
            raise QuotaExhausted(f"Veo quota exhausted: {text[:300]}",
                                 retry_after_s=_retry_after(resp.headers))
        if resp.status_code >= 400:
            raise ProviderError(
                f"Veo HTTP {resp.status_code}: {text[:400]}")

        op_name = _dig(_json(resp), ("name",), ("operation", "name"))
        if not op_name:
            raise ProviderError(
                f"Veo response carried no operation name: {text[:400]}")
        return self._await_and_download(op_name, prompt, seconds, out_path)

    def _await_and_download(self, op_name: str, prompt: str, seconds: float,
                            out_path: Path) -> GenResult:
        import requests  # noqa: PLC0415

        deadline = time.monotonic() + self.timeout_s
        while True:
            if time.monotonic() > deadline:
                raise ProviderError(
                    f"Veo operation {op_name} did not finish in "
                    f"{self.timeout_s:.0f}s")
            time.sleep(self.poll_interval_s)
            try:
                poll = requests.get(f"{self.endpoint}/{op_name}",
                                    headers=self._headers(), timeout=60)
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(f"Veo poll failed: {exc}") from exc
            if _looks_like_quota(poll.status_code, poll.text or ""):
                raise QuotaExhausted(
                    f"Veo quota exhausted while polling: {poll.text[:200]}",
                    retry_after_s=_retry_after(poll.headers))
            if poll.status_code >= 400:
                raise ProviderError(
                    f"Veo poll HTTP {poll.status_code}: {poll.text[:300]}")
            blob = _json(poll)
            if not blob.get("done"):
                continue
            if "error" in blob:
                err = json.dumps(blob["error"])[:400]
                if any(m in err for m in _QUOTA_MARKERS):
                    raise QuotaExhausted(f"Veo quota: {err}")
                raise ProviderError(f"Veo operation failed: {err}")
            uri = _dig(
                blob,
                ("response", "generateVideoResponse", "generatedSamples", "0",
                 "video", "uri"),
                ("response", "generatedVideos", "0", "video", "uri"),
                ("response", "predictions", "0", "videoUri"),
                ("response", "videos", "0", "uri"),
            )
            if not uri:
                raise ProviderError(
                    "Veo finished but no video URI was found in the "
                    f"response: {json.dumps(blob)[:500]}")
            return self._download(uri, prompt, seconds, out_path)

    def _download(self, uri: str, prompt: str, seconds: float,
                  out_path: Path) -> GenResult:
        import requests  # noqa: PLC0415

        out_path.parent.mkdir(parents=True, exist_ok=True)
        partial = out_path.with_suffix(out_path.suffix + ".partial")
        try:
            with requests.get(uri, headers={"x-goog-api-key": self._key},
                              stream=True, timeout=300) as r:
                if r.status_code >= 400:
                    raise ProviderError(
                        f"Veo download HTTP {r.status_code}: {r.text[:200]}")
                with open(partial, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except ProviderError:
            partial.unlink(missing_ok=True)
            raise
        except Exception as exc:  # noqa: BLE001
            partial.unlink(missing_ok=True)
            raise ProviderError(f"Veo download failed: {exc}") from exc
        if partial.stat().st_size == 0:
            partial.unlink(missing_ok=True)
            raise ProviderError("Veo returned a zero-byte video")
        partial.replace(out_path)
        return GenResult(out_path, self.name, seconds, prompt, self.model)


def _json(resp: Any) -> dict[str, Any]:
    try:
        out = resp.json()
        return out if isinstance(out, dict) else {"value": out}
    except Exception:  # noqa: BLE001
        raise ProviderError(
            f"non-JSON response from Veo: {(resp.text or '')[:300]}") from None


def _retry_after(headers: Any) -> float | None:
    for key in ("retry-after", "Retry-After", "x-ratelimit-reset"):
        try:
            val = headers.get(key)
        except Exception:  # noqa: BLE001
            val = None
        if val:
            try:
                return float(val)
            except (TypeError, ValueError):
                continue
    return None


# ------------------------------------------------------------- local

class LocalDiffusersProvider:
    """Open-source text-to-video via a local diffusers pipeline.

    Model-agnostic on purpose: the pipeline id comes from config, so the
    same code drives LTX-Video, Wan, CogVideoX or whatever the operator
    has weights for. It obeys the VRAM Law — the pipeline is built inside
    ``generate`` and dropped in ``finally``, never held across calls,
    because the clipping DAG needs the card back.
    """

    name = "local"

    def __init__(self, model_id: str, *, device: str = "cuda",
                 steps: int = 30, guidance_scale: float = 3.0,
                 seed: int = 1234, loras: list[str] | None = None,
                 lora_scale: float = 1.0,
                 step_cache_threshold: float = 0.0,
                 quantize: str = "none") -> None:
        self.model_id = model_id
        self.device = device
        self.steps = steps
        self.guidance_scale = guidance_scale
        self.seed = seed
        self.loras = list(loras or [])
        self.lora_scale = lora_scale
        self.step_cache_threshold = step_cache_threshold
        self.quantize = quantize

    def _apply_loras(self, pipe: Any) -> None:
        """Load LoRA weights, Wan2GP's main customisation surface.

        A pipeline without `load_lora_weights` is reported, not ignored:
        the operator listed weights in config and is entitled to know they
        did not take effect.
        """
        if not self.loras:
            return
        if not hasattr(pipe, "load_lora_weights"):
            log.warning("genvideo.lora_unsupported", model=self.model_id,
                        requested=len(self.loras),
                        note="this pipeline has no load_lora_weights; the "
                             "configured LoRAs were NOT applied")
            return
        loaded: list[str] = []
        for ref in self.loras:
            try:
                pipe.load_lora_weights(ref)
                loaded.append(ref)
            except Exception as exc:  # noqa: BLE001 - one bad LoRA is not fatal
                log.warning("genvideo.lora_failed", lora=ref,
                            error=str(exc)[:200])
        if loaded and hasattr(pipe, "fuse_lora"):
            try:
                pipe.fuse_lora(lora_scale=self.lora_scale)
            except Exception as exc:  # noqa: BLE001
                log.warning("genvideo.lora_fuse_failed", error=str(exc)[:200])
        log.info("genvideo.loras", requested=len(self.loras),
                 applied=len(loaded), scale=self.lora_scale)

    def _apply_step_cache(self, pipe: Any) -> None:
        """TeaCache-style step skipping, if this diffusers build has it.

        Deterministic for a FIXED threshold — the skip decision is a
        comparison against measured residuals, with no sampling — but the
        threshold changes the output, so it belongs in the params digest
        (it is, via GenVideoConfig). Not emulated by hand when the
        pipeline lacks support: a hand-rolled skip loop reaching into
        transformer internals is exactly the kind of thing that silently
        produces a flat brown rectangle.
        """
        if self.step_cache_threshold <= 0:
            return
        setter = getattr(pipe, "enable_teacache", None) or getattr(
            pipe, "enable_cache", None)
        if setter is None:
            log.warning("genvideo.step_cache_unsupported",
                        model=self.model_id,
                        threshold=self.step_cache_threshold,
                        note="this diffusers build exposes no cache hook; "
                             "generation runs at full step count")
            return
        try:
            setter(self.step_cache_threshold)
            log.info("genvideo.step_cache", threshold=self.step_cache_threshold)
        except Exception as exc:  # noqa: BLE001
            log.warning("genvideo.step_cache_failed", error=str(exc)[:200])

    def supports_start_image(self) -> bool:
        """Whether this pipeline can be given a first frame.

        Answered by INTROSPECTION at call time, not from a list of model
        names: the registry cannot know what the operator has installed,
        and a hardcoded list is how a working feature gets reported as
        missing. Cheap because it only inspects the loaded signature.
        """
        return bool(self._start_image_param)

    #: Set during `generate` once the pipeline is loaded. i2v pipelines
    #: disagree on the name — LTX takes `image`, Wan takes `image` too,
    #: some take `start_image` — so the accepted name is discovered.
    _start_image_param: str | None = None

    def available(self) -> bool:
        import importlib.util
        return (importlib.util.find_spec("diffusers") is not None
                and importlib.util.find_spec("torch") is not None)

    def generate(self, *, prompt: str, seconds: float, fps: int,
                 out_path: Path, negative: str = "",
                 aspect_ratio: str = "9:16",
                 start_image: Path | None = None) -> GenResult:
        if not self.available():
            raise ProviderUnavailable(
                "diffusers is not installed. `pip install diffusers` and set "
                "[genvideo].local_model_id to a text-to-video pipeline. "
                "Install deliberately: this venv's CUDA torch has been "
                "clobbered by careless pip runs three times.")
        from clipforge.gpu import ModelClass, gpu_session, hard_unload

        # Generate INSIDE the model's trained envelope, deliver at the
        # short-form size. See MAX_GEN_PIXELS: 704x1280 came back blank.
        width, height = _generation_dims(aspect_ratio)
        deliver = (DELIVERY_PORTRAIT if aspect_ratio == "9:16"
                   else DELIVERY_LANDSCAPE)
        frames = _latent_frames(seconds, fps)
        models: dict[str, Any] = {}
        try:
            with gpu_session(ModelClass.VL, budget_gb=16.0):
                import torch  # noqa: PLC0415
                from diffusers import DiffusionPipeline  # noqa: PLC0415

                # bfloat16, not float16: measured on LTX-Video, and it is
                # the dtype these video transformers are trained in — fp16
                # overflows in the VAE decode on some prompts.
                models["pipe"] = DiffusionPipeline.from_pretrained(
                    self.model_id, torch_dtype=torch.bfloat16)

                # ---- FIX 1: VAE must run in float32 ----
                # The transformer trains in bf16 but the VAE decoder
                # produces NaN in bf16 during latent→pixel reconstruction,
                # which manifests as solid brown frames. Keeping the
                # transformer in bf16 for speed while casting only the VAE
                # to fp32 is the documented fix for LTX-Video.
                if hasattr(models["pipe"], "vae"):
                    models["pipe"].vae.to(dtype=torch.float32)

                # ---- FIX 2: VAE tiling ----
                # Prevents latent overflow on longer generations and
                # reduces peak VRAM inside the VAE decode.
                if hasattr(models["pipe"].vae, "enable_tiling"):
                    models["pipe"].vae.enable_tiling()

                # Sequential offload rather than .to("cuda"): the
                # transformer plus a T5-class text encoder does not fit
                # alongside everything else this pipeline loads. Measured
                # 9.55 GB peak with offload on a 24 GB card.
                if hasattr(models["pipe"], "enable_model_cpu_offload"):
                    models["pipe"].enable_model_cpu_offload()
                else:
                    models["pipe"].to(self.device)

                # ---- Wan2GP-style controls -----------------------------
                # Each is applied only if the installed pipeline really
                # supports it, and each LOGS whether it took effect. A knob
                # that silently does nothing is the defect this project has
                # shipped most often, so "requested" and "applied" are
                # reported as separate facts.
                self._apply_loras(models["pipe"])
                self._apply_step_cache(models["pipe"])

                # ---- FIX 3: guidance_scale ----
                # LTX-Video needs CFG ~3.0 to produce actual content.
                # Without it the model defaults to unconditional generation
                # and every frame is a flat brown rectangle. Measured.
                call_kwargs: dict[str, Any] = dict(
                    prompt=prompt,
                    negative_prompt=negative or None,
                    width=width, height=height,
                    num_frames=frames,
                    num_inference_steps=self.steps,
                    guidance_scale=self.guidance_scale,
                )
                # ---- FIX 4: frame-rate conditioning + decode args ----
                # frame_rate conditions temporal generation and silently
                # defaults to 25; asking for 24 fps while the model assumes
                # 25 skews motion.
                #
                # The decode args are passed because they are the upstream
                # reference values, NOT because they fixed the blank
                # output. An earlier comment here claimed they were the
                # cause; the bisect disproved it — at 704x1280 the runs
                # with and without them were identical (std 2.67, motion
                # 0.88 both). Resolution was the whole story.
                #
                # Added only when the pipeline declares them: these are
                # LTX-family arguments and a different local model would
                # reject them outright.
                accepted = _accepted_params(models["pipe"])
                for name, value in (("decode_timestep", DECODE_TIMESTEP),
                                    ("decode_noise_scale", DECODE_NOISE_SCALE),
                                    ("frame_rate", int(fps))):
                    if name in accepted:
                        call_kwargs[name] = value

                # ---- Wan2GP: image-to-video continuity ----------------
                # The first frame is the previous shot's last frame, so a
                # cut lands inside one continuous scene. Only passed when
                # the pipeline declares the parameter; a t2v-only pipeline
                # is not asked, and the caller is told the shot was NOT
                # chained rather than being left to assume it was.
                self._start_image_param = next(
                    (n for n in ("image", "start_image", "init_image")
                     if n in accepted), None)
                if start_image is not None:
                    if self._start_image_param is None:
                        log.warning(
                            "genvideo.continuity_unsupported",
                            model=self.model_id,
                            note="pipeline takes no start image; this shot "
                                 "was generated from text alone")
                    else:
                        from PIL import Image  # noqa: PLC0415

                        with Image.open(start_image) as im:
                            call_kwargs[self._start_image_param] = (
                                im.convert("RGB").resize((width, height)))
                        log.info("genvideo.continuity_applied",
                                 param=self._start_image_param,
                                 frm=str(start_image))

                # ---- FIX 5: the seed the comments already claimed ----
                # With no generator the pipeline draws from torch's global
                # RNG, so a re-run of an identical brief produced a
                # different file — measured on two swarm runs of the same
                # task.
                #
                # Scope, stated precisely so this comment does not repeat
                # the failure it fixes: seeding pins the INITIAL LATENTS
                # (every draw from this generator); the CUDA kernels in the
                # bf16 denoise loop are otherwise unpinned
                # (torch.use_deterministic_algorithms is set nowhere in
                # this codebase). Byte-identity was therefore MEASURED, not
                # assumed: 2026-08-05, two full runs of the same prompt on
                # the reference machine (LTX-Video, seed 1234, 30 steps,
                # guidance 3.0, 49 frames @ 480x896, delivered 1080x1920
                # via libx264) produced byte-identical MP4s — 984,308 bytes,
                # sha256 b82769...0a5e both times, end to end through
                # denoise, VAE, upscale and encode. That proves §3.2 holds
                # HERE; it is same-machine evidence, not a cross-driver or
                # cross-GPU guarantee, and a new model or resolution
                # deserves the same one-command double-run check.
                #
                # The generator lives on the CPU deliberately: this
                # pipeline runs under enable_model_cpu_offload, and a CUDA
                # generator both fights the offload hooks and makes the
                # result depend on which device a submodule happened to be
                # on. diffusers seeds latents from a CPU generator and
                # moves them, which is the reproducible path.
                if "generator" in accepted:
                    call_kwargs["generator"] = (
                        torch.Generator(device="cpu").manual_seed(self.seed))
                else:
                    # Not silent: a pipeline that cannot be seeded cannot
                    # satisfy the Determinism Law, and the operator should
                    # learn that from a log rather than from two files.
                    log.warning("genvideo.unseedable_pipeline",
                                model=self.model_id,
                                note="pipeline __call__ takes no generator; "
                                     "output is not reproducible")
                log.info("genvideo.local_call", model=self.model_id,
                         seed=self.seed, seeded="generator" in accepted,
                         width=width, height=height, frames=frames)
                result = models["pipe"](**call_kwargs)
                video = _first_video(result)
                if video is None:
                    raise ProviderError(
                        f"{self.model_id} returned no frames")
                _write_video(video, out_path, fps, deliver=deliver)
        except ProviderError:
            raise
        except ImportError as exc:
            raise ProviderUnavailable(f"local pipeline unavailable: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"local generation failed ({self.model_id}): {exc}") from exc
        finally:
            hard_unload(models)
        return GenResult(out_path, self.name, seconds, prompt, self.model_id)


#: VAE decode conditioning, passed when the pipeline accepts it.
DECODE_TIMESTEP = 0.03
DECODE_NOISE_SCALE = 0.025

#: Pixel budget per frame for GENERATION. This is the finding that mattered
#: most: LTX-Video renders 704x480 (338k px) as real footage and 704x1280
#: (901k px) as a FLAT FILL — measured spatial std 45.7 vs 2.7 on identical
#: prompts and seeds. Asking a video model for a resolution outside its
#: trained envelope does not degrade gracefully; it returns an empty
#: rectangle that is structurally a valid video.
#:
#: So generation happens inside the envelope and the result is scaled to the
#: delivery size on encode. Upscaling a real image beats a blank one at
#: native resolution, and it is what every practical pipeline does.
#: 460k is measured, not guessed: 512x896 (459k) renders real footage and
#: 704x1280 (901k) is blank. Sits just above the largest proven-good size
#: so generation runs as large as the model reliably supports.
MAX_GEN_PIXELS = 460_000

#: Delivery size for 9:16 short-form. Generation is scaled up to this.
DELIVERY_PORTRAIT = (1080, 1920)
DELIVERY_LANDSCAPE = (1920, 1080)


def _generation_dims(aspect_ratio: str, *, budget: int = MAX_GEN_PIXELS,
                     multiple: int = 32) -> tuple[int, int]:
    """Largest on-grid size matching ``aspect_ratio`` within the budget.

    Returns (width, height). Both snap to ``multiple`` because the
    transformer patches a latent grid and rejects off-grid sizes outright.
    """
    ratio = (9.0 / 16.0) if aspect_ratio == "9:16" else (16.0 / 9.0)
    # width = ratio * height, and width * height <= budget
    height = (budget / ratio) ** 0.5
    width = ratio * height
    w = max(multiple, int(width // multiple) * multiple)
    h = max(multiple, int(height // multiple) * multiple)
    while w * h > budget and (w > multiple or h > multiple):
        if w >= h:
            w -= multiple
        else:
            h -= multiple
    return w, h


def last_frame(video: Path, dest: Path) -> Path | None:
    """Extract a video's final frame — the seed for the next shot.

    `-sseof -0.1` seeks from the END, which is the only reliable way to
    land on the last frame without knowing the duration. Returns None
    rather than raising: continuity is an enhancement, and failing to grab
    a frame must degrade the next shot to text-only, not kill the run.
    """
    from clipforge.ffmpeg import require_binary, run

    try:
        run([str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
             "-loglevel", "error", "-y", "-sseof", "-0.1", "-i", str(video),
             "-frames:v", "1", "-q:v", "2", str(dest)], timeout=60.0)
    except Exception as exc:  # noqa: BLE001 - degrade, never abort
        log.warning("genvideo.last_frame_failed", video=str(video),
                    error=str(exc)[:200])
        return None
    return dest if dest.is_file() else None


def _accepted_params(pipe: Any) -> frozenset[str]:
    """Argument names a pipeline's ``__call__`` accepts.

    Video pipelines differ in which conditioning arguments they take, and
    passing an unknown one is a hard TypeError. Introspecting keeps the
    LTX-specific arguments from breaking a different local model the
    operator points ``local_model_id`` at.
    """
    import inspect  # noqa: PLC0415

    try:
        return frozenset(inspect.signature(pipe.__call__).parameters)
    except (TypeError, ValueError):  # C-implemented or wrapped callables
        return frozenset()


def _round_to(value: int, multiple: int) -> int:
    """Nearest multiple of ``multiple``, at least one multiple.

    Video transformers patch the latent grid, so off-grid dimensions are
    rejected outright rather than resized. LTX wants multiples of 32.
    """
    return max(multiple, int(round(value / multiple)) * multiple)


def _latent_frames(seconds: float, fps: int) -> int:
    """Frame count these models actually accept: ``8n + 1``.

    The temporal VAE compresses in groups of 8 plus a keyframe, so a plain
    ``seconds * fps`` is rejected (144 frames is not 8n+1). Rounds to the
    nearest legal count of at least 9 — one group — because a request for
    a fraction of a group cannot be honoured at all.
    """
    want = max(1, int(round(seconds * fps)))
    groups = max(1, round((want - 1) / 8))
    return groups * 8 + 1


def _first_video(result: Any) -> Any:
    for attr in ("frames", "videos", "images"):
        val = getattr(result, attr, None)
        if val is not None and len(val):
            return val[0]
    return None


def _write_video(frames: Any, out_path: Path, fps: int,
                 deliver: tuple[int, int] | None = None) -> None:
    """Frames -> mp4 through ffmpeg, so the output matches what the rest
    of the DAG expects (yuv420p h264 the QA gate can probe).

    Accepts what diffusers actually returns, which is a LIST OF PIL
    IMAGES for the video pipelines — not the float array the first
    version of this assumed. ``np.asarray`` on that list yields an object
    array, so the conversion has to be explicit.
    """
    import subprocess  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from clipforge.ffmpeg import require_binary  # noqa: PLC0415

    seq = list(frames)
    if not seq:
        raise ProviderError("no frames to encode")
    if hasattr(seq[0], "convert"):                    # PIL images
        arr = np.stack([np.asarray(im.convert("RGB")) for im in seq])
    else:
        arr = np.asarray(seq)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ProviderError(f"unexpected frame array shape {arr.shape}")

    # ---- Blank-frame guard ----
    # The first version validated the plumbing (file exists, right
    # duration, right resolution) and never looked at a single pixel.
    # A flat-fill frame has near-zero spatial variance; a real scene has
    # hundreds. Check first and last frame so a fade-to-brown at the end
    # is caught too.
    _BLANK_VARIANCE_FLOOR = 10.0
    for label, frame_idx in (("first", 0), ("last", -1)):
        variance = float(np.var(arr[frame_idx].astype(np.float32)))
        if variance < _BLANK_VARIANCE_FLOOR:
            raise ProviderError(
                f"blank frame detected ({label} frame variance "
                f"{variance:.1f} < {_BLANK_VARIANCE_FLOOR}). "
                "The model produced a flat fill, not a video. Measured "
                "cause: a RESOLUTION OUTSIDE THE MODEL'S TRAINED ENVELOPE "
                "returns structurally perfect, completely empty frames "
                "(LTX-Video: 704x480 renders, 704x1280 is blank). Lower "
                "MAX_GEN_PIXELS, or use a model that supports this size. "
                "Also check guidance_scale and VAE dtype.")

    count, height, width = arr.shape[0], arr.shape[1], arr.shape[2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(out_path.suffix + ".partial")
    cmd = [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
           "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-"]
    if deliver and (deliver[0], deliver[1]) != (width, height):
        # Generation runs inside the model's trained envelope; delivery is
        # the short-form size. lanczos + a light sharpen, because a clean
        # upscale of real footage beats a blank frame at native size.
        cmd += ["-vf", (f"scale={deliver[0]}:{deliver[1]}:flags=lanczos,"
                        "unsharp=5:5:0.6:5:5:0.0,setsar=1")]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", "-f", "mp4", str(partial)]
    proc = subprocess.run(cmd, input=arr.tobytes(), capture_output=True,
                          timeout=600)
    if proc.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise ProviderError(
            f"encoding {count} frames failed: "
            f"{(proc.stderr or b'')[-300:].decode('utf-8', 'replace')}")
    partial.replace(out_path)
