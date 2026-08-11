"""Provider routing: premium first, local when metered out, premium again
when the window resets.

The whole point is that this decision is made from RECORDED state rather
than from whatever happened in this process. A run started an hour after
the quota died must not re-probe the exhausted API; a run started after the
window reset must go straight back to the good model without being told.

Three failure classes, three different consequences — collapsing them is
how a router either hammers a dead API or abandons a working one:

* ``QuotaExhausted``   → record with its reset time, fall back, and come
                         back automatically when it lifts.
* ``ProviderUnavailable`` → not configured on this machine. Skip it for
                         the whole run. NOT recorded as quota: a missing
                         API key is not a spent quota, and writing it to
                         the ledger would sideline the provider for a day
                         after the operator adds the key.
* ``ProviderError``    → transient. Fall back for this shot, and let the
                         ledger sideline it only after repeated failures.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
import inspect
from typing import Any, Callable, Sequence

from clipforge.genvideo.presets import (Preset, build_shot_prompt,
                                        split_into_beats)
from clipforge.genvideo.providers import (GenResult, Provider, ProviderError,
                                          ProviderUnavailable, QuotaExhausted)
from clipforge.genvideo.quota import QuotaLedger
from clipforge.log import get_logger

log = get_logger(__name__)


def _takes_start_image(provider: Any) -> bool:
    """Whether this provider's generate() declares a start_image.

    Introspected, not assumed from the provider's name: a future provider
    that gains i2v support should start receiving frames without anyone
    remembering to edit a list here.
    """
    try:
        return "start_image" in inspect.signature(provider.generate).parameters
    except (TypeError, ValueError):  # builtins / C-implemented callables
        return False


@dataclass
class ShotOutcome:
    index: int
    provider: str
    path: Path | None
    prompt: str
    seconds: float
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.path is not None


@dataclass
class SequenceResult:
    shots: list[ShotOutcome] = field(default_factory=list)
    providers_used: list[str] = field(default_factory=list)
    degraded: bool = False

    @property
    def paths(self) -> list[Path]:
        return [s.path for s in self.shots if s.path is not None]

    @property
    def ok_count(self) -> int:
        return sum(1 for s in self.shots if s.ok)


class GenerationRouter:
    """Picks a provider per shot and keeps the ledger honest."""

    def __init__(self, providers: Sequence[Provider], ledger: QuotaLedger,
                 *, clock: Callable[[], float] = time.time) -> None:
        if not providers:
            raise ValueError("a router needs at least one provider")
        self.providers = list(providers)
        self.ledger = ledger
        self.clock = clock
        #: Providers that reported themselves unconfigured this run. Held
        #: in memory only — the operator adding a key mid-day should not
        #: have to wait out a persisted penalty.
        self._unconfigured: set[str] = set()

    # ------------------------------------------------------------ pick

    def eligible(self) -> list[Provider]:
        """Providers that could run right now, in preference order."""
        out = []
        for p in self.providers:
            if p.name in self._unconfigured:
                continue
            if not self.ledger.available(p.name):
                continue
            out.append(p)
        return out

    def status(self) -> list[dict[str, object]]:
        """Per-provider view for the dashboard."""
        snap = self.ledger.snapshot()
        rows = []
        for p in self.providers:
            st = snap.get(p.name, {})
            unconfigured = p.name in self._unconfigured
            rows.append({
                "name": p.name,
                "configured": (not unconfigured) and _safe_available(p),
                "quota_ok": bool(st.get("available", True)),
                "available_in_s": float(st.get("available_in_s", 0.0) or 0.0),
                "calls": int(st.get("calls", 0) or 0),
                "seconds_generated": float(st.get("seconds_generated", 0.0) or 0.0),
                "last_reason": str(st.get("last_reason", "") or ""),
            })
        return rows

    # -------------------------------------------------------- generate

    def generate_shot(self, *, prompt: str, seconds: float, fps: int,
                      out_path: Path, negative: str = "",
                      aspect_ratio: str = "9:16",
                      start_image: Path | None = None) -> GenResult:
        """Try providers in order; raise only when all of them are out."""
        errors: list[str] = []
        for provider in self.providers:
            if provider.name in self._unconfigured:
                continue
            if not self.ledger.available(provider.name):
                wait = self.ledger.seconds_until_available(provider.name)
                log.info("genvideo.skipping_metered_provider",
                         provider=provider.name, available_in_s=round(wait, 1))
                errors.append(f"{provider.name}: metered out for "
                              f"{wait:.0f}s more")
                continue
            try:
                if not provider.available():
                    raise ProviderUnavailable(f"{provider.name} not configured")
                kw: dict[str, Any] = dict(
                    prompt=prompt, seconds=seconds, fps=fps,
                    out_path=out_path, negative=negative,
                    aspect_ratio=aspect_ratio)
                # Only the local provider takes a start frame today. Passing
                # it blindly would break the cloud provider's signature, and
                # a `**kwargs` catch-all there would swallow it silently —
                # which reads as continuity that never happened.
                if start_image is not None and _takes_start_image(provider):
                    kw["start_image"] = start_image
                result = provider.generate(**kw)
            except ProviderUnavailable as exc:
                # NOT a quota event. Skip for this run only.
                self._unconfigured.add(provider.name)
                log.info("genvideo.provider_unconfigured",
                         provider=provider.name, detail=str(exc)[:200])
                errors.append(f"{provider.name}: {exc}")
                continue
            except QuotaExhausted as exc:
                self.ledger.record_exhausted(
                    provider.name, reason=str(exc),
                    retry_after_s=getattr(exc, "retry_after_s", None))
                errors.append(f"{provider.name}: quota exhausted")
                continue
            except ProviderError as exc:
                self.ledger.record_error(provider.name, reason=str(exc))
                errors.append(f"{provider.name}: {exc}")
                continue
            self.ledger.record_success(provider.name, seconds=result.seconds)
            return result
        raise ProviderError(
            "no generation provider could produce this shot -> "
            + "; ".join(errors))

    def generate_sequence(self, *, brief: str, preset: Preset,
                          out_dir: Path, shots: int | None = None,
                          aspect_ratio: str = "9:16",
                          continuity: bool = False) -> SequenceResult:
        """Generate a whole piece, shot by shot.

        A failed shot does not abort the sequence: five good shots and one
        gap is a piece an editor can still cut, and the outcome record says
        exactly which beat is missing rather than losing the run.

        With ``continuity`` (Wan2GP-style i2v chaining) each shot starts
        from the previous shot's last frame, so the cuts fall inside one
        continuous scene. A shot that FAILED contributes no frame, and the
        chain restarts from text rather than reaching further back — the
        beat after a gap is a different moment, and seeding it from before
        the gap would assert a continuity the piece does not have.
        """
        count = int(shots or preset.default_shots)
        beats = split_into_beats(brief, count)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = SequenceResult()
        seed_frame: Path | None = None
        for i in range(count):
            prompt = build_shot_prompt(brief, preset, shot_index=i,
                                       total_shots=count, beat=beats[i])
            dest = out_dir / f"shot_{i:02d}.mp4"
            try:
                gen = self.generate_shot(
                    prompt=prompt, seconds=preset.shot_seconds,
                    fps=preset.fps, out_path=dest, negative=preset.avoid,
                    aspect_ratio=aspect_ratio,
                    start_image=seed_frame if continuity else None)
            except ProviderError as exc:
                log.error("genvideo.shot_failed", shot=i, error=str(exc)[:300])
                result.shots.append(ShotOutcome(
                    i, "none", None, prompt, preset.shot_seconds,
                    error=str(exc)[:300]))
                result.degraded = True
                seed_frame = None  # the chain is broken; do not span the gap
                continue
            result.shots.append(ShotOutcome(
                i, gen.provider, gen.path, prompt, gen.seconds))
            if continuity and gen.path is not None:
                from clipforge.genvideo.providers import last_frame

                seed_frame = last_frame(
                    Path(gen.path), out_dir / f"shot_{i:02d}.last.jpg")
            if gen.provider not in result.providers_used:
                result.providers_used.append(gen.provider)
        # More than one provider in one piece means the quota flipped
        # mid-sequence; the caller should say so rather than pretend the
        # piece is uniform.
        if len(result.providers_used) > 1:
            result.degraded = True
        return result


def _safe_available(provider: Provider) -> bool:
    try:
        return bool(provider.available())
    except Exception:  # noqa: BLE001 - status must never raise
        return False
