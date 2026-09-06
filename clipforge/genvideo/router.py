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

from clipforge.genvideo.presets import (Preset, beat_conflicts,
                                        build_shot_prompt,
                                        split_into_beats)
from clipforge.genvideo.providers import (GenResult, Provider, ProviderError,
                                          ProviderUnavailable, QuotaExhausted)
from clipforge.genvideo.quota import QuotaLedger
from clipforge.log import get_logger

log = get_logger(__name__)


def _takes_start_image(provider: Any) -> bool:
    """Whether this provider can actually START FROM a frame.

    ASK THE PROVIDER FIRST. A provider may declare `start_image` in its
    signature to satisfy the interface and still be text-to-video only:
    `SubprocessModelProvider` did exactly that, and said so in a
    `supports_start_image()` returning False whose comment explained the
    whole point -- "saying False is what makes the router stop threading
    last frames through here, rather than passing one that is silently
    ignored". This function asked the SIGNATURE instead, so the router
    believed the parameter and not the provider. Measured 2026-09-05: a
    five-shot batch with continuity ON came back byte-identical to one
    with it OFF, five last frames having been extracted, written to disk,
    handed over and dropped.

    Introspection stays as the fallback, for its original reason: a
    provider that gains i2v support without adding the method should still
    start receiving frames rather than waiting on someone to edit a list.
    """
    declared = getattr(provider, "supports_start_image", None)
    if callable(declared):
        try:
            return bool(declared())
        except Exception:  # noqa: BLE001 - a provider that cannot answer is a no
            return False
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
    #: Punchline marks the script put on this beat, carried through so the
    #: post layer can stamp them without re-parsing the screenplay and
    #: re-deriving a shot distribution that has already been decided.
    marks: list[str] = field(default_factory=list)
    #: What is SPOKEN over this beat, carried for the same reason the
    #: marks are. It is deliberately absent from `prompt`: dialogue is
    #: what a character says, not what the camera sees, and feeding it to
    #: a video model puts subtitles and mouth-shaped artefacts in frame.
    #: Kept here so the post layer can burn it, or a voice can read it,
    #: without re-parsing the screenplay -- which is what nothing did,
    #: leaving a Somali sketch's punchline parsed and thrown away.
    spoken: str = ""

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

    # ----------------------------------------------------------- close

    def close_providers(self) -> None:
        """Release providers that hold something between shots.

        `SubprocessModelProvider` keeps a worker — and, through it, a
        `gpu_session` and 13 GB of card — alive across a whole sequence
        on purpose: its model costs 81-103 s to load and 56-61 s to run,
        so a process per shot nearly doubles a brief. The cost of that
        choice is that somebody has to say when the sequence is over. A
        run that ends without this leaves the session held and the next
        GPU stage waiting on a lock nobody will release.

        Failures here are logged, not raised: a provider that will not
        shut down cleanly must not turn a finished piece into an error.
        """
        for provider in self.providers:
            close = getattr(provider, "close", None)
            if close is None:
                continue
            try:
                close()
            except Exception as exc:  # noqa: BLE001
                log.warning("genvideo.provider_close_failed",
                            provider=provider.name, error=str(exc)[:200])

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
                          continuity: bool = False,
                          screenplay: bool = False,
                          anchor: Path | None = None) -> SequenceResult:
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
        # `screenplay` reached this method for the first time on
        # 2026-08-18. `bta generate --screenplay` had been a declared flag
        # that nothing read: the CLI parsed it and then always split the
        # brief on full stops, so a Fountain script generated exactly as
        # if it were prose. The dashboard's own generate call went through
        # `build_storyboard`, which did honour it — which is why the flag
        # looked implemented from the UI side.
        marks: list[list[str]] = [[] for _ in range(count)]
        # Prose briefs have no dialogue; screenplay mode fills this in.
        lines: list[str] = ["" for _ in range(count)]
        if screenplay:
            from clipforge.screenplay import (  # noqa: PLC0415
                to_beats_with_dialogue)

            pairs = to_beats_with_dialogue(brief, count)
            if pairs:
                beats = [b for b, _, _ in pairs]
                marks = [m for _, m, _ in pairs]
                lines = [t for _, _, t in pairs]
                # The piece context handed to every shot becomes the
                # PICTURE-only synopsis. Left as the raw script it put
                # the dialogue back into each prompt one line after the
                # beat had removed it.
                from clipforge.screenplay import synopsis  # noqa: PLC0415

                brief = synopsis(brief)
            else:
                log.info("genvideo.screenplay_empty",
                         note="no scenes parsed; falling back to sentences")
                beats = split_into_beats(brief, count)
        else:
            beats = split_into_beats(brief, count)
        out_dir.mkdir(parents=True, exist_ok=True)
        result = SequenceResult()
        try:
            self._run_shots(result, beats, marks, lines, anchor=anchor,
                            brief=brief, preset=preset,
                            out_dir=out_dir, count=count,
                            aspect_ratio=aspect_ratio, continuity=continuity)
        finally:
            # Whatever happened — finished, failed, interrupted — the
            # sequence is over, so anything a provider was holding for
            # its duration goes back now.
            self.close_providers()
        # More than one provider in one piece means the quota flipped
        # mid-sequence; the caller should say so rather than pretend the
        # piece is uniform.
        if len(result.providers_used) > 1:
            result.degraded = True
        return result

    def _run_shots(self, result: SequenceResult, beats: list[str],
                   marks: list[list[str]], lines: list[str], *,
                   anchor: Path | None = None,
                   brief: str, preset: Preset,
                   out_dir: Path, count: int, aspect_ratio: str,
                   continuity: bool) -> None:
        """The shot loop itself, so `generate_sequence` can wrap it."""
        # THE ANCHOR, not a rolling chain. Set once from the first shot
        # that succeeds, and reused by every later shot.
        #
        # Seeding each shot from the PREVIOUS shot's last frame compounds
        # drift twice over: a shot is at its worst on its final frame, and
        # that worst frame then becomes the next shot's starting truth.
        # Measured 2026-09-05 on a three-shot batch -- by the end of shot
        # 2 the goat had fused with the child and three horns were growing
        # out of the toddler's scalp. Wardrobe continuity is worth nothing
        # if the subject decays into a chimera by the third cut.
        #
        # An anchor keeps every shot ONE generation from a clean reference
        # rather than N, so drift stays O(1) in sequence length. Shots are
        # no longer frame-continuous with their immediate predecessor,
        # which costs nothing in a format built on hard cuts: what has to
        # match across a cut is the child, the wardrobe and the courtyard,
        # and the anchor holds those better than a drifting chain did.
        # A SUPPLIED anchor beats a generated one, and beats it from shot
        # zero. Locking composition and wardrobe to a picture is the only
        # lever that actually works here: text does not win the argument.
        # A negative prompt naming "a woven mat", "a metal gate" and "two
        # children" produced a seated child on a woven mat in front of a
        # metal gate, with two children on it. Changing the seed moved the
        # composition more than any wording did.
        seed_frame: Path | None = Path(anchor) if anchor else None
        supplied = seed_frame is not None
        for i in range(count):
            prompt = build_shot_prompt(brief, preset, shot_index=i,
                                       total_shots=count, beat=beats[i])
            # A beat lands FIRST in the prompt, so a stale script beats
            # every correction in the niche and does it silently. Said out
            # loud, per shot, because four rounds of prompt work looked
            # ignored when it was being contradicted.
            clash = beat_conflicts(beats[i], preset.avoid)
            if clash:
                log.warning("genvideo.beat_conflicts_niche", shot=i,
                            asks_for=clash,
                            note="the script's own beat asks for things "
                                 "this niche forbids; the beat is first in "
                                 "the prompt and will win")
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
                    error=str(exc)[:300], marks=list(marks[i]),
                    spoken=lines[i]))
                result.degraded = True
                # The anchor SURVIVES a gap. When this was a rolling chain,
                # clearing it was right: seeding the beat after a gap from
                # before the gap asserted a continuity the piece did not
                # have. An anchor asserts something weaker and true -- that
                # this is the same child in the same place -- which a
                # missing beat does not falsify.
                continue
            result.shots.append(ShotOutcome(
                i, gen.provider, gen.path, prompt, gen.seconds,
                marks=list(marks[i]), spoken=lines[i]))
            if (continuity and gen.path is not None and seed_frame is None
                    and not supplied):
                from clipforge.genvideo.providers import first_frame

                seed_frame = first_frame(
                    Path(gen.path), out_dir / "anchor.jpg")
            if gen.provider not in result.providers_used:
                result.providers_used.append(gen.provider)


def _safe_available(provider: Provider) -> bool:
    try:
        return bool(provider.available())
    except Exception:  # noqa: BLE001 - status must never raise
        return False
