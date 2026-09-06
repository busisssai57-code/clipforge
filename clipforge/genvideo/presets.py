"""Creative presets — the four modes the generator is asked to cover.

Each preset is the difference between a prompt that produces stock-footage
mush and one that produces a shot. They encode what a human creator would
actually specify: lens and camera behaviour, lighting, pacing, grade, and
what NOT to do. Providers translate these into their own dialects; the
preset itself stays provider-neutral so the same brief renders on the
premium model and on the local fallback.

Shot lengths are deliberately short. Every current text-to-video model
degrades past a handful of seconds — coherence drifts, faces melt, motion
loops. Long pieces are built by generating SHOTS and cutting them, which
is also how the human job is actually done.
"""

from __future__ import annotations

from clipforge.log import get_logger

log = get_logger(__name__)

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    summary: str
    #: Appended to every shot prompt — style, lens, grade, motion.
    style: str
    #: What the model must avoid. Providers that support it pass this as a
    #: negative prompt; the rest fold it into the prompt text.
    avoid: str
    #: Seconds per generated shot.
    shot_seconds: float
    #: Frames per second requested from the provider.
    fps: int
    #: Whether the mode expects narration audio over the visuals.
    narrated: bool
    #: Default number of shots when the caller does not say.
    default_shots: int
    #: Cutting rhythm hint used when assembling shots into a sequence.
    cut_style: str
    keywords: tuple[str, ...] = field(default_factory=tuple)
    #: Chain each shot from the previous shot's last frame (i2v), so the
    #: cuts fall inside ONE continuous scene rather than five separate
    #: ones. Off by default: a piece whose beats are deliberately
    #: different moments should not assert a continuity it does not have.
    #: On for any niche that is one scene, which is what a 33-shot sketch
    #: of one child and one goat is.
    continuity: bool = False
    #: See Niche.loop_strength: pin the last frame as well as the first.
    loop_strength: float = 0.0


DOCUMENTARY = Preset(
    name="documentary",
    summary="Observational non-fiction: real places, real light, no gloss.",
    style=(
        "observational documentary cinematography, 35mm anamorphic lens, "
        "natural available light, handheld with subtle weight and breath, "
        "shallow depth of field, muted filmic colour grade with lifted "
        "blacks, patient slow push-in, authentic unposed subjects, "
        "photojournalistic framing"),
    avoid=("cartoon, 3d render, cgi, video game, text overlay, watermark, "
           "logo, distorted hands, extra limbs, stock-footage smile at camera"),
    shot_seconds=6.0,
    fps=24,
    narrated=True,
    default_shots=6,
    cut_style="slow",
    keywords=("archival", "verite", "observational"),
)

STORYTELLING = Preset(
    name="storytelling",
    summary="Narrative scenes with a subject, a place, and a turn.",
    style=(
        "cinematic narrative film still in motion, 50mm spherical lens, "
        "motivated practical lighting, gentle dolly and parallax, rich "
        "contrast with deep shadows, warm highlight roll-off, character "
        "centred in a legible environment, single clear action per shot"),
    avoid=("montage, collage, split screen, text overlay, watermark, "
           "morphing faces, extra fingers, jump in identity between frames"),
    shot_seconds=5.0,
    fps=24,
    narrated=True,
    default_shots=8,
    cut_style="medium",
    keywords=("narrative", "character", "scene"),
)

EXPLAINER = Preset(
    name="explainer",
    summary="Professional explainer: clean, bright, presenter-friendly.",
    style=(
        "clean professional explainer footage, bright soft key light with "
        "large source, shallow but legible depth of field, locked-off or "
        "slow smooth slider move, neutral modern colour grade, uncluttered "
        "backgrounds with negative space for captions, corporate documentary "
        "polish"),
    avoid=("clutter, busy background, harsh shadows, lens flare, text "
           "overlay, watermark, shaky handheld, distorted hands"),
    shot_seconds=5.0,
    fps=30,
    narrated=True,
    default_shots=6,
    cut_style="medium",
    keywords=("explainer", "corporate", "tutorial"),
)

MOTION_GRAPHICS = Preset(
    name="motion_graphics",
    summary="Abstract/graphic motion for titles, transitions and beds.",
    style=(
        "premium abstract motion graphics, clean vector and soft gradient "
        "shapes, smooth eased keyframe motion, shallow parallax depth, "
        "limited palette with one accent colour, seamless looping energy, "
        "broadcast title-sequence quality, generous negative space"),
    avoid=("photorealistic humans, faces, text, letters, watermark, logo, "
           "jitter, strobing, harsh flicker"),
    shot_seconds=4.0,
    fps=30,
    narrated=False,
    default_shots=4,
    cut_style="fast",
    keywords=("abstract", "title", "transition", "bed"),
)

PRESETS: dict[str, Preset] = {
    p.name: p for p in (DOCUMENTARY, STORYTELLING, EXPLAINER, MOTION_GRAPHICS)
}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(
            f"unknown preset {name!r}; available: "
            f"{', '.join(sorted(PRESETS))}") from None


def build_shot_prompt(brief: str, preset: Preset, *, shot_index: int,
                      total_shots: int, beat: str | None = None) -> str:
    """One shot's prompt: the beat, the brief, then the house style.

    Order is deliberate. Diffusion and autoregressive video models both
    weight early tokens most, so the SUBJECT leads and the style trails.
    Putting the style first produces four shots that look identical and
    ignore the brief.
    """
    subject = (beat or brief).strip().rstrip(".")
    position = (
        "opening establishing shot" if shot_index == 0 and total_shots > 1
        else "closing shot" if shot_index == total_shots - 1 and total_shots > 1
        else f"shot {shot_index + 1} of {total_shots}")
    parts = [subject]
    if beat and beat.strip() != brief.strip():
        parts.append(f"part of: {brief.strip().rstrip('.')}")
    parts.append(position)
    parts.append(preset.style)
    return ". ".join(p for p in parts if p) + "."


def split_into_beats(brief: str, shots: int) -> list[str]:
    """Split a brief into per-shot beats.

    Sentence-per-beat when the brief has enough sentences, otherwise the
    whole brief carries every shot and the position hint does the work of
    differentiating them. Deliberately not a model call: this runs before
    any provider is chosen and must behave identically for both.
    """
    import re

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", brief.strip())
                 if s.strip()]
    if not sentences:
        return [brief.strip()] * max(1, shots)
    if len(sentences) >= shots:
        # Distribute sentences across shots as evenly as possible.
        out: list[str] = []
        per = len(sentences) / float(shots)
        for i in range(shots):
            lo = int(round(i * per))
            hi = int(round((i + 1) * per)) or lo + 1
            out.append(" ".join(sentences[lo:hi]) or sentences[min(lo, len(sentences) - 1)])
        return out
    # Fewer sentences than shots: cycle them so each shot has a subject.
    return [sentences[i % len(sentences)] for i in range(shots)]


def build_storyboard(brief: str, preset: Preset, shots: int | None = None,
                     *, screenplay: bool = False) -> list[dict[str, Any]]:
    """Build a structured storyboard for the piece.

    Returns a list of shot specifications with beats, full prompts, durations,
    and frame rates — enabling UI preview & editing before generation.

    ``screenplay=True`` reads the brief as Fountain and takes one beat per
    SCENE instead of per sentence. That is the difference between the
    writer deciding where a shot begins and punctuation deciding — and
    dialogue is routed to the voice rather than into the picture prompt,
    where spoken words render as subtitles and mouth artefacts.
    """
    count = int(shots or preset.default_shots)
    if screenplay:
        from clipforge.screenplay import synopsis, to_beats  # noqa: PLC0415

        beats = to_beats(brief, count)
        if beats:
            # Same leak as the router's: the brief is appended to every
            # shot prompt as context, and in screenplay mode the brief is
            # the script - dialogue included.
            brief = synopsis(brief)
        if not beats:
            # An empty parse means the text had no usable blocks; falling
            # back is better than emitting a storyboard of blank shots,
            # and it is said out loud rather than silently substituted.
            log.info("storyboard.screenplay_empty",
                     note="no scenes parsed; falling back to sentence beats")
            beats = split_into_beats(brief, count)
    else:
        beats = split_into_beats(brief, count)
    storyboard: list[dict[str, Any]] = []

    for i in range(count):
        prompt = build_shot_prompt(brief, preset, shot_index=i, total_shots=count, beat=beats[i])
        storyboard.append({
            "shot_index": i,
            "beat": beats[i],
            "prompt": prompt,
            "seconds": preset.shot_seconds,
            "fps": preset.fps,
            "preset": preset.name,
        })
    return storyboard


def beat_conflicts(beat: str, avoid: str) -> list[str]:
    """Phrases the BEAT asks for that the preset's avoid list forbids.

    A beat is prepended to the shot prompt and lands FIRST, so it wins.
    That makes a stale screenplay stronger than every correction in the
    niche, silently.

    MEASURED 2026-09-05, and it cost four rounds. `gen_avoid` had "a
    woven mat", "a metal gate" and "a seated child" in it while the
    script's own beat said "a Somali toddler in a bright patterned shirt
    SITS ALONE ON A WOVEN MAT ... exterior, COURTYARD GATE". Every render
    came back with a seated child on a mat at a gate wearing the West
    African print the channel spec had just ruled out, and the prompt
    work looked like it was being ignored when it was being contradicted.

    The division this restores: a BEAT says what HAPPENS, a niche says
    what it LOOKS LIKE. When a script starts describing wardrobe and set
    dressing it is competing with the niche, and the older of the two
    usually wins for no better reason than word order.

    Matching is on whole phrases, lowercased, longest first, and only for
    avoid entries of two words or more -- single words like "crowd" or
    "text" collide with ordinary prose and would cry wolf on every beat.
    """
    beat_l = (beat or "").lower()
    hits: list[str] = []
    for phrase in (p.strip().lower() for p in (avoid or "").split(",")):
        if len(phrase.split()) < 2:
            continue
        # "a woven mat" should match "on a woven mat"; drop a leading
        # article so the phrase matches the way a writer would type it.
        needle = phrase
        for article in ("a ", "an ", "the "):
            if needle.startswith(article):
                needle = needle[len(article):]
                break
        if needle and needle in beat_l:
            hits.append(phrase)
    return sorted(set(hits), key=len, reverse=True)
