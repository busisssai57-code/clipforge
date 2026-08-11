"""Niches — a whole look, selectable by name.

A preset says how to GENERATE a shot. A niche says what the finished piece
should BE: the visual style, the colour grade, how captions look and where
they sit, the pacing, and the shot length. Selecting one should mean the
operator never touches another control.

This exists because the pipeline's defaults encode exactly one aesthetic —
loud viral captions, bright saturated footage, fast cuts — and that is
actively wrong for most formats. The reference piece that prompted this
module is the opposite of the house style in every respect: monochrome,
slow, and captioned in small quiet type. Rendering it with the defaults
would produce something that looks nothing like the channel.

Each niche is data, not code. Adding one is a dataclass literal.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CaptionStyle:
    """How burned-in text looks. Maps onto S5's ASS styling."""

    #: Font family. Must exist on the machine or libass silently substitutes.
    font: str
    #: Point size at 1080x1920. S5 scales from here.
    size: int
    #: &HBBGGRR& — ASS is BGR, not RGB. Getting this backwards is the
    #: classic libass mistake and it fails silently (blue reads as red).
    primary: str
    outline_colour: str
    outline: float
    shadow: float
    #: ASS numpad alignment. 2 = bottom-centre, 5 = middle-centre.
    alignment: int
    margin_v: int
    uppercase: bool
    #: "pop" = per-word karaoke spring. "quiet" = whole line, static, fades.
    animation: str
    #: Words per caption line. Small numbers read as poetry, large as prose.
    max_words: int


@dataclass(frozen=True)
class Niche:
    name: str
    label: str
    summary: str
    #: Appended to generation prompts.
    gen_style: str
    gen_avoid: str
    shot_seconds: float
    fps: int
    default_shots: int
    aspect: str
    #: ffmpeg filter chain applied to the picture at render time. Empty
    #: string = no grade.
    grade: str
    caption: CaptionStyle
    #: Jump-cut silence removal. Wrong for anything with musical timing.
    jumpcut: bool
    #: Speech cleanup mode passed to S6.
    enhance_speech: str
    keywords: tuple[str, ...] = field(default_factory=tuple)


#: Small, quiet, centred type. The reference piece sets its line in modest
#: white sans mid-frame with no outline — the restraint IS the aesthetic.
#: Loud karaoke here would read as a completely different channel.
_QUIET_CENTRE = CaptionStyle(
    font="Helvetica Neue", size=52, primary="&H00FFFFFF",
    outline_colour="&H00000000", outline=0.0, shadow=0.6,
    alignment=5, margin_v=0, uppercase=False, animation="quiet",
    max_words=7,
)

#: The house style everything else used: big, bold, bottom-third, per-word
#: pop with a heavy outline so it survives any background.
_VIRAL_POP = CaptionStyle(
    font="Arial Black", size=96, primary="&H0000FFFF",
    outline_colour="&H00000000", outline=4.0, shadow=1.0,
    alignment=2, margin_v=260, uppercase=True, animation="pop",
    max_words=4,
)

#: Monochrome grade matching the reference: full desaturation, crushed
#: blacks, blown highlights, vignette, and a light grain so the flat areas
#: do not band. Letterbox bars are part of the look, not an artefact.
_MONO_GRADE = (
    "hue=s=0,"
    "curves=all='0/0 0.25/0.10 0.6/0.72 1/1',"
    "eq=contrast=1.28:brightness=-0.03,"
    "vignette=PI/4.2,"
    "noise=alls=6:allf=t+u,"
    "pad=iw:ih:0:0:color=black"
)

ASCENDRO_MIND = Niche(
    name="dark_mindset",
    label="Dark Mindset",
    summary=("Monochrome, slow, aphoristic. A lone figure in a vast, misty "
             "landscape with one quiet line of text. Stoic/self-improvement "
             "short-form."),
    gen_style=(
        "high contrast black and white cinematography, lone distant "
        "silhouette walking away from camera, vast empty misty landscape, "
        "long exposure motion blur, heavy atmospheric fog, crushed blacks "
        "and blown highlights, grainy 35mm monochrome film stock, slow "
        "contemplative drift, deep negative space, melancholy but resolute, "
        "no faces visible, wide establishing scale"),
    gen_avoid=(
        "colour, saturated, bright daylight, cheerful, crowds, close-up "
        "face, text, watermark, logo, cartoon, 3d render, cgi, lens flare, "
        "fast motion, shaky"),
    shot_seconds=6.0,
    fps=30,
    default_shots=2,
    aspect="9:16",
    grade=_MONO_GRADE,
    caption=_QUIET_CENTRE,
    jumpcut=False,          # narration is paced; cutting its pauses kills it
    enhance_speech="gentle",
    keywords=("stoic", "mindset", "discipline", "solitude", "motivation"),
)

VIRAL_CLIPS = Niche(
    name="viral_clips",
    label="Viral Clips",
    summary="Loud, fast, bottom-third karaoke captions. Podcast/stream cuts.",
    gen_style=(
        "vivid high-energy footage, punchy saturated colour, crisp detail, "
        "dynamic camera movement, bright key light"),
    gen_avoid="dull, washed out, static, text, watermark, logo",
    shot_seconds=5.0,
    fps=30,
    default_shots=6,
    aspect="9:16",
    grade="eq=saturation=1.12:contrast=1.06",
    caption=_VIRAL_POP,
    jumpcut=True,
    enhance_speech="gentle",
    keywords=("podcast", "stream", "reaction", "interview"),
)

CINEMATIC_DOC = Niche(
    name="cinematic_doc",
    label="Cinematic Doc",
    summary="Filmic colour, restrained captions, patient cutting.",
    gen_style=(
        "observational documentary cinematography, 35mm anamorphic, natural "
        "available light, shallow depth of field, muted filmic grade with "
        "lifted blacks, patient slow push-in, photojournalistic framing"),
    gen_avoid="cartoon, cgi, video game, text, watermark, oversaturated",
    shot_seconds=6.0,
    fps=24,
    default_shots=6,
    aspect="9:16",
    grade="curves=all='0/0.04 0.5/0.5 1/0.96',eq=saturation=0.88:contrast=1.05",
    caption=CaptionStyle(
        font="Helvetica Neue", size=64, primary="&H00FFFFFF",
        outline_colour="&H00000000", outline=1.2, shadow=0.8,
        alignment=2, margin_v=200, uppercase=False, animation="quiet",
        max_words=6),
    jumpcut=False,
    enhance_speech="gentle",
    keywords=("documentary", "story", "essay"),
)

NICHES: dict[str, Niche] = {
    n.name: n for n in (ASCENDRO_MIND, VIRAL_CLIPS, CINEMATIC_DOC)
}


def get_niche(name: str) -> Niche:
    try:
        return NICHES[name]
    except KeyError:
        raise ValueError(
            f"unknown niche {name!r}; available: {', '.join(sorted(NICHES))}"
        ) from None


def niche_as_preset(niche: Niche):
    """A niche's generation half, as a ``genvideo.presets.Preset``.

    Generation (the router, the providers) only knows about Preset — it
    predates niches and has no reason to import this module. Rather than
    teach it a second vocabulary, adapt a niche INTO the shape it already
    understands. A niche is a strict superset of what generation needs
    (it additionally carries the grade and caption style, which apply
    after generation, not during it).
    """
    from clipforge.genvideo.presets import Preset

    return Preset(
        name=niche.name, summary=niche.summary, style=niche.gen_style,
        avoid=niche.gen_avoid, shot_seconds=niche.shot_seconds,
        fps=niche.fps, narrated=True, default_shots=niche.default_shots,
        cut_style="medium", keywords=niche.keywords,
    )


def resolve_preset(name: str):
    """A niche or a base creative preset, by one name.

    `bta generate --preset X` and the dashboard's preset field both take
    one string, and a niche IS a preset for generation purposes — so a
    niche name must resolve here too, not just in `get_niche`. Checked
    first: niches are the more specific, more common case now, and if a
    name ever collided the niche's fuller prompt should win.
    """
    from clipforge.genvideo.presets import PRESETS, get_preset

    if name in NICHES:
        return niche_as_preset(NICHES[name])
    try:
        return get_preset(name)
    except ValueError:
        available = sorted(set(NICHES) | set(PRESETS))
        raise ValueError(
            f"unknown preset or niche {name!r}; available: "
            f"{', '.join(available)}") from None


def niche_s5_params(niche: Niche) -> dict[str, object]:
    """Caption settings in the shape S5 expects."""
    c = niche.caption
    return {
        "font": c.font, "font_size": c.size,
        "highlight_color": c.primary, "base_color": c.primary,
        "outline_color": c.outline_colour, "outline": c.outline,
        "shadow": c.shadow, "alignment": c.alignment,
        "margin_v": c.margin_v, "uppercase": c.uppercase,
        "animation": c.animation, "max_words": c.max_words,
    }


def niche_summary() -> list[dict[str, object]]:
    """Dashboard-shaped list of every niche."""
    return [
        {
            "name": n.name,
            "label": n.label,
            "summary": n.summary,
            "aspect": n.aspect,
            "shots": n.default_shots,
            "shot_seconds": n.shot_seconds,
            "fps": n.fps,
            "captions": n.caption.animation,
            "graded": bool(n.grade),
            "jumpcut": n.jumpcut,
            "keywords": list(n.keywords),
        }
        for n in sorted(NICHES.values(), key=lambda x: x.label)
    ]
