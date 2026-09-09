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
    #: Whether this look wants the post layer at all — hook card, emoji
    #: punchline stickers, handle. Off for every existing niche, because
    #: stamping a hook card on a documentary would be a change of format,
    #: not of style.
    post_layer: bool = False
    #: How long a hook card holds, when there is one.
    hook_seconds: float = 2.0
    #: Stamp punchline emoji at all. Off for ari_goat on operator
    #: instruction ("no more emojis") and consistent with the channel's
    #: own spec, whose negative block bans on-screen text overlays and
    #: whose design calls for burned Somali TEXT -- a title and the
    #: punchline -- not emoji decoration.
    stickers: bool = True
    #: Whether the generation model's own audio belongs in the delivered
    #: piece. Off for ari_goat: the channel spec calls native model audio
    #: "serviceable for ambient/foley but... a SCRATCH LAYER", with the
    #: final mix built in post from a dhaanto bed, goat foley and the
    #: child's laugh at designed levels.
    #:
    #: MEASURED on a delivered piece, which is what the operator actually
    #: heard: shot 0 at -84 dBFS of low-level noise (entropy 0.06 -- not
    #: silence, hiss), shot 1 the same, and shot 2 at -27 with a -2.1
    #: peak of laughter arriving from nowhere. White noise, then a laugh
    #: with no cause. Keeping it is worse than having none, because a bed
    #: can be laid under silence and cannot be laid under hiss.
    model_audio: bool = True
    #: Condition the LAST frame of every shot as well as the first, at
    #: this strength. 0.0 is off.
    #:
    #: An anchor pins frame ZERO and nothing holds the shot after it.
    #: MEASURED on a 1.9s anchored render: the framing pushed in far
    #: enough to crop the goat's horns off the top, the goat's neck
    #: stretched, and the child's hand became a single elongated stick
    #: finger. Pinning both ends stops the drift and is also the channel
    #: spec's frames-to-video loop -- "the last frame flows back into the
    #: cold-open first frame; a seamless loop drives rewatches".
    #:
    #: Below 1.0 deliberately: at 1.0 the end IS the start and the beat
    #: has nowhere to move.
    loop_strength: float = 0.0
    #: Chain shots from the previous shot's last frame. MEASURED need,
    #: 2026-09-05: a five-shot ari_goat batch came back with the child in
    #: a DIFFERENT shirt and a different courtyard in every shot -- five
    #: good shots that cannot be cut into one afternoon. `continuity` was
    #: fully implemented in `generate_sequence`, supported by the ltx25
    #: provider (`SubprocessModelProvider.generate` declares
    #: `start_image`), and passed by nobody: `grep -rn "continuity="`
    #: outside router.py returned nothing, so it was always False. Same
    #: dead-parameter shape as `--model`, `quantize` and `holdout`.
    continuity: bool = False


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

#: Big bold caps, high on the frame. Only used when a sketch is subtitled
#: for silent viewing — the format's own text lives in the hook card, and
#: burned dialogue would fight it.
_SKETCH_CAPS = CaptionStyle(
    font="Arial Black", size=72, primary="&H00FFFFFF",
    outline_colour="&H00000000", outline=5.0, shadow=1.2,
    alignment=8, margin_v=190, uppercase=True, animation="pop",
    max_words=5,
)

GEEL_SKETCH = Niche(
    name="geel_sketch",
    label="Geel Sketch",
    summary=("Somali animal sketch comedy: photoreal camels and llamas "
             "living human lives, fast cuts, a hook question on frame one "
             "and emoji stamped on the punchlines."),
    gen_style=(
        "photorealistic anthropomorphic camel and llama characters with "
        "expressive human-like faces and gestures, Somali setting - open "
        "market stalls, acacia scrub, small shop interiors, hand-woven "
        "beadwork and bright patterned cloth, strong warm East African "
        "midday sun, saturated colour, crisp close-up portrait framing at "
        "eye level, one clear comic action per shot, subtle handheld "
        "weight"),
    gen_avoid=(
        "cartoon, flat illustration, plush toy, cute stylised big-eye "
        "character, text, subtitles, watermark, logo, dull grey light, "
        "wide empty establishing shot with no character, motion blur "
        "smear, melted faces, extra limbs"),
    #: 1.9s is the reference piece's MEDIAN shot, measured off 35 cuts in
    #: 65.1s. It is the load-bearing number of the format: this comedy
    #: works by cutting on every reaction, and the same shots at 5s read
    #: as an animation reel.
    shot_seconds=1.9,
    fps=30,
    #: The reference runs 35 shots. This defaults lower because each shot
    #: is a generation and an operator who types no number should not
    #: start a forty-minute local render by accident; --shots 35 is the
    #: faithful length and the rhythm is what the preset actually fixes.
    default_shots=12,
    #: 3:4, the reference's own frame. TikTok pillarboxes it and the
    #: extra height buys room for the hook card above the faces.
    aspect="3:4",
    grade="eq=saturation=1.16:contrast=1.07,unsharp=5:5:0.5",
    caption=_SKETCH_CAPS,
    #: A sketch is dialogue with comic pauses. Cutting the silences out
    #: removes the timing the joke is built on.
    jumpcut=False,
    enhance_speech="gentle",
    keywords=("comedy", "sketch", "character", "photoreal", "animal",
              "human", "detail"),
    post_layer=True,
)

#: The Ari Channel's own preset. Same cutting rhythm as GEEL_SKETCH --
#: that number is the format, not the subject -- but the subject is a real
#: child and a real animal, so nothing here may drift anthropomorphic.
#: Kept as a separate niche rather than a flag on GEEL_SKETCH because the
#: style and avoid lists are the whole difference and sharing them was
#: exactly how a camel-market render ended up on a toddler channel.
ARI_GOAT = Niche(
    name="ari_goat",
    label="Ari Soomaali",
    summary=("Somali toddler meets a goat: real child, real animal, fast "
             "cuts on every reaction, hook card on frame one and emoji "
             "stamped on the punchline."),
    #: "a single ... alone in frame" and the centring clause are load-
    #: bearing, not padding. The first render of this niche put TWO
    #: children in frame, both pushed to the edges with the centre empty
    #: -- and ari_bridge reserves 750-1050 for a face sitting dead
    #: centre, so an off-centre subject means the punchline block lands
    #: over nothing. Nothing in the old wording asked for one child or
    #: for the middle of the frame; "a Somali toddler" is not an
    #: instruction, it is a noun. Lower case throughout on purpose: an
    #: all-caps word here comes back drawn into the picture.
    #: The framing clause is SIZED, not stylistic. MEASURED on the first
    #: real render of this niche (shot_00, 2026-09-04): "tight framing"
    #: came back as a medium-wide shot with the child at roughly 40% of
    #: frame height. At the 704x1280 generation size and this model's
    #: 32-px latent grid, that leaves a hand about 1.4 latent cells
    #: across -- fingers cannot be represented in one and a half cells,
    #: and they were not: both arms came back as boneless tapered stumps
    #: and a six-fingered hand appeared in the lap. Naming the shot size
    #: explicitly roughly doubles the child's linear size and takes a
    #: hand to ~5 cells. There is no resolution lever left to pull
    #: instead: 704x1280 is 901,120 px against LTX-2.5's 921,600 budget.
    #: Toddler proportions are stated positively for the same reason the
    #: centring clause is -- "toddler" is a noun, not an instruction, and
    #: the model rendered a toddler head on an older child's legs.
    #:
    #: Three clauses were added after the FIRST framing test (2026-09-04,
    #: same seed 1234, same 30 steps at CFG 3.0, the prompt the only
    #: variable):
    #:
    #: * WARDROBE. Neither prompt had ever named clothing. The wide
    #:   version got a patterned shirt by luck; the close version came
    #:   back with the child wearing nothing, which is unpublishable on a
    #:   toddler channel. Luck is not a wardrobe department, so the shirt
    #:   and shorts are stated, and "a naked child" is in the avoid list.
    #: * COLOUR. Closing in did not just fix hands, it exposed the skin:
    #:   sunburnt orange, blotchy, with yellow-green patches on the
    #:   cheeks and red-rimmed eyes. The prompt was asking for "saturated
    #:   colour" AND "strong warm" sun, and then `grade` below multiplied
    #:   saturation by another 1.16 on top. That stacking is invisible on
    #:   a small distant subject and brutal on a face filling the frame,
    #:   so the prompt now asks for natural skin tones and the grade is
    #:   pulled back.
    #: * GOAT SCALE. Asking for the goat "close beside him" put a shaggy
    #:   mass across half the frame with its head drifting out of shot.
    #:
    #: The scale clause written for that third point then produced the
    #: WORST animal yet, and it is worth recording exactly why, because
    #: the mistake is one this file already warns about in another form.
    #: It read "a normal farm goat about the size of a large dog". A text
    #: encoder does not evaluate "about the size of" as a comparison --
    #: it sees the tokens `large dog`, and the render came back with a
    #: shaggy, long-snouted, hornless animal with a sloping back that the
    #: operator called a dog, because a dog is what the prompt asked for.
    #: Never name an animal you do not want in shot, not even as a ruler.
    #: The same clause carried "not crowding the child": these models do
    #: not honour negation in the POSITIVE prompt, so that reads as
    #: "crowding". Both belong in `gen_avoid`, which is a real negative
    #: prompt, and that is where they now are.
    #:
    #: So the goat is described positively and specifically instead --
    #: short-haired, smooth coat, small upright horns, standing behind --
    #: and the block was rewritten SHORTER (950 -> ~650 chars, near the
    #: 603 it started at), because length dilutes every clause.
    #:
    #: NOT for the audio, though -- that claim was made here and is
    #: WRONG, so it is corrected rather than quietly deleted. The short
    #: 647-char version measured -87.3 dBFS, the QUIETEST of the four
    #: renders, against -50.9 for the 603-char original. And the original
    #: run's own nine shots logged peaks of 0.76 / 0.23 / 0.23 / 0.065 /
    #: 0.039 alongside 0.00026 / 0.00036 / 0.00023 / 0.00032 -- silence
    #: about four times in nine with the ORIGINAL wording. LTX-2.5's
    #: audio is unreliable shot to shot; prompt length is not the lever,
    #: and nothing here should be shortened in the belief that it is.
    #:
    #: FIFTH round, against the channel's own production spec, which the
    #: operator supplied after calling the output trash. It named the
    #: fault precisely and it was not a rendering fault at all: the
    #: wardrobe was WEST AFRICAN. "bright patterned shirt" reliably
    #: produces an "Angelina" dashiki -- a Vlisco wax print designed in
    #: the Netherlands in 1962-63, a staple in Nigeria and Ghana and the
    #: American diaspora, and not Somali dress. For an audience that is
    #: 54.6% Somalia, 13.3% Ethiopia and 11.2% Kenya, that reads as
    #: "generically African", not "one of us", which on a channel whose
    #: whole moat is Horn-of-Africa specificity is a strategic leak
    #: rather than a pedantic one.
    #:
    #: So the wardrobe is now named in Somali terms: koofiyad (the
    #: embroidered cap), khamiis (plain cream tunic, gold thread at the
    #: collar only), macawiis (WOVEN indigo-and-white stripe, never
    #: printed petals). The spec's other corrections come with it --
    #: the goat is an indigenous Galla with ribbed horns and wattles
    #: rather than a generic brown one; the biome is acacia-commiphora
    #: savanna on ochre sand, because lush green broadleaf trees are a
    #: documented AI error that breaks it; the light is hard
    #: HIGH-ANGLE LATE-MORNING at ~70 degrees, not the late-afternoon
    #: warmth that was here; and the optics are stated (85mm, f2.2,
    #: camera at the child's head height) because the reference frames
    #: were reverse-engineered to them.
    #:
    #: FOURTH round, and the mistake was over-correction. "clean healthy
    #: baby skin in an even tone" removed the dirt of round three and
    #: took the child's ethnicity with it -- lighter skin, flatter nose
    #: bridge, a different face on a Somali channel -- because "even
    #: tone" is a smoothing instruction and "natural skin texture" had
    #: been dropped in the same edit. Skin is now anchored explicitly
    #: (deep brown Somali skin) with the texture clause restored, and
    #: the hair is named (short soft dark curly) after round three
    #: returned a wet-plastered scalp with a cracked crust on the crown.
    #: The goat moved from "behind him" to "behind his shoulder" to get
    #: its horns out of the 750-1050 band ari_bridge reserves for a face.
    gen_style=(
        "photorealistic documentary footage, a wide full-body two-shot "
        "of one Somali toddler standing barefoot on open sand beside "
        "one adult goat that stands taller than he does, the child "
        "looking up at the goat and the goat inclining its head down "
        "toward him, both animals and child complete in frame from head "
        "to feet, medium-brown skin with natural skin texture, "
        "texture, fine vellus hair and visible pores, short cropped "
        "dark hair, wearing a white embroidered koofiyad cap and a "
        "plain cream cotton khamiis tunic with fine gold-thread collar "
        "embroidery over a woven indigo-and-white striped macawiis "
        "wrap, barefoot, a soft rounded toddler body with a large head "
        "and short chubby limbs, one indigenous Somali Galla goat "
        "beside him, lean dairy frame, white coat with tan patches, "
        "long curved ribbed horns, pendulous ears, wattles under the "
        "jaw, open Somali acacia and commiphora savanna, ochre-tan "
        "compacted sand with scattered pebbles, pale sage desert scrub, "
        "flat-topped vachellia tortilis and myrrh scrub behind, hazy "
        "low hills, hard high-angle late-morning sun with short hard "
        "shadows and warm sand-bounce fill, light heat haze, 85mm lens "
        "at f2.2, shallow depth of field, camera low at the child head "
        "height looking slightly up so the goat towers, one steady "
        "composition, natural warm "
        "grade with Portra-like skin and protected highlights, filmic "
        "contrast, fine grain, one clear moment of reaction"),
    #: "anthropomorphic" and "clothing on the animal" are here and not in
    #: GEEL_SKETCH's list because for that niche they are the POINT. The
    #: rest of this list is the same failure set: a generator asked for a
    #: small child reaches for doll and waxy-skin territory unless told
    #: not to.
    #:
    #: This list is NOT load-bearing on its own and should not be trusted
    #: as though it were -- the framing above is what actually attacks
    #: the cause. shot_00 proved it: "extra limbs", "deformed hands" and
    #: "melted faces" were all already in this list and all three
    #: happened anyway. What changed here is the AXIS. The old list said
    #: "adult facial proportions on a child" and the face was the part
    #: that came out right; it was the body that came back adult, with
    #: shins about 1.8x head height where a toddler's are about 1x. The
    #: seated goat's neck roughly doubled in length across the 57 frames,
    #: so the stretch is named too.
    gen_avoid=(
        # The channel spec's own negative block, which is more thorough
        # than anything guessed here and carries two entries no amount of
        # staring at frames would have produced: "wax-print dashiki" and
        # "West African clothing" name the exact failure that made the
        # operator call the output trash, and "lush green broadleaf
        # trees" names a documented AI error that silently relocates the
        # piece out of the Horn of Africa.
        "plastic skin, waxy skin, airbrushed skin, doll face, porcelain "
        "skin, overly smooth skin, uncanny valley, dead eyes, "
        "cross-eyed, asymmetric eyes, deformed face, merged faces, "
        "morphing, identity drift, face melt, extra fingers, missing "
        "fingers, fused fingers, deformed hands, extra limbs, bad "
        "anatomy, warping, flicker, temporal inconsistency, background "
        "shifting, floating objects, duplicate goat, extra horns, "
        "subtitles, captions, text overlays, watermark, logo, garbled "
        "text, on-screen text, oversaturation, neon, HDR bloom, soap "
        "opera effect, cartoon, anime, 3d render, CGI, lush green "
        "broadleaf trees, wax-print dashiki, West African clothing, "
        # Kept from the measured rounds before the spec arrived: these
        # were each a real render on this box, not a precaution.
        "adult body proportions on a toddler, long adult legs on a "
        "small child, light skin, dirty skin, mud on the skin, bruises, "
        "misshapen head, flaky scalp, thinning hair, wet plastered "
        "hair, a naked child, bare chest, shirtless, dog, hyena, sheep, "
        "wolf, shaggy matted fur, long canine snout, hornless, two "
        "children, several children, a group of children, other people "
        "in the background, crowd, the child at the edge of the frame, "
        "empty centre of frame, the child's face turned away from the "
        "camera, back of the head, the framing changing part-way "
        "through the shot, "
        # The composition faults the operator identified by holding this
        # render against the reference that earned 428K views: a seated
        # child cropped at the lap with a goat's head floating behind
        # it, on a woven mat in front of a concrete gatepost. The
        # reference is a full-body two-shot on open sand where the goat
        # towers over a standing toddler -- the SIZE CONTRAST is the
        # joke, and a waist-up crop cannot tell it.
        "a seated child, a cropped or half-visible goat, a goat smaller "
        "than the child, a woven mat, a concrete wall, a metal gate, a "
        "fence, a courtyard, a close-up portrait"),
    #: Inherited deliberately from GEEL_SKETCH: 1.9s is the reference
    #: piece's median shot and the reason the format reads as comedy
    #: rather than as an animation reel.
    shot_seconds=1.9,
    fps=30,
    #: 33 x 1.9s = 62.7s of source, which is what ari_bridge's 62.5s
    #: MASTER_TARGET_DURATION is expecting. The default stays low for the
    #: same reason GEEL_SKETCH's does -- 33 shots is about six hours of
    #: local render and nobody should start that by omitting a flag.
    default_shots=12,
    #: 9:16 rather than GEEL_SKETCH's 3:4. ari_bridge's post_processor
    #: scale-to-covers then centre-crops to 1080x1920, so a 3:4 source
    #: loses roughly an eighth of its width off each side. Rendering the
    #: delivery frame natively costs nothing and crops nothing.
    aspect="9:16",
    #: Saturation 1.05, not GEEL_SKETCH's 1.16. That figure was set for a
    #: sketch shot wide, where the subject is small and the grade is what
    #: gives the frame its punch. This niche now fills the frame with a
    #: child's face, and 1.16 on top of a prompt already asking for warm
    #: low sun is what made the skin read as sunburnt. Contrast and the
    #: unsharp pass are unchanged.
    grade="eq=saturation=1.05:contrast=1.07,unsharp=5:5:0.5",
    caption=_SKETCH_CAPS,
    jumpcut=False,
    enhance_speech="gentle",
    #: One child, one goat, one afternoon -- so the shots must be one
    #: scene. See the field's own note for what five unchained shots
    #: looked like.
    continuity=True,
    stickers=False,
    model_audio=False,
    loop_strength=0.6,
    keywords=("comedy", "reaction", "child", "animal", "photoreal",
              "family", "detail"),
    post_layer=True,
)

NICHES: dict[str, Niche] = {
    n.name: n for n in (ASCENDRO_MIND, VIRAL_CLIPS, CINEMATIC_DOC,
                        GEEL_SKETCH, ARI_GOAT)
}


def get_niche(name: str) -> Niche:
    try:
        return NICHES[name]
    except KeyError:
        raise ValueError(
            f"unknown niche {name!r}; available: {', '.join(sorted(NICHES))}"
        ) from None




def niche_aspect(name: str) -> str | None:
    """The aspect a niche declares, or None if ``name`` is not a niche.

    `Niche.aspect` existed and reached nothing but a dashboard summary:
    generation read the config value, so a niche shot in 3:4 rendered
    9:16 and only the label said otherwise. Every niche before this one
    declared the config default, which is exactly why nobody saw it.
    """
    n = NICHES.get(name)
    return n.aspect if n else None






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
