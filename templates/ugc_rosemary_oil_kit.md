# UGC Kit — Rosemary & Peppermint Scalp Oil

Production kit for one 30-second TikTok Shop video built on a single
synthetic creator. The script is `examples/rosemary_oil_ugc.fountain`; the
look is the `tiktok_shop_ugc` niche in `clipforge/niches.py`. This file is
what sits between them: the prompts that make the creator, the voice that
speaks her, the composite that makes the product real, and the commands
that finish the piece.

| Input | Locked value |
|---|---|
| Product / category | Rosemary & Peppermint Scalp Oil — haircare |
| Angle | Dry, flaky scalp; shedding at the part line. First-person observation only |
| Persona | `MAYA-V1` — 26, olive complexion, dark brown eyes, messy bun, minimalist streetwear |
| Driving format | Bathroom-mirror GRWM, handheld vertical, window key camera-left |
| Niche | `tiktok_shop_ugc` |
| Delivery | 1080 × 1920, 30 fps, −14 LUFS integrated |

> **Claim posture.** Hair growth and hair loss claims are drug claims on a
> cosmetic listing. The script carries no timeline promise, no count and
> no clinical language — see §5 for the swap table and the boneyard note
> at the top of the Fountain file for why.

---

## 1. Dual-anchor character generation

Two reference images, generated before anything else and reused for every
clip. Anchor 1 fixes body, wardrobe and light; Anchor 2 fixes the face at
a detail level a video model can hold onto.

**Routing.** The README documents model selection as automatic and made on
one distinction — *"photoreal and human-subject work routes to Wan,
atmospheric and fast work to LTX"* — so both prompts below open with a
photoreal human subject rather than with mood. `tiktok_shop_ugc`'s
`gen_style` opens the same way and for the same reason. Where a generation
command exposes `--model`, Wan is the explicit choice for both anchors;
LTX does not do faces well enough to anchor an identity.

| Rule | Why |
|---|---|
| Fix one seed, record it here | A re-roll that drifts is not an anchor |
| Anchor 1 **1024 × 1536**, Anchor 2 **1024 × 1024** | Portrait for proportion, square for macro without engine crop |
| Generate 6–8, accept **one** | Identity is the scarcest asset in this pipeline |
| Anchor 2 generated **from** Anchor 1 at 0.35–0.45 denoise | Two independent text prompts produce two different people |

### Anchor 1 — medium GRWM frame (1024 × 1536)

```text
RAW photo, candid vertical portrait of a 26-year-old woman standing in a
small sunlit bathroom, medium-full shot from mid-thigh up, shot on a 35mm
lens at f/1.8, photographed not posed.

IDENTITY — BIOLOGICAL INVARIANTS (these never change across any image):
26 years old. Olive complexion, Fitzpatrick IV, warm neutral undertone,
visible natural skin unevenness across the cheeks. Dark brown eyes, deep
#3B2A1E, distinct limbal ring, slight downturn at the outer corners. Oval
face, soft high cheekbones, narrow nasal bridge with a small dorsal bump,
jaw very slightly fuller on the left. Full lower lip, thinner upper lip,
the right corner lifting first when she smiles.
MICRO-FEATURES: seven to nine faint freckles scattered across the nose
bridge and upper cheeks; one 2mm dark beauty mark 1cm below the outer
corner of the LEFT eye; a 4mm pale scar through the tail of the right
eyebrow. Thick natural unshaped eyebrows, the LEFT sitting about 1mm
higher than the right. No makeup beyond tinted balm; bare skin, real pores.

HAIR: dark brown, 2B wave, shoulder-blade length, worn in a loose messy
bun secured with a matte claw clip, two face-framing strands loose, honest
flyaways at the crown, roots slightly oily and a little flat. Centre part,
scalp faintly visible along the part line.

WARDROBE + FABRIC: oversized washed-black cotton crewneck tee, heavy
240gsm jersey, visible slub texture and soft vertical creasing at the
waist where it has been pushed up; sleeves pushed to the elbow.
High-waisted straight grey sweatpants, brushed fleece nap catching the
light. Barefoot on tile. Small gold huggie hoops, 10mm, both ears.

POSTURE: weight on the left hip, right shoulder dropped, chin very
slightly down and turned 10 degrees off-axis, one hand resting on the
counter edge. Relaxed, unposed, mid-thought.

ENVIRONMENT + LIGHT: cramped modern bathroom, white subway tile, matte
black tap, folded towel. Key light is soft north-facing window light from
camera LEFT at roughly 45 degrees, falling off across the right side of
her face; a dim warm bulb above the mirror fills the shadow side at about
a quarter stop. Visible soft shadow under the jaw.

TECHNICAL: RAW photo quality, natural film grain, true-to-life skin tone,
slight lens vignetting, mild sensor noise in the shadows, shallow depth of
field with the tile falling gently out of focus, imperfect handheld
framing, neutral white balance around 5200K. Photojournalistic,
unretouched.
```

### Anchor 2 — macro scalp and hairline texture (1024 × 1024)

Generated **from Anchor 1**. This is the anchor the demo beat leans on:
the product is applied to a scalp, and a scalp that reads as plastic sells
nothing.

```text
RAW macro close-up of the SAME 26-year-old woman as the reference image,
top of the head and hairline, centre part held open by two fingers, shot
on a 50mm lens at f/2.0, natural window light from camera LEFT.

CARRY OVER EXACTLY FROM REFERENCE: olive Fitzpatrick IV complexion with
warm neutral undertone; dark brown 2B hair, shoulder-blade length, in a
loose messy bun with two loose face-framing strands; the 2mm beauty mark
1cm below the outer corner of the LEFT eye where the face enters frame;
small gold huggie hoop, 10mm.

MACRO DETAIL EMPHASIS: individual hair strands with visible variation in
thickness and direction; scalp skin along the part with real texture —
visible pores, faint natural redness, fine flaking at the hairline; short
new growth standing up along the part; the hair shaft catching a specular
highlight where the window hits it; fingertips with natural nail texture
and visible knuckle creases holding the part open; fine vellus hair at the
temple.

TECHNICAL: RAW photo quality, natural grain, no beauty filter, no skin
smoothing, true-to-life colour, shallow depth of field falling off at the
crown, handheld micro-blur, 5200K neutral white balance.
```

### Negative prompt — both anchors and every clip

This is `tiktok_shop_ugc.gen_avoid`, and it is the same string the niche
carries so the two cannot drift apart:

```text
warped label text, garbled packaging text, unreadable product label,
deformed hands, malformed fingers, extra fingers, fused fingers, plastic
skin, waxy skin, airbrushed skin, poreless, beauty filter, skin smoothing,
oversaturated skin, orange skin, doll face, uncanny valley, dead eyes,
identity drift, face melt, morphing, temporal flicker, background
shifting, cartoon, anime, 3d render, CGI, studio lighting, ring light,
professional model, stock photo, fashion editorial, subtitles, captions,
text overlays, watermark, logo
```

### Acceptance gate

- [ ] Beauty mark under the **left** eye in both (engines mirror — reject if it swapped)
- [ ] Brow asymmetry visible, same direction, both images
- [ ] Gold huggie present and the same size in both
- [ ] Pores and individual hair strands visible at 100% zoom
- [ ] Hands, where visible, five fingers with correct joints
- [ ] Catchlight agrees with the stated camera-left key
- [ ] Seed recorded at the top of this file

---

## 2. Voice profile

| Parameter | Value | Note |
|---|---|---|
| Voice type | Young adult American female, 22–28, conversational and slightly breathy. **Not** narration, not "warm professional" | A polished read is the loudest AI tell on this platform |
| Model | Highest-quality multilingual / v2-class model available — **not** the low-latency turbo tier | Turbo flattens prosody and there is no latency requirement here |
| **Stability** | **42%** | Natural pitch drift without mid-word wobble; below 35% it slurs the product name |
| **Similarity / clarity** | **78%** | Above 85% it imports breath artefacts from the source sample |
| **Style exaggeration** | **18%** | Enough lift on "literally" and "way emptier" |
| Speaker boost | On | |
| Delivery | Upward inflection on the 0:00 line. Audible breath before "It's rosemary and peppermint". A smile in the voice on the last sentence | Generate 3 takes, keep the best 0:00–0:03 |
| Music bed | 92–98 BPM soft house / lo-fi, no vocal, no drop | Matched to the 156 wpm read so the cut does not feel rushed |
| Levels | VO −14 LUFS integrated, true peak ≤ −1 dBTP. Bed −24 LUFS under VO, −19 in the 0:00–0:03 gap and the final hold | S6 already normalises to the configured loudness; the bed is mixed before it gets there |
| SFX | Dropper click at 0:01, hair rustle at 0:09. Nothing else | Two sounds read as real, five read as sound design |

The pipeline's own `bta voiceover` speaks a script over a clip with Kokoro
or flite (§4). That is the local, no-credits path and it is a different
voice from the one specified above — use it for scratch timing, or when
the piece is not going out under a performed read.

---

## 3. Post-production composite

### 3.1 Label masking — which seconds must carry real product

Generated video cannot hold legible label text; the niche's negative
prompt fights it and does not win. Any frame where the label is wider than
about 8% of frame **and** would be readable must be real footage.

| Window | Defect | Fix |
|---|---|---|
| 0:00–0:03 | none — already a real plate | Shoot or license the dropper macro. It establishes the true label for the whole video, so the later glimpses read as the same bottle |
| 0:13.4–0:15.8 | Bottle held to lens; letterforms wander | Mask the label face, planar-track the plane, corner-pin a real product still. Matched grain, matched specular |
| 0:19.0–0:20.6 | Dropper at the scalp, angled, partly occluded by fingers | Tracked still feathered 4px — or cut to a 0.4s real insert, which is cheaper and cleaner if the hand crosses the label |
| 0:25.6–0:29.0 | Hero hold at the CTA | Do not fix; **replace** with a real locked-off plate. Match 5200K and the camera-left falloff |

*The avatar sells, the real product closes.* Any frame a buyer might
screenshot should be real.

### 3.2 Grain and the de-plastic pass

Order matters:

1. Assemble to the VO before lip-syncing, so only surviving frames get synced.
2. Lip-sync: mouth crop 384–512px, padding `[0, 12, 0, 0]` — the bottom pad
   keeps the chin from clipping on the 0:27 smile. Re-composite with a 6px
   feathered blend at the jaw, not a hard rectangle.
3. Colour-match the synced patch to the surrounding cheek; it comes back
   about half a stop bright and slightly desaturated.
4. De-plastic: 0.3px Gaussian, then unsharp at radius 1.2 / amount 0.4.
5. **Grain last: 1.5% monochrome neutral noise**, animated per frame, over
   generated and real footage alike so both share one noise floor. Below
   1.2% the waxy look survives; above 2.5% the platform's encoder turns it
   into blocking.

Grain goes on **after** the render, not in the niche's `grade`. The niche
grades the picture before captions are burned and before S6 encodes;
grain applied there is something the encoder then spends bitrate
smoothing back out. As a standalone pass over the delivered file:

```bash
ffmpeg -i clip.mp4 -vf "noise=alls=4:allf=t+u" -c:a copy clip.grain.mp4
```

`alls=4` is roughly the 1.5% band at this frame size. `allf=t+u` is
temporal and uniform — per frame, not a static overlay, which is the
difference between grain and a texture layer.

### 3.3 Safe zones

Working numbers at 1080 × 1920; verify on a device, since the UI shifts by
app version and region.

| Region | Keep clear |
|---|---|
| Top | `y < 220` |
| Right rail | `x > 800`, `y 780–1650` |
| Bottom, incl. the product anchor | `y > 1400` |

Effective text box `x 60 → 800`, `y 260 → 1380`.

This is the same constraint `_SHOP_POP.margin_v = 520` expresses in the
renderer (1920 − 1400, measured up from the bottom as ASS does). If you
change one, change the other, or burned-in captions land under the cart.
The house `_VIRAL_POP` sits at 260 and would do exactly that.

Note the 20px difference between the two numbers: `margin_v = 520` puts
the caption's lower edge at exactly y = 1400, flush against the band,
while the `y 260 → 1380` box above carries a cushion for manually placed
overlays. To hold the captions to the same cushion, override for the run:

```bash
.\.venv\Scripts\bta.exe process "D:\ugc\rosemary_cut.mp4" --niche tiktok_shop_ugc --margin-v 540
```

`--margin-v` is the only way to move them without editing the niche:
caption params are built from config and then updated from the niche, so
a selected niche wins over `config.toml` by construction. Precedence is
flag > niche > config.

---

## 4. Running it through the pipeline

Verified against the CLI in this tree (`clipforge/cli.py`), not against
the README.

**Render the assembled cut with the niche's look:**

```bash
.\.venv\Scripts\bta.exe process "D:\ugc\rosemary_cut.mp4" --niche tiktok_shop_ugc --clips 1
```

That applies the caption style, the grade, the speech cleanup and — since
the niche's pacing now reaches the clipper — its jump-cut setting. To
override the pacing for one run, the explicit flag still wins:

```bash
.\.venv\Scripts\bta.exe process "D:\ugc\rosemary_cut.mp4" --niche tiktok_shop_ugc --no-jumpcut
```

**Scratch voiceover** (local Kokoro/flite, not the §2 voice):

```bash
.\.venv\Scripts\bta.exe voiceover --clip rosemary_cut.mp4 --script-file "D:\ugc\vo.txt" --duck-db -12
```

`--script-file` rather than `--script`: the script has apostrophes and an
arrow in it, and a shell will mangle both.

**Deliver larger, if the source supports it:**

```bash
.\.venv\Scripts\bta.exe upscale --clip rosemary_cut.mp4 --height 2560
```

Resampling, not learned super-resolution — it will not invent label detail
the source lacks, which is the §3.1 problem and not something this solves.

**Prepare a draft post** (draft-only by design; it never presses Publish):

```bash
.\.venv\Scripts\bta.exe post --clip rosemary_cut.mp4 --platform tiktok --title "part-line thing" --caption "6 weeks in. #AIgenerated"
```

**Machine-readable result for automation:**

```bash
.\.venv\Scripts\bta.exe process "D:\ugc\rosemary_cut.mp4" --niche tiktok_shop_ugc --manifest "D:\ugc\result.json"
```

### What is not in this tree

The README documents `bta generate`, `bta swarm serve|plan|status|niches`
and a Fountain-driven screenplay mode. **None of those commands exist in
this checkout.** The generation half was removed along with the Fountain
parser and `/api/screenplay` — `clipforge/dashboard_live.html` says so at
the point where the composer used to be, and `clipforge/cli.py` registers
no `generate`, no `swarm` and no `niches` command.

So `examples/rosemary_oil_ugc.fountain` is a production document that
people read, exactly like `examples/geel_suuq.fountain` beside it. Nothing
in `clipforge/` parses either one today. The generation commands would
have looked like this, and are written down for whenever that half returns
rather than presented as things that run:

```bash
# NOT AVAILABLE IN THIS TREE — the generation half was removed.
.\.venv\Scripts\bta.exe generate --script examples/rosemary_oil_ugc.fountain --preset tiktok_shop_ugc --shots 12 --model wan
.\.venv\Scripts\bta.exe swarm plan --brief "rosemary oil UGC" --niche tiktok_shop_ugc
```

The niche itself is fully live regardless: it is data, the clipper reads
it, and `--niche tiktok_shop_ugc` works today.

---

## 5. Disclosure and claim compliance

1. Post editor → **More options** → **Content disclosure** → turn on
   **AI-generated content**.
2. Preserve C2PA metadata through export — do not strip it.
3. `#AIgenerated` in the caption. The export pack assembles the caption
   next to the clip; the toggle is still a human step, which is consistent
   with this pipeline producing files and stopping.
4. The creator is synthetic and must not resemble a real, identifiable
   person. If a generated face starts looking like someone, regenerate it.

| Do not say | Say instead |
|---|---|
| "Stops shedding in 14 days" | "My brush is way emptier" |
| "Regrows your hairline" | "My part looks less wide to me" |
| "Clinically proven" | *(omit unless the listing holds the study)* |
| "Cures dandruff / treats scalp disease" | "My scalp isn't flaking like it was" |
| "Works for everyone" | "This is just what happened for me" |

### Final QA

- [ ] 0.25× scrub — no identity drift, finger errors, or label warp outside the patched windows
- [ ] Blink rate roughly one per 3–5s; generated clips under-blink
- [ ] −14 LUFS integrated, ≤ −1 dBTP
- [ ] Every burned-in caption inside `x 60–800`, `y 260–1380`
- [ ] Cart area unobstructed for the full 30s
- [ ] AIGC toggle on, C2PA intact, hashtag present
- [ ] No growth, timeline or clinical claim in the VO
- [ ] All four label-legible windows are real footage
