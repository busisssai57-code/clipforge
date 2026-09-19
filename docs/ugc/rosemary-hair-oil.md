# UGC Kit — Rosemary & Peppermint Scalp Oil

Complete production kit for one 30-second TikTok Shop UGC video built on a
single AI creator. Everything below is copy-ready: the prompt blocks go
into the image and video tools verbatim, the script goes into the voice
tool verbatim, and the checklist is the edit.

| Input | Locked value |
|---|---|
| **Product / category** | Rosemary & Peppermint Scalp Oil — haircare, scalp treatment |
| **Selling angle / pain point** | Dry, flaky, itchy scalp; shedding concentrated at the part line. Personal-experience framing: *less hair in the brush, part looks less wide* |
| **Persona** | "MAYA-V1" — 26, female, olive complexion, dark brown eyes, messy bun, minimalist streetwear |
| **Driving video** | Bathroom-mirror GRWM: handheld vertical, arm's-length mirror selfie, natural window light, talks while applying, two small reframes |
| **Delivery** | 1080 × 1920, 30 fps, H.264, audio −14 LUFS integrated |

> **Claim posture.** Scalp oil sits next to a regulated claim class. Hair
> *growth* and hair *loss* claims are drug claims in most markets and are
> a TikTok Shop takedown risk on a cosmetic listing. This script is
> written entirely in first-person observation — what she saw in her own
> brush — with no timeline promise, no before/after count, and no
> "clinically proven". §4 has the phrase-level swap table. Do not
> reintroduce the "stops shedding in 14 days" framing into the VO without
> substantiation your listing can defend.

---

## 1. Dual-Anchor Character Generation

Two reference images, generated before anything else. Anchor 1 sets the
body, wardrobe and light. Anchor 2 sets the face at a detail level the
video engine can hold onto across clips. Every clip prompt in §3 refers
back to these two images — they are the identity, not the text.

**Generation discipline**

| Rule | Why |
|---|---|
| Fix one seed and record it at the top of the kit | Re-rolls that drift are unusable as anchors; you will need to regenerate a matching angle in three weeks |
| Anchor 1 at **1024 × 1536**, Anchor 2 at **1024 × 1024** | Portrait for body proportion, square for macro detail without engine crop |
| Generate 6–8 candidates, accept **one** | Identity is the scarcest asset in the pipeline; pick slowly |
| Anchor 2 must be generated **from** Anchor 1 (img2img / reference, 0.35–0.45 strength) | Two independent text prompts produce two different people |

### Anchor 1 — Full-body / medium foundation (1024 × 1536)

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
soft natural asymmetry when she smiles — the right corner lifts first.
MICRO-FEATURES: seven to nine faint freckles scattered across the nose
bridge and upper cheeks; one 2mm dark beauty mark 1cm below the outer
corner of the LEFT eye; a 4mm pale scar through the tail of the right
eyebrow. Thick natural unshaped eyebrows, the LEFT sitting about 1mm
higher than the right. No makeup beyond tinted balm; bare skin, real pores.

HAIR: dark brown, 2B wave, shoulder-blade length, worn in a loose messy
bun secured with a matte claw clip, two face-framing strands loose, honest
flyaways at the crown, roots slightly oily and a little flat. Centre part,
scalp faintly visible along the part line.

WARDROBE + FABRIC: oversized washed-black cotton crewneck tee, heavy 240gsm
jersey, visible slub texture and soft vertical creasing at the waist where
it has been pushed up; sleeves pushed to the elbow. High-waisted straight
grey sweatpants, brushed fleece nap catching the light. Barefoot on tile.
Small gold huggie hoops, 10mm, both ears. Thin gold ring, right index.

POSTURE: weight on the left hip, right shoulder dropped, chin very slightly
down and turned 10 degrees off-axis, one hand resting on the counter edge.
Relaxed, unposed, mid-thought.

ENVIRONMENT + LIGHT: cramped modern bathroom, white subway tile, matte
black tap, folded towel, a few real objects on the counter. Key light is
soft north-facing window light from camera LEFT at roughly 45 degrees,
falling off across the right side of her face; a dim warm bulb above the
mirror fills the shadow side at about a quarter stop. Visible soft shadow
under the jaw.

TECHNICAL: RAW photo quality, natural film grain, true-to-life skin tone,
slight lens vignetting, mild sensor noise in the shadows, shallow depth of
field with the tile falling gently out of focus, imperfect handheld
framing, neutral white balance around 5200K. Photojournalistic, unretouched.
```

### Anchor 2 — Macro close-up selfie (1024 × 1024)

Generated **from Anchor 1** as a reference image at 0.35–0.45 denoise so
the bone structure survives.

```text
RAW macro selfie close-up of the SAME 26-year-old woman as the reference
image, head and shoulders only, arm's-length phone distance, shot on a
50mm lens at f/2.0, natural window light from camera LEFT.

CARRY OVER EXACTLY FROM REFERENCE: olive Fitzpatrick IV complexion with
warm neutral undertone; dark brown #3B2A1E eyes with a distinct limbal
ring and slight outer downturn; oval face, soft high cheekbones, narrow
nasal bridge with a small dorsal bump; the seven-to-nine faint freckles
across nose bridge and upper cheeks; the 2mm beauty mark 1cm below the
outer corner of the LEFT eye; the 4mm pale scar through the right eyebrow
tail; thick unshaped brows with the LEFT sitting 1mm higher than the
right; dark brown 2B hair in a loose messy bun with two loose
face-framing strands.

MACRO DETAIL EMPHASIS: visible skin porosity across the nose and inner
cheeks, individual pores in focus; fine vellus hair catching the light
along the jawline and upper cheek; natural eyebrow asymmetry with
individual hairs growing in inconsistent directions at the brow head;
detailed iris — radial stromal fibres, a slightly darker outer ring, a
small crisp window catchlight at 10 o'clock in each eye; visible lower
lash line with unequal lash lengths; slight natural redness at the nose
sides and inner eye corners; faint dryness texture on the lower lip;
one or two small closed comedones near the chin, unretouched.

FIXED ACCESSORY: small gold huggie hoop, 10mm, visible on the LEFT ear,
catching a small specular highlight.

EXPRESSION: neutral-soft, lips barely parted, looking directly into the
lens, as if mid-sentence.

TECHNICAL: RAW photo quality, natural grain, no beauty filter, no skin
smoothing, true-to-life colour, shallow depth of field with the ears
falling slightly soft, handheld micro-blur, 5200K neutral white balance.
```

### Negative prompt (use on BOTH anchors and every clip)

```text
oversaturated skin, orange skin, waxy skin, plastic skin, airbrushed,
beauty filter, skin smoothing, poreless, glossy forehead, CGI, 3d render,
cartoon, anime, illustration, painting, digital art, doll-like, uncanny,
symmetrical face, perfect teeth, veneers, heavy makeup, contouring,
instagram filter, HDR, overexposed highlights, deformed hands, malformed
fingers, extra fingers, fused fingers, six fingers, missing fingers,
mangled hands, extra limbs, extra arms, deformed ears, asymmetrical eyes,
crossed eyes, dead eyes, lazy eye, distorted pupils, warped face,
elongated neck, long neck, bad anatomy, disfigured, mutated, blurry,
low resolution, jpeg artifacts, oversharpened, watermark, text, logo,
signature, duplicate person, twins, floating limbs, studio lighting,
ring light, professional model, stock photo, fashion editorial
```

### Anchor acceptance checklist

Do not proceed to §3 until every line is true of the accepted pair.

- [ ] Beauty mark is under the **left** eye in both images (engines mirror; reject if it swapped)
- [ ] Brow asymmetry visible and in the same direction in both
- [ ] Freckle density reads as the same person, not the same *type* of person
- [ ] Gold huggie present, same size, both anchors
- [ ] Pores visible at 100% zoom — no wax
- [ ] Hands, where visible, have five fingers with correct joint counts
- [ ] Catchlight direction agrees with the stated key light (camera left)

---

## 2. Thirty-Second Script & Audio Direction

Retention structure: motion before face, problem before product, texture
before claim, cart last. The cold-open line runs as voiceover over
product b-roll — the avatar is not on screen and not lip-synced until
0:03, which is both a retention choice and a cost saving.

| Time | Beat | On screen | Voiceover | On-screen text |
|---|---|---|---|---|
| **0:00–0:03** | Visual hook | REAL product macro. Dropper squeezed against dark slate, one amber bead swells and falls, rosemary sprig just in frame. Fast push-in. | *"Okay so the part-line thing finally stopped freaking me out."* | `the part-line thing` (0:00–0:02, centred high) |
| **0:03–0:12** | Problem / agitation | Avatar enters. Mirror selfie, medium, candid, talking directly to lens, small natural sway. At 0:09 she tips her head and parts her hair with two fingers. | *"My scalp was so dry it was flaking, and every time I brushed there'd be this little pile. I stopped wearing my hair up because the part was the first thing I'd see."* | none — face carries it |
| **0:12–0:22** | Demo / value stack | Cutaway: bottle into frame, label to camera. Dropper drawn up, oil beading along the part. Fingertips massage in small circles. Insert of oil texture on fingertips, light through it. | *"It's rosemary and peppermint — you literally feel it working, that cold tingle. Three drops down the part, massage, that's it. No grease, no wash day ruined."* | `rosemary + peppermint` (0:13) · `3 drops · massage · done` (0:17, stacked) |
| **0:22–0:30** | Offer / CTA | Back to mirror. Hair now down and brushed, she runs a hand through it, turns to lens, points down-left toward the cart. Hold on product on counter, in focus, last 1.5s. | *"Six weeks in and my brush is way emptier. It's in the orange cart below — grab it before the bundle sells out again."* | `ORANGE CART ↓` (0:24, lower-centre, above safe line) · `bundle deal` (0:27) |

**Word count:** 78 spoken words across 30s ≈ 156 wpm — conversational UGC
pace with room for two natural breaths. Do not write past 85.

### Audio specification

| Parameter | Value | Note |
|---|---|---|
| Voice type | Young adult American female, 22–28, conversational/"raw" — a voice described as casual, slightly breathy, imperfect. **Not** narration, not "warm professional", not audiobook | A polished read is the single loudest AI tell on this platform |
| Model | The highest-quality multilingual/v2-class model your account offers — **not** the low-latency turbo tier | Turbo flattens prosody; you have no latency requirement here |
| **Stability** | **42%** | Low enough for natural pitch drift, high enough to avoid mid-word wobble. Below 35% it slurs the product name |
| **Similarity / clarity** | **78%** | Above 85% it starts importing breath artefacts from the source sample |
| **Style exaggeration** | **18%** | Just enough lift on "literally" and "way emptier" |
| Speaker boost | On | |
| Delivery notes | Upward inflection on the 0:00 line. Audible breath before "It's rosemary and peppermint". Slight smile in the voice on the last sentence. | Generate 3 takes, keep the one with the best 0:00–0:03 |
| Music bed | 92–98 BPM soft house / lo-fi, warm, no vocal, no drop. Enters at 0:00, ducks under the VO. | Matching the bed to the 156 wpm read keeps the cut from feeling rushed |
| Levels | VO −14 LUFS integrated, true peak ≤ −1 dBTP. Bed at −24 LUFS under VO, −19 in the 0:00–0:03 gap and the final 1.5s. | −14 is the platform target; a bed at −24 stays audible without masking consonants |
| SFX | One soft dropper *click* at 0:01, one cloth/hair rustle at 0:09. Nothing else. | Two sounds read as real; five read as sound design |

---

## 3. Beat-by-Beat Video & Motion Prompts

Four segments: one real-footage hook plus three generated clips. The
table is the plan; the full copy-ready prompt for each clip is in the
block below it, because the prompts are long enough that a table cell
destroys them.

| Clip # | Timecode | Driving action / kinematics | AI video engine prompt | Motion weight |
|---|---|---|---|---|
| **Clip 0** | 0:00–0:03 | — *(no character; real product macro, shot or licensed)* | Not generated. See §4 — this is the plate that earns the label's legibility for the whole video. | — |
| **Clip 1** | 0:03–0:12 | Arm's-length mirror selfie. Handheld micro-drift, 2–3px sway. Talking to lens. At 0:09 head tips 15° right, left hand lifts, two fingers part the hair at the crown. | `[Identity Lock: Anchor 1 + 2]` + `[talks to mirror, small sway, tips head at 0:09 and parts hair with two fingers]` + `[35mm f/1.8, window key camera-left, handheld]` — full block below | **0.60** |
| **Clip 2** | 0:12–0:22 | Bottle raised into frame at chest height, label to lens, held 1.2s. Dropper drawn, lowered to the part. Fingertips make small circles at the scalp, elbow high. Camera holds. | `[Identity Lock]` + `[raises bottle to lens, draws dropper, applies along part, massages in small circles]` + `[50mm f/2.0, same window key, locked-off handheld]` — full block below | **0.70** |
| **Clip 3** | 0:22–0:30 | Hair now down. Right hand sweeps through it from crown to ends. Torso rotates 20° to camera, chin lifts, smile. Left hand points down-left off frame. Settles. | `[Identity Lock]` + `[sweeps hand through loose hair, turns to camera, smiles, points down-left]` + `[35mm f/1.8, window key camera-left, slight handheld settle]` — full block below | **0.55** |

**Motion weight, and what it actually maps to.** Higher weight follows the
driving footage more faithfully and drifts the face more. The numbers
above are tuned to that trade: Clip 2 is the one that must match the
driving kinematics exactly (hands near the scalp, product in frame), so
it takes the drift risk; Clip 3 carries the CTA and the face matters
more than the sweep, so it stays low.

| Engine family | Control that takes this number |
|---|---|
| Image-to-video with a motion slider (Kling, Hailuo, Luma) | Motion / creativity amount, direct |
| Video-to-video character swap (Runway Act-style, Viggle-style) | Driving-strength / retarget weight, direct |
| Diffusion v2v (AnimateDiff, WAN v2v) | Denoise strength ≈ the same number; ControlNet (OpenPose + depth) weight at 0.8–0.9 regardless |
| LivePortrait-style face retarget | Not applicable — set expression scale 1.0, and let this number drive the *body* pass only |

### Clip 1 — 0:03–0:12 (problem / agitation)

```text
[IDENTITY LOCK] The exact woman in the two reference images: 26, olive
Fitzpatrick IV complexion, dark brown #3B2A1E eyes with limbal ring,
freckles across nose bridge and upper cheeks, 2mm beauty mark below the
outer corner of the LEFT eye, 4mm scar through the right eyebrow tail,
thick unshaped brows with the left 1mm higher, dark brown 2B hair in a
loose messy bun with two loose face-framing strands, small gold huggie
hoops, oversized washed-black cotton tee. Face and wardrobe must remain
identical to the references for every frame.

[MOVEMENT] She holds the phone at arm's length toward a bathroom mirror
and talks directly into it, candid and unperformed. Natural micro-sway of
the shoulders, small weight shift on the left hip, two natural blinks,
eyebrows lifting slightly on emphasis. Partway through she tips her head
fifteen degrees to her right, raises her left hand, and parts the hair at
her crown with two fingers, holding it open for about a second before
letting it fall. Hands stay fully in frame throughout the parting gesture.

[CAMERA / LIGHT] 35mm lens at f/1.8, medium shot framed from the waist up,
handheld with believable micro-drift of two to three pixels, no zoom, no
push. Soft north-facing window key from camera LEFT at 45 degrees, warm
dim bulb fill on the shadow side, soft shadow under the jaw. White subway
tile falling gently out of focus behind her. RAW photographic quality,
natural film grain, unretouched skin with visible pores, 5200K neutral
white balance, imperfect handheld framing.

[NEGATIVE] see the shared negative block — plus: warped hands while
parting hair, fingers merging with hair, jewellery changing size, bun
untying, tee logo appearing, mirror reflection mismatch.
```

### Clip 2 — 0:12–0:22 (demo / value stack)

```text
[IDENTITY LOCK] Same woman as the two reference images — olive
Fitzpatrick IV skin, dark brown #3B2A1E eyes with limbal ring, freckles
across nose bridge and upper cheeks, 2mm beauty mark below the outer
corner of the LEFT eye, 4mm scar through the right eyebrow tail, left brow
1mm higher, dark brown 2B hair in a loose messy bun, gold huggie hoops,
oversized washed-black cotton tee. Identical face, hair and wardrobe in
every frame.

[MOVEMENT] She raises a small amber glass dropper bottle into frame at
chest height with her right hand, label turned toward the lens, and holds
it steady for a little over a second. She unscrews and draws the glass
dropper up, brings it to the top of her head, and releases three drops
along her centre part. She sets the bottle down out of frame, then works
both sets of fingertips into her scalp in small circles along the part
line, elbows raised, wrists relaxed. Deliberate, practised, unhurried.
Hands and fingers fully visible and anatomically correct throughout.

[CAMERA / LIGHT] 50mm lens at f/2.0, tighter medium shot from the chest
up, effectively locked off with only faint handheld breathing. Same soft
window key from camera LEFT at 45 degrees, same warm fill, same tile
background at the same depth. RAW photographic quality, natural grain,
visible skin texture, specular highlight on the oil where it catches the
window, 5200K neutral white balance.

[NEGATIVE] see the shared negative block — plus: warped or unreadable
label text, bottle changing shape or colour, dropper passing through the
hand, extra fingers at the scalp, oil rendering as solid white, hair
clipping through fingers.
```

### Clip 3 — 0:22–0:30 (offer / CTA)

```text
[IDENTITY LOCK] Same woman as the two reference images — olive
Fitzpatrick IV skin, dark brown #3B2A1E eyes with limbal ring, freckles
across nose bridge and upper cheeks, 2mm beauty mark below the outer
corner of the LEFT eye, 4mm scar through the right eyebrow tail, left brow
1mm higher, gold huggie hoops, oversized washed-black cotton tee. In this
clip only, the hair is DOWN — the same dark brown 2B wave, shoulder-blade
length, brushed, with the same two face-framing strands. Face and wardrobe
otherwise identical to the references.

[MOVEMENT] She sweeps her right hand through her loose hair from crown to
ends in one unhurried motion, letting it fall. Her torso rotates about
twenty degrees toward the camera, her chin lifts, and she gives a small
closed-mouth smile that reaches the eyes. Her left hand comes up and
points down and to her left, off the bottom of frame, held briefly. She
settles, one blink, still looking into the lens.

[CAMERA / LIGHT] 35mm lens at f/1.8, medium shot from the waist up,
handheld with a slight settle at the end of the move. Same soft window key
from camera LEFT at 45 degrees, same warm fill, same tile background. RAW
photographic quality, natural film grain, unretouched skin, 5200K neutral
white balance.

[NEGATIVE] see the shared negative block — plus: hand deforming while
passing through hair, hair melting or smearing at the ends, pointing hand
gaining fingers, smile distorting the jawline, wardrobe changing between
clips.
```

**Continuity gate between clips.** Before assembly, check that across
Clips 1–3 the window key stays camera-left, the gold huggies stay the same
size, the tee's neckline sits at the same depth, and the beauty mark stays
under the left eye. A clip that fails any of these gets regenerated, not
graded into agreement.

---

## 4. Post-Production & Compliance

### 4.1 Product occlusion — which seconds must carry real product

Generated video cannot hold legible label text. Any frame where the label
occupies more than about 8% of frame width **and** would be readable is a
frame that must be real. On this cut that is three windows:

| Window | What's wrong | Fix |
|---|---|---|
| **0:00–0:03** | Nothing — this is already a real plate | Shoot or license the dropper macro. This clip is doing double duty: it establishes the true label for the whole video, so the brief glimpses later read as the same bottle |
| **0:13.4–0:15.8** | Bottle held to lens in Clip 2. Label will warp, letterforms will wander | Rotoscope the bottle (or mask the label face only), planar-track the label plane, composite a real product still with matched grain and a matched specular highlight. 4-point corner pin is enough — the bottle is near-static for this beat |
| **0:19.0–0:20.6** | Dropper at the scalp; label small, angled, partly occluded by fingers | Cheapest correct fix: patch with a tracked still of the real dropper, feathered 4px. If the hand crosses the label, cut around it instead — a 0.4s insert of the real dropper covers the whole problem |
| **0:25.6–0:29.0** | Hero hold on the product at the CTA — the highest-stakes label moment in the video | Do not fix; **replace**. Cut to a real locked-off plate of the bottle on the counter for the final hold. Match white balance to 5200K and add the same window falloff from camera left |

Rule of thumb to carry into the next kit: *the avatar sells, the real
product closes.* Every frame a buyer might screenshot should be real
footage.

### 4.2 Lip-sync and the smoothing pass

Order matters — run these in sequence, not in parallel:

1. **Assemble first.** Cut Clips 1–3 to the VO before lip-syncing, so you
   only sync the frames that survive.
2. **Lip-sync pass.** Mouth crop at 384–512px. Padding `[top 0, bottom 12,
   left 0, right 0]` — the extra bottom pad keeps the chin from clipping
   when she smiles at 0:27. Smoothing **on**; disable it only if the sync
   visibly lags, since off produces per-frame jitter. Re-composite the
   synced mouth back with a **6px feathered** blend at the jaw, not a hard
   rectangle.
3. **Colour match the patch.** The synced region usually comes back half a
   stop brighter and slightly desaturated. Match it to the surrounding
   cheek before grading.
4. **De-plastic pass.** In order: a 0.3px Gaussian blur, then a light
   unsharp at radius 1.2 / amount 0.4 — this breaks the over-smooth
   surface without reintroducing a digital edge.
5. **Grain, last.** **1.5–2% monochrome neutral noise** over the whole
   timeline, including the real plates, so the AI clips and the real
   footage share a single noise floor. Below 1.2% the waxy look survives;
   above 2.5% TikTok's encoder turns it into blocking. Animate the grain
   per frame — static grain reads as a texture overlay.
6. **Final encode.** 1080 × 1920, 30 fps, H.264 High, CRF 18–20, audio
   −14 LUFS / ≤ −1 dBTP.

### 4.3 TikTok Shop safe zones

Working numbers at 1080 × 1920. Verify on a real device before you commit
a template — the UI shifts by app version, region and whether the video
carries a product anchor.

| Region | Keep clear | Why |
|---|---|---|
| Top | `y < 220` | For You / Following tabs, search icon |
| Right rail | `x > 800`, `y 780–1650` | Avatar, like, comment, share, spinning sound disc |
| Bottom | `y > 1400` | Handle, caption, music ticker — and the **product anchor / orange cart pill**, which is the one thing in this video that must never be covered |

**Effective text box: `x 60 → 800`, `y 260 → 1380`.** Put hook cards at
`y ≈ 700–1000`, where the thumb isn't and the eye already is.

The `ORANGE CART ↓` card at 0:24 is the trap — it wants to sit low, next
to the real cart, and that is exactly where the platform will cover it.
Park it at `y ≈ 1340` and let the arrow do the pointing.

If you render captions through this pipeline, `margin_v` in `config.toml`
(default **260**) is the same constraint expressed in the renderer; the
caption style's vertical margin and this table have to agree or burned-in
text will land under the UI.

### 4.4 Disclosure and claim compliance

**AIGC labelling — do this on every upload.**

1. In the post editor, open **More options** → **Content disclosure** (some
   builds: *Content disclosure and ads*), and turn on **AI-generated
   content**. This stamps the platform's own "Creator labeled as AI-generated"
   tag and is the disclosure the policy actually asks for.
2. Keep the C2PA metadata your generation tools attach — do not strip it
   in the export. TikTok reads it and may auto-label; an auto-label that
   agrees with your manual toggle is the clean outcome.
3. Add `#AIgenerated` in the caption as a belt-and-braces signal. It costs
   nothing in reach and it is evidence of good faith if the listing is
   reviewed.
4. The avatar is a synthetic person and must not resemble a real,
   identifiable individual. If a generated face starts looking like
   someone, regenerate it.

**Claim swaps — the VO in §2 already uses the right column.**

| Do not say | Say instead |
|---|---|
| "Stops shedding in 14 days" | "My brush is way emptier" |
| "Regrows your hairline" | "My part looks less wide to me" |
| "Clinically proven" | *(omit entirely unless your listing holds the study)* |
| "Cures dandruff / treats scalp disease" | "My scalp isn't flaking like it was" |
| "Works for everyone" | "This is just what happened for me" |

**Final QA before upload**

- [ ] Frame-by-frame scrub at 0.25× — no identity drift, no finger errors, no label warp outside the patched windows
- [ ] Blink rate looks human (roughly one every 3–5s; generated clips under-blink)
- [ ] Audio peaks ≤ −1 dBTP, integrated −14 LUFS
- [ ] Every burned-in text element inside `x 60–800`, `y 260–1380`
- [ ] Product anchor / cart area unobstructed for the full 30s
- [ ] AIGC toggle **on**, C2PA intact, `#AIgenerated` in caption
- [ ] VO contains no growth, timeline or clinical claim
- [ ] All four label-legible windows are real footage, not generated
