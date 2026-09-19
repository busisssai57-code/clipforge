# UGC Kit — {{PRODUCT_NAME}}

Blank kit. Fill the input table, then work top to bottom — each stage
consumes the one above it. `docs/ugc/rosemary-hair-oil.md` is a filled
copy of this file and is the better thing to read first.

| Input | Value |
|---|---|
| **Product / category** | `{{PRODUCT_NAME}}` — `{{CATEGORY}}` |
| **Selling angle / pain point** | `{{PAIN_POINT}}` |
| **Persona** | `{{PERSONA_ID}}` — `{{AGE}}`, `{{GENDER}}`, `{{SKIN_TONE}}`, `{{HAIR}}`, `{{WARDROBE}}` |
| **Driving video** | `{{DRIVING_VIDEO}}` — pacing, framing, physical actions |
| **Delivery** | 1080 × 1920, 30 fps, H.264, audio −14 LUFS integrated |
| **Anchor seed** | `{{SEED}}` — record it, you will need it again |

> **Claim posture.** Write the one-paragraph statement of what this
> category is *not allowed* to claim before you write a word of script.
> Supplements, skincare, haircare, and anything health-adjacent all carry
> claim classes that turn a good video into a takedown. §4.4 has the swap
> table; fill it for this product.

---

## 1. Dual-Anchor Character Generation

### Anchor 1 — Full-body / medium foundation (1024 × 1536)

```text
RAW photo, candid vertical portrait of a {{AGE}}-year-old {{GENDER}} in
{{ENVIRONMENT}}, {{FRAMING}}, shot on a 35mm lens at f/1.8, photographed
not posed.

IDENTITY — BIOLOGICAL INVARIANTS (these never change across any image):
{{AGE}} years old. {{SKIN_TONE}} complexion, Fitzpatrick {{FITZPATRICK}},
{{UNDERTONE}} undertone. {{EYE_COLOUR}} eyes, {{EYE_HEX}}, {{EYE_SHAPE}}.
{{FACE_SHAPE}}, {{CHEEKBONES}}, {{NOSE}}, {{JAW_ASYMMETRY}}.
MICRO-FEATURES: {{FRECKLES_OR_MARKS}}; {{BEAUTY_MARK_AND_SIDE}};
{{SCAR_OR_DISTINGUISHING_MARK}}. {{EYEBROWS_AND_ASYMMETRY}}. {{MAKEUP}}.

HAIR: {{HAIR_COLOUR}}, {{HAIR_TEXTURE}}, {{HAIR_LENGTH}}, worn
{{HAIR_STYLE}}, {{HAIR_IMPERFECTIONS}}.

WARDROBE + FABRIC: {{GARMENT}}, {{FABRIC_WEIGHT_AND_TEXTURE}},
{{HOW_IT_DRAPES}}. {{LOWER_GARMENT}}. {{FIXED_ACCESSORY}}.

POSTURE: {{WEIGHT_DISTRIBUTION}}, {{SHOULDER_LINE}}, {{HEAD_ANGLE}},
{{HANDS}}. Relaxed, unposed, mid-thought.

ENVIRONMENT + LIGHT: {{ROOM}}, {{SURFACES}}, {{PROPS}}. Key light is
{{KEY_QUALITY}} from camera {{KEY_DIRECTION}} at {{KEY_ANGLE}}, falling
off across {{FALLOFF_SIDE}}; {{FILL_SOURCE}} fills the shadow side at
about {{FILL_RATIO}}. Visible soft shadow under the jaw.

TECHNICAL: RAW photo quality, natural film grain, true-to-life skin tone,
slight lens vignetting, mild sensor noise in the shadows, shallow depth of
field, imperfect handheld framing, neutral white balance around
{{KELVIN}}K. Photojournalistic, unretouched.
```

### Anchor 2 — Macro close-up selfie (1024 × 1024)

Generate **from Anchor 1** as a reference image at 0.35–0.45 denoise.

```text
RAW macro selfie close-up of the SAME {{AGE}}-year-old {{GENDER}} as the
reference image, head and shoulders only, arm's-length phone distance,
shot on a 50mm lens at f/2.0, {{KEY_QUALITY}} from camera {{KEY_DIRECTION}}.

CARRY OVER EXACTLY FROM REFERENCE: {{SKIN_TONE}} Fitzpatrick
{{FITZPATRICK}} complexion; {{EYE_COLOUR}} {{EYE_HEX}} eyes;
{{FACE_SHAPE}}; {{FRECKLES_OR_MARKS}}; {{BEAUTY_MARK_AND_SIDE}};
{{SCAR_OR_DISTINGUISHING_MARK}}; {{EYEBROWS_AND_ASYMMETRY}};
{{HAIR_COLOUR}} {{HAIR_TEXTURE}} hair {{HAIR_STYLE}}.

MACRO DETAIL EMPHASIS: visible skin porosity across the nose and inner
cheeks; fine vellus hair along the jawline; natural eyebrow asymmetry with
individual hairs growing in inconsistent directions at the brow head;
detailed iris — radial stromal fibres, darker outer ring, small crisp
catchlight at {{CATCHLIGHT_CLOCK}} in each eye; visible lower lash line
with unequal lash lengths; {{SKIN_IMPERFECTION}}, unretouched.

FIXED ACCESSORY: {{FIXED_ACCESSORY}}, visible on the {{ACCESSORY_SIDE}},
catching a small specular highlight.

EXPRESSION: neutral-soft, lips barely parted, looking directly into the
lens, as if mid-sentence.

TECHNICAL: RAW photo quality, natural grain, no beauty filter, no skin
smoothing, true-to-life colour, shallow depth of field, handheld
micro-blur, {{KELVIN}}K neutral white balance.
```

### Negative prompt (both anchors and every clip — reusable as-is)

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

- [ ] Asymmetric micro-feature on the **same side** in both images (engines mirror)
- [ ] Brow asymmetry visible and in the same direction in both
- [ ] Freckles/marks read as the same person, not the same type of person
- [ ] Fixed accessory present, same size, both anchors
- [ ] Pores visible at 100% zoom — no wax
- [ ] Hands, where visible, anatomically correct
- [ ] Catchlight direction agrees with the stated key light
- [ ] Seed recorded in the input table

---

## 2. Thirty-Second Script & Audio Direction

| Time | Beat | On screen | Voiceover | On-screen text |
|---|---|---|---|---|
| **0:00–0:03** | Visual hook | REAL product b-roll, high motion, no avatar | `{{HOOK_LINE}}` | `{{HOOK_CARD}}` |
| **0:03–0:12** | Problem / agitation | Avatar enters, direct to lens, candid | `{{PROBLEM_VO}}` | usually none |
| **0:12–0:22** | Demo / value stack | Product in hand, texture insert, use demonstrated | `{{DEMO_VO}}` | `{{VALUE_CARDS}}` |
| **0:22–0:30** | Offer / CTA | Result beat, turn to camera, point to cart, hold on real product | `{{CTA_VO}}` | `ORANGE CART ↓` |

**Budget: 75–85 spoken words ≈ 150–160 wpm.** Count before you record.

### Audio specification

| Parameter | Value |
|---|---|
| Voice type | `{{VOICE_DESCRIPTION}}` — conversational, imperfect, not narration |
| Model | Highest-quality v2-class model available, not the turbo tier |
| Stability | `{{40–45}}%` |
| Similarity / clarity | `{{75–80}}%` |
| Style exaggeration | `{{15–20}}%` |
| Speaker boost | On |
| Music bed | `{{BPM}}` BPM `{{GENRE}}`, no vocal, no drop |
| Levels | VO −14 LUFS int. / ≤ −1 dBTP; bed −24 LUFS under VO, −19 in gaps |
| SFX | Two, maximum |

---

## 3. Beat-by-Beat Video & Motion Prompts

| Clip # | Timecode | Driving action / kinematics | AI video engine prompt | Motion weight |
|---|---|---|---|---|
| Clip 0 | 0:00–0:03 | — *(real product b-roll)* | not generated | — |
| Clip 1 | 0:03–{{T1}} | `{{KINEMATICS_1}}` | `[Identity Lock]` + `[{{MOVEMENT_1}}]` + `[{{LENS_LIGHT_1}}]` | `{{0.55–0.65}}` |
| Clip 2 | {{T1}}–{{T2}} | `{{KINEMATICS_2}}` | `[Identity Lock]` + `[{{MOVEMENT_2}}]` + `[{{LENS_LIGHT_2}}]` | `{{0.65–0.75}}` |
| Clip 3 | {{T2}}–0:30 | `{{KINEMATICS_3}}` | `[Identity Lock]` + `[{{MOVEMENT_3}}]` + `[{{LENS_LIGHT_3}}]` | `{{0.55–0.65}}` |

Higher motion weight follows the driving footage more faithfully and
drifts the face more. Spend the drift where the kinematics matter (the
demo clip) and save it where the face matters (the CTA clip).

Copy-ready block, one per clip:

```text
[IDENTITY LOCK] The exact person in the two reference images:
{{INVARIANTS_ONE_LINE}}. Face and wardrobe must remain identical to the
references for every frame.

[MOVEMENT] {{EXACT_BODY_MOVEMENT_FROM_DRIVING_FOOTAGE}}. Hands fully in
frame and anatomically correct throughout.

[CAMERA / LIGHT] {{LENS}} at {{APERTURE}}, {{SHOT_SIZE}}, {{CAMERA_MOVE}}.
{{KEY_QUALITY}} from camera {{KEY_DIRECTION}} at {{KEY_ANGLE}},
{{FILL}}, {{BACKGROUND}}. RAW photographic quality, natural film grain,
unretouched skin with visible pores, {{KELVIN}}K neutral white balance.

[NEGATIVE] shared negative block — plus: {{CLIP_SPECIFIC_FAILURES}}.
```

**Continuity gate.** Across all clips: key light on the same side, fixed
accessory the same size, wardrobe identical, asymmetric micro-feature on
the same side. A clip that fails gets regenerated, not graded into
agreement.

---

## 4. Post-Production & Compliance

### 4.1 Product occlusion

Any frame where the label is wider than ~8% of frame **and** legible must
be real footage. List every such window:

| Window | What's wrong | Fix |
|---|---|---|
| `{{TC}}` | `{{DEFECT}}` | `{{mask / corner-pin real still / replace with real plate}}` |

*The avatar sells, the real product closes.* Anything a buyer might
screenshot should be real.

### 4.2 Lip-sync and smoothing

1. Assemble to the VO before syncing.
2. Mouth crop 384–512px, padding `[0, 12, 0, 0]`, smoothing on.
3. Re-composite with a 6px feathered blend at the jaw.
4. Colour-match the patched region to the surrounding cheek.
5. De-plastic: 0.3px Gaussian, then unsharp radius 1.2 / amount 0.4.
6. **Grain last: 1.5–2% monochrome neutral noise**, animated per frame,
   over real and generated footage alike.
7. Encode 1080 × 1920, 30 fps, H.264 High, CRF 18–20, −14 LUFS.

### 4.3 Safe zones (1080 × 1920 — verify on a device)

| Region | Keep clear |
|---|---|
| Top | `y < 220` |
| Right rail | `x > 800`, `y 780–1650` |
| Bottom (incl. product anchor / cart pill) | `y > 1400` |

Effective text box `x 60 → 800`, `y 260 → 1380`. If you burn captions
through this pipeline, `margin_v` in `config.toml` (default 260) must
agree with this table.

### 4.4 Disclosure and claim compliance

1. Post editor → **More options** → **Content disclosure** → **AI-generated
   content**, on.
2. Preserve C2PA metadata through export.
3. `#AIgenerated` in the caption.
4. Synthetic person must not resemble a real, identifiable individual.

| Do not say | Say instead |
|---|---|
| `{{BANNED_CLAIM}}` | `{{COMPLIANT_PHRASING}}` |

### Final QA

- [ ] 0.25× scrub — no identity drift, finger errors, or label warp
- [ ] Blink rate roughly one per 3–5s
- [ ] −14 LUFS integrated, ≤ −1 dBTP
- [ ] All burned-in text inside the box
- [ ] Cart area unobstructed for the full 30s
- [ ] AIGC toggle on, C2PA intact, hashtag present
- [ ] No claim outside the approved column
- [ ] Every label-legible window is real footage
