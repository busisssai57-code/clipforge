# UGC kits

A UGC kit is everything needed to produce one 30-second TikTok Shop video
around a single synthetic creator: the two reference images that lock the
face, the script and voice settings, the per-clip video prompts, and the
edit-and-compliance checklist. One file per product.

| File | What it is |
|---|---|
| `rosemary-hair-oil.md` | Worked kit — rosemary scalp oil, mirror-GRWM driving footage. Read this one first |
| `_template.md` | Blank kit with `{{placeholders}}` |

These are **documents, not code**. Nothing in `clipforge/` reads them, and
the pipeline does not generate avatars — the kit is the human-and-model
process that produces the footage, and the pipeline is what finishes it.

## The four stages, and why they're in that order

1. **Dual anchors.** Two reference images — one full-body/medium at
   1024 × 1536 for body, wardrobe and light, one macro selfie at
   1024 × 1024 for the face. Both list the same biological invariants in
   the same words. Identity drift across clips is the defining failure of
   AI UGC, and it is cheaper to prevent here than to fix anywhere else.
2. **Script and audio.** Thirty seconds split 3 / 9 / 10 / 8: motion
   before face, problem before product, texture before claim, cart last.
   Budget 75–85 spoken words.
3. **Clip prompts.** Three generated clips, each written as
   `[Identity Lock] + [Exact Body Movement] + [Camera/Lens/Lighting]`,
   with a motion weight per clip. Higher weight follows the driving
   footage more faithfully and drifts the face more, so spend the drift on
   the demo clip and save it on the CTA clip.
4. **Post and compliance.** Which seconds must carry real product, the
   lip-sync and de-plastic settings, the safe-zone box, and the AIGC
   disclosure steps.

## Starting a new kit

```bash
cp docs/ugc/_template.md docs/ugc/<product-slug>.md
```

Fill the input table first — product, angle, persona, driving video — then
work top to bottom. Each stage consumes the one above it, so filling §3
before §2 means rewriting §3.

Two things are worth carrying between kits unchanged: the negative prompt
block, and the safe-zone table.

## Where the pipeline comes in

The kit produces a finished 30-second cut. If you run that cut through
this repo for captions, loudness and packaging:

- **Leave jump-cut off** (`--no-jumpcut`). The VO is already tight to the
  frame, and silence removal will cut the breaths the script deliberately
  leaves in.
- **Caption margin.** `margin_v` in `config/config.toml` (default 260) is
  the renderer's version of the safe-zone table in every kit's §4.3. If
  you change one, change the other, or burned-in text lands under the
  platform UI.
- **Grain goes last, after the render**, so the encoder isn't smoothing it
  back out — 1.5–2% is the band that survives TikTok's transcode.
- **The export pack** already assembles caption, hashtags and thumbnail
  next to the clip. `#AIgenerated` belongs in that caption, and the AIGC
  toggle still has to be set by hand in the post editor — the pipeline
  produces files and stops, so disclosure is an operator step by
  construction.
