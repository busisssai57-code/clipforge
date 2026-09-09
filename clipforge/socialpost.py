"""The post layer — what turns a stack of shots into a post.

A generated sequence and a piece of short-form comedy differ by things
that have nothing to do with the model: a hook card on frame one, emoji
stamped on the punchlines, and a handle in the corner of every frame. The
reference this was built from (a 65s Somali animal sketch, 35 shots, 1.86s
mean) carries all three, and without them the same shots read as an
animation reel rather than as a post.

Everything here is RASTERISED WITH PILLOW and overlaid, rather than drawn
by ffmpeg's drawtext:

* Colour emoji. drawtext renders CBDT/COLR glyphs as monochrome tofu on
  most ffmpeg builds, and a black-and-white 😂 is not the joke.
* Escaping. drawtext's filter-graph syntax eats colons, commas, quotes and
  backslashes, so any text taken from a script has to survive two layers
  of escaping to reach the screen. An image has no syntax.
* Fonts. libass and drawtext both substitute silently when a font is
  missing; here the chosen file is logged and a missing one is refused.

The layer is deterministic — the same text and geometry produce the same
PNG bytes, and sticker positions come from the sticker's INDEX rather than
from randomness — because §3.2 says a re-run of the same input must be
byte-identical, and "the emoji moved" is exactly the kind of difference
that would make a diff useless.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.log import get_logger

log = get_logger(__name__)


class PostError(Exception):
    """The post layer could not do what it was asked."""


#: Text faces, in preference order. Heavy weights first: a hook card is
#: read at arm's length on a phone in one second.
_TEXT_FONTS = ("arialbd.ttf", "seguibl.ttf", "segoeuib.ttf", "arial.ttf",
               "DejaVuSans-Bold.ttf")
#: Colour-emoji faces. Segoe UI Emoji is the Windows one; the others are
#: what Linux boxes carry. A machine with none of them gets no stickers,
#: said out loud rather than silently dropped.
_EMOJI_FONTS = ("seguiemj.ttf", "NotoColorEmoji.ttf", "AppleColorEmoji.ttc")
_FONT_DIRS = (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
              Path("/usr/share/fonts/truetype/dejavu"),
              Path("/usr/share/fonts/truetype/noto"),
              Path("/System/Library/Fonts"))


def find_font(candidates: tuple[str, ...]) -> Path | None:
    """First existing font file from ``candidates``, or None.

    Returned rather than raised so the caller decides whether the feature
    is optional (stickers) or required (the hook card).
    """
    for name in candidates:
        for d in _FONT_DIRS:
            p = d / name
            if p.is_file():
                return p
    return None


@dataclass(frozen=True)
class Sticker:
    """One emoji stamped on the picture.

    ``start`` and ``end`` are seconds into the finished piece. ``index``
    decides where it sits: positions cycle through a fixed ring so two
    stickers in a row do not land on top of each other, and so the same
    piece rendered twice puts them in the same places.
    """

    emoji: str
    start: float
    end: float
    index: int = 0


@dataclass(frozen=True)
class Subtitle:
    """One spoken line, held for exactly the shot that says it."""

    text: str
    start: float
    end: float


@dataclass(frozen=True)
class PostSpec:
    """Everything the post layer stamps on a finished sequence."""

    #: Burned over the opening, uppercase. The reference's is a question
    #: ("which one made you laugh?") — the hook is an ask, not a title.
    hook: str = ""
    #: How long the hook holds. Short: it is read once.
    hook_seconds: float = 2.0
    #: The operator's own handle. Deliberately a parameter with no
    #: default identity — copying a style is fair, stamping someone
    #: else's mark on your own video is not.
    watermark: str = ""
    stickers: tuple[Sticker, ...] = field(default_factory=tuple)
    #: Dialogue burned into the picture. The script's lines reached
    #: `sequence.srt` and stopped: a subtitle file beside a video is not
    #: a subtitle anyone watching a TikTok sees, because nothing in that
    #: player will ever load it. For a sketch whose punchline is a line,
    #: burning it in is the difference between telling the joke and
    #: shipping a silent clip of a baby.
    subtitles: tuple[Subtitle, ...] = field(default_factory=tuple)
    #: Fractions of the frame, so one spec renders at any size.
    hook_size: float = 0.062
    watermark_size: float = 0.030
    sticker_size: float = 0.155
    #: Smaller than the hook: the hook is an ask read once, a subtitle is
    #: read while the picture is doing the work.
    subtitle_size: float = 0.040

    def is_empty(self) -> bool:
        return not (self.hook or self.watermark or self.stickers)


#: Where stickers land, as (x, y) fractions of the frame — the anchor is
#: the sticker's CENTRE. Kept off the middle third so the sticker never
#: covers the face the shot is about, and away from the very bottom where
#: the platform's own UI sits.
_STICKER_SLOTS = ((0.76, 0.42), (0.24, 0.55), (0.72, 0.66), (0.28, 0.34))

#: Where a spoken line sits. Low, under everything: the sticker slots
#: occupy 0.34-0.66 and ari_bridge reserves 750-1050 of a 1920 frame
#: (0.39-0.55) for a face, so a subtitle any higher lands on the joke or
#: on the child. Above the handle at H-0.035 and clear of it.
_SUBTITLE_Y = 0.845


def _pillow():
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover - Pillow is a hard dep
        raise PostError("Pillow is required for the post layer") from exc
    return Image, ImageDraw, ImageFont


def _is_notdef(canvas) -> bool:
    """Whether this render is a font's substituted .notdef rectangle."""
    opaque = {px[:3] for px in canvas.getdata() if px[3] > 240}
    return len(opaque) <= 1


def render_emoji(emoji: str, px: int, dest: Path) -> Path | None:
    """Rasterise one emoji to an RGBA PNG, in colour.

    Returns None when the machine has no colour-emoji font — a missing
    sticker is a cosmetic loss and must not fail a render that otherwise
    worked. ``embedded_color=True`` is the whole point: without it Pillow
    draws the glyph's outline in a flat colour and the result looks like a
    typo rather than a sticker.
    """
    Image, ImageDraw, ImageFont = _pillow()
    face = find_font(_EMOJI_FONTS)
    if face is None:
        log.warning("post.no_emoji_font", tried=list(_EMOJI_FONTS),
                    note="stickers skipped; install a colour emoji font")
        return None
    # Segoe UI Emoji ships bitmap strikes at 109px; asking for anything
    # else makes Pillow raise "invalid pixel size" rather than scale, so
    # it is rendered at the native size and resized afterwards.
    try:
        font = ImageFont.truetype(str(face), 109)
    except OSError as exc:
        log.warning("post.emoji_font_unusable", font=str(face),
                    error=str(exc)[:200])
        return None
    canvas = Image.new("RGBA", (160, 160), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        draw.text((80, 80), emoji, font=font, anchor="mm", embedded_color=True)
    except Exception as exc:  # noqa: BLE001 - one bad glyph is not fatal
        log.warning("post.emoji_draw_failed", emoji=emoji, error=str(exc)[:200])
        return None
    box = canvas.getbbox()
    if box is None or _is_notdef(canvas):
        # Either the font drew nothing, or it drew .notdef — the flat
        # rectangle a font substitutes for a codepoint it does not have.
        # Both must be caught: an empty PNG overlays as an invisible
        # no-op and a tofu box overlays as a black rectangle stamped on
        # the punchline, and each reads as "stickers are broken" rather
        # than "that character is not an emoji in this font".
        #
        # .notdef is detected by COLOUR, not by shape: a colour emoji
        # antialiases into hundreds of tones (measured: 317 for the
        # laughing face, 34 for the deliberately monochrome black square)
        # while the substituted box is a single flat fill.
        log.warning("post.emoji_missing_glyph", emoji=emoji, font=face.name)
        return None
    canvas = canvas.crop(box).resize((px, px), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "PNG")
    return dest


def render_card(text: str, *, width: int, size_px: int, dest: Path,
                uppercase: bool = True) -> Path:
    """Rasterise a hook card: white caps, heavy black outline, wrapped.

    The outline rather than a box because the reference's card sits
    directly on the picture, and a solid plate reads as a slide. Wrapping
    is measured against the real font, not estimated from character
    counts — a Somali hook is long and a mis-estimate pushes it off frame.
    """
    Image, ImageDraw, ImageFont = _pillow()
    face = find_font(_TEXT_FONTS)
    if face is None:
        raise PostError(
            f"no usable text font found (tried {', '.join(_TEXT_FONTS)}); "
            "the hook card cannot be drawn")
    body = (text or "").strip()
    if uppercase:
        body = body.upper()
    font = ImageFont.truetype(str(face), size_px)
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    margin = int(width * 0.06)
    usable = width - 2 * margin

    lines: list[str] = []
    current = ""
    for word in body.split():
        trial = f"{current} {word}".strip()
        if probe.textlength(trial, font=font) <= usable or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)

    stroke = max(2, size_px // 9)
    line_h = int(size_px * 1.18)
    height = line_h * len(lines) + 2 * stroke + int(size_px * 0.3)
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for i, line in enumerate(lines):
        draw.text((width // 2, int(size_px * 0.15) + stroke + i * line_h),
                  line, font=font, anchor="ma", fill=(255, 255, 255, 255),
                  stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    dest.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dest, "PNG")
    return dest


def build_overlay(spec: PostSpec, *, width: int, height: int,
                  work_dir: Path) -> tuple[list[str], str]:
    """Assemble the overlay inputs and filter graph.

    Returns ``(extra_ffmpeg_inputs, filter_complex)`` with the final video
    on ``[v]``. Built separately from running ffmpeg so the graph is
    testable without a render: this is string-assembly around geometry,
    which is where these bugs live.
    """
    inputs: list[str] = []
    steps: list[str] = []
    chain = "[0:v]"
    idx = 1

    if spec.hook.strip():
        card = render_card(spec.hook, width=int(width * 0.88),
                           size_px=max(12, int(height * spec.hook_size)),
                           dest=work_dir / "hook.png")
        inputs += ["-i", str(card)]
        # Held from 0, then gone. `enable` rather than a shorter input so
        # the card cannot extend the piece.
        steps.append(
            f"{chain}[{idx}:v]overlay=x=(W-w)/2:y=H*0.11:"
            f"enable='lt(t,{spec.hook_seconds:.3f})'[v{idx}]")
        chain = f"[v{idx}]"
        idx += 1

    for st in spec.stickers:
        px = max(16, int(height * spec.sticker_size))
        png = render_emoji(st.emoji, px, work_dir / f"sticker{st.index}.png")
        if png is None:
            continue
        fx, fy = _STICKER_SLOTS[st.index % len(_STICKER_SLOTS)]
        inputs += ["-i", str(png)]
        steps.append(
            f"{chain}[{idx}:v]overlay=x=W*{fx:.3f}-w/2:y=H*{fy:.3f}-h/2:"
            f"enable='between(t,{st.start:.3f},{st.end:.3f})'[v{idx}]")
        chain = f"[v{idx}]"
        idx += 1

    for si, sub in enumerate(spec.subtitles):
        if not sub.text.strip():
            continue
        # A rendered card, not drawtext. The module docstring explains
        # why: drawtext's filter syntax eats colons, commas and quotes,
        # and a Somali line is exactly the sort of text that will one day
        # contain an apostrophe and silently break the whole graph.
        card = render_card(sub.text, width=int(width * 0.86),
                           size_px=max(10, int(height * spec.subtitle_size)),
                           dest=work_dir / f"sub{si}.png", uppercase=False)
        inputs += ["-i", str(card)]
        steps.append(
            f"{chain}[{idx}:v]overlay=x=(W-w)/2:y=H*{_SUBTITLE_Y:.3f}-h/2:"
            f"enable='between(t,{sub.start:.3f},{sub.end:.3f})'[v{idx}]")
        chain = f"[v{idx}]"
        idx += 1

    if spec.watermark.strip():
        mark = render_card(spec.watermark, width=int(width * 0.6),
                           size_px=max(8, int(height * spec.watermark_size)),
                           dest=work_dir / "watermark.png", uppercase=False)
        inputs += ["-i", str(mark)]
        # Last in the chain and always on: a handle that disappears under
        # a sticker is not a handle.
        steps.append(f"{chain}[{idx}:v]overlay=x=(W-w)/2:y=H-h-H*0.035"
                     f"[v{idx}]")
        chain = f"[v{idx}]"
        idx += 1

    if not steps:
        return [], ""
    # Rename the last labelled output to [v] so the caller maps one name.
    steps[-1] = steps[-1].rsplit("[v", 1)[0] + "[v]"
    return inputs, ";".join(steps)


def apply_post(src: Path, dest: Path, spec: PostSpec, *,
               work_dir: Path | None = None) -> Path:
    """Burn the post layer into ``src``, writing ``dest``.

    Audio is copied, never re-encoded: this pass touches the picture only,
    and a needless AAC round-trip would put a generation loss on every
    piece for nothing.
    """
    from clipforge.ffmpeg import probe, require_binary, run

    if spec.is_empty():
        raise PostError("nothing to stamp: the spec has no hook, watermark "
                        "or stickers")
    info = probe(src)
    width, height = int(info.width or 0), int(info.height or 0)
    if width <= 0 or height <= 0:
        raise PostError(f"could not read the size of {src}")
    work_dir = work_dir or dest.parent / ".post"
    work_dir.mkdir(parents=True, exist_ok=True)

    inputs, graph = build_overlay(spec, width=width, height=height,
                                  work_dir=work_dir)
    if not graph:
        raise PostError("the post layer produced no overlays (no usable "
                        "fonts?)")
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
           "-loglevel", "error", "-y", "-i", str(src), *inputs,
           "-filter_complex", graph, "-map", "[v]"]
    if info.a_codec:
        cmd += ["-map", "0:a", "-c:a", "copy"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(dest)]
    proc = run(cmd, timeout=1800.0)
    if proc.returncode != 0 or not dest.is_file():
        raise PostError(f"post layer failed: {(proc.stderr or '')[-400:]}")
    log.info("post.applied", src=str(src), dest=str(dest),
             hook=bool(spec.hook), stickers=len(spec.stickers),
             watermark=bool(spec.watermark))
    return dest


def spec_from_shots(shot_seconds: list[float], sticker_marks: list[list[str]],
                    *, hook: str = "", watermark: str = "",
                    hook_seconds: float = 2.0,
                    spoken: list[str] | None = None) -> PostSpec:
    """Turn per-shot marks into a timed spec.

    ``sticker_marks[i]`` is what shot ``i`` was marked with in the script.
    A sticker is held for its shot and no longer: it is punctuation on one
    beat, and a sticker that outlives its cut reads as a rendering fault.
    """
    stickers: list[Sticker] = []
    subtitles: list[Subtitle] = []
    t = 0.0
    n = 0
    for i, seconds in enumerate(shot_seconds):
        # The line is held for its shot and no longer, exactly like a
        # sticker: dialogue that outlives its cut is being said by the
        # wrong picture.
        line = (spoken[i] if spoken and i < len(spoken) else "") or ""
        if line.strip():
            subtitles.append(Subtitle(line.strip(), t, t + seconds))
        marks = sticker_marks[i] if i < len(sticker_marks) else []
        for emoji in marks:
            stickers.append(Sticker(emoji=emoji, start=round(t, 3),
                                    end=round(t + seconds, 3), index=n))
            n += 1
        t += seconds
    return PostSpec(hook=hook, hook_seconds=hook_seconds, watermark=watermark,
                    stickers=tuple(stickers), subtitles=tuple(subtitles))


#: A piece whose shots differ by more than this is one that plays silent
#: and then jumps. 20 dB is roughly the step from "quiet room" to
#: "conversation" -- audible as a fault rather than as dynamics.
AUDIO_SPREAD_WARN_DB = 20.0


def audio_spread(levels: "dict[str, float | None]") -> "tuple[float, str, str] | None":
    """The loudest and quietest measurable shot, and the gap between them.

    Pure, so the threshold is testable without encoding anything. Returns
    None when fewer than two shots could be measured -- one shot has no
    spread, and an unmeasurable one is not evidence of a quiet one.
    """
    known = {k: v for k, v in levels.items() if v is not None}
    if len(known) < 2:
        return None
    lo_k = min(known, key=lambda k: known[k])
    hi_k = max(known, key=lambda k: known[k])
    return known[hi_k] - known[lo_k], lo_k, hi_k


def measure_mean_dbfs(video: "Path") -> "float | None":
    """Mean volume of *video* in dBFS, or None if it cannot be measured."""
    import re
    import subprocess

    from clipforge.ffmpeg import require_binary

    try:
        proc = subprocess.run(
            [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
             "-i", str(video), "-af", "volumedetect", "-f", "null", "-"],
            capture_output=True, text=True, timeout=60.0)
    except Exception:  # noqa: BLE001 - a measurement is never worth a run
        return None
    hit = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", proc.stderr or "")
    return float(hit.group(1)) if hit else None


def report_audio_spread(paths: "list[Path]") -> "tuple[float, str, str] | None":
    """Warn when a sequence's shots are wildly different in loudness.

    MEASURED on a delivered three-shot sequence 2026-09-05: -84.3 and
    -74.4 dBFS for the first two shots and -26.9 for the third. That is
    3.8 seconds of silence and then a jump to the edge of clipping, in a
    5.7-second piece.

    It is REPORTED, not corrected, and the distinction matters. Loudness
    normalisation cannot fix it: EBU R128 gates silence out, so the
    integrated measurement reflects only the shot that has audio, and
    that shot already sits at the true-peak ceiling. A loudnorm pass was
    tried on this exact file and moved it by 0.0 LUFS. Nothing can create
    sound in a shot the model returned silent.

    Handling belongs downstream, where ari_channel's post layer excludes
    inaudible tracks and lets the music bed carry the piece. This exists
    so an operator who plays a raw sequence knows why it sounds broken.
    """
    levels = {p.name: measure_mean_dbfs(p) for p in paths}
    found = audio_spread(levels)
    if found is None:
        return None
    spread, quietest, loudest = found
    if spread < AUDIO_SPREAD_WARN_DB:
        return None
    log.warning("post.audio_spread", spread_db=round(spread, 1),
                quietest=quietest, loudest=loudest,
                levels={k: v for k, v in levels.items() if v is not None},
                note="shots differ enough that the piece will play silent "
                     "and then jump; the model returned no audio for some "
                     "of them, and no normalisation can create it")
    return found
