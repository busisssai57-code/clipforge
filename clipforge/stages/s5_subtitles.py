"""S5 — karaoke subtitles (.ass) for one clip window.

Spec §S5: word-level karaoke, active word highlighted, inside the 9:16 safe
area. CPU-only and fully deterministic: the same transcript and config always
produce byte-identical .ass output.

Design notes, each a correction to the sketch this replaces:

  * It is a real :class:`Stage`. The sketch was a bare class with no cache
    key, no artifact and no resumability, so it sat outside the DAG Law and
    re-ran from scratch on every invocation.
  * Every styling value comes from config (font, size, colours, outline,
    shadow, margin, words-per-line, max lines). The sketch hardcoded Arial 85
    with a 300px margin and 4 words, so the config section existed but did
    nothing.
  * Karaoke ``\\k`` durations are laid against the LINE's own start, with
    gaps between words emitted as unhighlighted spacers. ``\\k`` is a
    *cumulative* timeline, not a per-word duration in isolation — summing
    only word durations makes the highlight drift ahead of the audio by the
    total pause length, and a 30-60 s clip has plenty of pauses.
  * Times are clip-relative and quantized to centiseconds, which is the
    resolution the ASS format itself stores.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from clipforge.errors import FatalStageError
from clipforge.log import get_logger
from clipforge.paths import atomic_write_text
from clipforge.schemas.render import SubtitleArtifact
from clipforge.stages.base import Stage

log = get_logger(__name__)

#: ASS stores times as h:mm:ss.cc — centisecond resolution. Quantizing here
#: rather than at format time keeps the karaoke arithmetic and the printed
#: timestamps consistent with each other.
CS = 100


def _cs(seconds: float) -> int:
    """Seconds -> whole centiseconds, never negative."""
    return max(0, round(seconds * CS))


def _fmt(cs: int) -> str:
    """Centiseconds -> ASS h:mm:ss.cc."""
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _escape(text: str) -> str:
    """ASS-escape one word.

    A literal ``{`` opens an override block, so an unescaped brace in the
    transcript would swallow the rest of the line into a style directive.
    """
    return (text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            .replace("\n", " ").strip())


#: A word that deserves emphasis colour: contains a digit or currency —
#: amounts and counts are what short-form viewers' eyes snap to.
_EMPHASIS = re.compile(r"[\d$€£]")

#: Emphasis colour (ASS BGR): spring green — reads as "money/number" against
#: the yellow karaoke highlight without clashing with it.
_EMPHASIS_COLOUR = "&H007FFF00&"

#: Auto-emoji mapping for Opus Clip style visual cues on key words.
EMOJI_MAP: dict[str, str] = {
    "money": "💰", "cash": "💵", "dollar": "💵", "dollars": "💵", "rich": "🤑", "crypto": "🪙", "bitcoin": "🪙",
    "fire": "🔥", "hot": "🔥", "lit": "🔥", "burn": "🔥",
    "rocket": "🚀", "moon": "🌕", "growth": "📈", "chart": "📈", "scale": "⚡",
    "brain": "🧠", "mind": "🧠", "think": "💡", "idea": "💡", "smart": "💡",
    "time": "⏱️", "clock": "⏰", "fast": "⚡", "quick": "⚡", "speed": "⚡",
    "top": "🏆", "win": "🏆", "winner": "👑", "king": "👑", "best": "⭐", "secret": "🤫",
    "stop": "🛑", "warning": "⚠️", "danger": "🚨", "fake": "❌", "wrong": "❌",
    "love": "❤️", "heart": "❤️", "life": "✨", "death": "💀", "dead": "💀",
}

SUBTITLE_THEMES: dict[str, dict[str, str]] = {
    "viral_impact": {
        "font": "Arial Black",
        "highlight": "&H0000FFFF",  # ASS BGR Yellow
        "base": "&H00FFFFFF",       # ASS BGR White
        "outline": "3.0",
        "shadow": "1.5",
    },
    "neon_cyber": {
        "font": "Segoe UI Black",
        "highlight": "&H00FFFF00",  # ASS BGR Cyan
        "base": "&H00FF00FF",       # ASS BGR Magenta
        "outline": "3.5",
        "shadow": "2.0",
    },
    "clean_minimal": {
        "font": "Arial",
        "highlight": "&H00FFFFFF",  # ASS BGR Pure White
        "base": "&H00D0D0D0",       # Light Gray
        "outline": "2.0",
        "shadow": "0.0",
    },
    "podcast_gold": {
        "font": "Trebuchet MS",
        "highlight": "&H0000D7FF",  # ASS BGR Gold
        "base": "&H00FFFFFF",       # White
        "outline": "3.0",
        "shadow": "1.0",
    },
    "headline_box": {
        "font": "Impact",
        "highlight": "&H000000FF",  # ASS BGR Bright Red
        "base": "&H0000FFFF",       # ASS BGR Yellow
        "outline": "4.0",
        "shadow": "2.0",
    },
}


def _add_auto_emoji(word: str) -> str:
    """Append a matching emoji to key words if auto-emoji is enabled."""
    clean = re.sub(r"[^\w]", "", word).lower()
    emoji = EMOJI_MAP.get(clean)
    return f"{word} {emoji}" if emoji else word


#: How long a finished line may stay on screen after its last word ends,
#: waiting for the next line. Long enough to bridge a breath, short enough
#: that a real pause does not leave stale text sitting there.
_HOLD_MAX_S = 1.6


def _pop_events(groups: list[list[tuple[float, float, str]]],
                abs_start: float, per_line: int, max_lines: int,
                highlight: str, emphasis_colour: str) -> list[str]:
    """Word-by-word POP captions — one event per spoken word, NO GAPS.

    Karaoke ``\\k`` can only recolour; it cannot scale, and scale is what
    makes short-form captions feel alive. So each word gets its own event
    showing the whole line, with the active word scaled up and coloured and
    the rest resting. ``\\t`` runs the transform: overshoot to 124% in 90 ms,
    settle to 100% by 200 ms — a spring, not a linear ramp.

    CONTINUITY is the part that took a measurement to get right. Clamping
    each event to its own word's duration left the screen blank through
    every pause between words: measured 19 gaps totalling 15.3 s on a 59.5 s
    clip — **26% of the clip with no captions at all**, the largest hole
    2.49 s. Captions that blink out mid-sentence are exactly what makes
    auto-captioning look automatic. So each event now runs until the NEXT
    word begins, and the last word of a line holds (capped by
    ``_HOLD_MAX_S``) until the next line starts. The highlight still lands
    on the syllable; only the vanishing is gone.
    """
    out: list[str] = []
    # Flatten to find each word's successor across group boundaries.
    flat: list[tuple[int, int, float]] = []   # (group_idx, word_idx, start)
    for gi, group in enumerate(groups):
        for wi, (w_start, _e, _t) in enumerate(group):
            flat.append((gi, wi, w_start))
    next_start = {(gi, wi): flat[k + 1][2] if k + 1 < len(flat) else None
                  for k, (gi, wi, _s) in enumerate(flat)}

    for gi, group in enumerate(groups):
        if not group:
            continue
        for idx, (w_start, w_end, _text) in enumerate(group):
            start_cs = _cs(w_start - abs_start)
            nxt = next_start.get((gi, idx))
            if nxt is not None and idx + 1 < len(group):
                # Mid-line: run right up to the next word in this line.
                end_cs = _cs(nxt - abs_start)
            elif nxt is not None:
                # Line's last word: hold until the next line, but not past
                # the cap — a long silence should clear the screen.
                end_cs = _cs(min(nxt, w_end + _HOLD_MAX_S) - abs_start)
            else:
                # Final word of the clip: its own duration plus a short tail.
                end_cs = _cs(w_end + 0.3 - abs_start)
            end_cs = max(start_cs + 1, end_cs)
            rendered: list[str] = []
            for j, (_s, _e, txt) in enumerate(group):
                if j and j % per_line == 0:
                    rendered.append("\\N")
                elif j:
                    rendered.append(" ")
                if j == idx:
                    colour = (emphasis_colour if _EMPHASIS.search(txt)
                              else highlight)
                    # Overshoot then settle, and lift the active word above
                    # its neighbours so the scale-up never clips them.
                    rendered.append(
                        f"{{\\1c{colour}\\fscx118\\fscy118"
                        f"\\t(0,90,\\fscx124\\fscy124)"
                        f"\\t(90,200,\\fscx100\\fscy100)}}{txt}{{\\r}}")
                else:
                    rendered.append(txt)
            out.append(f"Dialogue: 0,{_fmt(start_cs)},{_fmt(end_cs)},"
                       f"Karaoke,,0,0,0,,{''.join(rendered)}")
    return out


class S5Subtitles(Stage[SubtitleArtifact]):
    name = "s5_subtitles"
    version = "5"
    artifact_type = SubtitleArtifact

    def _execute(self, *, cache_key: str, params: dict[str, Any],
                 **inputs: Any) -> SubtitleArtifact:
        transcript = inputs.get("transcript_artifact")
        campath = inputs.get("campath_artifact")
        if transcript is None or campath is None:
            raise FatalStageError(
                "S5 requires transcript_artifact and campath_artifact",
                stage=self.name)

        clip_start = float(campath.clip_start)
        clip_end = float(campath.clip_end)
        if clip_end <= clip_start:
            raise FatalStageError(
                f"S5: empty clip window [{clip_start}, {clip_end}]",
                stage=self.name)

        # TIME BASES, explicitly: the camera path's window is MEDIA-relative
        # (it is what S4 seeks with and S6 passes to -ss), while transcript
        # word times are ABSOLUTE stream time (T1). They coincide only when
        # abs_offset_s is 0 — true for a standalone file, false for every
        # chunk after the first of a recorded stream. Convert here, once.
        abs_offset = float(getattr(transcript, "abs_offset_s", 0.0) or 0.0)
        abs_start = clip_start + abs_offset
        abs_end = clip_end + abs_offset

        theme_name = str(params.get("theme", "")).strip()
        theme_cfg = SUBTITLE_THEMES.get(theme_name, {})

        font = str(params.get("font", theme_cfg.get("font", "Arial Black")))
        font_size = int(params.get("font_size", 72))
        highlight = str(params.get("highlight_color", theme_cfg.get("highlight", "&H0000FFFF")))
        base = str(params.get("base_color", theme_cfg.get("base", "&H00FFFFFF")))
        outline = float(params.get("outline", theme_cfg.get("outline", 3.0)))
        shadow = float(params.get("shadow", theme_cfg.get("shadow", 1.0)))
        margin_v = int(params.get("margin_v", 260))
        per_line = max(1, int(params.get("max_words_per_line", 4)))
        max_lines = max(1, int(params.get("max_lines", 2)))
        width = int(params.get("width", 1080))
        height = int(params.get("height", 1920))
        uppercase = bool(params.get("uppercase", False))
        animation = str(params.get("animation", "karaoke"))
        hook_text = str(params.get("hook_text", "") or "").strip()
        hook_seconds = float(params.get("hook_seconds", 3.0))
        auto_emojis = bool(params.get("auto_emojis", True))

        # Words inside the window, in absolute stream time. A word is kept if
        # it OVERLAPS the window rather than being strictly contained: the
        # sketch's `start >= clip_start and end <= clip_end` silently dropped
        # the words straddling either edge, which are exactly the ones a
        # viewer hears first and last.
        # Jump-cut pacing: when the CLI computed keep-intervals, every word
        # is retimed onto the COMPRESSED timeline here, before grouping —
        # the continuity and pop logic then operate on compressed times and
        # need no other changes. The intervals ride in params, so different
        # cuts hash to different artifacts.
        keeps = [(float(a), float(b))
                 for a, b in params.get("keep_intervals", [])]
        tmap = None
        if keeps:
            from clipforge.pacing import TimeMap  # noqa: PLC0415
            tmap = TimeMap(keeps)

        words: list[tuple[float, float, str]] = []
        for seg in transcript.segments:
            for w in seg.words:
                if w.start is None or w.end is None:
                    continue
                if float(w.end) <= abs_start or float(w.start) >= abs_end:
                    continue
                text = _escape(str(w.text))
                if auto_emojis:
                    text = _add_auto_emoji(text)
                if uppercase:
                    text = text.upper()
                if not text:
                    continue
                w_start, w_end = float(w.start), float(w.end)
                if tmap is not None:
                    rel_s = tmap.to_compressed(w_start - abs_start)
                    rel_e = max(rel_s + 0.02,
                                tmap.to_compressed(w_end - abs_start))
                    w_start, w_end = abs_start + rel_s, abs_start + rel_e
                words.append((w_start, w_end, text))
        words.sort(key=lambda t: (t[0], t[1]))

        # Group into lines of at most `per_line` words, then into events of at
        # most `max_lines` lines.
        groups = [words[i:i + per_line * max_lines]
                  for i in range(0, len(words), per_line * max_lines)]

        events: list[str] = []
        if animation == "pop":
            events = _pop_events(groups, abs_start, per_line, max_lines,
                                 highlight, _EMPHASIS_COLOUR)
        for group in (() if animation == "pop" else groups):
            if not group:
                continue
            line_start_cs = _cs(group[0][0] - abs_start)
            line_end_cs = _cs(group[-1][1] - abs_start)
            if line_end_cs <= line_start_cs:
                line_end_cs = line_start_cs + 1

            # Karaoke is a cumulative timeline from the EVENT's start. Track
            # the cursor so a pause between words becomes an unhighlighted
            # spacer instead of vanishing (which would slide every subsequent
            # highlight earlier than the audio).
            cursor = line_start_cs
            parts: list[str] = []
            for idx, (w_start, w_end, text) in enumerate(group):
                start_cs = max(cursor, _cs(w_start - abs_start))
                gap = start_cs - cursor
                if gap > 0:
                    parts.append(f"{{\\k{gap}}}")
                dur = max(1, _cs(w_end - abs_start) - start_cs)
                sep = "" if idx == len(group) - 1 else " "
                # Emphasis: amounts and numbers flip to green when sung
                # instead of the yellow highlight. Inline \1c changes the
                # POST-highlight colour only, so the resting state stays
                # uniform white and the pop lands exactly on the beat.
                if _EMPHASIS.search(text):
                    parts.append(f"{{\\k{dur}\\1c{_EMPHASIS_COLOUR}}}"
                                 f"{text}{sep}")
                else:
                    parts.append(f"{{\\k{dur}}}{text}{sep}")
                cursor = start_cs + dur

            # The event must outlive its karaoke timeline, or the tail words
            # never highlight.
            line_end_cs = max(line_end_cs, cursor)
            body = "".join(parts)
            if max_lines > 1 and len(group) > per_line:
                # Break into rendered lines at the per_line boundary.
                broken: list[str] = []
                count = 0
                for token in parts:
                    broken.append(token)
                    if token.startswith("{\\k") and not token.endswith("}"):
                        count += 1
                        if count % per_line == 0 and count < len(group):
                            broken.append("\\N")
                body = "".join(broken)
            events.append(
                f"Dialogue: 0,{_fmt(line_start_cs)},{_fmt(line_end_cs)},"
                f"Karaoke,,0,0,0,,{body}")

        # PrimaryColour is the colour a word becomes ONCE highlighted;
        # SecondaryColour is its colour beforehand. The sketch had these the
        # wrong way round, so the "highlight" was what the text started as.
        header = [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour,"
            " OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut,"
            " ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow,"
            " Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Karaoke,{font},{font_size},{highlight},{base},"
            f"&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline:g},{shadow:g},2,60,60,{margin_v},1",
            # Hook headline: top-anchored (alignment 8), larger than the
            # karaoke line, yellow-on-black. The first seconds decide
            # whether a viewer stays; a silent visual start with no framing
            # text is one of the things that made the output read as raw.
            f"Style: Hook,{font},{int(font_size * 1.25)},{highlight},"
            f"{highlight},&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,"
            f"{outline + 1:g},{shadow:g},8,60,60,180,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR,"
            " MarginV, Effect, Text",
        ]
        # Hook headline event: the clip's opening seconds carry a large
        # top-anchored title while the karaoke runs below. Kept out of the
        # karaoke timeline entirely — it is a static card, not a sung line.
        if hook_text:
            hook = _escape(hook_text.upper() if uppercase else hook_text)
            hook_words = hook.split()
            if len(hook_words) > 5:
                hook = (" ".join(hook_words[:5]) + "\\N"
                        + " ".join(hook_words[5:10]))
            effective_dur = (tmap.duration() if tmap is not None
                             else clip_end - clip_start)
            hook_end = _cs(min(hook_seconds, effective_dur))
            events.insert(0, (f"Dialogue: 1,{_fmt(0)},{_fmt(hook_end)},"
                              f"Hook,,0,0,0,,{hook}"))

        # "\n" explicitly, never os.linesep: the Determinism Law says the same
        # input produces the same BYTES, and libass accepts LF everywhere.
        content = "\n".join(header + events) + "\n"

        out_dir = self.artifacts_dir / self.name
        out_dir.mkdir(parents=True, exist_ok=True)
        ass_path = out_dir / f"{cache_key}.ass"
        atomic_write_text(ass_path, content)
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()

        log.info("s5.subtitles_written", path=str(ass_path),
                 lines=len(events), words=len(words))
        return SubtitleArtifact(
            cache_key=cache_key, stage=self.name,
            source_campath=campath.cache_key,
            ass_path=str(ass_path.resolve()),
            clip_start=clip_start, clip_end=clip_end,
            line_count=len(events), word_count=len(words),
            ass_sha256=digest)
