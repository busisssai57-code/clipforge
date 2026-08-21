"""Screenplay in, shots out — structured beats instead of guessed ones.

`split_into_beats` currently cuts a brief on sentence boundaries and hopes
they land where the shots should. That is fine for a one-line idea and
wrong for anything written: the moment a brief has a line of dialogue or a
scene change, punctuation stops predicting where a shot begins.

This parses Fountain — the plain-text screenplay convention — so the
BLOCKS the writer typed are the beats the generator gets. A scene heading
starts a shot, the action under it describes the picture, and the dialogue
under it is what gets spoken. Nothing is inferred from full stops.

Fountain rather than a bespoke syntax because it is what writers already
use, it is plain text (so it diffs, greps and survives a copy-paste), and
a screenplay written elsewhere pastes in and works.

Deliberately not a model call. It runs before a provider is chosen and
must behave identically for every one of them — the same reason
`split_into_beats` says about itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from clipforge.log import get_logger

log = get_logger(__name__)

BlockType = Literal["heading", "action", "character", "parenthetical",
                    "dialogue", "transition", "note"]

#: A scene heading. Fountain also allows forcing one with a leading dot.
_HEADING_RE = re.compile(
    r"^(INT\.?|EXT\.?|EST\.?|INT\.?/EXT\.?|I/E\.?)[\s.]", re.I)
#: A transition. Fountain: right-aligned, ends in "TO:", or forced with >.
_TRANSITION_RE = re.compile(r"^[A-Z0-9 ]+TO:$")
#: A character cue: upper-case, optionally with a (V.O.)-style extension.
#: Requires a letter so that "..." or "123" is action, not a character.
_CHARACTER_RE = re.compile(r"^[A-Z][A-Z0-9 .'\-]*(\s*\([^)]+\))?$")
#: Fountain's note syntax. Notes are the writer talking to production, not
#: to the camera, so they must never reach a picture prompt — before this
#: they were parsed as action and a "[[laugh here]]" landed in the frame
#: as text. Non-greedy so two notes on one line stay two notes.
_NOTE_RE = re.compile(r"\[\[(.+?)\]\]", re.S)


@dataclass
class Block:
    """One parsed line of screenplay."""

    type: BlockType
    text: str
    line_no: int
    #: Set on dialogue and parenthetical blocks: who is speaking.
    character: str | None = None
    #: Fountain notes found on this line, stripped out of ``text``.
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"type": self.type, "text": self.text, "line_no": self.line_no,
                "character": self.character, "notes": list(self.notes)}


@dataclass
class Shot:
    """One generated shot, assembled from the blocks under a heading."""

    heading: str = ""
    action: list[str] = field(default_factory=list)
    #: [(character, line)] in the order spoken.
    dialogue: list[tuple[str, str]] = field(default_factory=list)
    #: Notes written anywhere in this scene. A note that is nothing but
    #: emoji is a punchline mark the post layer stamps on the shot; the
    #: rest are production comments and go no further than here.
    notes: list[str] = field(default_factory=list)

    def beat(self) -> str:
        """The text handed to the prompt builder for this shot.

        Heading and action only. Dialogue is deliberately EXCLUDED: it is
        what the characters say, not what the camera sees, and feeding
        spoken lines to a video model puts subtitles and mouth-shaped
        artefacts in the frame. It travels separately, to the voice.
        """
        parts = [self.heading] if self.heading else []
        parts.extend(self.action)
        return " ".join(p.strip() for p in parts if p.strip())

    def spoken(self) -> str:
        return " ".join(line for _who, line in self.dialogue)

    def as_dict(self) -> dict:
        return {"heading": self.heading, "action": list(self.action),
                "dialogue": [{"character": c, "line": l}
                             for c, l in self.dialogue],
                "beat": self.beat()}


def _is_character(line: str, next_line: str | None) -> bool:
    """A character cue is upper-case AND followed by something to say.

    The second half matters: an all-caps action line ("THE DOOR SLAMS")
    is not a character cue, and treating it as one silently swallows the
    next line of action as dialogue.
    """
    if not line or line != line.upper():
        return False
    if not _CHARACTER_RE.match(line):
        return False
    if _HEADING_RE.match(line) or _TRANSITION_RE.match(line):
        return False
    return bool(next_line and next_line.strip())


#: A title-page key. Fountain's title page is `Key: Value` lines at the
#: very top, ending at the first blank line.
_TITLE_KEY_RE = re.compile(r"^([A-Za-z][A-Za-z ]{0,30}):\s*(.*)$")
#: The keys Fountain names. `hook` is this project's own addition: the
#: hook card is part of the piece, so the writer should be able to put
#: it in the script rather than remember a command-line flag.
_TITLE_KEYS = frozenset({
    "title", "credit", "author", "authors", "source",
    "draft date", "date", "contact", "copyright", "notes",
    "revision", "hook"})


def title_page(text: str) -> dict[str, str]:
    """The Fountain title page, or an empty dict.

    Recognised ONLY at the very top and only while every line is a key.
    A screenplay whose first line is action stays a screenplay, and a
    colon further down is dialogue punctuation, not a title key.
    """
    page: dict[str, str] = {}
    last: str | None = None
    for raw in (text or "").replace("\r\n", "\n").split("\n"):
        line = raw.rstrip()
        if not line.strip():
            break
        m = _TITLE_KEY_RE.match(line.strip())
        if m:
            last = m.group(1).strip().lower()
            page[last] = m.group(2).strip()
            continue
        if last is not None and raw.startswith((" ", "\t")):
            # Fountain allows a value to continue on indented lines.
            page[last] = f"{page[last]} {line.strip()}".strip()
            continue
        return {}
    # At least one CANONICAL key, or this is not a title page.
    # Without this a script opening on "GEEL: waa imisa" would
    # have its first line swallowed as a header - a made-up key is
    # legal Fountain, but only alongside a real one.
    return page if set(page) & _TITLE_KEYS else {}


def parse(text: str) -> list[Block]:
    """Parse Fountain-ish screenplay text into typed blocks.

    A title page is consumed here rather than parsed as action. Before
    this, `Title: ...` / `Credit: ...` at the top of a real screenplay
    became the FIRST SHOT - a generated picture of the words on the
    file's own header, and one of the twelve generations paid for it.
    """
    lines = (text or "").replace("\r\n", "\n").split("\n")
    if title_page(text):
        # Blank out the title page in place. Blanking rather than
        # slicing keeps every block's line_no pointing at the line the
        # writer sees in their editor.
        for i, raw in enumerate(lines):
            if not raw.strip():
                break
            lines[i] = ""
    blocks: list[Block] = []
    speaker: str | None = None

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        # Notes come off FIRST, before any line is classified. A note on a
        # character cue line would otherwise stop it being upper-case and
        # turn the speech under it into action.
        notes = [n.strip() for n in _NOTE_RE.findall(line) if n.strip()]
        if notes:
            line = _NOTE_RE.sub(" ", line).rstrip()
        stripped = line.strip()
        if not stripped:
            if notes:
                # A note on its own line still belongs to the scene, so it
                # is carried on a block of its own rather than dropped.
                blocks.append(Block("note", "", i, notes=notes))
            # A blank line ends a speech. Without this, action following
            # dialogue is attributed to whoever spoke last.
            speaker = None
            continue

        # Forced types, Fountain's escape hatches.
        if stripped.startswith(".") and not stripped.startswith(".."):
            blocks.append(Block("heading", stripped[1:].strip(), i,
                                notes=notes))
            speaker = None
            continue
        if stripped.startswith(">"):
            blocks.append(Block("transition", stripped[1:].strip(), i,
                                notes=notes))
            speaker = None
            continue

        if _HEADING_RE.match(stripped):
            blocks.append(Block("heading", stripped, i, notes=notes))
            speaker = None
            continue
        if _TRANSITION_RE.match(stripped):
            blocks.append(Block("transition", stripped, i, notes=notes))
            speaker = None
            continue

        if speaker is not None:
            if stripped.startswith("(") and stripped.endswith(")"):
                blocks.append(Block("parenthetical", stripped, i,
                                    character=speaker, notes=notes))
                continue
            blocks.append(Block("dialogue", stripped, i, character=speaker,
                                notes=notes))
            continue

        nxt = lines[i + 1].strip() if i + 1 < len(lines) else None
        if _is_character(stripped, nxt):
            # The cue may carry an extension: BOB (V.O.) speaks as BOB.
            name = re.sub(r"\s*\([^)]*\)\s*$", "", stripped).strip()
            blocks.append(Block("character", stripped, i, character=name,
                                notes=notes))
            speaker = name
            continue

        blocks.append(Block("action", stripped, i, notes=notes))

    return blocks


def to_shots(blocks: Iterable[Block]) -> list[Shot]:
    """Group blocks into shots. A heading starts one.

    Text before the first heading is still a shot — a writer who types
    three action lines and no heading has written one scene, and
    discarding it because it lacks a slugline would silently drop their
    whole brief.
    """
    shots: list[Shot] = []
    current: Shot | None = None

    for b in blocks:
        if b.type == "heading":
            current = Shot(heading=b.text)
            shots.append(current)
            current.notes.extend(b.notes)
            continue
        if current is None:
            current = Shot()
            shots.append(current)
        current.notes.extend(b.notes)
        if b.type == "action":
            current.action.append(b.text)
        elif b.type == "dialogue":
            current.dialogue.append((b.character or "", b.text))
        # character/parenthetical/transition carry no picture of their own:
        # the cue is bookkeeping and the transition is an edit, not a shot.

    return [s for s in shots if s.beat() or s.dialogue]


#: A note that is nothing but emoji (and whitespace) is a punchline mark.
#: Anything with a letter in it is a production comment — "[[check the
#: bead colours]]" must not become a sticker burnt into the picture.
_WORD_RE = re.compile(r"[0-9A-Za-z]")


def stickers(shot: Shot) -> list[str]:
    """The emoji this shot is marked with, in the order written.

    Reading them off the SCRIPT rather than a separate list is the point:
    the writer marks the punchline where they wrote it, and the mark
    cannot drift out of sync with the beat it belongs to.
    """
    out: list[str] = []
    for note in shot.notes:
        if not note.strip() or _WORD_RE.search(note):
            continue
        out.extend(ch for ch in note if not ch.isspace())
    return out


def to_beats(text: str, shots: int | None = None) -> list[str]:
    """Screenplay text to per-shot beats, ready for the prompt builder."""
    return [beat for beat, _marks in to_beats_with_marks(text, shots)]


def to_beats_with_marks(
        text: str, shots: int | None = None) -> list[tuple[str, list[str]]]:
    """Beats and their punchline marks, redistributed together.

    When ``shots`` is given the scenes are redistributed to match it —
    the operator's shot count wins over the writer's scene count, because
    the generator bills per shot and the picker is where that is decided.

    Marks travel WITH the beat through that redistribution rather than
    being matched up afterwards by index. Two scenes merged into one shot
    carry both their emoji; the alternative is a second distribution
    algorithm that has to stay in step with this one forever, and the
    symptom of it drifting would be a punchline stamped on the wrong shot.
    """
    parsed = [s for s in to_shots(parse(text)) if s.beat()]
    pairs = [(s.beat(), stickers(s)) for s in parsed]
    if not pairs:
        return []
    if not shots or shots == len(pairs):
        return pairs
    if shots < len(pairs):
        # Merge neighbours rather than truncating: dropping the tail of a
        # script silently loses the ending.
        out: list[tuple[str, list[str]]] = []
        per = len(pairs) / float(shots)
        for i in range(shots):
            lo, hi = int(round(i * per)), int(round((i + 1) * per))
            group = pairs[lo:max(hi, lo + 1)]
            out.append((" ".join(b for b, _ in group),
                        [m for _, marks in group for m in marks]))
        return out
    # More shots than scenes: repeat the last beat rather than inventing
    # material. The position hint in the prompt differentiates them, which
    # is what split_into_beats does in the same situation. The mark is NOT
    # repeated with it — one punchline is one punchline, and stamping the
    # same emoji on four shots is how a joke stops being one.
    out = list(pairs)
    while len(out) < shots:
        out.append((pairs[-1][0], []))
    return out


def synopsis(text: str, limit: int = 320) -> str:
    """The whole piece as PICTURE only, for a shot's "part of:" context.

    `build_shot_prompt` appends the brief to every shot so a model knows
    what the piece is. In screenplay mode that brief is the raw script -
    so the beat carefully excluded the dialogue and the next line put the
    entire screenplay, speech included, back into the prompt. The unit
    test that guarded this asserted on the BEAT, one layer below where
    the leak was.

    Truncated because this is context, not the subject: a 2,000-token
    script pasted into every shot buries the beat the shot is about.
    """
    beats = [s.beat() for s in to_shots(parse(text)) if s.beat()]
    joined = " ".join(beats)
    if len(joined) <= limit:
        return joined
    return joined[:limit].rsplit(" ", 1)[0] + "..."


def summary(text: str) -> dict:
    """Counts for the editor's status line."""
    blocks = parse(text)
    shots = to_shots(blocks)
    by_type: dict[str, int] = {}
    for b in blocks:
        by_type[b.type] = by_type.get(b.type, 0) + 1
    speakers = sorted({b.character for b in blocks
                       if b.type == "dialogue" and b.character})
    return {
        "blocks": [b.as_dict() for b in blocks],
        "shots": [s.as_dict() for s in shots],
        "counts": by_type,
        "speakers": speakers,
        "shot_count": len(shots),
        "word_count": len((text or "").split()),
    }
