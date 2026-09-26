"""Everything needed to post a clip, written next to it as files.

This is what exists INSTEAD of auto-publishing. The pipeline's standing
rule is that it produces files and stops: posting needs platform
credentials and acts outside this machine on the operator's behalf, which
is a decision a person should make per clip rather than a program make in
a loop. So this assembles the whole payload — caption, hashtags, a
thumbnail frame, chapters, per-platform text within each platform's real
limits — and leaves it on disk for one paste.

Nothing here contacts a network.

Two things it refuses to do, both because the failure would be invisible:
it will not silently truncate a caption past a platform limit (it trims
at a word boundary and says it trimmed), and it will not emit a thumbnail
from a frame that carries no picture.
"""

from __future__ import annotations

import os
import re
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.errors import ClipForgeError
from clipforge.log import get_logger
from clipforge.paths import atomic_write_json

log = get_logger(__name__)

#: Real caption limits, so text is fitted rather than rejected by the
#: platform after the operator has already pasted it.
PLATFORM_LIMITS: dict[str, int] = {
    "tiktok": 2200,
    "instagram": 2200,
    "youtube": 5000,
    "x": 280,
    "linkedin": 3000,
}

#: Hashtag counts that read as intentional rather than spammy.
PLATFORM_TAGS: dict[str, int] = {
    "tiktok": 5, "instagram": 8, "youtube": 6, "x": 3, "linkedin": 4,
}

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")

#: Words that carry no topical signal. Kept small and obvious — an
#: aggressive list would strip the words a niche is actually about.
_STOP = frozenset("""
the and for that with this you your are was were what when will would
have has had not but they them their there here from into out about
just like really very much more most some any all can could should
into over under then than only also even because while during
""".split())


@dataclass
class ExportPack:
    clip: Path
    title: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    chapters: list[dict[str, object]] = field(default_factory=list)
    thumbnail: Path | None = None
    platforms: dict[str, dict[str, object]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "clip": self.clip.name,
            "title": self.title,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "chapters": self.chapters,
            "thumbnail": self.thumbnail.name if self.thumbnail else None,
            "platforms": self.platforms,
            "notes": self.notes,
            "status": "DRAFT — nothing was posted. Copy and paste to publish.",
        }


def keywords(text: str, limit: int = 8) -> list[str]:
    """Topical words, most frequent first.

    Deliberately dumb and inspectable: frequency over a small stop list.
    A model could do better, but this runs with no GPU and its mistakes
    are the kind an operator can see and correct at a glance.
    """
    counts: dict[str, int] = {}
    for m in _WORD.finditer(text or ""):
        w = m.group(0).lower()
        if w in _STOP or len(w) < 4:
            continue
        counts[w] = counts.get(w, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [w for w, _ in ranked[:limit]]


def build_hashtags(text: str, niche_keywords: list[str] | None = None,
                   limit: int = 8) -> list[str]:
    """Hashtags from THIS clip's words, not a fixed house list.

    Every clip this project produced used to carry the same five
    hardcoded tags, which is worse than none — it marks the whole account
    as automated.
    """
    tags: list[str] = []
    for w in list(niche_keywords or []) + keywords(text, limit=limit * 2):
        tag = "#" + re.sub(r"[^a-z0-9]", "", str(w).lower())
        if len(tag) > 3 and tag not in tags:
            tags.append(tag)
        if len(tags) >= limit:
            break
    return tags


def fit_caption(text: str, limit: int) -> tuple[str, bool]:
    """Trim to ``limit`` at a WORD boundary. Returns (text, was_trimmed).

    Cutting mid-word produces a caption that looks like a bug to every
    viewer, and silently over-length text is rejected by the platform
    after the operator has already pasted it.
    """
    if limit < 1:
        raise ValueError("caption limit must be positive")
    text = (text or "").strip()
    if len(text) <= limit:
        return text, False
    # Reserve space for the ellipsis; it is part of the platform payload.
    cut = text[:limit - 1]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,.;:-") + "…", True


def chapters_from_segments(segments, *, max_chapters: int = 6,
                           min_gap_s: float = 4.0) -> list[dict[str, object]]:
    """Timestamped chapters from transcript segments."""
    out: list[dict[str, object]] = []
    last = -min_gap_s
    for seg in segments or []:
        start = float(getattr(seg, "start", 0.0) or 0.0)
        text = str(getattr(seg, "text", "") or "").strip()
        if not text or start - last < min_gap_s:
            continue
        label = text.split(".")[0].strip()
        if len(label) > 60:
            label = label[:57].rsplit(" ", 1)[0] + "…"
        out.append({"t": round(start, 2),
                    "time": f"{int(start // 60)}:{int(start % 60):02d}",
                    "label": label})
        last = start
        if len(out) >= max_chapters:
            break
    return out


def _atomic_replace(tmp: Path, dest: Path) -> None:
    """Move a fully-written temp file onto ``dest`` in one step.

    The same discipline S6 uses for the render itself (``.partial`` then
    replace) and for the same reason, one layer out: these sidecars are
    read by the dashboard on every four-second poll, so a write
    interrupted by a cancel or a crash must never be visible half-done.
    ``os.replace`` is atomic within a directory, which is why the temp is
    a sibling of the destination rather than in the system tmp.
    """
    os.replace(tmp, dest)


def _tmp_beside(dest: Path) -> Path:
    """A unique temp sibling of ``dest`` — same dir, so the replace is atomic.

    The destination's real suffix is preserved (``.partial`` sits in the
    middle, not at the end) because ffmpeg infers its output muxer from
    the extension: a ``.jpg.partial`` name makes it refuse to write, the
    same trap S6 documents for ``.mp4.partial``. ``os.replace`` ignores
    the extension, so this costs the JSON write nothing.
    """
    stem = dest.name[: -len(dest.suffix)] if dest.suffix else dest.name
    return dest.with_name(f".{stem}.{uuid.uuid4().hex[:8]}.partial{dest.suffix}")


def grab_thumbnail(clip: Path, dest: Path, *, at_s: float = 1.0) -> Path:
    """One frame, refusing to write a blank one.

    A thumbnail is the single most-seen frame of a clip; shipping a flat
    fill because the grab landed on a fade is the kind of silent failure
    this project keeps finding.
    """
    from clipforge.ffmpeg import require_binary

    clip, dest = Path(clip), Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Grab into a temp sibling and swap only once the frame is proven real.
    # Two things this buys, both correctness: a cancel mid-grab can no
    # longer leave a half-written .thumb.jpg for the gallery to render
    # broken, and a blank frame no longer DESTROYS a previously-good
    # thumbnail on its way to raising (the old code unlinked ``dest``).
    tmp = _tmp_beside(dest)
    try:
        proc = subprocess.run(
            [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner", "-y",
             "-ss", f"{max(0.0, at_s):.2f}", "-i", str(clip),
             "-frames:v", "1", "-q:v", "2", str(tmp)],
            capture_output=True, text=True, errors="replace", timeout=120)
        if proc.returncode != 0 or not tmp.is_file():
            raise ClipForgeError(
                f"thumbnail grab failed: {(proc.stderr or '')[-300:]}")

        try:
            import numpy as np
            from PIL import Image

            arr = np.asarray(Image.open(tmp).convert("L"), dtype=np.float32)
            if float(arr.std()) < 8.0:
                raise ClipForgeError(
                    f"thumbnail at {at_s:.1f}s has no picture (spread "
                    f"{float(arr.std()):.1f}); try a different timestamp")
        except ImportError:
            log.warning("export.thumbnail_unchecked",
                        note="numpy/Pillow missing; blank frame not verified")

        _atomic_replace(tmp, dest)
    finally:
        Path(tmp).unlink(missing_ok=True)
    return dest


def build_pack(clip: Path, *, title: str = "", transcript_text: str = "",
               segments=None, niche_keywords: list[str] | None = None,
               hook: str = "", thumbnail_at_s: float = 1.0,
               write: bool = True, max_hashtags: int = 8) -> ExportPack:
    """Assemble the full posting payload beside ``clip``."""
    clip = Path(clip)
    if not clip.is_file():
        raise ClipForgeError(f"no clip at {clip}")

    body = (hook or "").strip()
    if transcript_text and len(body) < 40:
        body = (body + " " + transcript_text.strip()).strip()
    # [editor] max_hashtags reaches this and only this: it was in
    # config.toml, documented, and read by nothing.
    tags = build_hashtags(transcript_text or title, niche_keywords,
                          limit=max_hashtags)

    pack = ExportPack(clip=clip, title=title.strip(), caption=body,
                      hashtags=tags,
                      chapters=chapters_from_segments(segments))

    try:
        pack.thumbnail = grab_thumbnail(
            clip, clip.with_suffix(".thumb.jpg"), at_s=thumbnail_at_s)
    except ClipForgeError as exc:
        pack.notes.append(f"no thumbnail: {exc}")

    for name, limit in PLATFORM_LIMITS.items():
        n_tags = PLATFORM_TAGS.get(name, 5)
        chosen = tags[:n_tags]
        full = (body + ("\n\n" + " ".join(chosen) if chosen else "")).strip()
        text, trimmed = fit_caption(full, limit)
        pack.platforms[name] = {
            "caption": text, "hashtags": chosen,
            "limit": limit, "trimmed": trimmed,
        }
        if trimmed:
            pack.notes.append(f"{name}: caption trimmed to {limit} chars")

    if write:
        dest = clip.with_suffix(".export.json")
        atomic_write_json(dest, pack.as_dict())
        log.info("export.pack_written", path=str(dest),
                 platforms=len(pack.platforms))
    return pack
