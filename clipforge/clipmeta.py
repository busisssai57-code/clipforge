"""Everything the UI knows about one rendered clip, assembled from disk.

The dashboard used to render ``clip.score``, ``clip.title`` and
``clip.duration_s`` from an endpoint that sent none of them, so the score
badge, the grade breakdown and the titles were dead markup — always
absent, never an error. :mod:`clipforge.scorecard` had the same problem
one level down: ``build_scorecard`` was written, tested and called from
nowhere. This module is the missing join.

Nothing here measures anything. Every field is read back from an artifact
a stage already wrote, and a field with no artifact behind it is absent
rather than estimated — the Authorization Law applies to reporting as
much as to rendering. ``provenance`` on the returned record names which
stages were found, so a UI can say *why* a number is missing instead of
showing a plausible zero.

**The join.** A clip's filename stem is its S6 cache key, and each stage
artifact names the one it consumed::

    <stem>.mp4
      └ s6_render/<stem>.json          duration, geometry, loudness
          └ source_subtitles → s5_subtitles/  clip_start, clip_end
              └ source_campath  → s4_tracking/   framing mode, camera path
                  └ source_ranking  → s3_semantic/  VL judgements
                      └ source_candidates → s2_prefilter/  heuristic scores
                          └ source_transcript → s1_transcribe/  word timings

That chain is exact — every hop is a recorded cache key, not a guess.
The one soft hop is the candidate index (S5 records the window, S2 records
the candidates, and the link between them is the window itself), so it is
matched on both edges with a tolerance and reported as unmatched when it
does not line up, rather than silently taking the nearest.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from clipforge.log import get_logger
from clipforge.paths import Workspace
from clipforge.scorecard import build_scorecard

log = get_logger(__name__)

#: A clip window is matched to its prefilter candidate by comparing both
#: edges. S5 stores clip_start/clip_end after snapping, so the two differ
#: by a fraction of a second; a whole second apart means a different
#: window and the candidate must not be claimed.
_WINDOW_TOLERANCE_S = 1.0

#: Sidecar renders share the clip's stem with a suffix (``.broll.mp4``,
#: ``.es.mp4``). They are their own gallery entries but resolve against
#: the SAME artifact chain — the stem to look up is the part before the
#: suffix, so a dub inherits its parent's transcript, score and QA rather
#: than showing up as an unscored orphan.
_SIDECAR_SUFFIXES = (".broll", ".draft", ".upscaled", ".vo")


def _language_suffixes() -> tuple[str, ...]:
    from clipforge.dubbing import LANGUAGES

    return tuple(f".{code}" for code, _ in LANGUAGES)


def _all_sidecar_suffixes() -> tuple[str, ...]:
    try:
        return _SIDECAR_SUFFIXES + _language_suffixes()
    except Exception:  # noqa: BLE001 - dubbing is optional
        return _SIDECAR_SUFFIXES


def _read(path: Path) -> dict[str, Any] | None:
    """Read one artifact, or None when it is absent or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        log.warning("clipmeta.artifact_unreadable", path=str(path),
                    error=str(exc)[:200])
        return None


def artifact_stem(filename: str) -> str:
    """The S6 cache key behind a clip filename, sidecars included.

    Suffixes are stripped REPEATEDLY, because sidecars compose: a voiceover
    over an upscale is ``<key>.upscaled.vo.mp4``, and removing one layer
    leaves ``<key>.upscaled``, which is not a cache key and has no artifact
    behind it — so the clip loses its score, transcript and QA and shows up
    in the gallery as an unscored orphan. Observed the first time the
    dashboard's voiceover button ran, because it acts on the newest clip
    and the newest clip was itself a sidecar.

    Bounded rather than ``while True``: a filename is attacker-adjacent
    (it arrives from a URL) and an unbounded strip loop on a crafted name
    is a needless hazard. Four layers is already more chaining than the
    pipeline can produce.
    """
    stem = Path(filename).stem
    suffixes = _all_sidecar_suffixes()
    for _ in range(4):
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        else:
            break
    return stem


def variant_of(filename: str) -> str | None:
    """``"broll"`` for ``x.broll.mp4``, ``"es"`` for a dub; None otherwise."""
    stem = Path(filename).stem
    for suffix in _all_sidecar_suffixes():
        if stem.endswith(suffix):
            return suffix.lstrip(".")
    return None


def variant_kind(variant: str | None) -> str | None:
    """``"dub"`` / ``"render"`` / None — what kind of sibling this is."""
    if not variant:
        return None
    try:
        from clipforge.dubbing import LANGUAGES

        if variant in {code for code, _ in LANGUAGES}:
            return "dub"
    except Exception:  # noqa: BLE001
        pass
    return "render"


# ------------------------------------------------------------------ record

@dataclass
class ClipMeta:
    """One clip, as much of it as the artifacts actually support."""

    filename: str
    stem: str
    rejected: bool = False
    variant: str | None = None
    size_mb: float = 0.0
    created_at: float = 0.0

    title: str = ""
    hook: str = ""
    caption: str = ""
    hashtags: list[str] = field(default_factory=list)
    platforms: dict[str, Any] = field(default_factory=dict)
    chapters: list[dict[str, Any]] = field(default_factory=list)

    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    encoder: str | None = None
    loudness_i: float | None = None
    loudness_tp: float | None = None

    #: Absolute position of this clip inside its source video.
    clip_start: float | None = None
    clip_end: float | None = None
    framing_mode: str | None = None
    source_width: int | None = None
    source_height: int | None = None
    #: The file this clip was cut from, recorded by S1. Without it the
    #: dashboard cannot offer "re-run with different settings" — that is
    #: the one action that turns every greyed-out control into a real one,
    #: because the pipeline is what actually applies them.
    source_path: str | None = None
    source_exists: bool = False

    score: dict[str, Any] | None = None
    justification: str = ""
    qa: dict[str, Any] | None = None
    has_thumb: bool = False
    has_export: bool = False
    #: Language codes with a translated .srt beside this clip.
    subtitle_tracks: list[str] = field(default_factory=list)
    #: Which stages backed this record — the UI explains gaps with it.
    provenance: dict[str, bool] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        q = f"?rejected=1" if self.rejected else ""
        return {
            "filename": self.filename,
            "stem": self.stem,
            "rejected": self.rejected,
            "variant": self.variant,
            "variant_kind": variant_kind(self.variant),
            "subtitle_tracks": self.subtitle_tracks,
            "size_mb": round(self.size_mb, 2),
            "created_at": self.created_at,
            "url": f"/api/clips/stream/{self.filename}{q}",
            "thumb_url": (f"/api/clips/thumb/{self.filename}{q}"
                          if self.has_thumb else None),
            "title": self.title,
            "hook": self.hook,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "chapters": self.chapters,
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
            "encoder": self.encoder,
            "loudness_i": self.loudness_i,
            "loudness_tp": self.loudness_tp,
            "clip_start": self.clip_start,
            "clip_end": self.clip_end,
            "framing_mode": self.framing_mode,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_path": self.source_path,
            "source_exists": self.source_exists,
            "score": self.score,
            "justification": self.justification,
            "qa": self.qa,
            "has_export": self.has_export,
            "provenance": self.provenance,
        }


# ------------------------------------------------------------------- chain

@dataclass
class _Chain:
    """The artifacts behind one clip, each None when it was not found."""

    s6: dict[str, Any] | None = None
    s5: dict[str, Any] | None = None
    s4: dict[str, Any] | None = None
    s3: dict[str, Any] | None = None
    s2: dict[str, Any] | None = None
    s1: dict[str, Any] | None = None
    qa: dict[str, Any] | None = None
    editor: dict[str, Any] | None = None
    candidate: dict[str, Any] | None = None
    candidate_index: int | None = None
    ranked_item: dict[str, Any] | None = None


class _Scan:
    """One pass over the artifact tree, so the gallery is linear again.

    Two things made :func:`list_clips` quadratic, and both are invisible
    from one clip's point of view:

    * **S7 and the editor pack are found by scanning.** Neither is keyed
      by the clip it describes, so ``_walk`` read and parsed every JSON
      in ``s7_qa`` and ``editor`` until it hit a match — per clip. With
      200 clips that is 200 directory sweeps of 200 files each.
    * **The chain is shared.** Ten clips cut from one video walk back to
      the same S3, S2 and S1 artifacts, and the transcript at the end of
      that chain is the biggest file in the workspace. Each clip was
      re-reading and re-parsing megabytes the previous one had just
      finished with, to pull one ``source_path`` string out of it.

    MEASURED on a 200-clip / 20-source workspace: 23,420 artifact reads
    per call, 0.79s — and the dashboard polls this every four seconds, so
    that is most of a core spent re-reading files that had not changed.
    Indexed and memoised: 1,260 reads, 0.095s.

    Deliberately per-call and short-lived. A cache that outlives the
    request would have to answer "is this artifact still current", and
    the honest answer needs a stat per file — which is most of the cost
    it was meant to save. One scan sees one consistent snapshot; the next
    poll takes a fresh one.
    """

    __slots__ = ("_blobs", "_qa", "_editor", "_arts", "_srt")

    def __init__(self, artifacts_root: Path) -> None:
        self._arts = Path(artifacts_root)
        self._blobs: dict[Path, dict[str, Any] | None] = {}
        self._qa: dict[str, dict[str, Any]] | None = None
        self._editor: dict[str, dict[str, Any]] | None = None
        self._srt: dict[Path, dict[str, set[str]]] = {}

    def read(self, path: Path) -> dict[str, Any] | None:
        """``_read``, but each path is parsed at most once per scan."""
        try:
            return self._blobs[path]
        except KeyError:
            blob = _read(path)
            self._blobs[path] = blob
            return blob

    def _index(self, subdir: str, key: str) -> dict[str, dict[str, Any]]:
        """Map ``key``'s value -> blob for every artifact in ``subdir``.

        Newest wins where two artifacts claim the same key: a re-run
        writes a second QA verdict naming the same clip, and the current
        one is the one the gallery should show. The previous code broke
        on the first glob hit, which is filesystem order — so which
        verdict won was genuinely arbitrary, and could change between two
        polls with nothing on disk having moved.
        """
        out: dict[str, dict[str, Any]] = {}
        directory = self._arts / subdir
        if not directory.is_dir():
            return out
        try:
            entries = sorted(directory.glob("*.json"),
                             key=lambda p: (_mtime(p), p.name))
        except OSError as exc:
            log.warning("clipmeta.index_failed", dir=str(directory),
                        error=str(exc)[:200])
            return out
        for path in entries:
            blob = self.read(path)
            if not blob:
                continue
            value = blob.get(key)
            if isinstance(value, str) and value:
                out[value] = blob
        return out

    def subtitle_tracks(self, clip: Path) -> list[str]:
        """Languages dubbed beside ``clip``, from one listing per folder.

        ``clips/*.srt`` was being globbed once per clip, and a glob lists
        the whole directory — so N clips in a folder cost N listings of N
        entries, for a lookup that is really one grouping pass.
        """
        folder = clip.parent
        index = self._srt.get(folder)
        if index is None:
            index = {}
            try:
                for path in folder.glob("*.srt"):
                    # <clip stem>.<lang>.srt — the stem can itself carry
                    # dots, so split from the right, never with .stem.
                    name = path.name[:-len(".srt")]
                    stem, _, lang = name.rpartition(".")
                    if stem and lang:
                        index.setdefault(stem, set()).add(lang)
            except OSError as exc:
                log.warning("clipmeta.srt_index_failed", dir=str(folder),
                            error=str(exc)[:200])
            self._srt[folder] = index
        return sorted(index.get(clip.name[:-len(clip.suffix)] or clip.name,
                                ()))

    def qa_for(self, stem: str) -> dict[str, Any] | None:
        if self._qa is None:
            self._qa = self._index("s7_qa", "source_clip")
        return self._qa.get(stem)

    def editor_for(self, candidate_id: str) -> dict[str, Any] | None:
        if self._editor is None:
            self._editor = self._index("editor", "candidate_id")
        return self._editor.get(candidate_id)


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _walk(ws: Workspace, stem: str, scan: _Scan | None = None) -> _Chain:
    """Follow the cache-key chain from a clip stem back to the transcript."""
    arts = Path(ws.artifacts)
    scan = scan if scan is not None else _Scan(arts)
    _read_one = scan.read
    chain = _Chain()

    chain.s6 = _read_one(arts / "s6_render" / f"{stem}.json")
    if chain.s6 is None:
        return chain

    sub_key = chain.s6.get("source_subtitles")
    if sub_key:
        chain.s5 = _read_one(arts / "s5_subtitles" / f"{sub_key}.json")
    if chain.s5:
        cam_key = chain.s5.get("source_campath")
        if cam_key:
            chain.s4 = _read_one(arts / "s4_tracking" / f"{cam_key}.json")
    if chain.s4:
        rank_key = chain.s4.get("source_ranking")
        if rank_key:
            chain.s3 = _read_one(arts / "s3_semantic" / f"{rank_key}.json")
    if chain.s3:
        cand_key = chain.s3.get("source_candidates")
        if cand_key:
            chain.s2 = _read_one(arts / "s2_prefilter" / f"{cand_key}.json")
    if chain.s2:
        tr_key = chain.s2.get("source_transcript")
        if tr_key:
            chain.s1 = _read_one(arts / "s1_transcribe" / f"{tr_key}.json")

    # QA is keyed by its own cache key and names the clip it judged, so
    # it cannot be found by following a pointer. It used to be found by
    # reading the whole s7_qa directory per clip; the scan indexes that
    # directory once instead — see :class:`_Scan`.
    chain.qa = scan.qa_for(stem)

    _match_candidate(chain)

    # The editor pack carries candidate_id but no pointer back to the
    # ranking that produced it — a real schema gap. With the candidate
    # index resolved above the id is exact; without it, no editor pack is
    # claimed rather than taking the most recent one and hoping.
    if chain.candidate_index is not None:
        chain.editor = scan.editor_for(f"cand_{chain.candidate_index:03d}")
    return chain


def _match_candidate(chain: _Chain) -> None:
    """Bind the clip window to the prefilter candidate that produced it.

    Both edges are compared. Matching on start alone would happily accept
    a candidate that begins at the same place and runs twice as long,
    which is exactly the confusion that makes a wrong score look right.
    """
    if not chain.s5 or not chain.s2:
        return
    try:
        want_start = float(chain.s5["clip_start"])
        want_end = float(chain.s5["clip_end"])
    except (KeyError, TypeError, ValueError):
        return

    best: tuple[float, int, dict[str, Any]] | None = None
    for idx, cand in enumerate(chain.s2.get("candidates") or []):
        try:
            delta = (abs(float(cand["start"]) - want_start)
                     + abs(float(cand["end"]) - want_end))
        except (KeyError, TypeError, ValueError):
            continue
        if best is None or delta < best[0]:
            best = (delta, idx, cand)

    if best is None or best[0] > _WINDOW_TOLERANCE_S * 2:
        log.info("clipmeta.candidate_unmatched",
                 clip_start=want_start, clip_end=want_end,
                 nearest_delta=(round(best[0], 3) if best else None))
        return

    chain.candidate_index = best[1]
    chain.candidate = best[2]
    if chain.s3:
        chain.ranked_item = next(
            (r for r in chain.s3.get("items") or []
             if r.get("candidate_index") == best[1]), None)


def _scorecard(chain: _Chain) -> dict[str, Any] | None:
    """The four-dimension breakdown, or None when nothing measured it."""
    if chain.candidate is None and chain.ranked_item is None:
        return None
    item = chain.ranked_item or {}
    scores = (chain.candidate or {}).get("scores") or {}
    if not scores and not item:
        return None
    card = build_scorecard(
        s2_scores=scores,
        visual_action=item.get("visual_action"),
        hook_strength=item.get("hook_strength"),
        comprehensibility=item.get("comprehensibility"),
        ranking_source=("vl" if (chain.s3 or {}).get("ranking_source")
                        in ("vl", "semantic") else "heuristic"),
    )
    return card.as_dict()


# ------------------------------------------------------------------ public

def resolve_clip(ws: Workspace, filename: str, *,
                 rejected: bool = False,
                 scan: "_Scan | None" = None) -> ClipMeta | None:
    """Assemble the full record for one clip file, or None if it is gone.

    ``scan`` lets a caller resolving many clips share one pass over the
    artifact tree; omitted, each call takes its own. See :class:`_Scan`.
    """
    root = (Path(ws.clips) / "rejected") if rejected else Path(ws.clips)
    path = root / filename
    if not path.is_file():
        return None

    scan = scan if scan is not None else _Scan(Path(ws.artifacts))
    stem = artifact_stem(filename)
    meta = ClipMeta(filename=filename, stem=stem, rejected=rejected,
                    variant=variant_of(filename))
    try:
        stat = path.stat()
        meta.size_mb = stat.st_size / (1024 * 1024)
        meta.created_at = stat.st_mtime
    except OSError:
        return None

    meta.has_thumb = path.with_suffix(".thumb.jpg").is_file()
    # Dub sidecars: <clip>.<lang>.srt beside the render, grouped once per
    # folder rather than globbed once per clip.
    meta.subtitle_tracks = scan.subtitle_tracks(path)

    # The export pack is the operator-facing copy: title, caption, tags and
    # chapters, already trimmed per platform. Prefer it over the editor
    # artifact because it is what the CLI actually shipped.
    pack = _read(path.with_suffix(".export.json"))
    if pack:
        meta.has_export = True
        meta.title = str(pack.get("title") or "")
        meta.caption = str(pack.get("caption") or "")
        meta.hashtags = [str(h) for h in (pack.get("hashtags") or [])]
        meta.platforms = pack.get("platforms") or {}
        meta.chapters = list(pack.get("chapters") or [])

    chain = _walk(ws, stem, scan)

    if chain.s6:
        meta.duration_s = chain.s6.get("duration_s")
        meta.width = chain.s6.get("width")
        meta.height = chain.s6.get("height")
        meta.encoder = chain.s6.get("encoder")
        meta.loudness_i = chain.s6.get("loudness_i")
        meta.loudness_tp = chain.s6.get("loudness_tp")
    if chain.s5:
        meta.clip_start = chain.s5.get("clip_start")
        meta.clip_end = chain.s5.get("clip_end")
    if chain.s4:
        meta.framing_mode = chain.s4.get("framing_mode")
        meta.source_width = chain.s4.get("src_width")
        meta.source_height = chain.s4.get("src_height")
    if chain.s1 and chain.s1.get("source_path"):
        meta.source_path = str(chain.s1["source_path"])
        # Recorded months ago on a machine whose disks may have moved. The
        # UI must not offer a re-run against a path that is gone.
        try:
            meta.source_exists = Path(meta.source_path).is_file()
        except OSError:
            meta.source_exists = False
    if chain.editor:
        meta.title = meta.title or str(chain.editor.get("title") or "")
        meta.hook = str(chain.editor.get("hook_text") or "")
    if chain.ranked_item:
        meta.title = meta.title or str(chain.ranked_item.get("title") or "")
        meta.hook = meta.hook or str(chain.ranked_item.get("hook") or "")
        meta.justification = str(chain.ranked_item.get("justification") or "")

    meta.score = _scorecard(chain)

    if chain.qa:
        checks = list(chain.qa.get("checks") or [])
        meta.qa = {
            "passed": bool(chain.qa.get("passed")),
            "failed_count": int(chain.qa.get("failed_count") or 0),
            "warned_count": int(chain.qa.get("warned_count") or 0),
            "checks": checks,
        }

    meta.provenance = {
        "render": chain.s6 is not None,
        "subtitles": chain.s5 is not None,
        "tracking": chain.s4 is not None,
        "ranking": chain.s3 is not None,
        "candidates": chain.s2 is not None,
        "transcript": chain.s1 is not None,
        "qa": chain.qa is not None,
        "editor": chain.editor is not None,
        "candidate_matched": chain.candidate_index is not None,
    }
    return meta


def list_clips(ws: Workspace) -> list[ClipMeta]:
    """Every clip on disk, newest first, quarantined ones flagged.

    One :class:`_Scan` covers the whole listing, which is what keeps this
    linear in the number of clips rather than quadratic — the dashboard
    polls it every four seconds, so the difference is a core.
    """
    scan = _Scan(Path(ws.artifacts))
    out: list[ClipMeta] = []
    for directory, rejected in ((Path(ws.clips), False),
                                (Path(ws.clips) / "rejected", True)):
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.mp4")):
            meta = resolve_clip(ws, path.name, rejected=rejected, scan=scan)
            if meta is not None:
                out.append(meta)
    out.sort(key=lambda m: m.created_at, reverse=True)
    return out


# -------------------------------------------------------------- transcript

def transcript_for(ws: Workspace, filename: str, *,
                   rejected: bool = False) -> dict[str, Any]:
    """Word-timed transcript for one clip, rebased to clip time.

    The editor's transcript panel needs three things the raw artifact does
    not directly give: times relative to the clip rather than the source,
    the silent gap between consecutive words (what the pause pills show),
    and the scene descriptions interleaved at their own timestamps.

    Returns ``{"available": False, "reason": ...}`` when a stage in the
    chain is missing. An empty word list would read as "this clip is
    silent", which is a different and wrong claim.
    """
    root = (Path(ws.clips) / "rejected") if rejected else Path(ws.clips)
    if not (root / filename).is_file():
        return {"available": False, "reason": "clip not found"}

    chain = _walk(ws, artifact_stem(filename))
    if chain.s1 is None:
        missing = next((n for n, ok in (("render", chain.s6), ("subtitles",
                        chain.s5), ("tracking", chain.s4), ("ranking",
                        chain.s3), ("candidates", chain.s2),
                        ("transcript", chain.s1)) if not ok), "transcript")
        return {"available": False,
                "reason": f"no {missing} artifact for this clip"}
    if chain.s5 is None:
        return {"available": False, "reason": "no subtitle artifact"}

    try:
        start = float(chain.s5["clip_start"])
        end = float(chain.s5["clip_end"])
    except (KeyError, TypeError, ValueError):
        return {"available": False, "reason": "subtitle artifact has no window"}

    # S1 times are chunk-relative; abs_offset_s puts them on the source
    # timeline, which is the timeline S5's window is expressed in.
    offset = float(chain.s1.get("abs_offset_s") or 0.0)

    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for seg in chain.s1.get("segments") or []:
        try:
            s_start = float(seg["start"]) + offset
            s_end = float(seg["end"]) + offset
        except (KeyError, TypeError, ValueError):
            continue
        if s_end < start or s_start > end:
            continue
        seg_words: list[dict[str, Any]] = []
        for w in seg.get("words") or []:
            try:
                w_start = float(w["start"]) + offset
                w_end = float(w["end"]) + offset
            except (KeyError, TypeError, ValueError):
                continue
            if w_end < start or w_start > end:
                continue
            entry = {
                "text": str(w.get("text") or ""),
                "start": round(w_start - start, 3),
                "end": round(w_end - start, 3),
                # Alignment confidence. Low-confidence words are the ones
                # worth eyeballing before publishing, so the UI can dim
                # them — but only because the aligner really reported it.
                "score": (float(w["score"]) if isinstance(
                    w.get("score"), (int, float)) else None),
                "speaker": w.get("speaker"),
            }
            seg_words.append(entry)
            words.append(entry)
        segments.append({
            "text": str(seg.get("text") or ""),
            "start": round(s_start - start, 3),
            "end": round(s_end - start, 3),
            "speaker": seg.get("speaker"),
            "words": seg_words,
        })

    # Gap to the PREVIOUS word, so the pill renders before the word that
    # follows the silence — the same place a person would read the pause.
    for prev, cur in zip(words, words[1:]):
        gap = cur["start"] - prev["end"]
        cur["gap_before"] = round(gap, 2) if gap >= 0.2 else None
    if words:
        words[0]["gap_before"] = None

    return {
        "available": True,
        "clip_start": start,
        "clip_end": end,
        "duration_s": round(end - start, 3),
        "language": chain.s1.get("language"),
        "diarized": bool(chain.s1.get("diarization_ok")),
        "word_count": len(words),
        "words": words,
        "segments": segments,
    }


def _fps_from(rational: Any) -> float | None:
    """Evaluate ``"24000/1001"``. Returns None rather than assuming 30."""
    if isinstance(rational, (int, float)) and rational > 0:
        return float(rational)
    if not isinstance(rational, str) or "/" not in rational:
        return None
    num, _, den = rational.partition("/")
    try:
        n, d = float(num), float(den)
    except ValueError:
        return None
    return n / d if d else None


#: The overlay interpolates between points, so a 1400-frame path does not
#: need to ship at full rate. 600 points is smooth at any preview size.
_CAMPATH_MAX_POINTS = 600


def campath_for(ws: Workspace, filename: str, *,
                rejected: bool = False) -> dict[str, Any]:
    """The S4 camera path — the crop the renderer actually followed.

    S4 stores one crop rectangle per source frame in source pixels. The
    overlay has to draw it over a preview of unknown size, so rectangles
    are normalised to fractions of the source frame here, and the frame
    index is converted to clip-relative seconds with the recorded fps.

    Without a recorded fps the frames cannot be placed on a timeline at
    all, so they are returned unplaced and flagged rather than timed
    against an assumed 30fps.
    """
    root = (Path(ws.clips) / "rejected") if rejected else Path(ws.clips)
    if not (root / filename).is_file():
        return {"available": False, "reason": "clip not found"}

    chain = _walk(ws, artifact_stem(filename))
    if chain.s4 is None:
        return {"available": False, "reason": "no tracking artifact"}

    src_w = float(chain.s4.get("src_width") or 0) or None
    src_h = float(chain.s4.get("src_height") or 0) or None
    fps = _fps_from(chain.s4.get("src_fps_rational"))

    raw = [f for f in (chain.s4.get("frames") or []) if isinstance(f, dict)]
    step = max(1, len(raw) // _CAMPATH_MAX_POINTS) if raw else 1
    frames: list[dict[str, Any]] = []
    for fr in raw[::step]:
        entry: dict[str, Any] = {"frame": fr.get("frame")}
        if fps and isinstance(fr.get("frame"), (int, float)):
            entry["t"] = round(float(fr["frame"]) / fps, 3)
        if src_w and src_h:
            try:
                entry["x"] = round(float(fr["x"]) / src_w, 5)
                entry["y"] = round(float(fr["y"]) / src_h, 5)
                entry["w"] = round(float(fr["w"]) / src_w, 5)
                entry["h"] = round(float(fr["h"]) / src_h, 5)
            except (KeyError, TypeError, ValueError):
                continue
        frames.append(entry)

    # Per-shot speaker assignment: which track the crop was following and
    # how confident S4 was about it.
    assignments = [a for a in (chain.s4.get("assignments") or [])
                   if isinstance(a, dict)]

    return {
        "available": True,
        "framing_mode": chain.s4.get("framing_mode"),
        "mar_shots": chain.s4.get("mar_shots"),
        "presence_shots": chain.s4.get("presence_shots"),
        "src_width": chain.s4.get("src_width"),
        "src_height": chain.s4.get("src_height"),
        "fps_rational": chain.s4.get("src_fps_rational"),
        "fps": round(fps, 4) if fps else None,
        "timed": fps is not None,
        "frame_count": len(raw),
        "sampled_every": step,
        "frames": frames,
        "assignments": assignments,
    }
