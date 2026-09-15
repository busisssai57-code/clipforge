"""S2 — candidate pre-filtering (spec §S2). CPU, deterministic, no models.

Slides 30–60 s windows over the S1 transcript and scores them on TEXT
heuristics only. A failure here is a bug, not bad luck — the stage's
declared failure mode is FATAL (spec §5), and every function in this module
is pure so the whole thing is exhaustively unit-testable.

Scoring components (each returns [0, 1]; weights are params):

  * ``boundary``  — window starts at a sentence/turn start and ends at
    terminal punctuation; mid-sentence cuts are penalized hard by
    construction (only sentence-aligned endpoints are ever generated) plus
    an explicit score for how cleanly the window closes.
  * ``qa``        — an interrogative followed by a substantive answer
    inside the window.
  * ``turns``     — speaker-turn density with a sweet-spot curve: dead air
    AND chaotic crosstalk both score low.
  * ``energy``    — words/second against a plateau.
  * ``laughter``  — laughter/interjection tokens.
  * ``selfcont``  — self-containedness: early pronouns without antecedents
    inside the window suggest the clip depends on missing context.

Determinism (§3.2): sorted iteration everywhere, no wall-clock, no random,
fixed tie-breaks (earlier window wins, then longer). All scores are
persisted per candidate — S3's fallback ordering depends on them, and
tuning sessions must be able to replay scoring offline (spec §S2).

Absolute time (T1): S1 artifacts carry absolute stream times already, so
windows here are absolute too, and cross-chunk dedup happens downstream via
``dedup_by_absolute_time``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Sequence

from clipforge.errors import FatalStageError
from clipforge.log import get_logger
from clipforge.schemas import (CandidatesArtifact, CandidateWindow,
                               TranscriptArtifact, TranscriptSegment)
from clipforge.stages.base import Stage

log = get_logger(__name__)

_UNSET = object()

_TERMINAL = (".", "!", "?", "…")
_INTERROGATIVE_LEAD = re.compile(
    r"^(who|what|when|where|why|how|is|are|was|were|do|does|did|can|could|"
    r"would|should|will|have|has|had)\b", re.IGNORECASE)
_LAUGH_TOKENS = re.compile(
    r"\b(haha+|hehe+|lol|lmao|rofl)\b|\[laugh(?:ter|s)?\]|\(laugh(?:ter|s)?\)",
    re.IGNORECASE)
_INTERJECTIONS = re.compile(
    r"\b(wow|whoa|woah|no way|oh my god|omg|insane|unbelievable|crazy|"
    r"let'?s go+|are you kidding)\b", re.IGNORECASE)
#: Pronouns that need an antecedent; opening a clip with them strands the
#: viewer ("so THAT was wild" — what was?).
_DANGLING_PRONOUNS = re.compile(
    r"\b(he|she|they|it|that|this|those|these|him|her|them)\b", re.IGNORECASE)


@dataclass(frozen=True)
class Sentence:
    """One sentence-ish unit with absolute times and its speaker."""

    start: float
    end: float
    text: str
    speaker: str | None
    terminal: bool  # ends with terminal punctuation
    turn_start: bool  # first sentence of a speaker turn


# --------------------------------------------------------------------------
# sentence extraction
# --------------------------------------------------------------------------


def split_sentences(segments: Sequence[TranscriptSegment]) -> list[Sentence]:
    """Deterministic sentence-ish units from S1 segments.

    WhisperX segments are usually sentence-shaped already; segments holding
    several sentences are split on terminal punctuation with times
    interpolated from word timings when present, else linearly.
    """
    out: list[Sentence] = []
    prev_speaker: Any = _UNSET  # sentinel: != any real speaker incl. None
    for seg in segments:
        parts = _split_segment_text(seg)
        for i, (start, end, text) in enumerate(parts):
            text = text.strip()
            if not text:
                continue
            turn_start = (seg.speaker != prev_speaker) and i == 0
            out.append(Sentence(
                start=start, end=end, text=text, speaker=seg.speaker,
                terminal=text.endswith(_TERMINAL), turn_start=turn_start))
        prev_speaker = seg.speaker
    # Mark the very first sentence as a turn start regardless.
    if out and not out[0].turn_start:
        out[0] = Sentence(out[0].start, out[0].end, out[0].text,
                          out[0].speaker, out[0].terminal, True)
    return out


def _split_segment_text(seg: TranscriptSegment) -> list[tuple[float, float, str]]:
    """(start, end, text) per sentence inside one segment."""
    text = seg.text.strip()
    if not text:
        return []
    # Find sentence boundaries: terminal punct followed by space or EOS.
    pieces = re.split(r"(?<=[.!?…])\s+", text)
    if len(pieces) == 1:
        return [(seg.start, seg.end, text)]

    # Apportion times: prefer word timings, else linear by char offset.
    if seg.words:
        return _apportion_by_words(seg, pieces)
    total_chars = max(1, len(text))
    out: list[tuple[float, float, str]] = []
    cursor = 0
    for piece in pieces:
        frac0 = cursor / total_chars
        cursor += len(piece) + 1
        frac1 = min(1.0, cursor / total_chars)
        dur = seg.end - seg.start
        out.append((seg.start + frac0 * dur, seg.start + frac1 * dur, piece))
    return out


def _apportion_by_words(seg: TranscriptSegment,
                        pieces: list[str]) -> list[tuple[float, float, str]]:
    """Assign word timings to sentence pieces by walking words in order."""
    out: list[tuple[float, float, str]] = []
    words = list(seg.words)
    wi = 0
    for piece in pieces:
        n_words = max(1, len(piece.split()))
        chunk = words[wi:wi + n_words]
        wi += n_words
        if chunk:
            out.append((chunk[0].start, chunk[-1].end, piece))
        else:  # ran out of word timings: fall back to the segment tail
            out.append((seg.end, seg.end, piece))
    return out


# --------------------------------------------------------------------------
# scoring components — each pure, each [0, 1]
# --------------------------------------------------------------------------


def score_boundary(window: Sequence[Sentence]) -> float:
    """Clean open + clean close. The generator only proposes sentence-aligned
    windows, so this scores the CLOSE quality and turn alignment."""
    if not window:
        return 0.0
    opens_on_turn = 1.0 if window[0].turn_start else 0.4
    closes_terminal = 1.0 if window[-1].terminal else 0.0  # hard penalty
    return 0.5 * opens_on_turn + 0.5 * closes_terminal


def score_qa(window: Sequence[Sentence]) -> float:
    """Interrogative followed by a substantive (≥8 words) answer within the
    window. The stronger pattern — question in the first half — scores 1."""
    for i, s in enumerate(window[:-1]):
        is_question = s.text.rstrip().endswith("?") or \
            (_INTERROGATIVE_LEAD.match(s.text) and len(s.text.split()) >= 4)
        if not is_question:
            continue
        answer_words = sum(len(t.text.split()) for t in window[i + 1:])
        if answer_words >= 8:
            early = i < max(1, len(window) // 2)
            return 1.0 if early else 0.7
    return 0.0


def score_turns(window: Sequence[Sentence], duration_s: float,
                *, lo: float = 0.05, hi: float = 0.35) -> float:
    """Turns/second on a trapezoid: a monologue (≈0) and chaotic crosstalk
    (>hi) both score low; conversation in the sweet spot scores 1."""
    if duration_s <= 0:
        return 0.0
    turns = sum(1 for s in window if s.turn_start)
    rate = turns / duration_s
    if rate <= 0:
        return 0.0
    if rate < lo:
        return rate / lo
    if rate <= hi:
        return 1.0
    return max(0.0, 1.0 - (rate - hi) / hi)


def score_energy(window: Sequence[Sentence], duration_s: float,
                 *, full_at: float = 2.5) -> float:
    """Words/second against a plateau: ~2.5 wps (lively speech) scores 1."""
    if duration_s <= 0:
        return 0.0
    words = sum(len(s.text.split()) for s in window)
    return min(1.0, (words / duration_s) / full_at)


def score_laughter(window: Sequence[Sentence]) -> float:
    hits = sum(len(_LAUGH_TOKENS.findall(s.text)) +
               len(_INTERJECTIONS.findall(s.text)) for s in window)
    return min(1.0, hits / 3.0)


def score_selfcontained(window: Sequence[Sentence], *, probe_words: int = 8) -> float:
    """Penalize windows whose OPENING leans on missing context: dangling
    pronouns in the first few words with no antecedent inside the window."""
    if not window:
        return 0.0
    opening = " ".join(window[0].text.split()[:probe_words])
    dangling = len(_DANGLING_PRONOUNS.findall(opening))
    return max(0.0, 1.0 - dangling / 3.0)


DEFAULT_WEIGHTS: dict[str, float] = {
    "boundary": 2.0,
    "qa": 1.5,
    "turns": 1.5,
    # The audience vote (live chat), when a chat log was supplied. Weighted
    # level with the structural turn signal: strong enough to lift a window
    # the room reacted to above one it sat quiet through, not so strong it
    # overrides sentence structure by itself. Contributes exactly 0 when no
    # chat log is present, so a run without one scores as it always has.
    "chat": 1.5,
    "energy": 0.75,
    "laughter": 0.75,
    "selfcont": 0.5,
}


#: Weights are multipliers on components already in [0, 1]. Anything beyond
#: this is a configuration error, not a tuning choice — and non-finite values
#: additionally break both the sum and the ranking.
MAX_ABS_WEIGHT = 1e6


def _finite_weight(key: str, value: Any) -> float:
    """Coerce and validate one operator-supplied weight.

    ``float(v)`` on an arbitrary param defeated ``digest_params``'
    ``allow_nan=False`` cache-key guard: a NaN *float* is rejected there, but
    the JSON *string* ``"nan"`` hashes cleanly and became a non-finite weight
    inside the stage. The artifact then carried bare ``NaN``/``Infinity``
    tokens — not valid JSON per RFC 8259 — and NMS ordering became
    permutation-dependent, because ``sorted`` cannot order NaN. Measured: two
    NaN scores among five candidates gave 39 distinct kept-sets across the
    120 input permutations.
    """
    try:
        w = float(value)
    except (TypeError, ValueError) as exc:
        raise FatalStageError(
            f"s2 weight {key!r} is not a number: {value!r}") from exc
    if not math.isfinite(w):
        raise FatalStageError(
            f"s2 weight {key!r} is not finite ({value!r}). Non-finite weights "
            "produce NaN scores, which cannot be ordered and serialize to "
            "invalid JSON.")
    if abs(w) > MAX_ABS_WEIGHT:
        raise FatalStageError(
            f"s2 weight {key!r} = {w} exceeds +/-{MAX_ABS_WEIGHT:g}; "
            "components are in [0, 1], so this can only be a mistake.")
    return w


def weighted_total(scores: dict[str, float],
                   weights: dict[str, float]) -> float:
    """Weighted sum that is exactly rounded and order-independent.

    Float addition is not associative, so a plain ``sum()`` makes the
    total's last bits depend on the order the components happen to be
    iterated in. A ``sorted()`` call was the only thing holding that
    stable — and deleting it left every gate green, because no test pinned
    the total's bits.

    ``math.fsum`` is exactly rounded and order-independent FOR FINITE INPUTS
    WHOSE PARTIAL SUMS DO NOT OVERFLOW. That caveat is real, not decorative:
    ``fsum`` raises ``OverflowError`` on an intermediate partial for some
    orderings and not others, so with a ``sorted()`` traversal the outcome
    would be decided by the component KEY NAMES. Weights are validated finite
    and bounded by :func:`_finite_weight` precisely so that domain holds; the
    sort keeps the traversal declared rather than incidental.
    """
    return math.fsum(weights.get(k, 0.0) * v for k, v in sorted(scores.items()))


def score_window(window: Sequence[Sentence], duration_s: float,
                 weights: dict[str, float],
                 chat_curve: Any = None) -> dict[str, float]:
    """All component scores plus the weighted total, keys SORTED on output
    (``total`` last) so serialization is byte-stable.

    ``chat_curve`` is an optional :class:`clipforge.ingest.chat.ChatCurve`.
    When absent (the default) the ``chat`` component is 0.0 and, at any
    weight, contributes nothing — so a run with no chat log produces the
    identical scores it did before this signal existed.
    """
    chat = 0.0
    if chat_curve is not None and window:
        from clipforge.ingest.chat import score_window as _chat_score  # noqa: PLC0415
        chat = _chat_score(window[0].start, window[-1].end, chat_curve)
    scores = {
        "boundary": score_boundary(window),
        "qa": score_qa(window),
        "turns": score_turns(window, duration_s),
        "chat": chat,
        "energy": score_energy(window, duration_s),
        "laughter": score_laughter(window),
        "selfcont": score_selfcontained(window),
    }
    # Rebuilt in sorted order: the docstring promised byte-stable
    # serialization but the dict was returned in construction order, so the
    # promise rested on nobody ever reordering the literal above.
    out = dict(sorted(scores.items()))
    out["total"] = weighted_total(scores, weights)
    return out


# --------------------------------------------------------------------------
# window generation + NMS
# --------------------------------------------------------------------------


def generate_windows(sentences: Sequence[Sentence], *, min_s: float,
                     max_s: float) -> list[tuple[int, int]]:
    """(start_idx, end_idx_inclusive) pairs, sentence-aligned, 30–60 s.

    Starts are turn starts (preferred) and sentence starts; ends are the
    sentences whose close lands inside [start+min, start+max]. Bounded:
    each start pairs with at most the few ends in its legal range.
    """
    out: list[tuple[int, int]] = []
    n = len(sentences)
    # Bounds compared in integer MILLISECONDS, for the same reason iou() is:
    # a window that is logically exactly 30.000 s was included or excluded
    # depending on the binary representation of its absolute offset. Measured
    # through the real stage: offset 8.3 gave 29.999999999999996 and DROPPED
    # the window; offset 16.7 gave 30.000000000000004 and kept it. Same
    # logical window, opposite decisions, decided by float luck — the exact
    # failure the iou() docstring says the Determinism Law forbids, left
    # standing one function away.
    min_ms = round(min_s * 1000.0)
    max_ms = round(max_s * 1000.0)
    for i in range(n):
        open_ms = round(sentences[i].start * 1000.0)
        for j in range(i, n):
            length_ms = round(sentences[j].end * 1000.0) - open_ms
            if length_ms < min_ms:
                continue
            if length_ms > max_ms:
                # `continue`, NOT `break`. Breaking assumes sentence END times
                # are non-decreasing, and they are not: _apportion_by_words
                # takes each sentence's end from its LAST MATCHED WORD, so one
                # stray word timestamp (a routine WhisperX alignment failure)
                # makes a later sentence end earlier than an earlier one. The
                # break then abandoned the scan and silently discarded legal,
                # in-bounds windows — measured 35 of 466 dropped (7.5%) from a
                # schema-legal transcript, 70 of 464 (15%) with a worse stray.
                continue
            out.append((i, j))
    return out


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Overlap ratio, computed in integer MILLISECONDS.

    The suppression test is a comparison against a configured threshold, so
    a pair sitting on that threshold decides the outcome. Subtracting
    second-scale floats makes both numerator and denominator carry
    representation error, and the ratio then lands on either side of the
    threshold for reasons nothing in the transcript records — the same
    input can rank differently on another machine, which the Determinism
    Law forbids. Integers make `inter` and `union` exact, so the quotient
    is the correctly-rounded double of an exact rational: reproducible
    everywhere. Millisecond resolution is far below word-timing precision.
    """
    a0, a1 = round(a[0] * 1000.0), round(a[1] * 1000.0)
    b0, b1 = round(b[0] * 1000.0), round(b[1] * 1000.0)
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def nms(candidates: list[CandidateWindow], *, iou_threshold: float,
        top_k: int) -> list[CandidateWindow]:
    """Greedy NMS by score. Deterministic tie-break: score desc, then
    earlier start, then longer. Returns at most top_k, score-descending.

    The boundary is deliberate and pinned by test: a candidate whose IoU is
    EXACTLY ``iou_threshold`` is KEPT — the config reads "suppress windows
    overlapping more than this". Nothing protected that choice before, so
    flipping to ``>=`` (silently dropping a whole class of candidates) left
    every gate green.
    """
    # The key must be a TOTAL order. Score/start/length alone is not: two
    # candidates can tie on all three, and the winner was then decided by
    # generate_windows' incidental emission order — "stable by discipline,
    # not by construction". Reachable in practice: a zero-duration segment
    # (what _apportion_by_words emits when word timings run out) among normal
    # ones, with word density above the energy plateau so every component
    # ties. Measured: 188 candidates, 12 colliding keys, and the GLOBAL
    # MAXIMUM key with multiplicity 2 — nms(c) and nms(reversed(c)) kept
    # different `text`, and `text` is S3's prompt input.
    ordered = sorted(candidates,
                     key=lambda c: (-c.total_score, c.start,
                                    -(c.end - c.start), c.text))
    kept: list[CandidateWindow] = []
    for cand in ordered:
        if any(iou((cand.start, cand.end), (k.start, k.end)) > iou_threshold
               for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= top_k:
            break
    return kept


# --------------------------------------------------------------------------
# the stage
# --------------------------------------------------------------------------


class S2Prefilter(Stage[CandidatesArtifact]):
    """`s1.transcript.json` → `s2.candidates.json` (spec §5). CPU, fatal."""

    name = "s2_prefilter"
    version = "1"
    vram_budget_gb = 0.0
    wall_budget_s = 60.0
    artifact_type = CandidatesArtifact

    def _execute(self, *, cache_key: str, params: dict[str, Any],
                 transcript: TranscriptArtifact | None = None,
                 **inputs: Any) -> CandidatesArtifact:
        if transcript is None:
            raise FatalStageError("S2 requires the S1 transcript artifact",
                                  stage=self.name)
        try:
            min_s = float(params.get("window_min_s", 30.0))
            max_s = float(params.get("window_max_s", 60.0))
            top_k = int(params.get("top_k", 10))
            iou_thr = float(params.get("nms_iou", 0.4))
            weights = dict(DEFAULT_WEIGHTS)
            weights.update({k: _finite_weight(k, v) for k, v in
                            dict(params.get("weights", {})).items()})

            from clipforge.ingest.chat import ChatCurve  # noqa: PLC0415
            chat_curve = ChatCurve.from_params(params.get("chat_curve"))
            chat_curve = chat_curve if chat_curve else None

            sentences = split_sentences(transcript.segments)
            candidates: list[CandidateWindow] = []
            for i, j in generate_windows(sentences, min_s=min_s, max_s=max_s):
                window = sentences[i:j + 1]
                start = window[0].start
                end = window[-1].end
                scores = score_window(window, end - start, weights,
                                      chat_curve=chat_curve)
                candidates.append(CandidateWindow(
                    start=start, end=end, total_score=scores["total"],
                    scores=scores,
                    text=" ".join(s.text for s in window)))

            kept = nms(candidates, iou_threshold=iou_thr, top_k=top_k)
            log.info("s2.candidates", generated=len(candidates),
                     kept=len(kept))
            return CandidatesArtifact(
                cache_key=cache_key,
                source_transcript=transcript.cache_key,
                candidates=kept)
        except FatalStageError:
            raise
        except Exception as exc:
            # Deterministic stage: ANY failure is a bug (spec §5), and it
            # must surface as the declared FATAL mode, typed.
            raise FatalStageError(
                f"S2 failed on a deterministic path: "
                f"{type(exc).__name__}: {exc}", stage=self.name) from exc
