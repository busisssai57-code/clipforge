"""Jump-cut pacing — silence removal computed from word timestamps.

Blueprint §9.2 / the "built my own editor" workflow: any gap between words
longer than a threshold is cut, producing the fast-paced short-form rhythm
natively. The transcript's word-level timestamps make this exact — no audio
re-analysis, no guessing.

Everything downstream must agree on the compressed timeline, so this module
produces two things and they are the ONLY two things any stage may use:

  * ``keep_intervals`` — window-relative [(start, end), ...] of retained
    time, computed once in the CLI and passed to S5 and S6 in params (they
    hash into cache keys: different cuts = different artifacts).
  * ``TimeMap`` — original window-relative time -> compressed time. S5 runs
    word times through it; S6 stamps sendcmd/fades/progress with it.

Cuts keep a pad on each side of the surrounding words so no attack or
release is clipped, and total cutting is capped so the clip never drops
below the spec's minimum duration (§S2: 30-60 s; QA enforces 29-61).
"""

from __future__ import annotations

from bisect import bisect_right
from typing import Sequence

#: A pause must exceed this to be cut at all. 0.6 s: breaths survive,
#: dead air does not.
DEFAULT_GAP_S = 0.6

#: Retained padding on each side of a cut, so the edit lands in silence
#: rather than on the neighbouring word's edge.
DEFAULT_PAD_S = 0.12

#: Never compress below this (QA's duration floor is 29.0 with tolerance).
MIN_DURATION_S = 29.5


#: Words an editor cuts on sight. Matched case-insensitively against the
#: whole token, so "um" goes and "umbrella" stays — a substring match here
#: would silently gut the transcript.
#:
#: Multi-word phrases ("you know", "i mean") are handled as adjacent-token
#: runs by `find_filler_spans`, because word timestamps are per-token.
FILLER_WORDS: frozenset[str] = frozenset({
    "um", "uh", "umm", "uhh", "erm", "er", "ah", "eh", "hm", "hmm",
    "mmm", "mm", "like", "basically", "literally", "actually",
    "honestly", "obviously", "anyway", "so", "well", "right", "okay",
})

#: Phrases, as token tuples. Cut only when the tokens are adjacent.
FILLER_PHRASES: tuple[tuple[str, ...], ...] = (
    ("you", "know"), ("i", "mean"), ("sort", "of"), ("kind", "of"),
    ("or", "something"), ("or", "whatever"),
)

#: Fillers that are ONLY fillers when they are not carrying meaning.
#: "so" opening a sentence is a filler; "so I quit" is not. These are cut
#: only mid-sentence, never as the first word after a pause — removing a
#: sentence's opening word leaves a clip that starts mid-thought.
_CONTEXTUAL = frozenset({"so", "well", "right", "okay", "like", "actually",
                         "literally", "basically", "honestly", "anyway",
                         "obviously"})

#: A filler shorter than this is inside the noise floor of word alignment;
#: cutting it produces an audible click for no perceptible gain.
MIN_FILLER_S = 0.08


def _norm(text: str) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalpha())


def find_filler_spans(
        words: Sequence[tuple[str, float, float]],
        *,
        gap_threshold: float = DEFAULT_GAP_S,
        aggressive: bool = False,
) -> list[tuple[float, float]]:
    """(start, end) spans of filler words to remove.

    ``words`` are (text, start, end) in window-relative seconds, sorted.

    Two rules keep this from mangling speech. A CONTEXTUAL filler ("so",
    "like", "well") is cut only mid-sentence — cutting the word that opens
    a sentence leaves a clip starting mid-thought, which is worse than the
    filler. And a span shorter than ``MIN_FILLER_S`` is left alone: it is
    within alignment noise, so cutting it clicks without shortening
    anything a viewer notices.

    ``aggressive`` also cuts contextual fillers at sentence starts. Off by
    default, because that is an editorial judgement rather than a fix.
    """
    spans: list[tuple[float, float]] = []
    n = len(words)
    i = 0
    while i < n:
        text, start, end = words[i]
        token = _norm(text)

        # Phrases first: "you know" should go as a unit, and matching the
        # single-word list first would cut only half of it.
        matched_phrase = 0
        for phrase in FILLER_PHRASES:
            k = len(phrase)
            if i + k <= n and all(
                    _norm(words[i + j][0]) == phrase[j] for j in range(k)):
                matched_phrase = max(matched_phrase, k)
        if matched_phrase:
            p_start, p_end = words[i][1], words[i + matched_phrase - 1][2]
            if p_end - p_start >= MIN_FILLER_S:
                spans.append((p_start, p_end))
            i += matched_phrase
            continue

        if token in FILLER_WORDS and end - start >= MIN_FILLER_S:
            starts_sentence = (
                i == 0 or (start - words[i - 1][2]) > gap_threshold)
            if token in _CONTEXTUAL and starts_sentence and not aggressive:
                i += 1
                continue
            spans.append((start, end))
        i += 1
    return spans


def subtract_spans(
        keeps: Sequence[tuple[float, float]],
        cuts: Sequence[tuple[float, float]],
        *,
        pad: float = 0.02,
        min_keep: float = 0.06,
) -> list[tuple[float, float]]:
    """Remove ``cuts`` from ``keeps``, returning the surviving intervals.

    ``pad`` shrinks each cut slightly at both ends so the edit lands just
    inside the filler rather than on the neighbouring word's attack.
    Fragments shorter than ``min_keep`` are dropped: a 20 ms sliver of
    audio between two cuts is a click, not speech.
    """
    out: list[tuple[float, float]] = []
    ordered = sorted((max(a + pad, a), max(b - pad, a + pad))
                     for a, b in cuts if b > a)
    for k_start, k_end in keeps:
        cursor = k_start
        for c_start, c_end in ordered:
            if c_end <= cursor or c_start >= k_end:
                continue
            if c_start > cursor:
                out.append((cursor, min(c_start, k_end)))
            cursor = max(cursor, min(c_end, k_end))
        if cursor < k_end:
            out.append((cursor, k_end))
    return [(a, b) for a, b in out if b - a >= min_keep]


def compute_keep_intervals(
        word_spans: Sequence[tuple[float, float]],
        window_s: float,
        *,
        gap_threshold: float = DEFAULT_GAP_S,
        pad: float = DEFAULT_PAD_S,
        min_duration: float = MIN_DURATION_S,
) -> list[tuple[float, float]]:
    """Window-relative keep-intervals after silence cuts.

    ``word_spans`` are (start, end) in WINDOW-relative seconds, sorted.
    Returns intervals covering everything except interior silences longer
    than ``gap_threshold``. If total cutting would push the clip under
    ``min_duration``, the SMALLEST cuts are restored first — many short
    breaths matter less to pacing than the few long dead-air holes, so the
    long ones are kept as cuts.
    """
    if not word_spans or window_s <= 0:
        return [(0.0, window_s)] if window_s > 0 else []

    cuts: list[tuple[float, float]] = []
    prev_end = None
    for start, end in word_spans:
        if prev_end is not None:
            gap = start - prev_end
            if gap > gap_threshold:
                a, b = prev_end + pad, start - pad
                if b - a > 0.05:
                    cuts.append((a, b))
        prev_end = max(prev_end or 0.0, end)

    if not cuts:
        return [(0.0, window_s)]

    # Enforce the duration floor: restore smallest cuts until we fit.
    total_cut = sum(b - a for a, b in cuts)
    over = (window_s - total_cut) < min_duration
    if over:
        by_size = sorted(cuts, key=lambda c: c[1] - c[0])
        kept_cuts: list[tuple[float, float]] = list(cuts)
        for small in by_size:
            if window_s - sum(b - a for a, b in kept_cuts) >= min_duration:
                break
            kept_cuts.remove(small)
        cuts = sorted(kept_cuts)
        if not cuts:
            return [(0.0, window_s)]

    keeps: list[tuple[float, float]] = []
    cursor = 0.0
    for a, b in cuts:
        if a > cursor:
            keeps.append((cursor, a))
        cursor = b
    if cursor < window_s:
        keeps.append((cursor, window_s))
    return keeps


def quantize_keeps_to_frames(
        keeps: Sequence[tuple[float, float]],
        fps_rational: str) -> list[tuple[int, int]]:
    """Keep-intervals -> integer FRAME index pairs on the exact pts grid.

    The panel measured what decimal-seconds 'quantization' does at
    30000/1001 fps: ±32 ms per-seam sawtooth, +200 ms caption drift and a
    +231 ms duration misreport — 6-60x WORSE than no quantization at all,
    because round(a*float_fps)/float_fps is not the pts grid and the :.3f
    re-round moved it further. Integers through exact Fractions are the
    grid. Video trims by frame number, audio by seconds derived from the
    same integers; both sides of every seam agree by construction.
    """
    from fractions import Fraction

    fps = Fraction(fps_rational)
    out: list[tuple[int, int]] = []
    for a, b in keeps:
        fa = int(round(a * fps))
        fb = int(round(b * fps))
        if fb - fa >= 1:
            out.append((fa, fb))
    return out


def frames_to_seconds(frame: int, fps_rational: str) -> float:
    """Exact frame boundary as a float second (float of an exact rational)."""
    from fractions import Fraction

    return float(frame / Fraction(fps_rational))


def keeps_seconds_from_frames(
        keep_frames: Sequence[tuple[int, int]],
        fps_rational: str) -> list[tuple[float, float]]:
    """The ONE conversion every consumer must use, so S5's caption map and
    S6's splice arithmetic are built from byte-identical values."""
    return [(frames_to_seconds(fa, fps_rational),
             frames_to_seconds(fb, fps_rational))
            for fa, fb in keep_frames]


def enforce_floor_on_frames(
        keep_frames: list[tuple[int, int]],
        window_frames: int,
        fps_rational: str,
        min_duration: float = MIN_DURATION_S) -> list[tuple[int, int]]:
    """Duration floor re-checked AFTER frame quantization.

    The floor was enforced pre-quantization only; adversarially phased
    boundaries then lost up to half a frame per edge and a measured 30.10 s
    plan rendered at 28.80 s — under QA's hard 29.0 floor. Restore the
    smallest cuts (as frame gaps) until the quantized total clears the floor.
    """
    from fractions import Fraction

    fps = Fraction(fps_rational)
    min_frames = int(min_duration * fps) + 1

    def total(frames: Sequence[tuple[int, int]]) -> int:
        return sum(fb - fa for fa, fb in frames)

    frames = sorted(keep_frames)
    while frames and total(frames) < min_frames:
        # Find the smallest interior gap (cut) and close it.
        gaps = [(frames[i + 1][0] - frames[i][1], i)
                for i in range(len(frames) - 1)]
        if not gaps:
            return [(0, window_frames)]
        _size, i = min(gaps)
        merged = (frames[i][0], frames[i + 1][1])
        frames = frames[:i] + [merged] + frames[i + 2:]
    return frames if frames else [(0, window_frames)]


class TimeMap:
    """Original window-relative time -> compressed (post-cut) time.

    Piecewise-linear and monotonic. Times inside a cut map to the cut
    point's compressed position (they collapse onto the seam, which is what
    a caption straddling a removed silence should do).
    """

    def __init__(self, keeps: Sequence[tuple[float, float]]) -> None:
        self._starts: list[float] = []
        self._offsets: list[float] = []   # compressed start of each keep
        acc = 0.0
        for a, b in keeps:
            self._starts.append(a)
            self._offsets.append(acc)
            acc += b - a
        self._keeps = list(keeps)
        self._total = acc

    def duration(self) -> float:
        return self._total

    def to_compressed(self, t: float) -> float:
        """Map an original time; interior-of-cut times land on the seam."""
        if not self._keeps:
            return max(0.0, t)
        i = bisect_right(self._starts, t) - 1
        if i < 0:
            return 0.0
        a, b = self._keeps[i]
        return self._offsets[i] + min(max(t - a, 0.0), b - a)

    def is_kept(self, t: float) -> bool:
        i = bisect_right(self._starts, t) - 1
        if i < 0:
            return False
        a, b = self._keeps[i]
        return a <= t < b
