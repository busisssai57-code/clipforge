"""S2 prefilter: pure heuristics, window law, NMS, determinism.

DRAFT — moves to tests/unit/ once round 7 completes (its meta-reviewer runs
the gate 3x checking determinism; a changing test count would pollute that).
"""

from pathlib import Path

import pytest

from clipforge.errors import FatalStageError
from clipforge.schemas import TranscriptArtifact, TranscriptSegment, Word
from clipforge.stages.base import digest_bytes
from clipforge.stages.s2_prefilter import (DEFAULT_WEIGHTS, S2Prefilter,
                                           Sentence, generate_windows, iou,
                                           nms, score_boundary, score_energy,
                                           score_laughter, score_qa,
                                           score_selfcontained, score_turns,
                                           score_window, split_sentences)
from clipforge.state import StateDB


def seg(start, end, text, speaker="SPEAKER_00", with_words=True):
    words = []
    if with_words:
        toks = text.split()
        step = (end - start) / max(1, len(toks))
        words = [Word(text=t, start=start + i * step,
                      end=start + (i + 1) * step, score=0.9, speaker=speaker)
                 for i, t in enumerate(toks)]
    return TranscriptSegment(start=start, end=end, text=text,
                             speaker=speaker, words=words)


def sent(start, end, text, speaker="S0", terminal=None, turn_start=False):
    return Sentence(start=start, end=end, text=text, speaker=speaker,
                    terminal=text.endswith((".", "!", "?", "…"))
                    if terminal is None else terminal,
                    turn_start=turn_start)


# ------------------------------------------------------------ split_sentences


def test_split_keeps_single_sentence_segment():
    out = split_sentences([seg(0, 5, "Hello world.")])
    assert len(out) == 1
    assert out[0].terminal and out[0].turn_start


def test_split_multi_sentence_segment_interpolates_times():
    out = split_sentences([seg(0, 10, "First point. Second point here.")])
    assert len(out) == 2
    assert out[0].start == 0.0
    assert out[0].end <= out[1].start + 1e-6
    assert out[1].end == pytest.approx(10.0, abs=0.5)


def test_split_marks_speaker_changes_as_turn_starts():
    out = split_sentences([
        seg(0, 5, "Question here?", "SPEAKER_00"),
        seg(5, 10, "Answer here.", "SPEAKER_01"),
        seg(10, 15, "Follow up.", "SPEAKER_01"),
    ])
    assert [s.turn_start for s in out] == [True, True, False]


def test_split_empty_and_whitespace_segments_dropped():
    assert split_sentences([seg(0, 1, "   ", with_words=False)]) == []


# ------------------------------------------------------------------ components


def test_boundary_rewards_clean_close():
    clean = [sent(0, 40, "A full thought ends here.", turn_start=True)]
    torn = [sent(0, 40, "and then we were about to", turn_start=True)]
    assert score_boundary(clean) > score_boundary(torn)
    assert score_boundary(torn) <= 0.5  # hard penalty for mid-sentence cut


def test_qa_needs_substantive_answer():
    q = sent(0, 5, "What happened next?", turn_start=True)
    thin = [q, sent(5, 8, "Nothing.", speaker="S1", turn_start=True)]
    fat = [q, sent(5, 20, "Well let me tell you the whole story of what "
                          "happened because it truly matters.",
                   speaker="S1", turn_start=True)]
    assert score_qa(fat) > score_qa(thin)
    assert score_qa(thin) == 0.0


def test_turns_sweet_spot_curve():
    def window_with_turns(n, dur=40.0):
        return [sent(i * dur / max(1, n), (i + 1) * dur / max(1, n),
                     "Words here.", turn_start=True) for i in range(n)]

    dead = score_turns(window_with_turns(0), 40.0)
    sweet = score_turns(window_with_turns(6), 40.0)     # 0.15/s
    chaos = score_turns(window_with_turns(40), 40.0)    # 1.0/s crosstalk
    assert dead == 0.0
    assert sweet == 1.0
    assert chaos < 0.5


def test_energy_plateau():
    lively = [sent(0, 10, " ".join(["word"] * 25))]     # 2.5 wps
    sparse = [sent(0, 10, "um okay")]
    assert score_energy(lively, 10.0) == 1.0
    assert score_energy(sparse, 10.0) < 0.2


def test_laughter_tokens_counted():
    with_laughs = [sent(0, 10, "hahaha no way that is insane [laughter]")]
    flat = [sent(0, 10, "the quarterly report shows growth")]
    assert score_laughter(with_laughs) > score_laughter(flat)
    assert score_laughter(flat) == 0.0


def test_selfcontained_penalizes_dangling_openers():
    dangling = [sent(0, 10, "So that was the craziest thing he did")]
    grounded = [sent(0, 10, "The marathon story starts in Berlin")]
    assert score_selfcontained(grounded) > score_selfcontained(dangling)


def test_score_window_is_weight_linear():
    w = [sent(0, 40, "What is the plan?", turn_start=True),
         sent(40, 45, "The plan is detailed and long and covers everything "
                      "we need to do tomorrow.", speaker="S1", turn_start=True)]
    zero = score_window(w, 45.0, {k: 0.0 for k in DEFAULT_WEIGHTS})
    assert zero["total"] == 0.0
    only_qa = score_window(w, 45.0, {"qa": 2.0})
    assert only_qa["total"] == pytest.approx(2.0 * only_qa["qa"])



# ------------------------------------------------------------- chat signal
#
# Live chat is an OPTIONAL scoring component: the audience vote re-ranks
# windows when a chat log was supplied, and contributes exactly nothing
# when one was not. Both halves are pinned, because "adds a signal without
# changing existing behaviour" is the promise that is easy to break.

def _chat_window():
    return [sent(100, 130, "Here comes the play of the game.", turn_start=True),
            sent(130, 145, "That was completely unbelievable honestly.",
                 speaker="S1", turn_start=True)]


def test_chat_absent_leaves_the_score_unchanged():
    """No curve -> the chat component is 0 and the total is exactly what it
    was before the signal existed."""
    from clipforge.stages.s2_prefilter import weighted_total
    w = _chat_window()
    scores = score_window(w, 45.0, DEFAULT_WEIGHTS, chat_curve=None)
    assert scores["chat"] == 0.0
    # total must equal the weighted sum of the NON-chat components.
    expected = weighted_total({k: v for k, v in scores.items()
                               if k not in ("chat", "total")}, DEFAULT_WEIGHTS)
    assert scores["total"] == pytest.approx(expected)


def test_chat_reaction_in_the_window_lifts_its_total():
    """A window the room reacted to outscores the identical window with a
    quiet chat, all else equal."""
    from clipforge.ingest.chat import ChatEvent, build_curve
    w = _chat_window()  # spans 100..145 s
    hot = build_curve([ChatEvent(t_s=132.0, user=f"u{i}") for i in range(12)])
    quiet = build_curve([ChatEvent(t_s=132.0, user="lonely")])
    s_hot = score_window(w, 45.0, DEFAULT_WEIGHTS, chat_curve=hot)
    s_quiet = score_window(w, 45.0, DEFAULT_WEIGHTS, chat_curve=quiet)
    assert s_hot["chat"] > s_quiet["chat"] >= 0.0
    assert s_hot["total"] > s_quiet["total"]


def test_chat_curve_survives_the_params_round_trip():
    """The curve reaches S2 through the cache-keyed params dict, so its
    serialized form must reconstruct the same scores."""
    from clipforge.ingest.chat import ChatCurve, ChatEvent, build_curve
    w = _chat_window()
    curve = build_curve([ChatEvent(t_s=132.0, user=f"u{i}") for i in range(12)])
    direct = score_window(w, 45.0, DEFAULT_WEIGHTS, chat_curve=curve)
    viaparams = score_window(w, 45.0, DEFAULT_WEIGHTS,
                             chat_curve=ChatCurve.from_params(curve.to_params()))
    assert direct["chat"] == viaparams["chat"]


# ---------------------------------------------------------- windows, iou, nms


def test_generate_windows_only_legal_lengths():
    sentences = [sent(i * 10.0, i * 10.0 + 9.0, "Ten seconds of talk here.")
                 for i in range(12)]
    for i, j in generate_windows(sentences, min_s=30.0, max_s=60.0):
        length = sentences[j].end - sentences[i].start
        assert 30.0 <= length <= 60.0


def test_iou_basics():
    assert iou((0, 10), (0, 10)) == 1.0
    assert iou((0, 10), (10, 20)) == 0.0
    assert iou((0, 10), (5, 15)) == pytest.approx(5 / 15)


def test_nms_deterministic_tiebreak():
    from clipforge.schemas import CandidateWindow

    a = CandidateWindow(start=0, end=40, total_score=5.0, scores={}, text="a")
    b = CandidateWindow(start=100, end=140, total_score=5.0, scores={}, text="b")
    got1 = nms([a, b], iou_threshold=0.4, top_k=1)
    got2 = nms([b, a], iou_threshold=0.4, top_k=1)
    assert got1 == got2 == [a], "tie must break to the earlier window"


def test_nms_suppresses_overlaps_keeps_best():
    from clipforge.schemas import CandidateWindow

    best = CandidateWindow(start=0, end=50, total_score=9.0, scores={}, text="x")
    shadow = CandidateWindow(start=5, end=55, total_score=8.0, scores={}, text="y")
    far = CandidateWindow(start=200, end=240, total_score=1.0, scores={}, text="z")
    kept = nms([shadow, best, far], iou_threshold=0.4, top_k=10)
    assert best in kept and far in kept and shadow not in kept


# -------------------------------------------------------------------- stage


@pytest.fixture()
def db(tmp_path: Path):
    d = StateDB(tmp_path / "s.db")
    yield d
    d.close()


def _transcript():
    segments = []
    t = 0.0
    for i in range(12):
        spk = "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01"
        text = (f"What about item {i} today?" if i % 2 == 0 else
                f"Item {i} deserves a long and thorough explanation indeed.")
        segments.append(seg(t, t + 7.5, text, spk))
        t += 7.5
    return TranscriptArtifact(cache_key="tkey", source_path="x.mp4",
                              segments=segments, turns=[])


def test_stage_produces_bounded_scored_candidates(db, tmp_path):
    s2 = S2Prefilter(db, tmp_path)
    art = s2.run(input_digest=digest_bytes(b"t"), params={"top_k": 3},
                 transcript=_transcript())
    assert 1 <= len(art.candidates) <= 3
    assert art.source_transcript == "tkey"
    assert all(c.scores["total"] == c.total_score for c in art.candidates)


def test_stage_missing_transcript_is_fatal(db, tmp_path):
    with pytest.raises(FatalStageError):
        S2Prefilter(db, tmp_path).run(input_digest=digest_bytes(b"t"),
                                      params={})


def test_stage_internal_error_is_fatal_typed(db, tmp_path):
    bad = _transcript()
    s2 = S2Prefilter(db, tmp_path)
    with pytest.raises(FatalStageError):
        s2.run(input_digest=digest_bytes(b"t"),
               params={"window_min_s": "not-a-number"}, transcript=bad)


def test_stage_cache_hit_skips_recompute(db, tmp_path):
    s2 = S2Prefilter(db, tmp_path)
    t = _transcript()
    a1 = s2.run(input_digest=digest_bytes(b"t"), params={}, transcript=t)
    a2 = s2.run(input_digest=digest_bytes(b"t"), params={}, transcript=t)
    assert a1.cache_key == a2.cache_key


def test_undiarized_transcript_still_yields_candidates(db, tmp_path):
    """No speakers at all (diarization_ok=False): turns score dies but the
    stage must still produce windows from the remaining heuristics."""
    segments = [seg(i * 7.5, (i + 1) * 7.5,
                    f"Sentence number {i} carries plenty of spoken words here.",
                    speaker=None) for i in range(12)]
    t = TranscriptArtifact(cache_key="nodiar", source_path="x.mp4",
                           segments=segments, turns=[], diarization_ok=False)
    art = S2Prefilter(db, tmp_path).run(input_digest=digest_bytes(b"t"),
                                        params={}, transcript=t)
    assert art.candidates, "undiarized content produced zero candidates"


def test_stage_threads_chat_curve_from_params(db, tmp_path):
    """The audience signal reaches the stage through the cache-keyed params
    dict: a chat reaction lands as a non-zero chat score on the candidate
    covering it, and the chat content changes the cache key (so a different
    chat re-computes, a moved file would not)."""
    from clipforge.ingest.chat import ChatEvent, build_curve

    t = _transcript()  # 0..90 s
    curve = build_curve([ChatEvent(t_s=32.0, user=f"u{i}") for i in range(15)])
    s2 = S2Prefilter(db, tmp_path)

    plain = s2.run(input_digest=digest_bytes(b"t"), params={}, transcript=t)
    withchat = s2.run(input_digest=digest_bytes(b"t"),
                      params={"chat_curve": curve.to_params()}, transcript=t)

    assert withchat.cache_key != plain.cache_key, "chat did not enter the key"
    assert all(c.scores.get("chat", 0.0) == 0.0 for c in plain.candidates)
    assert any(c.scores.get("chat", 0.0) > 0.0 for c in withchat.candidates), \
        "no candidate picked up the chat reaction at 32s"


def test_chat_peaks_seed_a_candidate_with_no_transcript(db, tmp_path):
    """A reaction with no words behind it becomes a candidate on the audience
    vote alone — the whole point of seeding. The transcript here stops at 90s;
    the chat erupts at 200s, where no sentence-aligned window can reach."""
    from clipforge.ingest.chat import ChatEvent, build_curve

    t = _transcript()  # sentences span 0..90 s
    curve = build_curve([ChatEvent(t_s=200.0 + (i % 3), user=f"u{i}")
                         for i in range(60)])
    s2 = S2Prefilter(db, tmp_path)
    art = s2.run(input_digest=digest_bytes(b"t"),
                 params={"chat_curve": curve.to_params(), "top_k": 10},
                 transcript=t)

    reaction = [c for c in art.candidates if c.start >= 150.0]
    assert reaction, "the 200s chat reaction seeded no candidate"
    c = reaction[0]
    assert c.scores["chat"] > 0.0
    assert 29.9 <= (c.end - c.start) <= 60.1, "seeded window is not clip-length"


def test_no_seeding_without_chat(db, tmp_path):
    """No chat curve -> no seeded candidates, identical to before."""
    t = _transcript()
    s2 = S2Prefilter(db, tmp_path)
    art = s2.run(input_digest=digest_bytes(b"t"), params={"top_k": 10},
                 transcript=t)
    assert all(c.start < 90.0 for c in art.candidates), \
        "a candidate appeared past the transcript with no chat to seed it"


def test_seeding_can_be_disabled(db, tmp_path):
    from clipforge.ingest.chat import ChatEvent, build_curve
    t = _transcript()
    curve = build_curve([ChatEvent(t_s=200.0, user=f"u{i}") for i in range(60)])
    s2 = S2Prefilter(db, tmp_path)
    art = s2.run(input_digest=digest_bytes(b"t"),
                 params={"chat_curve": curve.to_params(),
                         "chat_seed_peaks": False, "top_k": 10},
                 transcript=t)
    assert all(c.start < 150.0 for c in art.candidates), \
        "seeding was disabled but a peak candidate still appeared"


def test_seeding_is_deterministic(db, tmp_path):
    from clipforge.ingest.chat import ChatEvent, build_curve
    t = _transcript()
    ev = [ChatEvent(t_s=200.0 + (i % 4), user=f"u{i}") for i in range(60)]
    curve = build_curve(ev)
    s2 = S2Prefilter(db, tmp_path)
    a = s2.run(input_digest=digest_bytes(b"t"),
               params={"chat_curve": curve.to_params(), "top_k": 10},
               transcript=t)
    curve2 = build_curve(list(reversed(ev)))
    b = s2.run(input_digest=digest_bytes(b"t"),
               params={"chat_curve": curve2.to_params(), "top_k": 10},
               transcript=t)
    assert [(c.start, c.end, c.text) for c in a.candidates] == \
           [(c.start, c.end, c.text) for c in b.candidates]
