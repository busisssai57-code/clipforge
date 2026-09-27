"""Unit tests for S3.5 Editor Agent stage."""

from __future__ import annotations

from pathlib import Path
import pytest

from clipforge.schemas.editor import EditorArtifact
from clipforge.schemas.ranking import RankedArtifact, RankedItem
from clipforge.schemas.transcript import TranscriptArtifact, TranscriptSegment, Word
from clipforge.stages.s3_5_editor import S3_5_EditorAgent
from clipforge.state import StateDB


def test_editor_agent_execution(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "state.db")
    artifacts_dir = tmp_path / "artifacts"
    editor = S3_5_EditorAgent(db, artifacts_dir)

    words = [
        Word(text="What", start=0.0, end=0.3, score=0.95),
        Word(text="just", start=0.3, end=0.6, score=0.95),
        Word(text="happened?", start=0.6, end=1.0, score=0.95),
        Word(text="That", start=1.2, end=1.5, score=0.95),
        Word(text="was", start=1.5, end=1.8, score=0.95),
        Word(text="insane!", start=1.8, end=2.2, score=0.95),
    ]
    segment = TranscriptSegment(start=0.0, end=2.2, text="What just happened? That was insane!", words=words, speaker="SPEAKER_00")
    transcript = TranscriptArtifact(
        schema_version=1,
        stage="s1_transcribe",
        cache_key="transcript_key_123",
        source_path="/tmp/sample.mp4",
        abs_offset_s=0.0,
        language="en",
        diarization_ok=True,
        segments=[segment],
    )

    item = RankedItem(candidate_index=0, rank=1, hook_strength=8.5, justification="High tension question")
    ranked = RankedArtifact(
        schema_version=1,
        stage="s3_rank",
        cache_key="ranked_key_123",
        source_candidates="s2_candidates_key",
        ranking_source="semantic",
        items=[item],
    )


    params = {"style_profile": "viral_fast"}

    artifact = editor.run(
        input_digest="test_digest",
        params=params,
        ranked_artifact=ranked,
        transcript_artifact=transcript,
        candidate_id="cand_001",
    )

    assert isinstance(artifact, EditorArtifact)
    assert artifact.candidate_id == "cand_001"
    assert artifact.hook_score >= 0.7
    assert "youtube" in artifact.captions
    assert "tiktok" in artifact.captions
    assert "#Shorts" in artifact.hashtags
    assert len(artifact.title_variations) >= 2


# --- Hook grounding ------------------------------------------------------
#
# S3 writes the hook with frames and transcript in front of a VL model, and
# the model confabulated. A real run burned "UNSEEN MOMENTS FROM 'THE BIG
# BANG THEORY' REVEAL THE MAGIC" onto a clip about a creator's business.
# Nothing named a show. The editor took the hook unconditionally.

from clipforge.stages.s3_5_editor import _hook_is_grounded  # noqa: E402

_SAID = (
    "i mean people you know the show is headed then keep it a straight up "
    "secret when they asked monica to see it according to august none of "
    "them know we brainstorm all kinds of different outcomes"
)


def test_the_hallucinated_hook_from_the_real_run_is_rejected():
    assert not _hook_is_grounded(
        "UNSEEN MOMENTS FROM 'THE BIG BANG THEORY' REVEAL THE MAGIC", _SAID)


def test_an_invented_proper_noun_is_rejected():
    assert not _hook_is_grounded("MrBeast reveals his Netflix deal", _SAID)


def test_a_generic_hook_with_no_footing_in_the_clip_is_rejected():
    assert not _hook_is_grounded("You will not believe what happens next",
                                 _SAID)


def test_an_honest_paraphrase_still_wins():
    assert _hook_is_grounded("They kept the whole show a secret from Monica",
                             _SAID)


def test_an_empty_hook_is_never_grounded():
    assert not _hook_is_grounded("", _SAID)
    assert not _hook_is_grounded("   ", _SAID)


# ------------------------------------- capitalisation is not evidence of a name
#
# The guard treated every capitalised word as a proper noun that had to be
# spoken in the clip. MEASURED on 131 stored VL hooks from this workspace:
# 84 rejected, and 49 of those failed ONLY on Title Case — "Luxury Meets
# Espionage" rejected for "Meets" and "Espionage". Each rejection falls
# back to the raw first four seconds of speech, which is the flat
# on-screen text the operator called weak.


def test_a_title_case_hook_is_not_rejected_for_ordinary_words():
    """The hook talks about the same thing (3 of its 4 words are spoken),
    and only "Surprise" carries a capital it did not earn. That used to
    be a rejection."""
    from clipforge.stages.s3_5_editor import _hook_is_grounded

    said = "so we drove out to the old hotel and the whole floor was marble"
    assert _hook_is_grounded("Marble Floor Hotel Surprise", said)


def test_an_all_caps_hook_is_judged_on_its_words_not_its_case():
    from clipforge.stages.s3_5_editor import _hook_is_grounded

    said = "the whole floor was marble and the hotel was empty"
    assert _hook_is_grounded("EMPTY HOTEL, MARBLE FLOOR", said)


def test_the_invented_show_is_still_caught():
    """The case this guard exists for: a real run burned UNSEEN MOMENTS
    FROM 'THE BIG BANG THEORY' onto a clip about a creator's business."""
    from clipforge.stages.s3_5_editor import _hook_is_grounded

    said = "so I started this business in my garage and the first year was rough"
    assert not _hook_is_grounded("UNSEEN MOMENTS FROM 'THE BIG BANG THEORY'", said)


def test_a_name_invented_mid_sentence_is_still_caught():
    from clipforge.stages.s3_5_editor import _hook_is_grounded

    said = "so I started this business in my garage and the first year was rough"
    assert not _hook_is_grounded("he says Elon Musk called him", said)


def test_a_hook_about_something_else_entirely_is_still_caught():
    """The vocabulary floor is the other half of the guard, and it stays:
    35 of the 84 rejections were this, and they were right."""
    from clipforge.stages.s3_5_editor import _hook_is_grounded

    said = "the whole floor was marble and the hotel was empty"
    assert not _hook_is_grounded("Effortlessly manage your reel dashboard", said)


def test_title_case_detection_needs_more_than_one_word():
    from clipforge.stages.s3_5_editor import _is_title_or_caps

    assert not _is_title_or_caps("Wait")
    assert not _is_title_or_caps("he said Marble was cheap")
    assert _is_title_or_caps("Luxury Meets Espionage")
    assert _is_title_or_caps("EMPTY HOTEL MARBLE FLOOR")


def test_a_quoted_span_is_a_hard_claim_even_in_a_grounded_hook():
    """A quote is a claim about words that were SAID. This hook is about
    the right subject and shares most of its vocabulary, so the overlap
    floor lets it through — only the quote is wrong, and that is enough."""
    from clipforge.stages.s3_5_editor import _hook_is_grounded

    said = "i started this business in my garage and the first year was rough"
    grounded = "Started This Business In A Garage"
    assert _hook_is_grounded(grounded, said)
    assert not _hook_is_grounded(grounded + ' After "Shark Tank"', said), (
        "a quoted span nobody said was burned onto the clip")
