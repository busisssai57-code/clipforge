"""S3.5 Editor Agent stage — Human-grade AI editor & multi-platform copywriter."""

from __future__ import annotations

import re

from typing import Any
from clipforge.schemas.editor import EditorArtifact
from clipforge.schemas.ranking import RankedArtifact
from clipforge.schemas.transcript import TranscriptArtifact
from clipforge.stages.base import Stage


#: Words too common to prove a hook came from this clip rather than from the
#: model's imagination.
_HOOK_STOPWORDS = frozenset("""
a an the and or but if of to in on at for with from by as is are was were be
been being this that these those it its he she they them his her their you
your we our us i me my not no do does did so than then there here what when
where who why how all any both each few more most other some such only own
same too very can will just about into out up down over after before
""".split())


def _content_words(text: str) -> list[str]:
    """Lowercase words from ``text`` that carry meaning worth matching."""
    return [w for w in re.findall(r"[a-z0-9']+", text.lower())
            if len(w) > 1 and w not in _HOOK_STOPWORDS]


def _named_things(hook: str) -> list[str]:
    """Everything the hook asserts by NAME: quoted spans and proper nouns.

    The first word is skipped — a capital there is just a sentence opening.
    """
    named = [q.strip() for q in re.findall(r"['\"‘“]([^'\"’”]{2,})"
                                        r"['\"’”]", hook)]
    tokens = re.findall(r"[A-Za-z][\w']*", hook)
    named += [t for t in tokens[1:] if t[:1].isupper() and t.lower() != "i"]
    return named


def _hook_is_grounded(hook: str, transcript: str,
                      *, min_overlap: float = 0.4) -> bool:
    """Is everything this hook NAMES actually present in the clip?

    S3 writes the hook with the candidate's frames and transcript in front of
    a VL model, and a model asked for a punchy line will invent one. A real
    run burned "UNSEEN MOMENTS FROM 'THE BIG BANG THEORY'" onto a clip about
    a creator's business — naming a show nobody in the footage mentioned.

    Editorial paraphrase is the point of a hook and stays allowed. Naming a
    thing is different: it is a factual claim about the footage, so every
    quoted span and proper noun has to appear in what was actually said, and
    the hook overall has to share enough vocabulary to look like the same
    subject.
    """
    hook = (hook or "").strip()
    if not hook:
        return False
    haystack = " " + " ".join(_content_words(transcript)) + " "
    raw_hay = transcript.lower()
    for name in _named_things(hook):
        if name.lower() not in raw_hay:
            return False
    words = _content_words(hook)
    if not words:
        return False
    hits = sum(1 for w in words if f" {w} " in haystack)
    return (hits / len(words)) >= min_overlap



class S3_5_EditorAgent(Stage[EditorArtifact]):
    """Stage S3.5: AI Editor Agent.

    Analyzes transcript, hook strength, pacing, and generates platform-specific
    copywriting (titles, captions, hashtags, cover frame timestamp) acting as an
    AI Executive Producer / Human Clipper.
    """

    name: str = "editor"
    version: str = "2"
    vram_budget_gb: float = 0.0  # CPU-side NLP/prompt heuristics (or local LLM helper)
    wall_budget_s: float = 30.0
    artifact_type = EditorArtifact

    def _execute(
        self,
        *,
        cache_key: str,
        params: dict[str, Any],
        ranked_artifact: RankedArtifact,
        transcript_artifact: TranscriptArtifact,
        candidate_id: str,
        **kwargs: Any,
    ) -> EditorArtifact:
        start_s = kwargs.get("start_s", 0.0)
        end_s = kwargs.get("end_s", 30.0)


        # Collect transcript words within [start_s, end_s]
        clip_words = []
        for seg in transcript_artifact.segments:
            for w in seg.words:
                if start_s <= w.start <= end_s:
                    clip_words.append(w.text.strip())

        raw_text = " ".join(clip_words) if clip_words else "High energy moment from stream"

        # 1. Hook extraction (first 3-5 seconds)
        hook_words = []
        for seg in transcript_artifact.segments:
            for w in seg.words:
                if start_s <= w.start <= start_s + 4.0:
                    hook_words.append(w.text.strip())

        hook_text = " ".join(hook_words) if hook_words else raw_text[:50]

        # Calculate hook confidence score based on question marks, excitement, or length
        hook_score = 0.85 if ("?" in hook_text or "!" in hook_text) else 0.70

        # 2. Title — the VL model's, when it wrote one.
        #
        # S3 already has this candidate's FRAMES and transcript in front of a
        # loaded multimodal model, so it writes the title there and passes it
        # here. What follows is the fallback for the heuristic path only:
        # title-casing the first eight transcript words, which produced
        # "Aaron Judges Toughest Challenge Yet. We'Ll Be Against...!" —
        # mid-sentence, mangled apostrophes, no idea what was on screen. It
        # is a placeholder, and it is labelled as one.
        vl_item = next((it for it in ranked_artifact.items
                        if f"cand_{it.candidate_index:03d}" == candidate_id),
                       None)
        vl_title = (vl_item.title or "").strip() if vl_item else ""
        vl_hook = (vl_item.hook or "").strip() if vl_item else ""

        # The VL hook only wins if the clip can back it up. An ungrounded
        # hook is not a weaker hook, it is a false caption burned into the
        # picture, so it loses to the plain transcript opening every time.
        if vl_hook and _hook_is_grounded(vl_hook, raw_text):
            hook_text = vl_hook
            hook_score = 0.85 if ("?" in vl_hook or "!" in vl_hook) else 0.75

        if vl_title:
            title = vl_title
            main_topic = vl_title
        else:
            words_sample = raw_text.split()[:8]
            main_topic = (" ".join(words_sample).title() if words_sample
                          else "Insane Stream Moment")
            title = f"{main_topic}! 🔥"
        title_variations = [
            f"You won't believe what happened... ({main_topic})",
            f"Best moment live: {main_topic}",
            f"Wait for the end! 😱 #{main_topic.replace(' ', '')}",
        ]

        # 3. Multi-platform Copywriting
        hashtags = ["#Shorts", "#Viral", "#Trending", "#StreamClips", "#ClipForge"]

        captions = {
            "youtube": f"{title}\n\nWatch full stream moment! Drop a like and subscribe for daily clips.\n\n{' '.join(hashtags)}",
            "tiktok": f"{title} wait for the ending 💀 drop a comment below! {' '.join(hashtags)}",
            "instagram": f"Tag a friend who needs to see this! 👇\n.\n.\n{title}\n\n{' '.join(hashtags)}",
            "x_twitter": f"Unreal stream moment: {title}\n\n{' '.join(hashtags[:3])}",
        }

        # 4. Cover Frame Timestamp (recommend high visual/speech point ~1.5s in)
        cover_frame_offset_s = min(1.5, (end_s - start_s) / 2.0)

        style_profile = params.get("style_profile", "viral_fast")

        return EditorArtifact(
            cache_key=cache_key,
            candidate_id=candidate_id,
            hook_score=hook_score,
            hook_text=hook_text,
            title=title,
            title_variations=title_variations,
            captions=captions,
            hashtags=hashtags,
            cover_frame_offset_s=cover_frame_offset_s,
            style_profile=style_profile,
        )
