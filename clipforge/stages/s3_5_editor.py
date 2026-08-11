"""S3.5 Editor Agent stage — Human-grade AI editor & multi-platform copywriter."""

from __future__ import annotations

from typing import Any
from clipforge.schemas.editor import EditorArtifact
from clipforge.schemas.ranking import RankedArtifact
from clipforge.schemas.transcript import TranscriptArtifact
from clipforge.stages.base import Stage


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

        if vl_hook:
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
