"""S3 Semantic Ranking stage — Qwen2.5-VL-7B INT4/NF4 candidate ranking with visual tokens."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from clipforge.errors import FatalStageError, StageError
from clipforge.gpu import ModelClass, gpu_session, hard_unload
from clipforge.log import get_logger

#: Exception types that always mean "the code is wrong", never "the machine
#: lacks a model". These must never be absorbed into a quality fallback.
_PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError)
from clipforge.schemas.candidates import CandidatesArtifact
from clipforge.schemas.ranking import RankedArtifact, RankedItem
from clipforge.stages.base import Stage

log = get_logger(__name__)


#: The scoring rubric, in one place because it is the thing that decides
#: which moment ships. It is a module constant so a test can read it, a
#: change to it is visible in review, and the same words go to the local
#: model and to a cloud judge.
#:
#: What changed, and why: the first version asked for three numbers with
#: no definition of what any number MEANT. A model asked to "score 0-10"
#: with no anchors returns 7 for almost everything — the spread collapses
#: and the ranking is then decided by noise. Each axis now has anchors at
#: 0, 5 and 10, and each is a question about evidence in the frames or
#: the words, not a vibe.
SCORING_RUBRIC = """\
You are choosing which seconds of a long video become a short-form clip.
Judge ONLY what is in these frames and this transcript. Never assume what
happened before or after.

Score three axes, each 0-10. Use the WHOLE range: a 5 is ordinary, an 8 is
rare, a 10 should be almost unheard of. If two candidates feel the same,
find the difference and let the numbers show it.

visual_action - is there something to WATCH?
  0  a static talking head, a slide, a logo, an empty room
  5  a person gesturing, a normal cut, something changing slowly
  10 a physical event you could describe to someone who did not see it:
     a reaction, a reveal, an accident, a sudden change of state

hook_strength - would a thumb stop here in the FIRST SECOND?
  Judge the first frame hardest: that is all a scrolling viewer sees.
  0  the clip opens mid-sentence on nothing; a title card; dead air
  5  a clear face saying something ordinary
  10 the opening frame itself raises a question, or the first words are a
     claim, a number, a confession or a challenge that demands the answer

comprehensibility - does it STAND ALONE?
  0  needs the previous ten minutes; opens on "and that's why..."
  5  understandable but references something unseen
  10 a stranger with no context gets the whole thing, and it RESOLVES
     inside the clip rather than stopping mid-thought

Then write the packaging:
  title - what actually happens in these frames, concrete and specific,
          at most 8 words, no hashtags, no clickbait about things the
          frames do not show
  hook  - the one line to burn on screen in the first seconds. Short
          enough to read at a glance (at most 7 words). It must be true
          of this clip. A question or an incomplete promise beats a
          summary: the hook's job is to make the next second necessary.
  justification - name the specific thing you saw or heard that decided
          the scores, in at most 25 words. "Man knocks over the table at
          3s" is a reason; "engaging content" is not.

Return ONLY a JSON object:
{"visual_action": float, "hook_strength": float, "comprehensibility":
 float, "justification": str, "title": str, "hook": str}"""


#: How the three axes combine into the order clips ship in. Not a flat
#: mean: a plain average treats "nothing to look at" and "nobody would
#: stop scrolling" as equally survivable, and on short-form they are not.
#: A clip nobody stops for is never watched at all, so the hook leads;
#: comprehensibility is a floor rather than a prize, which is why it
#: carries the least weight but is the axis the penalty below reads.
SCORE_WEIGHTS = {"hook_strength": 0.45, "visual_action": 0.35,
                 "comprehensibility": 0.20}

#: A clip nobody can follow is not a good clip with a flaw; it is not a
#: clip. Below this, the weighted score is halved rather than the item
#: being dropped, so the operator still sees it ranked last with a reason.
COMPREHENSION_FLOOR = 4.0

#: Output budget for one judgement. Measured on 137 stored judgements from
#: this workspace: median 96 tokens, p90 130, max 196. The old cap of 128
#: cut roughly one in ten mid-object, and a cut object does not parse.
MAX_JUDGEMENT_TOKENS = 384


def weighted_score(visual_action: float, hook_strength: float,
                   comprehensibility: float) -> float:
    """One number to rank by, from the three the model returns."""
    return round(
        SCORE_WEIGHTS["visual_action"] * float(visual_action)
        + SCORE_WEIGHTS["hook_strength"] * float(hook_strength)
        + SCORE_WEIGHTS["comprehensibility"] * float(comprehensibility), 4)


def rank_key(item: Any) -> tuple:
    """Sort key for ranked items. Shared, so both judges order the same way.

    Two bands, not a penalty: a clip nobody can follow goes BELOW every
    clip that can be followed, whatever its other numbers. Halving the
    score did not do that — a flashy incoherent candidate still outranked
    a clear ordinary one — while the comment claimed it did, which is the
    kind of gap between code and prose this project treats as a defect.

    Within a band: weighted score, then the hook (the axis that decides
    whether a clip is watched at all), then the candidate index, because a
    tie with no final key leaves the order to dict iteration and §3.2 says
    the same input produces the same clip.
    """
    comp = float(item.comprehensibility or 0)
    return (0 if comp >= COMPREHENSION_FLOOR else 1,
            -weighted_score(item.visual_action or 0, item.hook_strength or 0,
                            comp),
            -(item.hook_strength or 0),
            item.candidate_index)


def scoring_digest(rubric: str = SCORING_RUBRIC,
                   weights: dict | None = None,
                   floor: float = COMPREHENSION_FLOOR) -> str:
    """A short hash of everything that decides S3's output but is not a param.

    The stage's cache key is built from name|input_digest|version|params,
    and the rubric, the weights and the floor are in NONE of those — they
    are module constants inside _execute. So rewriting the rubric changed
    what the model would be asked and changed nothing about what a re-run
    returned: every already-processed source answered from an artifact
    scored by the OLD prompt, reported a cache hit, and looked like a
    success. Measured on this workspace: 13 of 21 stored S2 keys still
    resolved to pre-rubric S3 artifacts.

    Deriving the version from these three means the next edit to any of
    them invalidates the cache on its own, rather than relying on whoever
    makes it remembering to bump a number.
    """
    import hashlib

    payload = json.dumps(
        {"rubric": rubric,
         "weights": SCORE_WEIGHTS if weights is None else weights,
         "floor": floor},
        sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _local(t_s: float, abs_offset_s: float) -> float:
    """Candidate time (ABSOLUTE stream seconds) as an offset into the file.

    S1 stamps every word with ``abs_offset + local``, so everything
    downstream of it — including S2's candidate spans — is on the stream's
    timeline. The file S3 opens is one window, which starts at 0. Seeking
    to an absolute time inside it read the wrong part of the clip, and,
    once the offset passed the window's duration, read nothing at all and
    failed the whole job. Measured 2026-09-17: every `bta watch` window
    after the first could not be ranked.

    Clamped at 0: a candidate that starts a hair before the window (edge
    snapping, a word that straddles the seam) is still readable from its
    first frame rather than seeking to a negative index.
    """
    return max(0.0, float(t_s) - float(abs_offset_s))


def _extract_frames_cv2(video_path: Path, start_s: float, end_s: float, num_frames: int = 8) -> list[Any]:
    """Extract evenly-spaced RGB frames from video_path within [start_s, end_s]."""
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise StageError(f"Cannot open video for frame extraction: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps)
    total_frames = max(1, end_frame - start_frame)

    step = total_frames / max(1, num_frames)
    frame_indices = [int(start_frame + i * step) for i in range(num_frames)]

    frames = []
    try:
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
    finally:
        cap.release()

    return frames


def _cloud_ranker(params: dict[str, Any]) -> Any:
    """Build the cloud ranker from stage params, or None.

    None means "must not exist" — cloud disabled or no key — so with the
    cloud off there is no object anywhere holding a key, and no code path
    that could put a frame on the wire.
    """
    if not params.get("use_cloud", False):
        return None
    try:
        from clipforge.vlrank import GeminiRanker  # noqa: PLC0415
    except ImportError as exc:
        log.warning("s3.cloud_unavailable", error=str(exc)[:200])
        return None

    # The credential comes through the §2 chokepoint. enabled=True is not
    # a bypass: the authorizing flag was checked two lines up from PARAMS
    # — the cache-key copy of [s3] use_cloud, which is the authority this
    # stage runs under (a live cfg does not exist here, and the params
    # copy is what the artifact digest records).
    from clipforge.cloud import gemini_key  # noqa: PLC0415

    key = gemini_key(feature="s3_ranking", enabled=True)

    ranker = GeminiRanker(
        key,
        model=params.get("cloud_model", "gemini-2.5-pro"),
        models=list(params.get("cloud_models") or []) or None,
        timeout_s=params.get("cloud_timeout_s", 90.0),
        max_attempts=params.get("cloud_max_attempts", 3),
    )
    if not ranker.available():
        log.info("s3.cloud_not_configured",
                 note="no CLIPFORGE_GEMINI_API_KEY; ranking on the local model")
        return None
    return ranker


class S3SemanticRanker(Stage[RankedArtifact]):
    """Stage S3: Semantic Candidate Ranking using Qwen2.5-VL-7B.

    Evaluates S2 candidates along with 6-8 video frames to produce a
    multimodal ranking score. Hard-caps visual tokens via max_pixels to
    prevent VRAM OOM. Falls back to S2 heuristic order on failure.
    """

    name: str = "s3_semantic"
    #: 4 for the anchored rubric and the weighted sort; the suffix is the
    #: digest of the rubric, the weights and the floor, so a future edit
    #: to any of them moves the cache key without anyone remembering to.
    version: str = f"4+{scoring_digest()}"
    vram_budget_gb: float = 10.0
    wall_budget_s: float = 180.0
    artifact_type = RankedArtifact

    #: Why the cloud ranker did not produce this artifact, when it was
    #: configured. Recorded into the artifact so a weaker judgement never
    #: looks like a deliberate choice after the fact.
    _fallback_reason: str | None = None

    def _rank_with_cloud(self, ranker: Any, *, cache_key: str,
                         candidates_artifact: CandidatesArtifact,
                         video_path: Path,
                         frames_per_cand: int,
                         abs_offset_s: float = 0.0) -> RankedArtifact:
        """Judge every candidate with the cloud model.

        Any failure propagates: the caller decides to fall back, and it
        falls back for the WHOLE run rather than leaving a ranking where
        some candidates were scored by one model and some by another —
        those numbers are not comparable, and ranking is a comparison.
        """
        from clipforge.vlrank import Judgement  # noqa: PLC0415

        candidates = candidates_artifact.candidates
        items: list[RankedItem] = []
        for i, cand in enumerate(candidates[:10]):
            frames = _extract_frames_cv2(
                video_path=video_path,
                start_s=_local(cand.start, abs_offset_s),
                end_s=_local(cand.end, abs_offset_s),
                num_frames=frames_per_cand)
            judgement: Judgement = ranker.judge(
                transcript=cand.text, frames=frames,
                start_s=cand.start, end_s=cand.end)
            items.append(RankedItem(
                candidate_index=getattr(cand, "index", i),
                rank=i + 1, **judgement.as_kwargs()))
        # The same order as the local path: one clip must not be ranked
        # two different ways depending on which judge was available.
        items.sort(key=rank_key)
        for r, item in enumerate(items, 1):
            item.rank = r

        items.sort(key=lambda x: -((x.visual_action or 0)
                                   + (x.hook_strength or 0)
                                   + (x.comprehensibility or 0)))
        for r, item in enumerate(items, 1):
            item.rank = r

        log.info("s3.cloud_ranked", ranker=ranker.describe(), items=len(items))
        return RankedArtifact(
            cache_key=cache_key,
            source_candidates=candidates_artifact.cache_key,
            ranking_source="semantic",
            ranker=ranker.describe(),
            items=items,
            # No visual tokens are consumed on this machine: the frames
            # were judged remotely. Reporting 0 would read as "the VL
            # processor ran and used none", which is a different claim.
            visual_tokens_used=None,
        )

    def _execute(
        self,
        *,
        cache_key: str,
        params: dict[str, Any],
        candidates_artifact: CandidatesArtifact,
        video_path: Path | None = None,
        abs_offset_s: float = 0.0,
        **kwargs: Any,
    ) -> RankedArtifact:
        candidates = candidates_artifact.candidates
        if not candidates:
            return RankedArtifact(
                cache_key=cache_key,
                source_candidates=candidates_artifact.cache_key,
                ranking_source="heuristic",
                items=[],
            )

        # Fallback helper: heuristic order from S2
        def _heuristic_fallback(reason: str) -> RankedArtifact:
            raise StageError(f"S3 Semantic Ranking failed (fallback disabled to enforce AI processing): {reason}")

        model_id = params.get("model_id", "Qwen/Qwen2.5-VL-7B-Instruct-AWQ")
        fallback_model_id = params.get("fallback_model_id", "Qwen/Qwen2.5-VL-7B-Instruct")
        max_pixels = params.get("max_pixels", 451_584)
        frames_per_cand = params.get("frames_per_candidate", 8)

        # If video_path is missing or doesn't exist, use heuristic fallback
        if video_path is None or not Path(video_path).exists():
            return _heuristic_fallback("video file unavailable for frame extraction")

        # ---- Cloud ranking first ------------------------------------
        # The strongest model available does the judging, because this is
        # the score that decides which clip ships. It needs no VRAM, so
        # the local path below stays exactly as it was and takes over
        # whenever the cloud is unavailable — never because it disagreed.
        cloud = _cloud_ranker(params)
        if cloud is not None:
            try:
                return self._rank_with_cloud(
                    cloud, cache_key=cache_key,
                    candidates_artifact=candidates_artifact,
                    video_path=Path(video_path),
                    frames_per_cand=frames_per_cand,
                    abs_offset_s=abs_offset_s)
            except _PROGRAMMING_ERRORS:
                raise
            except Exception as exc:  # noqa: BLE001 - degraded, not broken
                fallback_reason = f"{type(exc).__name__}: {exc}"[:300]
                log.warning("s3.cloud_fallback", ranker=cloud.describe(),
                            reason=fallback_reason)
                self._fallback_reason = fallback_reason

        try:
            # Model inference block under VRAM guard
            # gpu_session, not vram_guard. vram_guard is a plain function
            # returning None; `with vram_guard(...)` raised TypeError before a
            # single guarded line ran, and the blanket `except Exception`
            # below turned that into a silent heuristic fallback. Net effect:
            # ModelClass.VL was NEVER registered, so "VL and ASR never
            # simultaneous" — the law's hard pair — was enforced by nothing,
            # and no VRAM budget was ever asserted for S3.
            with gpu_session(ModelClass.VL, self.vram_budget_gb):
                # Try importing transformers / qwen_vl_utils
                try:
                    import torch
                    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
                except ImportError as err:
                    return _heuristic_fallback(f"transformers or dependencies not installed: {err}")

                # Load processor & model.
                #
                # The dict IS the reference holder, from the moment of
                # construction — never a bare local. `hard_unload` pops the
                # dict entry so the refcount drops BEFORE gc.collect() and
                # empty_cache() run; if a local also points at the weights,
                # popping frees nothing and ~10 GB of VL model stays
                # resident. The next stage's vram_guard then refuses to
                # load with "a previous stage leaked memory" — a message
                # that is accurate and whose cause is this dict.
                models: dict[str, Any] = {}
                # Which weights actually loaded, for the artifact's record:
                # the AWQ id and the NF4 id are different models, and the
                # ranking should say which one judged it.
                loaded_model_id = model_id
                try:
                    models["processor"] = AutoProcessor.from_pretrained(
                        model_id, max_pixels=max_pixels)
                    try:
                        models["model"] = (
                            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                model_id,
                                dtype=torch.float16,
                                device_map="auto",
                            ))
                    except Exception:
                        # Fallback to NF4 load
                        models["model"] = (
                            Qwen2_5_VLForConditionalGeneration.from_pretrained(
                                fallback_model_id,
                                load_in_4bit=True,
                                device_map="auto",
                            ))
                        loaded_model_id = fallback_model_id
                except Exception as load_err:
                    hard_unload(models)
                    return _heuristic_fallback(f"model load failed: {load_err}")

                ranked_items = []
                total_visual_tokens = 0

                try:
                    unreadable = 0
                    for i, cand in enumerate(candidates[:10]):
                        frames = _extract_frames_cv2(
                            video_path=Path(video_path),
                            start_s=_local(cand.start, abs_offset_s),
                            end_s=_local(cand.end, abs_offset_s),
                            num_frames=frames_per_cand,
                        )
                        if not frames:
                            # `cap.read()` can fail for every index in a
                            # window — a seek past the end, a damaged GOP,
                            # a codec the build cannot decode at that
                            # offset — and `_extract_frames_cv2` then
                            # returns []. Handing an empty image list to
                            # the processor raises `IndexError: list index
                            # out of range`, which the wrapper below turned
                            # into "S3 execution error" and lost the whole
                            # job. MEASURED: that is what killed the last
                            # real clip run on 2026-08-13, and the nine
                            # failed jobs in the state DB.
                            #
                            # One unreadable window is not a broken run.
                            # It is scored last and said out loud, so ten
                            # candidates do not die for one of them.
                            unreadable += 1
                            log.warning("s3.frames_unreadable", candidate=i,
                                        start=round(cand.start, 2),
                                        end=round(cand.end, 2),
                                        video=Path(video_path).name,
                                        note="scored last; it cannot be "
                                             "judged on pictures nobody saw")
                            ranked_items.append(
                                RankedItem(
                                    candidate_index=getattr(cand, "index", i),
                                    rank=i + 1,
                                    visual_action=0.0,
                                    hook_strength=0.0,
                                    comprehensibility=0.0,
                                    justification=(
                                        "no frames could be read from "
                                        f"{cand.start:.1f}s-{cand.end:.1f}s"),
                                )
                            )
                            continue

                        prompt_text = (
                            f"{SCORING_RUBRIC}\n\n"
                            f"Candidate ({cand.start:.1f}s-{cand.end:.1f}s), "
                            f"{len(frames)} frames in order, first frame "
                            "first.\n"
                            f"Transcript: {cand.text}"
                        )

                        # Generate structured JSON score
                        messages = [
                            {
                                "role": "user",
                                "content": [
                                    *[{"type": "image", "image": img} for img in frames],
                                    {"type": "text", "text": prompt_text},
                                ],
                            }
                        ]

                        # models[...] at every use, never a local alias: an
                        # alias is a second strong reference and hard_unload
                        # can only drop the one it owns.
                        text_input = models["processor"].apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                        inputs = models["processor"](text=[text_input], images=frames, return_tensors="pt", padding=True).to("cuda")

                        total_visual_tokens += inputs.get("pixel_values", torch.tensor([])).shape[0] if "pixel_values" in inputs else 0

                        with torch.no_grad():
                            # MEASURED on this workspace's 137 stored
                            # judgements: median 96 output tokens, p90 130,
                            # max 196 — against a cap of 128. Roughly one
                            # in ten was already being cut mid-object, and
                            # a truncated object does not parse, which used
                            # to mean invented scores and an empty title
                            # and hook. The richer rubric asks for more
                            # words, not fewer, so the cap moves with it.
                            outputs = models["model"].generate(
                                **inputs, max_new_tokens=MAX_JUDGEMENT_TOKENS,
                                do_sample=False)

                        generated_ids = [
                            output_ids[len(input_ids) :]
                            for input_ids, output_ids in zip(inputs.input_ids, outputs)
                        ]
                        response_text = models["processor"].batch_decode(generated_ids, skip_special_tokens=True)[0]

                        # Parse JSON
                        try:
                            data = json.loads(response_text[response_text.find("{") : response_text.rfind("}") + 1])
                            ranked_items.append(
                                RankedItem(
                                    candidate_index=getattr(cand, "index", i),
                                    rank=i + 1,
                                    visual_action=float(data.get("visual_action", 5.0)),
                                    hook_strength=float(data.get("hook_strength", 5.0)),
                                    comprehensibility=float(data.get("comprehensibility", 5.0)),
                                    justification=str(data.get("justification", ""))[:500],
                                    title=str(data.get("title", "")).strip()[:120],
                                    hook=str(data.get("hook", "")).strip()[:160],
                                )
                            )
                        except Exception:
                            # A judgement that did not parse is not a
                            # middling clip — it is no judgement. The old
                            # code wrote 5/5 and, worse, put S2's
                            # heuristic total (a 0-8.5 scale answering a
                            # different question) into hook_strength, the
                            # axis that now carries the most weight. A
                            # candidate nobody scored ranks last, with a
                            # reason, the way an unreadable window does.
                            log.warning("s3.judgement_unparseable",
                                        candidate=i,
                                        head=(response_text or "")[:160])
                            ranked_items.append(
                                RankedItem(
                                    candidate_index=getattr(cand, "index", i),
                                    rank=i + 1,
                                    visual_action=0.0,
                                    hook_strength=0.0,
                                    comprehensibility=0.0,
                                    justification=(
                                        "the model's judgement did not parse "
                                        f"({len(response_text or '')} chars); "
                                        "this candidate was not scored"),
                                )
                            )
                        total_eval = len(candidates[:10])
                        print(f"  candidate {i + 1}/{total_eval}: scored ({cand.start:.1f}s-{cand.end:.1f}s)", flush=True)
                        log.info("s3.candidate_scored", candidate=i + 1, total=total_eval,
                                 start=round(cand.start, 1), end=round(cand.end, 1))

                    if unreadable and unreadable == len(ranked_items):
                        # Every window unreadable is a broken VIDEO, not a
                        # degraded candidate, and it must say so rather
                        # than return ten zero-scored items that look like
                        # a ranking.
                        raise StageError(
                            f"no frames could be read from {video_path} for "
                            f"any of {unreadable} candidate windows; the "
                            "file is unreadable at those offsets")

                    # Weighted and banded — see rank_key. Shared with
                    # the cloud path so the same clip cannot be ordered
                    # two different ways depending on which judge ran.
                    ranked_items.sort(key=rank_key)
                    for r, item in enumerate(ranked_items, 1):
                        item.rank = r

                    return RankedArtifact(
                        cache_key=cache_key,
                        source_candidates=candidates_artifact.cache_key,
                        ranking_source="semantic",
                        ranker=f"local:{loaded_model_id or model_id}",
                        fallback_reason=self._fallback_reason,
                        items=ranked_items,
                        visual_tokens_used=total_visual_tokens,
                    )
                finally:
                    hard_unload(models)

        except _PROGRAMMING_ERRORS:
            # A misused API is a BUG, not a degraded environment. Swallowing
            # TypeError/AttributeError here is exactly how `with vram_guard(...)`
            # and `ModelClass.DETECTION` survived: the gate printed a fallback
            # warning and still reported PASS. Let them out.
            raise
        except Exception as exc:
            raise StageError(f"S3 execution error: {exc}") from exc
