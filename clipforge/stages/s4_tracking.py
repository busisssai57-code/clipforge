"""S4 Spatial Tracking & Active Speaker Camera Path stage."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

from clipforge.errors import RetryableStageError, StageError
from clipforge.ffmpeg import probe
from clipforge.gpu import ModelClass, gpu_session, hard_unload
from clipforge.log import get_logger

#: See s3_semantic â€” these always mean the code is wrong, never that the
#: machine lacks a model, so they must never become a quality fallback.
_PROGRAMMING_ERRORS = (TypeError, AttributeError, NameError)
from clipforge.schemas.campath import CamPathArtifact, CropFrame, SpeakerAssignment
from clipforge.schemas.ranking import RankedArtifact
from clipforge.schemas.transcript import TranscriptArtifact
from clipforge.stages.base import Stage

log = get_logger(__name__)


class OneEuroFilter:
    """Low-latency signal filter for smoothing camera pan & crop coordinates."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.007, d_cutoff: float = 1.0) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: float | None = None
        self.dx_prev: float = 0.0

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, dt: float = 1.0 / 30.0) -> float:
        if self.x_prev is None:
            self.x_prev = x
            return x

        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat


def _even(val: int) -> int:
    """Ensure val is an even integer (spec Â§S4)."""
    val = max(0, int(val))
    return val if val % 2 == 0 else val - 1


def _detect_shots(video_path: Path, start_s: float, duration_s: float,
                  fps: float, *, threshold: float = 0.30) -> list[int]:
    """Shot-boundary frame indices (window-relative) via ffmpeg scene scores.

    Input-side ``-ss`` is deliberate: it decodes only the window, and the
    select filter's ``pts_time`` then counts from the seek point, which is
    exactly the window-relative time base the caller needs. Detection
    granularity does not need the frame-exactness that made S6 use
    output-side seeking.
    """
    import re  # noqa: PLC0415
    import subprocess  # noqa: PLC0415

    from clipforge.ffmpeg import require_binary  # noqa: PLC0415

    try:
        proc = subprocess.run(
            [str(require_binary("ffmpeg")), "-nostdin", "-hide_banner",
             "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
             "-i", str(video_path),
             "-vf", f"select='gt(scene,{threshold})',metadata=print",
             "-an", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=300)
    except subprocess.TimeoutExpired:
        # Shot detection is an ENHANCEMENT to framing: a wedged decode
        # must not hang the stage. No boundaries = one shot, which the
        # subject-selection code handles as the ordinary case.
        log.warning("s4.shot_detect_timeout", video=str(video_path),
                    note="treating the window as a single shot")
        return []
    times = re.findall(r"pts_time:([\d.]+)", proc.stderr or "")
    out: list[int] = []
    last = -10.0
    for t in times:
        tv = float(t)
        # Merge boundaries closer than 0.4 s — flash frames and dissolves
        # produce clustered detections that would create unusable
        # micro-shots no editor would cut.
        if tv - last >= 0.4:
            out.append(int(round(tv * fps)))
            last = tv
    return out


#: A word worth pushing in on: an amount, a count, or a number. These are
#: the beats a viewer's eye should be pulled to, and they are the only ones
#: the transcript can identify without a model.
_PUNCH_WORD = re.compile(r"[\d$€£]")


def _mar_from_landmarks(pts: Any) -> float | None:
    """Mouth Aspect Ratio from FaceMesh-topology landmarks.

    MAR = vertical inner-lip opening / horizontal mouth width, using the
    canonical indices: 13 (upper inner lip), 14 (lower inner lip), 61 and
    291 (mouth corners). Open vowels spike it; bilabials (/m/, /p/) drive it
    to ~0 — so its VARIANCE over time separates a talker from a nodder,
    which no amount of face *detection* can do (blueprint §5.1).
    """
    try:
        up, low, left, right = pts[13], pts[14], pts[61], pts[291]
    except (IndexError, TypeError):
        return None
    width = math.hypot(right.x - left.x, right.y - left.y)
    if width <= 1e-6:
        return None
    return math.hypot(low.x - up.x, low.y - up.y) / width


#: MAR sampling rate. The panel measured 6 Hz starving series below the
#: minimum (a visibly articulating talker scored 0.0 and a BACK OF A HEAD
#: was framed instead) and aliasing syllable motion: talker-ordering flips
#: vs a 30 Hz reference fell from 10/24 shots at 6 Hz to 4/24 at 15 Hz.
MAR_SAMPLE_HZ = 15.0

#: Deltas whose samples are further apart than this many nominal sample
#: periods are detection-DROPOUT bridges, not lip motion; the panel measured
#: a flickering profile face outscoring an articulating one 3.8x purely on
#: cross-gap jumps.
MAR_MAX_GAP_PERIODS = 1.6

#: Face crop geometry: top fraction of the person box, half-width around the
#: head-keypoint anchor. Full-width crops credited a neighbour's face 5.1%
#: of the time in crowded frames (anatomically impossible MARs on a profile).
FACE_CROP_TOP_FRAC = 0.45
FACE_CROP_HALF_WIDTH_FRAC = 0.30
MIN_FACE_CROP_PX = 40


def _mar_activity(series: list[tuple[float, float]],
                  sample_period: float) -> float:
    """Speech activity in |ΔMAR| PER SECOND, dropout-gaps excluded.

    Change-per-time, not change-per-sample: per-sample means shift with the
    sampling rate, so a tau tuned at 6 Hz silently meant something else at
    15 Hz. Pairs separated by more than ``MAR_MAX_GAP_PERIODS`` nominal
    periods are detection dropouts and contribute nothing — bridging them
    injected large fake deltas. Fewer than 3 valid pairs is not evidence.
    """
    if len(series) < 3:
        return 0.0
    max_gap = MAR_MAX_GAP_PERIODS * sample_period
    num = den = 0.0
    pairs = 0
    for (t0, m0), (t1, m1) in zip(series, series[1:]):
        dt = t1 - t0
        if 0 < dt <= max_gap:
            num += abs(m1 - m0)
            den += dt
            pairs += 1
    if pairs < 3 or den <= 0:
        return 0.0
    return num / den


def _select_shot_subject(mar_series: dict[int, list[tuple[float, float]]],
                         presence: dict[int, float],
                         tau: float, sample_period: float) -> tuple[int, str]:
    """(track_id, method) for one shot: MAR speech activity, else presence.

    The MAR pick must clear ``tau`` (per-second units) AND beat the
    runner-up by 1.3x — two people trading lines within one shot is
    genuinely ambiguous, and presence is more stable than flip-flopping on
    a nose-length lead.
    """
    scored = sorted(((_mar_activity(s, sample_period), tid)
                     for tid, s in mar_series.items() if s),
                    reverse=True)
    if scored and scored[0][0] >= tau:
        best, runner = scored[0], (scored[1] if len(scored) > 1 else None)
        if runner is None or best[0] >= 1.3 * runner[0]:
            return best[1], "mar"
    return max(presence, key=presence.get), "presence"


def _is_resource_failure(exc: Exception) -> bool:
    """True when the machine could not do the work NOW.

    These must escape the stage (retryable) rather than degrade to a
    centre crop: masking a CUDA OOM as a framing choice reports success
    on a run whose headline feature silently did not happen. Matched
    structurally where possible; the RuntimeError case falls back to
    message sniffing because torch raises plain RuntimeError for most
    device failures.
    """
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, OSError):
        return True
    name = type(exc).__name__
    if name == "OutOfMemoryError":          # torch.cuda.OutOfMemoryError
        return True
    if isinstance(exc, RuntimeError):
        msg = str(exc).lower()
        return any(m in msg for m in
                   ("cuda", "out of memory", "device-side", "cudnn",
                    "cublas", "hip"))
    return False


def _face_crop_bounds(box: tuple[float, float, float, float],
                      head_x: float | None = None) -> tuple[int, int, int, int]:
    """(y1, y2, x1, x2) of the face crop inside a person box.

    Top ``FACE_CROP_TOP_FRAC`` of the box vertically; when a head keypoint
    is available, narrowed to ±``FACE_CROP_HALF_WIDTH_FRAC`` of the box
    width around it (floored at ``MIN_FACE_CROP_PX``). Pure so the geometry
    is testable without MediaPipe — the full-width crop is exactly the bug
    the panel measured (neighbour's face credited 5.1% of crowded samples).
    """
    x1, y1, x2, y2 = (int(v) for v in box)
    h = y2 - y1
    fy2 = y1 + max(1, int(h * FACE_CROP_TOP_FRAC))
    cx1, cx2 = x1, x2
    if head_x is not None:
        half = max(MIN_FACE_CROP_PX // 2,
                   int((x2 - x1) * FACE_CROP_HALF_WIDTH_FRAC))
        cx1 = max(x1, int(head_x) - half)
        cx2 = min(x2, int(head_x) + half)
    return max(0, y1), fy2, max(0, cx1), cx2


class _FaceMarSampler:
    """Lazy MediaPipe FaceLandmarker over per-person face crops.

    CPU-only (XNNPACK), so it does not touch the VRAM Law. Every failure
    path returns None rather than raising: MAR is an ENHANCEMENT to subject
    selection, and a missing model file or an undetectable face must degrade
    to the presence heuristic, not kill the stage.
    """

    def __init__(self, model_path: Path | None) -> None:
        self._path = model_path
        self._lm: Any = None
        self._dead = model_path is None or not model_path.exists()
        if self._dead:
            log.warning("s4.mar_unavailable",
                        model=str(model_path),
                        note="face_landmarker.task missing - subject "
                             "selection falls back to presence only")

    def mar(self, frame_bgr: Any, box: tuple[float, float, float, float],
            head_x: float | None = None) -> float | None:
        if self._dead:
            return None
        try:
            import cv2  # noqa: PLC0415
            import mediapipe as mp  # noqa: PLC0415

            if self._lm is None:
                from mediapipe.tasks import python as mp_python  # noqa: PLC0415
                from mediapipe.tasks.python import vision  # noqa: PLC0415

                self._lm = vision.FaceLandmarker.create_from_options(
                    vision.FaceLandmarkerOptions(
                        base_options=mp_python.BaseOptions(
                            model_asset_path=str(self._path)),
                        running_mode=vision.RunningMode.IMAGE, num_faces=1))
            ry1, fy2, cx1, cx2 = _face_crop_bounds(box, head_x)
            crop = frame_bgr[ry1:fy2, cx1:cx2]
            if (crop.shape[0] < MIN_FACE_CROP_PX
                    or crop.shape[1] < MIN_FACE_CROP_PX):
                return None       # face too small for reliable lip geometry
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            res = self._lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                           data=rgb))
            if not res.face_landmarks:
                return None
            return _mar_from_landmarks(res.face_landmarks[0])
        except Exception as exc:  # noqa: BLE001 - enhancement, never fatal
            if not getattr(self, "_warned", False):
                self._warned = True
                log.warning("s4.mar_error", error=f"{type(exc).__name__}: "
                            f"{str(exc)[:120]}")
            return None

    def close(self) -> None:
        if self._lm is not None:
            try:
                self._lm.close()
            except Exception:  # noqa: BLE001
                pass
            self._lm = None


def _apply_punch_ins(frames: list[CropFrame], transcript: Any, start_s: float,
                     end_s: float, fps: float, src_w: int, src_h: int,
                     scale: float, hold_s: float) -> list[CropFrame]:
    """Shrink the crop around emphasis beats — a push-in, not a jump cut.

    Ramp in over 0.25 s, hold, ramp out over 0.35 s. The asymmetry matters:
    a fast push and a slower release reads as intent; symmetric reads as a
    glitch. Aspect ratio is preserved exactly, and the crop is re-centred on
    its own midpoint so the subject does not slide during the zoom.
    """
    abs_offset = float(getattr(transcript, "abs_offset_s", 0.0) or 0.0)
    beats: list[float] = []
    for seg in getattr(transcript, "segments", []):
        for w in getattr(seg, "words", []):
            if w.start is None:
                continue
            t = float(w.start) - abs_offset           # media time
            if not (start_s <= t < end_s):
                continue
            if _PUNCH_WORD.search(str(w.text)):
                # One push per ~2 s: consecutive numbers ("five hundred
                # thousand") are one beat, not three.
                if not beats or t - beats[-1] > 2.0:
                    beats.append(t)
    if not beats:
        return frames

    ramp_in, ramp_out = 0.25, 0.35
    by_frame = {f.frame: f for f in frames}
    for beat in beats:
        f0 = int(round((beat - start_s) * fps))
        n_in = max(1, int(ramp_in * fps))
        n_hold = max(1, int(hold_s * fps))
        n_out = max(1, int(ramp_out * fps))
        for k in range(-n_in, n_hold + n_out):
            idx = f0 + k
            src = by_frame.get(idx)
            if src is None:
                continue
            if k < 0:                       # ramping in
                t = (k + n_in) / n_in
            elif k < n_hold:                # held
                t = 1.0
            else:                           # ramping out
                t = max(0.0, 1.0 - (k - n_hold) / n_out)
            z = 1.0 - scale * t
            new_w = _even(int(src.w * z))
            new_h = _even(int(src.h * z))
            # Re-centre on the ORIGINAL crop's midpoint, then clamp so the
            # tighter box never leaves the frame.
            cx = src.x + src.w / 2.0
            cy = src.y + src.h / 2.0
            new_x = _even(int(max(0, min(src_w - new_w, cx - new_w / 2.0))))
            new_y = _even(int(max(0, min(src_h - new_h, cy - new_h / 2.0))))
            by_frame[idx] = CropFrame(frame=idx, x=new_x, y=new_y,
                                      w=new_w, h=new_h)
    log.info("s4.punch_ins", beats=len(beats), scale=scale)
    return [by_frame[k] for k in sorted(by_frame)]


class S4Tracking(Stage[CamPathArtifact]):
    """Stage S4: Spatial Tracking, Active Speaker Detection, and Virtual Camera Path.

    Runs YOLO11-pose + ByteTrack on the clip window to track human centroids,
    correlates track movements with active speaker diarization turns, and
    generates a smoothed 9:16 portrait camera path using a One-Euro filter.
    """

    name: str = "s4_tracking"
    version: str = "6"
    vram_budget_gb: float = 3.0
    wall_budget_s: float = 120.0
    artifact_type = CamPathArtifact

    def _execute(
        self,
        *,
        cache_key: str,
        params: dict[str, Any],
        ranked_artifact: RankedArtifact,
        transcript_artifact: TranscriptArtifact | None = None,
        video_path: Path | None = None,
        start_s: float = 0.0,
        end_s: float = 30.0,
        **kwargs: Any,
    ) -> CamPathArtifact:
        # Fallback values & parameters
        asd_tau = params.get("asd_confidence_tau", 0.5)
        deadzone_frac = params.get("deadzone_frac", 0.02)
        max_pan_px = params.get("max_pan_px_per_frame", 40.0)
        one_euro_min_cutoff = params.get("one_euro_min_cutoff", 1.0)
        one_euro_beta = params.get("one_euro_beta", 0.007)
        punch_scale = float(params.get("punch_scale", 0.0))
        punch_hold_s = float(params.get("punch_hold_s", 0.9))
        mar_tau = float(params.get("mar_activity_tau", 0.18))
        # An INPUT, not a param: the model's IDENTITY (filename) is in
        # params and hashes into the cache key; its absolute location is
        # machine-plumbing and must not invalidate caches across moves.
        face_model_path = kwargs.get("face_model_path")

        # Source geometry is PROBED, never assumed. Defaulting to 1920x1080
        # when cv2 is absent wrote fabricated dimensions into the artifact,
        # and S6 built its crop from them: on the 1280x720 fixture that meant
        # crop height 1080 > frame height 720, so ffmpeg produced zero frames
        # and every clip render failed. A wrong number that looks plausible is
        # worse than no number â€” ffprobe is already a hard dependency here.
        if not video_path or not Path(video_path).exists():
            raise StageError(f"S4 requires the source media: {video_path}")
        info = probe(Path(video_path))
        src_w, src_h = int(info.width), int(info.height)
        fps = float(info.fps)
        fps_rational = info.fps_rational
        if src_w <= 0 or src_h <= 0:
            raise StageError(
                f"S4: probe returned degenerate geometry {src_w}x{src_h} "
                f"for {video_path}")

        total_duration = max(0.1, end_s - start_s)
        num_frames = int(total_duration * fps)

        # Target 9:16 aspect ratio crop dimensions
        crop_h = _even(src_h)
        crop_w = _even(int(crop_h * (9.0 / 16.0)))
        if crop_w > src_w:
            crop_w = _even(src_w)
            crop_h = _even(int(crop_w * (16.0 / 9.0)))

        center_x = (src_w - crop_w) // 2

        # Default fallback framing: static center crop
        def _build_center_crop(reason: str) -> CamPathArtifact:
            raise StageError(f"S4 Tracking failed (fallback disabled to enforce AI processing): {reason}")

        # Perform tracking run under VRAM guard
        try:
            # gpu_session, not vram_guard â€” and ModelClass.POSE, not
            # DETECTION, which does not exist (members are ASR, VL, POSE).
            # The AttributeError fired at ARGUMENT EVALUATION, before the
            # TypeError from misusing a plain function as a context manager
            # could even be reached, and the blanket `except Exception` turned
            # both into a silent center-crop. The gate printed
            # `s4.center_fallback reason='S4 tracking error: DETECTION'` on
            # every run and still reported ai 12/12 PASSED: S4's tracking path
            # has never actually executed under a gate.
            with gpu_session(ModelClass.POSE, self.vram_budget_gb):
                try:
                    from ultralytics import YOLO
                except ImportError as err:
                    return _build_center_crop(f"ultralytics YOLO not available: {err}")

                # Prefer the resolved workspace path (input) over the bare
                # filename (params): the filename hashes into the cache key
                # as the model's identity, while the path is plumbing that
                # ultralytics would otherwise resolve against the CWD and
                # "fix" by downloading a duplicate.
                pose_model_name = (kwargs.get("pose_model_path")
                                   or params.get("pose_model",
                                                 "yolo11m-pose.pt"))
                # The dict is the ONLY strong reference: a bare local would
                # survive hard_unload's pop, so gc/empty_cache would free
                # nothing and the pose weights would still be resident when
                # the next stage's vram_guard probes the driver.
                models: dict[str, Any] = {}
                try:
                    models["model"] = YOLO(pose_model_name)
                except Exception as load_err:
                    hard_unload(models)
                    return _build_center_crop(f"failed to load YOLO pose model: {load_err}")

                try:
                    # ---- ACTUAL tracking. The previous body loaded the
                    # model and never called it: `raw_x = last_target_x`,
                    # a deadzone check of that value against itself, a pan
                    # clamp of a zero delta, and a One-Euro filter smoothing
                    # a constant — then framing_mode="speaker" with a
                    # hardcoded confidence of 0.85 and used_fallback=False.
                    # A fabricated audit record wearing the model's VRAM as
                    # a costume. Every number below is measured or absent.
                    import cv2  # noqa: PLC0415 - ships with ultralytics

                    cap = cv2.VideoCapture(str(video_path))
                    if not cap.isOpened():
                        return _build_center_crop(
                            f"cv2 cannot open {video_path}")
                    try:
                        # Sample every STRIDE-th frame: pose inference at
                        # full rate is wasted work when the path is then
                        # smoothed by a filter whose cutoff is ~1 Hz.
                        stride = max(1, int(round(fps / MAR_SAMPLE_HZ)))
                        cap.set(cv2.CAP_PROP_POS_MSEC, start_s * 1000.0)
                        mar_sampler = _FaceMarSampler(
                            Path(face_model_path) if face_model_path
                            else None)
                        # frame_idx -> list[(tid, face_x, area, conf, mar)]
                        samples_raw: dict[
                            int, list[tuple[int, float, float, float,
                                            float | None]]] = {}
                        tracks_seen: set[int] = set()
                        for f in range(0, num_frames, stride):
                            ok, frame = cap.read()
                            if not ok:
                                break
                            # Skip the frames inside the stride window.
                            for _ in range(stride - 1):
                                cap.grab()
                            results = models["model"].track(
                                frame, persist=True, verbose=False,
                                classes=[0], conf=0.25)
                            if not results or results[0].boxes is None:
                                continue
                            boxes = results[0].boxes
                            if boxes.id is None:
                                continue
                            # Head keypoints when the pose model provides
                            # them. Framing on the BODY centroid was one of
                            # the two reasons the output read as amateur: a
                            # standing person's box centre is their sternum,
                            # so faces sat high and off-axis. COCO keypoints
                            # 0-4 are nose/eyes/ears; their mean is the face
                            # anchor, with the box's upper quarter as the
                            # fallback when the head is not visible.
                            kpts = getattr(results[0], "keypoints", None)
                            kxy = (kpts.xy.tolist()
                                   if kpts is not None and kpts.xy is not None
                                   else None)
                            kconf = (kpts.conf.tolist()
                                     if kpts is not None
                                     and kpts.conf is not None else None)
                            for i, (box, tid, conf) in enumerate(zip(
                                    boxes.xyxy.tolist(),
                                    boxes.id.int().tolist(),
                                    boxes.conf.tolist())):
                                x1, y1, x2, y2 = box
                                face_x = (x1 + x2) / 2.0
                                if kxy is not None and i < len(kxy):
                                    head = [kxy[i][j][0] for j in range(5)
                                            if j < len(kxy[i])
                                            and kconf is not None
                                            and i < len(kconf)
                                            and j < len(kconf[i])
                                            and kconf[i][j] > 0.3
                                            and kxy[i][j][0] > 0]
                                    if head:
                                        face_x = sum(head) / len(head)
                                tracks_seen.add(int(tid))
                                # MAR sampled on the same frame the tracker
                                # saw, crop anchored on the head keypoint —
                                # blueprint §5.1: the lip-geometry series is
                                # what turns "a person is here" into "this
                                # person is TALKING".
                                mar = mar_sampler.mar(frame, (x1, y1, x2, y2),
                                                      head_x=face_x)
                                samples_raw.setdefault(f, []).append(
                                    (int(tid), face_x,
                                     (x2 - x1) * (y2 - y1), float(conf),
                                     mar))
                    finally:
                        cap.release()
                        mar_sampler.close()

                    if not samples_raw:
                        return _build_center_crop(
                            "no persons detected in the clip window")

                    # SHOT-AWARE framing — the difference between "tracking
                    # demo" and "an editor framed this". The previous design
                    # followed the best subject per sample with one global
                    # smoother, which PANS ACROSS CUTS: when the source cuts
                    # to a person standing elsewhere, the crop slid over in
                    # ~0.3 s. Human editors never do that — the crop CUTS
                    # with the edit and holds STILL inside each shot. So:
                    # detect shot boundaries (ffmpeg scene scores), then per
                    # shot pick one subject (modal track weighted by
                    # presence), and lock a single median-centred crop unless
                    # the subject genuinely walks (spread > 8% of width), in
                    # which case follow with the smoother WITHIN the shot
                    # only. The One-Euro filter resets at every boundary.
                    boundaries = _detect_shots(video_path, start_s,
                                               total_duration, fps)
                    shot_edges = sorted({0, num_frames}
                                        | {min(num_frames, max(0, b))
                                           for b in boundaries})
                    shots = [(a, b) for a, b in zip(shot_edges,
                                                    shot_edges[1:]) if b > a]

                    n_samples = max(1, len(range(0, num_frames, stride)))
                    coverage = len(samples_raw) / n_samples
                    all_confs = [d[3] for dets in samples_raw.values()
                                 for d in dets[:1]]
                    mean_conf = sum(all_confs) / max(1, len(all_confs))
                    ids_used = []
                    for f_idx in sorted(samples_raw):
                        best = max(samples_raw[f_idx],
                                   key=lambda d: d[2] * d[3])
                        ids_used.append(best[0])
                    switches = sum(1 for a, b in zip(ids_used, ids_used[1:])
                                   if a != b)
                    primary_id = (max(set(ids_used), key=ids_used.count)
                                  if ids_used else -1)

                    turns = (transcript_artifact.turns
                             if transcript_artifact is not None else [])

                    if mean_conf < asd_tau or coverage < 0.3:
                        # Below τ, or persons are absent from most of the
                        # window: the spec's documented fallback framing.
                        return _build_center_crop(
                            f"mean detection conf {mean_conf:.2f} < τ "
                            f"{asd_tau} or person coverage {coverage:.0%} "
                            "< 30% of samples")

                    # ---- Solve one crop path PER SHOT -------------------
                    frames = []
                    y_lock = _even(max(0, (src_h - crop_h) // 2))
                    center_default = float(center_x)
                    prev_shot_x: float = center_default
                    locked_shots = moving_shots = 0
                    mar_shots = presence_shots = 0

                    for shot_a, shot_b in shots:
                        in_shot = {f: dets for f, dets in samples_raw.items()
                                   if shot_a <= f < shot_b}
                        if not in_shot:
                            # No person seen in this shot: hold the previous
                            # shot's framing rather than yanking to centre —
                            # b-roll inserts read better without a jump.
                            for f in range(shot_a, shot_b):
                                frames.append(CropFrame(
                                    frame=f,
                                    x=_even(int(max(0, min(src_w - crop_w,
                                                           prev_shot_x)))),
                                    y=y_lock, w=_even(crop_w),
                                    h=_even(crop_h)))
                            continue

                        # One subject per shot. MAR speech activity picks the
                        # talker when the lip-geometry evidence is decisive
                        # (blueprint §5.1); accumulated presence (area x
                        # conf) is the fallback when faces are too small,
                        # occluded, or nobody clears the activity floor.
                        presence: dict[int, float] = {}
                        mar_series: dict[int, list[tuple[float, float]]] = {}
                        for f_idx in sorted(in_shot):
                            for tid, _fx, area, conf, mar in in_shot[f_idx]:
                                presence[tid] = presence.get(tid, 0.0) \
                                    + area * conf
                                if mar is not None:
                                    mar_series.setdefault(tid, []).append(
                                        (f_idx / fps, mar))
                        shot_tid, method = _select_shot_subject(
                            mar_series, presence, mar_tau,
                            sample_period=stride / fps)
                        if method == "mar":
                            mar_shots += 1
                        else:
                            presence_shots += 1
                        targets = []
                        for f_idx in sorted(in_shot):
                            dets = in_shot[f_idx]
                            mine = [d for d in dets if d[0] == shot_tid]
                            pick = (mine[0] if mine
                                    else max(dets, key=lambda d: d[2] * d[3]))
                            targets.append((f_idx, pick[1]))

                        xs = sorted(t[1] for t in targets)
                        spread = xs[-1] - xs[0]
                        median_face = xs[len(xs) // 2]

                        if spread <= 0.08 * src_w:
                            # STATIC shot: one locked crop, centred on the
                            # median face position. No smoothing, no drift —
                            # this is what a human editor does.
                            locked_shots += 1
                            x_shot = median_face - crop_w / 2.0
                            x_shot = max(0.0, min(float(src_w - crop_w),
                                                  x_shot))
                            for f in range(shot_a, shot_b):
                                frames.append(CropFrame(
                                    frame=f, x=_even(int(x_shot)), y=y_lock,
                                    w=_even(crop_w), h=_even(crop_h)))
                            prev_shot_x = x_shot
                        else:
                            # MOVING shot: follow, but only inside the shot,
                            # with a fresh smoother — carrying filter state
                            # across a boundary is how pans leak across cuts.
                            moving_shots += 1
                            filter_x = OneEuroFilter(
                                min_cutoff=one_euro_min_cutoff,
                                beta=one_euro_beta)
                            dt = 1.0 / max(1.0, fps)
                            deadzone_px = deadzone_frac * src_w
                            s_idx = 0
                            last_x = targets[0][1] - crop_w / 2.0
                            for f in range(shot_a, shot_b):
                                while (s_idx + 1 < len(targets)
                                       and targets[s_idx + 1][0] <= f):
                                    s_idx += 1
                                if s_idx + 1 < len(targets):
                                    f0, x0 = targets[s_idx]
                                    f1, x1 = targets[s_idx + 1]
                                    w01 = (f - f0) / max(1, f1 - f0)
                                    cx = x0 + (x1 - x0) * w01
                                else:
                                    cx = targets[s_idx][1]
                                raw_x = cx - crop_w / 2.0
                                if abs(raw_x - last_x) < deadzone_px:
                                    raw_x = last_x
                                delta = raw_x - last_x
                                if abs(delta) > max_pan_px:
                                    raw_x = last_x + math.copysign(
                                        max_pan_px, delta)
                                smooth_x = filter_x.filter(raw_x, dt=dt)
                                last_x = smooth_x
                                cl_x = max(0, min(src_w - crop_w,
                                                  int(smooth_x)))
                                frames.append(CropFrame(
                                    frame=f, x=_even(cl_x), y=y_lock,
                                    w=_even(crop_w), h=_even(crop_h)))
                            prev_shot_x = last_x

                    frames.sort(key=lambda fr: fr.frame)

                    # PUNCH-INS. A flat crop for 60 s reads as a security
                    # camera; editors push in on the beat a point lands.
                    # Beats come from the transcript, not a timer: words
                    # carrying a number or amount are where the eye should
                    # be pulled. The crop w/h shrink toward the same centre
                    # and S6 scales back to 1080x1920, so a smaller crop IS
                    # a zoom — no extra filter, no resampling twice.
                    if punch_scale > 0 and transcript_artifact is not None:
                        frames = _apply_punch_ins(
                            frames, transcript_artifact, start_s, end_s, fps,
                            src_w, src_h, punch_scale, punch_hold_s)
                    # HONEST assignment, now with a real ASD signal. "mar"
                    # shots were chosen by measured lip-motion — visual
                    # active-speaker detection, not audio identity, and the
                    # label says which. Audio diarization turns still join
                    # the record when pyannote runs; until then MAR is the
                    # blueprint's (§5.1) visual half standing alone.
                    asd_label = ("MAR_VISUAL" if mar_shots > 0
                                 else "NO_DIARIZATION")
                    if turns:
                        # No turn↔track correlation is computed yet, so a
                        # per-turn assignment claiming the modal track would
                        # be an invented mapping. used_fallback=True until
                        # real correlation lands with pyannote (panel found
                        # this branch dormant-but-dishonest).
                        assignments = [
                            SpeakerAssignment(
                                turn_speaker=t.speaker,
                                track_id=primary_id,
                                confidence=mean_conf,
                                used_fallback=True)
                            for t in turns]
                    else:
                        assignments = [SpeakerAssignment(
                            turn_speaker=asd_label,
                            track_id=primary_id,
                            confidence=mean_conf,
                            used_fallback=mar_shots == 0)]

                    log.info("s4.shots", shots=len(shots),
                             locked=locked_shots, moving=moving_shots,
                             mar_selected=mar_shots,
                             presence_selected=presence_shots,
                             boundaries=len(boundaries))
                    log.info("s4.tracked", modal_track=primary_id,
                             tracks_seen=len(tracks_seen),
                             subject_switches=switches,
                             mean_conf=round(mean_conf, 3),
                             coverage=round(coverage, 3),
                             samples=len(samples_raw))
                    return CamPathArtifact(
                        cache_key=cache_key,
                        source_ranking=ranked_artifact.cache_key,
                        clip_start=start_s,
                        clip_end=end_s,
                        framing_mode="speaker",
                        frames=frames,
                        assignments=assignments,
                        mar_shots=mar_shots,
                        presence_shots=presence_shots,
                        src_width=src_w,
                        src_height=src_h,
                        src_fps_rational=fps_rational,
                    )
                finally:
                    hard_unload(models)

        except _PROGRAMMING_ERRORS:
            # A misused API is a BUG, not a degraded environment â€” see S3.
            raise
        except Exception as exc:
            # The catch-all exists for one class of cause only: a DEGRADED
            # ENVIRONMENT (ultralytics missing, a codec cv2 cannot open, a
            # model file that fails to deserialize). Resource failures are
            # NOT that: CUDA OOM, a dying disk, a decoder crash mean this
            # machine could not do the work NOW, and "quietly frame the
            # centre and report success" is exactly how the fake-tracking
            # body survived undetected. RetryableStageError so the operator
            # sees the real cause and can re-run once it clears.
            if _is_resource_failure(exc):
                raise RetryableStageError(
                    f"S4 tracking failed on a resource error, refusing to "
                    f"mask it as a centre crop: {type(exc).__name__}: {exc}",
                    stage=self.name) from exc
            raise StageError(f"S4 tracking error: {exc}") from exc

