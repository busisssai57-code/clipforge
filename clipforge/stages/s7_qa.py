"""S7 — mechanical quality gate over the rendered clip.

Every check below is measured from the FILE ON DISK by ffprobe/ffmpeg.
Nothing is trusted from the producing stages' artifacts except as the
*expected* side of a comparison — S6 reporting a duration is a claim, and
this stage's job is to check claims. That distinction exists because this
project has already shipped a stage that consumed real VRAM while
fabricating its audit record; a QA gate that reads the producer's own
numbers back to it would have blessed that stage too.

CPU-only, deterministic, no models. A clip that fails any ``severity="fail"``
check is moved to ``clips/rejected/`` by the caller; warnings ship but are
recorded in the artifact so an operator can audit deviations later.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from clipforge.errors import RetryableStageError
from clipforge.ffmpeg import require_binary
from clipforge.log import get_logger
from clipforge.schemas.qa import QAArtifact, QACheck
from clipforge.stages.base import Stage

log = get_logger(__name__)

#: Spec §S2: candidate windows are 30-60 s. One second of tolerance each way
#: covers sentence-aligned boundaries and container rounding.
MIN_CLIP_S = 29.0
MAX_CLIP_S = 61.0

#: True peak must stay under this AFTER the AAC encode. The loudnorm target
#: is -1.5 dBTP; AAC overshoots by a few tenths, so the SHIP ceiling is
#: -1.0 — anything above that risks audible inter-sample clipping on
#: phone speakers, which is where these clips live.
SHIP_TP_CEILING = -1.0

#: Integrated loudness tolerance around the target before a WARN, and the
#: band beyond which it is a FAIL. A headroom-limited source legitimately
#: lands a couple of LU quiet (recorded deviation); 4+ LU means the
#: normalizer did not run at all.
LOUDNESS_WARN_LU = 1.5
LOUDNESS_FAIL_LU = 4.0

#: A black or frozen run longer than this fails the clip; silence longer
#: than this fails the clip. Values chosen for short-form: 5 s of any of
#: these in a 30-60 s clip is a dead clip.
MAX_BLACK_RUN_S = 1.5
MAX_FREEZE_RUN_S = 5.0
MAX_SILENCE_RUN_S = 5.0


#: Ceiling on any single QA probe. Decoding a 60 s clip three times over
#: takes seconds; a minute of grace covers a cold disk. Without a timeout,
#: one ffmpeg wedged on a torn file hangs the unattended `bta agent` loop
#: forever — and QA is the LAST place a hang should be possible, because it
#: runs on every clip.
QA_PROBE_TIMEOUT_S = 120.0


def _run(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace",
                              timeout=QA_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        tail = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, "", (f"timed out after {QA_PROBE_TIMEOUT_S:.0f}s: "
                         f"{(tail or '')[-300:]}")
    return proc.returncode, proc.stdout or "", proc.stderr or ""


class S7QualityGate(Stage[QAArtifact]):
    name = "s7_qa"
    version = "1"
    artifact_type = QAArtifact

    def _execute(self, *, cache_key: str, params: dict[str, Any],
                 **inputs: Any) -> QAArtifact:
        clip = inputs.get("clip_artifact")
        subtitles = inputs.get("subtitle_artifact")
        campath = inputs.get("campath_artifact")
        if clip is None:
            raise RetryableStageError("S7 requires clip_artifact",
                                      stage=self.name)

        checks: list[QACheck] = []

        def check(name: str, severity: str, passed: bool,
                  measured: str, expected: str) -> None:
            checks.append(QACheck(name=name, severity=severity,  # type: ignore[arg-type]
                                  passed=bool(passed), measured=measured,
                                  expected=expected))

        clip_path = Path(clip.clip_path)
        out_w = int(params.get("width", 1080))
        out_h = int(params.get("height", 1920))
        target_i = float(params.get("loudness_i", -14.0))

        # ---- 1. Integrity: the bytes on disk are the bytes S6 hashed -----
        exists = clip_path.exists()
        check("file-exists", "fail", exists, str(clip_path),
              "rendered clip present on disk")
        if not exists:
            return self._finish(cache_key, clip, checks)

        digest = hashlib.sha256(clip_path.read_bytes()).hexdigest()
        check("sha256-integrity", "fail", digest == clip.clip_sha256,
              digest[:16], f"{clip.clip_sha256[:16]} (from ClipArtifact); "
              "mismatch means truncation or post-render tampering")

        # require_binary, not find_binary: the latter returns None on a
        # machine without ffmpeg, which then reaches subprocess as a
        # TypeError from deep inside the QA gate. require_binary raises
        # the actionable install/CLIPFORGE_FFMPEG_DIR message instead.
        ffprobe = require_binary("ffprobe")
        ffmpeg = require_binary("ffmpeg")

        # ---- 2. Container geometry, codecs, duration ---------------------
        rc, out, _err = _run([ffprobe, "-v", "error", "-print_format", "json",
                              "-show_format", "-show_streams",
                              str(clip_path)])
        if rc != 0:
            check("probe", "fail", False, f"ffprobe rc={rc}",
                  "clip must be probeable")
            return self._finish(cache_key, clip, checks)
        info = json.loads(out)
        streams = {s["codec_type"]: s for s in info.get("streams", [])}
        video = streams.get("video", {})
        audio = streams.get("audio", {})

        check("video-stream", "fail", bool(video),
              video.get("codec_name", "absent"), "one h264 video stream")
        check("audio-stream", "fail", bool(audio),
              audio.get("codec_name", "absent"), "one aac audio stream")
        if video:
            check("video-codec", "fail", video.get("codec_name") == "h264",
                  str(video.get("codec_name")), "h264")
            check("pixel-format", "fail", video.get("pix_fmt") == "yuv420p",
                  str(video.get("pix_fmt")),
                  "yuv420p (anything else breaks phone hardware decoders)")
            got_wh = (int(video.get("width", 0)), int(video.get("height", 0)))
            check("geometry", "fail", got_wh == (out_w, out_h),
                  f"{got_wh[0]}x{got_wh[1]}", f"{out_w}x{out_h} (9:16)")
        if audio:
            check("audio-codec", "fail", audio.get("codec_name") == "aac",
                  str(audio.get("codec_name")), "aac")

        dur = float(info.get("format", {}).get("duration", 0.0))
        check("duration-bounds", "fail", MIN_CLIP_S <= dur <= MAX_CLIP_S,
              f"{dur:.2f}s", f"[{MIN_CLIP_S}, {MAX_CLIP_S}]s (spec §S2 30-60)")
        check("duration-matches-artifact", "fail",
              abs(dur - float(clip.duration_s)) <= 0.5,
              f"{dur:.2f}s", f"{clip.duration_s:.2f}s ± 0.5 claimed by S6")
        # Jump-cut splice integrity: measured duration must land within one
        # frame of what the TimeMap PREDICTED. The panel showed concat
        # silently padding ~33 ms per seam — 231 ms over 10 seams — while
        # every other check stayed green: stream durations equalize by
        # construction, so only prediction-vs-measurement can see it.
        expected = getattr(clip, "expected_duration_s", None)
        if expected is not None:
            check("splice-duration-integrity", "fail",
                  abs(dur - float(expected)) <= 0.05,
                  f"{dur:.3f}s measured",
                  f"{float(expected):.3f}s predicted by the splice "
                  "arithmetic; divergence means seams were padded")
        if video and audio:
            vd = float(video.get("duration") or dur)
            ad = float(audio.get("duration") or dur)
            check("av-duration-match", "fail", abs(vd - ad) <= 0.75,
                  f"video {vd:.2f}s vs audio {ad:.2f}s",
                  "streams within 0.75s (gross A/V desync guard)")

        # ---- 3. faststart: moov before mdat -------------------------------
        head = clip_path.read_bytes()[:2 * 1024 * 1024]
        moov, mdat = head.find(b"moov"), head.find(b"mdat")
        faststart = moov != -1 and (mdat == -1 or moov < mdat)
        check("faststart", "warn", faststart,
              f"moov@{moov} mdat@{mdat}",
              "moov atom first (instant playback start on socials)")

        # ---- 4. Loudness, REMEASURED --------------------------------------
        rc, _out, err = _run([ffmpeg, "-nostdin", "-hide_banner",
                              "-i", str(clip_path),
                              "-af", "loudnorm=print_format=json",
                              "-f", "null", "-"])
        m_start, m_end = err.rfind("{"), err.rfind("}")
        if m_start != -1 and m_end > m_start:
            stats = json.loads(err[m_start:m_end + 1])
            meas_i = float(stats["input_i"])
            meas_tp = float(stats["input_tp"])
            check("true-peak-ceiling", "fail", meas_tp <= SHIP_TP_CEILING,
                  f"{meas_tp:.2f} dBTP",
                  f"<= {SHIP_TP_CEILING} dBTP (inter-sample clipping guard)")
            delta = abs(meas_i - target_i)
            check("loudness-target", "fail", delta <= LOUDNESS_FAIL_LU,
                  f"{meas_i:.2f} LUFS",
                  f"{target_i} ± {LOUDNESS_FAIL_LU} LU hard band")
            check("loudness-tolerance", "warn", delta <= LOUDNESS_WARN_LU,
                  f"{meas_i:.2f} LUFS",
                  f"{target_i} ± {LOUDNESS_WARN_LU} LU "
                  "(beyond this = headroom-limited source, recorded)")
        else:
            check("loudness-measurable", "fail", False,
                  "loudnorm produced no stats", "measurable audio")

        # ---- 5. Dead content: black, frozen, silent -----------------------
        rc, _out, err = _run([
            ffmpeg, "-nostdin", "-hide_banner", "-i", str(clip_path),
            "-vf", f"blackdetect=d={MAX_BLACK_RUN_S}:pix_th=0.10,"
                   f"freezedetect=n=-60dB:d={MAX_FREEZE_RUN_S}",
            "-af", f"silencedetect=noise=-50dB:d={MAX_SILENCE_RUN_S}",
            "-f", "null", "-"])
        blacks = re.findall(r"black_duration:([\d.]+)", err)
        check("black-frames", "fail", not blacks,
              (f"black runs {blacks}s" if blacks else "none"),
              f"no black run >= {MAX_BLACK_RUN_S}s")
        freezes = re.findall(r"freeze_duration: ?([\d.]+)", err)
        check("frozen-frames", "warn", not freezes,
              (f"freeze runs {freezes}s" if freezes else "none"),
              f"no frozen run >= {MAX_FREEZE_RUN_S}s (static content warns)")
        silences = re.findall(r"silence_duration: ?([\d.]+)", err)
        check("silence", "fail", not silences,
              (f"silent runs {silences}s" if silences else "none"),
              f"no silence >= {MAX_SILENCE_RUN_S}s")

        # ---- 6. Subtitles actually accompanied this clip ------------------
        if subtitles is not None:
            ass = Path(subtitles.ass_path)
            ass_ok = ass.exists()
            check("subtitle-file", "fail", ass_ok, str(ass),
                  "the .ass burned into this render still exists for audit")
            if ass_ok:
                ass_digest = hashlib.sha256(
                    ass.read_bytes()).hexdigest()
                check("subtitle-integrity", "fail",
                      ass_digest == subtitles.ass_sha256,
                      ass_digest[:16], f"{subtitles.ass_sha256[:16]}")
            check("subtitle-coverage", "warn",
                  subtitles.word_count > 0 and subtitles.line_count > 0,
                  f"{subtitles.word_count} words / "
                  f"{subtitles.line_count} events",
                  "at least one karaoke event when the window has speech")

        # ---- 7. Camera path sanity ----------------------------------------
        if campath is not None:
            bad = [f for f in campath.frames
                   if f.x % 2 or f.y % 2 or f.w % 2 or f.h % 2
                   or f.x + f.w > campath.src_width
                   or f.y + f.h > campath.src_height]
            check("campath-bounds", "fail", not bad,
                  f"{len(bad)} frames out of bounds or odd" if bad else "all "
                  f"{len(campath.frames)} frames even and in-bounds",
                  "every crop even-coordinate and inside the source frame")
            check("framing-recorded", "warn",
                  campath.framing_mode in ("speaker", "center", "dual_pane"),
                  campath.framing_mode, "an honest framing_mode")

        return self._finish(cache_key, clip, checks)

    def _finish(self, cache_key: str, clip: Any,
                checks: list[QACheck]) -> QAArtifact:
        failed = [c for c in checks if c.severity == "fail" and not c.passed]
        warned = [c for c in checks if c.severity == "warn" and not c.passed]
        passed = not failed
        for c in failed:
            log.warning("s7.check_failed", check=c.name,
                        measured=c.measured, expected=c.expected)
        for c in warned:
            log.info("s7.check_warned", check=c.name,
                     measured=c.measured, expected=c.expected)
        log.info("s7.verdict", passed=passed, checks=len(checks),
                 failed=len(failed), warned=len(warned))
        return QAArtifact(
            cache_key=cache_key, stage=self.name,
            source_clip=clip.cache_key, clip_path=str(clip.clip_path),
            passed=passed, checks=checks,
            failed_count=len(failed), warned_count=len(warned))
