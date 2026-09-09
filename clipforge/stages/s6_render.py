"""S6 — final 9:16 render: reframe, burn subtitles, normalize loudness.

Spec §S6: 1080x1920, NVENC with an automatic libx264 fallback, audio at
-14 LUFS integrated / -1.5 dBTP / 11 LRA.

Design notes, each a correction to the sketch this replaces:

  * It is a real :class:`Stage` with a cache key and an artifact. The sketch
    was a bare class that re-rendered on every run and left no audit record.
  * It USES the S4 camera path. The sketch did a static centre crop, which
    silently discarded every frame of tracking S4 had computed — the active
    speaker reframing, which is the product's whole point, was not in the
    output at all.
  * Clip length comes from the candidate window (spec §S2: 30-60 s). The
    sketch capped duration at 12 s under an invented "strict 10-12s maximum
    length rule" that appears nowhere in the spec and contradicts §S2.
  * Loudness normalization exists. The sketch had none, so §S6's -14 LUFS
    target was simply absent from the pipeline.
  * The ffmpeg binary is resolved through :mod:`clipforge.ffmpeg`, not the
    bare name, so the workspace's pinned build is used (this project ships a
    CLIPFORGE_FFMPEG_DIR override precisely because the ambient ffmpeg was
    the wrong one).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

from clipforge.errors import RetryableStageError
from clipforge.ffmpeg import probe, require_binary
from clipforge.log import get_logger
from clipforge.schemas.campath import CropFrame
from clipforge.schemas.render import ClipArtifact
from clipforge.stages.base import Stage

log = get_logger(__name__)

#: Loudness targets (spec §S6).
LOUDNESS_I = -14.0
#: Single-pass loudnorm does not land ON its TP target, it lands ABOVE it.
#: Measured on real source audio (45 s of speech, I=-14, LRA=11):
#:   TP=-1.5 -> -0.8 dBFS   (FAILS s7's -1.0 ceiling)
#:   TP=-2.0 -> -1.6 dBFS   (passes, 0.6 dB margin)
#:   TP=-2.5 -> -1.9 dBFS
#: At -1.5 every render landed within 0.1 dB of the ceiling, so whether a
#: clip shipped was a coin flip — 2 of 8 rendered clips were rejected for
#: true peak. -2.0 buys real margin for 0.2 LU of integrated loudness
#: (-14.5 -> -14.7 LUFS), well inside s7's -14.0 +/- 1.5 tolerance.
LOUDNESS_TP = -2.0
LOUDNESS_LRA = 11.0


def _even(v: int) -> int:
    """H.264 chroma subsampling needs even dimensions and offsets."""
    return int(v) - (int(v) % 2)


def _write_crop_commands(frames: list[Any], fps: Fraction,
                         dest: Path) -> tuple[int, int, int, int]:
    """Write the camera path as a ``sendcmd`` file; return the initial crop.

    The first design expressed the whole path as nested ``if(lt(n,...))``
    terms in the crop filter's x/y expressions. That was only ever exercised
    by a STATIC path (one run, no nesting); the first genuinely moving path —
    smoothed x changing almost every frame — produced hundreds of nesting
    levels and ffmpeg's expression parser rejected it, killing both encoders
    with zero frames out. ``sendcmd`` is the tool built for this: one line
    per change, linear in the path length, no nesting at all. ``crop``
    accepts runtime commands for exactly ``x``/``y``/``w``/``h``.

    Timestamps are CLIP-relative (``frame/fps``, no ``start_s`` offset),
    matching the ``setpts=PTS-STARTPTS`` rebase in the render. One origin for
    every stamp in the graph — these commands, the audio fades, the progress
    bar, the .ass — so the pieces cannot drift apart.
    """
    first = frames[0]
    x0, y0 = _even(first.x), _even(first.y)
    w, h = _even(first.w), _even(first.h)
    lines: list[str] = []
    last_x, last_y, last_w, last_h = x0, y0, w, h
    for f in frames[1:]:
        x, y = _even(f.x), _even(f.y)
        fw, fh = _even(f.w), _even(f.h)
        if x == last_x and y == last_y and fw == last_w and fh == last_h:
            continue
        t = float(int(f.frame) / fps)
        if x != last_x:
            lines.append(f"{t:.4f} crop@cam x {x};")
        if y != last_y:
            lines.append(f"{t:.4f} crop@cam y {y};")
        # w/h too — punch-ins are expressed as a SHRINKING crop, and sending
        # only x/y would compute the whole zoom and then throw it away.
        if fw != last_w:
            lines.append(f"{t:.4f} crop@cam w {fw};")
        if fh != last_h:
            lines.append(f"{t:.4f} crop@cam h {fh};")
        last_x, last_y, last_w, last_h = x, y, fw, fh
    dest.write_text("\n".join(lines) + ("\n" if lines else ""),
                    encoding="ascii")
    return w, h, x0, y0


#: Largest crop-width change one compressed frame may make across a splice
#: seam, as a fraction of source width (floored at 6 px for tiny sources).
#: A punch-in ramp cut mid-flight otherwise splices as an instant zoom pop —
#: the panel measured a 12% step in a single frame.
SEAM_MAX_W_STEP_FRAC = 0.015
SEAM_MIN_W_STEP_PX = 6

#: Subprocess ceilings. The render bound is generous — libx264 on a 60 s
#: 1080x1920 clip measures ~25 s here, so 30 min covers a pathological
#: source several times over — but it EXISTS, which is the point: a wedged
#: ffmpeg on a torn input otherwise hangs the unattended agent loop with no
#: output at all. Measurement decodes a single clip; two minutes is ample.
RENDER_TIMEOUT_S = 1800.0
MEASURE_TIMEOUT_S = 120.0


def _video_codec_args(vcodec: str, *, nvenc_preset: str, x264_preset: str,
                      cq: int) -> list[str]:
    """The ``-c:v ...`` half of the render command.

    A function, not an inline branch, so the preset actually reaching ffmpeg
    can be asserted without running a render. The x264 preset used to be the
    literal "medium" here while NVENC's was configurable — invisible on a
    machine with working NVENC, and the whole encoder on a machine without it.
    """
    if vcodec == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", nvenc_preset,
                "-rc", "vbr", "-cq", str(cq), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", x264_preset, "-crf", str(cq)]


def _remap_path_frames(path_frames: list[Any], tmap: Any, fps_f: float,
                       src_width: int, src_height: int) -> list[Any]:
    """Camera path -> the COMPRESSED timeline of a jump-cut render.

    Three things happen here and each one is load-bearing:

    1. frames inside a cut are dropped and survivors renumbered onto the
       compressed clock;
    2. two source frames can collide on one compressed index at a seam —
       the LATER one wins, because the first compressed frame after a cut
       shows the NEW scene and keep-first rendered it at the old camera
       position (a visible flick);
    3. crop width is rate-limited across the seam, so a punch-in that was
       mid-ramp when the cut landed eases in instead of popping.
    """
    remapped = []
    for f in path_frames:
        t = f.frame / fps_f
        if not tmap.is_kept(t):
            continue
        new_idx = int(round(tmap.to_compressed(t) * fps_f))
        remapped.append(CropFrame(frame=new_idx, x=f.x, y=f.y, w=f.w, h=f.h))

    by_idx: dict[int, CropFrame] = {}
    for f in sorted(remapped, key=lambda fr: fr.frame):
        by_idx[f.frame] = f               # later assignment wins per index
    ordered = [by_idx[k] for k in sorted(by_idx)]

    healed: list[CropFrame] = []
    max_step = max(SEAM_MIN_W_STEP_PX, int(SEAM_MAX_W_STEP_FRAC * src_width))
    for f in ordered:
        if healed:
            p = healed[-1]
            if abs(f.w - p.w) > max_step:
                dw = max(-max_step, min(max_step, f.w - p.w))
                w_new = _even(min(src_width, p.w + dw))
                h_new = _even(int(w_new * f.h / max(1, f.w)))
                # Widening toward the target can outgrow the source frame
                # when the path is already height-clamped; fall back to the
                # height bound and re-derive width from it, so the eased
                # crop stays inside the picture and keeps f's aspect.
                if h_new > src_height:
                    h_new = _even(src_height)
                    w_new = _even(min(src_width,
                                      int(h_new * f.w / max(1, f.h))))
                cx = f.x + f.w / 2.0
                cy = f.y + f.h / 2.0
                x_new = _even(int(max(0, min(src_width - w_new,
                                             cx - w_new / 2.0))))
                y_new = _even(int(max(0, min(src_height - h_new,
                                             cy - h_new / 2.0))))
                f = CropFrame(frame=f.frame, x=x_new, y=y_new,
                              w=w_new, h=h_new)
        healed.append(f)
    return healed


#: Speech-enhancement chains, applied BEFORE loudnorm so the normalizer
#: measures the cleaned signal rather than the noise floor.
#:
#: Default is "off", and that is a judgement about CONTENT, not caution:
#: this pipeline clips music and performance as readily as talking heads,
#: and spectral denoise + de-essing on a beat is destructive in a way no
#: amount of tuning fixes. Speech-dominant sources should turn it on.
#:
#: "gentle" is safe on mixed audio: an 80 Hz highpass (rumble/handling
#: noise, inaudible on a phone speaker), mild broadband denoise, and
#: speechnorm to even out mic-distance swings.
#: "strong" adds de-essing and compression — right for a lavalier or a
#: room mic, wrong for anything musical.
_SPEECH_CHAINS = {
    "off": "",
    "gentle": "highpass=f=80,afftdn=nr=10:nf=-28,speechnorm=e=6.25:r=0.0001",
    "strong": ("highpass=f=85,afftdn=nr=18:nf=-30,deesser=i=0.4,"
               "acompressor=threshold=-18dB:ratio=3:attack=15:release=180,"
               "speechnorm=e=12.5:r=0.0005"),
}


def _speech_chain(mode: str) -> str:
    """Filter prefix for the requested enhancement mode (empty when off).

    An unknown mode degrades to OFF rather than raising: a typo in config
    must not stop a render, and shipping unprocessed audio is the
    conservative failure.
    """
    chain = _SPEECH_CHAINS.get((mode or "off").strip().lower())
    if chain is None:
        log.warning("s6.unknown_enhance_mode", mode=mode,
                    known=sorted(_SPEECH_CHAINS), note="treating as off")
        return ""
    return f"{chain}," if chain else ""


def _measure_loudness(ffmpeg: str, path: Path) -> tuple[float, float]:
    """Measured integrated loudness and true peak of a rendered file.

    Measured, not assumed. A filter chain that silently no-ops (a typo'd
    filter name, an audio-less input) still produces a file, and reporting
    the TARGET as though it were the RESULT would make the artifact a
    statement of intent rather than of fact.
    """
    try:
        proc = subprocess.run(
            [ffmpeg, "-nostdin", "-hide_banner", "-i", str(path),
             "-af", "loudnorm=print_format=json", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=MEASURE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        log.error("s6.loudness_measure_timeout", path=str(path),
                  timeout_s=MEASURE_TIMEOUT_S)
        return float("nan"), float("nan")
    stderr = proc.stderr or ""
    start = stderr.rfind("{")
    end = stderr.rfind("}")
    if start == -1 or end == -1 or end < start:
        return float("nan"), float("nan")
    try:
        data = json.loads(stderr[start:end + 1])
        return float(data["input_i"]), float(data["input_tp"])
    except (ValueError, KeyError):
        return float("nan"), float("nan")


class S6Render(Stage[ClipArtifact]):
    name = "s6_render"
    version = "9"
    artifact_type = ClipArtifact

    def _execute(self, *, cache_key: str, params: dict[str, Any],
                 **inputs: Any) -> ClipArtifact:
        campath = inputs.get("campath_artifact")
        subtitles = inputs.get("subtitle_artifact")
        video_path = inputs.get("video_path")
        clips_dir = inputs.get("clips_dir")
        if campath is None or subtitles is None or video_path is None:
            raise RetryableStageError(
                "S6 requires campath_artifact, subtitle_artifact and "
                "video_path", stage=self.name)

        video_path = Path(video_path)
        if not video_path.exists():
            raise RetryableStageError(f"media vanished: {video_path}",
                                      stage=self.name)

        out_w = int(params.get("width", 1080))
        out_h = int(params.get("height", 1920))
        encoder = str(params.get("encoder", "h264_nvenc"))
        preset = str(params.get("nvenc_preset", "p5"))
        # The libx264 preset was hardcoded to "medium" while its NVENC
        # counterpart was configurable — and on a machine where NVENC is
        # unavailable (this one: the driver reports nvenc API 13.0, ffmpeg 8.x
        # requires 13.1) the x264 path is not a fallback, it is THE encoder.
        # Measured, 30s of 1080x1920 at crf 21: medium 17.9s/13.3 MB,
        # veryfast 9.9s/11.0 MB — 1.82x faster AND smaller, so that is the
        # default. ultrafast is 2.82x but writes 35 MB, which is why it is not.
        # crf is untouched, so the quality target is unchanged.
        x264_preset = str(params.get("x264_preset", "veryfast"))
        cq = int(params.get("cq", 21))
        audio_bitrate = str(params.get("audio_bitrate", "192k"))
        target_i = float(params.get("loudness_i", LOUDNESS_I))
        target_tp = float(params.get("loudness_tp", LOUDNESS_TP))
        target_lra = float(params.get("loudness_lra", LOUDNESS_LRA))

        start = float(campath.clip_start)
        duration = float(campath.clip_end) - start
        if duration <= 0:
            raise RetryableStageError(
                f"S6: non-positive clip duration {duration}", stage=self.name)

        try:
            fps = Fraction(campath.src_fps_rational)
        except (ValueError, ZeroDivisionError):
            fps = Fraction(30, 1)

        # Trust nothing about the upstream geometry: verify it against the
        # actual file. S4 defaulted to 1920x1080 when it could not probe, and
        # on a 1280x720 source that produced a crop taller than the frame —
        # ffmpeg then emitted ZERO frames and failed at the encoder, which
        # reads like an encoder problem and is not one.
        info = probe(video_path)
        if (int(info.width) != int(campath.src_width)
                or int(info.height) != int(campath.src_height)):
            raise RetryableStageError(
                f"S6: camera path declares source {campath.src_width}x"
                f"{campath.src_height} but {video_path.name} is "
                f"{info.width}x{info.height}. The crop would fall outside "
                "the frame; re-run S4 against this media.",
                stage=self.name)
        for f in campath.frames[:1] or []:
            if f.x + f.w > info.width or f.y + f.h > info.height:
                raise RetryableStageError(
                    f"S6: crop {f.w}x{f.h}+{f.x}+{f.y} exceeds the "
                    f"{info.width}x{info.height} frame", stage=self.name)

        clips_dir = Path(clips_dir) if clips_dir else (
            self.artifacts_dir.parent / "clips")
        clips_dir.mkdir(parents=True, exist_ok=True)
        out_path = clips_dir / f"{cache_key}.mp4"
        partial = out_path.with_suffix(".mp4.partial")

        ass = str(Path(subtitles.ass_path).resolve())
        # libass path escaping inside a filtergraph: backslashes become
        # forward slashes, then the drive colon is escaped, then the whole
        # value is single-quoted. On Windows an unescaped "C:" is parsed as
        # an option separator and the filter silently loses the file.
        ass_arg = ass.replace("\\", "/").replace(":", "\\:")

        # Jump-cut pacing (blueprint §9.2): keep-intervals computed by the
        # CLI from word timestamps. All timing below switches to the
        # COMPRESSED clock when they are present — the campath frames are
        # filtered to kept frames and renumbered, the fades and the
        # progress bar use the compressed duration, and the input is
        # spliced with trim+concat before the visual chain runs.
        keeps = [(float(a), float(b))
                 for a, b in params.get("keep_intervals", [])]
        tmap = None
        render_duration = duration
        path_frames = list(campath.frames)
        if keeps:
            from clipforge.pacing import TimeMap  # noqa: PLC0415

            tmap = TimeMap(keeps)
            render_duration = tmap.duration()
            path_frames = _remap_path_frames(
                path_frames, tmap, float(fps),
                campath.src_width, campath.src_height)
            log.info("s6.jumpcuts", segments=len(keeps),
                     cut_s=round(duration - render_duration, 2),
                     compressed_s=round(render_duration, 2))

        if path_frames:
            cmd_file = self.artifacts_dir / self.name / f"{cache_key}.cmds"
            cmd_file.parent.mkdir(parents=True, exist_ok=True)
            w, h, x0, y0 = _write_crop_commands(path_frames, fps, cmd_file)
            if cmd_file.stat().st_size > 0:
                cmd_arg = str(cmd_file.resolve()).replace("\\", "/") \
                                                 .replace(":", "\\:")
                crop_part = (f"sendcmd=f='{cmd_arg}',"
                             f"crop@cam={w}:{h}:{x0}:{y0}")
            else:
                # A perfectly STATIC path (the centre fallback) produces zero
                # command lines, and ffmpeg rejects an empty sendcmd file
                # outright: "No commands were specified". The mirror image of
                # the bug sendcmd replaced — the nested-if expression worked
                # only for static paths, this worked only for moving ones.
                # A static path needs no command channel at all.
                crop_part = f"crop={w}:{h}:{x0}:{y0}"
        else:
            # Documented fallback: static centre crop at the target aspect.
            w = _even(min(campath.src_width,
                          int(campath.src_height * 9 / 16)))
            h = _even(min(campath.src_height, int(w * 16 / 9)))
            crop_part = (f"crop={w}:{h}:"
                         f"{_even((campath.src_width - w) // 2)}:"
                         f"{_even((campath.src_height - h) // 2)}")
        # Progress bar: dark track + white fill sweeping left-to-right over
        # the clip window. drawbox re-evaluates w each frame because the
        # expression contains t; t is MEDIA time (output-side -ss), so the
        # sweep is anchored to the window exactly like the audio fades.
        bar_h = max(6, out_h // 160)
        progress = (
            f"drawbox=x=0:y=ih-{bar_h}:w=iw:h={bar_h}:"
            f"color=black@0.35:t=fill,"
            f"drawbox=x=0:y=ih-{bar_h}:"
            f"w='iw*min(1\\,max(0\\,t/{render_duration:.3f}))':"
            f"h={bar_h}:color=white@0.9:t=fill")
        # setpts FIRST: rebase to clip-relative before any filter that reads
        # timestamps (sendcmd, drawbox's t, ass). See the TIME BASE note in
        # _cmd — this is what makes the .ass line up with the picture.
        # Niche colour grade, BEFORE the progress bar and the burned-in
        # captions: the grade is meant to treat the FOOTAGE, and running it
        # over the overlays would desaturate white captions and the bar
        # along with the picture. It lives here rather than as a second
        # re-encode afterwards for two reasons — one encode instead of two,
        # and S7 then measures the pixels that actually ship (a grade
        # applied after the gate sails past every luma check).
        grade = str(params.get("grade", "") or "").strip().rstrip(",")
        grade_part = f"{grade}," if grade else ""
        vf = (f"setpts=PTS-STARTPTS,"
              f"{crop_part},"
              f"scale={out_w}:{out_h}:flags=lanczos,"
              f"setsar=1,"
              f"{grade_part}"
              f"{progress},"
              f"ass='{ass_arg}'")
        # require_binary: find_binary returns None on a machine without
        # ffmpeg, which reached subprocess as a bare TypeError instead of
        # the actionable install message three lines away.
        ffmpeg = str(require_binary("ffmpeg"))

        # SINGLE-pass dynamic loudnorm, on measured evidence — every fancier
        # configuration was tried against real renders and lost:
        #   two-pass linear   -> TP clean, but lands ~2.6 LU short whenever
        #                        the source lacks headroom (one fixed gain
        #                        cannot pass the peak ceiling);
        #   two-pass dynamic  -> VIOLATES the ceiling: feeding measured_*
        #                        stats with linear=false measured +1.77 dBTP
        #                        out against a -1.5 target, reproduced
        #                        standalone with the pipeline's exact values
        #                        (its limiter does not hold in that mode);
        #   single-pass       -> TP -1.1, same integrated loudness as the
        #                        two-pass attempts on this material.
        # A clip that is 2 LU quiet is a recorded deviation; a clip that
        # inter-sample-clips is broken. The artifact stores target AND
        # measured, and s6.loudness_off_target says when they diverge.
        # Edge fades: cutting mid-waveform leaves a step discontinuity whose
        # INTER-SAMPLE peak spikes far above anything in the audio — measured
        # +1.76 dBTP on a clip whose interior sits at -1.10, and the whole
        # excess vanished when the first 200 ms were excluded from the
        # measurement. 20 ms in / 60 ms out are inaudible and remove the step.
        #
        # Stamps are CLIP-relative, because `asetpts=PTS-STARTPTS` below
        # rebases the filter timeline to the clip. See the trim note in _cmd.
        fade_in, fade_out = 0.02, 0.06
        enhance = _speech_chain(str(params.get("enhance_speech", "off")))
        af_tail = (f"{enhance}"
                   f"loudnorm=I={target_i}:TP={target_tp}:LRA={target_lra}"
                   f":linear=false,"
                   f"afade=t=in:st=0:d={fade_in},"
                   f"afade=t=out:st={max(0.0, render_duration - fade_out):.3f}"
                   f":d={fade_out}")
        af = f"asetpts=PTS-STARTPTS,{af_tail}"

        # With jump-cuts, the input is spliced into kept segments FIRST
        # (trim+concat rebases each segment, so the visual/audio tails see
        # one continuous compressed timeline), then the normal chains run.
        # Every splice lands in a silence trough by construction — the cuts
        # are BETWEEN padded word boundaries — so no afade per seam is
        # needed; the loudnorm+edge fades handle the clip ends as before.
        filter_complex = None
        if tmap is not None and len(keeps) >= 1:
            # Video trims by FRAME NUMBER (exact); audio by seconds at
            # microsecond precision from the SAME frame boundaries. The
            # panel measured decimal-second trims silence-padding ~33 ms at
            # every seam where the rounded end swallowed an extra frame.
            from fractions import Fraction as _Fr  # noqa: PLC0415
            fps_frac = _Fr(campath.src_fps_rational)
            segs = []
            for i, (a, b) in enumerate(keeps):
                fa = int(round(a * fps_frac))
                fb = int(round(b * fps_frac))
                aa = float(fa / fps_frac)
                bb = float(fb / fps_frac)
                segs.append(f"[0:v]trim=start_frame={fa}:end_frame={fb},"
                            f"setpts=PTS-STARTPTS[v{i}]")
                segs.append(f"[0:a]atrim=start={aa:.6f}:end={bb:.6f},"
                            f"asetpts=PTS-STARTPTS[a{i}]")
            pairs = "".join(f"[v{i}][a{i}]" for i in range(len(keeps)))
            segs.append(f"{pairs}concat=n={len(keeps)}:v=1:a=1[vc][ac]")
            # vf minus its leading setpts (each segment is already rebased).
            vf_tail = vf.split(",", 1)[1] if vf.startswith("setpts=") else vf
            segs.append(f"[vc]{vf_tail}[vout]")
            segs.append(f"[ac]{af_tail}[aout]")
            filter_complex = ";".join(segs)

        def _cmd(vcodec: str) -> list[str]:
            # INPUT-side seek, for SPEED — and one time base for everything.
            #
            # Honest history: this was changed on a hypothesis that
            # output-side `-ss` broke burned-in subtitles for windows far
            # from zero. A differential render (same window with and without
            # the ass filter, peak |diff| 245/255) DISPROVED that — captions
            # burned in correctly either way. The hypothesis was wrong and
            # the comment that claimed it has been removed.
            #
            # The change is kept because the real benefit is measurable:
            # output-side `-ss` decodes from frame 0 and discards, so a clip
            # 22 minutes into a video pays for 22 minutes of decoding
            # (observed: 280 s for a window at 1350 s versus 70-130 s for
            # windows near the start). Input-side seek skips that.
            # `setpts=PTS-STARTPTS` then guarantees ONE origin for every
            # stamp in the graph — sendcmd, fades, progress bar, .ass — which
            # removes a whole class of time-base mistakes rather than any
            # specific observed one.
            base = [ffmpeg, "-nostdin", "-hide_banner", "-y",
                    "-ss", f"{start:.3f}", "-i", str(video_path),
                    "-t", f"{duration:.3f}"]
            if filter_complex is not None:
                base += ["-filter_complex", filter_complex,
                         "-map", "[vout]", "-map", "[aout]"]
            else:
                base += ["-vf", vf, "-af", af]
            base += _video_codec_args(vcodec, nvenc_preset=preset,
                                      x264_preset=x264_preset, cq=cq)
            # -f mp4 explicitly: the output is written to a `.mp4.partial`
            # name (Resumability Law — a killed render must not leave a
            # truncated .mp4 that looks finished), and ffmpeg infers the
            # muxer from the extension, which `.partial` is not.
            # -ar 48000: with no rate asked for, ffmpeg carries the source's
            # through, and a real render shipped 96 kHz AAC because the VOD
            # had it. Every short-form target expects 48 kHz; anything else
            # gets resampled by the platform, or refused.
            base += ["-pix_fmt", "yuv420p",
                     "-c:a", "aac", "-b:a", audio_bitrate, "-ar", "48000",
                     "-movflags", "+faststart", "-f", "mp4", str(partial)]
            return base

        def _render(vcodec: str) -> subprocess.CompletedProcess[str]:
            try:
                return subprocess.run(
                    _cmd(vcodec), capture_output=True, text=True,
                    encoding="utf-8", errors="replace",
                    timeout=RENDER_TIMEOUT_S)
            except subprocess.TimeoutExpired as exc:
                partial.unlink(missing_ok=True)
                tail = exc.stderr if isinstance(exc.stderr, str) else ""
                raise RetryableStageError(
                    f"S6 render wedged: ffmpeg exceeded "
                    f"{RENDER_TIMEOUT_S:.0f}s ({vcodec}). "
                    f"stderr tail: {(tail or '')[-300:]}",
                    stage=self.name) from exc

        used = encoder
        proc = _render(encoder)
        first_err = proc.stderr or ""
        if proc.returncode != 0 and encoder == "h264_nvenc":
            # Report what actually happened, not what we hope happened. This
            # said "nvenc unavailable" for a failure that had nothing to do
            # with the encoder (an unmuxable output name), which is how a
            # trivially fixable error gets read as a missing GPU feature.
            log.warning("s6.primary_encoder_failed", encoder=encoder,
                        note="retrying with libx264; if that also fails the "
                             "cause is not the encoder",
                        stderr=first_err[-400:])
            used = "libx264"
            proc = _render("libx264")
        if proc.returncode != 0:
            partial.unlink(missing_ok=True)
            detail = (proc.stderr or "")[-600:]
            if used != encoder:
                detail = (f"[{encoder}] {first_err[-300:]}\n"
                          f"[libx264] {detail}")
            raise RetryableStageError(f"S6 render failed: {detail}",
                                      stage=self.name)

        # .partial until complete, per the Resumability Law: a killed render
        # must never leave a truncated .mp4 that a later run treats as done.
        partial.replace(out_path)

        measured_i, measured_tp = _measure_loudness(ffmpeg, out_path)
        digest = hashlib.sha256(out_path.read_bytes()).hexdigest()
        # Duration MEASURED from the file, prediction recorded beside it.
        # A freshly rendered clip that CANNOT be probed is not a bookkeeping
        # hiccup to paper over — S7's duration-matches-artifact check exists
        # to catch fabricated numbers, and silently substituting the
        # prediction here would feed that gate a fabricated number. If the
        # file is unreadable one second after ffmpeg wrote it, the render
        # failed, whatever the exit code said.
        try:
            probed_duration = float(probe(out_path).duration_s)
        except Exception as exc:  # noqa: BLE001
            raise RetryableStageError(
                f"S6 wrote {out_path.name} but cannot probe it back "
                f"({type(exc).__name__}: {exc}); the render is not trusted",
                stage=self.name) from exc
        # PROBED, not `duration`. The artifact three lines down already
        # records the measured value — but this log line reported the
        # pre-splice window length, so a spliced clip logged 34.688s for a
        # file that is 31.1s. The comment above this block exists to forbid
        # exactly that substitution, and the logging was doing it: reading
        # the log, a working trim looked like a trim that had done nothing
        # (measured while adding the editor's trim, 2026-08-12).
        log.info("s6.render_complete", path=str(out_path), encoder=used,
                 duration_s=round(probed_duration, 3),
                 planned_duration_s=round(render_duration, 3),
                 loudness_i=measured_i, loudness_tp=measured_tp)
        # The artifact records what was MEASURED; this says out loud when
        # that misses what was ASKED FOR. A silent 3 LU miss is the kind of
        # thing that ships to a platform and gets normalized again on the
        # way in, which is where clips end up quiet.
        if measured_i == measured_i and abs(measured_i - target_i) > 1.0:
            log.warning("s6.loudness_off_target",
                        target_i=target_i, measured_i=measured_i,
                        measured_tp=measured_tp, target_tp=target_tp,
                        note="source headroom may make the target "
                             "unreachable under the true-peak ceiling")

        return ClipArtifact(
            cache_key=cache_key, stage=self.name,
            source_subtitles=subtitles.cache_key,
            clip_path=str(out_path.resolve()), clip_sha256=digest,
            duration_s=probed_duration,
            expected_duration_s=(render_duration if tmap is not None
                                 else None),
            width=out_w, height=out_h,
            encoder=used, loudness_i=measured_i, loudness_tp=measured_tp,
            target_loudness_i=target_i, target_loudness_tp=target_tp)
