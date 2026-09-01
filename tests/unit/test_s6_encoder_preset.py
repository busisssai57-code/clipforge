"""The encoder preset that actually reaches ffmpeg, pinned.

The x264 preset was hardcoded to "medium" while its NVENC counterpart was a
config field. That is invisible on a machine with working NVENC and is the
ENTIRE encoder on a machine without it — which is the case here (the driver
reports nvenc API 13.0, ffmpeg 8.x requires 13.1), so every render on this
box took the hardcoded path.

Measured on a 30s 1080x1920 encode at crf 21: medium 17.9s / 13.3 MB,
veryfast 9.9s / 11.0 MB — 1.82x faster and smaller, hence the default.
"""

from __future__ import annotations

from clipforge.config import S6Config
from clipforge.stages.s6_render import _video_codec_args


def test_x264_preset_is_used_not_hardcoded_medium():
    args = _video_codec_args("libx264", nvenc_preset="p5",
                             x264_preset="veryfast", cq=21)
    assert args == ["-c:v", "libx264", "-preset", "veryfast", "-crf", "21"]
    # The bug this replaces: the caller's preset ignored in favour of a literal.
    assert "medium" not in args


def test_x264_preset_is_honoured_for_any_value():
    # Not just the new default — an operator who sets it gets what they set.
    for preset in ("medium", "fast", "ultrafast"):
        args = _video_codec_args("libx264", nvenc_preset="p5",
                                 x264_preset=preset, cq=23)
        assert args[args.index("-preset") + 1] == preset
        assert args[args.index("-crf") + 1] == "23"


def test_nvenc_path_unchanged():
    # The GPU path must keep its own preset and rate control; this change is
    # scoped to the CPU encoder.
    args = _video_codec_args("h264_nvenc", nvenc_preset="p5",
                             x264_preset="veryfast", cq=21)
    assert args == ["-c:v", "h264_nvenc", "-preset", "p5",
                    "-rc", "vbr", "-cq", "21", "-b:v", "0"]
    assert "libx264" not in args


def test_config_exposes_x264_preset_with_fast_default():
    cfg = S6Config()
    assert cfg.x264_preset == "veryfast"
    # Strict model: the field must really exist, or config.toml would reject it.
    assert S6Config(x264_preset="medium").x264_preset == "medium"
