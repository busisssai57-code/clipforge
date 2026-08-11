"""Pure parsers in clipforge.ffmpeg — no binaries required."""

import pytest

from clipforge.errors import FfmpegError
from clipforge.ffmpeg import (LoudnormMeasurement, loudnorm_filter,
                              parse_loudnorm_json, parse_media_info)

# Captured shape of real `ffmpeg -af loudnorm=...:print_format=json` stderr.
LOUDNORM_STDERR = """
[Parsed_loudnorm_0 @ 000001f2b8c04c80]
{
\t"input_i" : "-23.62",
\t"input_tp" : "-6.47",
\t"input_lra" : "10.10",
\t"input_thresh" : "-34.19",
\t"output_i" : "-14.02",
\t"output_tp" : "-1.50",
\t"output_lra" : "8.70",
\t"output_thresh" : "-24.42",
\t"normalization_type" : "dynamic",
\t"target_offset" : "0.02"
}
"""


def test_parse_loudnorm_extracts_measurements():
    m = parse_loudnorm_json(LOUDNORM_STDERR)
    assert m.input_i == pytest.approx(-23.62)
    assert m.input_tp == pytest.approx(-6.47)
    assert m.input_lra == pytest.approx(10.10)
    assert m.input_thresh == pytest.approx(-34.19)
    assert m.target_offset == pytest.approx(0.02)


def test_parse_loudnorm_ignores_other_braces():
    noisy = "config={'a':1}\nsome {not json} text\n" + LOUDNORM_STDERR
    assert parse_loudnorm_json(noisy).input_i == pytest.approx(-23.62)


def test_parse_loudnorm_missing_raises_typed():
    with pytest.raises(FfmpegError, match="loudnorm"):
        parse_loudnorm_json("frame= 100 fps= 30 ...")


def test_loudnorm_filter_uses_measured_linear():
    m = LoudnormMeasurement(input_i=-23.62, input_tp=-6.47, input_lra=10.1,
                            input_thresh=-34.19, target_offset=0.02)
    f = loudnorm_filter(m, i=-14.0, tp=-1.5, lra=11.0)
    # T8: pass-2 must carry every measured value and force linear mode.
    assert "measured_I=-23.62" in f
    assert "measured_TP=-6.47" in f
    assert "measured_LRA=10.1" in f
    assert "measured_thresh=-34.19" in f
    assert "offset=0.02" in f
    assert f.endswith("linear=true")
    assert f.startswith("loudnorm=I=-14.0:TP=-1.5:LRA=11.0")


PROBE = {
    "format": {"duration": "901.234"},
    "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1920,
         "height": 1080, "avg_frame_rate": "60000/1001", "r_frame_rate": "60/1"},
        {"codec_type": "audio", "codec_name": "aac"},
    ],
}


def test_parse_media_info_prefers_avg_frame_rate():
    info = parse_media_info(PROBE)
    assert info.duration_s == pytest.approx(901.234)
    assert info.width == 1920 and info.height == 1080
    assert info.fps_rational == "60000/1001"  # T10: the EXACT rational survives
    assert info.fps == pytest.approx(59.94, abs=0.01)
    assert info.v_codec == "h264" and info.a_codec == "aac"


def test_parse_media_info_handles_missing_streams():
    info = parse_media_info({"format": {"duration": "5"}, "streams": []})
    assert info.duration_s == 5.0
    assert info.fps is None and info.width is None


def test_parse_media_info_zero_denominator():
    probe = {"format": {}, "streams": [{"codec_type": "video",
                                        "avg_frame_rate": "0/0"}]}
    assert parse_media_info(probe).fps is None
