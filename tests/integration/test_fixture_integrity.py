"""The bundled smoke fixture must stay valid — every later checkpoint's
end-to-end smoke (§9) builds on it. Needs real ffprobe, no GPU."""

from pathlib import Path

import pytest

from clipforge.ffmpeg import find_binary, probe

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_90s.mp4"

pytestmark = pytest.mark.skipif(find_binary("ffprobe") is None,
                                reason="ffprobe not installed")


def test_fixture_exists() -> None:
    assert FIXTURE.exists(), "bundled smoke input missing (spec §4)"


def test_fixture_shape() -> None:
    info = probe(FIXTURE)
    assert 89.0 <= info.duration_s <= 91.0
    assert info.width == 1280 and info.height == 720
    assert info.v_codec == "h264" and info.a_codec == "aac"
    assert info.fps == pytest.approx(30.0, abs=0.1)
