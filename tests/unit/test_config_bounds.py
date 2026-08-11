"""Config Field CONSTRAINTS, not just defaults.

CP2 round-1 finding CFG-BOUNDS: `verify/ai.py::_spec_constants` pins default
VALUES but never the Field bounds that stop an operator overriding them. So
`frames_per_candidate: Field(8, ge=6, le=8)` -> `Field(8, ge=1, le=64)` and
`nms_iou: Field(0.4, gt=0, lt=1)` -> `Field(0.4)` both left the entire gate
green while `assert cfg.s3.frames_per_candidate in (6, 7, 8)` stayed true.

Also CFG-LIVE: the config file the operator actually runs against was
validated by neither gate half — only config.example.toml was.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from clipforge.config import S2Config, S3Config, load_config

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("value", [5, 9, 12, 0, 64])
def test_frames_per_candidate_outside_6_to_8_is_rejected(value):
    """Spec §S3 says strictly 6-8 frames per candidate."""
    with pytest.raises(ValidationError):
        S3Config(frames_per_candidate=value)


@pytest.mark.parametrize("value", [6, 7, 8])
def test_frames_per_candidate_inside_the_spec_range_is_accepted(value):
    assert S3Config(frames_per_candidate=value).frames_per_candidate == value


@pytest.mark.parametrize("value", [0.0, 1.0, 1.5, -0.1])
def test_nms_iou_outside_the_open_unit_interval_is_rejected(value):
    """An IoU of 0 suppresses everything that touches; 1.0 suppresses
    nothing. Neither is a threshold."""
    with pytest.raises(ValidationError):
        S2Config(nms_iou=value)


def test_nms_iou_inside_the_interval_is_accepted():
    assert S2Config(nms_iou=0.4).nms_iou == 0.4


def test_the_live_operator_config_validates():
    """config/config.toml is what `clipforge watch` and `clipforge process`
    actually load; only config.example.toml was ever checked, so drift in the
    live file was invisible to the gate."""
    live = REPO / "config" / "config.toml"
    if not live.exists():
        pytest.skip("no live config in this checkout")
    cfg = load_config(live)
    # Spot-check the constants the spec fixes, on the file that ships.
    assert cfg.s1.batch_size <= 8
    assert 0.0 < cfg.s2.nms_iou < 1.0
    assert cfg.s3.frames_per_candidate in (6, 7, 8)
    assert cfg.orchestration.gpu_concurrency == 1
    assert cfg.posting.publish_mode == "draft"
