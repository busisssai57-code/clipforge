"""Unit tests for S4 dual-speaker podcast split screen crop calculations."""

from __future__ import annotations

from clipforge.schemas.campath import CropFrame
from clipforge.stages.s4_tracking import compute_split_screen_crops


def test_compute_split_screen_crops_bounds():
    box_a = (100.0, 200.0, 300.0, 600.0)  # Left speaker
    box_b = (1200.0, 200.0, 1400.0, 600.0) # Right speaker
    src_w, src_h = 1920, 1080

    top_crop, bot_crop = compute_split_screen_crops(box_a, box_b, src_w, src_h, frame_idx=10)

    assert isinstance(top_crop, CropFrame)
    assert isinstance(bot_crop, CropFrame)

    assert top_crop.frame == 10
    assert bot_crop.frame == 10

    # Ensure crops stay strictly inside source video dimensions
    assert 0 <= top_crop.x <= src_w - top_crop.w
    assert 0 <= top_crop.y <= src_h - top_crop.h
    assert 0 <= bot_crop.x <= src_w - bot_crop.w
    assert 0 <= bot_crop.y <= src_h - bot_crop.h

    # Ensure width and height are even
    assert top_crop.w % 2 == 0
    assert top_crop.h % 2 == 0
    assert bot_crop.w % 2 == 0
    assert bot_crop.h % 2 == 0
