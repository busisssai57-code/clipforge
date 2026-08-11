"""Unit tests for S5 subtitle theme presets and auto-emoji injection."""

from __future__ import annotations

import pytest
from clipforge.stages.s5_subtitles import _add_auto_emoji, EMOJI_MAP, SUBTITLE_THEMES


def test_auto_emoji_injects_matching_emojis():
    assert "💰" in _add_auto_emoji("money")
    assert "🚀" in _add_auto_emoji("rocket")
    assert "🔥" in _add_auto_emoji("FIRE!")
    assert "💡" in _add_auto_emoji("smart")


def test_auto_emoji_ignores_unmapped_words():
    assert _add_auto_emoji("table") == "table"
    assert _add_auto_emoji("ordinary") == "ordinary"


def test_subtitle_themes_have_valid_styles():
    assert "viral_impact" in SUBTITLE_THEMES
    assert "neon_cyber" in SUBTITLE_THEMES
    assert "clean_minimal" in SUBTITLE_THEMES
    assert "podcast_gold" in SUBTITLE_THEMES
    assert "headline_box" in SUBTITLE_THEMES

    for name, cfg in SUBTITLE_THEMES.items():
        assert "font" in cfg
        assert "highlight" in cfg
        assert "base" in cfg
        assert "outline" in cfg
        assert "shadow" in cfg
