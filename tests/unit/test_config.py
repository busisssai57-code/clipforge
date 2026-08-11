"""Config loading: fail-fast validation, defaults, watchlist ordering."""

from pathlib import Path

import pytest

from clipforge.config import AppConfig, load_config, load_watchlist
from clipforge.errors import ConfigError

REPO = Path(__file__).resolve().parents[2]


def test_example_config_is_valid():
    cfg = load_config(REPO / "config" / "config.example.toml")
    assert cfg.ingest.segment_time_s == 900
    assert cfg.s1.batch_size == 8
    assert cfg.s3.max_pixels > 0
    assert cfg.orchestration.gpu_concurrency == 1


def test_defaults_construct_without_any_toml():
    cfg = AppConfig()
    assert cfg.s2.top_k == 10
    assert cfg.s6.loudness_i == -14.0


def test_missing_file_raises_config_error(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "nope.toml")


def test_utf8_bom_config_loads(tmp_path):
    """Windows editors (Notepad, PowerShell Set-Content -Encoding UTF8) write
    a BOM; tomllib rejects it with a message that points nowhere useful.
    Operator-authored config must load regardless."""
    p = tmp_path / "bom.toml"
    p.write_bytes(b"\xef\xbb\xbf[ingest]\nsegment_time_s = 600\n")
    cfg = load_config(p)
    assert cfg.ingest.segment_time_s == 600


def test_non_utf8_config_gives_actionable_error(tmp_path):
    p = tmp_path / "latin1.toml"
    p.write_bytes(b"[workspace]\nroot = '\xff\xfe caf\xe9'\n")
    with pytest.raises(ConfigError, match="UTF-8"):
        load_config(p)


def test_watchlist_bom_loads(tmp_path):
    p = tmp_path / "channels.toml"
    p.write_bytes(b'\xef\xbb\xbf[[channels]]\nplatform="twitch"\nhandle="x"\n')
    assert len(load_watchlist(p).channels) == 1


def test_invalid_toml_raises_config_error(tmp_path):
    p = tmp_path / "bad.toml"
    p.write_text("this is not = [ toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid TOML"):
        load_config(p)


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "typo.toml"
    p.write_text("[ingest]\nsegment_tmie_s = 900\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_batch_size_hard_cap_enforced(tmp_path):
    p = tmp_path / "cap.toml"
    p.write_text("[s1]\nbatch_size = 16\n", encoding="utf-8")
    with pytest.raises(ConfigError):  # le=8 — the VRAM protection is unoverridable
        load_config(p)


def test_gpu_concurrency_cannot_exceed_one(tmp_path):
    p = tmp_path / "gpu.toml"
    p.write_text("[orchestration]\ngpu_concurrency = 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_window_bounds_validated(tmp_path):
    p = tmp_path / "win.toml"
    p.write_text("[s2]\nwindow_min_s = 60.0\nwindow_max_s = 30.0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="window_min_s"):
        load_config(p)


def test_watchlist_parses_and_sorts(tmp_path):
    p = tmp_path / "channels.toml"
    p.write_text(
        '[[channels]]\nplatform="twitch"\nhandle="b"\npriority=1\n'
        '[[channels]]\nplatform="twitch"\nhandle="a"\npriority=1\n'
        '[[channels]]\nplatform="youtube"\nhandle="c"\npriority=9\n'
        '[[channels]]\nplatform="kick"\nhandle="off"\nenabled=false\n',
        encoding="utf-8")
    wl = load_watchlist(p)
    order = [(c.platform, c.handle) for c in wl.enabled_sorted()]
    # priority desc, then platform/handle — fully deterministic:
    assert order == [("youtube", "c"), ("twitch", "a"), ("twitch", "b")]


def test_watchlist_bad_platform_rejected(tmp_path):
    p = tmp_path / "channels.toml"
    p.write_text('[[channels]]\nplatform="rumble"\nhandle="x"\n', encoding="utf-8")
    with pytest.raises(ConfigError):
        load_watchlist(p)


def test_repo_channels_toml_parses():
    wl = load_watchlist(REPO / "config" / "channels.toml")
    # The invariant is that no PLACEHOLDER ships enabled — recording a
    # channel is an authorization decision (spec §3.5) and a shipped
    # example must never make it for the operator. Channels the operator
    # has deliberately added are their call, so this no longer asserts the
    # live list is empty; it asserts nothing is enabled that nobody chose.
    placeholders = {"your_channel_here", "example", "changeme"}
    enabled = wl.enabled_sorted()
    leaked = [c.handle for c in enabled if c.handle.lower() in placeholders]
    assert not leaked, f"placeholder channels shipped enabled: {leaked}"
