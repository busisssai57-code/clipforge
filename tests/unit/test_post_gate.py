"""The draft-only amendment, enforced where a post leaves the machine.

`[posting]` carried six pins of the 2026-07-27 amendment — publish_mode,
smart_scheduling, require_approval, enabled_platforms, the timezone and
the delays — and `bta post` read NONE of them. Two field validators
refused bad values in config.toml, which makes them a law about the
config file rather than about what the program does.
"""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from clipforge import cli
from clipforge.config import AppConfig

runner = CliRunner()


@pytest.fixture()
def posting(tmp_path, monkeypatch):
    """A workspace, a clip, and a captured execute_post_job."""
    from clipforge.paths import Workspace

    ws = Workspace(tmp_path / "ws").ensure()
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"\x00" * 32)
    cfg = AppConfig()
    monkeypatch.setattr(cli, "_boot", lambda *a, **kw: (cfg, ws))

    calls: list[dict] = []

    class Result:
        status = "draft_saved"
        message = "ok"
        error = None

    def fake_execute(job, **kwargs):
        calls.append({"job": job, **kwargs})
        return Result()

    monkeypatch.setattr("clipforge.poster.scheduler.execute_post_job",
                        fake_execute)
    return cfg, clip, calls


def test_yes_cannot_skip_per_clip_approval(posting):
    """The third pin: every clip is approved individually. `--yes` made
    the flag that grants approval a command-line convenience."""
    cfg, clip, calls = posting
    res = runner.invoke(cli.app, ["post", "--clip", str(clip), "--yes"])
    assert res.exit_code == 2
    assert "approved individually" in res.output
    assert not calls, "a post was dispatched without a human answering"


def test_a_platform_outside_the_list_is_refused(posting):
    cfg, clip, calls = posting
    cfg.posting.enabled_platforms = ["youtube"]
    res = runner.invoke(cli.app, ["post", "--clip", str(clip),
                                  "--platform", "tiktok"], input="y\n")
    assert res.exit_code == 2
    assert "enabled_platforms" in res.output
    assert not calls


def test_scheduled_publishing_stops_the_command(posting, monkeypatch):
    """The validator refuses this in config.toml; the runtime now refuses
    it too, so a hand-built AppConfig cannot route around the pin."""
    cfg, clip, calls = posting
    monkeypatch.setattr(type(cfg.posting), "model_config",
                        {**type(cfg.posting).model_config, "validate_assignment": False},
                        raising=False)
    object.__setattr__(cfg.posting, "smart_scheduling", True)
    res = runner.invoke(cli.app, ["post", "--clip", str(clip)], input="y\n")
    assert res.exit_code == 2
    assert "smart_scheduling" in res.output
    assert not calls


def test_an_approved_post_carries_the_configured_mode_and_timezone(posting):
    cfg, clip, calls = posting
    cfg.posting.target_timezone_offset_hours = 1.0
    res = runner.invoke(cli.app, ["post", "--clip", str(clip)], input="y\n")
    assert res.exit_code == 0, res.output
    assert len(calls) == 1
    assert calls[0]["job"].publish_mode == "draft"
    assert calls[0]["timezone_offset_hours"] == 1.0
    assert calls[0]["approved"] is True


def test_declining_the_prompt_dispatches_nothing(posting):
    cfg, clip, calls = posting
    res = runner.invoke(cli.app, ["post", "--clip", str(clip)], input="n\n")
    assert res.exit_code == 1
    assert not calls
