"""`bta brand`: the post layer, finally wired to something.

clipforge/socialpost.py was written for the generation half, tested, and
then left with no caller when that half was deleted — the fifth time this
project has shipped a finished module nothing calls. These pin the entry
point, because "the function works" was already true and was not the
problem.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from clipforge import cli
from clipforge.socialpost import PostSpec, build_overlay

runner = CliRunner()


@pytest.fixture()
def clip(tmp_path, monkeypatch):
    """A workspace with one clip and its export pack."""
    from clipforge.paths import Workspace

    ws = Workspace(tmp_path / "ws").ensure()
    path = ws.clips / "abc.mp4"
    path.write_bytes(b"\x00" * 64)
    path.with_suffix(".export.json").write_text(
        json.dumps({"title": "A title", "hook": "wait for it"}),
        encoding="utf-8")

    from clipforge.config import AppConfig

    monkeypatch.setattr(cli, "_boot", lambda *a, **kw: (AppConfig(), ws))
    return path


# -------------------------------------------------------------- placement

def test_the_hook_card_can_be_moved_off_the_footages_own_text():
    """Measured on a shipped clip: at the 0.11 default the card landed on
    the source's own burned-in caption and both became unreadable. The
    default was chosen against generated shots, which carry nothing."""
    work = Path(__file__).parent
    high = build_overlay(PostSpec(hook="HI", hook_y=0.11), width=1080,
                         height=1920, work_dir=work / "_brandwork")[1]
    low = build_overlay(PostSpec(hook="HI", hook_y=0.30), width=1080,
                        height=1920, work_dir=work / "_brandwork")[1]
    assert "y=H*0.110" in high and "y=H*0.300" in low


def test_the_default_placement_is_unchanged():
    assert PostSpec().hook_y == 0.11


# ---------------------------------------------------------------- command

def test_branding_writes_a_new_file_and_leaves_the_clip_alone(clip, monkeypatch):
    """QA measured the original. A burned-in overlay is a taste decision,
    and the operator undoes it by deleting one file."""
    seen = {}

    def fake_apply(src, dest, spec, **kw):
        seen["src"], seen["spec"] = src, spec
        Path(dest).write_bytes(b"branded")
        return dest

    monkeypatch.setattr("clipforge.socialpost.apply_post", fake_apply)
    before = clip.read_bytes()
    res = runner.invoke(cli.app, ["brand", "abc.mp4", "--handle", "@me"])
    assert res.exit_code == 0, res.output
    assert clip.with_suffix(".branded.mp4").read_bytes() == b"branded"
    assert clip.read_bytes() == before, "the original clip was modified"
    assert seen["spec"].watermark == "@me"


def test_the_hook_comes_from_the_export_pack_when_not_given(clip, monkeypatch):
    seen = {}
    monkeypatch.setattr("clipforge.socialpost.apply_post",
                        lambda src, dest, spec, **kw: seen.update(spec=spec)
                        or Path(dest).write_bytes(b"x") or dest)
    res = runner.invoke(cli.app, ["brand", "abc.mp4"])
    assert res.exit_code == 0, res.output
    assert seen["spec"].hook == "wait for it", (
        "the editor already wrote a hook for this clip")


def test_an_explicit_hook_wins(clip, monkeypatch):
    seen = {}
    monkeypatch.setattr("clipforge.socialpost.apply_post",
                        lambda src, dest, spec, **kw: seen.update(spec=spec)
                        or Path(dest).write_bytes(b"x") or dest)
    runner.invoke(cli.app, ["brand", "abc.mp4", "--hook", "mine",
                            "--hook-y", "0.3"])
    assert seen["spec"].hook == "mine" and seen["spec"].hook_y == 0.3


def test_nothing_to_stamp_is_refused_before_ffmpeg_runs(clip, monkeypatch):
    clip.with_suffix(".export.json").unlink()
    called = []
    monkeypatch.setattr("clipforge.socialpost.apply_post",
                        lambda *a, **kw: called.append(1))
    res = runner.invoke(cli.app, ["brand", "abc.mp4"])
    assert res.exit_code == 2
    assert not called, "ffmpeg ran with nothing to draw"


def test_a_failure_is_reported_not_swallowed(clip, monkeypatch):
    from clipforge.socialpost import PostError

    def boom(*a, **kw):
        raise PostError("no usable fonts")

    monkeypatch.setattr("clipforge.socialpost.apply_post", boom)
    res = runner.invoke(cli.app, ["brand", "abc.mp4", "--handle", "@me"])
    assert res.exit_code == 1
    assert "no usable fonts" in res.output


def test_send_delivers_the_branded_cut_not_the_original(clip, monkeypatch):
    monkeypatch.setattr("clipforge.socialpost.apply_post",
                        lambda src, dest, spec, **kw: Path(dest).write_bytes(b"x")
                        or dest)
    sent = {}

    def fake_send(path, **kw):
        sent["path"] = Path(path)
        return "sent"

    monkeypatch.setattr("clipforge.notify.send_clip", fake_send)
    monkeypatch.setattr("clipforge.notify.target_from_config", lambda cfg: None)
    res = runner.invoke(cli.app, ["brand", "abc.mp4", "--handle", "@me",
                                  "--send"])
    assert res.exit_code == 0, res.output
    assert sent["path"].name == "abc.branded.mp4"
