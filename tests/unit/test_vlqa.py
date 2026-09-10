"""The VL judge chain.

The behaviour that matters is not "does Claude answer" — it is what happens
when it does not. A chain that quietly falls through to its weakest link and
reports a verdict either way is how a hosted judge appears to work for a
month after its key expired, so every test here is about the degraded path:
no key, a dead endpoint, unparseable output, no decodable frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from clipforge import vlqa


def cfg(*, vl_qa=True, use_cloud=False, frames=4):
    return NS(s7=NS(vl_qa=vl_qa, use_cloud=use_cloud, vl_frames=frames),
              s3=NS(local_model_id="x", vram_budget_gb=8.0))


# --- sampling ------------------------------------------------------------

def test_the_sample_points_are_fixed_so_a_verdict_can_be_replayed():
    assert vlqa.taps(4) == vlqa.taps(4)
    assert vlqa.taps(3) == (0.25, 0.5, 0.75)


def test_sampling_never_lands_on_the_first_or_last_frame():
    """Both are usually a fade, and a fade tells the judge nothing."""
    for n in (1, 2, 6, 16):
        t = vlqa.taps(n)
        assert min(t) > 0.0 and max(t) < 1.0


# --- parsing -------------------------------------------------------------

def test_a_fenced_json_reply_is_still_read():
    v = vlqa._parse('```json\n{"ok": false, "findings": ["a"]}\n```', "x", 3)
    assert v.ok is False and v.findings == ["a"] and v.frames_seen == 3


def test_prose_with_no_json_is_not_treated_as_a_failure():
    """A judge that rambles must not quarantine a clip."""
    v = vlqa._parse("Looks fine to me!", "x", 2)
    assert v.ok is True and "no JSON" in v.note


def test_unparseable_json_is_not_treated_as_a_failure():
    # Braces present so the extractor finds a candidate, contents invalid.
    # Without the closing brace this takes the "no JSON" path instead, which
    # is a different branch and would leave this one untested.
    v = vlqa._parse('{"ok": tru}', "x", 2)
    assert v.ok is True and "unparseable" in v.note


def test_findings_imply_not_ok_when_the_model_omits_the_flag():
    v = vlqa._parse(json.dumps({"findings": ["face cut by the frame edge"]}),
                    "x", 4)
    assert v.ok is False


def test_blank_findings_are_dropped():
    v = vlqa._parse(json.dumps({"ok": True, "findings": ["", "  "]}), "x", 1)
    assert v.findings == []


# --- the chain -----------------------------------------------------------

def test_the_judge_is_off_when_the_operator_turns_it_off():
    v = vlqa.judge(Path("x.mp4"), cfg=cfg(vl_qa=False), duration_s=10)
    assert not v.ran and "vl_qa is off" in v.note


def test_a_clip_with_no_decodable_frame_yields_a_verdict_not_an_exception():
    v = vlqa.judge(Path("does-not-exist.mp4"), cfg=cfg(), duration_s=10)
    assert not v.ran and "frame" in v.note


def test_the_chain_is_tried_in_order_and_local_is_last():
    assert vlqa.CHAIN == ("anthropic", "openai", "gemini", "local")


def test_a_link_with_no_key_is_skipped_with_a_reason(monkeypatch):
    """Skipped, and SAID so. Silence here is the failure mode."""
    monkeypatch.setattr(vlqa, "sample_frames", lambda *a, **k: [b"jpg"])
    monkeypatch.setattr("clipforge.cloud.provider_key",
                        lambda *a, **k: None)
    monkeypatch.setattr(vlqa, "judge_local",
                        lambda *a, **k: vlqa.Verdict(provider="local"))
    v = vlqa.judge(Path("x.mp4"), cfg=cfg(), duration_s=10)
    assert v.provider == "local"
    assert len(v.skipped) == 3
    assert all("no key" in s for s in v.skipped)


def test_the_first_link_with_a_key_wins(monkeypatch):
    monkeypatch.setattr(vlqa, "sample_frames", lambda *a, **k: [b"jpg"])
    monkeypatch.setattr("clipforge.cloud.provider_key",
                        lambda cfg, feature, provider: "k")
    monkeypatch.setattr(
        vlqa, "_HOSTED",
        {n: (lambda f, k, *, model, _n=n: vlqa.Verdict(provider=_n))
         for n in ("anthropic", "openai", "gemini")})
    v = vlqa.judge(Path("x.mp4"), cfg=cfg(), duration_s=10)
    assert v.provider == "anthropic" and v.skipped == []


def test_a_link_that_raises_does_not_take_the_pipeline_down(monkeypatch):
    """A 429 from a hosted judge must cost a skip, not a failed render."""
    monkeypatch.setattr(vlqa, "sample_frames", lambda *a, **k: [b"jpg"])
    monkeypatch.setattr("clipforge.cloud.provider_key",
                        lambda cfg, feature, provider: "k"
                        if provider == "anthropic" else None)

    def boom(*a, **k):
        raise RuntimeError("HTTP 429: slow down")

    monkeypatch.setattr(vlqa, "_HOSTED", dict(vlqa._HOSTED, anthropic=boom))
    monkeypatch.setattr(vlqa, "judge_local",
                        lambda *a, **k: vlqa.Verdict(provider="local"))
    v = vlqa.judge(Path("x.mp4"), cfg=cfg(), duration_s=10)
    assert v.provider == "local"
    assert any("429" in s for s in v.skipped)


def test_when_every_link_declines_the_verdict_says_so(monkeypatch):
    monkeypatch.setattr(vlqa, "sample_frames", lambda *a, **k: [b"jpg"])
    monkeypatch.setattr("clipforge.cloud.provider_key", lambda *a, **k: None)

    def boom(*a, **k):
        raise RuntimeError("no weights")

    monkeypatch.setattr(vlqa, "judge_local", boom)
    v = vlqa.judge(Path("x.mp4"), cfg=cfg(), duration_s=10)
    assert not v.ran and "declined or failed" in v.note


# --- the gate ------------------------------------------------------------

def test_the_judge_cannot_get_a_key_while_use_cloud_is_off():
    """The whole point of the chokepoint, asserted from this side of it."""
    from clipforge import cloud

    assert cloud.provider_key(cfg(use_cloud=False), feature="vl_qa",
                              provider="gemini") is None


def test_an_unregistered_provider_cannot_ask_for_a_key():
    from clipforge import cloud

    with pytest.raises(cloud.UnknownCloudProvider):
        cloud.provider_key(cfg(use_cloud=True), feature="vl_qa",
                           provider="some-new-vendor")


def test_vl_qa_is_a_registered_cloud_feature():
    from clipforge import cloud

    assert "vl_qa" in cloud.registered_features()
