"""Metered-cloud → local failover, and the return trip. Pinned.

The requirement: use the premium provider until quota runs out, fall back
to the open-source one, and go BACK when the window resets. The hard part
is not the fallback — it is (a) remembering exhaustion across processes so
a dead API is not re-probed every run, and (b) not confusing "no API key"
with "quota spent", which would sideline a provider for a day the moment
the operator forgets to set a key.

Time is injected everywhere. Nothing here sleeps.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from clipforge.genvideo.presets import get_preset
from clipforge.genvideo.providers import (GenResult, ProviderError,
                                          ProviderUnavailable, QuotaExhausted)
from clipforge.genvideo.quota import MAX_BACKOFF_S, QuotaLedger
from clipforge.genvideo.router import GenerationRouter


class FakeClock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class FakeProvider:
    """Scriptable provider. `script` is consumed one entry per call:
    None = succeed, or an exception instance to raise."""

    def __init__(self, name: str, script=None, configured: bool = True):
        self.name = name
        self.script = list(script or [])
        self.configured = configured
        self.calls = 0

    def available(self) -> bool:
        return self.configured

    def generate(self, *, prompt, seconds, fps, out_path, negative="",
                 aspect_ratio="9:16"):
        self.calls += 1
        if self.script:
            outcome = self.script.pop(0)
            if outcome is not None:
                raise outcome
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_bytes(b"\x00" * 16)
        return GenResult(Path(out_path), self.name, seconds, prompt,
                         f"{self.name}-model")


@pytest.fixture()
def clock():
    return FakeClock()


@pytest.fixture()
def ledger(tmp_path, clock):
    return QuotaLedger.load(tmp_path / "quota.json", clock=clock)


def _router(ledger, clock, *providers):
    return GenerationRouter(list(providers), ledger, clock=clock)


def _shot(router, tmp_path, name="a.mp4"):
    return router.generate_shot(prompt="p", seconds=5.0, fps=24,
                                out_path=tmp_path / name)


# ------------------------------------------------------- happy path

def test_the_premium_provider_is_used_while_it_has_quota(ledger, clock,
                                                          tmp_path):
    veo, local = FakeProvider("veo"), FakeProvider("local")
    res = _shot(_router(ledger, clock, veo, local), tmp_path)
    assert res.provider == "veo"
    assert local.calls == 0, "the fallback must not run while quota remains"


# ----------------------------------------------------- the fallback

def test_quota_exhaustion_falls_back_to_the_local_provider(ledger, clock,
                                                            tmp_path):
    veo = FakeProvider("veo", script=[QuotaExhausted("out of quota")])
    local = FakeProvider("local")
    res = _shot(_router(ledger, clock, veo, local), tmp_path)
    assert res.provider == "local"
    assert not ledger.available("veo"), "exhaustion was not recorded"


def test_exhaustion_survives_a_new_process(tmp_path, clock):
    """The ledger is the point: a second run must not re-probe a dead API."""
    path = tmp_path / "quota.json"
    first = QuotaLedger.load(path, clock=clock)
    veo = FakeProvider("veo", script=[QuotaExhausted("out")])
    _router(first, clock, veo, FakeProvider("local")).generate_shot(
        prompt="p", seconds=5.0, fps=24, out_path=tmp_path / "a.mp4")

    reborn = QuotaLedger.load(path, clock=clock)         # new "process"
    veo2, local2 = FakeProvider("veo"), FakeProvider("local")
    res = _router(reborn, clock, veo2, local2).generate_shot(
        prompt="p", seconds=5.0, fps=24, out_path=tmp_path / "b.mp4")
    assert res.provider == "local"
    assert veo2.calls == 0, (
        "a fresh process re-probed an API it had already recorded as out "
        "of quota")


# -------------------------------------------------- the return trip

def test_it_returns_to_the_premium_provider_when_the_window_resets(
        ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[QuotaExhausted("out", retry_after_s=3600)])
    local = FakeProvider("local")
    router = _router(ledger, clock, veo, local)
    assert _shot(router, tmp_path, "a.mp4").provider == "local"

    clock.advance(3599)
    assert _shot(router, tmp_path, "b.mp4").provider == "local", (
        "came back before the window actually reset")

    clock.advance(2)
    assert _shot(router, tmp_path, "c.mp4").provider == "veo", (
        "did not return to the premium provider after the reset")


def test_the_providers_own_retry_after_is_respected(ledger, clock, tmp_path):
    """A per-minute limit must not strand the premium provider for a day."""
    veo = FakeProvider("veo", script=[QuotaExhausted("slow down",
                                                     retry_after_s=30)])
    router = _router(ledger, clock, veo, FakeProvider("local"))
    _shot(router, tmp_path, "a.mp4")
    clock.advance(31)
    assert _shot(router, tmp_path, "b.mp4").provider == "veo"


def test_an_absurd_retry_after_is_clamped(ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[QuotaExhausted("bad", retry_after_s=1e12)])
    router = _router(ledger, clock, veo, FakeProvider("local"))
    _shot(router, tmp_path, "a.mp4")
    assert ledger.seconds_until_available("veo") <= MAX_BACKOFF_S


# ------------------------------- an unconfigured provider is not quota

def test_a_missing_api_key_is_not_recorded_as_spent_quota(ledger, clock,
                                                           tmp_path):
    """Writing 'unconfigured' to the ledger would sideline the provider
    for a day AFTER the operator finally sets the key."""
    veo = FakeProvider("veo", script=[ProviderUnavailable("no key")])
    local = FakeProvider("local")
    router = _router(ledger, clock, veo, local)
    assert _shot(router, tmp_path, "a.mp4").provider == "local"
    assert ledger.available("veo"), (
        "a missing key was recorded as a quota penalty")
    assert ledger.state("veo").calls == 0


def test_an_unconfigured_provider_is_skipped_for_the_rest_of_the_run(
        ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[ProviderUnavailable("no key")])
    router = _router(ledger, clock, veo, FakeProvider("local"))
    _shot(router, tmp_path, "a.mp4")
    _shot(router, tmp_path, "b.mp4")
    assert veo.calls == 1, "kept retrying a provider that has no credentials"


# ------------------------------------------------- transient failures

def test_a_transient_error_falls_back_without_burning_the_quota_window(
        ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[ProviderError("502 upstream")])
    router = _router(ledger, clock, veo, FakeProvider("local"))
    assert _shot(router, tmp_path, "a.mp4").provider == "local"
    assert ledger.available("veo"), (
        "one transient error should not cost a quota window")


def test_repeated_errors_sideline_a_broken_provider(ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[ProviderError("boom")] * 3)
    router = _router(ledger, clock, veo, FakeProvider("local"))
    for i in range(3):
        _shot(router, tmp_path, f"{i}.mp4")
    assert not ledger.available("veo"), (
        "a provider failing every call must stop being tried first")


def test_everything_failing_raises_rather_than_returning_nothing(
        ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[QuotaExhausted("out")])
    local = FakeProvider("local", script=[ProviderError("no weights")])
    with pytest.raises(ProviderError) as err:
        _shot(_router(ledger, clock, veo, local), tmp_path)
    assert "veo" in str(err.value) and "local" in str(err.value), (
        "the error must name what every provider said")


# ------------------------------------------------------- sequences

def test_a_sequence_keeps_going_when_one_shot_fails(ledger, clock, tmp_path):
    """Five good shots and a gap is still cuttable; losing the run is not."""
    local = FakeProvider("local", script=[None, ProviderError("hiccup"), None])
    router = _router(ledger, clock, local)
    out = router.generate_sequence(brief="A river at dawn. Then the town.",
                                   preset=get_preset("documentary"),
                                   out_dir=tmp_path / "seq", shots=3)
    assert out.ok_count == 2
    assert out.degraded is True
    assert [s.error != "" for s in out.shots] == [False, True, False]


def test_a_sequence_that_flips_provider_midway_is_marked_degraded(
        ledger, clock, tmp_path):
    veo = FakeProvider("veo", script=[None, QuotaExhausted("out")])
    router = _router(ledger, clock, veo, FakeProvider("local"))
    out = router.generate_sequence(brief="One. Two. Three.",
                                   preset=get_preset("explainer"),
                                   out_dir=tmp_path / "seq", shots=3)
    assert out.ok_count == 3
    assert out.providers_used == ["veo", "local"]
    assert out.degraded is True, (
        "a piece whose shots came from two different models is not uniform "
        "and the caller must be told")


def test_status_reports_each_provider_for_the_dashboard(ledger, clock,
                                                         tmp_path):
    veo = FakeProvider("veo", script=[QuotaExhausted("out", retry_after_s=600)])
    router = _router(ledger, clock, veo, FakeProvider("local"))
    _shot(router, tmp_path, "a.mp4")
    rows = {r["name"]: r for r in router.status()}
    assert rows["veo"]["quota_ok"] is False
    assert 0 < rows["veo"]["available_in_s"] <= 600
    assert rows["local"]["quota_ok"] is True
    assert rows["local"]["calls"] == 1
