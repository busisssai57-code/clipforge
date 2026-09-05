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

import subprocess

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


# ------------------------------------------- i2v capability, honestly

class HonestT2VProvider(FakeProvider):
    """Declares start_image to satisfy the interface, cannot use it.

    This is not hypothetical: SubprocessModelProvider was exactly this
    until 2026-09-05, and the router threaded five last frames into it.
    """

    def supports_start_image(self) -> bool:
        return False

    def generate(self, *, prompt, seconds, fps, out_path, negative="",
                 aspect_ratio="9:16", start_image=None):
        return super().generate(prompt=prompt, seconds=seconds, fps=fps,
                                out_path=out_path, negative=negative,
                                aspect_ratio=aspect_ratio)


def test_a_provider_that_says_it_cannot_chain_is_believed():
    """The signature says yes and the provider says no. The provider wins.

    Measured 2026-09-05: because this asked the signature, a five-shot
    batch with continuity ON came back BYTE-IDENTICAL to one with it OFF.
    The frames were extracted, written, handed over and dropped.
    """
    from clipforge.genvideo.router import _takes_start_image

    assert _takes_start_image(HonestT2VProvider("t2v")) is False


def test_a_provider_that_can_chain_still_receives_frames():
    from clipforge.genvideo.router import _takes_start_image

    class I2V(HonestT2VProvider):
        def supports_start_image(self) -> bool:
            return True

    assert _takes_start_image(I2V("i2v")) is True


def test_a_provider_with_no_opinion_falls_back_to_the_signature():
    """The original behaviour, kept: a provider that gains i2v support
    without adding the method should still start receiving frames."""
    from clipforge.genvideo.router import _takes_start_image

    assert _takes_start_image(ChainProvider("chain")) is True
    assert _takes_start_image(FakeProvider("plain")) is False


def test_the_ltx25_provider_now_declares_i2v():
    from clipforge.genvideo.models import REGISTRY
    from clipforge.genvideo.subproc import SubprocessModelProvider

    assert SubprocessModelProvider(REGISTRY["ltx25"]).supports_start_image()


def test_the_provider_puts_the_start_frame_in_the_request():
    """Claiming i2v and not sending the frame is the same bug, one layer up.

    A mutant that deleted this key left all 25 other tests green: the
    provider said supports_start_image() -> True, the router threaded the
    frame in, generate() accepted it and never forwarded it. Asserted
    against the source because building a real request spawns the worker.
    """
    import inspect

    from clipforge.genvideo import subproc

    src = inspect.getsource(subproc.SubprocessModelProvider.generate)
    assert '"start_image"' in src, (
        "the request must carry the frame, or supports_start_image() lies")
    assert "str(start_image) if start_image else None" in src, (
        "and it must carry the CALLER's frame, not a constant")


def test_the_worker_actually_runs_the_i2v_pipeline():
    """Declaring i2v without doing it is the defect this replaced."""
    import inspect

    from clipforge.genvideo import _ltx_worker

    src = inspect.getsource(_ltx_worker._generate)
    assert 'req.get("start_image")' in src
    assert "active(image=seed_frame" in src, (
        "a start frame must reach the pipeline as `image=`")
    assert "result = pipe(**kw)" in src, "the no-frame path must still exist"
    built = inspect.getsource(_ltx_worker._i2v_pipe)
    assert "from_pipe" in built, "a second full load would not fit on the card"


# ------------------------------------------------- attention dispatch

@pytest.mark.parametrize("backend", ["", "_native_flash", "_not_a_backend"])
def test_the_attention_context_yields_exactly_once(backend):
    """A generator context manager may not yield twice.

    The first version wrapped the body in try/except around a `with` and
    yielded again from the handler, so ANY error inside the render came
    back as "generator didn't stop after throw()" and a kernel that was
    documented as optional killed the shot. Measured 2026-09-05: a whole
    speed run lost to it.
    """
    from clipforge.genvideo import _ltx_worker

    with pytest.raises(ValueError, match="boom"):
        with _ltx_worker._attention(backend):
            raise ValueError("boom")


@pytest.mark.parametrize("backend", ["", "_not_a_backend"])
def test_an_unusable_backend_still_runs_the_shot(backend):
    """A kernel is an optimisation, not a requirement."""
    from clipforge.genvideo import _ltx_worker

    ran = False
    with _ltx_worker._attention(backend):
        ran = True
    assert ran


# --------------------------------------------- guidance pass counting

def test_the_pipeline_defaults_would_cost_four_passes_a_step():
    """Why a 1.9s shot took ten minutes, as arithmetic.

    LTX2Pipeline invokes self.transformer THREE times per step: the CFG
    batch (2 passes' compute), an STG pass, and a modality-isolation pass.
    ITS defaults (stg 1.0, modality 3.0) switch both extras on, so 30
    steps is 120 passes. This pins what we would inherit by saying
    nothing -- which is exactly what this project did until measured.
    """
    import dataclasses

    from clipforge.genvideo.models import REGISTRY, guidance_passes_per_step

    as_pipeline_ships = dataclasses.replace(
        REGISTRY["ltx25"], stg_scale=1.0, audio_stg_scale=1.0,
        modality_scale=3.0, audio_modality_scale=3.0)
    assert guidance_passes_per_step(as_pipeline_ships) == 4
    assert guidance_passes_per_step(as_pipeline_ships) * 30 == 120


def test_our_schedule_costs_two_passes_a_step():
    """MEASURED: 616.9s at 4 passes, 344.7s at 2, and the faster frame is
    not worse. The extras are off because they bought nothing here."""
    from clipforge.genvideo.models import REGISTRY, guidance_passes_per_step

    spec = REGISTRY["ltx25"]
    assert guidance_passes_per_step(spec) == 2
    assert guidance_passes_per_step(spec) * spec.steps == 60


def test_the_audio_scales_alone_keep_both_extra_passes_alive():
    """The `or` is the whole point, and it is expensive.

    Zeroing only the VIDEO scales changes nothing: the pipeline's guards
    are `stg_scale > 0 OR audio_stg_scale > 0`. This model's audio has
    measured silent on every shot, so those are two passes per step spent
    guiding a track that is thrown away.
    """
    import dataclasses

    from clipforge.genvideo.models import REGISTRY, guidance_passes_per_step

    # Start from what the PIPELINE ships, since our own spec now has the
    # audio scales off too -- the trap is only visible from the defaults
    # you inherit by saying nothing.
    shipped = dataclasses.replace(
        REGISTRY["ltx25"], stg_scale=1.0, audio_stg_scale=1.0,
        modality_scale=3.0, audio_modality_scale=3.0)
    half = dataclasses.replace(shipped, stg_scale=0.0, modality_scale=1.0)
    assert guidance_passes_per_step(half) == 4, (
        "zeroing only the video scales changes nothing: the guards are "
        "`stg_scale > 0 OR audio_stg_scale > 0`")


def test_the_scales_reach_the_worker_request():
    import inspect

    from clipforge.genvideo import subproc

    src = inspect.getsource(subproc.SubprocessModelProvider.generate)
    for key in ("stg_scale", "audio_stg_scale", "modality_scale",
                "audio_modality_scale"):
        assert f'"{key}"' in src, f"{key} must reach the worker"


# ------------------------------------------------- Wan2GP step cache

def test_the_subprocess_provider_carries_the_step_cache_threshold():
    """It used to log `step_cache_unsupported` and drop the value.

    That was true of the worker as written and false of the pipeline:
    LTX2Pipeline has no CacheMixin, but LTX2VideoTransformer3DModel does,
    so the hook was one level below where LocalDiffusersProvider looks. The
    threshold must reach the request or the worker cannot act on it.
    """
    from clipforge.genvideo.models import REGISTRY
    from clipforge.genvideo.subproc import SubprocessModelProvider

    provider = SubprocessModelProvider(REGISTRY["ltx25"],
                                       step_cache_threshold=0.05)
    assert provider.step_cache_threshold == 0.05


def test_the_worker_sets_the_cache_on_every_request_not_at_load():
    """The pipeline is cached across calls keyed on (model_id, quantize).

    A threshold left on the transformer would outlive the run that asked
    for it -- the same unkeyed-cache defect the loader already guards
    against. Asserted against the source because engaging it needs 13 GB
    of weights and a card.
    """
    import inspect

    from clipforge.genvideo import _ltx_worker

    src = inspect.getsource(_ltx_worker)
    assert "_apply_step_cache(pipe, float(req.get(\"step_cache\"" in src, (
        "the cache must be set from the REQUEST, inside _generate")
    apply_src = inspect.getsource(_ltx_worker._apply_step_cache)
    assert "disable_cache" in apply_src, (
        "each request must clear the previous request's cache state first")
    assert "FirstBlockCacheConfig" in apply_src


def test_a_zero_threshold_disables_rather_than_skips():
    """0.0 must actively turn the cache OFF, not leave whatever was set."""
    import inspect

    from clipforge.genvideo import _ltx_worker

    src = inspect.getsource(_ltx_worker._apply_step_cache)
    disable_at = src.index("disable_cache")
    guard_at = src.index("if threshold <= 0")
    assert disable_at < guard_at, (
        "disable_cache must run BEFORE the zero-threshold early return, "
        "or a run at 0.0 inherits the previous run's cache")


# ------------------------------------------------------- continuity

class ChainProvider(FakeProvider):
    """A provider that records the start_image it was handed per shot."""

    def __init__(self, name: str):
        super().__init__(name)
        self.start_images: list = []

    def generate(self, *, prompt, seconds, fps, out_path, negative="",
                 aspect_ratio="9:16", start_image=None):
        self.start_images.append(start_image)
        self.calls += 1
        # Honour the script like FakeProvider does. Without this a
        # failure-injection test cannot inject a failure and passes while
        # proving nothing -- which is what the gap test did first time.
        if self.script:
            outcome = self.script.pop(0)
            if outcome is not None:
                raise outcome
        # A REAL file, not 16 zero bytes: the chain is built by pulling the
        # last frame out of the previous shot with ffmpeg, so a fake that
        # writes rubbish tests nothing and reports the feature as broken.
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
             "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=0.3",
             "-pix_fmt", "yuv420p", str(out_path)],
            check=True, capture_output=True)
        return GenResult(Path(out_path), self.name, seconds, prompt,
                         f"{self.name}-model")


def test_continuity_anchors_every_shot_to_the_same_frame(ledger, clock, tmp_path):
    """Shot 0 starts from text; every later shot starts from THE ANCHOR.

    Not from its immediate predecessor. A rolling chain compounds drift
    twice over -- a shot is worst on its last frame, and that frame then
    becomes the next shot's truth -- and by shot 2 of a real batch the
    goat had fused with the child, three horns growing out of the
    toddler's scalp. An anchor keeps every shot one generation from a
    clean reference rather than N.
    """
    local = ChainProvider("local")
    router = _router(ledger, clock, local)
    router.generate_sequence(brief="One. Two. Three.",
                             preset=get_preset("documentary"),
                             out_dir=tmp_path / "seq", shots=3,
                             continuity=True)
    assert local.start_images[0] is None, "the first shot has nothing to chain from"
    assert all(x is not None for x in local.start_images[1:]), (
        "later shots must be seeded: %r" % (local.start_images,))
    # THE anchor property: one reference, not a moving one.
    seeds = set(map(str, local.start_images[1:]))
    assert len(seeds) == 1, (
        "every seeded shot must use the SAME anchor; a per-shot seed is "
        "the rolling chain that compounded drift: %r" % (seeds,))
    assert "anchor" in next(iter(seeds)), (
        "the anchor is a named file, not shot N's trailing frame")


def test_a_failed_shot_does_not_destroy_the_anchor(ledger, clock, tmp_path):
    """A gap falsifies frame-continuity, not the identity of the scene.

    When this was a rolling chain, clearing the seed after a failure was
    right. An anchor asserts something weaker and still true -- same
    child, same place -- so a missing beat must not send every later shot
    back to text and change the wardrobe mid-piece.
    """
    local = ChainProvider("local")
    local.script = [None, ProviderError("hiccup"), None, None]
    router = _router(ledger, clock, local)
    router.generate_sequence(brief="One. Two. Three. Four.",
                             preset=get_preset("documentary"),
                             out_dir=tmp_path / "seq", shots=4,
                             continuity=True)
    # The shot IMMEDIATELY after the gap must carry the anchor -- not be
    # sent back to text and re-anchored on itself, which is what clearing
    # the seed does and which a weaker assertion here did not catch.
    assert local.start_images[2] is not None, (
        "the beat after a gap must still be anchored, not restarted from "
        "text: %r" % (local.start_images,))
    assert str(local.start_images[2]) == str(local.start_images[1] or
                                             local.start_images[2]), "same anchor"
    seeded = {str(x) for x in local.start_images if x is not None}
    assert len(seeded) == 1, (
        "a gap must not mint a second anchor: %r" % (seeded,))


def test_without_continuity_no_shot_is_seeded(ledger, clock, tmp_path):
    """The default must stay a genuine default, not a no-op flag."""
    local = ChainProvider("local")
    router = _router(ledger, clock, local)
    router.generate_sequence(brief="One. Two. Three.",
                             preset=get_preset("documentary"),
                             out_dir=tmp_path / "seq", shots=3)
    assert local.start_images == [None, None, None]


def test_the_cli_passes_the_presets_continuity_through(): 
    """The wiring, not the feature.

    `continuity` was fully implemented in `generate_sequence`, supported by
    the ltx25 provider, and passed by NOBODY -- the CLI called
    generate_sequence without it, so every run ever made used the default.
    A five-shot ari_goat batch on 2026-09-05 came back with the child in a
    different shirt in every shot, which is what that looks like from the
    outside. Asserted against the source because the real call renders.
    """
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli)
    assert "continuity=chosen.continuity" in src, (
        "the CLI must hand generate_sequence the preset's own answer; "
        "omitting it is what kept this feature off for every run")


def test_the_one_scene_niche_asks_for_continuity():
    """ari_goat is one child, one goat, one afternoon."""
    from clipforge.niches import resolve_preset

    assert resolve_preset("ari_goat").continuity is True
    # And it stays a per-niche decision, not a new global default.
    assert resolve_preset("documentary").continuity is False


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
