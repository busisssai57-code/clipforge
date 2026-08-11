"""The Determinism Law, applied to local text-to-video generation.

§3.2 says same input bytes + same config = same output bytes, and names
fixed seeds as the mechanism. The local diffusers path shipped with no
seed at all: ``call_kwargs`` carried prompt, size, steps and guidance and
nothing that pinned the noise, so every run drew from torch's global RNG.

That was not caught by inspection because a comment in the same file
described a measurement taken "on identical prompts and seeds" — the code
was read as if the claim were true of it. It was found instead by running
the same swarm task twice and comparing bytes: shot_00 came back as
2,684,792 bytes on the first run and 2,625,493 on the second.

These tests pin the fix at the two points where it can be reverted: the
call kwargs, and the wiring that carries the configured seed down to the
provider.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

import pytest

from clipforge.genvideo.providers import LocalDiffusersProvider

torch = pytest.importorskip("torch")


class _FakePipe:
    """Stands in for a diffusers video pipeline.

    ``__call__`` takes the arguments LTX-family pipelines take, because
    the provider introspects the signature to decide what to pass — a
    fake with ``**kwargs`` would accept everything and prove nothing.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.vae = types.SimpleNamespace(
            to=lambda **_: None, enable_tiling=lambda: None)

    def to(self, *_a, **_k) -> "_FakePipe":
        return self

    def enable_model_cpu_offload(self) -> None:
        pass

    def __call__(self, *, prompt=None, negative_prompt=None, width=None,
                 height=None, num_frames=None, num_inference_steps=None,
                 guidance_scale=None, generator=None, decode_timestep=None,
                 decode_noise_scale=None, frame_rate=None):
        self.calls.append({"prompt": prompt, "generator": generator})
        # A real pipeline consumes the generator; draw from it so the test
        # can compare the noise two runs would actually have started from.
        draw = (torch.randn(4, generator=generator).tolist()
                if generator is not None else None)
        return types.SimpleNamespace(frames=[[object()]], _draw=draw)


class _UnseedablePipe(_FakePipe):
    """A pipeline whose ``__call__`` takes no generator at all."""

    def __call__(self, *, prompt=None, negative_prompt=None, width=None,
                 height=None, num_frames=None, num_inference_steps=None,
                 guidance_scale=None):
        self.calls.append({"prompt": prompt, "generator": None})
        return types.SimpleNamespace(frames=[[object()]])


@pytest.fixture
def fake_diffusers(monkeypatch):
    """Install a fake ``diffusers`` module and neutralise the encode step.

    ``_write_video`` shells out to ffmpeg over real frames; this test is
    about what reaches the pipeline, so the encode is replaced rather
    than performed.
    """
    made: list[_FakePipe] = []

    def _install(pipe_cls=_FakePipe):
        mod = types.ModuleType("diffusers")
        # ``available()`` asks importlib for a spec; a bare ModuleType has
        # ``__spec__ = None``, which find_spec treats as an error.
        mod.__spec__ = importlib.machinery.ModuleSpec("diffusers", None)

        class _DP:
            @staticmethod
            def from_pretrained(model_id, **_kw):
                pipe = pipe_cls()
                made.append(pipe)
                return pipe

        mod.DiffusionPipeline = _DP  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "diffusers", mod)
        monkeypatch.setattr(
            "clipforge.genvideo.providers._write_video",
            lambda *a, **k: None)
        return made

    return _install


def _run(provider: LocalDiffusersProvider, tmp_path: Path):
    provider.generate(prompt="a lone figure on a shoreline", seconds=2.0,
                      fps=24, out_path=tmp_path / "shot.mp4")


def test_generator_is_passed_and_carries_the_configured_seed(
        fake_diffusers, tmp_path):
    """The call must carry a generator, seeded from the provider."""
    made = fake_diffusers()
    _run(LocalDiffusersProvider("fake/model", seed=99), tmp_path)

    assert len(made) == 1
    gen = made[0].calls[0]["generator"]
    assert gen is not None, (
        "no generator reached the pipeline: generation draws from torch's "
        "global RNG and §3.2 cannot hold")
    assert gen.initial_seed() == 99


def test_same_seed_reproduces_the_same_noise(fake_diffusers, tmp_path):
    """Two runs at one seed start from identical noise; a different seed
    does not. This is the property the byte-size mismatch violated."""
    made = fake_diffusers()
    _run(LocalDiffusersProvider("fake/model", seed=1234), tmp_path)
    _run(LocalDiffusersProvider("fake/model", seed=1234), tmp_path)
    _run(LocalDiffusersProvider("fake/model", seed=4321), tmp_path)

    first, second, other = (p.calls[0]["generator"] for p in made)
    a = torch.randn(8, generator=first).tolist()
    b = torch.randn(8, generator=second).tolist()
    c = torch.randn(8, generator=other).tolist()
    assert a == b
    assert a != c


def test_generator_is_on_cpu_because_the_pipeline_is_offloaded(
        fake_diffusers, tmp_path):
    """A CUDA generator fights ``enable_model_cpu_offload`` and makes the
    result depend on where a submodule happened to live."""
    made = fake_diffusers()
    _run(LocalDiffusersProvider("fake/model", seed=7), tmp_path)
    assert made[0].calls[0]["generator"].device.type == "cpu"


def test_unseedable_pipeline_is_reported_not_ignored(
        fake_diffusers, tmp_path, capsys, caplog):
    """A pipeline that takes no generator cannot satisfy §3.2. Passing
    silently is how the first version looked correct for two weeks.

    Both sinks are read, and that is not belt-and-braces. structlog's
    destination depends on whether :mod:`clipforge.log` has configured it
    yet: run alone, this file renders to the console and ``caplog`` is
    empty; run inside the full suite another test has already installed
    the stdlib bridge and ``capsys`` is empty instead. Asserting on one of
    them passes in one invocation and fails in the other — which is
    exactly what happened before this comment existed.
    """
    import logging

    made = fake_diffusers(_UnseedablePipe)
    with caplog.at_level(logging.WARNING):
        _run(LocalDiffusersProvider("fake/model", seed=7), tmp_path)
    emitted = capsys.readouterr().out + caplog.text

    assert made[0].calls[0]["generator"] is None
    assert "unseedable" in emitted, (
        "an unseedable pipeline must say so; silence reads as determinism")


def test_build_router_threads_the_configured_seed(monkeypatch, tmp_path):
    """Both construction paths must carry the seed.

    The fallback branch is the one that matters: it is taken whenever no
    registered model fits, which is the common state on a machine with no
    weights downloaded yet.
    """
    from clipforge import genvideo
    from clipforge.config import load_config
    from clipforge.paths import Workspace

    # Anchored on __file__ (test_config.py convention) so the test does
    # not depend on pytest's invocation cwd.
    repo = Path(__file__).resolve().parents[2]
    cfg = load_config(repo / "config" / "config.example.toml")
    cfg.genvideo.local_seed = 4242
    cfg.genvideo.use_cloud = False
    ws = Workspace(tmp_path)

    # selected-model path
    router = genvideo.build_router(cfg, ws)
    local = [p for p in router.providers if p.name == "local"]
    assert local and all(p.seed == 4242 for p in local)

    # fallback path: nothing in the registry fits
    monkeypatch.setattr("clipforge.genvideo.models.select_model",
                        lambda **_: (_ for _ in ()).throw(ValueError("none")))
    router = genvideo.build_router(cfg, ws)
    local = [p for p in router.providers if p.name == "local"]
    assert local and all(p.seed == 4242 for p in local), (
        "the fallback branch dropped the seed — the branch taken when no "
        "weights are downloaded is the one most likely to run")
