"""The provider whose model lives in another interpreter.

None of this loads LTX-2.5 — a 22 GB checkpoint is not a unit test. What
is testable here is everything around it, and it is the part that has
historically broken in this repo: a feature with no caller, a resource
held past the work that needed it, and a claim in a docstring nothing
executes.

So: the interpreter is resolved the way the spec says, the builder
actually reaches for this provider, the router gives the card back when a
sequence ends however it ends, and the JSON line protocol survives a
round trip against the REAL worker file (its `ping` op imports nothing)
and against a stub that returns frames.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from clipforge.genvideo import subproc
from clipforge.genvideo.models import LTX_25, WAN22_TI2V_5B
from clipforge.genvideo.providers import ProviderUnavailable
from clipforge.genvideo.subproc import SubprocessModelProvider, interpreter_for

REPO = Path(__file__).resolve().parents[2]


# ------------------------------------------------------- interpreter

@pytest.fixture()
def stubbed_weights(monkeypatch):
    """Let the stub worker stand in for the 22 GB checkpoint.

    ``available()`` insists three things are present: the interpreter, the
    worker file and the WEIGHTS. The tests below replace the first two
    deliberately — that is the whole design, a stub worker over the real
    line protocol — and then fell at the third on any machine without the
    download, which is every machine but the one this was written on. So
    nine tests covering the pipe, the lifecycle and the card handoff were
    reporting a broken provider instead of exercising it.

    Not autouse: `test_a_missing_interpreter_is_unavailable_not_a_crash`
    is about that check saying no, and must keep reaching it.
    """
    from clipforge.genvideo import models as _models
    monkeypatch.setattr(_models, "weights_present", lambda spec: True)



def test_interpreter_is_none_for_a_model_that_runs_in_this_venv():
    assert interpreter_for(WAN22_TI2V_5B) is None


def test_interpreter_resolves_against_the_repo(monkeypatch):
    monkeypatch.delenv("CLIPFORGE_ALT_PYTHON", raising=False)
    got = interpreter_for(LTX_25, repo_root=Path("/repo"))
    assert got == Path("/repo") / LTX_25.interpreter


def test_the_env_override_wins(monkeypatch):
    """An operator may keep the second environment anywhere."""
    monkeypatch.setenv("CLIPFORGE_ALT_PYTHON", "/elsewhere/python")
    assert interpreter_for(LTX_25) == Path("/elsewhere/python")


def test_a_missing_interpreter_is_unavailable_not_a_crash(monkeypatch):
    monkeypatch.setenv("CLIPFORGE_ALT_PYTHON", "/no/such/python")
    provider = SubprocessModelProvider(LTX_25)
    assert provider.available() is False
    with pytest.raises(ProviderUnavailable) as exc:
        provider.generate(prompt="x", seconds=1.0, fps=24,
                          out_path=Path("unused.mp4"))
    # The message has to explain WHY a second interpreter exists, or the
    # next person deletes it as a stray venv.
    assert "huggingface-hub" in str(exc.value)


# ------------------------------------------------------------ wiring

def _cfg_and_ws(tmp_path):
    from clipforge.config import load_config
    from clipforge.paths import Workspace

    cfg = load_config(REPO / "config" / "config.example.toml")
    cfg.genvideo.use_cloud = False
    return cfg, Workspace(tmp_path)


def test_the_builder_reaches_for_the_subprocess_provider(monkeypatch, tmp_path):
    """The bug this repo keeps meeting is a feature with no caller.

    `models.py` can describe a second interpreter and `subproc.py` can
    implement one, and the pipeline still builds a `LocalDiffusersProvider`
    that cannot load the checkpoint. This asserts the branch.
    """
    from clipforge import genvideo

    cfg, ws = _cfg_and_ws(tmp_path)
    monkeypatch.setattr("clipforge.genvideo.models.select_model",
                        lambda **_: LTX_25)
    router = genvideo.build_router(cfg, ws, prefer="ltx25")
    picked = [p for p in router.providers
              if isinstance(p, SubprocessModelProvider)]
    assert picked, [p.name for p in router.providers]
    assert picked[0].quantize == "nf4", (
        "the model's own requirement must reach the worker; unquantized "
        "it is 35 GB of transformer for a 24 GB card")


def test_a_model_without_an_interpreter_still_loads_in_process(monkeypatch,
                                                               tmp_path):
    """The new branch must not swallow the ordinary path."""
    from clipforge import genvideo
    from clipforge.genvideo.providers import LocalDiffusersProvider

    cfg, ws = _cfg_and_ws(tmp_path)
    monkeypatch.setattr("clipforge.genvideo.models.select_model",
                        lambda **_: WAN22_TI2V_5B)
    router = genvideo.build_router(cfg, ws)
    assert any(isinstance(p, LocalDiffusersProvider) for p in router.providers)
    assert not any(isinstance(p, SubprocessModelProvider)
                   for p in router.providers)


# ------------------------------------------------------------ closing

class _CountingProvider:
    """A provider that holds something and records being released."""

    name = "counting"

    def __init__(self, fail: bool = False):
        self.closed = 0
        self.fail = fail

    def available(self) -> bool:
        return True

    def generate(self, *, prompt, seconds, fps, out_path, negative="",
                 aspect_ratio="9:16", start_image=None):
        from clipforge.genvideo.providers import GenResult, ProviderError

        if self.fail:
            raise ProviderError("no")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(b"x" * 2048)
        return GenResult(out_path, self.name, seconds, prompt, "fake")

    def close(self) -> None:
        self.closed += 1


def _router(provider, tmp_path):
    from clipforge.genvideo.quota import QuotaLedger
    from clipforge.genvideo.router import GenerationRouter

    return GenerationRouter([provider],
                            QuotaLedger.load(tmp_path / "quota.json"))


@pytest.mark.parametrize("failing", [False, True])
def test_a_sequence_gives_the_card_back_however_it_ends(tmp_path, failing):
    """A worker held past its sequence blocks every later GPU stage.

    The failing case is the one that matters: the provider raising is
    exactly when a `finally` gets forgotten.
    """
    from clipforge.genvideo.presets import PRESETS

    provider = _CountingProvider(fail=failing)
    router = _router(provider, tmp_path)
    router.generate_sequence(brief="a camel. a market.",
                             preset=next(iter(PRESETS.values())),
                             out_dir=tmp_path / "out", shots=2)
    assert provider.closed == 1


def test_broll_closes_providers_too(tmp_path, monkeypatch):
    """B-roll walks cues itself, so it owns the shutdown too."""
    from clipforge.broll import BRollCue, render_broll

    provider = _CountingProvider(fail=True)   # no cue survives; still closes
    router = _router(provider, tmp_path)
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    render_broll(clip, [BRollCue(start_s=0.0, duration_s=1.0,
                                 subject="camel", prompt="a camel")],
                 router=router, work_dir=tmp_path / "work",
                 width=1080, height=1920)
    assert provider.closed == 1


# ----------------------------------------------------------- protocol

def test_the_real_worker_answers_ping_with_clean_stdout():
    """`ping` imports nothing, so the actual worker file can be run here.

    This is the guard on `sys.stdout = sys.stderr`: diffusers,
    transformers and bitsandbytes all print progress, and one stray line
    on stdout would be read as a reply.
    """
    proc = subprocess.run([sys.executable, str(subproc.WORKER)],
                          input='{"op": "ping"}\n{"op": "quit"}\n',
                          capture_output=True, text=True, timeout=120)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1, f"stdout carried more than the reply: {lines}"
    assert json.loads(lines[0])["ok"] is True
    assert json.loads(lines[0])["loaded"] is False


def test_an_unknown_op_is_an_answer_not_a_hang():
    proc = subprocess.run([sys.executable, str(subproc.WORKER)],
                          input='{"op": "nonsense"}\nnot json\n{"op": "quit"}\n',
                          capture_output=True, text=True, timeout=120)
    replies = [json.loads(ln) for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(replies) == 2 and not any(r["ok"] for r in replies)
    assert "unknown op" in replies[0]["error"]
    assert "unparseable" in replies[1]["error"]


_STUB = '''
import json, sys
import numpy as np
_REPLY = sys.stdout
sys.stdout = sys.stderr
print("library noise on stdout")          # must not reach the parent
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    if req["op"] == "quit":
        break
    if req["op"] == "ping":
        _REPLY.write(json.dumps({"ok": True, "loaded": False}) + "\\n")
    else:
        rng = np.random.default_rng(req["seed"])
        arr = rng.integers(0, 255, size=(9, 64, 64, 3), dtype=np.uint8)
        np.save(req["out"], arr)
        _REPLY.write(json.dumps({"ok": True, "npy": req["out"], "frames": 9,
                                 "height": 64, "width": 64,
                                 "echo": req}) + "\\n")
    _REPLY.flush()
'''


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_frames_round_trip_from_a_worker_into_a_real_video(
        tmp_path, monkeypatch, stubbed_weights):
    """End to end over the pipe, with the model replaced by noise.

    Proves the parts that have nothing to do with the model: the child
    starts, the protocol survives library output on its stdout, frames
    come back as an array, and the PARENT does the encoding — the
    blank-frame guard and ffmpeg flags stay in one place.
    """
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB, encoding="utf-8")
    monkeypatch.setattr(subproc, "WORKER", stub)
    # A budget the test machine certainly has: this asserts nothing about
    # VRAM, it just must not fail on a card that is busy elsewhere.
    spec = dataclasses.replace(LTX_25, vram_gb=0.1)
    provider = SubprocessModelProvider(spec, seed=99, interpreter=Path(sys.executable))
    out = tmp_path / "shot_00.mp4"
    try:
        result = provider.generate(prompt="a camel", seconds=0.4, fps=24,
                                   out_path=out)
    finally:
        provider.close()
    assert out.is_file() and out.stat().st_size > 1024
    assert result.model == LTX_25.model_id
    # The temp array is not left behind next to the deliverable.
    assert not list(out.parent.glob("*.npy"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_the_worker_is_reused_across_shots(tmp_path, monkeypatch,
                                           stubbed_weights):
    """The whole reason this provider is stateful.

    Loading is 81-103 s and a generation 56-61 s, so a process per shot
    nearly doubles a brief. If a later change starts a worker per call,
    nothing else in the suite would notice.
    """
    stub = tmp_path / "stub_worker.py"
    stub.write_text(_STUB, encoding="utf-8")
    monkeypatch.setattr(subproc, "WORKER", stub)
    spec = dataclasses.replace(LTX_25, vram_gb=0.1)
    provider = SubprocessModelProvider(spec, interpreter=Path(sys.executable))
    try:
        provider.generate(prompt="one", seconds=0.4, fps=24,
                          out_path=tmp_path / "a.mp4")
        first = provider._proc.pid
        provider.generate(prompt="two", seconds=0.4, fps=24,
                          out_path=tmp_path / "b.mp4")
        assert provider._proc.pid == first, "a second worker was started"
    finally:
        provider.close()
    assert provider._proc is None


# ------------------------------------------------------- the dead flag

def test_generate_passes_model_through_to_selection(monkeypatch, tmp_path):
    """`--model` has to reach `select_model`, which is where it means
    something.

    Until 2026-08-20 the chain was dead at every link: `swarm plan
    --model` put a key in a payload, the payload carried it through two
    task kinds, `Generator.run` built an argv without it, `bta generate`
    had no such flag, and `build_router(prefer=...)` — which implements
    the one path that lets an unverified model run at all — had no caller
    in the product. The registry's own note told operators to "select it
    explicitly with --model ltx25".
    """
    from clipforge import genvideo

    seen = {}

    def _spy(**kw):
        seen.update(kw)
        return WAN22_TI2V_5B

    cfg, ws = _cfg_and_ws(tmp_path)
    monkeypatch.setattr("clipforge.genvideo.models.select_model", _spy)
    genvideo.build_router(cfg, ws, prefer="ltx25")
    assert seen.get("prefer") == "ltx25"


def test_the_swarm_hands_its_model_key_to_the_generate_call(monkeypatch):
    """The board carried the key; the subprocess call dropped it."""
    from clipforge.swarm import roles

    captured: list[list[str]] = []

    def _fake_run(argv, **_):
        captured.append(list(argv))
        return "ok"

    monkeypatch.setattr(roles, "_run", _fake_run)
    monkeypatch.setattr(roles, "_read_manifest", lambda _p: ["piece.mp4"])
    role = roles.Generator(board=None)
    task = roles.Task(id="t1", kind="generate",
                      payload={"brief": "a camel", "model": "ltx25"})
    role.run(task)
    argv = captured[0]
    assert "--model" in argv and argv[argv.index("--model") + 1] == "ltx25"


def test_the_cli_flag_reaches_the_router(monkeypatch, tmp_path):
    """The link the other test cannot see: `bta generate --model`.

    Typer command functions are ordinary functions (see
    `test_cli_sentinels`), so this calls one — with a sentinel raised
    from inside `build_router` the moment it is reached, because what is
    being asserted is the argument, not the generation.
    """
    import typer

    from clipforge import cli

    class _Reached(Exception):
        def __init__(self, prefer):
            self.prefer = prefer

    def _spy(cfg, ws, **kw):
        raise _Reached(kw.get("prefer"))

    monkeypatch.setattr("clipforge.genvideo.build_router", _spy)
    cfg_path = tmp_path / "config.toml"
    base = (REPO / "config" / "config.example.toml").read_text(encoding="utf-8")
    # The example config ships generation OFF, which is the right default
    # and would make this test assert only that the guard works.
    cfg_path.write_text(
        base.replace('root = "workspace"', f'root = {str(tmp_path / "ws")!r}')
        + "\n[genvideo]\nenabled = true\nuse_cloud = false\n",
        encoding="utf-8")
    with pytest.raises((_Reached, typer.Exit)) as exc:
        cli.generate("a camel in a market", config=cfg_path, model="ltx25")
    assert isinstance(exc.value, _Reached), (
        "generate exited before it built a router; the flag was never used")
    assert exc.value.prefer == "ltx25"


_FAILING_STUB = '''
import json, sys
_REPLY = sys.stdout
sys.stdout = sys.stderr
for line in sys.stdin:
    if not line.strip():
        continue
    req = json.loads(line)
    if req["op"] == "quit":
        break
    if req["op"] == "ping":
        _REPLY.write(json.dumps({"ok": True, "loaded": False}) + "\\n")
    else:
        _REPLY.write(json.dumps({"ok": False, "error": "out of memory"}) + "\\n")
    _REPLY.flush()
'''


def test_a_failed_generation_releases_the_card(tmp_path, monkeypatch,
                                               stubbed_weights):
    """Failing while still holding the GPU breaks the fallback.

    The router answers a ProviderError by trying the next provider, which
    loads its own model on the same card. A worker that survives its own
    failure is 13 GB the fallback cannot have, and nothing else can see
    it — the child's memory is outside the residency registry.
    """
    from clipforge.genvideo.providers import ProviderError

    stub = tmp_path / "failing_worker.py"
    stub.write_text(_FAILING_STUB, encoding="utf-8")
    monkeypatch.setattr(subproc, "WORKER", stub)
    spec = dataclasses.replace(LTX_25, vram_gb=0.1)
    provider = SubprocessModelProvider(spec, interpreter=Path(sys.executable))
    with pytest.raises(ProviderError) as exc:
        provider.generate(prompt="x", seconds=0.4, fps=24,
                          out_path=tmp_path / "a.mp4")
    assert "out of memory" in str(exc.value)
    assert provider._proc is None, "the worker outlived its own failure"


def test_a_worker_that_dies_on_startup_is_reported_at_once(
        tmp_path, monkeypatch, stubbed_weights):
    """A dead child never fills the queue, so waiting on it learns nothing.

    Measured while writing this file: a stub that exited immediately took
    the full 126 s ping timeout and was then reported as "did not
    answer", when the reason had been in its stderr from the first
    second. The queue is now polled in slices with a liveness check
    between them, and the child's own last words come back with the exit
    code.
    """
    import time as _time

    stub = tmp_path / "broken_worker.py"
    stub.write_text("\n".join([
        "import sys",
        'sys.stderr.write("CUDA driver version is insufficient")',
        "raise SystemExit(3)",
    ]), encoding="utf-8")
    monkeypatch.setattr(subproc, "WORKER", stub)
    monkeypatch.setattr(subproc, "START_TIMEOUT_S", 120.0)
    spec = dataclasses.replace(LTX_25, vram_gb=0.1)
    provider = SubprocessModelProvider(spec, interpreter=Path(sys.executable))
    started = _time.monotonic()
    with pytest.raises(Exception) as exc:
        provider.generate(prompt="x", seconds=0.4, fps=24,
                          out_path=tmp_path / "a.mp4")
    elapsed = _time.monotonic() - started
    assert elapsed < 30, f"took {elapsed:.0f}s to notice a dead worker"
    assert "exit=3" in str(exc.value), str(exc.value)
    assert "CUDA driver version is insufficient" in str(exc.value), (
        "the child's own stderr is the only diagnosis the parent can "
        f"offer: {exc.value}")


# ------------------------------------------------------------ timeout

def test_the_timeout_moves_with_the_work_not_with_a_constant():
    """A constant timeout is right for exactly one schedule.

    Two stood here — 900 s for the first call, 600 s after — sized
    against 8 steps at 512x896. Raising quality to 30 steps at the
    model's own envelope made one six-second shot a ~26-minute job, and
    the 900 s cap killed the worker at 15:30 with the render still
    running: a good render reported as "did not answer".
    """
    from clipforge.genvideo.subproc import MIN_CALL_TIMEOUT_S, call_timeout_s

    small = call_timeout_s(49, 512, 896, 8, first_call=False)
    steppier = call_timeout_s(49, 512, 896, 30, first_call=False)
    bigger = call_timeout_s(49, 704, 1280, 30, first_call=False)
    longer = call_timeout_s(145, 704, 1280, 30, first_call=False)
    assert small < steppier < bigger < longer, (
        "the budget must grow with steps, with pixels and with frames")

    # The shot that actually died: measured ~26 min, and the budget has
    # to clear it with room rather than by a hair.
    assert longer > 26 * 60 * 1.2

    # A first call additionally pays the load.
    assert (call_timeout_s(49, 512, 896, 30, first_call=True)
            > steppier)

    # And a trivial call still gets a floor, so a stub worker on a busy
    # machine is not raced.
    assert call_timeout_s(1, 64, 64, 1, first_call=False) == MIN_CALL_TIMEOUT_S


# ---------------------------------------------------- dropped controls

def test_loras_this_worker_cannot_apply_are_reported(caplog):
    """A customisation that vanishes is worse than one that is declined.

    `LocalDiffusersProvider._apply_loras` logs `lora_unsupported` when a
    pipeline cannot take them. The subprocess path took `spec` and `seed`
    and nothing else, so `[genvideo] loras` configured against ltx25 went
    nowhere with no line in the log.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        SubprocessModelProvider(LTX_25, loras=["a.safetensors"],
                                step_cache_threshold=0.3)
    assert "lora_unsupported" in caplog.text
    assert "step_cache_unsupported" in caplog.text


def test_a_model_with_no_extra_controls_is_quiet(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        SubprocessModelProvider(LTX_25)
    assert "unsupported" not in caplog.text


# ------------------------------------------------------- the envelope

def test_every_registry_model_generates_inside_its_own_envelope():
    """The size that renders must be legal for the spec that authorised it.

    Selection scores models against the size the GLOBAL cap produces and
    the provider generates at the model's own envelope, so the size that
    actually renders was never checked against the spec. It is legal by
    construction — the budget and the grid both come from the spec — and
    this pins that, because asking past a model's envelope returns blank
    frames rather than an error.
    """
    from clipforge.genvideo.models import REGISTRY
    from clipforge.genvideo.providers import _generation_dims

    for key, spec in REGISTRY.items():
        for aspect in ("9:16", "16:9", "3:4", "4:5", "1:1"):
            w, h = _generation_dims(aspect, budget=spec.max_pixels,
                                    multiple=spec.dim_multiple)
            assert spec.supports(w, h), (
                f"{key} would generate {w}x{h} at {aspect}, which its own "
                f"envelope rejects")
