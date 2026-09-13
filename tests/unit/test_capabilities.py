"""capabilities.py — the probe must report reality, not assumptions.

The module's own docstring states the contract: every tile is backed by a
live test, and a missing capability names its fix. The highest-value case
here is the dubbing tile's tri-state, because collapsing "config
unreadable" into "off by decision" is the exact fault-dressed-as-choice
the module documents having shipped once already.
"""

from __future__ import annotations

import subprocess
import types

import pytest

from clipforge import capabilities
from clipforge.capabilities import Capability


# ---------------------------------------------------------- filter probing

@pytest.fixture(autouse=True)
def _clear_filter_cache():
    """The filter list is lru_cached per process; isolate every test."""
    capabilities._ffmpeg_filters.cache_clear()
    yield
    capabilities._ffmpeg_filters.cache_clear()


_FILTERS_STDOUT = """Filters:
  T.. = Timeline support
 ... anull            A->A       Pass the source unchanged.
 TS. cas              V->V       Contrast Adaptive Sharpen.
 ..C libplacebo       N->V       Render using libplacebo.
 T.. overlay          VV->V      Overlay a video source on top of the input.
 this line has no arrow and must be ignored
"""


def test_filter_list_parsed_from_real_ffmpeg_output(monkeypatch):
    import clipforge.ffmpeg as ff

    monkeypatch.setattr(ff, "require_binary", lambda name: "ffmpeg")
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return types.SimpleNamespace(stdout=_FILTERS_STDOUT, returncode=0)

    monkeypatch.setattr(capabilities.subprocess, "run", fake_run)

    assert capabilities.has_filter("cas")
    assert capabilities.has_filter("libplacebo")
    assert capabilities.has_filter("overlay")
    # the names come from the -filters listing, not some other probe
    assert "-filters" in calls[0]
    assert not capabilities.has_filter("flite")
    # header, legend and arrowless lines never become filter names
    assert not capabilities.has_filter("Filters:")
    assert not capabilities.has_filter("this")
    # cached: one subprocess for any number of lookups
    assert len(calls) == 1


def test_filter_probe_failure_reports_nothing_available(monkeypatch):
    """A broken ffmpeg yields an EMPTY set — never a hardcoded 'yes'."""
    import clipforge.ffmpeg as ff

    def boom(name):
        raise FileNotFoundError("no ffmpeg on PATH")

    monkeypatch.setattr(ff, "require_binary", boom)
    assert capabilities._ffmpeg_filters() == frozenset()
    assert not capabilities.has_filter("overlay")


def test_filter_probe_tolerates_empty_stdout(monkeypatch):
    import clipforge.ffmpeg as ff

    monkeypatch.setattr(ff, "require_binary", lambda name: "ffmpeg")
    monkeypatch.setattr(
        capabilities.subprocess, "run",
        lambda cmd, **kw: types.SimpleNamespace(stdout=None, returncode=1))
    assert capabilities._ffmpeg_filters() == frozenset()


# ------------------------------------------------------------- probe wiring

class _Cfg:
    """Opaque config sentinel; the probe must pass it through untouched."""


def _patch_probe_world(monkeypatch, *, filters=(), kokoro=False,
                       local_model=False, asr=True,
                       load_config=None, cloud_enabled=None,
                       has_key=False, voices=None,
                       local_translator=True,
                       pose=False, face=False, gpu="", weights=False,
                       nvenc=False, hf_token=False, tools=(), judges=()):
    """Pin every external the probe consults to an explicit value.

    ``local_translator`` defaults True because that is the real state of
    any machine that can download: the local NLLB route needs no key and
    no permission, which is exactly why it makes the dubbing tile LIVE
    under the cloud-off decision. Tests that want a dark tile must now
    remove BOTH routes, which is the point of the feature.

    the tree: the S6 composite path is not written. It is a knob rather than
    a constant so the fully-equipped world can still assert that no tile is
    hardcoded dead.
    """
    import clipforge.cloud as cloud
    import clipforge.config as config
    import clipforge.dubbing as dubbing
    import clipforge.enhance as enhance
    import clipforge.translate as translate

    monkeypatch.setattr(translate, "local_translator_available",
                        lambda cfg: local_translator)

    fset = frozenset(filters)
    monkeypatch.setattr(capabilities, "has_filter", lambda n: n in fset)
    monkeypatch.setattr(enhance, "kokoro_available", lambda root=None: kokoro)
    monkeypatch.setattr(capabilities, "_has_module",
                        lambda n: asr and n in ("whisperx", "faster_whisper"))
    monkeypatch.setattr(
        config, "load_config",
        load_config if load_config else (lambda p: _Cfg()))
    monkeypatch.setattr(
        cloud, "cloud_enabled",
        cloud_enabled if cloud_enabled else (lambda cfg, feat: False))
    monkeypatch.setattr(cloud, "has_gemini_key", lambda: has_key)
    monkeypatch.setattr(dubbing, "installed_voice_languages",
                        lambda root=None: dict(voices or {}))

    # The pipeline tiles. Each probe is a module-level helper precisely so
    # it can be pinned here: two of them encode a frame or import torch for
    # real, and a unit test must not do either.
    monkeypatch.setattr(capabilities, "_pose_ready", lambda: (pose, face))
    monkeypatch.setattr(capabilities, "_cuda_device", lambda: gpu)
    monkeypatch.setattr(
        capabilities, "_ranking_weights",
        lambda: (weights, "Qwen2.5-VL-7B-Instruct-AWQ: 6.9 GB cached"
                 if weights else "Qwen2.5-VL-7B-Instruct-AWQ: 0.0 GB cached "
                 "(weights missing)"))
    monkeypatch.setattr(
        capabilities, "_nvenc",
        lambda: (True, "NVENC encodes", "") if nvenc else
        (False, "NVENC unusable: driver too old", "Update the NVIDIA driver."))
    monkeypatch.setattr(capabilities, "_hf_token_present", lambda: hf_token)
    monkeypatch.setattr(capabilities, "_tool", lambda name: name in tools)
    monkeypatch.setattr(capabilities, "_hosted_judges", lambda: tuple(judges))


def _tile(caps: list[Capability], key: str) -> Capability:
    match = [c for c in caps if c.key == key]
    assert len(match) == 1, f"expected exactly one {key!r} tile"
    return match[0]


def test_probe_reports_every_advertised_tile_exactly_once(monkeypatch):
    _patch_probe_world(monkeypatch)
    keys = [c.key for c in capabilities.probe()]
    assert keys == ["voiceover", "upscale", "broll", "splitscreen",
                    "dubbing", "publish", "speech",
                    "tracking", "ranking", "diarization", "gpu_encode",
                    "vl_judge", "grab", "live"]


def test_unavailable_tiles_name_their_fix(monkeypatch):
    """The contract: a missing capability says what would fix it."""
    _patch_probe_world(monkeypatch)
    for cap in capabilities.probe():
        if not cap.available:
            assert cap.blocker, f"{cap.key} is unavailable with no blocker"


def test_available_tiles_carry_no_blocker(monkeypatch):
    _patch_probe_world(
        monkeypatch, filters=("flite", "libplacebo", "cas", "overlay",
                              "afftdn", "speechnorm", "xstack"),
        kokoro=True, local_model=True,
        cloud_enabled=lambda cfg, feat: True, has_key=True,
        voices={"en": "af_heart"},
        pose=True, face=True, gpu="RTX 3090 (24 GB)", weights=True,
        nvenc=True, hf_token=True, tools=("yt-dlp", "streamlink"),
        judges=("anthropic",))
    for cap in capabilities.probe():
        if cap.available:
            assert cap.blocker == "", f"{cap.key} available yet blocked"


def test_fully_equipped_world_reports_every_implemented_tile_live(monkeypatch):
    """Installed dependencies light up every implemented feature.

    A capability explicitly marked ``by_policy`` is a truthful roadmap tile,
    not a missing dependency that a fully equipped machine should disguise as
    available.
    """
    _patch_probe_world(
        monkeypatch, filters=("flite", "libplacebo", "cas", "overlay",
                              "afftdn", "speechnorm", "xstack"),
        kokoro=True, local_model=True,
        cloud_enabled=lambda cfg, feat: True, has_key=True,
        voices={"en": "af_heart"},
        pose=True, face=True, gpu="RTX 3090 (24 GB)", weights=True,
        nvenc=True, hf_token=True, tools=("yt-dlp", "streamlink"),
        judges=("anthropic",))
    caps = capabilities.probe()
    for cap in caps:
        if not cap.by_policy:
            assert cap.available, f"{cap.key} dead in a fully equipped world"

    # The exemption above is a hole: any broken tile can now hide behind
    # by_policy=True and this guard will wave it through. Pin the exempt set
    # so widening it costs a deliberate edit here, in the same change - the
    # ratchet EXPECTED_GPU_TESTS uses for the same reason.
    exempt = {c.key for c in caps if c.by_policy}
    assert exempt == {"broll", "splitscreen"}, (
        f"a tile exempted itself from the equipped-world guard: {exempt}")


def test_roadmap_tiles_are_explicitly_blocked(monkeypatch):
    _patch_probe_world(monkeypatch)
    for key in ("broll", "splitscreen"):
        tile = _tile(capabilities.probe(), key)
        assert not tile.available
        assert tile.by_policy
        assert "removed" in tile.blocker




def test_speech_names_whichever_filter_is_missing(monkeypatch):
    """afftdn present, speechnorm absent: the blocker names speechnorm."""
    _patch_probe_world(monkeypatch, filters=("afftdn",))
    tile = _tile(capabilities.probe(), "speech")
    assert not tile.available
    assert "speechnorm" in tile.blocker



# ----------------------------------------------- dubbing tile tri-state

def test_dubbing_cloud_off_is_policy(monkeypatch):
    """Config read, flag really false → by_policy with the shared blocker."""
    _patch_probe_world(monkeypatch, cloud_enabled=lambda cfg, feat: False,
                       local_translator=False)
    tile = _tile(capabilities.probe(), "dubbing")
    assert not tile.available
    assert tile.by_policy is True
    assert "local translator cannot run" in tile.blocker


def test_dubbing_cloud_off_ignores_a_key_that_is_still_present(monkeypatch):
    """The flag decides, not the credential — and this is a live state.

    Cloud was switched off by operator decision on 2026-08-05, but the
    key it used may still sit in ``.env``. Every other cloud-off test here
    also leaves ``has_key`` False, so dropping ``cloud_on`` from the
    translator expression keeps them all green while the tile lights up
    LIVE on any machine that kept its key — the keyed/unkeyed split that
    only shows in production, which is the shape the chokepoint round
    already closed once in ``cloud.py``.
    """
    _patch_probe_world(monkeypatch, cloud_enabled=lambda cfg, feat: False,
                       has_key=True, voices={"en": "af_heart"},
                       local_translator=False)
    tile = _tile(capabilities.probe(), "dubbing")
    assert not tile.available, "a present key must not re-enable dubbing"
    assert tile.by_policy is True
    assert "local translator cannot run" in tile.blocker


def test_dubbing_is_live_on_the_local_route_under_cloud_off(monkeypatch):
    """The headline behaviour of the local translator, and the one thing
    no other test here asserts.

    Cloud off, no key, and the tile is still LIVE — because the local
    route needs neither. `by_policy` stays True, but it now records that
    the ENGINE was chosen by decision, not that the feature is gone. A
    mutation sweep found this missing: making the probe ignore the local
    route entirely left all twenty capability tests green.
    """
    _patch_probe_world(monkeypatch, cloud_enabled=lambda cfg, feat: False,
                       has_key=False, local_translator=True,
                       voices={"en": "af_heart"})
    tile = _tile(capabilities.probe(), "dubbing")
    assert tile.available, "the local route must keep dubbing alive"
    assert tile.blocker == ""
    assert "local NLLB-200" in tile.note  # and it names which engine ran
    assert tile.by_policy is True


def test_dubbing_unreadable_config_is_a_fault_not_a_decision(monkeypatch):
    """load_config raising must NOT be reported as the cloud-off choice."""

    def broken(path):
        raise ValueError("corrupt toml")

    _patch_probe_world(monkeypatch, load_config=broken)
    tile = _tile(capabilities.probe(), "dubbing")
    assert not tile.available
    assert tile.by_policy is False  # the regression this module documents
    assert "could not be read" in tile.blocker


def test_dubbing_cloud_on_but_keyless_is_not_policy(monkeypatch):
    _patch_probe_world(monkeypatch, cloud_enabled=lambda cfg, feat: True,
                       has_key=False, local_translator=False)
    tile = _tile(capabilities.probe(), "dubbing")
    assert not tile.available
    assert tile.by_policy is False
    assert "no usable Gemini key" in tile.blocker
    assert "off by decision" not in tile.blocker


def test_dubbing_no_asr_outranks_the_cloud_question(monkeypatch):
    """No ASR means the tile is DARK, not merely differently worded.

    ASR used to sit first in the blocker chain while `available` ignored
    it — harmless only because a missing cloud key made the tile dark
    anyway. When the local translator landed, a translator became almost
    always present, this branch became unreachable, and a machine with no
    ASR advertised dubbing as LIVE. The availability assertion is the part
    that catches that; the blocker string alone did not.
    """
    _patch_probe_world(monkeypatch, asr=False,
                       cloud_enabled=lambda cfg, feat: False,
                       local_translator=True, has_key=True)
    tile = _tile(capabilities.probe(), "dubbing")
    assert not tile.available, "no ASR cannot report a live dubbing tile"
    assert tile.blocker == "no ASR installed"
    assert tile.by_policy is False  # policy needs ASR present AND flag false


def test_dubbing_available_lists_installed_voice_languages(monkeypatch):
    _patch_probe_world(monkeypatch,
                       cloud_enabled=lambda cfg, feat: True, has_key=True,
                       voices={"es": "ef_dora", "en": "af_heart"})
    tile = _tile(capabilities.probe(), "dubbing")
    assert tile.available
    assert tile.blocker == ""
    assert "en, es" in tile.note  # sorted, so the note is deterministic


def test_dubbing_translator_without_voices_is_subtitles_only(monkeypatch):
    _patch_probe_world(monkeypatch,
                       cloud_enabled=lambda cfg, feat: True, has_key=True,
                       voices={})
    tile = _tile(capabilities.probe(), "dubbing")
    assert tile.available  # subtitles half stands on its own
    assert "no Kokoro voice installed" in tile.blocker


# ------------------------------------------------------------------ summary

def test_summary_is_the_dashboard_shape(monkeypatch):
    _patch_probe_world(monkeypatch)
    rows = capabilities.summary()
    assert rows and all(
        set(r) == {"key", "label", "available", "blocker", "note",
                   "by_policy"} for r in rows)
    # JSON-serializable end to end — this is what web.py ships verbatim
    import json

    json.dumps(rows)


# --- the pipeline tiles ----------------------------------------------------

def test_reframe_is_live_whenever_the_pose_weights_are_present(monkeypatch):
    """The regression: the dashboard's Reframe tool gated on cap:'tracking'
    and this probe never emitted the key, so capOk() dimmed it everywhere."""
    _patch_probe_world(monkeypatch, pose=True)
    tile = _tile(capabilities.probe(), "tracking")
    assert tile.available and tile.blocker == ""


def test_reframe_without_the_face_model_says_how_it_chooses(monkeypatch):
    _patch_probe_world(monkeypatch, pose=True, face=False)
    note = _tile(capabilities.probe(), "tracking").note
    assert "screen presence" in note and "face_landmarker" in note


def test_ranking_needs_the_weights_and_a_gpu_and_names_which_is_missing(monkeypatch):
    _patch_probe_world(monkeypatch, weights=False, gpu="RTX")
    assert "weights missing" in _tile(capabilities.probe(), "ranking").blocker
    _patch_probe_world(monkeypatch, weights=True, gpu="")
    assert "CUDA" in _tile(capabilities.probe(), "ranking").blocker


def test_a_dead_nvenc_is_reported_with_the_software_fallback(monkeypatch):
    """Not a dead end: renders still work, and the tile must say so."""
    _patch_probe_world(monkeypatch, nvenc=False)
    tile = _tile(capabilities.probe(), "gpu_encode")
    assert not tile.available
    assert "driver" in tile.blocker
    assert "libx264" in tile.note


def test_speaker_labels_explain_that_clips_still_render_without_a_token(monkeypatch):
    _patch_probe_world(monkeypatch, hf_token=False)
    blocker = _tile(capabilities.probe(), "diarization").blocker
    assert "CLIPFORGE_HF_TOKEN" in blocker and "still render" in blocker


def test_the_judge_names_the_local_link_when_the_cloud_is_off(monkeypatch):
    _patch_probe_world(monkeypatch, weights=True, gpu="RTX",
                       judges=("gemini",),
                       cloud_enabled=lambda cfg, feat: False)
    tile = _tile(capabilities.probe(), "vl_judge")
    assert tile.available
    assert "local Qwen" in tile.note
    assert "Gemini has a key but [s7] use_cloud is off" in tile.note


def test_the_judge_names_the_first_hosted_link_when_authorised(monkeypatch):
    _patch_probe_world(monkeypatch, weights=True, gpu="RTX",
                       judges=("gemini", "anthropic"),
                       cloud_enabled=lambda cfg, feat: True)
    assert _tile(capabilities.probe(), "vl_judge").note == "Claude answers first"


def test_a_hosted_key_without_authorisation_does_not_light_the_judge(monkeypatch):
    """A key is possibility, not permission."""
    _patch_probe_world(monkeypatch, weights=False, gpu="",
                       judges=("anthropic",),
                       cloud_enabled=lambda cfg, feat: False)
    assert not _tile(capabilities.probe(), "vl_judge").available


def test_sources_report_their_own_tool(monkeypatch):
    _patch_probe_world(monkeypatch, tools=("yt-dlp",))
    caps = capabilities.probe()
    assert _tile(caps, "grab").available
    assert not _tile(caps, "live").available
    assert "streamlink" in _tile(caps, "live").blocker


def test_the_expensive_probes_are_cached_for_the_life_of_the_process():
    """NVENC encodes a frame and CUDA imports torch; a page load must not
    pay for either twice."""
    assert hasattr(capabilities._nvenc, "cache_info")
    assert hasattr(capabilities._cuda_device, "cache_info")

