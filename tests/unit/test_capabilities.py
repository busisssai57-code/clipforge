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
                       local_translator=True, split_screen_render=False):
    """Pin every external the probe consults to an explicit value.

    ``local_translator`` defaults True because that is the real state of
    any machine that can download: the local NLLB route needs no key and
    no permission, which is exactly why it makes the dubbing tile LIVE
    under the cloud-off decision. Tests that want a dark tile must now
    remove BOTH routes, which is the point of the feature.

    ``split_screen_render`` defaults False because that is the real state of
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
    monkeypatch.setattr(capabilities, "_renders_split_screen",
                        lambda: split_screen_render)

    fset = frozenset(filters)
    monkeypatch.setattr(capabilities, "has_filter", lambda n: n in fset)
    monkeypatch.setattr(enhance, "kokoro_available", lambda root=None: kokoro)
    monkeypatch.setattr(capabilities, "_local_model_installed",
                        lambda: local_model)
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


def _tile(caps: list[Capability], key: str) -> Capability:
    match = [c for c in caps if c.key == key]
    assert len(match) == 1, f"expected exactly one {key!r} tile"
    return match[0]


def test_probe_reports_every_advertised_tile_exactly_once(monkeypatch):
    _patch_probe_world(monkeypatch)
    keys = [c.key for c in capabilities.probe()]
    assert keys == ["voiceover", "upscale", "broll", "dubbing",
                    "publish", "speech", "splitscreen"]


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
        voices={"en": "af_heart"}, split_screen_render=True)
    for cap in capabilities.probe():
        if cap.available:
            assert cap.blocker == "", f"{cap.key} available yet blocked"


def test_fully_equipped_world_reports_every_tile_live(monkeypatch):
    """With everything installed, no tile may be hardcoded dead."""
    _patch_probe_world(
        monkeypatch, filters=("flite", "libplacebo", "cas", "overlay",
                              "afftdn", "speechnorm", "xstack"),
        kokoro=True, local_model=True,
        cloud_enabled=lambda cfg, feat: True, has_key=True,
        voices={"en": "af_heart"}, split_screen_render=True)
    for cap in capabilities.probe():
        assert cap.available, f"{cap.key} dead in a fully equipped world"


def test_broll_weights_without_overlay_names_the_filter(monkeypatch):
    """Weights present, overlay filter absent: unavailable, and the
    blocker must name the overlay filter rather than stay empty."""
    _patch_probe_world(monkeypatch, local_model=True, filters=())
    tile = _tile(capabilities.probe(), "broll")
    assert not tile.available
    assert tile.blocker and "overlay" in tile.blocker


def test_speech_names_whichever_filter_is_missing(monkeypatch):
    """afftdn present, speechnorm absent: the blocker names speechnorm."""
    _patch_probe_world(monkeypatch, filters=("afftdn",))
    tile = _tile(capabilities.probe(), "speech")
    assert not tile.available
    assert "speechnorm" in tile.blocker


# ------------------------------------------------------------ split screen
#
# This tile was reported as `has_filter("xstack")`, which is a fact about
# ffmpeg rather than about BTA. S4 computes the crops correctly, but nothing
# in the render path composites them, so the tile read LIVE on every ordinary
# build — on the dashboard and on splitscreen.html — for a feature that does
# not exist. These tests pin the report to the render path instead.

def test_split_screen_dark_when_the_render_path_is_missing(monkeypatch):
    """xstack present and the path absent is the exact shipped bug."""
    _patch_probe_world(monkeypatch, filters=("xstack",),
                       split_screen_render=False)
    tile = _tile(capabilities.probe(), "splitscreen")
    assert not tile.available
    assert "not implemented" in tile.blocker
    # Naming the filter here would send someone to fix ffmpeg for a path
    # nobody has written.
    assert "xstack missing" not in tile.blocker


def test_split_screen_names_the_filter_once_the_path_exists(monkeypatch):
    """Path written, filter absent: now xstack IS the thing to fix."""
    _patch_probe_world(monkeypatch, filters=(), split_screen_render=True)
    tile = _tile(capabilities.probe(), "splitscreen")
    assert not tile.available
    assert tile.blocker == "xstack missing"


def test_split_screen_live_only_when_both_hold(monkeypatch):
    _patch_probe_world(monkeypatch, filters=("xstack",),
                       split_screen_render=True)
    tile = _tile(capabilities.probe(), "splitscreen")
    assert tile.available
    assert tile.blocker == ""


def test_split_screen_probe_is_live_not_a_hardcoded_false(monkeypatch):
    """The probe must flip on its own when S6 grows the entry point.

    A hardcoded False would pass every test above and then quietly under-report
    the feature forever once somebody built it.
    """
    from clipforge.stages import s6_render

    assert not capabilities._renders_split_screen()
    monkeypatch.setattr(s6_render, "build_split_screen_filter",
                        lambda *a, **k: "", raising=False)
    assert capabilities._renders_split_screen()


def test_split_screen_tile_cannot_outrun_its_caller():
    """Structural: no production caller means the tile may not claim a path.

    Deliberately reads the tree rather than a fixture. `compute_split_screen_crops`
    sat callable and tested, with its only caller a unit test, while the tile
    advertised it — so the invariant worth enforcing is the module->caller
    edge itself, not any particular spelling of the probe.
    """
    from pathlib import Path

    import clipforge

    pkg = Path(clipforge.__file__).resolve().parent
    callers = sorted(
        p.name for p in pkg.rglob("*.py")
        if p.name != "s4_tracking.py"
        and "compute_split_screen_crops(" in p.read_text(encoding="utf-8")
    )
    if not callers:
        assert not capabilities._renders_split_screen(), (
            "nothing in clipforge/ calls compute_split_screen_crops, yet the "
            "render probe claims a split-screen path exists")


def test_voiceover_flite_only_names_the_kokoro_upgrade(monkeypatch):
    _patch_probe_world(monkeypatch, filters=("flite",), kokoro=False)
    tile = _tile(capabilities.probe(), "voiceover")
    assert tile.available
    assert "Kokoro" in tile.note  # upgrade path named, not implied


def test_voiceover_with_no_tts_at_all_goes_dark(monkeypatch):
    """The module's one rule, tested at the only point it can fail.

    Every other voiceover test above runs in a world with flite, Kokoro or
    both, so all of them stay green if ``available`` is hardcoded True —
    including the fully-equipped sweep, which asserts availability. The
    empty world is the case that distinguishes a probe from a constant.
    """
    _patch_probe_world(monkeypatch, filters=(), kokoro=False)
    tile = _tile(capabilities.probe(), "voiceover")
    assert not tile.available
    # and it names BOTH fixes, per the docstring's "in the same breath"
    assert "libflite" in tile.blocker and "Kokoro" in tile.blocker


def test_upscale_never_claims_super_resolution(monkeypatch):
    _patch_probe_world(monkeypatch, filters=("libplacebo", "cas"))
    tile = _tile(capabilities.probe(), "upscale")
    assert tile.available
    assert "RESAMPLING" in tile.note  # honest-quality note survives


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
