"""The §2 chokepoint, pinned structurally.

The law ("Cloud: None. No hosted inference") used to live in per-feature
leaf checks, and failed the way distributed invariants fail: the s3 flag
shipped True for a day because nothing audited a new feature's own flag.
The chokepoint moves the invariant to the CREDENTIAL — only
clipforge.cloud may read the Gemini key — and these tests are what stop
the N+1th feature from quietly bringing its own key read.

Two kinds of pin here, deliberately:

* **sweeps** over the package source (a new offending file fails even if
  its author never heard of this test), and
* **behavioural** checks on the gate itself, using call-recording stubs
  rather than raising ones — an exception-based pin gets swallowed by the
  very ``except Exception`` blocks this codebase (correctly) wraps
  secrets reads in (test-harness trap #11, VERIFICATION.md 2026-08-05).
"""

from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace as NS

import pydantic
import pytest

import clipforge
from clipforge import cloud
from clipforge.cloud import (UnknownCloudFeature, cloud_enabled, gemini_key,
                             has_gemini_key, registered_features)

PKG = Path(clipforge.__file__).resolve().parent

#: The only files allowed to spell the credential's attribute name in
#: code: the chokepoint itself and the model that defines the field.
_CREDENTIAL_READERS = {"cloud.py", "config.py"}


def _package_sources():
    for py in sorted(PKG.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        yield py.relative_to(PKG).as_posix(), py.read_text(encoding="utf-8")


# ------------------------------------------------------------------ sweeps

def test_the_credential_is_read_only_at_the_chokepoint():
    """No module outside the chokepoint may touch ``gemini_api_key``.

    The lowercase attribute name is the tell for CODE (``Secrets()
    .gemini_api_key``, ``cfg.secrets.gemini_api_key``); operator-facing
    messages say CLIPFORGE_GEMINI_API_KEY (uppercase) and stay legal.
    """
    offenders = [rel for rel, text in _package_sources()
                 if rel not in _CREDENTIAL_READERS
                 and "gemini_api_key" in text]
    assert offenders == [], (
        f"{offenders} read the hosted-inference credential directly; "
        "route through clipforge.cloud.gemini_key (the §2 chokepoint) — "
        "an inline read is how a feature ships cloud-on unaudited")


def test_the_env_var_is_never_read_directly_anywhere():
    """Not even the chokepoint reads the raw environment variable —
    Secrets (pydantic-settings) is the one env/.env reader, and the
    chokepoint is its one caller for this credential."""
    pattern = re.compile(
        r"environ(\.get)?\s*[\(\[]\s*['\"]CLIPFORGE_GEMINI_API_KEY")
    offenders = [rel for rel, text in _package_sources()
                 if pattern.search(text)]
    assert offenders == [], (
        f"{offenders} read CLIPFORGE_GEMINI_API_KEY straight from the "
        "environment, bypassing both Secrets and the chokepoint")


def test_every_use_cloud_flag_in_the_config_is_registered():
    """A new ``use_cloud`` flag on the config MUST have a registered
    feature, or flipping it changes nothing the chokepoint can see.

    This is the anti-regression for the exact 2026-08-04 failure: a
    feature section landing with its own flag that no invariant audits.
    Introspected from AppConfig so the test finds a future section
    without being edited.
    """
    from clipforge.config import AppConfig

    flagged_sections = []
    for name, field in AppConfig.model_fields.items():
        anno = field.annotation
        if (isinstance(anno, type) and issubclass(anno, pydantic.BaseModel)
                and "use_cloud" in anno.model_fields):
            flagged_sections.append(name)
    assert flagged_sections, "expected at least the s3 and genvideo flags"

    for section in flagged_sections:
        all_off = NS(**{s: NS(use_cloud=False) for s in flagged_sections})
        one_on = NS(**{s: NS(use_cloud=(s == section))
                       for s in flagged_sections})
        states_off = {f: cloud_enabled(all_off, f)
                      for f in registered_features()}
        states_on = {f: cloud_enabled(one_on, f)
                     for f in registered_features()}
        assert states_off != states_on, (
            f"config section {section!r} has a use_cloud flag that no "
            "registered cloud feature reads — the chokepoint cannot see "
            "it, which is how the s3 flag shipped True for a day")


# ------------------------------------------------------------- gate itself

def test_unregistered_feature_raises_not_defaults():
    with pytest.raises(UnknownCloudFeature):
        cloud_enabled(NS(), "brand_new_cloud_thing")
    with pytest.raises(UnknownCloudFeature):
        gemini_key(NS(), feature="brand_new_cloud_thing")


def test_disabled_feature_never_touches_secrets(monkeypatch):
    """Flag off => the environment is provably never read.

    Call-recording stub, not a raising one: the gate wraps its Secrets
    read in ``except Exception``, so a raising stub would be swallowed
    and the test would pass on the very regression it pins.
    """
    import clipforge.config as config_mod

    calls: list[str] = []

    class _Recorder:
        def __init__(self) -> None:
            calls.append("constructed")
            self.gemini_api_key = "key-that-must-never-flow"

    monkeypatch.setattr(config_mod, "Secrets", _Recorder)

    cfg = NS(s3=NS(use_cloud=False), genvideo=NS(use_cloud=False))
    assert gemini_key(cfg, feature="s3_ranking") is None
    assert gemini_key(cfg, feature="genvideo") is None
    assert gemini_key(feature="s3_ranking", enabled=False) is None
    assert calls == [], (
        "the disabled path read Secrets; the flag must gate the read, "
        "not just the return value")


def test_enabled_feature_gets_the_key_through_the_gate(monkeypatch):
    import clipforge.config as config_mod

    calls: list[str] = []

    class _Recorder:
        def __init__(self) -> None:
            calls.append("constructed")
            self.gemini_api_key = "k-granted"

    monkeypatch.setattr(config_mod, "Secrets", _Recorder)

    cfg = NS(s3=NS(use_cloud=True), genvideo=NS(use_cloud=False))
    assert gemini_key(cfg, feature="s3_ranking") == "k-granted"
    assert calls == ["constructed"]
    # ...and the flag still partitions features sharing the credential:
    assert gemini_key(cfg, feature="genvideo") is None


def test_absent_config_sections_read_as_disabled():
    """A partial cfg can never mean permission."""
    assert cloud_enabled(NS(), "s3_ranking") is False
    assert cloud_enabled(NS(), "genvideo") is False
    assert gemini_key(NS(), feature="translation") is None


def test_probe_helper_returns_bool_not_the_key(monkeypatch):
    import clipforge.config as config_mod

    class _WithKey:
        def __init__(self) -> None:
            self.gemini_api_key = "secret-value"

    monkeypatch.setattr(config_mod, "Secrets", _WithKey)
    result = has_gemini_key()
    assert result is True, "probe should see the key exists"
    assert result != "secret-value"


def test_translation_rides_the_ranker_flag():
    """One decision, one flag: translation uses the same model and the
    same payload class (transcript text) as ranking, so a config where
    they disagree must be unrepresentable."""
    for flag in (True, False):
        cfg = NS(s3=NS(use_cloud=flag))
        assert (cloud_enabled(cfg, "translation")
                == cloud_enabled(cfg, "s3_ranking"))
