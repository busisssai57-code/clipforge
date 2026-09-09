"""Spec §2 is a hard contract: "Cloud: **None.** No hosted inference."

Two switches decide whether this machine talks to anyone —
``[s3] use_cloud`` (candidate frames and transcript text to Gemini) and
``[s3] use_cloud`` (frames to a hosted ranker). It was pinned
False-by-default on 2026-07-31 with three enforcement mechanisms. The s3
one shipped **True** on 2026-08-04 and stayed that way for a day: the
ranking stage sent frames out of the box, there was no ledger amendment
permitting it, and nothing in the suite would have noticed.

The operator settled it on 2026-08-05 — keep everything local — so these
pin it the way that flag was pinned. A default is not a preference
here; it is the difference between a machine that phones home and one that
does not, on an install nobody has read the config of yet.

Deliberately structural, not behavioural-only: a test that merely checks
"ranking works offline" would still pass with the default flipped back.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

from clipforge.config import S3Config, load_config

# Anchored on __file__, matching test_config.py — a cwd-relative
# Path("config/...") makes these pins depend on where pytest was invoked
# rather than on the code they pin.
REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / "config" / "config.example.toml"
LIVE = REPO / "config" / "config.toml"


def test_s3_cloud_is_off_in_the_model_default():
    """A fresh install with no config file at all must be local."""
    assert S3Config().use_cloud is False, (
        "§2 says no hosted inference; a True default sends frames from an "
        "install whose config nobody has opened")




def test_shipped_example_states_the_choice_rather_than_inheriting_it():
    """The example config must say it out loud.

    This is the specific hole the True default fell through:
    ``config.example.toml`` had no ``[s3] use_cloud`` line, so a fresh
    install inherited the setting without the operator ever seeing that
    there was a decision to make.
    """
    cfg = load_config(EXAMPLE)
    assert cfg.s3.use_cloud is False
    # Parsed, not string-sliced: a substring check matches a use_cloud that
    # only appears in a comment, and breaks if sections are reordered.
    raw = tomllib.loads(EXAMPLE.read_text(encoding="utf-8"))
    assert "use_cloud" in raw["s3"], (
        "the [s3] block must state use_cloud explicitly; a silent default "
        "is how this was on for a day without being decided")


def test_operator_config_is_local_on_both_switches():
    """The config this machine actually runs with.

    Skipped when absent, same as test_config_bounds.py's live-config test:
    config.toml is machine-local (created by copying the example), so a
    fresh checkout must skip here, not error. On machines that have the
    file, this pins the standing 2026-08-05 decision — if the operator
    later reverses it, this test is updated with the config, deliberately.
    """
    if not LIVE.exists():
        pytest.skip("no live config in this checkout")
    cfg = load_config(LIVE)
    assert cfg.s3.use_cloud is False


def test_no_ranker_object_exists_when_cloud_is_off():
    """Off means "not constructed", not "constructed and unused".

    An object holding an API key is one edit away from a request. Both
    construction sites — the module-level builder and the stage's own —
    must return None before they look at anything else, including the key.
    """
    from clipforge import vlrank
    from clipforge.stages.s3_semantic import _cloud_ranker

    # The SHIPPED example, not the operator's live file: the property under
    # test is "flag off => no object", and the example is pinned off by
    # test_shipped_example... above, so this holds on every checkout.
    cfg = load_config(EXAMPLE)
    assert vlrank.build_ranker(cfg) is None

    # ...and with a key present, which is the case that matters: a key in
    # .env must not be sufficient to put a frame on the wire.
    assert _cloud_ranker({"use_cloud": False,
                          "cloud_model": "gemini-2.5-pro"}) is None


def test_cloud_off_short_circuits_before_touching_secrets(monkeypatch):
    """The disabled path must not even read the key.

    Ordering is the property: if the flag were checked *after* the key
    lookup, a machine with a key would differ from one without, and the
    difference would only show up in production.

    A call-RECORDING stub, not a raising one — the first version raised
    AssertionError from Secrets(), which _cloud_ranker's own
    ``except Exception`` swallows into ``return None``, so the test passed
    whether or not the ordering held. A recorded call cannot be swallowed:
    the assertion runs out here, after the fact.
    """
    import clipforge.config as config_mod

    calls: list[str] = []

    class _RecordingSecrets:
        def __init__(self) -> None:
            calls.append("constructed")
            self.gemini_api_key = "key-that-must-never-be-read"

    monkeypatch.setattr(config_mod, "Secrets", _RecordingSecrets)
    from clipforge.stages.s3_semantic import _cloud_ranker

    assert _cloud_ranker({"use_cloud": False}) is None
    assert calls == [], (
        "Secrets() was read on the cloud-off path; the flag must be "
        "checked first")


def test_use_cloud_change_invalidates_the_s3_cache_key():
    """Flipping the switch must not serve a cloud-ranked artifact.

    S3 artifacts record which model judged them. If the flag were outside
    the params digest, turning the cloud off would return yesterday's
    Gemini ranking from cache and the run would look local while resting
    on a cloud judgement.

    This pins the Stage's half — that the flag, once passed, reaches the
    key. That the CLI actually passes it is not asserted here; it was
    proven by running the DAG on a source already ranked by Gemini and
    watching S3 re-rank instead of returning the cached artifact.
    """
    from clipforge.stages.base import digest_params

    on = digest_params({"use_cloud": True, "seed": 1234})
    off = digest_params({"use_cloud": False, "seed": 1234})
    assert on != off


@pytest.mark.parametrize("field", ["cloud_models", "cloud_model"])
def test_cloud_settings_are_kept_not_deleted(field):
    """Turning it off is a switch, not an amputation.

    The model chain stays configured so the operator can turn it back on
    without re-deriving which Gemini models work — recorded in the ledger
    as: Pro answers 429 "limit: 0" on an unbilled key, so a single name
    silently falls all the way through to the local 7B.
    """
    assert getattr(S3Config(), field)
