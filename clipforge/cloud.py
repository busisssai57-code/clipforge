"""The §2 chokepoint: hosted inference runs on ONE credential, and only
this module may read it.

Spec §2 is a hard contract: *"Cloud: **None.** No hosted inference."*
Until 2026-08-05 the law was enforced by per-feature leaf checks —
``[s3] use_cloud`` in two places, and a
fifth re-derivation in the capability probe — and that pattern failed in
exactly the predictable way: genvideo was pinned off-by-default on
2026-07-31, and the s3 flag then shipped **True for a day** (2026-08-04)
because a new feature bringing its own flag is precisely what nobody
audits.

**The chokepoint is the KEY, not the network.** Ingestion (yt-dlp,
streamlink) and model-weight downloads legitimately use the network, so a
socket-level guard would need an allowlist and would rot. Hosted
inference, by contrast, cannot happen without the Gemini credential. So:

* every cloud-capable feature is REGISTERED here, mapped to the config
  flag that authorizes it — an unregistered name raises, so the next
  feature must add a row in the file whose docstring is this law;
* the only code that reads ``Secrets().gemini_api_key`` is
  :func:`gemini_key` below, and it returns None unless the feature's flag
  says yes;
* a structural test (``tests/unit/test_cloud_chokepoint.py``) sweeps the
  package and fails on any other read of the credential — including the
  raw environment variable — so the N+1th feature cannot quietly bring
  its own key read.

Probes that only need "is a key present at all" use
:func:`has_gemini_key`, which never returns the credential and opens no
gate.
"""

from __future__ import annotations

from typing import Any, Callable

from clipforge.log import get_logger

log = get_logger(__name__)


class UnknownCloudFeature(KeyError):
    """A cloud feature that was never registered asked for the gate.

    Raising (rather than defaulting to disabled) is deliberate: a typo'd
    or unregistered feature name should fail the first test that touches
    it, not silently read as "cloud off" until someone wires it for real.
    """


#: feature -> the config flag that authorizes it. getattr-chained so a
#: partial cfg (tests, reconstructed shims) reads as DISABLED rather than
#: crashing — absence of a section can never mean permission.
_FEATURES: dict[str, Callable[[Any], bool]] = {
    # S3 ranking sends candidate frames + transcript text to Gemini.
    "s3_ranking": lambda cfg: bool(
        getattr(getattr(cfg, "s3", None), "use_cloud", False)),
    # Translation rides the ranker's flag and credential: same model,
    # same payload class (transcript text), same decision.
    "translation": lambda cfg: bool(
        getattr(getattr(cfg, "s3", None), "use_cloud", False)),
    # Generation sends brief/prompt text to Veo.
}


def registered_features() -> tuple[str, ...]:
    return tuple(sorted(_FEATURES))


def cloud_enabled(cfg: Any, feature: str) -> bool:
    """Does the operator's config authorize this feature to use the cloud?"""
    try:
        flag = _FEATURES[feature]
    except KeyError:
        raise UnknownCloudFeature(
            f"cloud feature {feature!r} is not registered; add it to "
            f"clipforge.cloud._FEATURES (known: {registered_features()}). "
            "Registration is the point — an unregistered flag is how the "
            "s3 ranker shipped cloud-on for a day.") from None
    return flag(cfg)


#: Features already logged as granted this process — the grant line is the
#: audit signal, and one per feature per process keeps it a signal.
_granted: set[str] = set()


def gemini_key(cfg: Any = None, *, feature: str,
               enabled: bool | None = None) -> str | None:
    """THE credential read. Returns the key iff the feature is authorized.

    Pass ``cfg`` and the flag is looked up in the registry; pass
    ``enabled`` explicitly when the caller's authority is a stage-params
    copy of the flag (S3's cache-key params) rather than a live config.
    Either way an unknown ``feature`` raises — see UnknownCloudFeature.

    The Secrets import happens at call time, on the enabled path only, so
    the disabled path provably never touches the environment (pinned by a
    call-recording test — an exception-based pin would be swallowed by
    callers' own except blocks, test-harness trap #11).
    """
    if feature not in _FEATURES:
        raise UnknownCloudFeature(
            f"cloud feature {feature!r} is not registered; add it to "
            f"clipforge.cloud._FEATURES (known: {registered_features()})")
    if enabled is None:
        if cfg is None:
            raise TypeError("gemini_key needs cfg or an explicit enabled=")
        enabled = cloud_enabled(cfg, feature)
    if not enabled:
        return None

    try:
        from clipforge.config import Secrets  # noqa: PLC0415 - call-time, see docstring
        key = Secrets().gemini_api_key
    except Exception as exc:  # noqa: BLE001 - unreadable .env is "no key", loudly
        log.warning("cloud.secrets_unreadable", feature=feature,
                    error=str(exc)[:200])
        return None
    if not key:
        return None
    if feature not in _granted:
        _granted.add(feature)
        log.warning("cloud.key_granted", feature=feature,
                    note="hosted-inference credential handed out "
                         "(flag-authorized); content may leave this machine")
    return key


def has_gemini_key() -> bool:
    """Probe-only: is a credential present at all.

    Never returns the key and opens no gate — for capability tiles that
    report "you could enable this", not for code that would send.
    """
    try:
        from clipforge.config import Secrets  # noqa: PLC0415

        return bool(Secrets().gemini_api_key)
    except Exception:  # noqa: BLE001 - absence is the answer
        return False
