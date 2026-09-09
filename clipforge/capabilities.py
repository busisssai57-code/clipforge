"""What this machine can actually do, probed rather than assumed.

Every feature here is reported by TESTING for it — an ffmpeg filter is
looked up in the live filter list, a model is checked for on disk, a
package is checked for importability. Nothing is hardcoded to "available".

That matters because the dashboard reads this. A tile that says LIVE when
the underlying thing is missing is how this project shipped blank videos
and dead knobs: the report and the reality drifted, and only the report
was ever looked at. A capability that is missing must say what would fix
it, in the same breath.
"""

from __future__ import annotations

import functools
import importlib.util
import subprocess
from dataclasses import dataclass
from pathlib import Path

from clipforge.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class Capability:
    key: str
    label: str
    available: bool
    #: What is missing, and what would fix it. Empty when available.
    blocker: str = ""
    #: Honest quality note — available does not mean good.
    note: str = ""
    #: True when the feature is deliberately not built rather than merely
    #: unconfigured. Keeps "we chose not to" distinct from "you can fix it".
    by_policy: bool = False


@functools.lru_cache(maxsize=1)
def _ffmpeg_filters() -> frozenset[str]:
    """Filter names this ffmpeg build actually exposes."""
    try:
        from clipforge.ffmpeg import require_binary

        proc = subprocess.run([str(require_binary("ffmpeg")), "-hide_banner",
                               "-filters"], capture_output=True, text=True,
                              errors="replace", timeout=60)
    except Exception as exc:  # noqa: BLE001
        log.warning("capabilities.ffmpeg_probe_failed", error=str(exc)[:200])
        return frozenset()
    names = set()
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        # " TS cas   V->V   Contrast Adaptive Sharpen." -> parts[1] is the name
        if len(parts) >= 3 and "->" in parts[2]:
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    return name in _ffmpeg_filters()


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _translator_blocker_text() -> str:
    """`bta dub`'s own refusal text, or a bare fallback if config is gone."""
    try:
        from clipforge.config import load_config
        from clipforge.dubbing import translator_blocker
        from pathlib import Path as _P

        return translator_blocker(load_config(_P("config/config.toml")))
    except Exception:  # noqa: BLE001
        return "no translator is available on this machine"


def probe() -> list[Capability]:
    """Every advertised feature, with its real state on this machine."""
    caps: list[Capability] = []

    # ---- voiceover -------------------------------------------------
    flite = has_filter("flite")
    try:
        from clipforge.enhance import kokoro_available

        kokoro = kokoro_available()
    except Exception:  # noqa: BLE001
        kokoro = False
    caps.append(Capability(
        key="voiceover", label="Voiceover",
        available=kokoro or flite,
        blocker="" if (kokoro or flite) else
            "no local TTS: this ffmpeg lacks libflite and Kokoro weights "
            "are not in workspace/models/kokoro",
        note=("Kokoro-82M, neural and natural — measured 3.8x realtime on "
              "CPU. Apache-2.0, runs entirely locally"
              if kokoro else
              "flite only: intelligible but plainly synthetic. Fetching "
              "Kokoro-82M (~330 MB) upgrades this to a neural voice"
              if flite else ""),
    ))

    # ---- upscaling -------------------------------------------------
    placebo = has_filter("libplacebo")
    cas = has_filter("cas")
    caps.append(Capability(
        key="upscale", label="Upscale",
        available=placebo or cas,
        blocker="" if (placebo or cas) else
            "no libplacebo and no cas filter in this ffmpeg build",
        note=("GPU resampling via libplacebo plus contrast-adaptive "
              "sharpening. This is high-quality RESAMPLING, not learned "
              "super-resolution — it will not invent detail that is not "
              "in the source" if placebo else
              "contrast-adaptive sharpening only; no GPU resampler"),
    ))

    # ---- b-roll ----------------------------------------------------
    # bta-site still ships broll.html and studio.html, both carrying
    # data-cap="broll", and dashboard_live.html still lists the tile. The
    # capability answer is the only thing that can contradict them, so
    # deleting the tile did not retire the feature - it left the claim
    # standing with nothing to argue against it.
    #
    # The gallery no longer recognises b-roll variants either: ".broll" came
    # out of clipmeta._SIDECAR_SUFFIXES when the generator went.
    caps.append(Capability(
        key="broll", label="AI B-roll",
        available=False,
        blocker="b-roll was removed with the generation half; nothing here renders an insert",
        note="The site still describes the local workflow. It is gone, not pending.",
        by_policy=True,
    ))

    # ---- split screen ----------------------------------------------
    # Tracking geometry exists, but nothing composites two speakers. xstack
    # being present in the ffmpeg build is not proof the product can make a
    # split-screen clip - deriving LIVE from an installed filter is the exact
    # mistake this module was written to stop.
    caps.append(Capability(
        key="splitscreen", label="Split screen",
        available=False,
        blocker="the split-screen render path was removed; no CLI or API composites two panes",
        note="Tracked speaker geometry survives, but nothing renders it.",
        by_policy=True,
    ))

    # ---- dubbing ---------------------------------------------------
    # Two halves with different requirements, and conflating them is how
    # this probe used to report a flat "no": translated SUBTITLES need a
    # translator, translated AUDIO additionally needs a voice that speaks
    # the target. The translator is the S3 cloud ranker; the voices are
    # whatever Kokoro files are installed.
    asr = _has_module("whisperx") or _has_module("faster_whisper")
    # Tri-state on purpose: None means the config could not be READ, which
    # is a fault, not a decision. The first version collapsed that into
    # cloud_on=False, so a missing or corrupt config.toml was reported as
    # the operator's dated cloud-off decision — with by_policy=True — which
    # is the exact fault-dressed-as-choice this module exists to prevent.
    cloud_on: bool | None = None
    cloud_translator = False
    local_translator = False
    engine = ""
    try:
        from clipforge.cloud import cloud_enabled, has_gemini_key
        from clipforge.config import load_config
        from clipforge.translate import local_translator_available
        from pathlib import Path as _P

        _cfg = load_config(_P("config/config.toml"))
        cloud_on = cloud_enabled(_cfg, "translation")
        # Ordering matters: cloud_on is learned BEFORE the key probe, so a
        # malformed .env leaves "cloud is on, key unreadable" intact
        # instead of clobbering it back to "off by decision".
        # has_gemini_key never returns the credential — the probe reports
        # possibility; only the chokepoint's gemini_key() hands out keys.
        cloud_translator = bool(cloud_on and has_gemini_key())
        # The local route needs no credential and no permission, which is
        # why it is what makes this tile LIVE under the cloud-off decision.
        local_translator = local_translator_available(_cfg)
    except Exception:  # noqa: BLE001 - absence is the answer, not a crash
        pass  # both stay False; cloud_on keeps whatever was learned
    # ASR gates AVAILABILITY, not just the wording. It used to sit first in
    # the blocker chain while `available` ignored it — harmless only because
    # a missing cloud key made the tile dark anyway. Now that the local
    # route almost always supplies a translator, that branch became
    # unreachable and a machine with no ASR would have advertised dubbing
    # as LIVE. Caught by the test that pinned the blocker's precedence.
    translator = (cloud_translator or local_translator) and asr
    if not asr:
        engine = ""
    elif cloud_translator:
        engine = "Gemini"
    elif local_translator:
        engine = "local NLLB-200"
    try:
        from clipforge.dubbing import installed_voice_languages

        dub_langs = sorted(installed_voice_languages())
    except Exception:  # noqa: BLE001
        dub_langs = []

    if translator and dub_langs:
        dub_note = (f"translated by {engine}; subtitles in any supported "
                    f"language; dubbed audio for {', '.join(dub_langs)} "
                    f"(the installed Kokoro voices)")
        dub_block = ""
    elif translator:
        dub_note = f"translated by {engine}; subtitles in any supported language"
        dub_block = ("no Kokoro voice installed, so audio cannot be dubbed. "
                     "Add a voice to workspace/models/kokoro/voices")
    else:
        dub_note = "needs a translator"
        # Distinguish "no ASR" from "config unreadable" from "neither route
        # can run". Since the local translator landed, cloud being off is
        # no longer a blocker at all — it only decides WHICH engine runs —
        # so the cloud-off text moved to a note and this branch is reached
        # only when the local model is uncached AND barred from downloading.
        if not asr:
            dub_block = "no ASR installed"
        elif cloud_on is None:
            dub_block = ("config/config.toml could not be read, so whether "
                         "a translator is allowed is unknown — fix the "
                         "config before believing this tile")
        else:
            # Single source with `bta dub`'s refusal, so the tile and the
            # error cannot drift.
            dub_block = _translator_blocker_text()
    caps.append(Capability(
        key="dubbing", label="Dubbing",
        available=translator,
        blocker=dub_block,
        note=dub_note,
        # ``is False``, not ``not cloud_on``: policy requires the config to
        # have actually been read and to actually say false. It now marks
        # WHICH ENGINE was chosen by decision, not that the tile is dark —
        # a locally-translated tile is both available and by_policy.
        by_policy=bool(asr and cloud_on is False),
    ))

    # ---- publishing ------------------------------------------------
    caps.append(Capability(
        key="publish", label="Export pack",
        available=True,
        note=("caption, hashtags, thumbnail and chapters written beside "
              "each clip, fitted to every platform's real character limit "
              "— one paste to post. Draft upload is available for "
              "platforms you have signed into with `bta auth`: it fills "
              "the upload form and stops. It never presses Publish"),
    ))

    # ---- things that already run ------------------------------------
    # Both filters are required, so the blocker names whichever is absent
    # — checking only afftdn once made a speechnorm-less build unavailable
    # with an empty blocker.
    no_speech = [n for n in ("afftdn", "speechnorm") if not has_filter(n)]
    caps.append(Capability(
        key="speech", label="Speech enhancement",
        available=not no_speech,
        blocker="" if not no_speech else f"{' and '.join(no_speech)} missing",
        note="highpass, denoise, de-ess and level — applied before loudness",
    ))
    return caps


def summary() -> list[dict[str, object]]:
    """Dashboard-shaped view."""
    return [
        {"key": c.key, "label": c.label, "available": c.available,
         "blocker": c.blocker, "note": c.note, "by_policy": c.by_policy}
        for c in probe()
    ]
