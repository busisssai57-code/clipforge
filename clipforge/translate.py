"""Translation, local-first — the route the cloud-off decision removed.

When cloud inference was switched off on 2026-08-05, dubbing was the only
capability that went dark, because the single translator wired into this
build was the Gemini one. The blocker text has named the replacement ever
since ("a local translation model (NLLB-200 or M2M100, ~2.5 GB) — is not
built yet"). This module is that route.

**Why NLLB-200 distilled 600M.** It translates directly between the
languages this project offers without pivoting through English, it is
small enough to run on CPU in seconds for the ~30 cues a clip has, and it
is a plain seq2seq model — no chat template, no JSON envelope, no schema
to repair. The failure mode that dominates cloud translation here (a model
that helpfully merges or renumbers lines) is structurally impossible
because each cue is translated as its own sequence.

**CPU deliberately.** §3.1 says exactly one model class is resident at a
time, and translation is not worth a GPU permit: it would have to contend
with the VL and ASR stages for the one `GPULock`, and a 600M seq2seq on
CPU costs seconds. Running it on CPU keeps it entirely outside the VRAM
Law rather than making it a new participant — the same reasoning that put
a seeded generator on CPU.

**Determinism (§3.2).** Greedy decoding, `do_sample=False`, `num_beams=1`,
fixed max length. No sampling means no seed to get wrong, which is the
defect this project already shipped once in local video generation.
"""

from __future__ import annotations

from typing import Any, Protocol

from clipforge.errors import ClipForgeError
from clipforge.log import get_logger

log = get_logger(__name__)

#: Small, direct, Apache-2.0-adjacent (CC-BY-NC for NLLB — see README note).
#: Overridable so a machine with the 1.3B already cached can point at it.
DEFAULT_LOCAL_MODEL = "facebook/nllb-200-distilled-600M"

#: NLLB uses FLORES-200 codes (language + script), not ISO-639-1. Mapping
#: them here rather than guessing: "zh" alone is ambiguous between scripts,
#: and an unmapped language must be REFUSED, not approximated with English.
FLORES: dict[str, str] = {
    "en": "eng_Latn", "es": "spa_Latn", "pt": "por_Latn",
    "fr": "fra_Latn", "de": "deu_Latn", "it": "ita_Latn",
    "nl": "nld_Latn", "pl": "pol_Latn", "tr": "tur_Latn",
    "ru": "rus_Cyrl", "ar": "arb_Arab", "hi": "hin_Deva",
    "ja": "jpn_Jpan", "ko": "kor_Hang", "zh": "zho_Hans",
    "id": "ind_Latn", "vi": "vie_Latn",
}

#: A subtitle cue is one short spoken line. This bounds a pathological
#: generation rather than the content — cues are already short.
_MAX_NEW_TOKENS = 200


class Translator(Protocol):
    """What `dub_clip` needs. Both routes satisfy it identically."""

    name: str

    def translate(self, lines: list[str], *, target: str,
                  source: str | None = None) -> list[str]:
        ...


def _flores(code: str | None) -> str | None:
    return FLORES.get((code or "").lower().split("-")[0]) or None


class LocalTranslator:
    """NLLB-200 on CPU, one cue at a time.

    Per-cue translation is the point, not a simplification: it makes the
    one-to-one line mapping a structural property instead of an instruction
    the model may ignore. `translate_lines` on the cloud path has to CHECK
    the returned count and raise when it drifts; here it cannot drift.
    """

    def __init__(self, model_id: str = DEFAULT_LOCAL_MODEL):
        self.model_id = model_id
        self.name = f"local:{model_id}"
        self._tok: Any = None
        self._model: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from transformers import (AutoModelForSeq2SeqLM,  # noqa: PLC0415
                                      AutoTokenizer)
        except ImportError as exc:  # pragma: no cover - transformers is a dep
            raise ClipForgeError(
                "transformers is not installed, so the local translator "
                "cannot run") from exc

        log.info("translate.loading", model=self.model_id)
        # No device_map / no .to("cuda"): see the module docstring. A CPU
        # load also means this never has to take the GPULock.
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self._model.eval()

    def translate(self, lines: list[str], *, target: str,
                  source: str | None = None) -> list[str]:
        if not lines:
            return []
        tgt = _flores(target)
        if tgt is None:
            raise ClipForgeError(
                f"the local translator has no FLORES code for {target!r}; "
                "refusing rather than translating into the wrong language")
        # An unknown SOURCE is recoverable — NLLB needs one, and the clip's
        # language is whatever ASR reported. English is the documented
        # assumption, logged, not silent.
        src = _flores(source)
        if src is None:
            log.info("translate.source_assumed", given=source, using="eng_Latn")
            src = "eng_Latn"

        self._load()
        import torch  # noqa: PLC0415

        self._tok.src_lang = src
        bos = self._convert_target(tgt)

        out: list[str] = []
        with torch.inference_mode():
            for line in lines:
                text = (line or "").strip()
                if not text:
                    # An empty cue stays empty: it is a timed silence, and
                    # asking a seq2seq model to translate "" returns noise.
                    out.append("")
                    continue
                enc = self._tok(text, return_tensors="pt", truncation=True,
                                max_length=512)
                gen = self._model.generate(
                    **enc, forced_bos_token_id=bos,
                    max_new_tokens=_MAX_NEW_TOKENS,
                    # §3.2: greedy and unsampled, so two runs of the same
                    # cue produce the same bytes without needing a seed.
                    do_sample=False, num_beams=1)
                out.append(self._tok.batch_decode(
                    gen, skip_special_tokens=True)[0].strip())

        # Structural, but assert it anyway: this is the invariant the whole
        # feature rests on, and a future batched rewrite could break it.
        if len(out) != len(lines):
            raise ClipForgeError(
                f"translator produced {len(out)} lines for {len(lines)} cues")
        return out

    def _convert_target(self, flores_code: str) -> int:
        """Resolve the forced BOS token across transformers versions.

        `lang_code_to_id` was removed from NllbTokenizer in transformers
        4.45+; `convert_tokens_to_ids` works on both. Picking one and
        hoping is how this project has lost an afternoon before (whisperx
        renaming `use_auth_token`), so both are tried and a failure names
        the code rather than raising a bare KeyError.
        """
        table = getattr(self._tok, "lang_code_to_id", None)
        if isinstance(table, dict) and flores_code in table:
            return int(table[flores_code])
        tok_id = self._tok.convert_tokens_to_ids(flores_code)
        unk = getattr(self._tok, "unk_token_id", None)
        if tok_id is None or tok_id == unk:
            raise ClipForgeError(
                f"this tokenizer does not know the language code "
                f"{flores_code!r}; the model may not be an NLLB checkpoint")
        return int(tok_id)


class CloudTranslator:
    """The existing Gemini route, unchanged, behind the same interface."""

    def __init__(self, ranker: Any):
        self._ranker = ranker
        self.name = f"cloud:{getattr(ranker, 'model', 'gemini')}"

    def translate(self, lines: list[str], *, target: str,
                  source: str | None = None) -> list[str]:
        from clipforge.dubbing import _LANG_NAME  # noqa: PLC0415
        from clipforge.vlrank import translate_lines  # noqa: PLC0415

        return translate_lines(self._ranker, lines,
                               target_language=_LANG_NAME.get(target, target),
                               source_language=source)


def local_weights_present(model_id: str = DEFAULT_LOCAL_MODEL) -> bool:
    """True when the model is already cached, so nothing will download.

    Probes the HF cache without importing torch or instantiating anything —
    the capability tile calls this on every dashboard load.
    """
    try:
        from huggingface_hub import scan_cache_dir  # noqa: PLC0415

        return any(repo.repo_id == model_id
                   for repo in scan_cache_dir().repos)
    except Exception:  # noqa: BLE001 - absence is the answer, not a crash
        return False


def local_translator_available(cfg: Any) -> bool:
    """Whether the local route could run, without loading anything.

    Called on every dashboard load, so it must stay cheap: it reads config
    and scans the HF cache directory, and never imports torch or
    transformers. "Cached, or allowed to fetch" is the honest test — a
    machine that will download on first use really can dub, it just pays
    once, and the note says so.
    """
    return (local_weights_present(_local_model_id(cfg))
            or _allow_download(cfg))


def build_translator(cfg: Any) -> Translator | None:
    """The one translator this machine should use, or None.

    Cloud first when it is BOTH enabled and keyed — it is better, and a
    machine that has deliberately turned it on should get it. Otherwise
    local, which needs no credential and no permission. None only when
    neither route can run, and `translator_blocker` explains which.
    """
    from clipforge.vlrank import build_ranker  # noqa: PLC0415

    ranker = build_ranker(cfg)
    if ranker is not None:
        return CloudTranslator(ranker)

    model_id = _local_model_id(cfg)
    if not local_weights_present(model_id) and not _allow_download(cfg):
        return None
    return LocalTranslator(model_id)


def _local_model_id(cfg: Any) -> str:
    section = getattr(cfg, "dubbing", None)
    return str(getattr(section, "local_model_id", None) or DEFAULT_LOCAL_MODEL)


def _allow_download(cfg: Any) -> bool:
    """Whether a first run may fetch ~2.4 GB.

    Defaulting to True matches every other model here (YOLO, LTX,
    face_landmarker all fetch on first use) and keeps the feature usable
    out of the box; the capability note says plainly that the first run
    downloads, so it is never a surprise.
    """
    section = getattr(cfg, "dubbing", None)
    value = getattr(section, "allow_model_download", None)
    return True if value is None else bool(value)
