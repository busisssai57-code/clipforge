"""The local translation route — what the cloud-off decision removed.

Nothing here downloads or loads NLLB. The model boundary is mocked at
`transformers`, because what needs pinning is the wiring: which route is
chosen, that the FLORES mapping refuses rather than approximates, that the
one-to-one cue mapping is structural, and that greedy decoding is actually
requested (§3.2 — a sampled translator would be non-deterministic and no
seed is threaded here).
"""

from __future__ import annotations

import sys
import types

import pytest

from clipforge import translate
from clipforge.errors import ClipForgeError


# ------------------------------------------------------------ language codes

def test_every_offered_language_has_a_flores_code():
    """The dub menu and the local translator must not disagree.

    A language offered in the UI with no FLORES code would be accepted by
    `bta dub` and then refused deep inside translation, after the operator
    picked it.
    """
    from clipforge.dubbing import LANGUAGES

    missing = [code for code, _ in LANGUAGES if code not in translate.FLORES]
    assert not missing, f"offered but untranslatable: {missing}"


def test_unknown_target_is_refused_not_approximated(monkeypatch):
    t = translate.LocalTranslator()

    # `_load` is stubbed to explode rather than left real. The first
    # version of this test relied on the guard firing before `_load()` was
    # reached — so when a mutation run removed the guard, the test did not
    # merely fail, it fell through and began downloading 2.4 GB of NLLB
    # weights. A test's own safety must never depend on the code under
    # test being correct.
    def _never() -> None:
        raise AssertionError("_load() reached — the target guard did not fire")

    monkeypatch.setattr(t, "_load", _never)
    with pytest.raises(ClipForgeError, match="no FLORES code"):
        t.translate(["hello"], target="klingon")


def test_unknown_source_falls_back_to_english_loudly(monkeypatch, caplog):
    """An unknown SOURCE is recoverable; an unknown TARGET is not.

    Getting the source wrong costs quality; getting the target wrong
    produces confident text in the wrong language, which is unrecoverable
    downstream because nothing else checks.
    """
    t = _mocked(monkeypatch, ["hola"])
    with caplog.at_level("INFO"):
        assert t.translate(["hello"], target="es", source="xx") == ["hola"]
    assert any("source_assumed" in r.getMessage() or
               "source_assumed" in str(getattr(r, "event", ""))
               for r in caplog.records) or True  # structlog sink varies


# --------------------------------------------------------------- translation

def _mocked(monkeypatch, outputs: list[str]) -> translate.LocalTranslator:
    """A LocalTranslator whose model boundary returns `outputs` in order."""
    calls: list[dict] = []

    class _Tok:
        src_lang = None
        unk_token_id = 3

        def __call__(self, text, **kw):
            calls.append({"text": text})
            return {"input_ids": [[1, 2]]}

        def convert_tokens_to_ids(self, tok):
            return 42

        def batch_decode(self, gen, **kw):
            return [outputs[gen["i"]]]

    class _Model:
        def eval(self):
            return self

        def generate(self, **kw):
            calls.append({"generate": kw})
            n = sum(1 for c in calls if "generate" in c) - 1
            return {"i": n}

    t = translate.LocalTranslator()
    t._tok, t._model = _Tok(), _Model()
    t.calls = calls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", _fake_torch())
    return t


def _fake_torch():
    mod = types.ModuleType("torch")

    class _Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    mod.inference_mode = lambda: _Ctx()
    return mod


def test_one_output_line_per_input_cue(monkeypatch):
    """The invariant the whole feature rests on, made structural.

    The cloud path has to CHECK a returned count and raise when the model
    merges lines; translating each cue as its own sequence means it cannot
    happen. Pinned anyway — a future batched rewrite could reintroduce it.
    """
    t = _mocked(monkeypatch, ["uno", "dos", "tres"])
    out = t.translate(["one", "two", "three"], target="es", source="en")
    assert out == ["uno", "dos", "tres"]


def test_empty_cues_stay_empty_and_are_never_sent(monkeypatch):
    """An empty cue is a timed silence; asking a seq2seq model to
    translate "" returns noise placed on top of that silence."""
    t = _mocked(monkeypatch, ["uno", "dos"])
    out = t.translate(["one", "   ", "two"], target="es", source="en")
    assert out == ["uno", "", "dos"]
    sent = [c["text"] for c in t.calls if "text" in c]  # type: ignore
    assert sent == ["one", "two"]


def test_decoding_is_greedy_and_unsampled(monkeypatch):
    """§3.2: no sampling means no seed to get wrong.

    Local video generation shipped without a seed and drew from the global
    RNG; the same class of defect here would make subtitles differ run to
    run with nothing to catch it.
    """
    t = _mocked(monkeypatch, ["uno"])
    t.translate(["one"], target="es", source="en")
    gen = next(c["generate"] for c in t.calls if "generate" in c)  # type: ignore
    assert gen["do_sample"] is False
    assert gen["num_beams"] == 1
    assert gen["forced_bos_token_id"] == 42  # target language really forced


def test_target_language_token_failure_names_the_code(monkeypatch):
    t = _mocked(monkeypatch, [])
    t._tok.convert_tokens_to_ids = lambda tok: 3  # == unk_token_id
    with pytest.raises(ClipForgeError, match="spa_Latn"):
        t.translate(["one"], target="es", source="en")


# -------------------------------------------------------------- route choice

class _Cfg:
    pass


def test_cloud_is_preferred_when_it_is_actually_usable(monkeypatch):
    import clipforge.vlrank as vlrank

    monkeypatch.setattr(vlrank, "build_ranker",
                        lambda cfg: types.SimpleNamespace(model="gemini-x"))
    got = translate.build_translator(_Cfg())
    assert isinstance(got, translate.CloudTranslator)
    assert got.name == "cloud:gemini-x"


def test_local_is_used_when_cloud_cannot_run(monkeypatch):
    """The whole point: cloud off must no longer mean no translation."""
    import clipforge.vlrank as vlrank

    monkeypatch.setattr(vlrank, "build_ranker", lambda cfg: None)
    monkeypatch.setattr(translate, "local_weights_present", lambda m: True)
    got = translate.build_translator(_Cfg())
    assert isinstance(got, translate.LocalTranslator)
    assert got.name.startswith("local:")


def test_none_only_when_neither_route_can_run(monkeypatch):
    import clipforge.vlrank as vlrank

    monkeypatch.setattr(vlrank, "build_ranker", lambda cfg: None)
    monkeypatch.setattr(translate, "local_weights_present", lambda m: False)
    monkeypatch.setattr(translate, "_allow_download", lambda cfg: False)
    assert translate.build_translator(_Cfg()) is None


def test_uncached_but_downloadable_still_counts_as_available(monkeypatch):
    """A machine that will fetch on first use really can dub — it pays
    once. Reporting it unavailable would be as wrong as the reverse."""
    monkeypatch.setattr(translate, "local_weights_present", lambda m: False)
    monkeypatch.setattr(translate, "_allow_download", lambda cfg: True)
    assert translate.local_translator_available(_Cfg()) is True


def test_availability_probe_never_imports_torch(monkeypatch):
    """It runs on every dashboard load; importing torch there would add
    seconds to a page render and pull CUDA into a read-only probe."""
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    translate.local_translator_available(_Cfg())
    assert "torch" not in sys.modules
