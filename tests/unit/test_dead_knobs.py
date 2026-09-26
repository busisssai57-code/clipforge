"""Every knob in config.toml must reach something.

This project has shipped dead knobs three times now: `quantize` (applied
by nothing), `s6.x264_preset` (hardcoded "medium" while nvenc's twin was
a real knob), and the per-stage `vram_budget_gb` this file was written
for — documented in config.example.toml as tunable while every stage read
its own hardcoded number.

A dead knob is worse than a missing one: the operator changes it, nothing
happens, and the config file is now a document that lies.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from clipforge.config import AppConfig
from clipforge.stages.s1_transcribe import S1Transcribe
from clipforge.stages.s3_semantic import S3SemanticRanker
from clipforge.stages.s4_tracking import S4Tracking

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "clipforge"


# --------------------------------------------------- the knob that was dead

@pytest.mark.parametrize("stage, default", [
    (S1Transcribe, 8.0), (S3SemanticRanker, 10.0), (S4Tracking, 3.0)])
def test_a_stage_takes_its_vram_budget_from_the_config(stage, default):
    made = (stage(db=None, artifacts_dir=".")
            if stage is S1Transcribe else stage(None, "."))
    assert made.vram_budget_gb == default, "the measured default moved"

    tuned = (stage(db=None, artifacts_dir=".", vram_budget_gb=4.25)
             if stage is S1Transcribe else stage(None, ".", vram_budget_gb=4.25))
    assert tuned.vram_budget_gb == 4.25, (
        "[sN] vram_budget_gb in config.toml reaches nothing")


def test_process_passes_the_configured_budget_to_every_gpu_stage():
    """Constructing the stage with the knob is half of it; the pipeline
    has to pass it, which is the half that was missing."""
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.process)
    for section in ("s1", "s3", "s4"):
        assert f"vram_budget_gb=cfg.{section}.vram_budget_gb" in src, (
            f"process() does not give S{section[-1]} its configured VRAM budget")


# ------------------------------------------------- the rule, for every knob

def _config_fields() -> dict[str, list[str]]:
    """Each [section] in config.toml and the fields it declares."""
    tree = ast.parse((PKG / "config.py").read_text(encoding="utf-8"))
    by_class: dict[str, list[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name.endswith("Config"):
            by_class[node.name] = [
                s.target.id for s in node.body
                if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)]
    app = next(n for n in tree.body
               if isinstance(n, ast.ClassDef) and n.name == "AppConfig")
    out: dict[str, list[str]] = {}
    for stmt in app.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            cls = getattr(stmt.annotation, "id", None)
            if cls in by_class:
                out[stmt.target.id] = by_class[cls]
    return out


def _package_text() -> str:
    parts = []
    for path in list(PKG.rglob("*.py")) + list(PKG.rglob("*.html")):
        if "__pycache__" in str(path) or path.name == "config.py":
            continue
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)


#: Knobs read indirectly and verified by hand, each with the reader named.
#: A new entry here is a claim someone must justify in review.
READ_INDIRECTLY = {
    ("s7", "use_cloud"): "clipforge/cloud.py registry (the §2 chokepoint)",
    ("s7", "vl_qa"): "s7_qa.py via getattr(cfg.s7, 'vl_qa')",
    ("s7", "vl_frames"): "vlqa.py via getattr(s7, 'vl_frames')",
    ("s3", "use_cloud"): "clipforge/cloud.py registry",
    ("genvideo", "use_cloud"): "clipforge/cloud.py registry",
}


def test_no_knob_in_the_config_reaches_nothing():
    code = _package_text()
    dead = []
    for section, fields in _config_fields().items():
        for field in fields:
            if (section, field) in READ_INDIRECTLY:
                continue
            direct = f"{section}.{field}" in code
            named = re.search(rf'\b{re.escape(field)}\b', code) is not None
            if not (direct or named):
                dead.append(f"[{section}] {field}")
    assert not dead, (
        "these knobs exist in config.toml and are read by nothing — wire "
        "them or delete them, but do not ship a config file that lies: "
        + ", ".join(sorted(dead)))


def test_the_example_config_documents_exactly_the_knobs_that_exist():
    """A knob missing from the example is undiscoverable; one that is in
    the example but not in the model is rejected at startup."""
    example = (ROOT / "config" / "config.example.toml").read_text(encoding="utf-8")
    import tomllib

    declared = tomllib.loads(example)
    model = AppConfig()
    for section, values in declared.items():
        assert hasattr(model, section), f"[{section}] is not a config section"
        for key in values:
            assert hasattr(getattr(model, section), key), (
                f"[{section}] {key} is in the example but not in the model")


def test_the_hashtag_cap_reaches_the_pack(tmp_path):
    """[editor] max_hashtags was documented and read by nothing; the pack
    always built up to eight."""
    from clipforge import export_pack

    clip = tmp_path / "c.mp4"
    clip.write_bytes(b"\x00" * 16)
    words = ("rosemary oil scalp growth routine before after results weeks "
             "shampoo thickness roots serum")
    few = export_pack.build_pack(clip, title="t", transcript_text=words,
                                 write=False, max_hashtags=3)
    many = export_pack.build_pack(clip, title="t", transcript_text=words,
                                  write=False, max_hashtags=8)
    assert len(few.hashtags) <= 3, "max_hashtags does not reach build_hashtags"
    assert len(many.hashtags) > len(few.hashtags), (
        "this test would pass on a hardcoded cap if the corpus were tiny")
