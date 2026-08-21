"""Typer sentinels must never be mistaken for user choices.

`grab` and `agent` call `process()` as a plain Python function. Typer
command functions are ordinary functions, so an argument the caller omits
keeps its `OptionInfo` DEFAULT OBJECT rather than the value Typer would
have parsed — and `OptionInfo` is truthy. Measured before the fix:

    bool(process's jumpcut default) is True

so `jumpcut if jumpcut is not None else cfg.pacing.enabled` chose the
sentinel, and jump-cut silence removal ran on both unattended paths even
though [pacing] is off by default. Pacing alters the source's rhythm,
which the config documents as an editorial choice rather than a
correction, so this was the pipeline overriding the operator silently.
"""

from __future__ import annotations

import inspect

import pytest
from typer.models import ArgumentInfo, OptionInfo

from clipforge import cli


def test_a_leaked_option_sentinel_becomes_the_intended_default():
    assert cli._cli_value(OptionInfo(default=None), None) is None
    assert cli._cli_value(ArgumentInfo(default=None), 7) == 7


def test_real_values_survive_including_falsey_ones():
    """--no-jumpcut is False and must NOT be turned back into the config
    default; that would make the flag unusable in the off direction."""
    assert cli._cli_value(False, None) is False
    assert cli._cli_value(True, None) is True
    assert cli._cli_value(0, 5) == 0
    assert cli._cli_value(0.0, 1.0) == 0.0
    assert cli._cli_value(None, 3) is None


def test_the_sentinel_really_is_truthy():
    """The whole bug rests on this. If a future Typer makes OptionInfo
    falsey, the normaliser is still correct but this comment is not."""
    assert bool(OptionInfo(default=None)) is True


def test_a_leaked_sentinel_does_not_force_pacing_on():
    """THE bug, stated behaviourally.

    `grab`/`agent` omit `jumpcut`, so the sentinel arrives here. Pacing
    must fall back to the CONFIG value — which is False by default,
    because silence removal alters the source's rhythm and that is an
    editorial choice rather than a correction.

    Deliberately behavioural: the previous version of this test asserted
    that a particular assignment appeared in `process`'s source, and a
    mutant defeated it by leaving that text inside a comment while
    disabling the code.
    """
    sentinel = inspect.signature(cli.process).parameters["jumpcut"].default
    assert cli._pacing_enabled(sentinel, config_default=False) is False
    assert cli._pacing_enabled(sentinel, config_default=True) is True


def test_an_explicit_flag_beats_the_config_in_both_directions():
    assert cli._pacing_enabled(True, config_default=False) is True
    assert cli._pacing_enabled(False, config_default=True) is False
    assert cli._pacing_enabled(None, config_default=True) is True
    assert cli._pacing_enabled(None, config_default=False) is False


def test_process_uses_the_shared_pacing_decision():
    """Guard against the decision being re-inlined somewhere else."""
    src = inspect.getsource(cli.process)
    assert "_pacing_enabled(jumpcut" in src
    assert "jumpcut if jumpcut is not None" not in src, (
        "the raw ternary is back; it reads the truthy sentinel as a choice")


@pytest.mark.parametrize("param", ["abs_offset", "clips"])
def test_process_normalises_the_other_options_it_reads(param):
    src = inspect.getsource(cli.process)
    assert f"{param} = _cli_value({param}," in src, (
        f"process() reads {param} without normalising the Typer sentinel")


def test_process_defaults_are_still_sentinels():
    """Control: if Typer ever starts handing out real defaults, the
    normaliser becomes dead code and this test says so rather than
    leaving it to rot."""
    sig = inspect.signature(cli.process)
    assert isinstance(sig.parameters["jumpcut"].default, OptionInfo)


def test_verify_help_names_only_modules_that_exist():
    """A help string is a promise about what the gate can run.

    `verify`'s argument advertised "all | skeleton | ingestion | ai |
    compositing | orchestration" while `clipforge/verify/` held four
    modules; `clipforge verify compositing` answered "unknown verify
    module". Nothing failed, which is the point — an operator reading the
    help would believe two more gates existed and had passed.

    Read out of the signature rather than hardcoded, so the test fails on
    the next module named in help and never written.
    """
    import importlib

    param = inspect.signature(cli.verify).parameters["module"]
    named = [m.strip() for m in param.default.help.split("|")]
    missing = []
    for name in named:
        if name == "all":
            continue
        try:
            importlib.import_module(f"clipforge.verify.{name}")
        except ImportError:
            missing.append(name)
    assert not missing, f"help offers verify modules that do not exist: {missing}"
