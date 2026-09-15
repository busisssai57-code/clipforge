"""A request value must reach the CLI as data, never as a flag.

Nothing in this API is shell-quoted, because nothing goes through a
shell — commands are spawned as an argv list. The parser that *does* get
to reinterpret these strings is Typer's: an argument beginning with ``-``
is an option, so a request naming its source ``--niche`` or its brief
``--shots`` steers the pipeline somewhere the endpoint never meant to
send it. Same class of bug as shell injection, one layer in.

These call the handlers directly, as the other web tests do, and assert
on the argv that would have been spawned.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from clipforge import web


@pytest.fixture()
def spawned(monkeypatch):
    """Capture argv instead of starting a process."""
    seen: list[list[str]] = []

    def _fake(kind, description, args):
        seen.append(list(args))
        return "task-test"

    monkeypatch.setattr(web, "_spawn_task", _fake)
    return seen


@pytest.mark.parametrize("hostile", ["--help", "-h", "--niche=evil",
                                     "  --clips"])
def test_a_source_that_would_read_as_a_flag_is_refused(spawned, hostile):
    with pytest.raises(HTTPException) as err:
        web.start_process(web.ProcessRequest(source=hostile, clips=1))
    assert err.value.status_code == 400
    assert not spawned


def test_a_brief_that_would_read_as_a_flag_is_refused(spawned):
    with pytest.raises(HTTPException) as err:
        web.start_generation(web.GenerateRequest(brief="--shots 999"))
    assert err.value.status_code == 400
    assert not spawned


def test_a_hostile_url_is_refused_before_the_scheme_check_passes_it(spawned):
    with pytest.raises(HTTPException):
        web.start_grab(web.GrabRequest(url="-x", clips=1))
    assert not spawned


def test_an_ordinary_source_still_goes_through(spawned):
    web.start_process(web.ProcessRequest(source="a video.mp4", clips=2))
    assert spawned == [["process", "a video.mp4", "--clips", "2"]]


def test_a_windows_path_is_not_mistaken_for_a_flag(spawned):
    """The common case must keep working: only a LEADING dash is refused."""
    web.start_process(web.ProcessRequest(source=r"D:\media\clip-01.mp4",
                                         clips=1))
    assert spawned[0][1] == r"D:\media\clip-01.mp4"


def test_a_url_with_dashes_in_it_still_works(spawned):
    web.start_grab(web.GrabRequest(url="https://ex.com/a-b-c?x=-1", clips=1))
    assert spawned[0][1] == "https://ex.com/a-b-c?x=-1"


def test_a_null_byte_is_refused(spawned):
    """It truncates in the C API, so what the CLI sees is not what was
    validated here."""
    with pytest.raises(HTTPException):
        web.start_process(web.ProcessRequest(source="ok.mp4\x00--niche=evil",
                                             clips=1))
    assert not spawned
