"""``interactive_login`` — the sign-in loop, which had no tests at all.

This is the function the operator actually collided with: five YouTube
sign-in attempts over an hour, one DNS failure, one 15-minute timeout. It
returned False and saved nothing, which was correct — but nothing pinned
that correctness, and its most important behaviour is a *refusal* that is
invisible when it breaks.

The load-bearing case is ``_wait_for_enter`` swallowing ``EOFError``. With
no interactive stdin, ``input()`` raises immediately; treating that as a
keypress would report a successful sign-in that never happened and save an
unauthenticated session, which then fails later at upload time with a
message about the upload rather than about the login.

Playwright is never imported here — the function takes a ``page`` and only
ever touches ``.goto`` and ``.url``, so a stub covers it exactly.
"""

from __future__ import annotations

import sys

import pytest

from clipforge.poster import browser


class _Page:
    """The whole surface ``interactive_login`` uses: goto() and .url."""

    def __init__(self, urls, *, goto_raises=None):
        # `urls` is consumed one entry per poll; the last value repeats.
        self._urls = list(urls)
        self.goto_raises = goto_raises
        self.goto_calls: list[str] = []
        self.closed = False

    def goto(self, url, **kw):
        self.goto_calls.append(url)
        if self.goto_raises is not None:
            raise self.goto_raises

    @property
    def url(self):
        if self.closed:
            raise RuntimeError("Target page, context or browser has been closed")
        return self._urls.pop(0) if len(self._urls) > 1 else self._urls[0]


@pytest.fixture
def clock(monkeypatch):
    """Deterministic time: sleep advances a fake monotonic clock.

    Without this the timeout case takes fifteen real minutes, which is why
    it is the kind of test that does not get written.
    """
    now = {"t": 0.0}
    monkeypatch.setattr(browser.time, "monotonic", lambda: now["t"])
    monkeypatch.setattr(browser.time, "sleep",
                        lambda s: now.__setitem__("t", now["t"] + s))
    return now


@pytest.fixture(autouse=True)
def _no_console(monkeypatch):
    """Default to a non-tty stdin; the ENTER thread is opted into per test."""
    monkeypatch.setattr(sys, "stdin", None)


def _login(page, **kw):
    return browser.interactive_login(
        page, platform=kw.pop("platform", "youtube"),
        start_url=kw.pop("start_url", "https://studio.example/signin"),
        done_when=kw.pop("done_when", ("/channel/",)),
        where=kw.pop("where", "your channel page"), **kw)


# ------------------------------------------------------------------ success

def test_reaching_the_destination_completes_on_its_own(clock):
    """URL polling is what finishes a login; ENTER is only the override."""
    page = _Page(["https://studio.example/signin",
                  "https://studio.example/signin?challenge=2fa",
                  "https://studio.example/channel/UC123"])
    assert _login(page) is True


def test_a_substring_match_is_what_counts(clock):
    page = _Page(["https://x.example/channel/UC1?flow=1"])
    assert _login(page, done_when=("/channel/",)) is True


def test_unrelated_urls_never_complete_the_login(clock, capsys):
    """A consent interstitial is not a signed-in channel page."""
    page = _Page(["https://accounts.example/consent"])
    assert _login(page) is False
    assert "Gave up" in capsys.readouterr().out


# ------------------------------------------------------------------ refusals

def test_non_interactive_stdin_never_counts_as_a_keypress(clock, monkeypatch):
    """The one that matters: ``input()`` raising is NOT a confirmation.

    Under a non-tty stdin the ENTER thread must not start at all. If it did
    and its EOFError were read as a press, this returns True and the caller
    saves a session for an account nobody signed into.
    """
    started: list[object] = []
    real_thread = browser.threading.Thread

    class _Spy(real_thread):
        def start(self):
            started.append(self)
            super().start()

    monkeypatch.setattr(browser.threading, "Thread", _Spy)
    page = _Page(["https://accounts.example/consent"])

    assert _login(page) is False
    assert started == [], "no ENTER thread may run without a console"


def test_eof_on_a_tty_still_does_not_confirm(clock, monkeypatch):
    """Belt and braces: even with a console, a raising ``input()`` is not a
    press. The thread is allowed to start here; only a real line counts."""
    monkeypatch.setattr(
        sys, "stdin", type("S", (), {"isatty": staticmethod(lambda: True)})())
    monkeypatch.setattr(browser, "input", lambda: (_ for _ in ()).throw(
        EOFError("no stdin")), raising=False)

    page = _Page(["https://accounts.example/consent"])
    assert _login(page) is False


def test_a_closed_window_returns_false_and_says_nothing_was_saved(clock,
                                                                  capsys):
    page = _Page(["https://studio.example/signin"])
    page.closed = True
    assert _login(page) is False
    assert "Nothing was saved" in capsys.readouterr().out


def test_timeout_gives_up_and_reports_the_real_budget(clock, capsys):
    page = _Page(["https://accounts.example/consent"])
    assert _login(page) is False
    # the clock advanced to the documented ceiling, not some other number
    assert clock["t"] >= browser.INTERACTIVE_LOGIN_TIMEOUT_S
    out = capsys.readouterr().out
    assert f"{int(browser.INTERACTIVE_LOGIN_TIMEOUT_S / 60)} minutes" in out


# ------------------------------------------------- documented ordering rules

def test_instructions_print_before_navigation(clock, capsys, monkeypatch):
    """Documented failure 1: a heavy sign-in page with a silent console
    looks frozen. The operator must be told what to do FIRST."""
    seen: list[str] = []

    class _SlowPage(_Page):
        def goto(self, url, **kw):
            seen.append("goto:" + capsys.readouterr().out)
            super().goto(url, **kw)

    page = _SlowPage(["https://studio.example/channel/UC1"])
    _login(page)

    assert seen, "goto was never called"
    printed_before_goto = seen[0][len("goto:"):]
    assert "Sign in there yourself" in printed_before_goto
    assert "never sees your password" in printed_before_goto


def test_a_failed_navigation_does_not_abort_the_login(clock, capsys):
    """Documented failure 2: the operator can navigate themselves, so a
    goto timeout must not throw away the browser they are about to use."""
    page = _Page(["https://studio.example/channel/UC1"],
                 goto_raises=RuntimeError("net::ERR_NAME_NOT_RESOLVED"))

    assert _login(page) is True  # login still completes by URL detection
    out = capsys.readouterr().out
    assert "Could not open" in out and "RuntimeError" in out
    assert "Navigate there in the window" in out


def test_no_enter_prompt_is_offered_without_a_console(clock, capsys):
    page = _Page(["https://studio.example/channel/UC1"])
    _login(page)
    assert "Press ENTER" not in capsys.readouterr().out
