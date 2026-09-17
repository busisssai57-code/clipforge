"""A saved platform sign-in is the account. Treat it that way on disk.

What ``bta auth tiktok`` writes under ``workspace/auth`` is a live
session cookie jar for the operator's real TikTok / YouTube / Instagram /
X account — for anyone who can read the file, it IS that account, no
password and no second factor required. It was being written with the
default umask.
"""

from __future__ import annotations

import os
import stat
import sys

import pytest

from clipforge.poster import browser

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="POSIX mode bits; on Windows chmod only moves the read-only bit")


def _mode(path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_the_auth_directory_is_owner_only(tmp_path):
    auth = tmp_path / "auth"
    browser.get_auth_file(auth, "tiktok")
    assert _mode(auth) == 0o700


def test_a_saved_session_file_is_owner_only(tmp_path):
    session = tmp_path / "auth" / "tiktok_session.json"
    session.parent.mkdir(parents=True)
    session.write_text('{"cookies": []}', encoding="utf-8")
    session.chmod(0o644)
    browser._restrict(session)
    assert _mode(session) == 0o600


def test_locking_down_is_advisory_and_never_raises(tmp_path):
    """A session that could not be locked down still beats no session."""
    browser._restrict(tmp_path / "does-not-exist.json")


def test_the_renderer_sandbox_is_not_disabled():
    """--no-sandbox turned off the strongest boundary in a browser that
    then loads whatever a social platform serves. It is not what makes
    automation detectable, so it bought nothing for what it cost."""
    import inspect

    # Comments are stripped first: the reason the flags are gone is
    # written next to where they used to be, and a substring search over
    # raw source would match the explanation and call it a regression.
    src = inspect.getsource(browser.launch_stealth_browser)
    code = "\n".join(line.split("#", 1)[0]
                     for line in src.splitlines())
    assert "--no-sandbox" not in code
    assert "--disable-setuid-sandbox" not in code
