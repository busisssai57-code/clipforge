"""grab fetches the live chat alongside the video.

The download command is built by `_grab_ytdlp_cmd` so its flags can be
pinned without a network round trip. The point: a livestream VOD lands a
`<title>.live_chat.json` beside the media, which `process` already
auto-discovers into the S2 audience signal — so this closes the loop from
a bare URL to a chat-aware clip run.
"""

from __future__ import annotations

from pathlib import Path

from clipforge.cli import _grab_ytdlp_cmd


def test_the_command_requests_live_chat():
    cmd = _grab_ytdlp_cmd("https://example.test/watch?v=x", Path("/dl"), "node")
    assert "--write-subs" in cmd
    i = cmd.index("--sub-langs")
    assert cmd[i + 1] == "live_chat", "must fetch ONLY live chat, not all subs"


def test_the_output_stem_matches_what_process_discovers():
    """The video and the chat file share a stem, so chat.discover_beside
    finds `<stem>.live_chat.json` next to `<stem>.mp4`."""
    cmd = _grab_ytdlp_cmd("u", Path("/dl"), None)
    i = cmd.index("-o")
    assert cmd[i + 1] == "/dl/%(title).60s.%(ext)s"


def test_height_cap_and_url_are_preserved():
    cmd = _grab_ytdlp_cmd("https://example.test/v", Path("/dl"), "deno")
    assert cmd[-1] == "https://example.test/v"
    assert any("height<=1080" in a for a in cmd), "the 1080 cap was dropped"
    assert cmd[cmd.index("--js-runtimes") + 1] == "deno"


def test_no_runtime_omits_the_flag():
    cmd = _grab_ytdlp_cmd("u", Path("/dl"), None)
    assert "--js-runtimes" not in cmd
