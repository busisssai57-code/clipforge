"""Taking orders from Telegram: what it obeys, and what it refuses.

A bot token is a password. Anyone who finds one can message the bot, and
this process spends GPU hours and reaches the network — so the
authorisation is the feature, and these are the tests that matter.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from clipforge.botctl import HELP, BotControl

TOKEN = "123456:AAFAKE-token-for-tests-only"
MINE = "7889536775"
STRANGER = "5150"


def msg(text: str, *, chat: str = MINE, uid: int = 1) -> dict:
    return {"update_id": uid,
            "message": {"chat": {"id": int(chat)}, "text": text}}


@pytest.fixture()
def bot(tmp_path):
    said: list[str] = []
    ran: list[tuple[str, str]] = []

    def fake_get(url, params):
        if url.endswith("sendMessage"):
            said.append(params["text"])
            return {"ok": True}
        return {"ok": True, "result": []}

    control = BotControl(
        token=TOKEN, chat_id=MINE, ws_root=tmp_path,
        run_clip=lambda url: ran.append(("clip", url)) or f"clipped {url}",
        run_status=lambda: "watching 1 channel(s)",
        run_retry=lambda: "queue: {'sent': 2}",
        get=fake_get)
    control.said, control.ran = said, ran
    return control


# --------------------------------------------------------- the boundary

def test_a_stranger_is_ignored_entirely(bot):
    """Not refused with an explanation — ignored. A stranger should not
    learn that the bot does anything at all."""
    assert bot.handle(msg("/clip https://evil.example/x", chat=STRANGER)) is None
    assert bot.said == [], "a stranger got a reply"
    assert bot.ran == [], "a stranger started a job"


def test_a_stranger_cannot_read_status_either(bot):
    assert bot.handle(msg("/status", chat=STRANGER)) is None
    assert bot.said == []


def test_an_unknown_verb_runs_nothing(bot):
    bot.handle(msg("/deleteeverything"))
    assert bot.ran == []
    assert "do not know" in bot.said[-1]


def test_there_is_no_verb_that_runs_a_shell(bot):
    """The fixed list IS the design: no argument is ever interpolated into
    a command line, and nothing takes a subcommand name."""
    import inspect

    from clipforge import botctl

    src = inspect.getsource(botctl)
    assert "shell=True" not in src
    for verb in ("/exec", "/run", "/sh", "/eval"):
        assert verb not in src


def test_a_non_link_is_refused_before_anything_starts(bot):
    bot.handle(msg("/clip file:///C:/Windows/System32"))
    assert bot.ran == []
    assert "does not look like a link" in bot.said[-1]


# ------------------------------------------------------------- the verbs

def test_status_answers_from_the_heartbeat(bot):
    bot.handle(msg("/status"))
    assert bot.said[-1] == "watching 1 channel(s)"


def test_retry_flushes_the_queue(bot):
    bot.handle(msg("/retry"))
    assert "sent" in bot.said[-1]


def test_help_is_the_answer_to_anything_that_is_not_a_command(bot):
    bot.handle(msg("hello?"))
    assert bot.said[-1] == HELP


def test_a_group_style_mention_still_works(bot):
    bot.handle(msg("/status@myopenclaw2026_bot"))
    assert bot.said[-1] == "watching 1 channel(s)"


def test_clip_acknowledges_first_then_reports(bot):
    bot.handle(msg("/clip https://youtu.be/abc"))
    deadline = time.monotonic() + 5
    while len(bot.said) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "On it" in bot.said[0], "the phone waited with no acknowledgement"
    assert bot.said[-1] == "clipped https://youtu.be/abc"
    assert bot.ran == [("clip", "https://youtu.be/abc")]


def test_a_second_job_is_refused_not_queued(bot):
    """One GPU. A queue on a phone is a way to start six hours of work by
    tapping six times."""
    release = threading.Event()
    bot.run_clip = lambda url: (release.wait(timeout=5), "done")[1]
    bot.handle(msg("/clip https://a.example/1"))
    time.sleep(0.1)
    bot.handle(msg("/clip https://a.example/2", uid=2))
    assert "Already working" in bot.said[-1]
    release.set()


def test_a_failing_job_still_answers_the_phone(bot):
    def boom(url):
        raise RuntimeError("yt-dlp exploded")

    bot.run_clip = boom
    bot.handle(msg("/clip https://a.example/1"))
    deadline = time.monotonic() + 5
    while "failed" not in " ".join(bot.said) and time.monotonic() < deadline:
        time.sleep(0.02)
    assert "yt-dlp exploded" in bot.said[-1]


def test_the_token_never_reaches_the_chat(bot):
    def boom(url):
        raise RuntimeError(f"connecting to /bot{TOKEN}/sendVideo failed")

    bot.run_clip = boom
    bot.handle(msg("/clip https://a.example/1"))
    deadline = time.monotonic() + 5
    while len(bot.said) < 2 and time.monotonic() < deadline:
        time.sleep(0.02)
    assert TOKEN not in " ".join(bot.said)
    assert "<token>" in bot.said[-1]


# --------------------------------------------------------------- the loop

def test_another_poller_owning_the_bot_is_reported_not_spun_on(tmp_path):
    """The OpenClaw gateway polls the same bot. Two pollers fight, and
    Telegram says so; a hot retry loop would make it worse."""
    calls: list[float] = []

    def conflicted(url, params):
        calls.append(time.monotonic())
        if len(calls) >= 2:
            control.stop()
        return {"ok": False, "description": "Conflict: terminated by other getUpdates request"}

    control = BotControl(token=TOKEN, chat_id=MINE, ws_root=tmp_path,
                         get=conflicted)
    started = time.monotonic()
    thread = threading.Thread(target=control.run, daemon=True)
    thread.start()
    thread.join(timeout=2)
    control.stop()
    assert len(calls) <= 2, "it hammered Telegram through a conflict"


def test_the_cli_wires_the_operators_own_chat():
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.bot)
    assert "notify.target_from_config" in src, (
        "the bot must obey the chat clips are delivered to, resolved the "
        "same way")
    assert "chat_id=target.chat_id" in src
