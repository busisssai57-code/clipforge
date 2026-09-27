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

from clipforge.botctl import HELP, MAX_CONFLICTS, BotControl, is_fetchable_url

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


@pytest.mark.parametrize("target", [
    "file:///C:/Windows/System32",
    "http://127.0.0.1:8765/api/jobs",     # this project's own control API
    "http://localhost/x",
    "http://[::1]/x",
    "http://169.254.169.254/latest/meta-data/",   # cloud metadata
    "http://192.168.1.1/x",               # the router
    "not a url at all",
])
def test_a_link_that_points_inward_is_refused_before_anything_starts(bot, target):
    """A scheme test alone let /clip aim yt-dlp's generic extractor at
    anything reachable from this machine."""
    bot.handle(msg(f"/clip {target}"))
    assert bot.ran == [], f"{target} started a job"
    assert bot.said, "nothing was said"


def test_an_ordinary_link_still_works(bot):
    """Control: the guard must not refuse everything."""
    bot.handle(msg("/clip https://youtu.be/abc"))
    import time as _t
    deadline = _t.monotonic() + 5
    while not bot.ran and _t.monotonic() < deadline:
        _t.sleep(0.02)
    assert bot.ran == [("clip", "https://youtu.be/abc")]


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

def test_a_conflicting_poller_is_announced_and_then_given_up_on(tmp_path):
    """The OpenClaw gateway polls the same bot. Two pollers split the
    operator's commands at random, which from the phone looks like the
    bot ignoring them — so it has to SAY so, and then stand down rather
    than retry into a log nobody reads.

    The previous version of this test was theatre: its fake stopped the
    loop on the second call, so the assertion held whether or not the
    back-off existed. This one counts what was said and waits on a clock
    it controls.
    """
    said: list[str] = []
    polls = {"n": 0}

    def conflicted(url, params):
        if url.endswith("sendMessage"):
            said.append(params["text"])
            return {"ok": True}
        polls["n"] += 1
        if polls["n"] > MAX_CONFLICTS + 3:
            raise AssertionError(
                f"it kept polling through {polls['n']} conflicts; the "
                "bail-out is gone")
        return {"ok": False,
                "description": "Conflict: terminated by other getUpdates request"}

    control = BotControl(token=TOKEN, chat_id=MINE, ws_root=tmp_path,
                         get=conflicted)
    control._stop.wait = lambda _timeout=None: False    # no real sleeping
    # A cap that is gone must FAIL this test, not hang it: without the
    # bail-out the loop is infinite, and a hanging suite tells nobody
    # anything.
    control.run()                                        # returns on its own

    assert polls["n"] <= MAX_CONFLICTS + 1, "it hammered Telegram"
    assert any("polling this bot" in s for s in said), (
        "the operator was never told why their commands vanish")
    assert any("stopping" in s.lower() for s in said), (
        "it retried for ever instead of standing down")


def test_the_cli_wires_the_operators_own_chat():
    import inspect

    from clipforge import cli

    src = inspect.getsource(cli.bot)
    assert "notify.target_from_config" in src, (
        "the bot must obey the chat clips are delivered to, resolved the "
        "same way")
    assert "chat_id=target.chat_id" in src


# --------------------------------------------------- what the audit found

def test_a_blank_chat_id_is_refused_at_construction(tmp_path):
    """With chat_id="" every comparison against a message with no chat
    object succeeded: the authorisation failed OPEN."""
    for bad in ("", "   ", "0", "-1001234567890", "abc"):
        with pytest.raises(ValueError):
            BotControl(token=TOKEN, chat_id=bad, ws_root=tmp_path)


def test_a_message_with_no_chat_is_ignored(bot):
    assert bot.handle({"update_id": 9, "message": {"text": "/clip https://x/y"}}) is None
    assert bot.ran == [] and bot.said == []


def test_an_empty_sender_never_matches_even_if_the_id_is_emptied(bot):
    """Defence in depth. Construction refuses a blank chat id, so this
    state is unreachable today — but the comparison itself must not be
    what stands between a stranger and a job: an empty string equals an
    empty string."""
    bot.chat_id = ""
    assert bot.handle({"update_id": 9, "message": {"text": "/clip https://x/y"}}) is None
    assert bot.ran == []


def test_stop_cancels_the_running_job(bot):
    cancelled = []
    bot._job = lambda: cancelled.append(True)
    bot.handle(msg("/stop"))
    assert cancelled == [True]
    assert "Stopping" in bot.said[-1]


def test_stop_with_nothing_running_says_so(bot):
    bot.handle(msg("/stop"))
    assert "Nothing is running" in bot.said[-1]


def test_a_job_that_ends_always_frees_the_next_one(bot):
    """A job that returns without releasing the lock refuses every later
    /clip with "already working on one", while the supervisor sees a live
    process and restarts nothing."""
    def boom(url):
        raise RuntimeError("killed")

    bot.run_clip = boom
    bot.handle(msg("/clip https://a.example/1"))
    import time as _t
    deadline = _t.monotonic() + 5
    while bot._busy.locked() and _t.monotonic() < deadline:
        _t.sleep(0.02)
    assert not bot._busy.locked(), "the bot wedged after a failed job"
    assert bot._job is None


def test_a_restart_does_not_replay_the_command_it_died_on(tmp_path):
    """Telegram keeps an unconfirmed update for 24 h. Without draining,
    a restart re-runs the /clip that was in flight — a two-hour GPU job,
    unasked."""
    calls: list[dict] = []

    def fake(url, params):
        calls.append({"url": url, **params})
        if url.endswith("getUpdates") and len([c for c in calls if "getUpdates" in c["url"]]) == 1:
            return {"ok": True, "result": [
                {"update_id": 41, "message": {"chat": {"id": int(MINE)},
                                              "text": "/clip https://x/y"}}]}
        return {"ok": True, "result": []}

    control = BotControl(token=TOKEN, chat_id=MINE, ws_root=tmp_path,
                         run_clip=lambda u: "never", get=fake)
    offset = control.drain()
    assert offset == 42
    confirms = [c for c in calls if "getUpdates" in c["url"] and c.get("offset") == 42]
    assert confirms, "the backlog was read but never confirmed"


def test_run_drains_before_it_starts_listening(tmp_path):
    """The whole point: run() must not hand the queued /clip to a job.
    Calling drain() from a test proves nothing if run() never calls it."""
    ran: list[str] = []
    polls = {"n": 0}

    def fake(url, params):
        if url.endswith("sendMessage"):
            return {"ok": True}
        polls["n"] += 1
        if polls["n"] == 1:      # the backlog, from before the crash
            return {"ok": True, "result": [
                {"update_id": 7, "message": {"chat": {"id": int(MINE)},
                                             "text": "/clip https://x/y"}}]}
        control.stop()
        return {"ok": True, "result": []}

    control = BotControl(token=TOKEN, chat_id=MINE, ws_root=tmp_path,
                         run_clip=lambda u: ran.append(u) or "done", get=fake)
    control.run()
    assert ran == [], "a restart re-ran the command it died on"


def test_the_flood_guard_keeps_one_line_per_sender_per_minute(bot):
    for i in range(50):
        bot.handle(msg("/status", chat=STRANGER, uid=i))
    assert bot.said == []
    assert len(bot._last_seen) == 1


def test_url_guard_allows_a_hostname_and_refuses_a_bare_ip():
    assert is_fetchable_url("https://youtube.com/watch?v=x")[0]
    assert not is_fetchable_url("https://8.8.8.8/x")[0]
