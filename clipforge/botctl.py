"""Run ClipForge from the Telegram bot that delivers its clips.

The phone is where the operator already sees finished clips, so it is the
natural place to ask for one. This is the listening half of
`clipforge.notify`: same bot, same chat, the other direction.

The whole design is the authorisation. A bot token is a password that
anybody who finds it can use, and this process can spend hours of GPU and
reach the network, so:

* **One chat.** Messages from any other chat id are logged and dropped.
  The id is the operator's own, resolved exactly as delivery resolves it.
* **A fixed verb list.** `/clip`, `/status`, `/retry`, `/help` — each
  mapped to one function here. Nothing takes a shell string, nothing
  interpolates a message into a command line, and there is no verb that
  runs an arbitrary subcommand.
* **One job at a time.** A second `/clip` while one is running is
  refused, not queued: the GPU allows one job, and a queue on a phone is
  a way to start six hours of work by tapping six times.

Long-polling rather than a webhook: no port to open, nothing to expose,
and it works behind any router.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from clipforge.log import get_logger

log = get_logger(__name__)

API = "https://api.telegram.org"

HELP = (
    "What I can do:\n"
    "/clip <url> - download it and cut clips, then send them here\n"
    "/status - what the watcher is doing right now\n"
    "/retry - resend anything that failed to arrive\n"
    "/help - this")


@dataclass
class BotControl:
    """The listening loop. Injected all the way down so it is testable."""

    token: str
    chat_id: str
    ws_root: Path
    #: (verb, argument) -> what to say back. Replaced in tests.
    run_clip: Callable[[str], str] = None  # type: ignore[assignment]
    run_status: Callable[[], str] = None   # type: ignore[assignment]
    run_retry: Callable[[], str] = None    # type: ignore[assignment]
    get: Callable[..., Any] = None         # type: ignore[assignment]
    post: Callable[..., Any] = None        # type: ignore[assignment]
    poll_timeout_s: int = 25
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _busy: threading.Lock = field(default_factory=threading.Lock, init=False)

    # ------------------------------------------------------------- plumbing

    def _call(self, method: str, **params: Any) -> dict:
        url = f"{API}/bot{self.token}/{method}"
        if self.get is not None:
            return self.get(url, params)
        try:
            body = urllib.parse.urlencode(params).encode()
            with urllib.request.urlopen(url, data=body,
                                        timeout=self.poll_timeout_s + 15) as r:
                return json.load(r)
        except Exception as exc:  # noqa: BLE001 - a poll failure is not fatal
            return {"ok": False, "description": self.redact(str(exc))}

    def redact(self, text: str) -> str:
        return text.replace(self.token, "<token>") if self.token else text

    def say(self, text: str) -> None:
        self._call("sendMessage", chat_id=self.chat_id, text=text[:4000])

    def stop(self) -> None:
        self._stop.set()

    # -------------------------------------------------------------- the loop

    def run(self, offset: int | None = None) -> None:
        """Poll until stopped. Never raises out of a message."""
        log.info("botctl.listening", chat=str(self.chat_id))
        while not self._stop.is_set():
            res = self._call("getUpdates", timeout=self.poll_timeout_s,
                             offset=offset or 0, allowed_updates='["message"]')
            if not res.get("ok"):
                desc = str(res.get("description") or "")
                if "Conflict" in desc:
                    # Another poller (the OpenClaw gateway) owns this bot.
                    log.error("botctl.conflict", note=desc[:160])
                    self._stop.wait(30)
                    continue
                log.warning("botctl.poll_failed", error=desc[:160])
                self._stop.wait(5)
                continue
            for update in res.get("result") or []:
                offset = int(update.get("update_id", 0)) + 1
                try:
                    self.handle(update)
                except Exception as exc:  # noqa: BLE001 - one bad message
                    log.error("botctl.handler_failed",
                              error=self.redact(f"{type(exc).__name__}: {exc}")[:200])

    def handle(self, update: dict) -> str | None:
        """Act on one update. Returns what was said, for tests."""
        msg = update.get("message") or {}
        sender = str((msg.get("chat") or {}).get("id", ""))
        text = (msg.get("text") or "").strip()
        if sender != str(self.chat_id):
            # Not an error the stranger should learn anything from.
            log.warning("botctl.ignored_chat", chat=sender[:24],
                        note="only the operator's own chat is obeyed")
            return None
        if not text.startswith("/"):
            return self._reply(HELP)

        verb, _, argument = text.partition(" ")
        verb = verb.split("@", 1)[0].lower()      # /clip@thebot -> /clip
        argument = argument.strip()

        if verb in ("/help", "/start"):
            return self._reply(HELP)
        if verb == "/status":
            return self._reply(self.run_status())
        if verb == "/retry":
            return self._reply(self.run_retry())
        if verb == "/clip":
            if not argument:
                return self._reply("Send a link: /clip https://...")
            if not argument.lower().startswith(("http://", "https://")):
                return self._reply("That does not look like a link.")
            if not self._busy.acquire(blocking=False):
                return self._reply(
                    "Already working on one. This machine runs a single job "
                    "at a time; ask again when it lands.")
            self._reply("On it. I will send the clips here when they are cut.")
            threading.Thread(target=self._clip_then_report, args=(argument,),
                             name="botctl-clip", daemon=True).start()
            return "On it."
        return self._reply(f"I do not know {verb}.\n\n{HELP}")

    def _reply(self, text: str) -> str:
        self.say(text)
        return text

    def _clip_then_report(self, url: str) -> None:
        try:
            self.say(self.run_clip(url))
        except Exception as exc:  # noqa: BLE001 - always answer the phone
            self.say(self.redact(f"That failed: {type(exc).__name__}: {exc}")[:500])
        finally:
            self._busy.release()


def clip_a_url(url: str, *, config: Path, timeout_s: float = 7200.0) -> str:
    """Run `bta grab` on a link and report what came out.

    A subprocess, not an in-process call: a job started from a phone must
    not be able to take the listener down with it, and the CLI is the
    interface this project actually tests.
    """
    import subprocess
    import sys

    cmd = [sys.executable, "-m", "clipforge.cli", "grab", url,
           "--config", str(config)]
    log.info("botctl.clip_started", url=url[:120])
    try:
        proc = subprocess.run(cmd, cwd=str(Path(config).resolve().parent.parent),
                              capture_output=True, text=True, errors="replace",
                              timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return "That took longer than two hours and I stopped it."
    tail = (proc.stdout or "").strip().splitlines()[-6:]
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()[-4:]
        return "It failed:\n" + "\n".join(tail + err)[-1200:]
    return "Done:\n" + "\n".join(tail)[-1200:]
