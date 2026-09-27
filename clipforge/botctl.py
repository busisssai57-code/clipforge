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

#: Conflicts to tolerate before standing down. Retrying for ever means the
#: operator's commands keep vanishing into whichever poller wins the race.
MAX_CONFLICTS = 5

#: Hosts a clip may be fetched from. A scheme test alone let /clip point
#: yt-dlp's generic extractor at 127.0.0.1:8765 (this project's own API),
#: at a LAN router, or at 169.254.169.254 — the address a cloud metadata
#: service answers on. Measured: all four were accepted.
def is_fetchable_url(raw: str) -> tuple[bool, str]:
    """(ok, why not). Public http(s) only; no IP literals, no private nets."""
    import ipaddress
    from urllib.parse import urlparse

    try:
        parsed = urlparse((raw or "").strip())
    except ValueError:
        return False, "that is not a URL I can read"
    if parsed.scheme not in ("http", "https"):
        return False, "links only, over http or https"
    host = (parsed.hostname or "").strip()
    if not host:
        return False, "that link has no host"
    if host.lower() in ("localhost", "localhost.localdomain"):
        return False, "that points back at this machine"
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return True, ""          # a name: let the resolver and yt-dlp judge
    if (addr.is_loopback or addr.is_private or addr.is_link_local
            or addr.is_reserved or addr.is_multicast):
        return False, "that points inside this network"
    return False, "give me a link to a page, not a bare IP"


HELP = (
    "What I can do:\n"
    "/clip <url> - download it and cut clips, then send them here\n"
    "/status - what the watcher is doing right now\n"
    "/retry - resend anything that failed to arrive\n"
    "/stop - cancel the clip I am working on\n"
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
    #: The running job, so /stop can end it.
    _job: Any = field(default=None, init=False)
    #: When each sender was last answered, for the flood guard.
    _last_seen: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        # An empty chat id made every comparison succeed against a message
        # with no chat object — the authorisation failed OPEN. Unreachable
        # through the CLI today, which is not a reason to allow it here.
        cid = str(self.chat_id).strip()
        if not cid.lstrip("-").isdigit() or int(cid) <= 0:
            raise ValueError(
                f"refusing to listen for chat id {self.chat_id!r}: it must "
                "be one positive Telegram user id")
        self.chat_id = cid

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

    def drain(self) -> int | None:
        """Discard whatever is queued and return the next offset.

        Telegram keeps an unconfirmed update for 24 hours and hands it
        back on the next poll. Without this, a restart re-executes the
        /clip that was in flight when the bot died — a two-hour GPU job,
        again, unasked.
        """
        res = self._call("getUpdates", timeout=0, offset=0)
        ups = res.get("result") or []
        if not ups:
            return None
        last = int(ups[-1].get("update_id", 0)) + 1
        self._call("getUpdates", timeout=0, offset=last)
        log.info("botctl.backlog_discarded", count=len(ups),
                 note="commands sent while the bot was down are not replayed")
        return last

    def run(self, offset: int | None = None) -> None:
        """Poll until stopped. Never raises out of a message."""
        log.info("botctl.listening", chat=str(self.chat_id))
        if offset is None:
            offset = self.drain()
        conflicts = 0
        while not self._stop.is_set():
            res = self._call("getUpdates", timeout=self.poll_timeout_s,
                             offset=offset or 0, allowed_updates='["message"]')
            if not res.get("ok"):
                desc = str(res.get("description") or "")
                if "Conflict" in desc:
                    # Another poller (the OpenClaw gateway) owns this bot.
                    # Both pollers then get a random half of the operator's
                    # commands, which from the phone looks like the bot
                    # ignoring them. Say so once, then give up rather than
                    # retry for ever into a log nobody reads.
                    conflicts += 1
                    log.error("botctl.conflict", attempt=conflicts,
                              note=desc[:160])
                    if conflicts == 1:
                        self.say("Something else is polling this bot "
                                 "(the OpenClaw gateway?). Commands will go "
                                 "missing until one of us stops.")
                    if conflicts >= MAX_CONFLICTS:
                        log.error("botctl.giving_up", attempts=conflicts,
                                  note="another poller owns this bot")
                        self.say("Still not the only one listening. I am "
                                 "stopping; restart me once the other "
                                 "poller is off.")
                        return
                    self._stop.wait(30)
                    continue
                conflicts = 0
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
        if not sender or sender != str(self.chat_id):
            # Not an error the stranger should learn anything from. Logged
            # at most once a minute per sender: the line is small, but an
            # unrotated stderr log and a flood of messages are a bad pair,
            # and the forensic value of the millionth copy is zero.
            now = time.monotonic()
            last = self._last_seen.get(sender or "?", 0.0)
            if now - last > 60.0:
                self._last_seen[sender or "?"] = now
                log.warning("botctl.ignored_chat", chat=(sender or "")[:24],
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
        if verb == "/stop":
            job = self._job
            if job is None:
                return self._reply("Nothing is running.")
            self._cancel_job()
            return self._reply("Stopping it.")
        if verb == "/clip":
            if not argument:
                return self._reply("Send a link: /clip https://...")
            ok, why = is_fetchable_url(argument)
            if not ok:
                return self._reply(why.capitalize() + ".")
            if not self._busy.acquire(blocking=False):
                return self._reply(
                    "Already working on one. This machine runs a single job "
                    "at a time; ask again when it lands.")
            self._reply("On it. I will send the clips here when they are "
                        "cut. /stop cancels it.")
            threading.Thread(target=self._clip_then_report, args=(argument,),
                             name="botctl-clip", daemon=True).start()
            return "On it."
        return self._reply(f"I do not know {verb}.\n\n{HELP}")

    def _reply(self, text: str) -> str:
        self.say(text)
        return text

    def _cancel_job(self) -> None:
        """End the running clip job, children and all."""
        job = self._job
        if job is None:
            return
        try:
            job()
        except Exception as exc:  # noqa: BLE001
            log.warning("botctl.cancel_failed",
                        error=f"{type(exc).__name__}: {exc}"[:200])

    def _clip_then_report(self, url: str) -> None:
        try:
            self.say(self.run_clip(url))
        except Exception as exc:  # noqa: BLE001 - always answer the phone
            self.say(self.redact(f"That failed: {type(exc).__name__}: {exc}")[:500])
        finally:
            # ALWAYS, whatever happened: a job that ends without releasing
            # this refuses every later /clip with "already working on one"
            # and nothing restarts the bot, because its process is alive.
            self._job = None
            try:
                self._busy.release()
            except RuntimeError:
                pass


def _kill_tree(proc: Any) -> None:
    """Kill a process AND its children.

    `subprocess.run(timeout=...)` kills only the direct child. `bta grab`
    spawns yt-dlp as a grandchild that inherits the pipes, so the timeout
    fired, the parent went on waiting for those pipes to close, and the
    download carried on. The job never returned, the busy lock was never
    released, and every later /clip was refused — while the supervisor saw
    a live process and restarted nothing.
    """
    try:
        import psutil

        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            with contextlib.suppress(Exception):
                child.kill()
        with contextlib.suppress(Exception):
            parent.kill()
        return
    except Exception:  # noqa: BLE001 - psutil missing or the pid is gone
        pass
    with contextlib.suppress(Exception):
        proc.kill()


def clip_a_url(url: str, *, config: Path, timeout_s: float = 7200.0,
               register: "Callable[[Any], None] | None" = None) -> str:
    """Run the grab command on a link and report what came out.

    A subprocess, not an in-process call: a job started from a phone must
    not be able to take the listener down with it, and the CLI is the
    interface this project actually tests.

    ``register`` is handed a canceller so /stop can end the job.
    """
    import subprocess
    import sys

    # Absolute config, and the REPO as cwd. The old expression
    # (config.parent.parent) named D:\ for a bare filename and C:\Users
    # for a config in the home directory — and because the workspace root
    # is a RELATIVE path, the child would then build a second, empty
    # workspace there and deliver from a different .env.
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, "-m", "clipforge.cli", "grab",
           "--config", str(Path(config).resolve()),
           # Everything after -- is a positional, so a link that begins
           # with a dash cannot arrive as a flag.
           "--", url]
    log.info("botctl.clip_started", url=url[:120])
    proc = subprocess.Popen(cmd, cwd=str(root), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True,
                            errors="replace")
    if register is not None:
        register(lambda: _kill_tree(proc))
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_tree(proc)
        with contextlib.suppress(Exception):
            out, err = proc.communicate(timeout=30)
        return ("That ran past the time limit and I stopped it "
                f"({timeout_s / 3600:.0f}h).")
    tail = (out or "").strip().splitlines()[-6:]
    if proc.returncode != 0:
        if proc.returncode in (-9, -15, 1) and not tail:
            return "Stopped."
        errs = (err or "").strip().splitlines()[-4:]
        return "It failed:\n" + "\n".join(tail + errs)[-1200:]
    return "Done:\n" + "\n".join(tail)[-1200:]
