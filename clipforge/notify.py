"""Send each accepted clip to the operator's phone over Telegram.

Private delivery, not publishing: one clip, one chat, the operator's own.
Nothing here posts to a platform, and nothing is sent unless
``[notify] telegram = true``.

Three properties this module exists to hold:

* **It never sinks a clip.** A clip that passed QA is on disk before this
  runs; a dead network, a revoked token or a 413 is logged and queued for
  retry, never raised into the pipeline.
* **A failed send is not a lost send.** Failures land in
  ``workspace/outbox/telegram/`` and `bta watch` retries them on a timer,
  so a phone that was offline for an hour still gets the clip.
* **Each clip arrives once.** A ``<clip>.telegram.json`` sidecar records the
  delivery; re-running `bta process` on the same source reuses the cached
  render and must not buzz the phone again.

The bot token is a credential, so it is never logged — including inside
exception text, where ``requests`` embeds the request URL (and with it the
token, which Telegram puts in the path).
"""

from __future__ import annotations

import contextlib
import json
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from clipforge.log import get_logger

log = get_logger(__name__)

API = "https://api.telegram.org"
#: Bot API upload ceiling is 50 MB; leave headroom for multipart framing.
MAX_UPLOAD_BYTES = 49 * 1024 * 1024
#: Telegram's caption limit for media messages.
CAPTION_LIMIT = 1024

#: One send at a time inside this process: the dispatcher's worker and the
#: watch retry timer both call send_clip, and an upload takes ~a minute.
_SEND_LOCK = threading.Lock()


@dataclass(frozen=True)
class TelegramTarget:
    token: str
    chat_id: str
    #: Where the credential came from, for logs: "env" or "openclaw".
    source: str

    def redact(self, text: str) -> str:
        return text.replace(self.token, "<token>") if self.token else text


# ------------------------------------------------------------- credentials

def _read_openclaw(config_path: Path) -> tuple[str | None, str | None]:
    """Token and chat id as the OpenClaw gateway has them, or Nones.

    The chat id comes from the gateway's DM allow-list, and only when it
    names exactly one person: with two, sending clips to "the" operator
    would be a guess about whose phone this is.
    """
    path = Path(config_path).expanduser()
    token = chat = None
    try:
        tg = json.loads(path.read_text(encoding="utf-8"))["channels"]["telegram"]
        accounts = tg.get("accounts") or {}
        token = ((accounts.get("default") or {}).get("botToken")
                 or tg.get("botToken"))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        pass
    try:
        allow = json.loads((path.parent / "credentials" /
                            "telegram-allowFrom.json").read_text(encoding="utf-8"))
        ids = [str(i) for i in (allow.get("allowFrom") or [])]
        if len(ids) == 1:
            chat = ids[0]
        elif ids:
            log.warning("notify.ambiguous_chat", candidates=len(ids),
                        note="set CLIPFORGE_TELEGRAM_CHAT_ID to choose one")
    except (OSError, ValueError, AttributeError):
        pass
    return token or None, chat


def resolve_target(openclaw_config: Path, *,
                   token: str | None = None,
                   chat_id: str | None = None) -> TelegramTarget | None:
    """Explicit values (from .env via Secrets) win; OpenClaw fills gaps."""
    source = "env"
    if not (token and chat_id):
        oc_token, oc_chat = _read_openclaw(openclaw_config)
        if not token and oc_token:
            token, source = oc_token, "openclaw"
        if not chat_id and oc_chat:
            chat_id = oc_chat
            source = "openclaw" if source == "openclaw" else "env+openclaw"
    if not (token and chat_id):
        return None
    return TelegramTarget(token=str(token), chat_id=str(chat_id), source=source)


def target_from_config(cfg: Any) -> TelegramTarget | None:
    from clipforge.config import Secrets

    secrets = Secrets()
    return resolve_target(cfg.notify.openclaw_config,
                          token=secrets.telegram_bot_token,
                          chat_id=secrets.telegram_chat_id)


# ----------------------------------------------------------------- caption

def caption_for(clip: Path) -> str:
    """Title, caption and hashtags from the clip's export pack, if any."""
    clip = Path(clip)
    parts: list[str] = []
    try:
        pack = json.loads(clip.with_suffix(".export.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pack = {}
    title = str(pack.get("title") or "").strip()
    body = str(pack.get("caption") or "").strip()
    tags = " ".join(f"#{str(t).lstrip('#')}" for t in (pack.get("hashtags") or []))
    for piece in (title, body if body != title else "", tags):
        if piece:
            parts.append(piece)
    parts.append(clip.name)
    text = "\n\n".join(parts)
    if len(text) > CAPTION_LIMIT:
        text = text[:CAPTION_LIMIT - 1].rstrip() + "…"
    return text


# -------------------------------------------------------------------- send

Post = Callable[..., Any]


def _default_post(*args: Any, **kwargs: Any) -> Any:
    import requests

    return requests.post(*args, **kwargs)


def _marker(clip: Path) -> Path:
    return Path(clip).with_suffix(".telegram.json")


def _outbox(ws_root: Path) -> Path:
    return Path(ws_root) / "outbox" / "telegram"


def _entry(ws_root: Path, clip: Path) -> Path:
    return _outbox(ws_root) / f"{Path(clip).stem}.json"


def _queue(ws_root: Path, clip: Path, caption: str, error: str, *,
           state: str = "failed", bump: bool = False) -> None:
    """Record this clip as owed to the phone. Atomic; keeps the attempt count.

    Written BEFORE the upload as well as after a failure: a kill between
    "QA passed" and the delivery marker used to lose the clip silently,
    because nothing but a failure ever wrote the entry. A duplicate after
    a crash mid-upload is the better error.
    """
    entry = _entry(ws_root, clip)
    attempts = 0
    try:
        attempts = int(json.loads(entry.read_text(encoding="utf-8")).get("attempts", 0))
    except (OSError, ValueError):
        pass
    if bump:
        # Counted once per delivery ATTEMPT, at the write-ahead. Counting
        # again on the failure that ends the same attempt would double
        # every number the operator reads.
        attempts += 1
    entry.parent.mkdir(parents=True, exist_ok=True)
    tmp = entry.with_suffix(".json.partial")
    tmp.write_text(json.dumps({
        "clip": str(Path(clip).resolve()), "caption": caption,
        "error": error[:500], "attempts": attempts, "state": state,
        "pid": os.getpid(), "queued_at": time.time()}, indent=2),
        encoding="utf-8")
    tmp.replace(entry)


@contextlib.contextmanager
def _claim(ws_root: Path, clip: Path):
    """Exclusive right to send THIS clip, across processes.

    `bta telegram --retry` is documented as safe to run while `bta watch`
    is going, and both walk the same outbox. Without a claim each would
    see no marker, and the phone would get every clip twice — at ~57 s of
    uplink each. An OS lock, like the workspace lock, because the kernel
    drops it even on a hard kill; a lock FILE with a staleness rule would
    not survive this machine's power-offs.
    """
    box = _outbox(ws_root)
    box.mkdir(parents=True, exist_ok=True)
    lock_path = box / f"{Path(clip).stem}.lock"
    handle = open(lock_path, "a+b")
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            yield False
            return
        try:
            yield True
        finally:
            with contextlib.suppress(OSError):
                if sys.platform == "win32":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _chat_refusal(target: TelegramTarget, post: Post) -> str | None:
    """Why the chat will not take a message right now, or None if it will.

    A ``sendChatAction`` costs a few hundred bytes and fails exactly the
    way a ``sendVideo`` would (blocked bot, unknown chat, revoked token).
    Measured 2026-09-17: a 26 MB clip took 57 s to upload only to be told
    "bot was blocked by the user"; without this check the retry timer
    would repeat that upload every ten minutes for as long as the block
    lasts. It also shows "sending video..." on the phone while uploading.
    """
    try:
        body = post(f"{API}/bot{target.token}/sendChatAction",
                    data={"chat_id": target.chat_id, "action": "upload_video"},
                    timeout=30).json()
    except Exception as exc:  # noqa: BLE001
        return target.redact(f"{type(exc).__name__}: {exc}")
    if body.get("ok"):
        return None
    return target.redact(f"{body.get('error_code')}: {body.get('description')}")


def send_clip(clip: Path, *, target: TelegramTarget | None, ws_root: Path,
              caption: str | None = None, again: bool = False,
              post: Post = _default_post) -> str:
    """Deliver one clip. Returns an outcome word; never raises.

    ``sent`` delivered · ``already_sent`` marker present · ``queued`` failed
    and kept for retry · ``in_flight`` someone else is sending it right
    now · ``missing`` the file is gone.
    """
    with _SEND_LOCK:
        with _claim(ws_root, clip) as mine:
            if not mine:
                log.info("notify.in_flight", clip=Path(clip).name,
                         note="another process is sending this clip")
                return "in_flight"
            return _send_claimed(clip, target=target, ws_root=ws_root,
                                 caption=caption, again=again, post=post)


def _send_claimed(clip: Path, *, target: TelegramTarget | None, ws_root: Path,
                  caption: str | None, again: bool, post: Post) -> str:
    """The body of :func:`send_clip`, with this clip's claim held."""
    clip = Path(clip)
    try:
        if not clip.is_file():
            return "missing"
        if _marker(clip).exists() and not again:
            return "already_sent"
        text = caption if caption is not None else caption_for(clip)
        if target is None:
            _queue(ws_root, clip, text, "no Telegram credentials found")
            log.error("notify.unconfigured", clip=clip.name,
                      note="set CLIPFORGE_TELEGRAM_BOT_TOKEN and "
                           "CLIPFORGE_TELEGRAM_CHAT_ID, or configure the "
                           "OpenClaw Telegram channel; queued for retry")
            return "queued"

        # Owed to the phone from here on, whatever happens next.
        _queue(ws_root, clip, text, "delivery in flight", state="sending",
               bump=True)

        refusal = _chat_refusal(target, post)
        if refusal is not None:
            _queue(ws_root, clip, text, refusal)
            hint = (" - open the bot in Telegram and tap Unblock/Start"
                    if "blocked" in refusal.lower() else "")
            log.warning("notify.chat_unreachable", clip=clip.name,
                        error=refusal[:300], note=f"queued, not uploaded{hint}")
            return "queued"

        size = clip.stat().st_size
        if size > MAX_UPLOAD_BYTES:
            # Telegram will not take it. Say so on the phone rather than
            # silently skipping: the operator still learns a clip exists.
            method, data, files = "sendMessage", {
                "chat_id": target.chat_id,
                "text": (f"New clip is {size / 1024 ** 2:.0f} MB, over "
                         f"Telegram's 50 MB bot limit. It is on the PC:\n"
                         f"{clip}\n\n{text}")[:4096]}, None
        else:
            method, data, files = "sendVideo", {
                "chat_id": target.chat_id, "caption": text,
                "supports_streaming": "true"}, True

        try:
            if files:
                with clip.open("rb") as fh:
                    resp = post(f"{API}/bot{target.token}/{method}", data=data,
                                files={"video": (clip.name, fh, "video/mp4")},
                                # One number, not (connect, read): urllib3
                                # applies the CONNECT timeout to writing the
                                # body, so (15, 600) aborted a 26 MB upload
                                # on a slow uplink after 15 s of stall.
                                timeout=600)
            else:
                resp = post(f"{API}/bot{target.token}/{method}", data=data,
                            timeout=(15, 60))
            body = resp.json()
        except Exception as exc:  # noqa: BLE001 - network, TLS, bad JSON
            err = target.redact(f"{type(exc).__name__}: {exc}")
            _queue(ws_root, clip, text, err)
            log.warning("notify.send_failed", clip=clip.name, error=err[:300],
                        note="queued; `bta watch` retries it")
            return "queued"

        if not body.get("ok"):
            err = target.redact(f"{body.get('error_code')}: "
                                f"{body.get('description')}")
            _queue(ws_root, clip, text, err)
            log.warning("notify.rejected", clip=clip.name, error=err[:300])
            return "queued"

        result = body.get("result") or {}
        _marker(clip).write_text(json.dumps({
            "sent_at": time.time(), "method": method,
            "message_id": result.get("message_id"),
            "source": target.source}, indent=2), encoding="utf-8")
        _entry(ws_root, clip).unlink(missing_ok=True)
        log.info("notify.sent", clip=clip.name, method=method,
                 size_mb=round(size / 1024 ** 2, 1), source=target.source)
        return "sent"
    except Exception as exc:  # noqa: BLE001 - delivery never sinks a clip
        log.error("notify.crashed", clip=clip.name,
                  error=(target.redact(f"{type(exc).__name__}: {exc}")
                         if target else f"{type(exc).__name__}: {exc}")[:300])
        return "queued"


def flush_outbox(*, target: TelegramTarget | None, ws_root: Path,
                 post: Post = _default_post) -> dict[str, int]:
    """Retry every queued delivery. Entries whose clip is gone are dropped."""
    counts: dict[str, int] = {}
    box = _outbox(ws_root)
    if not box.is_dir():
        return counts
    for entry in sorted(box.glob("*.json")):
        try:
            item = json.loads(entry.read_text(encoding="utf-8"))
            clip = Path(item["clip"])
        except (OSError, ValueError, KeyError):
            entry.unlink(missing_ok=True)
            continue
        outcome = send_clip(clip, target=target, ws_root=ws_root,
                            caption=item.get("caption"), post=post)
        if outcome in ("missing", "already_sent"):
            entry.unlink(missing_ok=True)
        counts[outcome] = counts.get(outcome, 0) + 1
    if counts:
        log.info("notify.outbox_flushed", **counts)
    return counts


def check(target: TelegramTarget | None, *, get: Callable[..., Any] | None = None) -> str:
    """Validate the credential with getMe; returns the bot's @username."""
    if target is None:
        raise ValueError("no Telegram credentials found")
    if get is None:
        import requests

        get = requests.get
    try:
        body = get(f"{API}/bot{target.token}/getMe", timeout=15).json()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(target.redact(f"{type(exc).__name__}: {exc}")) from None
    if not body.get("ok"):
        raise ValueError(f"Telegram refused the token: {body.get('description')}")
    return "@" + str((body.get("result") or {}).get("username"))
