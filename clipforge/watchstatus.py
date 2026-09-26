"""What `bta watch` is doing right now, for anything that is not it.

The watcher is meant to run unattended — started at logon, hidden, for
days. Before this, the only way to know whether it was recording, holding
clips for an idle machine, or dead since the last reboot was to read a
JSONL log. "Nothing arrived on my phone" and "nothing was on" look
identical from outside, which is the failure this file exists to end.

A heartbeat file, not a port: the dashboard, the CLI and a person with a
text editor can all read it, and it survives the watcher dying — with a
timestamp that says how long ago it stopped being true.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from clipforge.log import get_logger

log = get_logger(__name__)

#: Older than this and the watcher is presumed gone, not quiet. Six times
#: the default write cadence, so a slow sweep never reads as death.
STALE_AFTER_S = 90.0


def path_for(ws_root: Path) -> Path:
    return Path(ws_root) / "watch_status.json"


def write(ws_root: Path, **fields: Any) -> None:
    """Atomically record the current state. Never raises."""
    dest = path_for(ws_root)
    payload = {"updated_at": time.time(), "pid": os.getpid(), **fields}
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".json.partial")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True),
                       encoding="utf-8")
        tmp.replace(dest)
    except OSError as exc:
        log.debug("watchstatus.write_failed", error=str(exc)[:200])


def read(ws_root: Path) -> dict[str, Any]:
    """The last heartbeat, plus whether it is still true.

    ``running`` is the honest answer to "is the watcher up": a heartbeat
    from an hour ago means it died an hour ago, not that it is idle.
    """
    dest = path_for(ws_root)
    try:
        data = json.loads(dest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"running": False, "note": "watch has not run in this "
                                          "workspace, or never wrote a "
                                          "heartbeat"}
    age = time.time() - float(data.get("updated_at") or 0)
    data["age_s"] = round(age, 1)
    data["running"] = age <= STALE_AFTER_S
    if not data["running"]:
        data["note"] = (f"last heartbeat {age / 60:.0f} min ago - `bta watch` "
                        "is not running")
    return data


def clear(ws_root: Path) -> None:
    """Mark the watcher as stopped on a clean exit."""
    try:
        path_for(ws_root).unlink(missing_ok=True)
    except OSError:
        pass


def summarize(status: dict[str, Any]) -> str:
    """One line a person can read, for the CLI and the dashboard tile."""
    if not status.get("running"):
        return status.get("note", "not running")
    bits = [f"watching {status.get('channels', '?')} channel(s)"]
    live = status.get("live") or []
    bits.append(f"LIVE: {', '.join(live)}" if live else "none live")
    pending = status.get("queue_pending")
    if pending:
        bits.append(f"{pending} window(s) queued")
    gate = status.get("gate")
    bits.append(f"clipping held: {gate}" if gate else "clipping free to run")
    owed = status.get("telegram_outbox")
    if owed:
        bits.append(f"{owed} clip(s) owed to the phone")
    return " | ".join(bits)
