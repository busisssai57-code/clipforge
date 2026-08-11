"""MCP server — drive this pipeline from Claude, Cursor or any MCP client.

Transport is stdio JSON-RPC 2.0, implemented here against the standard
library rather than the `mcp` SDK. That is deliberate: the setup story on
the website is "point your client at `bta mcp`", and a server that needs
its own pip install before that command works is a worse story than 200
lines of protocol. There is nothing exotic in stdio MCP — newline
delimited JSON in, newline delimited JSON out.

**Every tool goes through the local HTTP API.** It would be faster to
import the stages directly, but that would be a second way to run the
pipeline, and a second way drifts from the first. `bta web` is already the
supervised path with task tracking, cancellation and logs; this is a thin
client of it. If the server is not running, tools say so and name the
command that fixes it rather than failing obscurely.

**stdout is protocol.** Anything written there that is not a JSON-RPC
message corrupts the session, so all diagnostics go to stderr. This is
the single easiest way to break an MCP server and it fails silently from
the client's point of view.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

#: Protocol revisions this server knows how to speak. The client's
#: requested version is echoed back when we recognise it, because a client
#: that asked for an older revision should not be handed a newer one.
SUPPORTED_PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")
DEFAULT_PROTOCOL = "2025-06-18"

SERVER_NAME = "bta"
SERVER_VERSION = "1.0.0"


class ApiDown(Exception):
    """The local pipeline server is not answering."""


# ---------------------------------------------------------------------------
# HTTP client for the local API
# ---------------------------------------------------------------------------

class Api:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def call(self, path: str, payload: dict | None = None,
             timeout: float = 30.0) -> Any:
        url = self.base + path
        data = None
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers,
                                     method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            raise RuntimeError(f"{path} returned HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise ApiDown(
                f"No pipeline at {self.base}. Start it with "
                f"`bta web --port {self.base.rsplit(':', 1)[-1]}` and try again."
            ) from exc
        return json.loads(body) if body.strip() else None


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------

def _tools() -> list[dict[str, Any]]:
    """Tool definitions.

    Descriptions carry the routing information — when to reach for a tool,
    and what it will refuse to do. A client picks tools by reading these,
    so a vague description is a broken tool.
    """
    return [
        {
            "name": "bta_capabilities",
            "description": (
                "What this machine can ACTUALLY do right now, probed live: ffmpeg "
                "filters are looked up in the real filter list and model weights are "
                "checked on disk. Call this before promising the user a feature — "
                "each entry reports available, a blocker naming what is missing and "
                "what would fix it, and by_policy for things deliberately switched off."
            ),
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bta_health",
            "description": "Check that the local pipeline server is running. Returns its status and version.",
            "inputSchema": {"type": "object", "properties": {}},
        },
        {
            "name": "bta_list_clips",
            "description": (
                "List rendered clips on disk, newest first, with title, hook, caption, "
                "hashtags, duration, dimensions, loudness and score. Use this to answer "
                "questions about what has already been produced."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "Max clips to return (default 20).",
                              "minimum": 1, "maximum": 200},
                    "query": {"type": "string",
                              "description": "Case-insensitive filter over title, hook, caption and filename."},
                },
            },
        },
        {
            "name": "bta_clip_video",
            "description": (
                "Clip a LOCAL video file into ranked vertical shorts. Returns a task_id "
                "immediately — the render runs in the background, so poll bta_task_status. "
                "The path must already exist on this machine; nothing is uploaded."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source": {"type": "string", "description": "Absolute path to a video file on this machine."},
                    "clips": {"type": "integer", "description": "How many clips to produce (default 3).",
                              "minimum": 1, "maximum": 20},
                },
                "required": ["source"],
            },
        },
        {
            "name": "bta_grab_url",
            "description": (
                "Download a video by URL (anything yt-dlp supports) and clip it. Returns a "
                "task_id to poll with bta_task_status. Use bta_clip_video instead when the "
                "file is already on disk."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Video URL."},
                    "clips": {"type": "integer", "description": "How many clips to produce (default 3).",
                              "minimum": 1, "maximum": 20},
                },
                "required": ["url"],
            },
        },
        {
            "name": "bta_task_status",
            "description": (
                "Poll a background task started by bta_clip_video, bta_grab_url or "
                "bta_generate. Returns status (running/completed/failed) and the tail of "
                "its real log lines."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "task_id": {"type": "string"},
                    "lines": {"type": "integer", "description": "How many trailing log lines (default 20).",
                              "minimum": 1, "maximum": 200},
                },
                "required": ["task_id"],
            },
        },
        {
            "name": "bta_generate",
            "description": (
                "Generate an original short video from a text brief using local weights. "
                "Seeded, so the same brief renders the same bytes. Returns a task_id. "
                "Check bta_capabilities first — this needs text-to-video weights on disk."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "brief": {"type": "string", "description": "What the piece should be."},
                    "niche": {"type": "string",
                              "description": "Named look controlling style, grade, captions and pacing."},
                    "seed": {"type": "integer", "description": "Explicit seed for a reproducible render."},
                },
                "required": ["brief"],
            },
        },
        {
            "name": "bta_clip_detail",
            "description": (
                "Everything the pipeline recorded about one clip: the export pack copy "
                "(per-platform captions, hashtags, chapters), framing decisions, loudness "
                "measurements and the source timestamps it was cut from."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {"filename": {"type": "string", "description": "Clip filename from bta_list_clips."}},
                "required": ["filename"],
            },
        },
    ]


def _run_tool(api: Api, name: str, args: dict[str, Any]) -> str:
    """Execute one tool and return its result as text for the model."""
    if name == "bta_health":
        return json.dumps(api.call("/api/health"), indent=2)

    if name == "bta_capabilities":
        caps = api.call("/api/capabilities")
        lines = []
        for c in caps:
            mark = "LIVE" if c.get("available") else "BLOCKED"
            if c.get("available") and c.get("by_policy"):
                mark = "LIVE (engine chosen by policy)"
            detail = c.get("note") if c.get("available") else c.get("blocker")
            lines.append(f"{mark:<32} {c.get('label')}\n{'':<32} {detail}")
        return "\n".join(lines) or "No capabilities reported."

    if name == "bta_list_clips":
        clips = api.call("/api/clips", timeout=60.0) or []
        query = (args.get("query") or "").lower().strip()
        if query:
            def hit(c: dict) -> bool:
                hay = " ".join(str(c.get(k) or "") for k in
                               ("title", "hook", "caption", "filename", "stem")).lower()
                return all(w in hay for w in query.split())
            clips = [c for c in clips if hit(c)]
        limit = int(args.get("limit") or 20)
        clips = clips[:limit]
        if not clips:
            return "No clips match. Render some with bta_clip_video or bta_grab_url."
        out = []
        for c in clips:
            out.append(json.dumps({
                "filename": c.get("filename"),
                "title": c.get("title"),
                "hook": c.get("hook"),
                "duration_s": c.get("duration_s"),
                "size": f"{c.get('width')}x{c.get('height')}",
                "loudness_tp": c.get("loudness_tp"),
                "score": c.get("score"),
                "created_at": c.get("created_at"),
            }, indent=2))
        return "\n".join(out)

    if name == "bta_clip_video":
        src = args.get("source")
        if not src:
            return "source is required."
        res = api.call("/api/process", {"source": src, "clips": int(args.get("clips") or 3)})
        return (f"Started. task_id={res.get('task_id')}\n"
                f"Poll it with bta_task_status. Rendering happens on this machine's GPU.")

    if name == "bta_grab_url":
        url = args.get("url")
        if not url:
            return "url is required."
        res = api.call("/api/grab", {"url": url, "clips": int(args.get("clips") or 3)})
        return (f"Started. task_id={res.get('task_id')}\n"
                f"Poll it with bta_task_status.")

    if name == "bta_task_status":
        tid = args.get("task_id")
        if not tid:
            return "task_id is required."
        res = api.call(f"/api/tasks/{tid}/logs")
        tail = (res.get("lines") or [])[-int(args.get("lines") or 20):]
        return f"status: {res.get('status')}\n\n" + "\n".join(tail)

    if name == "bta_generate":
        payload: dict[str, Any] = {"brief": args["brief"]}
        if args.get("niche"):
            payload["niche"] = args["niche"]
        if args.get("seed") is not None:
            payload["seed"] = int(args["seed"])
        res = api.call("/api/generate", payload)
        return (f"Started. task_id={res.get('task_id')}\n"
                f"Poll it with bta_task_status.")

    if name == "bta_clip_detail":
        fn = args.get("filename")
        if not fn:
            return "filename is required."
        return json.dumps(api.call(f"/api/clips/{urllib.parse.quote(fn)}/detail"), indent=2)

    raise KeyError(f"unknown tool: {name}")


# ---------------------------------------------------------------------------
# JSON-RPC plumbing
# ---------------------------------------------------------------------------

def _result(msg_id: Any, payload: dict) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "result": payload}


def _error(msg_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": code, "message": message}}


def handle(msg: dict, api: Api) -> dict | None:
    """Route one JSON-RPC message. Returns None for notifications."""
    method = msg.get("method")
    msg_id = msg.get("id")
    params = msg.get("params") or {}

    if method == "initialize":
        asked = params.get("protocolVersion")
        version = asked if asked in SUPPORTED_PROTOCOLS else DEFAULT_PROTOCOL
        return _result(msg_id, {
            "protocolVersion": version,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": SERVER_VERSION},
            "instructions": (
                "BTA turns long video into ranked vertical clips, entirely on this "
                "machine. Call bta_capabilities before offering a feature — it probes "
                "what is actually installed instead of assuming. Renders are background "
                "tasks: the clip/grab/generate tools return a task_id to poll."
            ),
        })

    # Notifications carry no id and get no reply.
    if method in ("notifications/initialized", "initialized", "notifications/cancelled"):
        return None

    if method == "ping":
        return _result(msg_id, {})

    if method == "tools/list":
        return _result(msg_id, {"tools": _tools()})

    if method == "tools/call":
        name = params.get("name") or ""
        args = params.get("arguments") or {}
        try:
            text = _run_tool(api, name, args)
            is_error = False
        except ApiDown as exc:
            text, is_error = str(exc), True
        except KeyError as exc:
            return _error(msg_id, -32602, str(exc))
        except Exception as exc:  # noqa: BLE001 — surfaced to the model, not swallowed
            text, is_error = f"{type(exc).__name__}: {exc}", True
        return _result(msg_id, {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        })

    # Optional capabilities we do not implement — answer properly rather
    # than letting the client hang waiting for a reply that never comes.
    if method in ("resources/list", "prompts/list"):
        key = method.split("/")[0]
        return _result(msg_id, {key: []})

    if msg_id is None:
        return None
    return _error(msg_id, -32601, f"method not found: {method}")


def serve(base_url: str = "http://127.0.0.1:8765") -> int:
    """Read JSON-RPC from stdin, write replies to stdout, until EOF."""
    api = Api(base_url)
    # Binary-safe line reading; clients send UTF-8.
    stdin = sys.stdin
    out = sys.stdout

    print(f"[bta mcp] serving on stdio, pipeline at {base_url}", file=sys.stderr, flush=True)

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            out.write(json.dumps(_error(None, -32700, "parse error")) + "\n")
            out.flush()
            continue

        # A batch is a list; each element is handled independently.
        messages = msg if isinstance(msg, list) else [msg]
        replies = []
        for one in messages:
            try:
                reply = handle(one, api)
            except Exception as exc:  # noqa: BLE001
                reply = _error(one.get("id"), -32603, f"internal error: {exc}")
            if reply is not None:
                replies.append(reply)

        for reply in replies:
            out.write(json.dumps(reply) + "\n")
        if replies:
            out.flush()

    return 0
