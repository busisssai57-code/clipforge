"""Serve the BTA dashboard on the local network (phones, tablets).

Local-only by design: this binds a plain HTTP server to the LAN so another
device on the same network can open the dashboard. Nothing is uploaded and
nothing leaves the network — it is the same files the browser already reads
off disk, reachable over Wi-Fi instead of file://.

Why this exists rather than `python -m http.server`: iOS Safari will not
play a video unless the server answers HTTP Range requests with 206
Partial Content. The stdlib handler answers 200 with the whole file, so
clips show a black player with a slash through it on an iPhone. This adds
range support and advertises it.

    python tools/serve_dashboard.py [--port 8770] [--root workspace]

Ctrl-C to stop. While it runs, everything under --root is readable by any
device on your network, so stop it when you are done.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class _RangeFile:
    """File wrapper that yields at most ``length`` bytes to copyfileobj."""

    def __init__(self, fh, length: int) -> None:
        self._fh = fh
        self._left = length

    def read(self, size: int = -1) -> bytes:
        if self._left <= 0:
            return b""
        if size is None or size < 0:
            size = self._left
        chunk = self._fh.read(min(size, self._left))
        self._left -= len(chunk)
        return chunk

    def close(self) -> None:
        self._fh.close()


class RangeHandler(SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler + RFC 7233 single-range support."""

    def end_headers(self) -> None:
        # Advertise range support on EVERY response: Safari checks this
        # before it will even attempt to stream.
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def send_head(self):
        rng = self.headers.get("Range")
        path = self.translate_path(self.path)
        if not rng or os.path.isdir(path):
            return super().send_head()

        try:
            unit, _, spec = rng.partition("=")
            if unit.strip().lower() != "bytes" or "," in spec:
                raise ValueError("unsupported range unit or multi-range")
            size = os.path.getsize(path)
            first, _, last = spec.strip().partition("-")
            if first == "":                      # suffix: last N bytes
                start, end = max(0, size - int(last)), size - 1
            else:
                start = int(first)
                end = int(last) if last else size - 1
            end = min(end, size - 1)
            if start > end or start >= size:
                raise ValueError("unsatisfiable")
        except (ValueError, OSError):
            # Anything we cannot parse degrades to a normal 200 rather than
            # failing the request.
            return super().send_head()

        try:
            fh = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fh.seek(start)
        self.send_response(206)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Content-Length", str(end - start + 1))
        self.end_headers()
        return _RangeFile(fh, end - start + 1)

    def log_message(self, fmt, *args) -> None:      # quieter console
        if "206" in (args[1] if len(args) > 1 else ""):
            return
        super().log_message(fmt, *args)


def _lan_addresses() -> list[str]:
    """Best-effort list of addresses another device could reach."""
    found: list[str] = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))           # no packets sent
        found.append(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    for info in socket.getaddrinfo(socket.gethostname(), None,
                                   socket.AF_INET):
        ip = info[4][0]
        if not ip.startswith(("127.", "169.254.")) and ip not in found:
            found.append(ip)
    return found


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--root", default="workspace",
                    help="directory to serve (must contain dashboard.html)")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not (root / "dashboard.html").exists():
        print(f"no dashboard.html in {root} — run `bta process ...` first",
              file=sys.stderr)
        return 1

    handler = partial(RangeHandler, directory=str(root))
    try:
        httpd = ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    except OSError as exc:
        print(f"could not bind port {args.port}: {exc}\n"
              "Windows reserves some port ranges for Hyper-V/WSL; try "
              "--port 8771 or another free port.", file=sys.stderr)
        return 1

    print(f"serving {root} on port {args.port}")
    print("open one of these on your phone (same Wi-Fi):")
    for ip in _lan_addresses():
        print(f"    http://{ip}:{args.port}/dashboard.html")
    print("\nCtrl-C to stop. While this runs, anything under the served "
          "directory is readable by devices on your network.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
