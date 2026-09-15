"""Remote access for the control API — who may reach it, and from where.

`bta web` used to bind 127.0.0.1 and that was the whole security model:
nothing else could connect, so nothing else needed checking. The moment it
binds anything wider, that model is gone. This API spawns CLI subprocesses
(`generate`, `process`, `post`) on the host machine, so an unauthenticated
listener on 0.0.0.0 is remote code execution for every device on the
network — including the guest wifi and, behind a tunnel, the internet.

So the rule this module enforces is:

    Loopback stays open. Anything else needs the token.

Three ways to present it, because three different clients need it:

* ``Authorization: Bearer <token>`` — scripts and the MCP bridge.
* ``?t=<token>`` in the URL — the one-shot link you paste onto a phone.
* the ``bta_access`` cookie — set once by /login, so the dashboard's own
  fetches carry it without every call being rewritten.

And because typing a 43-character token on a phone keyboard is how people
end up disabling auth instead, :class:`PairingCodes` issues a short code
that is worth exactly one exchange: type six digits into the login page,
get the cookie, the code dies. It is guessable in a way the token is not,
so it expires fast and burns after a handful of wrong guesses.

The authorization decision is a pure function of (client address,
presented credentials, policy). It is not a middleware, not a FastAPI
dependency, and it touches no globals — because the one thing that must
never be true of this file is "it looked right and nobody could test it".
"""

from __future__ import annotations

import ipaddress
import os
import re
import secrets
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from clipforge.log import get_logger

log = get_logger(__name__)

#: Cookie the login exchange sets. Named, not generic, so it cannot
#: collide with whatever else is on localhost during development.
COOKIE_NAME = "bta_access"
#: Header a script can send instead of Authorization.
TOKEN_HEADER = "x-bta-token"
#: Query parameter for the paste-once link.
TOKEN_QUERY = "t"

#: Environment variables the CLI uses to hand policy to the server
#: process. uvicorn imports the app by string, so the policy cannot be
#: passed as an argument — it travels here.
ENV_TOKEN = "BTA_WEB_TOKEN"
ENV_REQUIRE_AUTH = "BTA_WEB_REQUIRE_AUTH"
ENV_TRUST_LOOPBACK = "BTA_WEB_TRUST_LOOPBACK"
ENV_PUBLIC_URL = "BTA_WEB_PUBLIC_URL"


# ------------------------------------------------------------------ token

def token_path(workspace_root: Path | str) -> Path:
    """Where the access token lives — inside the workspace, not the repo.

    The workspace is already the thing that holds clips, state and saved
    platform sessions; a token belongs with them, and keeping it out of
    the source tree means it cannot be committed by accident.
    """
    return Path(workspace_root) / "access_token"


def load_or_create_token(workspace_root: Path | str) -> str:
    """Return the persisted access token, creating one on first use.

    Persisted rather than regenerated per run: a phone that paired
    yesterday should still work today, and a token that changes on every
    restart trains the operator to turn auth off.
    """
    path = token_path(workspace_root)
    try:
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    except (FileNotFoundError, OSError):
        pass
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    _restrict(path)
    log.info("remote.token_created", path=str(path))
    return token


def rotate_token(workspace_root: Path | str) -> str:
    """Replace the token, invalidating every paired device."""
    path = token_path(workspace_root)
    token = secrets.token_urlsafe(32)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n", encoding="utf-8")
    _restrict(path)
    log.info("remote.token_rotated", path=str(path))
    return token


def _restrict(path: Path) -> None:
    """Best-effort owner-only permissions.

    On POSIX this is 0600. On Windows chmod only moves the read-only bit,
    so it is close to meaningless — the real protection there is that the
    file sits in the operator's own profile. Attempted anyway rather than
    skipped, and never fatal: a token that could not be locked down is
    still better than no token.
    """
    try:
        os.chmod(path, 0o600)
    except OSError as exc:  # noqa: BLE001 - advisory only
        log.debug("remote.chmod_failed", path=str(path), error=str(exc))


# ------------------------------------------------------------- addresses

def is_loopback(host: str | None) -> bool:
    """True when a client address is this machine talking to itself.

    Handles the three spellings a server actually sees: dotted IPv4,
    ``::1``, and the IPv4-mapped ``::ffff:127.0.0.1`` that a dual-stack
    listener reports. Anything unparseable is NOT loopback — an address
    this function cannot understand is one it must not vouch for.
    """
    if not host:
        return False
    raw = host.strip()
    if raw.lower() in ("localhost", "::1"):
        return True
    # A bracketed or port-suffixed form can reach here from headers.
    if raw.startswith("[") and "]" in raw:
        raw = raw[1:raw.index("]")]
    elif raw.count(":") == 1 and "." in raw:
        raw = raw.split(":", 1)[0]
    try:
        addr = ipaddress.ip_address(raw)
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return addr.is_loopback


#: Extra Host values the operator vouches for, comma separated. Needed
#: when the dashboard is reached by a name this module cannot derive —
#: an mDNS alias, a hosts-file entry, a reverse proxy's own vhost.
ENV_ALLOWED_HOSTS = "BTA_WEB_ALLOWED_HOSTS"

#: Host suffixes that belong to the two remote transports this tool
#: offers. Both are per-machine names nobody else can point at us.
_HOST_SUFFIXES = (".ts.net", ".trycloudflare.com")


def normalise_host(value: str | None) -> str:
    """The bare hostname from a Host header: no port, no brackets, lower."""
    raw = (value or "").strip().lower()
    if not raw:
        return ""
    if raw.startswith("[") and "]" in raw:           # [::1]:8011
        return raw[1:raw.index("]")]
    if raw.count(":") == 1:                          # host:8011
        return raw.split(":", 1)[0]
    return raw                                       # bare IPv6 or name


def allowed_hosts(extra: str | None = None) -> set[str]:
    """Every Host value this server legitimately answers to.

    Names, not addresses, are the rebinding vector: an attacker page on
    ``evil.com`` can make its own DNS answer 127.0.0.1 and then reach a
    loopback server from the victim's browser as same-origin — CORS never
    enters into it, because the browser believes it IS the origin. What
    the attacker cannot forge is the Host header, so pinning the names we
    answer to is the fix.

    IP literals are handled separately by :func:`host_is_allowed` (any
    literal is fine: reaching us by address means no DNS was involved).
    """
    names = {"localhost", "localhost.localdomain"}
    try:
        own = socket.gethostname().strip().lower()
    except OSError:
        own = ""
    if own:
        names.add(own)
        names.add(f"{own}.local")
        names.add(own.split(".", 1)[0])
    for source in (extra, os.environ.get(ENV_ALLOWED_HOSTS)):
        for item in (source or "").replace(";", ",").split(","):
            name = normalise_host(item)
            if name:
                names.add(name)
    public = normalise_host(os.environ.get(ENV_PUBLIC_URL, "").split("//")[-1])
    if public:
        names.add(public)
    return names


def host_is_allowed(value: str | None, *, extra: str | None = None) -> bool:
    """Is this Host header one this server should answer to?

    Permissive in the two ways that cost nothing and strict in the one
    that matters:

    * any IP literal passes — a browser that got here by address did no
      DNS lookup, so there was nothing to rebind;
    * the tailnet and quick-tunnel suffixes pass, because those names are
      issued to this machine and cannot be aimed elsewhere;
    * an absent Host passes, because a non-browser client (curl -H, a
      script, an HTTP/1.0 probe) is not the thing being defended against
      and already needs the token;
    * every other DNS name must be one we derived or the operator listed.
    """
    host = normalise_host(value)
    if not host:
        return True
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        pass
    if any(host.endswith(suffix) for suffix in _HOST_SUFFIXES):
        return True
    return host in allowed_hosts(extra)


def is_private(host: str | None) -> bool:
    """True for RFC1918 / link-local / CGNAT (Tailscale) addresses."""
    if not host:
        return False
    try:
        addr = ipaddress.ip_address(host.strip())
    except ValueError:
        return False
    mapped = getattr(addr, "ipv4_mapped", None)
    if mapped is not None:
        addr = mapped
    return bool(addr.is_private or addr.is_link_local)


def lan_addresses() -> list[str]:
    """Every local IPv4 a phone on this network could plausibly reach.

    Two sources, because either alone misses cases: a UDP socket
    "connected" to a public address reveals the interface the OS routes
    through (no packet is sent), and the hostname lookup catches the
    other adapters. Loopback is excluded — it is never the answer to
    "what do I type on my phone".
    """
    found: list[str] = []

    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # 203.0.113.0/24 is TEST-NET-3: routable-looking, never routed.
        probe.connect(("203.0.113.1", 9))
        found.append(probe.getsockname()[0])
    except OSError:
        pass
    finally:
        probe.close()

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None,
                                       socket.AF_INET):
            found.append(info[4][0])
    except (OSError, socket.gaierror):
        pass

    out: list[str] = []
    for addr in found:
        if addr in out or is_loopback(addr):
            continue
        try:
            parsed = ipaddress.ip_address(addr)
        except ValueError:
            continue
        # 169.254.x.x is what an adapter falls back to when DHCP failed.
        # It is technically "private" and never once the answer to "what
        # do I type on my phone", so offering it is offering a dead link.
        if parsed.is_link_local or not parsed.is_private:
            continue
        out.append(addr)
    # The routed interface came first (the UDP probe) and stays first —
    # that is the address a phone on the same wifi will actually reach,
    # and the rest are usually virtual adapters (WSL, Hyper-V, VPNs).
    return out


def tailscale_addresses() -> list[str]:
    """Tailnet IPs, if Tailscale is installed and up.

    This is the honest answer to "from anywhere": a tailnet address works
    from any device signed into the same account, over an encrypted link,
    with nothing exposed to the public internet. It is checked before any
    tunnel is offered because it is strictly the safer option.
    """
    import shutil

    exe = shutil.which("tailscale")
    if not exe:
        return []
    try:
        proc = subprocess.run([exe, "ip", "-4"], capture_output=True,
                              text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return []
    if proc.returncode != 0:
        return []
    out = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        try:
            ipaddress.ip_address(line)
        except ValueError:
            continue
        out.append(line)
    return out


def access_urls(port: int, token: str | None = None, *,
                include_tailscale: bool = True) -> dict[str, list[str]]:
    """Every URL this server is reachable at, grouped by how far it goes.

    The token is embedded when given, because the whole point of the
    ``local`` / ``lan`` lists is that they are pasted or scanned onto
    another device, and a URL that then demands a separate secret is a
    URL that gets abandoned halfway.
    """
    suffix = f"/?{TOKEN_QUERY}={token}" if token else "/"

    def _url(host: str) -> str:
        bracket = f"[{host}]" if ":" in host else host
        return f"http://{bracket}:{port}{suffix}"

    urls = {
        "local": [_url("127.0.0.1")],
        "lan": [_url(a) for a in lan_addresses()],
        "tailscale": ([_url(a) for a in tailscale_addresses()]
                      if include_tailscale else []),
    }
    return urls


# ------------------------------------------------------------- pairing

@dataclass
class _Code:
    code: str
    expires_at: float
    attempts_left: int


class PairingCodes:
    """Short-lived, single-use codes that trade up for the real token.

    Six digits is 10^6, which is only unguessable because of what is
    around it: a code lives for minutes, dies on redemption, and burns
    after a few wrong tries. Those three limits are the security here —
    the digit count is a UX choice, and stating that plainly is better
    than implying six digits are strong.
    """

    #: Long enough to walk to another room, short enough that an
    #: abandoned code is not left standing all afternoon.
    TTL_S = 600.0
    #: Wrong guesses before the code is destroyed. Deliberately small.
    MAX_ATTEMPTS = 5

    def __init__(self, *, ttl_s: float | None = None,
                 max_attempts: int | None = None) -> None:
        self._codes: dict[str, _Code] = {}
        self.ttl_s = float(ttl_s if ttl_s is not None else self.TTL_S)
        self.max_attempts = int(max_attempts if max_attempts is not None
                                else self.MAX_ATTEMPTS)

    def _now(self) -> float:
        return time.monotonic()

    def _sweep(self) -> None:
        now = self._now()
        for code in [c for c, v in self._codes.items() if v.expires_at <= now]:
            self._codes.pop(code, None)

    def issue(self) -> tuple[str, float]:
        """Mint a code. Returns (code, seconds_valid)."""
        self._sweep()
        # Only one code outstanding at a time: two live codes double the
        # guessing surface for no benefit, since a second device can
        # simply ask for a second code after the first is used.
        self._codes.clear()
        code = f"{secrets.randbelow(1_000_000):06d}"
        self._codes[code] = _Code(code, self._now() + self.ttl_s,
                                  self.max_attempts)
        log.info("remote.pairing_issued", ttl_s=self.ttl_s)
        return code, self.ttl_s

    def redeem(self, candidate: str) -> bool:
        """Consume a code. True only for a live, correct, unused code."""
        self._sweep()
        cleaned = re.sub(r"\D", "", candidate or "")
        if not self._codes:
            return False
        # Compare against every live code in constant time, then decide,
        # so a wrong guess and a right guess take the same path.
        hit: _Code | None = None
        for entry in self._codes.values():
            if secrets.compare_digest(entry.code, cleaned):
                hit = entry
        if hit is not None:
            self._codes.pop(hit.code, None)
            log.info("remote.pairing_redeemed")
            return True
        for entry in list(self._codes.values()):
            entry.attempts_left -= 1
            if entry.attempts_left <= 0:
                self._codes.pop(entry.code, None)
                log.warning("remote.pairing_burned",
                            reason="too many wrong attempts")
        return False

    def active(self) -> bool:
        self._sweep()
        return bool(self._codes)

    def clear(self) -> None:
        self._codes.clear()


# -------------------------------------------------------------- policy

@dataclass(frozen=True)
class AccessPolicy:
    """What this server requires of a caller.

    ``token=None`` means auth is off, which is only ever legitimate when
    the server is bound to loopback. The CLI refuses the other
    combination; this dataclass records it rather than deciding it.
    """

    token: str | None = None
    trust_loopback: bool = True
    #: Set when the operator explicitly asked for an open network bind.
    #: Nothing here reads it — it exists so /api/access can report the
    #: truth about the machine's exposure instead of inferring it.
    insecure: bool = False

    @property
    def enforced(self) -> bool:
        return bool(self.token)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "AccessPolicy":
        src = os.environ if env is None else env
        token = (src.get(ENV_TOKEN) or "").strip() or None
        require = (src.get(ENV_REQUIRE_AUTH) or "").strip().lower()
        trust = (src.get(ENV_TRUST_LOOPBACK) or "1").strip().lower()
        if require in ("0", "false", "no", "off"):
            token = None
        return cls(token=token,
                   trust_loopback=trust not in ("0", "false", "no", "off"),
                   insecure=(require in ("0", "false", "no", "off")))


@dataclass(frozen=True)
class Decision:
    """The outcome of one authorization check."""

    allowed: bool
    reason: str
    #: True when the caller proved itself with something that is not yet
    #: a cookie (a URL token, a header, a pairing code) — the server
    #: should persist it so the next request needs no query string.
    set_cookie: bool = False
    #: Credential to write into that cookie. Never the pairing code.
    cookie_value: str | None = None


#: Request headers that mean "something forwarded this to me". Any one of
#: them present proves the peer address is a proxy's, not the real
#: client's — see :attr:`Presented.forwarded` for why that voids loopback
#: trust.
FORWARD_HEADERS = ("x-forwarded-for", "x-real-ip", "forwarded",
                   "cf-connecting-ip", "x-forwarded-host")


@dataclass(frozen=True)
class Presented:
    """Whatever credential material arrived with a request."""

    client_host: str | None = None
    header_token: str | None = None
    cookie_token: str | None = None
    query_token: str | None = None
    pairing_code: str | None = None
    #: True when the request carried proxy-forwarding headers. A request
    #: that reached us through a proxy has the PROXY's address as its
    #: peer, so 127.0.0.1 no longer means "this machine" — it means "the
    #: proxy runs here". `cloudflared tunnel --url http://127.0.0.1:PORT`
    #: is exactly that shape, which is how a public URL was inheriting
    #: loopback trust and reaching a subprocess-spawning API with no
    #: token at all. Fails closed: a direct local client never sends
    #: these, so refusing them costs nothing and closes the hole even
    #: where the operator put their own reverse proxy in front.
    forwarded: bool = False
    #: The Host header, minus any port. Checked against the names this
    #: server answers to so a DNS-rebinding page cannot drive it.
    host_header: str | None = None


def _matches(candidate: str | None, token: str) -> bool:
    if not candidate:
        return False
    return secrets.compare_digest(candidate.strip(), token)


def decide(presented: Presented, policy: AccessPolicy, *,
           pairing: PairingCodes | None = None) -> Decision:
    """Authorize one request. Pure — same inputs, same answer, always.

    Order matters and is deliberate:

    1. Auth off → allow, and say so, so the caller can surface it.
    2. Loopback → allow when trusted AND nothing forwarded the request.
       This keeps `bta web` on a laptop exactly as it was, which is why
       the token can be mandatory everywhere else without anyone wanting
       it turned off. The forwarding check is what keeps a tunnel or a
       reverse proxy — both of which connect FROM 127.0.0.1 — from
       inheriting that trust on behalf of the whole internet.
    3. A real token in any of its three carriers → allow.
    4. A live pairing code → allow ONCE and hand back the token to store.
    5. Everything else → refuse, with the reason the operator needs.
    """
    if not policy.enforced:
        return Decision(True, "auth disabled")

    token = policy.token or ""

    if (policy.trust_loopback and is_loopback(presented.client_host)
            and not presented.forwarded):
        return Decision(True, "loopback")

    if _matches(presented.cookie_token, token):
        return Decision(True, "cookie")
    if _matches(presented.header_token, token):
        return Decision(True, "header")
    if _matches(presented.query_token, token):
        # Arrived in a URL: store it so the next request is a normal one
        # and the token stops travelling in links and history.
        return Decision(True, "query token", set_cookie=True,
                        cookie_value=token)

    if presented.pairing_code and pairing is not None:
        if pairing.redeem(presented.pairing_code):
            return Decision(True, "pairing code", set_cookie=True,
                            cookie_value=token)
        return Decision(False, "that pairing code is wrong or expired")

    if presented.cookie_token or presented.header_token or presented.query_token:
        return Decision(False, "the access token is not valid for this "
                               "machine — it may have been rotated")
    return Decision(False, "this device is not paired with BTA")


# --------------------------------------------------------------- tunnel

#: cloudflared prints its assigned hostname once, in a banner, on stderr.
_TRYCF = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")


def parse_tunnel_url(line: str) -> str | None:
    """Pull a quick-tunnel URL out of one line of cloudflared output."""
    hit = _TRYCF.search(line or "")
    return hit.group(0) if hit else None


@dataclass
class Tunnel:
    """A running `cloudflared` quick tunnel."""

    process: subprocess.Popen
    url: str | None = None
    lines: list[str] = field(default_factory=list)

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()


def start_cloudflare_tunnel(port: int, *, timeout_s: float = 30.0,
                            popen=subprocess.Popen) -> Tunnel:
    """Expose ``port`` on a public https URL via a cloudflared quick tunnel.

    This is the "anywhere" that does not need Tailscale on both ends, and
    it is also the only option here that puts the control API on the
    public internet. The CLI will not start one without auth enforced;
    that check lives there because it is a policy decision, not a
    transport one.

    Raises FileNotFoundError with an actionable message when cloudflared
    is absent, rather than returning a Tunnel with url=None that the
    caller has to interpret.
    """
    import shutil

    exe = shutil.which("cloudflared")
    if not exe:
        raise FileNotFoundError(
            "cloudflared is not installed. Get it from "
            "https://developers.cloudflare.com/cloudflare-one/connections/"
            "connect-networks/downloads/ (or `winget install "
            "--id Cloudflare.cloudflared`), then run this again.")

    proc = popen([exe, "tunnel", "--url", f"http://127.0.0.1:{port}"],
                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                 bufsize=1)
    tunnel = Tunnel(process=proc)

    deadline = time.monotonic() + timeout_s
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                break
            continue
        tunnel.lines.append(line.rstrip())
        url = parse_tunnel_url(line)
        if url:
            tunnel.url = url
            break

    if not tunnel.url:
        tunnel.stop()
        tail = "\n".join(tunnel.lines[-8:]) or "(no output)"
        raise RuntimeError(
            f"cloudflared did not report a tunnel URL within {timeout_s:.0f}s."
            f"\n{tail}")

    # Keep draining, or the pipe fills and cloudflared blocks on write.
    import threading

    def _drain() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            tunnel.lines.append(line.rstrip())
            if len(tunnel.lines) > 200:
                del tunnel.lines[:100]

    threading.Thread(target=_drain, daemon=True).start()
    log.info("remote.tunnel_up", url=tunnel.url)
    return tunnel
