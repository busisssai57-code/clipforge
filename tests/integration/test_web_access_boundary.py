"""The control API's front door, exercised through the real ASGI stack.

test_remote_access.py pins the decision function; this pins the wiring
around it. Both are needed, because every bug this file was written for
lived in the gap between them — the policy was right and nothing fed it
the facts it needed.

Two holes are pinned here:

* **the tunnel bypass.** ``cloudflared tunnel --url http://127.0.0.1:PORT``
  connects to this server from loopback, so every request off the public
  internet arrived with ``request.client.host == "127.0.0.1"`` and was
  allowed by the loopback rule. The token was set and never consulted:
  the public URL was an unauthenticated shell onto an API whose whole
  job is spawning subprocesses.
* **DNS rebinding.** A page on an attacker-owned name whose DNS answers
  127.0.0.1 reaches a loopback server as *same-origin*, so CORS never
  runs and the browser sends the request happily. The Host header is the
  only part of it the attacker cannot choose.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clipforge import remote, web

TOKEN = "integration-token-long-enough-to-be-real"


@pytest.fixture()
def client(monkeypatch):
    """A server with auth enforced, as `--lan` and `--tunnel` leave it."""
    previous = web.current_policy()
    web.set_policy(remote.AccessPolicy(token=TOKEN, trust_loopback=True))
    # TestClient presents 'testclient' as the peer address; pin it to
    # loopback so these tests are about the headers, not the transport.
    monkeypatch.setattr(remote, "is_loopback", lambda host: True)
    with TestClient(web.app, base_url="http://127.0.0.1:8011") as c:
        yield c
    web.set_policy(previous)


def test_a_direct_local_request_needs_no_token(client):
    assert client.get("/api/health").status_code == 200


def test_a_tunnelled_request_is_refused(client):
    """cloudflared stamps cf-connecting-ip on everything it forwards."""
    r = client.get("/api/health", headers={"cf-connecting-ip": "203.0.113.7"})
    assert r.status_code == 401


def test_a_proxied_request_is_refused(client):
    r = client.get("/api/health", headers={"x-forwarded-for": "203.0.113.7"})
    assert r.status_code == 401


def test_a_tunnelled_request_with_the_token_is_served(client):
    r = client.get("/api/health", headers={"cf-connecting-ip": "203.0.113.7",
                                           remote.TOKEN_HEADER: TOKEN})
    assert r.status_code == 200


def test_a_tunnelled_post_cannot_spawn_a_run(client):
    """The endpoint that matters: this one starts a process on the host."""
    r = client.post("/api/process", json={"source": "x.mp4", "clips": 1},
                    headers={"cf-connecting-ip": "203.0.113.7"})
    assert r.status_code == 401


def test_a_rebinding_host_is_refused(client):
    r = client.get("/api/health", headers={"host": "evil.example"})
    assert r.status_code == 421


def test_the_rebinding_check_covers_the_open_login_path(client):
    """/login needs no credential by design, so the Host check has to run
    before that exemption or rebinding simply walks through it."""
    r = client.get("/login", headers={"host": "evil.example"})
    assert r.status_code == 421


def test_reaching_the_server_by_address_still_works(client):
    r = client.get("/api/health", headers={"host": "192.168.1.9:8011"})
    assert r.status_code == 200


# ----------------------------------------------------- open redirect

def test_login_will_not_bounce_to_another_site(client):
    """?next= rides in a link anyone can send. An absolute URL there turns
    the pairing page into an open redirect wearing this server's name."""
    r = client.get(f"/login?{remote.TOKEN_QUERY}={TOKEN}&next=https://evil.example",
                   follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/"


def test_login_will_not_bounce_to_a_protocol_relative_url(client):
    r = client.get(f"/login?{remote.TOKEN_QUERY}={TOKEN}&next=//evil.example",
                   follow_redirects=False)
    assert r.headers["location"] == "/"


def test_login_will_not_hand_back_a_javascript_uri(client):
    """The page's own location.replace(next) would run it as script."""
    r = client.get(f"/login?{remote.TOKEN_QUERY}={TOKEN}&next=javascript:alert(1)",
                   follow_redirects=False)
    assert r.headers["location"] == "/"


def test_a_real_destination_still_survives_login(client):
    r = client.get(f"/login?{remote.TOKEN_QUERY}={TOKEN}&next=/dashboard",
                   follow_redirects=False)
    assert r.headers["location"] == "/dashboard"
