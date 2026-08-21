"""Who may reach the control API, and from where.

`bta web --lan` puts an API that spawns pipeline subprocesses onto a
network. The authorization decision is therefore the most safety-relevant
code added for remote access, and it is written as a pure function
precisely so it can be tested without a server, a socket or a browser.

Every test here is a sentence about the policy, not about FastAPI.
"""

from __future__ import annotations

import pytest

from clipforge import remote

TOKEN = "s3cret-token-value-long-enough"
POLICY = remote.AccessPolicy(token=TOKEN, trust_loopback=True)


def _decide(**kw):
    pairing = kw.pop("pairing", None)
    policy = kw.pop("policy", POLICY)
    return remote.decide(remote.Presented(**kw), policy, pairing=pairing)


# ------------------------------------------------------------ loopback

@pytest.mark.parametrize("host", ["127.0.0.1", "::1", "::ffff:127.0.0.1",
                                  "127.0.0.5", "localhost"])
def test_loopback_is_recognised_in_every_spelling(host):
    """A dual-stack listener reports IPv4-mapped addresses; a server that
    only knew '127.0.0.1' would demand a token from the machine it runs
    on, which is how operators end up disabling auth."""
    assert remote.is_loopback(host) is True


@pytest.mark.parametrize("host", ["10.0.0.86", "192.168.1.4", "8.8.8.8",
                                  "100.64.0.1", "", None, "not-an-address",
                                  "127.0.0.1.evil.com"])
def test_everything_else_is_not_loopback(host):
    """Unparseable input must be refused, not vouched for. A hostname that
    merely CONTAINS 127.0.0.1 is the classic bypass."""
    assert remote.is_loopback(host) is False


def test_the_local_machine_still_needs_no_token():
    assert _decide(client_host="127.0.0.1").allowed is True


def test_loopback_trust_can_be_switched_off():
    strict = remote.AccessPolicy(token=TOKEN, trust_loopback=False)
    assert _decide(client_host="127.0.0.1", policy=strict).allowed is False


# --------------------------------------------------------------- token

def test_a_lan_client_without_a_token_is_refused():
    d = _decide(client_host="192.168.1.50")
    assert d.allowed is False
    assert "not paired" in d.reason


@pytest.mark.parametrize("field", ["header_token", "cookie_token",
                                   "query_token"])
def test_the_token_is_accepted_from_every_carrier(field):
    """Three carriers because three clients need them: scripts send a
    header, the dashboard sends a cookie, and a pasted link carries a
    query parameter."""
    assert _decide(client_host="192.168.1.50", **{field: TOKEN}).allowed


def test_a_wrong_token_is_refused_and_says_so_specifically():
    d = _decide(client_host="192.168.1.50", header_token="wrong")
    assert d.allowed is False
    # Distinguished from "never paired": a rotated token is a different
    # problem with a different fix.
    assert "not valid" in d.reason


def test_a_url_token_is_promoted_to_a_cookie():
    """So the secret stops travelling in links, history and referrers
    after the first request."""
    d = _decide(client_host="192.168.1.50", query_token=TOKEN)
    assert d.allowed and d.set_cookie is True and d.cookie_value == TOKEN


def test_a_cookie_token_does_not_re_issue_itself():
    d = _decide(client_host="192.168.1.50", cookie_token=TOKEN)
    assert d.allowed and d.set_cookie is False


def test_auth_disabled_allows_everyone_and_admits_it():
    d = _decide(client_host="192.168.1.50",
                policy=remote.AccessPolicy(token=None))
    assert d.allowed and d.reason == "auth disabled"


def test_whitespace_around_a_token_does_not_break_it():
    """Copy-paste from a terminal brings a trailing newline."""
    assert _decide(client_host="192.168.1.50",
                   header_token=f"  {TOKEN}\n").allowed


# ------------------------------------------------------------- pairing

def test_a_pairing_code_admits_a_device_and_hands_back_the_token():
    codes = remote.PairingCodes()
    code, _ttl = codes.issue()
    d = _decide(client_host="192.168.1.50", pairing_code=code, pairing=codes)
    assert d.allowed and d.cookie_value == TOKEN and d.set_cookie


def test_a_pairing_code_works_exactly_once():
    codes = remote.PairingCodes()
    code, _ = codes.issue()
    assert codes.redeem(code) is True
    assert codes.redeem(code) is False


def test_a_pairing_code_expires():
    codes = remote.PairingCodes(ttl_s=0.0)
    code, _ = codes.issue()
    assert codes.redeem(code) is False


def test_wrong_guesses_burn_the_code():
    """Six digits is only safe because of this. Without a burn limit, a
    live code is 10^6 guesses away from a shell on the host."""
    codes = remote.PairingCodes(max_attempts=3)
    code, _ = codes.issue()
    for _ in range(3):
        codes.redeem("000000" if code != "000000" else "111111")
    assert codes.redeem(code) is False, "a burned code must stay dead"


def test_only_one_code_is_live_at_a_time():
    codes = remote.PairingCodes()
    first, _ = codes.issue()
    second, _ = codes.issue()
    assert codes.redeem(first) is False
    assert codes.redeem(second) is True


def test_a_code_is_read_leniently_but_matched_exactly():
    """Phones insert spaces and dashes into one-time codes."""
    codes = remote.PairingCodes()
    code, _ = codes.issue()
    spaced = f"{code[:3]} {code[3:]}"
    assert codes.redeem(spaced) is True


def test_no_pairing_code_means_the_generic_refusal():
    codes = remote.PairingCodes()
    d = _decide(client_host="192.168.1.50", pairing=codes)
    assert d.allowed is False


# ----------------------------------------------------------- addresses

def test_link_local_addresses_are_never_offered():
    """169.254.x.x is what an adapter falls back to when DHCP failed. It
    is 'private' and it is never reachable from a phone, so offering it
    is offering a dead link."""
    assert remote.is_private("169.254.83.107") is True   # by RFC
    assert "169.254.83.107" not in remote.lan_addresses()


def test_lan_addresses_exclude_loopback():
    assert not any(remote.is_loopback(a) for a in remote.lan_addresses())


def test_access_urls_carry_the_token_so_a_link_just_works():
    urls = remote.access_urls(8765, TOKEN)
    assert all(TOKEN in u for u in urls["local"])
    assert urls["local"][0].startswith("http://127.0.0.1:8765/")


def test_access_urls_without_a_token_are_bare():
    urls = remote.access_urls(8765, None)
    assert urls["local"] == ["http://127.0.0.1:8765/"]


# --------------------------------------------------------------- token file

def test_a_token_persists_across_restarts(tmp_path):
    """A phone paired yesterday must still work today. A token that
    changed per run would train the operator to turn auth off."""
    first = remote.load_or_create_token(tmp_path)
    assert remote.load_or_create_token(tmp_path) == first


def test_rotating_invalidates_the_old_token(tmp_path):
    first = remote.load_or_create_token(tmp_path)
    assert remote.rotate_token(tmp_path) != first


def test_a_generated_token_is_not_guessable(tmp_path):
    assert len(remote.load_or_create_token(tmp_path)) >= 32


# --------------------------------------------------------------- env

def test_policy_from_env_reads_the_token():
    p = remote.AccessPolicy.from_env({remote.ENV_TOKEN: TOKEN})
    assert p.token == TOKEN and p.enforced is True


def test_policy_from_env_defaults_to_no_auth():
    assert remote.AccessPolicy.from_env({}).enforced is False


def test_explicitly_disabling_auth_drops_a_supplied_token():
    p = remote.AccessPolicy.from_env({remote.ENV_TOKEN: TOKEN,
                                      remote.ENV_REQUIRE_AUTH: "0"})
    assert p.enforced is False and p.insecure is True


# -------------------------------------------------------------- tunnel

def test_a_quick_tunnel_url_is_recognised():
    line = ("2026-08-12T19:00:00Z INF |  https://brave-ox-hums.trycloudflare.com"
            "                          |")
    assert remote.parse_tunnel_url(line) == \
        "https://brave-ox-hums.trycloudflare.com"


def test_ordinary_output_yields_no_url():
    assert remote.parse_tunnel_url("INF Requesting new quick tunnel...") is None
