"""Structural checks on the live dashboard.

dashboard_live.html is 4,000 lines of markup and one inline script, and it
had no tests. Deleting the generation half broke it in four separate ways
and every one of them was silent:

  * `/api/models` sat in the BOOT Promise.all with no catch, so one 404
    rejected the whole await. NICHES, CAPS and DUBLANGS were never assigned:
    every capability tile read unavailable and the composer had no styles.
  * `/api/generated` and `/api/generated/status` sat in the POLL the same
    way, so refresh() threw every four seconds and the connection dot never
    turned green.
  * Removing a markup block left handlers bound to elements that no longer
    existed, and `$('#gone').onclick = ...` throws on null, killing every
    statement after it in the script.
  * A regex removal left an unbalanced brace, and a SyntaxError in the one
    inline script means nothing on the page runs at all.

None of that shows up in pytest, in `verify all`, or in a diff. It shows up
when a person opens the page. These checks read the file instead.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "clipforge" / "dashboard_live.html"
WEB = ROOT / "clipforge" / "web.py"

_ROUTE = re.compile(r"@app\.(?:get|post|put|delete)\(\"([^\"]+)\"")
_CALL = re.compile(r"['\"`](/api/[A-Za-z0-9/_-]+)")
_HANDLER = re.compile(r"\$\('#([A-Za-z0-9_-]+)'\)\.(onclick|onchange|oninput|addEventListener)")


@pytest.fixture(scope="module")
def dash() -> str:
    return DASH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def routes() -> set[str]:
    return {r.split("{")[0].rstrip("/")
            for r in _ROUTE.findall(WEB.read_text(encoding="utf-8"))}


def script_of(html: str) -> str:
    """The one long inline script, which is where everything lives."""
    blocks = re.findall(r"<script[^>]*>(.*?)</script>", html, re.S)
    return max(blocks, key=len) if blocks else ""


def element_ids(html: str) -> set[str]:
    return set(re.findall(r'id="([A-Za-z0-9_-]+)"', html))


def dead_calls(html: str, routes: set[str]) -> list[str]:
    bad = []
    for call in sorted(set(_CALL.findall(html))):
        path = call.rstrip("/")
        if not any(path == r or path.startswith(r + "/") for r in routes):
            bad.append(call)
    return bad


def dangling_handlers(html: str) -> list[str]:
    ids = element_ids(html)
    return [f"#{m.group(1)}.{m.group(2)}"
            for m in _HANDLER.finditer(html) if m.group(1) not in ids]


def unbalanced(src: str) -> list[str]:
    """Brace/paren/bracket balance outside strings, comments and regex.

    Not a parser. It is enough to catch a removal that ate a closing brace,
    which is the failure that actually happened.

    Template literals need real handling rather than a counter: `${a}` opens
    and closes inside the string, and an earlier version counted the "${"
    while the string scanner swallowed the "}", so every healthy template
    read as unbalanced. A hole is scanned as code, to its matching brace.

    BRACES ONLY. Regex literals are not tracked - telling `/[a-z(]/` from a
    division needs the parser this deliberately is not - and on the real file
    they left '(' and '[' each short by one while the source was valid. A
    narrow check that holds beats a broad one that cries wolf, and the
    failure this exists to catch was an eaten closing brace.
    """
    depth = {"{": 0}
    close = {"}": "{"}

    def scan(i: int, stop: str | None) -> int:
        """Scan code from i. With stop='}' return at the matching brace."""
        n = len(src)
        while i < n:
            c = src[i]
            if stop and c == "}":
                return i
            if c in "\"'":
                i = skip_quote(i, c)
            elif c == "`":
                i = skip_template(i)
            elif src.startswith("//", i):
                j = src.find(chr(10), i)
                i = n if j < 0 else j
            elif src.startswith("/*", i):
                j = src.find("*/", i)
                i = n if j < 0 else j + 1
            elif c in depth:
                depth[c] += 1
            elif c in close:
                depth[close[c]] -= 1
            i += 1
        return i

    def skip_quote(i: int, q: str) -> int:
        i += 1
        while i < len(src):
            if src[i] == "\\":
                i += 2
                continue
            if src[i] == q:
                return i
            i += 1
        return i

    def skip_template(i: int) -> int:
        i += 1
        while i < len(src):
            if src[i] == "\\":
                i += 2
                continue
            if src[i] == "`":
                return i
            if src.startswith("${", i):
                i = scan(i + 2, "}")     # the hole is code, not text
                continue
            i += 1
        return i

    scan(0, None)
    return [f"{k!r} unbalanced by {v}" for k, v in depth.items() if v]


# --- the checks ----------------------------------------------------------

def test_the_dashboard_and_the_api_are_both_readable(dash, routes):
    """Guard the readers first: a vacuous pass here hides every check below."""
    assert len(dash) > 50_000, "dashboard_live.html looks truncated"
    assert len(routes) >= 20, f"only {len(routes)} routes parsed from web.py"
    assert len(script_of(dash)) > 50_000, "inline script not found"


def test_every_endpoint_the_dashboard_calls_is_served(dash, routes):
    dead = dead_calls(dash, routes)
    assert not dead, ("the dashboard calls endpoints web.py does not serve: "
                      + ", ".join(dead))


def test_no_handler_is_bound_to_an_element_that_does_not_exist(dash):
    bad = dangling_handlers(dash)
    assert not bad, ("$('#id').on... on missing elements, which throws and "
                     "kills the rest of the script: " + ", ".join(bad))


def test_the_inline_script_is_balanced(dash):
    bad = unbalanced(script_of(dash))
    assert not bad, "inline script does not balance: " + "; ".join(bad)


def test_every_capability_key_the_tools_gate_on_is_real(dash):
    """A tile gating on a key the probe never emits is permanently dark."""
    from clipforge import capabilities
    known = {c.key for c in capabilities.probe()} | {"tracking"}
    used = set(re.findall(r"cap:'([a-z_]+)'", dash))
    unknown = sorted(used - known)
    assert not unknown, f"tool tiles gate on unknown capability keys: {unknown}"


def test_the_boot_and_poll_fetches_all_carry_their_own_catch(dash):
    """One dead endpoint must not blank the whole page.

    Both Promise.all blocks lost every assignment after them when a single
    leg rejected. Each leg carries `.catch` now, and this is what keeps it
    that way.
    """
    src = script_of(dash)
    bad = []
    for m in re.finditer(r"await Promise\.all\(\[(.*?)\]\)", src, re.S):
        for leg in re.findall(r"api\('(/api/[^']+)'\)(\.catch)?", m.group(1)):
            if not leg[1]:
                bad.append(leg[0])
    assert not bad, ("Promise.all legs with no .catch — one 404 rejects the "
                     "whole await: " + ", ".join(bad))


# --- teeth ---------------------------------------------------------------
# Every validator above gets a deliberately broken input. A check that
# cannot fail is not a check.

def test_teeth_a_dead_endpoint_is_reported():
    assert dead_calls("fetch('/api/gone')", {"/api/clips"}) == ["/api/gone"]


def test_teeth_a_served_path_parameter_is_not_called_dead():
    assert dead_calls("fetch('/api/clips/thumb/x.mp4')",
                      {"/api/clips/thumb"}) == []


def test_teeth_an_empty_route_set_condemns_every_call():
    assert dead_calls("fetch('/api/clips')", set()) == ["/api/clips"]


def test_teeth_a_dangling_handler_is_reported():
    html = '<div id="a"></div>' + """$('#b').onclick=x"""
    assert dangling_handlers(html) == ["#b.onclick"]


def test_teeth_a_handler_on_a_real_element_is_not_reported():
    html = '<div id="b"></div>' + """$('#b').onclick=x"""
    assert dangling_handlers(html) == []


def test_teeth_an_eaten_closing_brace_is_reported():
    assert unbalanced("function f(){ if(x){ y() }")


def test_teeth_balanced_source_with_braces_in_strings_is_clean():
    assert unbalanced("const s='}}}'; const t=`${a}`; function f(){ return 1 }") == []
