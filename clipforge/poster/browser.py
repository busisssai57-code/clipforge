"""Playwright browser automation session manager with anti-detection stealth."""

from __future__ import annotations

import random
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator


from clipforge.errors import ClipForgeError
from clipforge.log import get_logger

log = get_logger(__name__)


def _restrict(path: Path, *, directory: bool = False) -> None:
    """Best-effort owner-only permissions on a saved sign-in.

    What lands here is a live session for the operator's TikTok, YouTube,
    Instagram or X account — cookies that are, for anyone who reads them,
    that account. It was being written with the default umask, so on a
    shared or multi-user box every other user could take it. The access
    token gets this treatment already (``clipforge.remote._restrict``);
    the thing that is worth more was missing it.

    Advisory, never fatal: on Windows chmod only moves the read-only bit,
    and a session that could not be locked down is still better than no
    session at all. It is logged rather than raised.
    """
    try:
        path.chmod(0o700 if directory else 0o600)
    except OSError as exc:  # noqa: BLE001 - advisory only
        log.debug("poster.chmod_failed", path=str(path), error=str(exc))


def get_auth_file(auth_dir: Path, platform: str) -> Path:
    """Path to platform storage state JSON file."""
    auth_dir = Path(auth_dir)
    auth_dir.mkdir(parents=True, exist_ok=True)
    _restrict(auth_dir, directory=True)
    return auth_dir / f"{platform}_session.json"


@contextmanager
def launch_stealth_browser(
    auth_dir: Path,
    platform: str,
    headless: bool = True,
    save_session_on_exit: bool = False,
) -> Generator[tuple[Any, Any], None, None]:
    """Launch Playwright browser context with persistent storage state & stealth options."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise ClipForgeError(
            "Playwright is not installed. Run: pip install playwright && playwright install chromium"
        ) from exc

    session_file = get_auth_file(auth_dir, platform)

    with sync_playwright() as p:
        user_data_dir = auth_dir / f"{platform}_profile"
        user_data_dir.mkdir(parents=True, exist_ok=True)
        # The profile holds the same cookies the session file does, so
        # locking down one and not the other protects nothing.
        _restrict(user_data_dir, directory=True)

        browser_type = p.chromium
        # No --no-sandbox here. It was disabling Chromium's renderer
        # sandbox in a browser that then loads whatever a social platform
        # serves, which trades the strongest boundary in the process for
        # nothing — the sandbox is not what makes automation detectable,
        # and the two flags below are the ones that address that.
        args = [
            "--disable-blink-features=AutomationControlled",
            "--disable-infobars",
            "--window-size=1280,800",
        ]

        context = browser_type.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
            args=args,
            viewport={"width": 1280, "height": 800},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            ),
        )

        try:
            # Apply stealth script if available
            try:
                from playwright_stealth import stealth_sync
                page = context.pages[0] if context.pages else context.new_page()
                stealth_sync(page)
            except Exception:
                page = context.pages[0] if context.pages else context.new_page()

            yield context, page

            if save_session_on_exit or not session_file.exists():
                context.storage_state(path=str(session_file))
                _restrict(session_file)
                log.info("poster.session_saved", platform=platform,
                         path=str(session_file))
        finally:
            context.close()


#: How long an interactive sign-in may take before we stop waiting. Real
#: logins involve a password manager, 2FA and sometimes a phone — minutes,
#: not seconds.
INTERACTIVE_LOGIN_TIMEOUT_S = 900.0
_POLL_INTERVAL_S = 2.0


def interactive_login(page: Any, *, platform: str, start_url: str,
                      done_when: tuple[str, ...], where: str) -> bool:
    """Drive a human sign-in and return True once it looks complete.

    Three things were wrong with doing this inline, and all three are the
    kind that make a working tool look frozen:

    1. **The instructions printed AFTER ``page.goto``.** A sign-in page is
       heavy and often redirects; until it settled, the operator had a
       browser window and a silent console, with no way to know the
       process was waiting on them. Instructions come first now.
    2. **A slow or failed navigation aborted the whole login.** The
       operator can navigate themselves; a goto timeout is not a reason to
       throw away the browser they are about to type into.
    3. **ENTER was the only way to finish.** Now the URL is polled too, so
       reaching the destination completes on its own — ENTER remains as
       the manual override for when detection does not fire.

    Returns False if the window was closed before sign-in completed, so
    the caller can decline to save a session that does not exist.
    """
    # ASCII only: operator output goes to a cp1252 console and an em dash
    # renders as a replacement char there.
    print(f"\n[CLIPFORGE] A browser window is opening for {platform}.")
    print("[CLIPFORGE] Sign in there yourself. This app never sees your "
          "password.")
    print(f"[CLIPFORGE] It finishes on its own once you reach {where}.")
    interactive = sys.stdin is not None and sys.stdin.isatty()
    if interactive:
        print("[CLIPFORGE] Press ENTER here if it does not, or to cancel.")
    print()
    sys.stdout.flush()

    try:
        # Generous, and non-fatal: a sign-in page that is slow to settle is
        # normal, and the operator can navigate on their own regardless.
        page.goto(start_url, wait_until="domcontentloaded", timeout=60_000)
    except Exception as exc:  # noqa: BLE001 - reported, not fatal
        log.warning("poster.login_goto_failed", platform=platform,
                    url=start_url, error=str(exc)[:200])
        print(f"[CLIPFORGE] Could not open {start_url} automatically "
              f"({type(exc).__name__}). Navigate there in the window.")

    pressed = threading.Event()

    def _wait_for_enter() -> None:
        try:
            input()
        except (EOFError, KeyboardInterrupt, OSError):
            # NOT a confirmation. With no interactive stdin `input()`
            # raises immediately, and treating that as "the operator
            # pressed ENTER" would report a successful sign-in that never
            # happened and save an unauthenticated session. Only a real
            # line counts.
            return
        pressed.set()

    # Only offer the manual override where there is a console to press it
    # on; daemon so a wedged read can never hold up process exit.
    if interactive:
        threading.Thread(target=_wait_for_enter, daemon=True).start()

    deadline = time.monotonic() + INTERACTIVE_LOGIN_TIMEOUT_S
    while time.monotonic() < deadline:
        if pressed.is_set():
            log.info("poster.login_confirmed_manually", platform=platform)
            return True
        try:
            url = page.url
        except Exception:  # noqa: BLE001 - window closed mid-login
            log.warning("poster.login_window_closed", platform=platform)
            print("[CLIPFORGE] The browser window was closed before sign-in "
                  "completed. Nothing was saved.")
            return False
        if any(fragment in url for fragment in done_when):
            log.info("poster.login_detected", platform=platform, url=url[:120])
            print("[CLIPFORGE] Signed in. Saving the session.")
            return True
        time.sleep(_POLL_INTERVAL_S)

    log.warning("poster.login_timed_out", platform=platform)
    print(f"[CLIPFORGE] Gave up after "
          f"{int(INTERACTIVE_LOGIN_TIMEOUT_S / 60)} minutes.")
    return False


def human_type(page: Any, selector: str, text: str, min_delay_ms: int = 30, max_delay_ms: int = 120) -> None:
    """Type text into element with randomized human keystroke delays."""
    element = page.locator(selector).first
    element.click()
    for char in text:
        element.type(char, delay=random.randint(min_delay_ms, max_delay_ms))
        if random.random() < 0.05:
            time.sleep(random.uniform(0.1, 0.3))


def human_delay(min_s: float = 1.0, max_s: float = 3.0) -> None:
    """Randomized human pause delay."""
    time.sleep(random.uniform(min_s, max_s))
