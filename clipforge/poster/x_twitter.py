"""API-less X (Twitter) poster via X Web UI automation."""

from __future__ import annotations

import time
from pathlib import Path

from clipforge.log import get_logger
from clipforge.poster.base import BaseSocialPoster
from clipforge.errors import ClipForgeError
from clipforge.poster.browser import human_delay, human_type, interactive_login, launch_stealth_browser
from clipforge.schemas.poster import PostJob, PostResult

log = get_logger(__name__)


class XTwitterPoster(BaseSocialPoster):
    """Automates X (Twitter) video posts via X Web interface."""

    platform = "x_twitter"

    def login_interactive(self, auth_dir: Path) -> None:
        log.info("x_twitter.login_interactive", msg="Opening X (Twitter). Please log in in the browser window.")
        with launch_stealth_browser(auth_dir, self.platform,
                                    headless=False,
                                    save_session_on_exit=True) as (context, page):
            ok = interactive_login(
                page, platform=self.platform,
                start_url='https://x.com/login',
                done_when=('x.com/home', 'twitter.com/home',),
                where='your X home timeline')
            if not ok:
                raise ClipForgeError(
                    f"{self.platform} sign-in did not complete; no session saved")

    def upload_clip(self, job: PostJob, auth_dir: Path, headless: bool = True) -> PostResult:
        log.info("x_twitter.upload_start", job_id=job.job_id, clip=job.clip_path)

        with launch_stealth_browser(auth_dir, self.platform, headless=headless, save_session_on_exit=True) as (context, page):
            try:
                page.goto("https://x.com/compose/post", wait_until="domcontentloaded")
                human_delay(2.0, 4.0)

                # Attach media file
                file_input = page.locator("input[data-testid='fileInput']").first
                if not file_input.is_visible():
                    file_input = page.locator("input[type='file']").first

                file_input.set_input_files(str(job.clip_path))
                human_delay(3.0, 5.0)

                # Write text content
                post_box = page.locator("div[data-testid='tweetTextarea_0']").first
                if post_box.is_visible():
                    post_box.click()
                    text_content = f"{job.title}\n\n{' '.join(job.hashtags[:3])}"
                    human_type(page, "div[data-testid='tweetTextarea_0']", text_content[:280])

                human_delay(2.0, 3.0)

                # DRAFT ONLY — the tweetButton click was removed, not
                # branched around. A human posts (VERIFICATION.md 2026-07-27).
                status = "draft_saved"

                log.info("x_twitter.upload_done", job_id=job.job_id, status=status)
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status=status,
                    post_url=page.url,
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

            except Exception as exc:
                log.error("x_twitter.upload_failed", job_id=job.job_id, error=str(exc))
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status="failed",
                    error_message=str(exc),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
