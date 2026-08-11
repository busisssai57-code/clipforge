"""API-less Instagram Reels poster via Instagram Web UI automation."""

from __future__ import annotations

import time
from pathlib import Path

from clipforge.log import get_logger
from clipforge.poster.base import BaseSocialPoster
from clipforge.errors import ClipForgeError
from clipforge.poster.browser import human_delay, human_type, interactive_login, launch_stealth_browser
from clipforge.schemas.poster import PostJob, PostResult

log = get_logger(__name__)


class InstagramReelsPoster(BaseSocialPoster):
    """Automates Instagram Reels uploading via Instagram Web interface."""

    platform = "instagram"

    def login_interactive(self, auth_dir: Path) -> None:
        log.info("instagram.login_interactive", msg="Opening Instagram. Please log in in the browser window.")
        with launch_stealth_browser(auth_dir, self.platform,
                                    headless=False,
                                    save_session_on_exit=True) as (context, page):
            ok = interactive_login(
                page, platform=self.platform,
                start_url='https://www.instagram.com',
                done_when=('instagram.com/?', 'instagram.com/direct', 'instagram.com/explore',),
                where='your Instagram feed')
            if not ok:
                raise ClipForgeError(
                    f"{self.platform} sign-in did not complete; no session saved")

    def upload_clip(self, job: PostJob, auth_dir: Path, headless: bool = True) -> PostResult:
        log.info("instagram.upload_start", job_id=job.job_id, clip=job.clip_path)

        with launch_stealth_browser(auth_dir, self.platform, headless=headless, save_session_on_exit=True) as (context, page):
            try:
                page.goto("https://www.instagram.com", wait_until="domcontentloaded")
                human_delay(2.0, 4.0)

                # Click Create (+) button
                create_nav = page.locator("svg[aria-label='New post'], svg[aria-label='Create']").first
                if create_nav.is_visible():
                    create_nav.click()
                    human_delay(1.5, 3.0)

                # Upload file
                file_input = page.locator("input[type='file']").first
                file_input.set_input_files(str(job.clip_path))
                human_delay(3.0, 5.0)

                # Crop ratio: click ratio selector & select 9:16 vertical if prompted
                crop_btn = page.locator("button:has(svg[aria-label='Select crop'])").first
                if crop_btn.is_visible():
                    crop_btn.click()
                    human_delay(1.0, 1.5)
                    ratio_9_16 = page.locator("button:has-text('9:16')").first
                    if ratio_9_16.is_visible():
                        ratio_9_16.click()
                        human_delay(1.0, 1.5)

                # Next -> Cover / Edit -> Next -> Caption
                next_btn = page.locator("div[role='button']:has-text('Next')").first
                if next_btn.is_visible():
                    next_btn.click()
                    human_delay(1.5, 2.5)

                if next_btn.is_visible():
                    next_btn.click()
                    human_delay(1.5, 2.5)

                # Write caption
                caption_area = page.locator("div[aria-label='Write a caption...'], textarea").first
                if caption_area.is_visible():
                    caption_text = f"{job.title}\n\n{job.caption}\n\n{' '.join(job.hashtags)}"
                    human_type(page, "div[aria-label='Write a caption...'], textarea", caption_text[:2200])

                human_delay(2.0, 3.0)

                # DRAFT ONLY — the Share click was removed, not branched
                # around. A human shares (VERIFICATION.md 2026-07-27).
                status = "draft_saved"

                log.info("instagram.upload_done", job_id=job.job_id, status=status)
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status=status,
                    post_url=page.url,
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

            except Exception as exc:
                log.error("instagram.upload_failed", job_id=job.job_id, error=str(exc))
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status="failed",
                    error_message=str(exc),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
