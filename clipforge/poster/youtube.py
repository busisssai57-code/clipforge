"""API-less YouTube Shorts poster via YouTube Studio web UI automation."""

from __future__ import annotations

import time
from pathlib import Path

from clipforge.log import get_logger
from clipforge.poster.base import BaseSocialPoster
from clipforge.errors import ClipForgeError
from clipforge.poster.browser import human_delay, human_type, interactive_login, launch_stealth_browser
from clipforge.schemas.poster import PostJob, PostResult

log = get_logger(__name__)


class YouTubeShortsPoster(BaseSocialPoster):
    """Automates YouTube Shorts uploading via YouTube Studio web interface."""

    platform = "youtube"

    def login_interactive(self, auth_dir: Path) -> None:
        log.info("youtube.login_interactive", msg="Opening YouTube Studio. Please log in in the browser window.")
        with launch_stealth_browser(auth_dir, self.platform,
                                    headless=False,
                                    save_session_on_exit=True) as (context, page):
            ok = interactive_login(
                page, platform=self.platform,
                start_url='https://studio.youtube.com',
                done_when=('studio.youtube.com/channel', 'studio.youtube.com/video',),
                where='the YouTube Studio dashboard')
            if not ok:
                raise ClipForgeError(
                    f"{self.platform} sign-in did not complete; no session saved")

    def upload_clip(self, job: PostJob, auth_dir: Path, headless: bool = True) -> PostResult:
        log.info("youtube.upload_start", job_id=job.job_id, clip=job.clip_path)

        with launch_stealth_browser(auth_dir, self.platform, headless=headless, save_session_on_exit=True) as (context, page):
            try:
                page.goto("https://studio.youtube.com", wait_until="domcontentloaded")
                human_delay(2.0, 4.0)

                # Check if logged in
                if "channel" not in page.url and "studio.youtube.com" not in page.url:
                    return PostResult(
                        job_id=job.job_id,
                        platform=self.platform,
                        status="failed",
                        error_message="Not logged into YouTube Studio. Run 'clipforge auth youtube' first.",
                        completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    )

                # Click Create button if present or upload icon
                create_btn = page.locator("#create-icon, button:has-text('Create')").first
                if create_btn.is_visible():
                    create_btn.click()
                    human_delay(1.0, 2.0)
                    upload_option = page.locator("tp-yt-paper-item:has-text('Upload videos'), #text-item-0").first
                    if upload_option.is_visible():
                        upload_option.click()
                        human_delay(1.0, 2.0)

                # Upload file
                file_input = page.locator("input[type='file']").first
                file_input.set_input_files(str(job.clip_path))
                human_delay(4.0, 6.0)

                # Set title & description
                title_box = page.locator("#textbox[aria-label*='title'], #title-textarea #textbox").first
                if title_box.is_visible():
                    title_box.click()
                    page.keyboard.press("Control+A")
                    page.keyboard.press("Backspace")
                    human_type(page, "#textbox[aria-label*='title']", f"{job.title} #Shorts")

                desc_box = page.locator("#textbox[aria-label*='description'], #description-textarea #textbox").first
                if desc_box.is_visible():
                    desc_box.click()
                    page.keyboard.press("Control+A")
                    page.keyboard.press("Backspace")
                    full_desc = f"{job.caption}\n\n{' '.join(job.hashtags)}"
                    human_type(page, "#textbox[aria-label*='description']", full_desc[:4000])

                # Select "Not made for kids"
                not_kids = page.locator("tp-yt-paper-radio-button[name='VIDEO_MADE_FOR_KIDS_NOT_MFK']").first
                if not_kids.is_visible():
                    not_kids.click()
                    human_delay(1.0, 2.0)

                # Click Next until Visibility tab
                next_btn = page.locator("#next-button").first
                for _ in range(3):
                    if next_btn.is_visible() and next_btn.is_enabled():
                        next_btn.click()
                        human_delay(1.5, 3.0)

                # DRAFT ONLY. The visibility radio ("PUBLIC") and the #done
                # button that finalizes the upload were REMOVED, not left
                # behind an `if` — dead publish code is one type-edit from
                # live publish code. The upload is left as a YouTube Studio
                # draft; a human sets visibility and publishes
                # (VERIFICATION.md, 2026-07-27 amendment).
                close_btn = page.locator("#close-button, button[aria-label='Close']").first
                if close_btn.is_visible():
                    close_btn.click()
                status = "draft_saved"

                log.info("youtube.upload_done", job_id=job.job_id, status=status)
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status=status,
                    post_url=page.url,
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

            except Exception as exc:
                log.error("youtube.upload_failed", job_id=job.job_id, error=str(exc))
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status="failed",
                    error_message=str(exc),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
