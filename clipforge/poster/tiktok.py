"""API-less TikTok poster via TikTok Studio web UI automation."""

from __future__ import annotations

import time
from pathlib import Path

from clipforge.log import get_logger
from clipforge.poster.base import BaseSocialPoster
from clipforge.errors import ClipForgeError
from clipforge.poster.browser import human_delay, human_type, interactive_login, launch_stealth_browser
from clipforge.schemas.poster import PostJob, PostResult

log = get_logger(__name__)


class TikTokPoster(BaseSocialPoster):
    """Automates TikTok video uploading via TikTok Creator Center web interface."""

    platform = "tiktok"

    def login_interactive(self, auth_dir: Path) -> None:
        log.info("tiktok.login_interactive", msg="Opening TikTok Login. Please log in in the browser window.")
        with launch_stealth_browser(auth_dir, self.platform,
                                    headless=False,
                                    save_session_on_exit=True) as (context, page):
            ok = interactive_login(
                page, platform=self.platform,
                start_url='https://www.tiktok.com/creator-center/upload',
                done_when=('creator-center', 'tiktokstudio', '/upload',),
                where='the TikTok upload page')
            if not ok:
                raise ClipForgeError(
                    f"{self.platform} sign-in did not complete; no session saved")

    def upload_clip(self, job: PostJob, auth_dir: Path, headless: bool = True) -> PostResult:
        log.info("tiktok.upload_start", job_id=job.job_id, clip=job.clip_path)

        with launch_stealth_browser(auth_dir, self.platform, headless=headless, save_session_on_exit=True) as (context, page):
            try:
                page.goto("https://www.tiktok.com/creator-center/upload", wait_until="domcontentloaded")
                human_delay(2.0, 4.0)

                # Set file input inside iframe or root page
                file_input = page.locator("iframe").content_frame.locator("input[type='file']").first if page.locator("iframe").count() > 0 else page.locator("input[type='file']").first
                
                if not file_input.is_visible():
                    file_input = page.locator("input[type='file']").first

                file_input.set_input_files(str(job.clip_path))
                human_delay(5.0, 8.0)

                # Set caption
                caption_text = f"{job.title} {job.caption} {' '.join(job.hashtags)}"
                editor_box = page.locator(".public-DraftEditor-content, div[contenteditable='true']").first
                if editor_box.is_visible():
                    editor_box.click()
                    page.keyboard.press("Control+A")
                    page.keyboard.press("Backspace")
                    human_type(page, ".public-DraftEditor-content, div[contenteditable='true']", caption_text[:2000])

                human_delay(2.0, 3.0)

                # DRAFT ONLY. The "click Post/Publish" branch was removed
                # rather than left behind an if: dead publish code is one
                # type-edit away from live publish code, and the amendment
                # (VERIFICATION.md, 2026-07-27) is that a human performs
                # every publish. Automation stops at the draft.
                draft_btn = page.locator("button:has-text('Save as draft'), button:has-text('Save draft')").first
                if draft_btn.is_visible():
                    draft_btn.click()
                    human_delay(2.0, 3.0)
                status = "draft_saved"

                log.info("tiktok.upload_done", job_id=job.job_id, status=status)
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status=status,
                    post_url=page.url,
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )

            except Exception as exc:
                log.error("tiktok.upload_failed", job_id=job.job_id, error=str(exc))
                return PostResult(
                    job_id=job.job_id,
                    platform=self.platform,
                    status="failed",
                    error_message=str(exc),
                    completed_at=time.strftime("%Y-%m-%dT%H:%M:%SZ"),
                )
