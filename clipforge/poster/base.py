"""Abstract base class for API-less social media platform posters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from clipforge.schemas.poster import PostJob, PostResult


class BaseSocialPoster(ABC):
    """Base poster interface for Playwright-based browser automation."""

    platform: str = "base"

    @abstractmethod
    def login_interactive(self, auth_dir: Path) -> None:
        """Launch interactive browser for user to log into the platform.

        Saves persistent browser storage state / cookies under auth_dir.
        """

    @abstractmethod
    def upload_clip(self, job: PostJob, auth_dir: Path, headless: bool = True) -> PostResult:
        """Automates browser navigation to upload clip and post/save draft."""
