"""Configuration models — pydantic v2, fail-fast at startup.

Two files (spec §4):
  * ``config/config.toml``   — all tunables (``config.example.toml`` documents them)
  * ``config/channels.toml`` — the operator's watchlist

Loading is strict: unknown keys are rejected (``extra="forbid"``) so a typo'd
tunable fails at startup instead of silently using a default for days.
Secrets (HF token) come from the environment / ``.env``, never from TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from clipforge.errors import ConfigError

Platform = Literal["youtube", "twitch", "kick"]


class _StrictModel(BaseModel):
    """Base: unknown keys are config errors, not surprises."""

    model_config = {"extra": "forbid"}


# --------------------------------------------------------------------------
# config.toml sections
# --------------------------------------------------------------------------


class WorkspaceConfig(_StrictModel):
    root: Path = Path("workspace")


class IngestConfig(_StrictModel):
    poll_interval_s: float = Field(60.0, gt=0, description="Live/VOD poll cadence per channel")
    segment_time_s: int = Field(900, ge=60, le=3600, description="Chunk length (spec: 900)")
    playlist_end: int = Field(5, ge=1, le=50, description="yt-dlp --playlist-end for VOD discovery")
    overlap_s: int = Field(60, ge=0, description="T1: trailing overlap carried into next chunk")
    segment_ready_stable_s: float = Field(20.0, gt=0, description="Size-unchanged window ⇒ segment closed")
    backoff_base_s: float = Field(5.0, gt=0)
    backoff_max_s: float = Field(300.0, gt=0)
    twitch_disable_ads: bool = True
    kick_enabled: bool = Field(False, description="T4: Kick is best-effort, off by default")
    quality: str = Field("best", description="streamlink stream quality selector")
    #: The VOD backlog is a different product from live capture: it clips
    #: uploads the operator may never have asked for, and on a watched
    #: channel it is usually the live broadcast that matters.
    youtube_vods: bool = Field(
        True, description="Also clip a YouTube channel's published VODs")


class DiskConfig(_StrictModel):
    free_floor_gb: float = Field(50.0, gt=0, description="Pause ingestion below this free space")
    retention_hours: float = Field(48.0, gt=0, description="Delete processed chunks older than this")


class S1Config(_StrictModel):
    model: str = "large-v2"
    batch_size: int = Field(8, ge=1, le=8, description="HARD CAP — VRAM protection (spec §S1)")
    compute_type: str = "float16"
    language: str | None = Field(None, description="None = autodetect; pin for determinism in prod")
    vram_budget_gb: float = Field(8.0, gt=0)


class S2Config(_StrictModel):
    window_min_s: float = Field(30.0, gt=0)
    window_max_s: float = Field(60.0, gt=0)
    top_k: int = Field(10, ge=1)
    nms_iou: float = Field(0.4, gt=0, lt=1)


class S3Config(_StrictModel):
    #: OFF BY DEFAULT, and the default is the point. §2 of the spec is a
    #: hard contract — "Cloud: **None.** No hosted inference" — and this
    #: flag decides whether candidate frames and transcript text leave the
    #: machine. It shipped as True from 2026-08-04 until the operator
    #: settled it on 2026-08-05: keep everything local.
    #:
    #: With it on, Gemini reads the frames and transcript and the local
    #: Qwen below becomes the fallback. With it off the cloud ranker is
    #: never constructed (``vlrank.build_ranker`` returns None), so no
    #: frame can leave by accident.
    use_cloud: bool = False
    #: Ordered preference, strongest first. Gemini Pro is not on the free
    #: API tier — an unbilled key gets 429 "limit: 0" for it, not 403 — so
    #: a single name would drop straight past Flash to the local 7B. The
    #: chain uses Pro as soon as billing is enabled on the key.
    cloud_models: list[str] = ["gemini-2.5-pro", "gemini-2.5-flash"]
    cloud_model: str = "gemini-2.5-pro"
    cloud_timeout_s: float = Field(90.0, gt=0)
    cloud_max_attempts: int = Field(3, ge=1, le=6)

    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    fallback_model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct"  # loaded NF4 via bitsandbytes
    frames_per_candidate: int = Field(8, ge=6, le=8, description="strictly 6–8 (spec §S3)")
    max_pixels: int = Field(451_584, gt=0, description="T7: per-frame pixel cap for the VL processor (≈768×588)")
    seed: int = 1234
    vram_budget_gb: float = Field(10.0, gt=0)


class S4Config(_StrictModel):
    pose_model: str = "yolo11m-pose.pt"
    asd_confidence_tau: float = Field(0.5, ge=0, le=1, description="Below τ: documented fallback framing")
    deadzone_frac: float = Field(0.02, ge=0, description="Ignore drift under this fraction of frame width")
    max_pan_px_per_frame: float = Field(40.0, gt=0)
    one_euro_min_cutoff: float = Field(1.0, gt=0)
    one_euro_beta: float = Field(0.007, ge=0)
    vram_budget_gb: float = Field(3.0, gt=0)
    #: Push-in depth on emphasis beats, as a fraction of the crop. 0 = off.
    #: Capped at 0.25: past a quarter the source pixels run out and the
    #: upscale to 1080x1920 goes visibly soft.
    punch_scale: float = Field(0.08, ge=0, le=0.25,
                               description="Zoom depth on emphasis beats")
    punch_hold_s: float = Field(0.9, ge=0, le=5.0)
    #: MediaPipe FaceLandmarker model filename under workspace/models/.
    #: Missing file ⇒ MAR is disabled and subject selection falls back to
    #: presence, loudly logged — never a crash.
    face_model: str = "face_landmarker.task"
    #: |ΔMAR| PER SECOND a face must exceed to count as SPEAKING. Per-second
    #: units so the threshold means the same thing at any sampling rate
    #: (the old per-sample 0.03 at ~6 Hz is ≈0.18/s). Below it, everyone in
    #: the shot is just sitting there and presence decides.
    mar_activity_tau: float = Field(0.18, ge=0, le=10)


class PacingConfig(_StrictModel):
    """Jump-cut silence removal (blueprint §9.2). Off by default: it alters
    the source's rhythm, which is an editorial choice, not a correction."""

    enabled: bool = False
    #: A pause must exceed this to be cut. Below it, breaths survive.
    gap_threshold_s: float = Field(0.6, gt=0, le=5)
    #: Retained padding each side of a cut so no word edge is clipped.
    pad_s: float = Field(0.12, ge=0, le=1)


class S5Config(_StrictModel):
    font: str = "Arial Black"
    font_size: int = Field(72, gt=0)
    highlight_color: str = Field("&H0000FFFF", description="ASS BGR — yellow active word")
    base_color: str = Field("&H00FFFFFF", description="ASS BGR — white")
    outline: float = Field(3.0, ge=0)
    shadow: float = Field(1.0, ge=0)
    margin_v: int = Field(260, ge=0, description="Vertical margin inside 9:16 safe area")
    max_words_per_line: int = Field(3, ge=1, le=8)
    max_lines: int = Field(1, ge=1, le=2)
    #: "pop" = one event per word, active word scaled and coloured (the
    #: short-form look). "karaoke" = a single swept line, calmer and cheaper.
    animation: Literal["pop", "karaoke"] = "pop"
    uppercase: bool = False


class S6Config(_StrictModel):
    width: int = Field(1080, description="Output is portrait 9:16")
    height: int = Field(1920)
    encoder: str = Field("h264_nvenc", description="Falls back to libx264 on session exhaustion")
    nvenc_preset: str = "p5"
    #: libx264 preset for the CPU path. Not a rarely-touched fallback knob:
    #: whenever NVENC is unavailable this IS the encoder, so it gets the same
    #: configurability its NVENC counterpart always had.
    x264_preset: str = "veryfast"
    cq: int = Field(21, ge=0, le=51)
    audio_bitrate: str = "192k"
    loudness_i: float = Field(-14.0, description="Integrated loudness target (LUFS)")
    loudness_tp: float = Field(-2.0)
    loudness_lra: float = Field(11.0)
    #: Speech cleanup applied BEFORE loudnorm, so the normalizer measures
    #: the cleaned signal. Default "off" is a judgement about CONTENT, not
    #: caution: this pipeline clips music and performance as readily as
    #: talking heads, and spectral denoise + de-essing on a beat is
    #: destructive. "gentle" is safe on mixed audio; "strong" suits a
    #: lavalier or a room mic and is wrong for anything musical.
    enhance_speech: Literal["off", "gentle", "strong"] = "off"


class S7Config(_StrictModel):
    """Quality control.

    The deterministic checks always run and never leave the machine. The VL
    judge is a second opinion on the things a container check cannot see -
    whether a face is framed, whether the captions collide with burned-in
    text, whether the opening frame earns a scroll-stop.

    ``use_cloud`` is the only switch here that can send anything anywhere,
    and it is registered as the ``vl_qa`` feature in clipforge.cloud. It
    ships FALSE: the standing decision of 2026-08-05 is that hosted
    inference is off, and with no Anthropic or OpenAI key on this machine
    turning it on would not reach the primary judge at all - it would fall
    through to Gemini and send clip frames to Google, which is a different
    thing from what "Claude first" asks for. One flag flips it the moment a
    key exists.
    """

    #: The local judge runs regardless; this is hosted inference only.
    use_cloud: bool = Field(False, description=(
        "Authorize the vl_qa feature to use a hosted VL judge (see "
        "clipforge.cloud). False = the local Qwen VL judge only."))
    vl_qa: bool = Field(True, description=(
        "Run the VL judge at all. False = deterministic checks only."))
    vl_frames: int = Field(6, ge=1, le=16, description=(
        "How many frames the judge sees. Sampled at fixed fractions, so a "
        "verdict can be replayed against a past run."))


class EditorConfig(_StrictModel):
    style_profile: str = Field("viral_fast", description="Editor agent profile: viral_fast, educational, conversational")
    max_hashtags: int = Field(5, ge=1, le=20, description="Max hashtags per post")


class PostingConfig(_StrictModel):
    """Publishing config — DRAFT-ONLY, with a mandatory human gate.

    The spec's §10 Non-goals and §3.5 Authorization Law both forbid
    publishing outright ("this pipeline *produces files and stops*"). The
    operator amended that on 2026-07-27 to allow posting under three
    conditions, which are enforced here rather than left to documentation:

      1. ``publish_mode`` is pinned to ``"draft"``. The type no longer
         admits ``"public"`` at all, so no config file, env var, or code
         path can select autonomous publishing.
      2. ``smart_scheduling`` is pinned False — nothing may queue itself to
         fire at a later time without a human present.
      3. Every post requires explicit per-clip approval
         (``require_approval``), checked at dispatch.

    Relaxing any of these is a spec amendment and belongs in
    VERIFICATION.md, not in a config edit.
    """

    enabled_platforms: list[str] = Field(
        default_factory=lambda: ["youtube", "tiktok", "instagram", "x_twitter"],
        description="Target social platforms for posting"
    )
    #: Literal["draft"] — deliberately a single-member type, not a default.
    publish_mode: Literal["draft"] = Field(
        "draft", description="DRAFT ONLY. Autonomous publishing is not "
                             "available; a human completes every post.")
    smart_scheduling: bool = Field(
        False, description="Must stay False: unattended scheduled publishing "
                           "is exactly what the human gate exists to prevent.")
    require_approval: bool = Field(
        True, description="Must stay True: each clip is approved individually "
                          "before any browser automation runs.")
    target_timezone_offset_hours: float = Field(-5.0, description="Target audience timezone offset (e.g. -5.0 for EST)")
    headless: bool = Field(True, description="Run browser automation headless")
    # delay_min_s / delay_max_s were here and were read by nothing: each
    # automator picks its own pacing per action (`human_delay(2.0, 4.0)`
    # while a page settles, `(1.0, 1.5)` between keystrokes), which a
    # single global pair cannot express. A knob that cannot change the
    # behaviour it names is worse than no knob.

    @field_validator("smart_scheduling")
    @classmethod
    def _no_unattended_scheduling(cls, v: bool) -> bool:
        if v:
            raise ValueError(
                "posting.smart_scheduling must be false - scheduled "
                "publishing without a human present is disallowed by the "
                "draft-only amendment (see VERIFICATION.md)")
        return v

    @field_validator("require_approval")
    @classmethod
    def _approval_is_mandatory(cls, v: bool) -> bool:
        if not v:
            raise ValueError(
                "posting.require_approval must be true - every clip is "
                "approved individually (see VERIFICATION.md)")
        return v





class OrchestrationConfig(_StrictModel):
    ingest_concurrency: int = Field(4, ge=1)
    gpu_concurrency: int = Field(1, ge=1, le=1, description="LAW: exactly 1 (spec §6)")
    # render_concurrency was here, unread: renders are serialised behind
    # the one-GPU-stage law, so a second one never starts. It said the
    # opposite.
    #: Bounded ⇒ backpressure. Read by the watch→DAG dispatcher: when it is
    #: full, the window is dropped from the CLIP queue (loudly) and
    #: ingestion continues. The media is still on disk; unrecorded stream
    #: would not be.
    queue_maxsize: int = Field(32, ge=1, description="Bounded queues ⇒ backpressure")
    #: Clips rendered per ingested window in `bta watch`. One by default:
    #: a live stream produces a window every few minutes, and asking for
    #: three clips each would queue GPU work faster than it drains.
    clips_per_window: int = Field(
        1, ge=1, le=10,
        description="Clips per ingested window in watch mode")


class WatchConfig(_StrictModel):
    """`bta watch`: when the GPU half of the loop is allowed to run.

    Recording is never gated — a live stream cannot be replayed. Clipping
    is, because it saturates the one GPU the operator is also using.
    """

    clip_only_when_idle: bool = Field(
        True, description="Hold rendered-clip work until the machine is idle")
    idle_after_s: float = Field(
        300.0, ge=0, description="Keyboard/mouse quiet this long = idle")
    gpu_busy_pct: int = Field(
        40, ge=1, le=100,
        description="Another process using this much GPU = not idle")
    idle_poll_s: float = Field(
        30.0, gt=0, description="Re-check cadence while waiting for idle")
    #: Asymmetric with idle_after_s on purpose: a running job yields as
    #: soon as the operator touches anything, and only resumes once they
    #: have been away for the full idle window again.
    preempt_within_s: float = Field(
        60.0, gt=0,
        description="Input this recent pauses a job already running")
    min_free_vram_gb: float = Field(
        8.0, ge=0,
        description="Another process holding the GPU's memory = not idle")


class NotifyConfig(_StrictModel):
    """Private delivery of each accepted clip to the operator's own phone.

    This is NOT publishing: the clip goes to one Telegram chat, the
    operator's, and nothing is posted anywhere. The credential lives in
    ``.env`` (CLIPFORGE_TELEGRAM_BOT_TOKEN / CLIPFORGE_TELEGRAM_CHAT_ID) or,
    when those are absent, is READ from the OpenClaw gateway's config
    rather than copied, so a rotated bot token is picked up here too.
    """

    telegram: bool = Field(
        False, description="Send every clip that passes QA to Telegram")
    openclaw_config: Path = Field(
        Path("~/.openclaw/openclaw.json"),
        description="Fallback source for the bot token and chat id")
    retry_interval_s: float = Field(
        600.0, gt=0, description="`bta watch` retries failed sends this often")


class AppConfig(_StrictModel):
    """Root of config.toml."""

    workspace: WorkspaceConfig = WorkspaceConfig()
    ingest: IngestConfig = IngestConfig()
    disk: DiskConfig = DiskConfig()
    s1: S1Config = S1Config()
    s2: S2Config = S2Config()
    s3: S3Config = S3Config()
    s4: S4Config = S4Config()
    s5: S5Config = S5Config()
    s6: S6Config = S6Config()
    s7: S7Config = S7Config()
    editor: EditorConfig = EditorConfig()
    pacing: PacingConfig = PacingConfig()
    posting: PostingConfig = PostingConfig()
    orchestration: OrchestrationConfig = OrchestrationConfig()
    watch: WatchConfig = WatchConfig()
    notify: NotifyConfig = NotifyConfig()


    @field_validator("s2")
    @classmethod
    def _window_bounds_sane(cls, v: S2Config) -> S2Config:
        if v.window_min_s >= v.window_max_s:
            raise ValueError("s2.window_min_s must be < s2.window_max_s")
        return v


class Secrets(BaseSettings):
    """Environment-sourced secrets. Never stored in TOML, never logged."""

    model_config = SettingsConfigDict(env_prefix="CLIPFORGE_", env_file=".env",
                                      env_file_encoding="utf-8", extra="ignore")

    hf_token: str | None = Field(None, description="Hugging Face token for gated pyannote models (T5)")
    #: Google AI Studio key for Veo. Absent = the cloud provider reports
    #: itself unconfigured and generation runs entirely on the local model.
    #: Set as CLIPFORGE_GEMINI_API_KEY in .env. Never logged.
    gemini_api_key: str | None = Field(
        None, description="Google AI Studio key")
    #: VL quality control runs a chain: Anthropic first, OpenAI second,
    #: Gemini third, and the local Qwen ranker last. Each is optional and an
    #: absent key means that link reports itself unconfigured and the chain
    #: moves on - it never silently downgrades without saying which ran.
    anthropic_api_key: str | None = Field(
        None, description="Anthropic key for the primary VL judge")
    openai_api_key: str | None = Field(
        None, description="OpenAI key for the second VL judge")
    #: Private clip delivery (see NotifyConfig). Both optional: when absent,
    #: clipforge.notify reads them from the OpenClaw gateway config.
    telegram_bot_token: str | None = Field(
        None, description="Telegram bot token for clip delivery")
    telegram_chat_id: str | None = Field(
        None, description="The operator's Telegram chat id")


# --------------------------------------------------------------------------
# channels.toml
# --------------------------------------------------------------------------


class ChannelSpec(_StrictModel):
    platform: Platform
    handle: str = Field(min_length=1, description="Channel handle/slug as the platform names it")
    enabled: bool = True
    priority: int = Field(0, description="Higher = polled first when contended")


class Watchlist(_StrictModel):
    channels: list[ChannelSpec] = []

    @field_validator("channels")
    @classmethod
    def _no_duplicates(cls, channels: list[ChannelSpec]) -> list[ChannelSpec]:
        """Reject the same channel twice, case-insensitively.

        Per-channel isolation (VOD dedup, session resume, segment
        directories) is keyed on (platform, handle); two entries differing
        only in case would run two loops that race the same downloads into
        the same files.
        """
        seen: dict[tuple[str, str], str] = {}
        for c in channels:
            key = (c.platform, c.handle.lower().lstrip("@"))
            if key in seen:
                raise ValueError(
                    f"duplicate channel {c.platform}:{c.handle!r} "
                    f"(already listed as {seen[key]!r}) - one entry per channel")
            seen[key] = c.handle
        return channels

    def enabled_sorted(self) -> list[ChannelSpec]:
        """Deterministic processing order: priority desc, then platform/handle."""
        return sorted((c for c in self.channels if c.enabled),
                      key=lambda c: (-c.priority, c.platform, c.handle))


# --------------------------------------------------------------------------
# loaders
# --------------------------------------------------------------------------


#: UTF-8 byte-order mark. Notepad and PowerShell's `Set-Content -Encoding
#: UTF8` both write one by default on Windows, and tomllib does NOT skip it —
#: it fails with a baffling "Invalid statement (at line 1, column 1)" that
#: points at config the operator can see is perfectly valid. Strip it.
_UTF8_BOM = b"\xef\xbb\xbf"


def _read_toml(path: Path) -> dict:
    try:
        raw = Path(path).read_bytes()
    except FileNotFoundError as exc:
        raise ConfigError(f"Config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"Cannot read config {path}: {exc}") from exc
    try:
        return tomllib.loads(raw.removeprefix(_UTF8_BOM).decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid UTF-8 ({exc}). Re-save it as UTF-8 "
            "(VS Code: 'Save with Encoding'; Notepad: Encoding = UTF-8).") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"Invalid TOML in {path}: {exc}") from exc


def load_config(path: Path) -> AppConfig:
    """Parse + validate config.toml. Raises ConfigError with a precise message."""
    data = _read_toml(Path(path))
    try:
        return AppConfig.model_validate(data)
    except Exception as exc:  # pydantic ValidationError → our typed error
        raise ConfigError(f"Invalid config {path}:\n{exc}") from exc


def load_watchlist(path: Path) -> Watchlist:
    """Parse + validate channels.toml.

    Accepts ``[[channels]]`` array-of-tables. The Authorization Law lives at
    this boundary: only operator-listed channels are ever touched.
    """
    data = _read_toml(Path(path))
    try:
        return Watchlist.model_validate(data)
    except Exception as exc:
        raise ConfigError(f"Invalid watchlist {path}:\n{exc}") from exc
