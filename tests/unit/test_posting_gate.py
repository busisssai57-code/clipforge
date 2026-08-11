"""The draft-only human gate on publishing.

The spec forbids publishing outright (§10 Non-goals; §3.5 "produces files
and stops"). The operator amended that on 2026-07-27 to permit posting
under three conditions. Those conditions are the amendment — if they are
not mechanically enforced, the amendment is just a comment. Every test here
fails if its pin is loosened.
"""

from pathlib import Path

import pytest

from clipforge.config import AppConfig, PostingConfig, load_config
from clipforge.errors import ConfigError
from clipforge.poster.scheduler import execute_post_job
from clipforge.schemas.poster import PostJob


def _job(**kw) -> PostJob:
    base = dict(job_id="j1", clip_path="clips/a.mp4", platform="youtube",
                title="t", caption="c")
    base.update(kw)
    return PostJob(**base)


# ----------------------------------------------------- pin 1: draft only


def test_public_publish_mode_cannot_be_represented():
    """Not a default that could be overridden — a type that cannot hold
    'public'. No config file, env var, or code path can select it."""
    with pytest.raises(Exception):
        PostJob(job_id="j", clip_path="c.mp4", platform="youtube",
                title="t", caption="c", publish_mode="public")
    assert _job().publish_mode == "draft"


def test_config_rejects_public_publish_mode(tmp_path: Path):
    p = tmp_path / "cfg.toml"
    p.write_text("[posting]\npublish_mode = 'public'\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


def test_default_posting_config_is_draft_only():
    cfg = AppConfig()
    assert cfg.posting.publish_mode == "draft"


# --------------------------------------------- pin 2: no auto-scheduling


def test_smart_scheduling_cannot_be_enabled():
    with pytest.raises(Exception, match="smart_scheduling"):
        PostingConfig(smart_scheduling=True)
    assert PostingConfig().smart_scheduling is False


def test_config_file_cannot_enable_scheduling(tmp_path: Path):
    p = tmp_path / "cfg.toml"
    p.write_text("[posting]\nsmart_scheduling = true\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(p)


# ------------------------------------------- pin 3: per-clip human approval


def test_approval_cannot_be_disabled_in_config():
    with pytest.raises(Exception, match="require_approval"):
        PostingConfig(require_approval=False)
    assert PostingConfig().require_approval is True


def test_dispatch_refuses_an_unapproved_job(tmp_path: Path):
    """The choke point between ClipForge and the outside world."""
    with pytest.raises(ConfigError, match="not approved"):
        execute_post_job(_job(), auth_dir=tmp_path)


def test_approval_is_keyword_only_and_defaults_false(tmp_path: Path):
    """Approval must be typed out per job — never satisfied by argument
    order or by a truthy value drifting in from config."""
    import inspect

    sig = inspect.signature(execute_post_job)
    param = sig.parameters["approved"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is False
    # Positional attempts cannot reach it.
    with pytest.raises(TypeError):
        execute_post_job(_job(), tmp_path, True, -5.0, True)  # type: ignore[misc]


def test_dispatch_rechecks_publish_mode_not_just_config(tmp_path: Path,
                                                        monkeypatch):
    """A hand-built job must not route around the config-level pin."""
    job = _job()
    object.__setattr__(job, "publish_mode", "public")  # bypass validation
    with pytest.raises(ConfigError, match="only 'draft'"):
        execute_post_job(job, auth_dir=tmp_path, approved=True)


def test_approved_draft_job_reaches_the_platform_automator(tmp_path: Path,
                                                           monkeypatch):
    """The positive arm: with approval and draft mode, dispatch proceeds —
    otherwise these pins could 'pass' by blocking everything."""
    import clipforge.poster.scheduler as sched
    from clipforge.schemas.poster import PostResult

    called: dict[str, object] = {}

    class FakePoster:
        platform = "youtube"

        def upload_clip(self, job, auth_dir, headless=True):
            called["job_id"] = job.job_id
            called["headless"] = headless
            return PostResult(job_id=job.job_id, platform="youtube",
                              status="draft_saved", completed_at="now")

    monkeypatch.setattr(sched, "get_poster", lambda p: FakePoster())
    result = execute_post_job(_job(), auth_dir=tmp_path, approved=True)

    assert called["job_id"] == "j1"
    assert result.status == "draft_saved", (
        "automation must leave a DRAFT for a human to publish")


def test_no_code_path_publishes_autonomously():
    """Structural sweep: nothing may pass publish_mode='public' anywhere."""
    import clipforge

    root = Path(clipforge.__file__).parent
    offenders: list[str] = []
    for py in root.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        if '"public"' in text or "'public'" in text:
            # Only the config/schema docstrings may MENTION it.
            for i, line in enumerate(text.splitlines(), 1):
                if ("public" in line and "=" in line
                        and "publish_mode" in line and "#" not in line.split("public")[0]):
                    offenders.append(f"{py.name}:{i}: {line.strip()}")
    assert not offenders, f"autonomous publish path found: {offenders}"
