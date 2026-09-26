"""Telegram delivery of accepted clips.

Offline: every HTTP call goes through an injected ``post`` that records
what would have been sent. The properties pinned are the ones that matter
on a phone: a clip arrives once, a failure is retried rather than lost,
a blocked bot does not burn an upload every retry, and the token never
reaches a log or the outbox.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clipforge import notify
from clipforge.notify import TelegramTarget

TOKEN = "123456:AAFAKE-token-for-tests-only"
TARGET = TelegramTarget(token=TOKEN, chat_id="42", source="env")
JPEG_MAGIC = bytes([0xFF, 0xD8]) + b"jpeg"


class Resp:
    def __init__(self, body):
        self._body = body

    def json(self):
        return self._body


class FakeTelegram:
    """Records calls; answers per method."""

    def __init__(self, **answers):
        self.calls: list[tuple[str, dict, bool]] = []
        self.answers = answers

    def __call__(self, url, data=None, files=None, timeout=None):
        method = url.rsplit("/", 1)[-1]
        assert TOKEN in url  # the token travels in the URL, and only there
        self.calls.append((method, dict(data or {}), files is not None))
        answer = self.answers.get(method, {"ok": True, "result": {"message_id": 7}})
        if isinstance(answer, Exception):
            raise answer
        return Resp(answer)

    def methods(self) -> list[str]:
        return [c[0] for c in self.calls]


@pytest.fixture()
def clip(tmp_path) -> Path:
    clips = tmp_path / "ws" / "clips"
    clips.mkdir(parents=True)
    path = clips / "abc.mp4"
    path.write_bytes(b"\x00" * 1024)
    (clips / "abc.export.json").write_text(json.dumps({
        "title": "He did not see it coming", "caption": "Watch till the end",
        "hashtags": ["speed", "#live"]}), encoding="utf-8")
    return path


def ws_root(clip: Path) -> Path:
    return clip.parent.parent


# ------------------------------------------------------------------ sending

def test_a_clip_is_sent_once_with_its_export_pack_caption(clip):
    tg = FakeTelegram()
    assert notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg) == "sent"
    assert tg.methods() == ["sendChatAction", "sendVideo"]
    _, data, has_file = tg.calls[1]
    assert has_file and data["chat_id"] == "42"
    assert "He did not see it coming" in data["caption"]
    assert "#speed #live" in data["caption"]
    assert clip.with_suffix(".telegram.json").is_file()

    # Re-running process on the same source reuses the cached render: the
    # phone must not buzz twice for one clip.
    assert notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg) == "already_sent"
    assert tg.methods() == ["sendChatAction", "sendVideo"]


def test_again_resends_a_delivered_clip(clip):
    tg = FakeTelegram()
    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg)
    assert notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                            again=True, post=tg) == "sent"
    assert tg.methods().count("sendVideo") == 2


def test_a_blocked_bot_queues_without_uploading(clip):
    """Measured: a 26 MB upload took 57 s only to be refused. The cheap
    chat-action probe must stop the upload from happening at all."""
    tg = FakeTelegram(sendChatAction={
        "ok": False, "error_code": 403,
        "description": "Forbidden: bot was blocked by the user"})
    outcome = notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg)
    assert outcome == "queued"
    assert "sendVideo" not in tg.methods()
    entry = json.loads((ws_root(clip) / "outbox" / "telegram" / "abc.json")
                       .read_text(encoding="utf-8"))
    assert "blocked" in entry["error"]
    assert not clip.with_suffix(".telegram.json").exists()


def test_a_network_failure_is_queued_and_the_token_is_redacted(clip):
    # requests embeds the full URL - token included - in its error text.
    boom = ConnectionError(f"Max retries exceeded with url: /bot{TOKEN}/sendVideo")
    tg = FakeTelegram(sendVideo=boom)
    assert notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg) == "queued"
    raw = (ws_root(clip) / "outbox" / "telegram" / "abc.json").read_text(encoding="utf-8")
    assert TOKEN not in raw, "the bot token was written to the outbox"
    assert "<token>" in raw


def test_a_clip_is_owed_to_the_phone_before_the_upload_starts(clip):
    """A kill mid-upload (Ctrl+C, or this box's hard power-offs) used to
    lose the delivery: nothing wrote the outbox entry until a FAILURE, and
    the marker is only written ~57 s later when the upload finishes."""
    seen: list[dict] = []

    def snapshot_then_die(url, data=None, files=None, timeout=None):
        if url.endswith("sendVideo"):
            entry = ws_root(clip) / "outbox" / "telegram" / "abc.json"
            seen.append(json.loads(entry.read_text(encoding="utf-8")))
            raise KeyboardInterrupt("Ctrl+C mid-upload")
        return Resp({"ok": True, "result": {}})

    try:
        notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                         post=snapshot_then_die)
    except KeyboardInterrupt:
        pass
    assert seen and seen[0]["state"] == "sending", (
        "the clip was not recorded as owed before the upload began")
    assert seen[0]["clip"].endswith("abc.mp4")


def test_a_second_sender_cannot_claim_a_clip_being_sent(clip):
    """`bta telegram --retry` is documented as safe to run while watch is
    going, and both walk the same outbox. Without an exclusive claim each
    sees no marker and uploads the same clip — ~57 s of uplink each, and
    the clip arrives twice.

    The claim is an OS lock, so a second PROCESS is what it guards; here a
    second handle on the same file stands in for it, which is the same
    kernel check.
    """
    with notify._claim(ws_root(clip), clip) as mine:
        assert mine
        with notify._claim(ws_root(clip), clip) as second:
            assert not second, "two senders both claimed the same clip"
    # Released afterwards, or the next retry could never send it.
    with notify._claim(ws_root(clip), clip) as later:
        assert later


def test_a_clip_already_being_sent_is_not_sent_again(clip):
    held = notify._claim(ws_root(clip), clip)
    assert held.__enter__()
    try:
        import threading

        outcome: list[str] = []
        # A thread, because _SEND_LOCK already serialises this process and
        # the claim is what protects against the OTHER process.
        notify._SEND_LOCK = threading.Lock()
        t = threading.Thread(target=lambda: outcome.append(
            notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                             post=FakeTelegram())))
        t.start()
        t.join(timeout=10)
        assert outcome == ["in_flight"], outcome
    finally:
        held.__exit__(None, None, None)


def test_retries_count_attempts_and_success_clears_the_outbox(clip):
    down = FakeTelegram(sendVideo=ConnectionError("offline"))
    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=down)
    notify.flush_outbox(target=TARGET, ws_root=ws_root(clip), post=down)
    entry = ws_root(clip) / "outbox" / "telegram" / "abc.json"
    assert json.loads(entry.read_text(encoding="utf-8"))["attempts"] == 2

    up = FakeTelegram()
    counts = notify.flush_outbox(target=TARGET, ws_root=ws_root(clip), post=up)
    assert counts == {"sent": 1}
    assert not entry.exists()
    # The caption survives the round trip through the outbox.
    assert "He did not see it coming" in up.calls[1][1]["caption"]


def test_the_outbox_forgets_clips_that_were_deleted(clip):
    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                     post=FakeTelegram(sendVideo=ConnectionError("offline")))
    clip.unlink()
    tg = FakeTelegram()
    assert notify.flush_outbox(target=TARGET, ws_root=ws_root(clip), post=tg) == {"missing": 1}
    assert tg.calls == []
    assert not list((ws_root(clip) / "outbox" / "telegram").glob("*.json"))


def test_no_credentials_queues_so_the_clip_arrives_once_configured(clip):
    assert notify.send_clip(clip, target=None, ws_root=ws_root(clip)) == "queued"
    tg = FakeTelegram()
    assert notify.flush_outbox(target=TARGET, ws_root=ws_root(clip), post=tg) == {"sent": 1}


def test_an_oversized_clip_becomes_a_message_not_a_silent_skip(clip, monkeypatch):
    monkeypatch.setattr(notify, "MAX_UPLOAD_BYTES", 100)
    tg = FakeTelegram()
    assert notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg) == "sent"
    assert tg.methods() == ["sendChatAction", "sendMessage"]
    assert "50 MB" in tg.calls[1][1]["text"]


def test_a_missing_clip_is_reported_not_raised(tmp_path):
    assert notify.send_clip(tmp_path / "nope.mp4", target=TARGET,
                            ws_root=tmp_path, post=FakeTelegram()) == "missing"


def test_captions_are_fitted_to_telegrams_limit(clip):
    (clip.with_suffix(".export.json")).write_text(
        json.dumps({"title": "x" * 3000}), encoding="utf-8")
    assert len(notify.caption_for(clip)) <= notify.CAPTION_LIMIT


# -------------------------------------------------------------- credentials

def write_openclaw(tmp_path: Path, token: str | None, ids: list[str]) -> Path:
    cfg = tmp_path / "openclaw" / "openclaw.json"
    (cfg.parent / "credentials").mkdir(parents=True)
    tg = {"accounts": {"default": {"botToken": token}}} if token else {}
    cfg.write_text(json.dumps({"channels": {"telegram": tg}}), encoding="utf-8")
    (cfg.parent / "credentials" / "telegram-allowFrom.json").write_text(
        json.dumps({"version": 1, "allowFrom": ids}), encoding="utf-8")
    return cfg


def test_openclaw_supplies_token_and_chat(tmp_path):
    cfg = write_openclaw(tmp_path, TOKEN, ["99"])
    target = notify.resolve_target(cfg)
    assert target == TelegramTarget(token=TOKEN, chat_id="99", source="openclaw")


def test_env_values_win_over_openclaw(tmp_path):
    cfg = write_openclaw(tmp_path, "other:token", ["99"])
    target = notify.resolve_target(cfg, token=TOKEN, chat_id="42")
    assert (target.token, target.chat_id, target.source) == (TOKEN, "42", "env")


def test_two_allowed_chats_is_not_guessed(tmp_path):
    cfg = write_openclaw(tmp_path, TOKEN, ["1", "2"])
    assert notify.resolve_target(cfg) is None


def test_nothing_configured_is_none(tmp_path):
    assert notify.resolve_target(tmp_path / "missing.json") is None


def test_check_redacts_the_token_from_errors():
    def get(url, timeout=None):
        raise ConnectionError(f"failed: {url}")

    with pytest.raises(ValueError) as err:
        notify.check(TARGET, get=get)
    assert TOKEN not in str(err.value)


# --------------------------------------------------- what lands on the phone

def test_a_portrait_clip_is_sent_as_portrait(clip, monkeypatch):
    """Without width/height/duration Telegram guesses, and a 9:16 clip can
    arrive letterboxed inside a landscape box."""
    monkeypatch.setattr(notify, "video_fields",
                        lambda c: {"width": "1080", "height": "1920",
                                   "duration": "38"})
    tg = FakeTelegram()
    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=tg)
    _, data, _ = tg.calls[1]
    assert (data["width"], data["height"], data["duration"]) == ("1080", "1920", "38")


def test_a_failed_probe_still_sends_the_clip(clip, monkeypatch):
    def boom(_c):
        raise OSError("ffprobe missing")

    monkeypatch.setattr(notify, "video_fields", boom)
    tg = FakeTelegram()
    # The probe is a nicety; the delivery is the product.
    try:
        outcome = notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                                   post=tg)
    except OSError:
        raise AssertionError("a failed probe must not sink the delivery")
    assert outcome == "queued"       # it failed loudly, and kept the clip


def test_the_clips_own_thumbnail_rides_along(clip, monkeypatch):
    monkeypatch.setattr(notify, "thumbnail_bytes", lambda c: JPEG_MAGIC)
    sent: dict = {}

    def capture(url, data=None, files=None, timeout=None):
        if url.endswith("sendVideo"):
            sent.update(files or {})
        return Resp({"ok": True, "result": {"message_id": 3}})

    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip), post=capture)
    assert "thumbnail" in sent and sent["thumbnail"][1] == JPEG_MAGIC
    assert "video" in sent


def test_the_caption_is_the_title_not_a_content_hash(clip):
    caption = notify.caption_for(clip)
    assert caption.startswith("He did not see it coming")
    assert "abc.mp4" not in caption, (
        "a 64-char content hash tells the operator nothing")


def test_a_clip_with_no_export_pack_still_says_something(clip):
    clip.with_suffix(".export.json").unlink()
    assert notify.caption_for(clip) == clip.name


# ------------------------------------------------- private delivery only

def test_a_group_or_channel_id_is_refused(tmp_path):
    """The amendment allows one outbound path: the operator's own chat. A
    negative id is a group or channel — an audience — which is publishing
    by another name."""
    cfg = write_openclaw(tmp_path, TOKEN, ["-1001234567890"])
    assert notify.resolve_target(cfg) is None
    assert notify.resolve_target(cfg, token=TOKEN,
                                 chat_id="-1001234567890") is None


def test_a_personal_chat_id_is_accepted(tmp_path):
    cfg = write_openclaw(tmp_path, TOKEN, ["8675309"])
    target = notify.resolve_target(cfg)
    assert target is not None and target.chat_id == "8675309"


def test_a_nonsense_chat_id_is_refused(tmp_path):
    cfg = write_openclaw(tmp_path, TOKEN, ["@somechannel"])
    assert notify.resolve_target(cfg) is None


def test_the_outbox_is_empty_when_nothing_is_owed(clip):
    """Its whole purpose is to be empty. The per-clip lock file outlived
    the delivery, so one stayed behind for every clip ever sent."""
    box = ws_root(clip) / "outbox" / "telegram"
    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                     post=FakeTelegram())
    assert sorted(p.name for p in box.iterdir()) == [], (
        "a delivered clip left files behind in the outbox")


def test_a_vanished_clip_leaves_nothing_behind(clip):
    notify.send_clip(clip, target=TARGET, ws_root=ws_root(clip),
                     post=FakeTelegram(sendVideo=ConnectionError("offline")))
    clip.unlink()
    notify.flush_outbox(target=TARGET, ws_root=ws_root(clip),
                        post=FakeTelegram())
    box = ws_root(clip) / "outbox" / "telegram"
    assert list(box.iterdir()) == []
