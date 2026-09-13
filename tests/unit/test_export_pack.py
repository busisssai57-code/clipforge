"""Export drafts must fit their limits and survive interrupted saves."""

import json

import pytest

from clipforge import export_pack, paths
from clipforge.errors import AtomicWriteError


@pytest.mark.parametrize("limit", [1, 2, 10, 280, 2200, 5000])
@pytest.mark.parametrize("text", ["x" * 6000, "word " * 1500, "字幕" * 3000],
                         ids=["unbroken", "words", "unicode"])
def test_trimmed_caption_including_ellipsis_fits(limit, text):
    fitted, trimmed = export_pack.fit_caption(text, limit)
    assert trimmed
    assert len(fitted) <= limit
    assert fitted.endswith("…")


def test_caption_at_limit_is_unchanged():
    assert export_pack.fit_caption("x" * 280, 280) == ("x" * 280, False)


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_caption_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="positive"):
        export_pack.fit_caption("draft", limit)


def test_pack_save_failure_preserves_previous_draft(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fixture")
    dest = clip.with_suffix(".export.json")
    previous = '{"title": "Approved draft"}'
    dest.write_text(previous, encoding="utf-8")
    monkeypatch.setattr(export_pack, "grab_thumbnail", lambda *a, **k: None)

    def fail_sync(fd):
        raise OSError("disk unavailable")

    monkeypatch.setattr(paths.os, "fsync", fail_sync)
    with pytest.raises(AtomicWriteError):
        export_pack.build_pack(clip, title="Replacement")
    assert dest.read_text(encoding="utf-8") == previous
    assert not list(tmp_path.glob("*.partial"))


def test_saved_pack_has_bounded_platform_payloads(tmp_path, monkeypatch):
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"fixture")
    monkeypatch.setattr(export_pack, "grab_thumbnail", lambda *a, **k: None)
    export_pack.build_pack(clip, transcript_text="x" * 6000)
    pack = json.loads(clip.with_suffix(".export.json").read_text(encoding="utf-8"))
    assert set(pack["platforms"]) == set(export_pack.PLATFORM_LIMITS)
    for platform in pack["platforms"].values():
        assert len(platform["caption"]) <= platform["limit"]
