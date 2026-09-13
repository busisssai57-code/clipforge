"""Exercise draft edits against actual sidecar files."""

import json

import pytest
from fastapi import HTTPException

from clipforge import web
from clipforge.paths import Workspace


@pytest.fixture()
def draft(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "workspace").ensure()
    monkeypatch.setattr(web, "_workspace", lambda: ws)
    clip = ws.clips / "sample.mp4"
    clip.write_bytes(b"fixture")
    return clip.with_suffix(".export.json")


def test_edit_keeps_platform_limits_and_tag_counts(draft):
    draft.write_text(json.dumps({"platforms": {"x": {"limit": 280}}}),
                     encoding="utf-8")
    result = web.edit_clip_text(web.ClipTextRequest(
        filename="sample.mp4", caption="x" * 281,
        hashtags=[" one ", "#two", "three", "four"]))
    saved = json.loads(draft.read_text(encoding="utf-8"))
    assert saved == result["pack"]
    assert saved["hashtags"] == ["#one", "#two", "#three", "#four"]
    assert saved["platforms"]["x"]["hashtags"] == ["#one", "#two", "#three"]
    assert len(saved["platforms"]["x"]["caption"]) <= 280
    assert saved["platforms"]["x"]["trimmed"]


@pytest.mark.parametrize("pack", [
    [], None, "draft", {"platforms": [1]},
    {"platforms": {"x": []}}, {"hashtags": [123]},
    {"platforms": {"x": {"limit": 0}}},
    {"platforms": {"x": {"limit": "bad"}}},
    {"platforms": {"x": {"limit": True}}},
])
def test_corrupt_draft_is_reported_and_preserved(draft, pack):
    previous = json.dumps(pack)
    draft.write_text(previous, encoding="utf-8")
    with pytest.raises(HTTPException) as error:
        web.edit_clip_text(web.ClipTextRequest(filename="sample.mp4", title="edit"))
    assert error.value.status_code == 422
    assert draft.read_text(encoding="utf-8") == previous
