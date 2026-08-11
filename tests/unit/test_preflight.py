"""Doctor checks are total (never raise) and messages are actionable."""

from pathlib import Path

from clipforge import preflight


def test_all_checks_are_total(tmp_path: Path, monkeypatch):
    """run_all must never raise, whatever the machine looks like.
    Network probes are stubbed — unit tests stay hermetic."""
    monkeypatch.setattr(preflight, "_probe_gated_repo", lambda repo, token: None)
    results = preflight.run_all(tmp_path / "ws", disk_floor_gb=1.0)
    assert results, "doctor produced no results"
    for r in results:
        assert r.name and r.message
        assert r.severity in ("required", "optional")
        if not r.ok:
            # Every failure must tell the operator what to DO.
            assert r.fix, f"check {r.name} failed without a fix hint"


def test_render_reports_verdict(tmp_path: Path):
    results = [
        preflight.CheckResult("good", True, "required", "fine"),
        preflight.CheckResult("bad", False, "required", "broken", fix="do X"),
        preflight.CheckResult("meh", False, "optional", "degraded", fix="do Y"),
    ]
    text, ok = preflight.render(results)
    assert not ok
    assert "FAIL" in text and "WARN" in text and "PASS" in text
    assert "do X" in text and "do Y" in text

    text2, ok2 = preflight.render([results[0]])
    assert ok2 and "all required checks passed" in text2


def test_workspace_writable_check(tmp_path: Path):
    r = preflight.check_workspace_writable(tmp_path / "new_ws")
    assert r.ok
    assert (tmp_path / "new_ws").exists()


def test_disk_check_floor(tmp_path: Path):
    r_ok = preflight.check_disk(tmp_path, floor_gb=0.001)
    assert r_ok.ok
    r_fail = preflight.check_disk(tmp_path, floor_gb=10_000_000.0)
    assert not r_fail.ok and r_fail.fix


def test_disk_check_relative_nonexistent_path():
    """The workspace may be a relative path that doesn't exist yet — the
    check must walk to an existing ancestor, not fail on an empty anchor."""
    r = preflight.check_disk(Path("some_ws_that_does_not_exist"), floor_gb=0.001)
    assert r.ok


# ---------------------------------------------------------------- T5 probing


def _with_token(monkeypatch):
    monkeypatch.setenv("CLIPFORGE_HF_TOKEN", "hf_fake_token_for_tests")


def test_hf_token_missing_fails_with_both_urls(monkeypatch):
    monkeypatch.delenv("CLIPFORGE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    r = preflight.check_hf_token(probe=lambda repo, token: True)
    assert not r.ok
    for repo in preflight.PYANNOTE_GATED_REPOS:
        assert repo in r.fix, "fix must list the exact URLs to accept (T5)"


def test_hf_token_present_but_terms_not_accepted_fails(monkeypatch):
    """THE T5 scenario: valid token, never clicked accept. Doctor must fail
    now, not twenty minutes into the first S1 run."""
    _with_token(monkeypatch)
    r = preflight.check_hf_token(probe=lambda repo, token: False)
    assert not r.ok
    assert "DENIED" in r.message
    for repo in preflight.PYANNOTE_GATED_REPOS:
        assert repo in r.fix


def test_hf_token_offline_passes_with_caveat(monkeypatch):
    _with_token(monkeypatch)
    r = preflight.check_hf_token(probe=lambda repo, token: None)
    assert r.ok
    assert "could not be verified" in r.message


def test_hf_token_fully_verified(monkeypatch):
    _with_token(monkeypatch)
    r = preflight.check_hf_token(probe=lambda repo, token: True)
    assert r.ok
    assert "accessible" in r.message


def test_module_probe_stub_actually_intercepts(monkeypatch):
    """Round-2 finding: a def-time probe default made monkeypatching the
    module attribute ineffective (live HTTPS from 'hermetic' tests). The
    probe must now resolve at CALL time."""
    _with_token(monkeypatch)
    calls = []
    monkeypatch.setattr(preflight, "_probe_gated_repo",
                        lambda repo, token: calls.append(repo) or True)
    r = preflight.check_hf_token()  # no explicit probe arg
    assert r.ok
    assert sorted(calls) == sorted(preflight.PYANNOTE_GATED_REPOS)


def test_render_output_is_ascii_safe(tmp_path: Path):
    """Round-2 finding: doctor output crashed cp1252 pipes (U+2192). The
    rendered report must stay ASCII-encodable forever."""
    results = [
        preflight.CheckResult("bad", False, "required", "broken",
                              fix="do X\nthen Y"),
        preflight.CheckResult("good", True, "required", "fine"),
    ]
    text, _ = preflight.render(results)
    text.encode("ascii")  # raises UnicodeEncodeError on regression


def test_real_check_output_is_ascii_safe(tmp_path: Path, monkeypatch):
    """Round-3 finding: non-ASCII lived in actual check MESSAGE/FIX strings,
    which the scaffold-only test above missed. Render the REAL checks on a
    machine where most of them fail (maximum fix-text coverage) and require
    the whole report to be ASCII."""
    monkeypatch.setattr(preflight, "_probe_gated_repo", lambda repo, token: None)
    monkeypatch.delenv("CLIPFORGE_HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    text, _ = preflight.render(
        preflight.run_all(tmp_path / "ws", disk_floor_gb=10_000_000.0))
    text.encode("ascii")  # any non-ASCII in any check string fails here
