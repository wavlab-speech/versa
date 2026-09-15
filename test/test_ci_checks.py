"""Ensure CI cannot report skipped lanes or unpinned model assets as validated."""

import json
import sys
from types import SimpleNamespace

import pytest

from ci.check_test_report import check_report
from ci import prepare_wavlm_smoke


@pytest.mark.parametrize(
    "body,message",
    [
        ("", "no executed"),
        ("<testcase><skipped/></testcase>", "skip"),
        ("<testcase><failure/></testcase>", "failures"),
        ("<testcase><error/></testcase>", "errors"),
    ],
)
def test_incomplete_junit_report_fails(tmp_path, body, message):
    """Reject empty, skipped, failed, and collection-error reports."""
    path = tmp_path / "report.xml"
    path.write_text(f"<testsuites><testsuite>{body}</testsuite></testsuites>")
    with pytest.raises(ValueError, match=message):
        check_report(path)


def test_executed_junit_report_passes(tmp_path):
    """Accept pytest's successful report shape with an empty testcase element."""
    path = tmp_path / "report.xml"
    path.write_text(
        '<testsuites><testsuite><testcase name="test_ok"/></testsuite></testsuites>'
    )
    check_report(path)


@pytest.mark.parametrize("revision", ["main", "abc123", "x" * 40, "a" * 41])
def test_model_preparation_rejects_unpinned_revision(tmp_path, revision):
    """Reject mutable refs before importing the model downloader or writing files."""
    with pytest.raises(ValueError, match="commit SHA"):
        prepare_wavlm_smoke.prepare(revision, str(tmp_path), tmp_path / "manifest.json")
    assert not list(tmp_path.iterdir())


def test_model_manifest_matches_downloaded_snapshot(tmp_path, monkeypatch):
    """Preserve the exact requested revision, cache, snapshot, and dependency versions."""
    calls = []

    def download(**kwargs):
        """Record the requested assets without downloading a model."""
        calls.append(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=download)
    )
    monkeypatch.setattr(prepare_wavlm_smoke, "version", lambda name: "test-version")
    manifest = tmp_path / "manifest.json"
    prepare_wavlm_smoke.prepare("A" * 40, str(tmp_path), manifest)
    result = json.loads(manifest.read_text())
    assert calls == [
        {
            "repo_id": "microsoft/wavlm-base-sv",
            "revision": "a" * 40,
            "cache_dir": str(tmp_path),
            "allow_patterns": ["*.json", "*.safetensors", "*.bin"],
        }
    ]
    assert result["revision"] == "a" * 40
    assert result["snapshot_path"] == str(tmp_path / "snapshot")
    assert result["packages"]["transformers"] == "test-version"


@pytest.mark.parametrize("snapshot", [None, "/tmp/wavlm-cache/snapshots/pinned"])
def test_prepared_snapshot_config(monkeypatch, snapshot):
    """Route a prepared snapshot to the HF backend and preserve local defaults."""
    from test.test_pipeline.test_speaker_wavlm import _load_wavlm_speaker_config

    monkeypatch.delenv("VERSA_WAVLM_MODEL_PATH", raising=False)
    original = _load_wavlm_speaker_config()
    if snapshot:
        monkeypatch.setenv("VERSA_WAVLM_MODEL_PATH", snapshot)
    config = _load_wavlm_speaker_config()
    if snapshot:
        assert config == [
            {**metric, "model_tag": snapshot, "backend": "huggingface"}
            for metric in original
        ]
    else:
        assert config == original
