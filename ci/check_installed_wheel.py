"""Smoke-test the installed wheel from a clean directory using Python -I."""

import csv
import json
import subprocess
import sys
import sysconfig
import tempfile
from importlib.metadata import distribution
from pathlib import Path


def main():
    """Verify installed metadata, backend-free discovery, CLIs, and report output."""
    import versa
    from versa.metric_discovery import create_metric_discovery_registry
    from versa.metric_registry import _metric_specs_by_name
    from versa.prompt_bank import list_protocols, render_protocol, validate_bank
    from versa.reporting import analyze_records

    dist = distribution("versa-speech-audio-toolkit")
    installed = Path(dist.locate_file("versa/__init__.py")).resolve()
    assert Path(versa.__file__).resolve() == installed, versa.__file__
    assert Path(sys.prefix).resolve() in installed.parents, installed
    assert versa.__version__ == dist.version

    registry = create_metric_discovery_registry(include_runtime_imports=False)
    names = set(registry.list_metrics()) | set(registry.list_aliases())
    assert {"pesq", "mapss", "speaker", "spk_similarity"} <= names
    assert not any("/test/" in str(path) for path in dist.files)
    assert any(name.startswith("qwen2_audio_") for name in names)
    assert any(name.startswith("qwen_omni_") for name in names)
    assert names <= _metric_specs_by_name().keys()
    bank = validate_bank()
    assert Path(bank.protocols[0].source_file).suffix == ".yaml"
    assert len(list_protocols()) == len(bank.protocols) >= 8
    rendered = render_protocol("speech.emotion.v1")
    assert rendered.text.endswith("\n") and "unclear" in rendered.text
    assert rendered.bank_schema_version == bank.schema_version
    rows = [{"key": "one", "smoke_score": 1.0}, {"key": "two", "smoke_score": 3.0}]
    assert analyze_records(rows)
    forbidden = {"torch", "torchaudio", "transformers"} & sys.modules.keys()
    forbidden.update(
        name
        for name in sys.modules
        if name.startswith(
            (
                "versa.utterance_metrics.",
                "versa.corpus_metrics.",
                "versa.sequence_metrics.",
            )
        )
    )
    assert not forbidden, sorted(forbidden)

    commands = {
        entry.name for entry in dist.entry_points if entry.group == "console_scripts"
    }
    expected = {"versa-score", "versa-aggregate", "versa-visualize"}
    assert expected <= commands
    scripts = Path(sysconfig.get_path("scripts"))

    def run(command, *args):
        """Run an installed console script with an isolated working directory."""
        return subprocess.run(
            [str(scripts / command), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        ).stdout

    with tempfile.TemporaryDirectory(prefix="versa-wheel-smoke-") as directory:
        import os

        os.chdir(directory)
        for command in sorted(expected):
            assert "usage:" in run(command, "--help").lower()
        listing = run("versa-score", "--list-metrics")
        assert "mapss" in listing and "qwen2_audio_" in listing
        source = Path("scores.jsonl")
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        run("versa-aggregate", str(source), "--out", "summary.csv")
        with Path("summary.csv").open(encoding="utf-8", newline="") as handle:
            table = list(csv.DictReader(handle))
        assert any(
            row.get("metric") == "smoke_score" and float(row["mean"]) == 2.0
            for row in table
        ), table
        run("versa-visualize", str(source), "--out", "report.html")
        assert "smoke_score" in Path("report.html").read_text(encoding="utf-8")
    print(
        f"Installed wheel {dist.version}: {len(names)} metric names/aliases, "
        f"{len(bank.protocols)} prompt-bank protocols, all CLIs and reports passed"
    )


if __name__ == "__main__":
    main()
