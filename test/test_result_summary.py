"""Regression coverage for shared result aggregation without model downloads."""

import csv
import json
import subprocess
import sys

import numpy as np
import pytest

from versa.bin.aggregate_results import aggregate_results
from versa.reporting import (
    analyze_records,
    write_csv_report,
    write_html_report,
    write_markdown_report,
)
from versa.result_summary import compute_summary
from versa.scorer_shared import compute_summary as scorer_summary


def test_partial_results_share_policy_across_entrypoints(tmp_path):
    """Nulls, heterogeneous fields, metadata and repeated keys retain consistent counts."""
    records = [
        {
            "key": "a",
            "pesq": 2,
            "bad": None,
            "whisper_wer_insert": 1,
            "custom_wer": 0.1,
            "status": 1,
            "seed": 12,
            "_private": 9,
        },
        {
            "key": "a",
            "pesq": None,
            "bad": float("nan"),
            "whisper_wer_insert": 3,
            "custom_wer": 0.3,
            "later": 8,
        },
        {"key": "c", "pesq": 4, "bad": float("inf"), "later": "bad"},
        {"key": "d", "pesq": "bad", "later": False},
        {"key": "e", "later": {"value": 20}},
    ]
    expected = {"pesq": 3, "whisper_wer_insert": 4, "custom_wer": 0.2, "later": 8}
    assert compute_summary(records) == expected == scorer_summary(records)
    analysis = analyze_records(records)
    assert analysis["summary"] == expected
    metrics = {item.name: item for item in analysis["metrics"]}
    assert (
        metrics["pesq"].count,
        metrics["pesq"].missing,
        metrics["pesq"].invalid,
    ) == (2, 1, 2)
    assert metrics["bad"].aggregate is None
    assert (metrics["bad"].missing, metrics["bad"].invalid) == (2, 3)
    assert metrics["whisper_wer_insert"].aggregate == 4
    assert metrics["whisper_wer_insert"].mean == 2
    logdir = tmp_path / "logs"
    logdir.mkdir()
    (logdir / "result.1.txt").write_text(
        "\n".join(json.dumps(row) for row in records[:2]) + "\n\n"
    )
    (logdir / "result.2.txt").write_text(
        "\n".join(json.dumps(row) for row in records[2:])
    )
    out = tmp_path / "scores"
    aggregate_results(str(logdir), str(out), 2)
    exported = [
        json.loads(line, parse_constant=reject_constant)
        for line in (out / "utt_result.txt").read_text().splitlines()
    ]
    assert [row["key"] for row in exported] == [row["key"] for row in records]
    assert exported[1]["bad"] is None
    assert analyze_records(exported)["summary"] == expected
    averages = dict(
        line.split(": ") for line in (out / "avg_result.txt").read_text().splitlines()
    )
    assert {key: float(value) for key, value in averages.items()} == expected


def reject_constant(value):
    """Reject the nonstandard JSON NaN and Infinity extensions in exported records."""
    raise AssertionError(value)


@pytest.mark.parametrize("unit", ["wer", "cer", "per"])
def test_only_explicit_operation_counts_are_summed(unit):
    """Error rates and unrelated names must not inherit the count reducer."""
    row = {
        f"custom_{unit}_{operation}": 2
        for operation in ("insert", "delete", "replace", "equal")
    }
    row.update(
        {f"custom_{unit}": 0.5, f"custom_{unit}_rate": 0.5, f"custom_{unit}_other": 0.5}
    )
    summary = compute_summary([row, row])
    for key, value in row.items():
        assert summary[key] == (4 if value == 2 else 0.5)


def test_numpy_scalars_and_numeric_group_metadata():
    """Scorer and report accept numpy real scalars and exclude the grouping field."""
    records = [
        {"key": "a", "pesq": np.float32(2), "system": 1},
        {"key": "b", "pesq": np.int64(4), "system": 2},
    ]
    analysis = analyze_records(records, group_by="system")
    assert analysis["summary"] == {"pesq": 3}
    assert set(analysis["groups"]["rankings"]) == {"pesq"}


@pytest.mark.parametrize("records", [[], [{"key": "a", "pesq": None}]])
def test_empty_and_all_invalid_exports(tmp_path, records):
    """Every report format can represent an empty or entirely failed run."""
    analysis = analyze_records(records)
    assert analysis["summary"] == {}
    for writer, suffix in [
        (write_csv_report, "csv"),
        (write_markdown_report, "md"),
        (write_html_report, "html"),
    ]:
        path = tmp_path / f"report.{suffix}"
        writer(analysis, str(path))
        assert path.stat().st_size
    (tmp_path / "result.1.txt").write_text("")
    aggregate_results(str(tmp_path), str(tmp_path / "out"), 1)
    assert (tmp_path / "out" / "utt_result.txt").read_text() == ""
    assert (tmp_path / "out" / "avg_result.txt").read_text() == ""


def test_malformed_record_fails_before_overwriting(tmp_path):
    """Parsing errors identify the source line and preserve existing output files."""
    (tmp_path / "result.1.txt").write_text('{"key":"ok"}\n{"key":')
    (tmp_path / "utt_result.txt").write_text("keep")
    with pytest.raises(ValueError, match=r"result\.1\.txt:2"):
        aggregate_results(str(tmp_path), str(tmp_path), 1)
    assert (tmp_path / "utt_result.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="at least 1"):
        aggregate_results(str(tmp_path), str(tmp_path), 0)


def test_report_exposes_count_reducer(tmp_path):
    """Exports distinguish corpus operation totals from per-observation means."""
    analysis = analyze_records([{"wer_insert": 1}, {"wer_insert": 3}])
    path = tmp_path / "report.csv"
    write_csv_report(analysis, str(path))
    with path.open() as handle:
        row = next(csv.DictReader(handle))
    assert row["reducer"] == "sum"
    assert float(row["aggregate"]) == 4
    assert float(row["mean"]) == 2


def test_shared_summary_and_reporting_do_not_import_backends():
    """A fresh process must summarize and report without importing model packages."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from versa.reporting import analyze_records; import sys; assert analyze_records([{'score': 2}])['summary'] == {'score': 2}; assert not {'torch', 'transformers', 'versa.scorer_shared'} & sys.modules.keys()",
        ],
        check=True,
        timeout=10,
    )
