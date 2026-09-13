"""Regression tests for aggregation of numbered scoring chunks."""

import json

import pytest

from versa.bin.aggregate_results import aggregate_results
from versa.reporting import analyze_records, read_result_records


def test_partial_scores_round_trip_and_match_report_means(tmp_path):
    """Keep duplicate rows and structured values while averaging valid scores."""
    records = [
        {"key": "a", "pesq": None, "text": "it's quiet", "flag": True},
        {"key": "b", "pesq": 2, "later": 4, "detail": {"label": "clear"}},
        {"key": "b", "pesq": 4, "later": "failed"},
        {"key": "c", "pesq": "failed", "later": 8, "_status": 1},
    ]
    for index, chunk in enumerate((records[:2], records[2:]), start=1):
        (tmp_path / f"result.{index}.txt").write_text(
            "\n" + "\n".join(json.dumps(row) for row in chunk) + "\n",
            encoding="utf-8",
        )
    output = tmp_path / "scores"

    aggregate_results(str(tmp_path), str(output), 2)

    assert [
        json.loads(line)
        for line in (output / "utt_result.txt").read_text().splitlines()
    ] == records
    averages = dict(
        line.split(": ")
        for line in (output / "avg_result.txt").read_text().splitlines()
    )
    report = analyze_records(read_result_records(str(output / "utt_result.txt")))
    assert (
        {key: float(value) for key, value in averages.items()}
        == {summary.name: summary.mean for summary in report["metrics"]}
        == {"later": 6, "pesq": 3}
    )


def test_nonfinite_values_are_null_in_standard_json(tmp_path):
    """Normalize nonfinite constants and overflowing exponents, even when nested."""
    (tmp_path / "result.1.txt").write_text(
        '{"key":"a","score":NaN,"nested":[Infinity,-Infinity,1e999]}\n'
        '{"key":"b","score":2,"invalid":null}\n',
        encoding="utf-8",
    )
    aggregate_results(str(tmp_path), str(tmp_path), 1)
    lines = (tmp_path / "utt_result.txt").read_text().splitlines()
    assert json.loads(lines[0]) == {
        "key": "a",
        "score": None,
        "nested": [None, None, None],
    }
    assert "NaN" not in "".join(lines)
    assert "Infinity" not in "".join(lines)
    assert (tmp_path / "avg_result.txt").read_text() == "score: 2.0\n"


@pytest.mark.parametrize("content", ["", "\n\n", '{"key":"a","score":null}\n'])
def test_empty_or_all_invalid_scores_have_no_average(tmp_path, content):
    """Do not crash or invent a zero score when no valid observations exist."""
    (tmp_path / "result.1.txt").write_text(content, encoding="utf-8")
    aggregate_results(str(tmp_path), str(tmp_path), 1)
    assert (tmp_path / "avg_result.txt").read_text() == ""
    assert len((tmp_path / "utt_result.txt").read_text().splitlines()) == (
        1 if content.strip() else 0
    )


@pytest.mark.parametrize("bad_line", ['{"key":', "[]", "null"])
def test_bad_record_does_not_overwrite_outputs(tmp_path, bad_line):
    """Identify the failing chunk and line before touching existing results."""
    (tmp_path / "result.1.txt").write_text(
        '{"key":"a","score":2}\n' + bad_line + "\n", encoding="utf-8"
    )
    for name in ("utt_result.txt", "avg_result.txt"):
        (tmp_path / name).write_text("existing", encoding="utf-8")
    with pytest.raises(ValueError, match=r"result\.1\.txt:2"):
        aggregate_results(str(tmp_path), str(tmp_path), 1)
    for name in ("utt_result.txt", "avg_result.txt"):
        assert (tmp_path / name).read_text() == "existing"


@pytest.mark.parametrize("nj", [0, -1])
def test_invalid_job_count(tmp_path, nj):
    """Reject nonpositive job counts without creating output files."""
    with pytest.raises(ValueError, match="nj must be at least 1"):
        aggregate_results(str(tmp_path), str(tmp_path), nj)
    assert not (tmp_path / "utt_result.txt").exists()
