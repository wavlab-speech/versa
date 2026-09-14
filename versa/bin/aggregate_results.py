#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Aggregate and report VERSA results."""

import argparse
import json
import logging
import math
import os

from tqdm import tqdm

from versa.result_summary import compute_summary

from versa.reporting import (
    analyze_records,
    read_result_records,
    write_csv_report,
    write_html_report,
    write_markdown_report,
)


def get_parser() -> argparse.Namespace:
    """Get parser of aggregate results."""
    parser = argparse.ArgumentParser(
        description="Aggregate chunked results or generate polished result reports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSONL result file or directory. If omitted, --logdir/--scoredir/--nj mode is used.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default=None,
        help="Input log directory.",
    )
    parser.add_argument(
        "--scoredir",
        type=str,
        default=None,
        help="Output scoring directory.",
    )
    parser.add_argument(
        "--nj",
        type=int,
        default=None,
        help="Number of sub jobs",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output report path for input mode.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "md", "html"],
        default="auto",
        help="Report format for input mode.",
    )
    parser.add_argument(
        "--group-by",
        default=None,
        help="Optional record field used for per-metric ranking.",
    )
    return parser


def _finite_json_float(token: str):
    """Convert a JSON float token, replacing overflow with null."""
    value = float(token)
    return value if math.isfinite(value) else None


def aggregate_results(logdir: str, scoredir: str, nj: int) -> None:
    """Combine numbered JSONL chunks using the shared finite-score reducers.

    Preserve row order and duplicates. Missing and invalid values do not enter
    reductions; named error-operation counts are summed. Fields without finite
    observations are omitted. Nonfinite JSON
    constants become null, including in nested results. Empty chunks produce
    empty output files. Reject malformed records before opening either output.
    """
    if nj < 1:
        raise ValueError("nj must be at least 1")
    logging.info("Aggregating results...")
    score_info = []
    for i in range(nj):
        path = os.path.join(logdir, f"result.{i + 1}.txt")
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(
                        line,
                        parse_constant=lambda value: None,
                        parse_float=_finite_json_float,
                    )
                except ValueError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object in {path}:{line_number}")
                score_info.append(record)
    summary = compute_summary(score_info)
    os.makedirs(scoredir, exist_ok=True)
    with open(
        os.path.join(scoredir, "utt_result.txt"), "w", encoding="utf-8"
    ) as f, open(os.path.join(scoredir, "avg_result.txt"), "w", encoding="utf-8") as f2:
        for info in tqdm(score_info):
            f.write(json.dumps(info, allow_nan=False) + "\n")
        for key, value in summary.items():
            f2.write(f"{key}: {value}\n")

    logging.info("Done.")


def generate_report(
    input_path: str,
    output_path: str,
    report_format: str = "auto",
    group_by: str = None,
) -> None:
    """Generate a CSV, Markdown, or HTML report from result records."""
    records = read_result_records(input_path)
    analysis = analyze_records(records, group_by=group_by)

    if report_format == "auto":
        suffix = os.path.splitext(output_path)[1].lower()
        report_format = {
            ".csv": "csv",
            ".md": "md",
            ".markdown": "md",
            ".html": "html",
            ".htm": "html",
        }.get(suffix, "csv")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    if report_format == "csv":
        write_csv_report(analysis, output_path)
    elif report_format == "md":
        write_markdown_report(analysis, output_path)
    else:
        write_html_report(analysis, output_path)

    logging.info(
        "Wrote %s report for %s utterances and %s metrics to %s",
        report_format,
        analysis["record_count"],
        analysis["metric_count"],
        output_path,
    )


def main() -> None:
    """Run main function."""
    parser = get_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    if args.input:
        output_path = args.out or "metrics_report.csv"
        generate_report(args.input, output_path, args.format, args.group_by)
        return

    if args.logdir is None or args.scoredir is None or args.nj is None:
        parser.error("either provide input, or provide --logdir, --scoredir, and --nj")

    aggregate_results(args.logdir, args.scoredir, args.nj)


if __name__ == "__main__":
    main()
