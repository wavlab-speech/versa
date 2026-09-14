"""Reporting helpers for VERSA scoring results."""

import ast
import csv
import html
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from versa.definition import MetricRegistry
from versa.result_summary import (
    discover_numeric_metrics,
    is_count_metric,
    numeric_observations,
    reduce_values,
)

METRIC_CATEGORIES = {
    "audio_quality": [
        "dns_overall",
        "dns_p808",
        "nisqa",
        "utmos",
        "plcmos",
        "singmos",
        "sheet_ssqa",
        "utmosv2",
        "scoreq_nr",
        "scoreq_ref",
        "noresqa",
        "torch_squim_mos",
        "warpq",
        "dnsmos_pro_bvcc",
        "dnsmos_pro_nisqa",
        "dnsmos_pro_vcc2018",
    ],
    "speech_enhancement": [
        "torch_squim_pesq",
        "torch_squim_stoi",
        "torch_squim_si_sdr",
        "se_si_snr",
        "se_ci_sdr",
        "se_sar",
        "se_sdr",
        "pesq",
        "stoi",
        "sir",
        "sar",
        "sdr",
        "ci-sdr",
        "ci_sdr",
        "si-snr",
        "si_snr",
        "visqol",
    ],
    "psychoacoustic": [
        "pysepm_fwsegsnr",
        "pysepm_wss",
        "pysepm_cd",
        "pysepm_c_sig",
        "pysepm_c_bak",
        "pysepm_c_ovl",
        "pysepm_csii_high",
        "pysepm_csii_mid",
        "pysepm_csii_low",
        "pysepm_ncm",
        "pysepm_llr",
        "pam",
        "pam_score",
        "srmr",
    ],
    "asr_wer_cer": [
        "espnet_wer",
        "espnet_cer",
        "owsm_wer",
        "owsm_cer",
        "whisper_wer",
        "whisper_cer",
        "wer",
        "cer",
        "asr_match_error_rate",
    ],
    "semantic": [
        "speech_bert",
        "speech_bleu",
        "speech_token_distance",
        "clap_score",
    ],
    "similarity": ["emotion_similarity", "spk_similarity", "singer_similarity"],
    "pitch_f0": ["f0_corr", "f0corr", "f0_rmse", "f0rmse", "mcd"],
    "audio_features": ["speaking_rate", "log_wmse"],
    "aesthetics": [
        "audiobox_aesthetics_CE",
        "audiobox_aesthetics_CU",
        "audiobox_aesthetics_PC",
        "audiobox_aesthetics_PQ",
    ],
    "security": ["asvspoof_score", "nomad"],
}


@dataclass
class MetricSummary:
    """Hold finite-value statistics, completeness counts, extrema, and z-score outliers."""

    name: str
    category: str
    count: int
    missing: int
    invalid: int
    mean: float
    median: float
    std: float
    stderr: float
    ci95_low: float
    ci95_high: float
    minimum: float
    maximum: float
    higher_is_better: Optional[bool]
    best_key: str
    best_value: float
    worst_key: str
    worst_value: float
    outliers: List[Tuple[str, float, float]]
    reducer: str
    aggregate: Optional[float]


def read_result_records(input_path: str) -> List[Dict[str, Any]]:
    """Read VERSA result records from a file or directory."""
    paths = _collect_input_paths(input_path)
    records: List[Dict[str, Any]] = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    try:
                        record = ast.literal_eval(line)
                    except (SyntaxError, ValueError):
                        try:
                            record = _literal_eval_with_special_floats(line)
                        except (SyntaxError, ValueError) as literal_exc:
                            raise ValueError(
                                f"Could not parse {path}:{line_number} as JSON or Python literal"
                            ) from literal_exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected object in {path}:{line_number}")
                record = dict(record)
                record.setdefault("_source_file", path.name)
                records.append(record)
    return records


def analyze_records(
    records: Sequence[Dict[str, Any]],
    *,
    group_by: Optional[str] = None,
    outlier_limit: int = 3,
    registry: Optional[MetricRegistry] = None,
) -> Dict[str, Any]:
    """Summarize every row, retaining duplicate keys as separate observations.

    Empty inputs produce empty reports. The summary mapping follows the scorer's
    reducers; metric means remain descriptive per-observation statistics.
    """
    metrics = [name for name in discover_numeric_metrics(records) if name != group_by]
    metric_summaries = [
        summarize_metric(
            metric, records, outlier_limit=outlier_limit, registry=registry
        )
        for metric in sorted(metrics)
    ]
    categories: Dict[str, List[MetricSummary]] = defaultdict(list)
    for summary in metric_summaries:
        categories[summary.category].append(summary)

    groups = {}
    if group_by:
        groups = summarize_groups(records, metrics, group_by, registry=registry)

    return {
        "summary": {
            item.name: item.aggregate
            for item in metric_summaries
            if item.aggregate is not None
        },
        "records": records,
        "metrics": metric_summaries,
        "categories": dict(sorted(categories.items())),
        "groups": groups,
        "group_by": group_by,
        "record_count": len(records),
        "metric_count": len(metric_summaries),
    }


def summarize_metric(
    metric: str,
    records: Sequence[Dict[str, Any]],
    *,
    outlier_limit: int = 3,
    registry: Optional[MetricRegistry] = None,
) -> MetricSummary:
    """Summarize finite values of one metric across result records.

    Absent keys count as missing; nonnumeric and nonfinite values count as
    invalid. Statistics are zero when no valid values exist. Confidence limits
    use mean +/- 1.96 standard errors, and outliers have absolute z-score >= 2.
    Unknown score direction is ranked as higher-is-better for extrema."""
    observations, missing, invalid = numeric_observations(records, metric)
    values = [
        (str(records[index].get("key") or f"utt_{index + 1}"), value)
        for index, value in observations
    ]

    numeric = [value for _, value in values]
    count = len(numeric)
    mean = sum(value / count for value in numeric) if count else 0.0
    sorted_values = sorted(numeric)
    median = _median(sorted_values)
    std = _sample_std(numeric, mean)
    stderr = std / math.sqrt(count) if count else 0.0
    ci_delta = 1.96 * stderr
    minimum = min(numeric) if numeric else 0.0
    maximum = max(numeric) if numeric else 0.0
    higher_is_better = metric_direction(metric, registry=registry)

    if values and higher_is_better is False:
        best_key, best_value = min(values, key=lambda item: item[1])
        worst_key, worst_value = max(values, key=lambda item: item[1])
    elif values:
        best_key, best_value = max(values, key=lambda item: item[1])
        worst_key, worst_value = min(values, key=lambda item: item[1])
    else:
        best_key, best_value, worst_key, worst_value = "", 0.0, "", 0.0

    outliers = []
    if count > 1 and std > 0:
        scored = [
            (key, value, (value - mean) / std)
            for key, value in values
            if abs((value - mean) / std) >= 2.0
        ]
        outliers = sorted(scored, key=lambda item: abs(item[2]), reverse=True)[
            :outlier_limit
        ]

    return MetricSummary(
        name=metric,
        category=metric_category(metric, registry=registry),
        count=count,
        missing=missing,
        invalid=invalid,
        mean=mean,
        median=median,
        std=std,
        stderr=stderr,
        ci95_low=mean - ci_delta,
        ci95_high=mean + ci_delta,
        minimum=minimum,
        maximum=maximum,
        higher_is_better=higher_is_better,
        best_key=best_key,
        best_value=best_value,
        worst_key=worst_key,
        worst_value=worst_value,
        outliers=outliers,
        reducer="sum" if is_count_metric(metric) else "mean",
        aggregate=reduce_values(metric, numeric),
    )


def summarize_groups(
    records: Sequence[Dict[str, Any]],
    metrics: Sequence[str],
    group_by: str,
    registry: Optional[MetricRegistry] = None,
) -> Dict[str, Any]:
    """Group records by a field and rank group means separately for each metric.

    Missing group fields become ``unknown``. Groups with no finite values for
    a metric are omitted from its ranking; unknown direction sorts descending."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get(group_by, "unknown"))].append(record)

    metric_rankings = {}
    for metric in metrics:
        direction = metric_direction(metric, registry=registry)
        rows = []
        for group, group_records in grouped.items():
            summary = summarize_metric(
                metric, group_records, outlier_limit=0, registry=registry
            )
            if summary.count:
                rows.append(
                    {
                        "group": group,
                        "mean": summary.mean,
                        "std": summary.std,
                        "count": summary.count,
                    }
                )
        reverse = direction is not False
        metric_rankings[metric] = sorted(
            rows, key=lambda row: row["mean"], reverse=reverse
        )

    return {
        "sizes": {key: len(value) for key, value in grouped.items()},
        "rankings": metric_rankings,
    }


def write_csv_report(analysis: Dict[str, Any], output_path: str) -> None:
    """Overwrite a UTF-8 CSV with one row per metric from ``analyze_records``."""
    fields = [
        "metric",
        "reducer",
        "aggregate",
        "category",
        "count",
        "missing",
        "invalid",
        "mean",
        "median",
        "std",
        "stderr",
        "ci95_low",
        "ci95_high",
        "min",
        "max",
        "higher_is_better",
        "best_key",
        "best_value",
        "worst_key",
        "worst_value",
        "outliers",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for summary in analysis["metrics"]:
            writer.writerow(_summary_row(summary))


def write_markdown_report(analysis: Dict[str, Any], output_path: str) -> None:
    """Overwrite a UTF-8 Markdown summary and outlier report from an analysis."""
    lines = [
        "# VERSA Results Report",
        "",
        f"- Utterances: {analysis['record_count']}",
        f"- Metrics: {analysis['metric_count']}",
        "",
        "## Summary",
        "",
        "| Metric | Category | Count | Reducer | Aggregate | Mean | Std | 95% CI | Missing | Invalid | Best | Worst |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for summary in analysis["metrics"]:
        lines.append(
            "| {metric} | {category} | {count} | {reducer} | {aggregate} | {mean} | {std} | {ci} | {missing} | {invalid} | {best} | {worst} |".format(
                reducer=summary.reducer,
                aggregate=_fmt(summary.aggregate),
                metric=summary.name,
                category=summary.category,
                count=summary.count,
                mean=_fmt(summary.mean),
                std=_fmt(summary.std),
                ci=f"{_fmt(summary.ci95_low)} to {_fmt(summary.ci95_high)}",
                missing=summary.missing,
                invalid=summary.invalid,
                best=f"{summary.best_key} ({_fmt(summary.best_value)})",
                worst=f"{summary.worst_key} ({_fmt(summary.worst_value)})",
            )
        )
    lines.extend(["", "## Outlier Examples", ""])
    any_outliers = False
    for summary in analysis["metrics"]:
        if not summary.outliers:
            continue
        any_outliers = True
        lines.append(f"### {summary.name}")
        for key, value, z_score in summary.outliers:
            lines.append(f"- {key}: {_fmt(value)} (z={_fmt(z_score)})")
        lines.append("")
    if not any_outliers:
        lines.append("No z-score outliers >= 2.0 were detected.")
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def write_html_report(analysis: Dict[str, Any], output_path: str) -> None:
    """Overwrite a self-contained HTML report with summaries, charts, and rankings."""
    category_rows = []
    for category, summaries in analysis["categories"].items():
        expected_values = analysis["record_count"] * len(summaries)
        observed_values = sum(summary.count for summary in summaries)
        missing_values = sum(summary.missing for summary in summaries)
        invalid_values = sum(summary.invalid for summary in summaries)
        category_rows.append(
            {
                "category": category,
                "metrics": len(summaries),
                "observed": observed_values,
                "missing": missing_values,
                "invalid": invalid_values,
                "coverage": (
                    observed_values / expected_values if expected_values else 0.0
                ),
            }
        )

    top_metrics = sorted(
        analysis["metrics"], key=lambda item: item.count, reverse=True
    )[:12]
    radar_svg = _radar_svg(top_metrics)
    sunburst_svg = _sunburst_svg(category_rows)
    rows_html = "\n".join(_metric_html_row(summary) for summary in analysis["metrics"])
    category_html = "\n".join(
        f"<tr><td>{html.escape(row['category'])}</td><td>{row['metrics']}</td><td>{row['observed']}</td><td>{row['missing']}</td><td>{row['invalid']}</td><td>{_fmt(row['coverage'] * 100)}%</td></tr>"
        for row in category_rows
    )
    outlier_html = _outlier_html(analysis["metrics"])
    ranking_html = _ranking_html(analysis)

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VERSA Results Report</title>
<style>
:root {{ color-scheme: light; --ink: #18202a; --muted: #5d6b7a; --line: #d9e0e7; --panel: #f7f9fb; --accent: #1f8a70; --accent2: #c05621; }}
body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: #fff; }}
header {{ padding: 34px 40px 22px; border-bottom: 1px solid var(--line); }}
h1 {{ margin: 0 0 8px; font-size: 30px; letter-spacing: 0; }}
h2 {{ margin: 0 0 14px; font-size: 19px; }}
main {{ padding: 26px 40px 44px; }}
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }}
.stat {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
.stat strong {{ display: block; font-size: 25px; }}
.stat span, .muted {{ color: var(--muted); font-size: 13px; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; margin: 22px 0 26px; }}
.panel {{ border: 1px solid var(--line); border-radius: 8px; padding: 18px; overflow: auto; }}
table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }}
th {{ color: var(--muted); font-weight: 650; background: #fbfcfd; position: sticky; top: 0; }}
.metric-table td:nth-child(n+3), .metric-table th:nth-child(n+3) {{ text-align: right; }}
.metric-table td:first-child, .metric-table th:first-child {{ text-align: left; }}
svg text {{ font-family: inherit; fill: var(--muted); font-size: 11px; }}
.footer {{ color: var(--muted); font-size: 12px; margin-top: 20px; }}
</style>
</head>
<body>
<header>
<h1>VERSA Results Report</h1>
<div class="muted">Generated from scoring results with summary statistics, confidence intervals, rankings, failures, and outlier examples.</div>
</header>
<main>
<section class="summary">
<div class="stat"><strong>{analysis['record_count']}</strong><span>utterances</span></div>
<div class="stat"><strong>{analysis['metric_count']}</strong><span>numeric metrics</span></div>
<div class="stat"><strong>{len(analysis['categories'])}</strong><span>metric categories</span></div>
<div class="stat"><strong>{sum(s.missing + s.invalid for s in analysis['metrics'])}</strong><span>missing or invalid metric values</span></div>
</section>
<section class="grid">
<div class="panel"><h2>Radar Overview</h2>{radar_svg}</div>
<div class="panel"><h2>Category Sunburst</h2>{sunburst_svg}</div>
</section>
<section class="panel"><h2>Category Summary</h2><table><thead><tr><th>Category</th><th>Metrics</th><th>Observed Values</th><th>Missing</th><th>Invalid</th><th>Coverage</th></tr></thead><tbody>{category_html}</tbody></table></section>
{ranking_html}
<section class="panel"><h2>Metric Summary</h2><table class="metric-table"><thead><tr><th>Metric</th><th>Category</th><th>Count</th><th>Reducer</th><th>Aggregate</th><th>Mean</th><th>Std</th><th>95% CI</th><th>Missing</th><th>Invalid</th><th>Best</th><th>Worst</th></tr></thead><tbody>{rows_html}</tbody></table></section>
<section class="panel"><h2>Outlier Examples</h2>{outlier_html}</section>
<div class="footer">Tip: CSV and Markdown exports are available from the same CLI for downstream analysis.</div>
</main>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(document)


def metric_category(metric: str, registry: Optional[MetricRegistry] = None) -> str:
    """Infer a reporting category from registry metadata and metric-name heuristics."""
    metadata = _metadata_for_metric(metric, registry)
    if metadata and metadata.category.value in {"non_match", "distributional"}:
        return metadata.category.value
    normalized = _strip_prefix(metadata.name if metadata else metric).lower()
    for category, names in METRIC_CATEGORIES.items():
        if normalized in {name.lower() for name in names}:
            return category
    if any(token in normalized for token in ["wer", "cer"]):
        return "asr_wer_cer"
    if any(token in normalized for token in ["similarity", "sim"]):
        return "similarity"
    if any(token in normalized for token in ["mos", "quality", "nisqa", "utmos"]):
        return "audio_quality"
    if any(token in normalized for token in ["pesq", "stoi", "sdr", "snr"]):
        return "speech_enhancement"
    if any(token in normalized for token in ["f0", "pitch", "mcd"]):
        return "pitch_f0"
    if any(token in normalized for token in ["distance", "dtw", "rmse"]):
        return "distance"
    return "other"


def metric_direction(
    metric: str, registry: Optional[MetricRegistry] = None
) -> Optional[bool]:
    """Infer whether larger scores are better, returning None for unknown names.

    This is a name-based heuristic, not a backend-provided score guarantee."""
    metadata = _metadata_for_metric(metric, registry)
    normalized = _strip_prefix(metadata.name if metadata else metric).lower()
    if any(token in normalized for token in ["wer", "cer", "error", "rmse", "mcd"]):
        return False
    if "distance" in normalized and "token_distance" not in normalized:
        return False
    if any(
        token in normalized
        for token in [
            "similarity",
            "sim",
            "corr",
            "mos",
            "quality",
            "nisqa",
            "utmos",
            "pesq",
            "stoi",
            "sdr",
            "snr",
            "bleu",
            "bert",
            "clap",
        ]
    ):
        return True
    return None


def _metadata_for_metric(metric: str, registry: Optional[MetricRegistry]):
    """Find the first registry entry matching a metric name or stripped suffix."""
    if registry is None:
        return None
    for candidate in _metric_name_candidates(metric):
        metadata = registry.get_metadata(candidate)
        if metadata is not None:
            return metadata
    return None


def _metric_name_candidates(metric: str) -> List[str]:
    """Return distinct metric identifiers obtained by progressively removing prefixes."""
    candidates = [metric, _strip_prefix(metric)]
    parts = metric.split("_")
    for index in range(1, len(parts)):
        candidates.append("_".join(parts[index:]))
    unique = []
    for candidate in candidates:
        if candidate and candidate not in unique:
            unique.append(candidate)
    return unique


def _collect_input_paths(input_path: str) -> List[Path]:
    """List a result file or supported files directly inside a directory.

    Raise FileNotFoundError if the path contains no recognized result files."""
    path = Path(input_path)
    if path.is_file():
        return [path]
    if path.is_dir():
        paths: List[Path] = []
        for pattern in ("*.jsonl", "*.json", "*.txt", "*.jl"):
            paths.extend(sorted(path.glob(pattern)))
        if paths:
            return paths
    raise FileNotFoundError(f"No result files found at {input_path}")


def _literal_eval_with_special_floats(line: str) -> Any:
    """Parse a Python literal record allowing bare infinity and NaN spellings."""

    class SpecialFloatTransformer(ast.NodeTransformer):
        """Replace supported special-float names with constants before literal evaluation."""

        def visit_Name(self, node: ast.Name) -> ast.AST:
            """Convert inf/Infinity and nan/NaN names while preserving source locations."""
            if node.id in {"inf", "Infinity"}:
                return ast.copy_location(ast.Constant(float("inf")), node)
            if node.id in {"nan", "NaN"}:
                return ast.copy_location(ast.Constant(float("nan")), node)
            return node

    tree = ast.parse(line, mode="eval")
    tree = SpecialFloatTransformer().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.literal_eval(tree)


def _sample_std(values: Sequence[float], mean: float) -> float:
    """Return sample standard deviation, or zero for fewer than two observations."""
    if len(values) <= 1:
        return 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def _median(sorted_values: Sequence[float]) -> float:
    """Return the median of already-sorted values, or zero for an empty sequence."""
    count = len(sorted_values)
    if not count:
        return 0.0
    midpoint = count // 2
    if count % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def _strip_prefix(metric: str) -> str:
    """Remove one prefix only if the remainder is a known reporting metric."""
    parts = metric.split("_")
    if len(parts) > 1 and parts[0] not in {"se", "si", "ci", "f0"}:
        candidate = "_".join(parts[1:])
        known = {name.lower() for names in METRIC_CATEGORIES.values() for name in names}
        if candidate.lower() in known:
            return candidate
    return metric


def _fmt(value: Any) -> str:
    """Format finite floats to four significant digits and stringify other values."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.4g}"
    return str(value)


def _summary_row(summary: MetricSummary) -> Dict[str, Any]:
    """Convert a metric summary to CSV fields, flattening its outlier list."""
    return {
        "metric": summary.name,
        "reducer": summary.reducer,
        "aggregate": summary.aggregate,
        "category": summary.category,
        "count": summary.count,
        "missing": summary.missing,
        "invalid": summary.invalid,
        "mean": summary.mean,
        "median": summary.median,
        "std": summary.std,
        "stderr": summary.stderr,
        "ci95_low": summary.ci95_low,
        "ci95_high": summary.ci95_high,
        "min": summary.minimum,
        "max": summary.maximum,
        "higher_is_better": summary.higher_is_better,
        "best_key": summary.best_key,
        "best_value": summary.best_value,
        "worst_key": summary.worst_key,
        "worst_value": summary.worst_value,
        "outliers": "; ".join(
            f"{key}:{_fmt(value)} (z={_fmt(z_score)})"
            for key, value, z_score in summary.outliers
        ),
    }


def _metric_html_row(summary: MetricSummary) -> str:
    """Render one metric summary as a table row with escaped names and keys."""
    ci = f"{_fmt(summary.ci95_low)} to {_fmt(summary.ci95_high)}"
    best = f"{html.escape(summary.best_key)} ({_fmt(summary.best_value)})"
    worst = f"{html.escape(summary.worst_key)} ({_fmt(summary.worst_value)})"
    return (
        "<tr>"
        f"<td>{html.escape(summary.name)}</td>"
        f"<td>{html.escape(summary.category)}</td>"
        f"<td>{summary.count}</td>"
        f"<td>{summary.reducer}</td>"
        f"<td>{_fmt(summary.aggregate)}</td>"
        f"<td>{_fmt(summary.mean)}</td>"
        f"<td>{_fmt(summary.std)}</td>"
        f"<td>{html.escape(ci)}</td>"
        f"<td>{summary.missing}</td>"
        f"<td>{summary.invalid}</td>"
        f"<td>{best}</td>"
        f"<td>{worst}</td>"
        "</tr>"
    )


def _outlier_html(summaries: Sequence[MetricSummary]) -> str:
    """Render escaped outlier lists, or a message when no outliers were found."""
    blocks = []
    for summary in summaries:
        if not summary.outliers:
            continue
        items = "".join(
            f"<li>{html.escape(key)}: {_fmt(value)} (z={_fmt(z_score)})</li>"
            for key, value, z_score in summary.outliers
        )
        blocks.append(f"<h3>{html.escape(summary.name)}</h3><ul>{items}</ul>")
    return (
        "\n".join(blocks)
        if blocks
        else '<p class="muted">No z-score outliers >= 2.0 were detected.</p>'
    )


def _ranking_html(analysis: Dict[str, Any]) -> str:
    """Render the top group for each ranked metric, or empty text if none exist."""
    groups = analysis.get("groups") or {}
    rankings = groups.get("rankings") or {}
    if not rankings:
        return ""
    rows = []
    for metric, ranking in rankings.items():
        if not ranking:
            continue
        top = ranking[0]
        rows.append(
            f"<tr><td>{html.escape(metric)}</td><td>{html.escape(top['group'])}</td><td>{_fmt(top['mean'])}</td><td>{top['count']}</td></tr>"
        )
    if not rows:
        return ""
    group_by = html.escape(str(analysis.get("group_by")))
    return f"<section class=\"panel\"><h2>Per-Metric Ranking by {group_by}</h2><table><thead><tr><th>Metric</th><th>Top Group</th><th>Mean</th><th>Count</th></tr></thead><tbody>{''.join(rows)}</tbody></table></section>"


def _radar_svg(summaries: Sequence[MetricSummary]) -> str:
    """Plot absolute means scaled to the largest absolute mean in this selection.

    The plot does not normalize metric ranges or account for score direction."""
    if not summaries:
        return '<p class="muted">No metrics available.</p>'
    width = 420
    height = 360
    cx = width / 2
    cy = height / 2
    radius = 118
    max_mean = max(abs(summary.mean) for summary in summaries) or 1.0
    points = []
    labels = []
    for index, summary in enumerate(summaries):
        angle = -math.pi / 2 + 2 * math.pi * index / len(summaries)
        scaled = min(abs(summary.mean) / max_mean, 1.0)
        x = cx + math.cos(angle) * radius * scaled
        y = cy + math.sin(angle) * radius * scaled
        points.append(f"{x:.2f},{y:.2f}")
        lx = cx + math.cos(angle) * (radius + 34)
        ly = cy + math.sin(angle) * (radius + 34)
        labels.append(
            f'<text x="{lx:.2f}" y="{ly:.2f}" text-anchor="middle">{html.escape(summary.name[:18])}</text>'
        )
    rings = []
    for factor in (0.25, 0.5, 0.75, 1.0):
        ring_points = []
        for index in range(len(summaries)):
            angle = -math.pi / 2 + 2 * math.pi * index / len(summaries)
            ring_points.append(
                f"{cx + math.cos(angle) * radius * factor:.2f},{cy + math.sin(angle) * radius * factor:.2f}"
            )
        rings.append(
            f"<polygon points=\"{' '.join(ring_points)}\" fill=\"none\" stroke=\"#d9e0e7\" stroke-width=\"1\"/>"
        )
    axes = []
    for index in range(len(summaries)):
        angle = -math.pi / 2 + 2 * math.pi * index / len(summaries)
        axes.append(
            f'<line x1="{cx}" y1="{cy}" x2="{cx + math.cos(angle) * radius:.2f}" y2="{cy + math.sin(angle) * radius:.2f}" stroke="#e4e9ee"/>'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Radar overview">'
        + "".join(rings)
        + "".join(axes)
        + f"<polygon points=\"{' '.join(points)}\" fill=\"#1f8a7040\" stroke=\"#1f8a70\" stroke-width=\"2\"/>"
        + "".join(labels)
        + "</svg>"
    )


def _sunburst_svg(category_rows: Sequence[Dict[str, Any]]) -> str:
    """Render category wedges sized by their number of metrics."""
    if not category_rows:
        return '<p class="muted">No categories available.</p>'
    width = 420
    height = 360
    cx = width / 2
    cy = height / 2
    total = sum(row["metrics"] for row in category_rows) or 1
    start = -math.pi / 2
    colors = [
        "#1f8a70",
        "#c05621",
        "#4267ac",
        "#8a6f2a",
        "#287f9e",
        "#9a4f7a",
        "#5f7f32",
        "#555f6f",
    ]
    paths = []
    labels = []
    for index, row in enumerate(category_rows):
        sweep = 2 * math.pi * row["metrics"] / total
        end = start + sweep
        paths.append(
            _arc_path(cx, cy, 58, 132, start, end, colors[index % len(colors)])
        )
        mid = (start + end) / 2
        labels.append(
            f"<text x=\"{cx + math.cos(mid) * 158:.2f}\" y=\"{cy + math.sin(mid) * 158:.2f}\" text-anchor=\"middle\">{html.escape(row['category'][:16])}</text>"
        )
        start = end
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Category sunburst">'
        f'<circle cx="{cx}" cy="{cy}" r="48" fill="#f7f9fb" stroke="#d9e0e7"/>'
        f'<text x="{cx}" y="{cy - 4}" text-anchor="middle">VERSA</text>'
        f'<text x="{cx}" y="{cy + 12}" text-anchor="middle">metrics</text>'
        + "".join(paths)
        + "".join(labels)
        + "</svg>"
    )


def _arc_path(
    cx: float,
    cy: float,
    inner: float,
    outer: float,
    start: float,
    end: float,
    color: str,
) -> str:
    """Build a filled SVG annular sector from radii and angles in radians."""
    large = 1 if end - start > math.pi else 0
    p1 = (cx + math.cos(start) * outer, cy + math.sin(start) * outer)
    p2 = (cx + math.cos(end) * outer, cy + math.sin(end) * outer)
    p3 = (cx + math.cos(end) * inner, cy + math.sin(end) * inner)
    p4 = (cx + math.cos(start) * inner, cy + math.sin(start) * inner)
    d = (
        f"M {p1[0]:.2f} {p1[1]:.2f} "
        f"A {outer} {outer} 0 {large} 1 {p2[0]:.2f} {p2[1]:.2f} "
        f"L {p3[0]:.2f} {p3[1]:.2f} "
        f"A {inner} {inner} 0 {large} 0 {p4[0]:.2f} {p4[1]:.2f} Z"
    )
    return f'<path d="{d}" fill="{color}" fill-opacity="0.82" stroke="#fff" stroke-width="2"/>'
