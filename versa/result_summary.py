"""Dependency-light numeric reduction policy shared by scoring and reports."""

import math
from numbers import Real

RESERVED_FIELDS = {
    "key",
    "status",
    "provenance",
    "metadata",
    "model",
    "model_id",
    "model_revision",
    "protocol_id",
    "protocol_version",
    "protocol_digest",
    "bank_schema_version",
    "prompt_mode",
    "seed",
    "candidate_order",
}
COUNT_SUFFIXES = tuple(
    f"{unit}_{operation}"
    for unit in ("wer", "cer", "per")
    for operation in ("insert", "delete", "replace", "equal")
)


def to_float(value):
    """Return a real scalar as float, excluding booleans and numeric strings."""
    if isinstance(value, Real) and not isinstance(value, bool):
        try:
            return float(value)
        except OverflowError:
            return math.inf if value > 0 else -math.inf
    return None


def is_score_field(name):
    """Exclude reserved metadata, private fields, and legacy transcript fields."""
    return (
        name not in RESERVED_FIELDS
        and not name.startswith("_")
        and "text" not in name.lower()
    )


def discover_numeric_metrics(records):
    """Find numeric or null score fields, retaining all-invalid numeric fields."""
    return sorted(
        {
            name
            for record in records
            for name, value in record.items()
            if is_score_field(name) and (value is None or to_float(value) is not None)
        }
    )


def is_count_metric(name):
    """Recognize only WER/CER/PER insert/delete/replace/equal count suffixes."""
    return any(
        name == suffix or name.endswith("_" + suffix) for suffix in COUNT_SUFFIXES
    )


def numeric_observations(records, metric):
    """Return (row index, finite value) pairs and missing/invalid row counts.

    Every row is an observation, including repeated utterance keys; keys alone
    cannot distinguish retries from evaluations of different systems.
    """
    values = []
    missing = invalid = 0
    for index, record in enumerate(records):
        if metric not in record:
            missing += 1
            continue
        value = to_float(record[metric])
        if value is None or not math.isfinite(value):
            invalid += 1
        else:
            values.append((index, value))
    return values, missing, invalid


def reduce_values(metric, values):
    """Sum named operation counts; average other finite values; omit empty data."""
    if not values:
        return None
    if is_count_metric(metric):
        return sum(values)
    return sum(value / len(values) for value in values)


def compute_summary(score_info):
    """Reduce all finite score fields; omit fields with no valid observations."""
    summary = {}
    for metric in discover_numeric_metrics(score_info):
        values, _, _ = numeric_observations(score_info, metric)
        value = reduce_values(metric, [value for _, value in values])
        if value is not None:
            summary[metric] = value
    return summary
