"""Dependency-light completion and status contract shared by scoring and resume.

Result rows carry a versioned completion envelope under ``_versa_completion``
that records, for every metric attempted on an utterance, the execution status
and the identity of the evaluation that produced it. Resume consults these
records instead of the presence of a key, so a failed metric, a newly added
metric, or a changed configuration is recomputed while successful work is kept.

The envelope field name starts with an underscore, so
``versa.result_summary.is_score_field`` already excludes it from numeric
summaries, rankings, and report score columns.
"""

import hashlib
import json
import os
from numbers import Integral, Real

COMPLETION_SCHEMA_VERSION = 1
COMPLETION_FIELD = "_versa_completion"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"
STATUS_ABSTAINED = "abstained"
ALL_STATUSES = (STATUS_SUCCESS, STATUS_FAILED, STATUS_SKIPPED, STATUS_ABSTAINED)

# Retry policy: a metric that failed or was skipped is retried on resume. An
# abstention is a valid, reproducible outcome of the same inputs, so it is kept
# and not retried; delete the row or disable resume to force recomputation.
TERMINAL_STATUSES = frozenset({STATUS_SUCCESS, STATUS_ABSTAINED})

ERROR_CONFIGURATION = "configuration"
ERROR_BACKEND_SETUP = "backend_setup"
ERROR_INVALID_INPUT = "invalid_input"
ERROR_INFERENCE = "inference"

# Execution placement options do not change the evaluation itself, so they are
# excluded from metric identity to avoid recomputing on an unrelated rerun.
NON_EVALUATION_CONFIG_KEYS = frozenset({"use_gpu", "cache_dir", "io"})

LEGACY_RECOMPUTE = "recompute"
LEGACY_TRUST = "trust"

INPUT_IDENTITY_PATH = "path"
INPUT_IDENTITY_CONTENT = "content"

_ERROR_MESSAGE_LIMIT = 300
_CONTENT_CHUNK = 1024 * 1024


def _canonical(value):
    """Convert a configuration value into a deterministic JSON-safe form."""
    if isinstance(value, dict):
        return {str(name): _canonical(item) for name, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        # Exact, so two large adjacent integers cannot share one identity.
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        # An integral float and the same integer describe one configuration.
        return int(number) if number.is_integer() else number
    return repr(value)


def digest(payload):
    """Return a short stable digest of an arbitrary canonicalizable payload."""
    encoded = json.dumps(_canonical(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def evaluation_config(config):
    """Drop execution-placement options that do not affect evaluation output."""
    return {
        name: value
        for name, value in (config or {}).items()
        if name not in NON_EVALUATION_CONFIG_KEYS
    }


def metric_signature(name, config=None, identity=None):
    """Digest the canonical metric name, evaluation config, and model identity.

    ``identity`` carries evaluation-affecting facts a metric knows but its
    configuration does not, such as a resolved checkpoint revision, a prompt
    protocol digest, or a preprocessing policy. Metrics expose it by defining
    ``evaluation_identity()``; see ``metric_signatures``."""
    return digest(
        {
            "schema": COMPLETION_SCHEMA_VERSION,
            "name": name,
            "config": evaluation_config(config),
            "identity": identity or {},
        }
    )


def metric_signatures(metrics):
    """Build the required name-to-signature mapping for loaded metric objects.

    ``metrics`` maps a metric name to a metric instance. An instance may define
    ``evaluation_identity()`` returning a small mapping of evaluation-affecting
    facts; failures to produce one are ignored so an unmodified metric keeps a
    configuration-only identity."""
    signatures = {}
    for name, metric in metrics.items():
        identity = None
        hook = getattr(metric, "evaluation_identity", None)
        if callable(hook):
            try:
                identity = hook()
            except Exception:  # pragma: no cover - defensive, metric-defined hook
                identity = None
        signatures[name] = metric_signature(
            name, getattr(metric, "config", None), identity
        )
    return signatures


def input_signature(paths, policy=INPUT_IDENTITY_PATH, text=None):
    """Digest the inputs of one utterance under the selected identity policy.

    ``path`` (the default) records only the resolved input locations and the
    transcript, so replacing a file in place is not detected. ``content``
    additionally hashes readable local files, which detects edited audio at the
    cost of reading every input once per run. Entries that are not readable
    local files, such as Kaldi arrays, fall back to their textual form."""
    if policy not in (INPUT_IDENTITY_PATH, INPUT_IDENTITY_CONTENT):
        raise ValueError(f"Unknown input identity policy: {policy}")

    items = []
    for path in paths:
        if path is None:
            items.append(None)
            continue
        label = os.fspath(path) if isinstance(path, (str, os.PathLike)) else repr(path)
        if policy == INPUT_IDENTITY_CONTENT:
            items.append({"path": label, "content": _content_digest(path)})
        else:
            items.append({"path": label})
    return digest({"policy": policy, "inputs": items, "text": text})


def _content_digest(path):
    """Hash a readable local file, or return None when it cannot be read."""
    if not isinstance(path, (str, os.PathLike)):
        return None
    try:
        hasher = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_CONTENT_CHUNK), b""):
                hasher.update(chunk)
    except OSError:
        return None
    return hasher.hexdigest()[:16]


def new_completion(inputs=None, policy=INPUT_IDENTITY_PATH):
    """Create an empty completion envelope for one utterance."""
    return {
        "schema": COMPLETION_SCHEMA_VERSION,
        "input": {"policy": policy, "signature": inputs},
        "metrics": {},
    }


def get_completion(row):
    """Return the completion envelope of a row, or None for a legacy row."""
    envelope = (row or {}).get(COMPLETION_FIELD)
    if not isinstance(envelope, dict):
        return None
    if envelope.get("schema") != COMPLETION_SCHEMA_VERSION:
        return None
    return envelope


def ensure_completion(row, inputs=None, policy=INPUT_IDENTITY_PATH):
    """Return the row's envelope, creating and attaching one when absent."""
    envelope = get_completion(row)
    if envelope is None:
        envelope = new_completion(inputs, policy)
        row[COMPLETION_FIELD] = envelope
    return envelope


def record_metric_status(
    row,
    name,
    signature,
    status,
    fields=(),
    error=None,
    error_category=None,
):
    """Record one metric outcome in the row's envelope.

    ``fields`` names the result keys the metric contributed, so a later resume
    can remove stale values before recomputing it. ``error`` is truncated for
    the record; full diagnostics belong in the logs."""
    if status not in ALL_STATUSES:
        raise ValueError(f"Unknown metric status: {status}")
    envelope = ensure_completion(row)
    entry = {"status": status, "signature": signature, "fields": sorted(fields)}
    if error is not None:
        entry["error"] = str(error)[:_ERROR_MESSAGE_LIMIT]
        entry["error_category"] = error_category or ERROR_INFERENCE
    envelope["metrics"][name] = entry
    return entry


def metric_entry(row, name):
    """Return the recorded entry for one metric, or None when it has none."""
    envelope = get_completion(row)
    if envelope is None:
        return None
    entry = envelope["metrics"].get(name)
    return entry if isinstance(entry, dict) else None


def classify_outcome(value, error=None):
    """Map a metric return value or exception to a status and error category.

    An exception is an inference failure. A metric that returns nothing, or a
    mapping whose values are all null, abstained on these inputs. Any other
    value is a success, including strings and structured results."""
    if error is not None:
        return STATUS_FAILED, ERROR_INFERENCE
    if value is None:
        return STATUS_ABSTAINED, None
    if isinstance(value, dict) and all(item is None for item in value.values()):
        return STATUS_ABSTAINED, None
    return STATUS_SUCCESS, None


def pending_metrics(row, signatures, inputs=None, legacy=LEGACY_RECOMPUTE):
    """List the metrics that still need computation for an existing row.

    A metric is pending unless the row records it as terminal under the same
    signature. Changed inputs make every metric pending. A legacy row without
    an envelope is recomputed unless ``legacy`` is ``trust``, which accepts the
    historical rule that any recorded field means completed work."""
    if row is None:
        return sorted(signatures)

    envelope = get_completion(row)
    if envelope is None:
        if legacy == LEGACY_TRUST:
            return sorted(
                name for name in signatures if not _legacy_row_has_values(row)
            )
        return sorted(signatures)

    recorded_inputs = envelope.get("input", {}).get("signature")
    if inputs is not None and recorded_inputs != inputs:
        # A record with a missing or different input identity cannot be shown
        # to describe these inputs, so everything is recomputed.
        return sorted(signatures)

    pending = []
    for name, signature in signatures.items():
        entry = envelope["metrics"].get(name)
        if not isinstance(entry, dict):
            pending.append(name)
        elif entry.get("signature") != signature:
            pending.append(name)
        elif entry.get("status") not in TERMINAL_STATUSES:
            pending.append(name)
    return sorted(pending)


def _legacy_row_has_values(row):
    """Report whether a pre-contract row carries any value besides its key."""
    return any(name != "key" for name in row)


def drop_metric_fields(row, name):
    """Remove the values a previous run attributed to one metric."""
    entry = metric_entry(row, name)
    if entry is None:
        return
    for field in entry.get("fields", ()):
        row.pop(field, None)


def merge_rows(existing, fresh):
    """Merge freshly computed metric results into a previously stored row.

    Values of recomputed metrics are replaced rather than accumulated, so a
    metric that stops reporting a field does not leave a stale one behind.
    Changed inputs invalidate all old metric entries and their score fields,
    including metrics that have not yet run in a metric-oriented pass."""
    if existing is None:
        return dict(fresh)

    merged = dict(existing)
    fresh_envelope = get_completion(fresh) or new_completion()
    for name in fresh_envelope["metrics"]:
        drop_metric_fields(merged, name)

    merged_envelope = dict(ensure_completion(merged))
    merged_envelope["metrics"] = dict(merged_envelope["metrics"])
    if merged_envelope["input"].get("signature") != fresh_envelope["input"].get(
        "signature"
    ):
        for name in merged_envelope["metrics"]:
            drop_metric_fields(merged, name)
        merged_envelope["metrics"] = {}
    merged.pop(COMPLETION_FIELD, None)
    for field, value in fresh.items():
        if field != COMPLETION_FIELD:
            merged[field] = value

    merged_envelope["input"] = fresh_envelope["input"]
    merged_envelope["metrics"].update(fresh_envelope["metrics"])
    merged[COMPLETION_FIELD] = merged_envelope
    return merged


class RunStatus:
    """Counts of requested, loaded, and executed work for one scoring run.

    Denominators are explicit: ``requested_metrics`` counts configured metric
    entries, ``loaded_metrics`` counts the ones whose backend was constructed,
    and the utterance counters partition the configured inputs into scored,
    resumed, and skipped."""

    def __init__(self):
        """Start every counter at zero with no recorded failures."""
        self.requested_metrics = 0
        self.loaded_metrics = 0
        self.failed_metric_loads = []
        self.total_utterances = 0
        self.scored_utterances = 0
        self.resumed_utterances = 0
        self.skipped_utterances = 0
        self.metric_status_counts = {status: 0 for status in ALL_STATUSES}
        self.metric_error_counts = {}

    def record_load(self, requested, loaded, failures=None):
        """Record metric loading outcomes before any utterance is scored.

        ``failures`` maps a metric name to why it could not be used, such as
        ``configuration`` for an unmet input requirement or ``backend_setup``
        for a backend that raised while being constructed."""
        self.requested_metrics += requested
        self.loaded_metrics += loaded
        for name, category in (failures or {}).items():
            self.failed_metric_loads.append(name)
            self.metric_error_counts[category] = (
                self.metric_error_counts.get(category, 0) + 1
            )

    def record_metric_entry(self, entry):
        """Count one recorded metric outcome and its error category."""
        status = entry.get("status")
        if status in self.metric_status_counts:
            self.metric_status_counts[status] += 1
        category = entry.get("error_category")
        if category:
            self.metric_error_counts[category] = (
                self.metric_error_counts.get(category, 0) + 1
            )

    def record_row(self, row, resumed=False, metric_names=None):
        """Count a row, optionally restricting outcomes to the current metrics.

        Resumed rows may retain failures from metrics no longer requested.
        Those historical entries must not make the current strict run fail."""
        if resumed:
            self.resumed_utterances += 1
        else:
            self.scored_utterances += 1
        envelope = get_completion(row)
        if envelope is None:
            return
        for name, entry in envelope["metrics"].items():
            if metric_names is not None and name not in metric_names:
                continue
            if isinstance(entry, dict):
                self.record_metric_entry(entry)

    def record_skipped_utterance(self):
        """Count an utterance dropped before scoring, such as invalid audio."""
        self.skipped_utterances += 1

    def merge_metric_counts(self, other):
        """Fold the metric outcomes of one metric pass into this status.

        Only metric-level counts are merged, because each pass evaluates
        different metrics. Utterance counters are not: the same invalid input
        is dropped again by every pass, so the caller counts each utterance
        once after all passes finish."""
        for status, count in other.metric_status_counts.items():
            self.metric_status_counts[status] = (
                self.metric_status_counts.get(status, 0) + count
            )
        for category, count in other.metric_error_counts.items():
            self.metric_error_counts[category] = (
                self.metric_error_counts.get(category, 0) + count
            )

    @property
    def has_failures(self):
        """Report whether any metric failed to load, run, or receive its input."""
        return bool(
            self.failed_metric_loads
            or self.metric_status_counts[STATUS_FAILED]
            or self.skipped_utterances
        )

    def as_dict(self):
        """Return the counters as a plain mapping for logs and reports."""
        return {
            "requested_metrics": self.requested_metrics,
            "loaded_metrics": self.loaded_metrics,
            "failed_metric_loads": sorted(self.failed_metric_loads),
            "total_utterances": self.total_utterances,
            "scored_utterances": self.scored_utterances,
            "resumed_utterances": self.resumed_utterances,
            "skipped_utterances": self.skipped_utterances,
            "metric_status_counts": dict(self.metric_status_counts),
            "metric_error_counts": dict(sorted(self.metric_error_counts.items())),
        }

    def describe(self):
        """Render a one-line human-readable completeness summary."""
        counts = self.as_dict()
        return (
            "metrics {loaded}/{requested} loaded; utterances "
            "{scored} scored, {resumed} resumed, {skipped} skipped of {total}; "
            "metric outcomes {statuses}".format(
                loaded=counts["loaded_metrics"],
                requested=counts["requested_metrics"],
                scored=counts["scored_utterances"],
                resumed=counts["resumed_utterances"],
                skipped=counts["skipped_utterances"],
                total=counts["total_utterances"],
                statuses=counts["metric_status_counts"],
            )
        )
