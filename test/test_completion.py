"""Unit contracts for the versioned completion and status records."""

import json

import pytest

from versa.completion import (
    COMPLETION_FIELD,
    COMPLETION_SCHEMA_VERSION,
    ERROR_BACKEND_SETUP,
    ERROR_INFERENCE,
    INPUT_IDENTITY_CONTENT,
    INPUT_IDENTITY_PATH,
    LEGACY_RECOMPUTE,
    LEGACY_TRUST,
    RunStatus,
    STATUS_ABSTAINED,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCESS,
    classify_outcome,
    ensure_completion,
    input_signature,
    merge_rows,
    metric_signature,
    metric_signatures,
    pending_metrics,
    record_metric_status,
)
from versa.result_summary import is_score_field


class _Metric:
    """Minimal stand-in exposing the configuration a metric instance carries."""

    def __init__(self, config, identity=None):
        """Store the configuration and an optional evaluation identity."""
        self.config = config
        self._identity = identity

    def evaluation_identity(self):
        """Return the identity hook value, raising when it is an exception."""
        if isinstance(self._identity, Exception):
            raise self._identity
        return self._identity


def test_metric_signature_ignores_execution_placement_only():
    """Placement options are not evaluation inputs; real options are."""
    base = metric_signature("m", {"model_tag": "a"})

    assert base == metric_signature(
        "m", {"model_tag": "a", "use_gpu": True, "cache_dir": "/tmp", "io": "dir"}
    )
    assert base != metric_signature("m", {"model_tag": "b"})
    assert base != metric_signature("other", {"model_tag": "a"})


def test_metric_signature_is_stable_across_key_order_and_value_types():
    """Canonicalization makes the digest independent of dict and numeric form."""
    assert metric_signature("m", {"a": 1, "b": [1, 2]}) == metric_signature(
        "m", {"b": (1.0, 2), "a": 1.0}
    )


def test_metric_signature_separates_adjacent_large_integers():
    """Exact integers keep their identity beyond the exact float range."""
    assert metric_signature("m", {"seed": 2**53}) != metric_signature(
        "m", {"seed": 2**53 + 1}
    )
    assert metric_signature("m", {"threshold": 0.1}) != metric_signature(
        "m", {"threshold": 0.2}
    )


def test_metric_signatures_include_the_metric_identity_hook():
    """A metric that reports a checkpoint revision changes its identity."""
    signatures = metric_signatures(
        {
            "plain": _Metric({"model_tag": "a"}),
            "pinned": _Metric({"model_tag": "a"}, {"revision": "abc"}),
            "broken": _Metric({"model_tag": "a"}, RuntimeError("no revision")),
        }
    )

    assert signatures["plain"] != signatures["pinned"]
    # A hook that raises leaves the configuration-only identity in place.
    assert signatures["broken"] == metric_signature("broken", {"model_tag": "a"})


def test_input_signature_policies(tmp_path):
    """Path identity ignores edited bytes; content identity detects them."""
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"first")
    before_path = input_signature([audio], INPUT_IDENTITY_PATH)
    before_content = input_signature([audio], INPUT_IDENTITY_CONTENT)
    audio.write_bytes(b"second")

    assert input_signature([audio], INPUT_IDENTITY_PATH) == before_path
    assert input_signature([audio], INPUT_IDENTITY_CONTENT) != before_content
    assert input_signature([audio], INPUT_IDENTITY_PATH, text="hi") != before_path
    assert input_signature([None], INPUT_IDENTITY_CONTENT)
    with pytest.raises(ValueError):
        input_signature([audio], "mtime")


def test_classify_outcome_separates_failure_from_abstention():
    """Only an exception is a failure; an empty result is an abstention."""
    assert classify_outcome(0.0) == (STATUS_SUCCESS, None)
    assert classify_outcome("male") == (STATUS_SUCCESS, None)
    assert classify_outcome({"a": 1, "b": None}) == (STATUS_SUCCESS, None)
    assert classify_outcome(None) == (STATUS_ABSTAINED, None)
    assert classify_outcome({"a": None}) == (STATUS_ABSTAINED, None)
    assert classify_outcome(None, ValueError("x")) == (STATUS_FAILED, ERROR_INFERENCE)


def test_pending_metrics_detects_every_invalidating_change():
    """Missing, failed, changed, and unknown-input records are all pending."""
    row = {"key": "utt", "a": 1.0, "b": 2.0}
    ensure_completion(row, "inputs-1")
    record_metric_status(row, "a", "sig-a", STATUS_SUCCESS, ["a"])
    record_metric_status(row, "b", "sig-b", STATUS_FAILED, error="boom")

    signatures = {"a": "sig-a", "b": "sig-b", "c": "sig-c"}
    assert pending_metrics(row, signatures, "inputs-1") == ["b", "c"]
    assert pending_metrics(row, {"a": "sig-a-v2"}, "inputs-1") == ["a"]
    assert pending_metrics(row, signatures, "inputs-2") == ["a", "b", "c"]
    assert pending_metrics(None, signatures) == ["a", "b", "c"]


def test_pending_metrics_treats_abstention_as_complete():
    """An abstention is a reproducible outcome, so resume keeps it."""
    row = {"key": "utt", "a": None}
    ensure_completion(row)
    record_metric_status(row, "a", "sig-a", STATUS_ABSTAINED, ["a"])

    assert pending_metrics(row, {"a": "sig-a"}) == []


def test_pending_metrics_retries_a_skipped_metric():
    """A skipped metric never ran, so it is retried."""
    row = {"key": "utt"}
    ensure_completion(row)
    record_metric_status(row, "a", "sig-a", STATUS_SKIPPED)

    assert pending_metrics(row, {"a": "sig-a"}) == ["a"]


def test_legacy_rows_recompute_by_default_and_can_be_trusted():
    """Rows without identity are rescored unless the caller opts into trust."""
    legacy = {"key": "utt", "a": 1.0}
    empty = {"key": "utt"}

    assert pending_metrics(legacy, {"a": "sig-a"}, None, LEGACY_RECOMPUTE) == ["a"]
    assert pending_metrics(legacy, {"a": "sig-a"}, None, LEGACY_TRUST) == []
    assert pending_metrics(empty, {"a": "sig-a"}, None, LEGACY_TRUST) == ["a"]


def test_missing_stored_input_identity_is_not_trusted():
    """A record that cannot prove which inputs produced it is recomputed."""
    row = {"key": "utt", "a": 1.0}
    ensure_completion(row)
    record_metric_status(row, "a", "sig-a", STATUS_SUCCESS, ["a"])

    assert pending_metrics(row, {"a": "sig-a"}, "inputs-1") == ["a"]
    # Without a current identity to compare against, the record still stands.
    assert pending_metrics(row, {"a": "sig-a"}) == []


def test_unknown_schema_version_is_treated_as_legacy():
    """A record from a future contract is not silently trusted."""
    row = {"key": "utt", COMPLETION_FIELD: {"schema": 99, "metrics": {}}}

    assert pending_metrics(row, {"a": "sig-a"}) == ["a"]


def test_merge_rows_replaces_recomputed_fields_only():
    """Recomputed metrics drop their stale fields; other metrics are kept."""
    existing = {"key": "utt", "a": 1.0, "b_one": 2.0, "b_two": 3.0}
    ensure_completion(existing, "inputs-1")
    record_metric_status(existing, "a", "sig-a", STATUS_SUCCESS, ["a"])
    record_metric_status(existing, "b", "sig-b", STATUS_SUCCESS, ["b_one", "b_two"])

    fresh = {"key": "utt", "b_one": 9.0}
    ensure_completion(fresh, "inputs-1")
    record_metric_status(fresh, "b", "sig-b2", STATUS_SUCCESS, ["b_one"])

    merged = merge_rows(existing, fresh)

    assert merged["a"] == 1.0
    assert merged["b_one"] == 9.0
    assert "b_two" not in merged
    metrics = merged[COMPLETION_FIELD]["metrics"]
    assert metrics["a"]["signature"] == "sig-a"
    assert metrics["b"]["signature"] == "sig-b2"
    assert merged[COMPLETION_FIELD]["schema"] == COMPLETION_SCHEMA_VERSION


def test_merge_rows_upgrades_a_legacy_row_without_losing_values():
    """Merging into a pre-contract row keeps its unrelated stored values."""
    merged = merge_rows({"key": "utt", "legacy": 1.0}, {"key": "utt", "a": 2.0})

    assert merged["legacy"] == 1.0
    assert merged["a"] == 2.0


def test_completion_field_is_not_a_score_and_serializes():
    """The envelope is excluded from score discovery and stays JSON-safe."""
    row = {"key": "utt"}
    ensure_completion(row, "inputs-1")
    record_metric_status(row, "a", "sig-a", STATUS_FAILED, error=RuntimeError("boom"))

    assert not is_score_field(COMPLETION_FIELD)
    restored = json.loads(json.dumps(row))
    entry = restored[COMPLETION_FIELD]["metrics"]["a"]
    assert entry["error"] == "boom"
    assert entry["error_category"] == ERROR_INFERENCE


def test_record_metric_status_rejects_an_unknown_status():
    """The status vocabulary is closed so reports cannot drift."""
    with pytest.raises(ValueError):
        record_metric_status({"key": "utt"}, "a", "sig-a", "done")


def test_run_status_reports_denominators_and_failures():
    """Counts partition the run and any failure makes it incomplete."""
    status = RunStatus()
    status.total_utterances = 3
    status.record_load(2, 1, {"missing": ERROR_BACKEND_SETUP})
    row = {"key": "utt"}
    ensure_completion(row)
    record_metric_status(row, "a", "sig-a", STATUS_SUCCESS, ["a"])
    status.record_row(row)
    status.record_row(row, resumed=True)

    counts = status.as_dict()
    assert counts["requested_metrics"] == 2
    assert counts["loaded_metrics"] == 1
    assert counts["failed_metric_loads"] == ["missing"]
    assert counts["scored_utterances"] == 1
    assert counts["resumed_utterances"] == 1
    assert counts["metric_status_counts"][STATUS_SUCCESS] == 2
    assert counts["metric_error_counts"] == {ERROR_BACKEND_SETUP: 1}
    assert status.has_failures
    assert "utterances 1 scored" in status.describe()


def test_run_status_merges_metric_outcomes_only():
    """Merging metric passes accumulates outcomes and leaves utterances alone."""
    status = RunStatus()
    for _ in range(3):
        one_pass = RunStatus()
        one_pass.record_metric_entry({"status": STATUS_SUCCESS})
        one_pass.record_skipped_utterance()
        status.merge_metric_counts(one_pass)

    assert status.metric_status_counts[STATUS_SUCCESS] == 3
    # The caller counts each utterance once after every pass has finished.
    assert status.skipped_utterances == 0


def test_run_status_without_failures_is_complete():
    """A clean run reports no failures so strict mode succeeds."""
    status = RunStatus()
    status.record_load(1, 1)

    assert not status.has_failures


def test_merge_changed_inputs_invalidates_metrics_not_yet_recomputed():
    """A partial rerun cannot certify old scores under its new input identity."""
    existing = {"key": "utt", "a": 1.0, "b": 2.0}
    ensure_completion(existing, "old-inputs")
    record_metric_status(existing, "a", "sig-a", STATUS_SUCCESS, ["a"])
    record_metric_status(existing, "b", "sig-b", STATUS_SUCCESS, ["b"])
    original = json.loads(json.dumps(existing))
    fresh = {"key": "utt", "a": 3.0}
    ensure_completion(fresh, "new-inputs")
    record_metric_status(fresh, "a", "sig-a", STATUS_SUCCESS, ["a"])

    merged = merge_rows(existing, fresh)

    assert merged["a"] == 3.0
    assert "b" not in merged
    assert pending_metrics(merged, {"a": "sig-a", "b": "sig-b"}, "new-inputs") == ["b"]
    assert existing == original


def test_run_status_filters_historical_metric_failures():
    """Only failures of currently requested metrics affect strict completion."""
    row = {"key": "utt"}
    record_metric_status(row, "a", "sig-a", STATUS_SUCCESS)
    record_metric_status(row, "removed", "sig-b", STATUS_FAILED, error="old failure")
    status = RunStatus()
    status.record_row(row, resumed=True, metric_names={"a"})
    assert status.resumed_utterances == 1
    assert status.metric_status_counts[STATUS_SUCCESS] == 1
    assert not status.metric_error_counts
    assert not status.has_failures
    status.record_row(row, resumed=True, metric_names={"removed"})
    assert status.has_failures
