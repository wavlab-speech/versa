"""Resume and run-status contracts exercised through the real scorer.

The metrics here are dependency-free and defined at module level so the local
multiprocessing path can pickle them.
"""

import json
import os
import pathlib

import pytest

from versa.completion import (
    COMPLETION_FIELD,
    ERROR_CONFIGURATION,
    RunStatus,
    STATUS_ABSTAINED,
    STATUS_FAILED,
    STATUS_SUCCESS,
)
from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)
from versa import scorer_shared
from versa.result_summary import compute_summary
from versa.scorer_shared import VersaScorer, find_files

FAIL_MARKER = "fail"
ABSTAIN_MARKER = "abstain"
SUCCEED_MARKER = "succeed"
# Worker processes are spawned, so behavior that must reach them travels
# through the environment rather than through class attributes.
FLAKY_MODE_VAR = "VERSA_TEST_FLAKY_MODE"
STABLE_VALUE_VAR = "VERSA_TEST_STABLE_VALUE"


def _metadata(name, requires_reference=False):
    """Build dependency-free metadata for a test metric."""
    return MetricMetadata(
        name=name,
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=requires_reference,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=[],
        description="Dependency-light test metric.",
    )


class StableMetric(BaseMetric):
    """Return a configurable constant and count its invocations."""

    calls = 0

    def _setup(self):
        """Read the constant this instance reports."""
        self.value = self.config.get("value", 1.0)

    def compute(self, predictions, references=None, metadata=None):
        """Count the call and return the current constant.

        The environment overrides the configured value so a test can change
        the result without changing the metric identity."""
        type(self).calls += 1
        override = os.environ.get(STABLE_VALUE_VAR)
        return {"stable_score": float(override) if override else self.value}

    def get_metadata(self):
        """Expose the metric under its registered name."""
        return _metadata("stable")


class SecondMetric(StableMetric):
    """A separate metric used to test newly configured metrics."""

    calls = 0

    def compute(self, predictions, references=None, metadata=None):
        """Count the call and return a distinct score field."""
        type(self).calls += 1
        return {"second_score": 2.0}

    def get_metadata(self):
        """Expose the metric under its registered name."""
        return _metadata("second")


class FlakyMetric(BaseMetric):
    """Fail or abstain on the first pass and succeed afterwards."""

    calls = 0

    def _setup(self):
        """No backend is required for this metric."""

    def compute(self, predictions, references=None, metadata=None):
        """Raise, abstain, or score depending on the mode in the environment."""
        type(self).calls += 1
        mode = os.environ.get(FLAKY_MODE_VAR, FAIL_MARKER)
        if mode == FAIL_MARKER:
            raise RuntimeError("backend exploded")
        if mode == ABSTAIN_MARKER:
            return None
        return {"flaky_score": 4.0}

    def get_metadata(self):
        """Expose the metric under its registered name."""
        return _metadata("flaky")


class ReferenceMetric(StableMetric):
    """A metric that cannot load without paired references."""

    calls = 0

    def get_metadata(self):
        """Declare a reference requirement to exercise load failures."""
        return _metadata("needs_reference", requires_reference=True)


@pytest.fixture
def scorer(monkeypatch):
    """Build a scorer whose registry contains only the test metrics."""
    StableMetric.calls = 0
    SecondMetric.calls = 0
    FlakyMetric.calls = 0
    ReferenceMetric.calls = 0
    monkeypatch.setenv(FLAKY_MODE_VAR, FAIL_MARKER)
    monkeypatch.delenv(STABLE_VALUE_VAR, raising=False)
    registry = MetricRegistry()
    for cls in (StableMetric, SecondMetric, FlakyMetric, ReferenceMetric):
        registry.register(cls, cls().get_metadata())
    return VersaScorer(registry)


@pytest.fixture
def gen_files():
    """Map three utterance keys onto one real sample waveform."""
    sample_file = next(iter(find_files("test/test_samples/test2").values()))
    return {f"utterance-{index}": sample_file for index in range(3)}


def _read_jsonl(path):
    """Read a JSONL result file into records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _status(row, name):
    """Return the recorded status of one metric in a result row."""
    return row[COMPLETION_FIELD]["metrics"][name]["status"]


def _score(scorer, gen_files, configs, output_file, **kwargs):
    """Score with a freshly loaded suite, returning the utterance records."""
    suite = scorer.load_metrics(configs, use_gt=False)
    return scorer.score_utterances(
        gen_files,
        suite,
        output_file=str(output_file),
        io="soundfile",
        **kwargs,
    )


@pytest.mark.parametrize("num_workers", [1, 2])
def test_failed_metric_is_retried_and_success_is_kept(
    scorer, gen_files, tmp_path, num_workers, monkeypatch
):
    """A failed metric is recomputed on resume while successes are reused."""
    output_file = tmp_path / "scores.jsonl"
    configs = [{"name": "stable"}, {"name": "flaky"}]

    first = _score(scorer, gen_files, configs, output_file, num_workers=num_workers)
    assert [_status(row, "flaky") for row in first] == [STATUS_FAILED] * 3
    assert [_status(row, "stable") for row in first] == [STATUS_SUCCESS] * 3
    assert all(row["flaky"] is None for row in first)
    assert [row["stable_score"] for row in first] == [1.0] * 3

    monkeypatch.setenv(FLAKY_MODE_VAR, SUCCEED_MARKER)
    monkeypatch.setenv(STABLE_VALUE_VAR, "7.0")
    second = _score(
        scorer,
        gen_files,
        configs,
        output_file,
        resume=True,
        num_workers=num_workers,
    )

    assert [row["flaky_score"] for row in second] == [4.0] * 3
    # The successful metric kept its stored value instead of rerunning.
    assert [row["stable_score"] for row in second] == [1.0] * 3
    assert "flaky" not in second[0]
    rows = _read_jsonl(output_file)
    assert [row["key"] for row in rows] == list(gen_files)
    assert rows == second


def test_newly_configured_metric_runs_without_recomputing_the_others(
    scorer, gen_files, tmp_path
):
    """Adding a metric evaluates only the new one on resume."""
    output_file = tmp_path / "scores.jsonl"
    _score(scorer, gen_files, [{"name": "stable"}], output_file)
    StableMetric.calls = 0

    resumed = _score(
        scorer,
        gen_files,
        [{"name": "stable"}, {"name": "second"}],
        output_file,
        resume=True,
    )

    assert StableMetric.calls == 0
    assert SecondMetric.calls == 3
    assert [row["second_score"] for row in resumed] == [2.0] * 3
    assert [row["stable_score"] for row in resumed] == [1.0] * 3


def test_changed_metric_configuration_invalidates_stored_results(
    scorer, gen_files, tmp_path
):
    """A configuration that changes the evaluation forces recomputation."""
    output_file = tmp_path / "scores.jsonl"
    _score(scorer, gen_files, [{"name": "stable", "value": 1.0}], output_file)
    StableMetric.calls = 0

    resumed = _score(
        scorer,
        gen_files,
        [{"name": "stable", "value": 5.0}],
        output_file,
        resume=True,
    )

    assert StableMetric.calls == 3
    assert [row["stable_score"] for row in resumed] == [5.0] * 3


def test_changed_input_content_invalidates_stored_results(scorer, gen_files, tmp_path):
    """Under content identity, editing the audio forces recomputation."""
    audio = tmp_path / "input.wav"
    audio.write_bytes(pathlib.Path(next(iter(gen_files.values()))).read_bytes())
    files = {"utterance-0": str(audio)}
    output_file = tmp_path / "scores.jsonl"
    options = {"input_identity": "content"}
    _score(scorer, files, [{"name": "stable"}], output_file, **options)
    StableMetric.calls = 0

    _score(scorer, files, [{"name": "stable"}], output_file, resume=True, **options)
    assert StableMetric.calls == 0

    raw = bytearray(audio.read_bytes())
    raw[-1] = (raw[-1] + 1) % 256
    audio.write_bytes(bytes(raw))
    _score(scorer, files, [{"name": "stable"}], output_file, resume=True, **options)

    assert StableMetric.calls == 1


def test_path_identity_does_not_detect_edited_audio(scorer, gen_files, tmp_path):
    """The default policy compares locations, which a rewritten file keeps."""
    audio = tmp_path / "input.wav"
    audio.write_bytes(pathlib.Path(next(iter(gen_files.values()))).read_bytes())
    files = {"utterance-0": str(audio)}
    output_file = tmp_path / "scores.jsonl"
    _score(scorer, files, [{"name": "stable"}], output_file)
    raw = bytearray(audio.read_bytes())
    raw[-1] = (raw[-1] + 1) % 256
    audio.write_bytes(bytes(raw))
    StableMetric.calls = 0

    _score(scorer, files, [{"name": "stable"}], output_file, resume=True)

    assert StableMetric.calls == 0


def test_abstention_is_recorded_and_not_retried(
    scorer, gen_files, tmp_path, monkeypatch
):
    """An abstaining metric keeps its null value and is treated as complete."""
    monkeypatch.setenv(FLAKY_MODE_VAR, ABSTAIN_MARKER)
    output_file = tmp_path / "scores.jsonl"
    first = _score(scorer, gen_files, [{"name": "flaky"}], output_file)
    assert [_status(row, "flaky") for row in first] == [STATUS_ABSTAINED] * 3
    assert all(row["flaky"] is None for row in first)

    FlakyMetric.calls = 0
    _score(scorer, gen_files, [{"name": "flaky"}], output_file, resume=True)

    assert FlakyMetric.calls == 0


def test_resume_repairs_truncated_duplicate_and_unknown_rows(
    scorer, gen_files, tmp_path
):
    """Malformed, duplicated, and foreign rows survive resume correctly."""
    output_file = tmp_path / "scores.jsonl"
    complete = _score(scorer, gen_files, [{"name": "stable"}], output_file)
    keys = list(gen_files)
    duplicated = json.dumps(complete[0])
    foreign = json.dumps({"key": "not-configured", "stable_score": 9.0})
    partial = json.dumps(complete[1])[:-5]
    output_file.write_text(
        "\n".join([duplicated, foreign, duplicated, partial]) + "\n",
        encoding="utf-8",
    )
    StableMetric.calls = 0

    resumed = _score(scorer, gen_files, [{"name": "stable"}], output_file, resume=True)

    # Only the two utterances without a usable record are recomputed.
    assert StableMetric.calls == 2
    assert [row["key"] for row in resumed] == keys
    written = _read_jsonl(output_file)
    assert [row["key"] for row in written] == [keys[0], "not-configured"] + keys[1:]
    assert written.count(complete[0]) == 1


def test_resume_agrees_with_an_uninterrupted_run(scorer, gen_files, tmp_path):
    """Interrupting and resuming yields the records of a single full run."""
    configs = [{"name": "stable"}, {"name": "second"}]
    reference_file = tmp_path / "reference.jsonl"
    reference = _score(scorer, gen_files, configs, reference_file)

    partial_file = tmp_path / "partial.jsonl"
    first_key = list(gen_files)[0]
    _score(scorer, {first_key: gen_files[first_key]}, configs, partial_file)
    resumed = _score(scorer, gen_files, configs, partial_file, resume=True)

    assert resumed == reference
    assert _read_jsonl(partial_file) == _read_jsonl(reference_file)


def test_metric_oriented_resume_skips_completed_metrics(
    scorer, gen_files, tmp_path, monkeypatch
):
    """Metric-oriented resume neither reloads nor recomputes finished metrics."""
    output_file = tmp_path / "scores.jsonl"
    configs = [{"name": "stable"}, {"name": "flaky"}]
    scorer.score_utterances_by_metric(
        gen_files,
        configs,
        output_file=str(output_file),
        io="soundfile",
    )
    assert StableMetric.calls == 3 and FlakyMetric.calls == 3

    monkeypatch.setenv(FLAKY_MODE_VAR, SUCCEED_MARKER)
    StableMetric.calls = 0
    FlakyMetric.calls = 0
    loaded = []
    original_load = scorer.load_metrics

    def record_load(configs, **kwargs):
        """Record which metrics had their backend constructed."""
        loaded.extend(config["name"] for config in configs)
        return original_load(configs, **kwargs)

    monkeypatch.setattr(scorer, "load_metrics", record_load)
    resumed = scorer.score_utterances_by_metric(
        gen_files,
        configs,
        output_file=str(output_file),
        io="soundfile",
        resume=True,
    )

    assert loaded == ["flaky"]
    assert StableMetric.calls == 0
    assert FlakyMetric.calls == 3
    assert [row["flaky_score"] for row in resumed] == [4.0] * 3
    assert [row["stable_score"] for row in resumed] == [1.0] * 3
    assert _read_jsonl(output_file) == resumed


def test_metric_oriented_and_utterance_modes_agree(scorer, gen_files, tmp_path):
    """Both loop orders produce the same records for the same configuration."""
    configs = [{"name": "stable"}, {"name": "second"}]
    by_metric = scorer.score_utterances_by_metric(
        gen_files,
        configs,
        output_file=str(tmp_path / "by_metric.jsonl"),
        io="soundfile",
    )
    by_utterance = _score(scorer, gen_files, configs, tmp_path / "by_utterance.jsonl")

    assert by_metric == by_utterance


def test_failed_metric_is_excluded_from_numeric_summaries(scorer, gen_files, tmp_path):
    """Completion records never become scores or ranking dimensions."""
    records = _score(
        scorer, gen_files, [{"name": "stable"}, {"name": "flaky"}], tmp_path / "s.jsonl"
    )
    summary = compute_summary(records)

    assert COMPLETION_FIELD not in summary
    assert not any(name.startswith(COMPLETION_FIELD) for name in summary)
    assert summary["stable_score"] == 1.0


def test_run_status_counts_loads_failures_and_resumed_utterances(
    scorer, gen_files, tmp_path
):
    """The status contract reports explicit denominators for one run."""
    output_file = tmp_path / "scores.jsonl"
    status = RunStatus()
    suite = scorer.load_metrics(
        [{"name": "stable"}, {"name": "needs_reference"}],
        use_gt=False,
        run_status=status,
    )
    scorer.score_utterances(
        gen_files,
        suite,
        output_file=str(output_file),
        io="soundfile",
        run_status=status,
    )

    counts = status.as_dict()
    assert counts["requested_metrics"] == 2
    assert counts["loaded_metrics"] == 1
    assert counts["failed_metric_loads"] == ["needs_reference"]
    assert counts["metric_error_counts"] == {ERROR_CONFIGURATION: 1}
    assert counts["total_utterances"] == 3
    assert counts["scored_utterances"] == 3
    assert counts["resumed_utterances"] == 0
    assert counts["metric_status_counts"][STATUS_SUCCESS] == 3
    assert status.has_failures

    resumed_status = RunStatus()
    scorer.score_utterances(
        gen_files,
        suite,
        output_file=str(output_file),
        io="soundfile",
        resume=True,
        run_status=resumed_status,
    )

    resumed_counts = resumed_status.as_dict()
    assert resumed_counts["scored_utterances"] == 0
    assert resumed_counts["resumed_utterances"] == 3
    assert not resumed_status.has_failures


class _LazyMapping(dict):
    """Mimic a Kaldi mapping whose indexing loads the audio array."""

    def __init__(self, entries):
        """Store archive entries and expose them the way kaldiio does."""
        super().__init__(entries)
        self._dict = dict(entries)

    def __getitem__(self, key):
        """Fail so a test can prove planning never loads audio."""
        raise AssertionError("resume planning loaded audio from a lazy mapping")


def test_resume_planning_does_not_load_lazy_audio():
    """Input identity uses the stored archive entry, not the loaded array."""
    mapping = _LazyMapping({"utt": "archive.ark:17"})

    signatures = scorer_shared._utterance_input_signatures(mapping, mapping, None)

    assert list(signatures) == ["utt"]
    assert (
        signatures["utt"]
        == scorer_shared._utterance_input_signatures(
            {"utt": "archive.ark:17"}, {"utt": "archive.ark:17"}, None
        )["utt"]
    )


def test_interrupted_resume_keeps_stored_work_and_recovers(
    scorer, gen_files, tmp_path, monkeypatch
):
    """A resumed run that crashes loses nothing and the next resume repairs it."""
    output_file = tmp_path / "scores.jsonl"
    configs = [{"name": "stable"}, {"name": "flaky"}]
    _score(scorer, gen_files, configs, output_file)

    monkeypatch.setenv(FLAKY_MODE_VAR, SUCCEED_MARKER)
    real_load_audio = scorer_shared.load_audio
    state = {"loads": 0}

    def crashing_load_audio(path, io):
        """Load normally once, then fail as an interrupted run would."""
        state["loads"] += 1
        if state["loads"] > 1:
            raise RuntimeError("interrupted")
        return real_load_audio(path, io)

    monkeypatch.setattr(scorer_shared, "load_audio", crashing_load_audio)
    with pytest.raises(RuntimeError, match="interrupted"):
        _score(scorer, gen_files, configs, output_file, resume=True)

    # Every utterance still has its stored successful metric.
    interrupted = _read_jsonl(output_file)
    by_key = {row["key"]: row for row in interrupted}
    assert set(by_key) == set(gen_files)
    assert all(row["stable_score"] == 1.0 for row in by_key.values())

    monkeypatch.setattr(scorer_shared, "load_audio", real_load_audio)
    recovered = _score(scorer, gen_files, configs, output_file, resume=True)

    assert [row["flaky_score"] for row in recovered] == [4.0] * 3
    written = _read_jsonl(output_file)
    assert [row["key"] for row in written] == list(gen_files)
    assert written == recovered
