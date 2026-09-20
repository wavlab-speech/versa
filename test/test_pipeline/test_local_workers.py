import json

import pytest

from versa.bin.scorer import get_parser
from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)
from versa.scorer_shared import VersaScorer, find_files


class ConstantMetric(BaseMetric):
    calls = 0

    def _setup(self):
        pass

    def compute(self, predictions, references=None, metadata=None):
        ConstantMetric.calls += 1
        return 1.0

    def get_metadata(self):
        return MetricMetadata(
            name="constant",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=False,
            auto_install=False,
            dependencies=[],
            description="Dependency-light test metric.",
        )


def _scorer_and_files():
    registry = MetricRegistry()
    registry.register(ConstantMetric, ConstantMetric().get_metadata())
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics([{"name": "constant"}], use_gt=False)
    sample_file = next(iter(find_files("test/test_samples/test2").values()))
    gen_files = {f"utterance-{index}": sample_file for index in range(3)}
    return scorer, metric_suite, gen_files


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_num_workers_cli_defaults_to_serial_and_accepts_parallel():
    parser = get_parser()

    assert parser.parse_args([]).num_workers == 1
    assert parser.parse_args(["--num_workers", "2"]).num_workers == 2


def test_parallel_matches_serial_and_preserves_input_order(tmp_path):
    scorer, metric_suite, gen_files = _scorer_and_files()
    serial_scores = scorer.score_utterances(
        gen_files, metric_suite, io="soundfile", num_workers=1
    )
    output_file = tmp_path / "parallel.jsonl"
    parallel_scores = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=str(output_file),
        io="soundfile",
        num_workers=2,
    )

    assert parallel_scores == serial_scores
    assert [score["key"] for score in parallel_scores] == list(gen_files)
    assert _read_jsonl(output_file) == parallel_scores


def test_parallel_resume_trusts_legacy_rows_on_request(tmp_path):
    scorer, metric_suite, gen_files = _scorer_and_files()
    keys = list(gen_files)
    completed_key = keys[-1]
    output_file = tmp_path / "scores.jsonl"
    output_file.write_text(
        json.dumps({"key": completed_key, "constant": 3.0}) + "\n",
        encoding="utf-8",
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=str(output_file),
        io="soundfile",
        resume=True,
        num_workers=2,
        legacy_resume="trust",
    )

    assert [score["key"] for score in score_info] == keys
    assert score_info[-1] == {"key": completed_key, "constant": 3.0}
    written = _read_jsonl(output_file)
    assert [row["key"] for row in written] == [completed_key, keys[0], keys[1]]
    assert written[0] == {"key": completed_key, "constant": 3.0}
    assert [row["constant"] for row in written[1:]] == [1.0, 1.0]


def test_parallel_resume_recomputes_rows_without_completion_records(tmp_path):
    scorer, metric_suite, gen_files = _scorer_and_files()
    keys = list(gen_files)
    output_file = tmp_path / "scores.jsonl"
    output_file.write_text(
        json.dumps({"key": keys[-1], "constant": 3.0}) + "\n",
        encoding="utf-8",
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=str(output_file),
        io="soundfile",
        resume=True,
        num_workers=2,
    )

    # A legacy row carries no metric identity, so its stale score is replaced.
    assert [score["key"] for score in score_info] == keys
    assert [score["constant"] for score in score_info] == [1.0, 1.0, 1.0]
    # The rewritten row keeps the position it had in the resumed file.
    written = _read_jsonl(output_file)
    assert [row["key"] for row in written] == [keys[-1], keys[0], keys[1]]
    assert [row["constant"] for row in written] == [1.0, 1.0, 1.0]


def test_one_worker_uses_existing_serial_path(monkeypatch):
    scorer, metric_suite, gen_files = _scorer_and_files()
    monkeypatch.setattr(
        scorer,
        "_score_utterances_parallel",
        lambda *args, **kwargs: pytest.fail("parallel path was used"),
    )

    ConstantMetric.calls = 0
    score_info = scorer.score_utterances(
        gen_files, metric_suite, io="soundfile", num_workers=1
    )

    assert len(score_info) == len(gen_files)
    assert ConstantMetric.calls == len(gen_files)
