import importlib.util
import math
import os

import pytest
import yaml

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.utterance_metrics.scoreq import register_scoreq_metric

TEST_INFO = {"scoreq_ref": 1.0068472623825073, "scoreq_nr": 1.7731}
RUN_REAL_MODEL_TESTS = os.environ.get("VERSA_RUN_REAL_MODEL_TESTS") == "1"


def _load_scoreq_config():
    with open("egs/separate_metrics/scoreq.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _sample_files():
    gen_path = "test/test_samples/test2"
    gt_path = "test/test_samples/test1"
    if not os.path.isdir(gen_path) or not os.path.isdir(gt_path):
        pytest.skip("Required test sample directories are not available")
    return find_files(gen_path), find_files(gt_path)


def _scoreq_is_available():
    return importlib.util.find_spec("scoreq_versa") is not None


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not _scoreq_is_available(),
    reason="scoreq_versa is not installed; run tools/install_scoreq.sh first",
)
def test_scoreq_pipeline_with_real_model():
    """Run ScoreQ through the registry/scorer path with real model inference."""
    gen_files, gt_files = _sample_files()
    score_config = _load_scoreq_config()

    registry = MetricRegistry()
    register_scoreq_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(score_config, use_gt=True, use_gpu=False)

    assert len(score_config) > 0, "no scoring function is provided"
    assert set(metric_suite.metrics) == {"scoreq_ref", "scoreq_nr"}

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        gt_files=gt_files,
        output_file=None,
        io="soundfile",
    )
    summary = compute_summary(score_info)

    for key, expected_value in TEST_INFO.items():
        assert key in summary
        if math.isinf(expected_value):
            assert math.isinf(summary[key])
        else:
            assert summary[key] == pytest.approx(expected_value, abs=1e-4)


if __name__ == "__main__":
    if not RUN_REAL_MODEL_TESTS:
        raise SystemExit("Set VERSA_RUN_REAL_MODEL_TESTS=1 to run this check")
    pytest.main([__file__, "-q", "-s"])
