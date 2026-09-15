import math
import os

import pytest
import yaml

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.utterance_metrics.speaker import (
    is_transformers_available,
    register_speaker_metric,
)

RUN_REAL_MODEL_TESTS = os.environ.get("VERSA_RUN_REAL_MODEL_TESTS") == "1"


def _load_wavlm_speaker_config():
    """Load the example, optionally selecting the CI-prepared immutable snapshot."""
    with open(
        "egs/separate_metrics/spk_similarity_wavlm.yaml", "r", encoding="utf-8"
    ) as f:
        config = yaml.safe_load(f)
    model_path = os.environ.get("VERSA_WAVLM_MODEL_PATH")
    if model_path:
        for metric in config:
            metric["model_tag"] = model_path
            metric["backend"] = "huggingface"
    return config


def _sample_files():
    gen_path = "test/test_samples/test2"
    gt_path = "test/test_samples/test1"
    if not os.path.isdir(gen_path) or not os.path.isdir(gt_path):
        pytest.skip("Required test sample directories are not available")
    return find_files(gen_path), find_files(gt_path)


@pytest.mark.real_model
@pytest.mark.skipif(
    not RUN_REAL_MODEL_TESTS,
    reason="Set VERSA_RUN_REAL_MODEL_TESTS=1 to run real model-backed checks",
)
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_speaker_wavlm_pipeline_with_real_model(record_testsuite_property):
    """Run the WavLM speaker backend through the registry/scorer path."""
    gen_files, gt_files = _sample_files()
    score_config = _load_wavlm_speaker_config()

    registry = MetricRegistry()
    register_speaker_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(score_config, use_gt=True, use_gpu=False)

    assert len(score_config) > 0, "no scoring function is provided"
    assert set(metric_suite.metrics) == {"speaker"}

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        gt_files=gt_files,
        output_file=None,
        io="soundfile",
    )
    summary = compute_summary(score_info)

    assert "spk_similarity" in summary
    assert math.isfinite(summary["spk_similarity"])
    assert -1.0 <= summary["spk_similarity"] <= 1.0
    record_testsuite_property("spk_similarity", summary["spk_similarity"])


if __name__ == "__main__":
    if not RUN_REAL_MODEL_TESTS:
        raise SystemExit("Set VERSA_RUN_REAL_MODEL_TESTS=1 to run this check")
    pytest.main([__file__, "-q", "-s"])
