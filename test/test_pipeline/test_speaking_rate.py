import os

import pytest
import yaml

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, compute_summary
from versa.utils_shared import find_files
from versa.utterance_metrics.speaking_rate import (
    SpeakingRateMetric,
    register_speaking_rate_metric,
)

TEST_INFO = {
    "speaking_rate": 4.0,
}


class DummyWhisperModel:
    def transcribe(self, audio, beam_size=5):
        return {"text": "one two three four"}


class DummyWhisper:
    @staticmethod
    def load_model(model_tag, device="cpu"):
        return DummyWhisperModel()


class DummyTextCleaner:
    def __init__(self, cleaner):
        self.cleaner = cleaner


@pytest.fixture()
def mocked_speaking_rate_dependencies(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.speaking_rate.whisper", DummyWhisper)
    monkeypatch.setattr(
        "versa.utterance_metrics.speaking_rate.TextCleaner", DummyTextCleaner
    )


def test_speaking_rate_metric_class_uses_existing_keys(
    mocked_speaking_rate_dependencies,
):
    metric = SpeakingRateMetric({"use_gpu": False})
    scores = metric.compute([0.0] * 16000, metadata={"sample_rate": 16000})

    assert scores == {
        "speaking_rate": pytest.approx(4.0),
        "whisper_hyp_text": "one two three four",
    }


def test_speaking_rate_registration():
    registry = MetricRegistry()

    register_speaking_rate_metric(registry)

    assert registry.get_metric("speaking_rate") is SpeakingRateMetric
    assert registry.get_metric("speaking_rate_metric") is SpeakingRateMetric
    assert registry.get_metric("swr") is SpeakingRateMetric
    assert registry.get_metadata("speaking_rate").requires_reference is False


def info_update(mocked_speaking_rate_dependencies=None):

    # find files
    if os.path.isdir("test/test_samples/test2"):
        gen_files = find_files("test/test_samples/test2")

    # find reference file
    if os.path.isdir("test/test_samples/test1"):
        gt_files = find_files("test/test_samples/test1")

    with open("egs/separate_metrics/speaking_rate.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)

    registry = MetricRegistry()
    register_speaking_rate_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        score_config,
        use_gt=(True if gt_files is not None else False),
        use_gpu=False,
    )

    assert len(score_config) > 0, "no scoring function is provided"

    score_info = scorer.score_utterances(
        gen_files, metric_suite, gt_files=gt_files, output_file=None, io="soundfile"
    )
    summary = compute_summary(score_info)
    print("Summary: {}".format(summary), flush=True)

    for key in TEST_INFO:
        assert summary[key] == pytest.approx(TEST_INFO[key], abs=1e-4)
    print("check successful", flush=True)


def test_speaking_rate_pipeline_with_registry(mocked_speaking_rate_dependencies):
    info_update(mocked_speaking_rate_dependencies)


if __name__ == "__main__":
    info_update()
