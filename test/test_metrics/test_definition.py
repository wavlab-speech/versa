import pytest

from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricFactory,
    MetricMetadata,
    MetricRegistry,
    MetricSuite,
    MetricType,
)
from versa.scorer_shared import compute_summary


class DummyMetric(BaseMetric):
    def _setup(self):
        pass

    def compute(self, predictions, references=None, metadata=None):
        return {"dummy": 1.0}

    def get_metadata(self):
        return DUMMY_METADATA


DUMMY_METADATA = MetricMetadata(
    name="dummy",
    category=MetricCategory.INDEPENDENT,
    metric_type=MetricType.FLOAT,
    requires_reference=False,
    requires_text=False,
    gpu_compatible=False,
    auto_install=False,
    dependencies=["definitely_missing_dependency_for_versa_test"],
    description="Dummy metric for registry tests.",
)


def test_metric_factory_create_suite_with_missing_dependency_and_default_config():
    registry = MetricRegistry()
    registry.register(DummyMetric, DUMMY_METADATA)

    suite = MetricFactory(registry).create_metric_suite(["dummy"])

    assert suite.compute_all(predictions=None) == {"dummy": {"dummy": 1.0}}


def test_metric_suite_compute_parallel_is_explicitly_unimplemented():
    suite = MetricSuite({})

    with pytest.raises(NotImplementedError, match="compute_parallel"):
        suite.compute_parallel(predictions=None)


def test_compute_summary_infers_numeric_scores_without_metric_registry():
    score_info = [
        {
            "key": "utt1",
            "pesq": 1.0,
            "match_details": {"ok": True},
            "ref_text": "hello",
            "espnet_wer_insert": 1,
        },
        {
            "key": "utt2",
            "pesq": 3.0,
            "match_details": {"ok": False},
            "ref_text": "world",
            "espnet_wer_insert": 2,
        },
    ]

    assert compute_summary(score_info) == {
        "espnet_wer_insert": 3,
        "pesq": 2.0,
    }
