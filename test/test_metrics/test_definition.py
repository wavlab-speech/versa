from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricFactory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)


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
