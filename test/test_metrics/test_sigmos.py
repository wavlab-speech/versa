import numpy as np

from versa.definition import MetricRegistry
from versa.utterance_metrics import sigmos
from versa.utterance_metrics.sigmos import SigmosMetric, register_sigmos_metric


class DummySigmosModel:
    def run(self, audio, sr=None):
        return {
            "SIGMOS_COL": 1.0,
            "SIGMOS_DISC": 2.0,
            "SIGMOS_LOUD": 3.0,
            "SIGMOS_REVERB": 4.0,
            "SIGMOS_SIG": 5.0,
            "SIGMOS_OVRL": 6.0,
        }


def test_sigmos_metric_class_returns_expected_keys(monkeypatch):
    monkeypatch.setattr(
        sigmos, "sigmos_setup", lambda model_dir=None: DummySigmosModel()
    )

    metric = SigmosMetric({"model_dir": "unused"})
    result = metric.compute(
        np.zeros(48000, dtype=np.float32), metadata={"sample_rate": 48000}
    )

    assert result == {
        "SIGMOS_COL": 1.0,
        "SIGMOS_DISC": 2.0,
        "SIGMOS_LOUD": 3.0,
        "SIGMOS_REVERB": 4.0,
        "SIGMOS_SIG": 5.0,
        "SIGMOS_OVRL": 6.0,
    }


def test_register_sigmos_metric():
    registry = MetricRegistry()

    register_sigmos_metric(registry)

    assert registry.get_metric("sigmos") is SigmosMetric
    assert registry.get_metric("sig_mos") is SigmosMetric
    assert registry.get_metadata("sigmos").requires_reference is False
