import numpy as np

from versa.definition import MetricRegistry
from versa.utterance_metrics import wvmos
from versa.utterance_metrics.wvmos import WvmosMetric, register_wvmos_metric


def test_wvmos_metric_class_returns_existing_key(monkeypatch):
    monkeypatch.setattr(wvmos, "wvmos_setup", lambda use_gpu=False: object())
    monkeypatch.setattr(
        wvmos,
        "wvmos_calculate",
        lambda model, pred_x, gen_sr: {"wvmos": 0.75},
    )

    metric = WvmosMetric({"use_gpu": False})
    result = metric.compute(
        np.zeros(16000, dtype=np.float32), metadata={"sample_rate": 16000}
    )

    assert result == {"wvmos": 0.75}


def test_register_wvmos_metric():
    registry = MetricRegistry()

    register_wvmos_metric(registry)

    assert registry.get_metric("wvmos") is WvmosMetric
    assert registry.get_metric("wv_mos") is WvmosMetric
    assert registry.get_metadata("wvmos").requires_reference is False
