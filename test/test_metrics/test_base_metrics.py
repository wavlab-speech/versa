import numpy as np
import pytest
import torch

from versa.definition import MetricRegistry
from versa.sequence_metrics.signal_metric import SignalMetric, register_signal_metric
from versa.utterance_metrics.pysepm import PysepmMetric, register_pysepm_metric
from versa.utterance_metrics.squim import (
    SquimNoRefMetric,
    SquimRefMetric,
    register_squim_metric,
)


def _audio_pair(length=16000):
    t = np.linspace(0, 1, length, endpoint=False)
    pred = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    ref = 0.5 * np.sin(2 * np.pi * 221 * t).astype(np.float32)
    return pred, ref


def test_signal_metric_class_returns_existing_keys():
    pred, ref = _audio_pair()
    metric = SignalMetric()

    scores = metric.compute(pred, ref)

    assert set(scores) == {"sdr", "sir", "sar", "si_snr", "ci_sdr"}
    assert all(isinstance(value, (float, np.floating)) for value in scores.values())


def test_register_signal_metric():
    registry = MetricRegistry()

    register_signal_metric(registry)

    assert registry.get_metric("signal_metric") is SignalMetric
    assert registry.get_metric("signal") is SignalMetric
    assert registry.get_metadata("signal_metric").requires_reference is True


def test_squim_no_ref_metric_uses_cached_model(monkeypatch):
    class DummyObjectiveBundle:
        @staticmethod
        def get_model():
            return lambda pred_x: (
                torch.tensor([0.6]),
                torch.tensor([1.2]),
                torch.tensor([-3.4]),
            )

    monkeypatch.setattr("versa.utterance_metrics.squim.SQUIM_AVAILABLE", True)
    monkeypatch.setattr(
        "versa.utterance_metrics.squim.SQUIM_OBJECTIVE", DummyObjectiveBundle
    )

    pred, _ = _audio_pair()
    metric = SquimNoRefMetric()
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {
        "torch_squim_stoi": pytest.approx(0.6),
        "torch_squim_pesq": pytest.approx(1.2),
        "torch_squim_si_sdr": pytest.approx(-3.4),
    }


def test_squim_ref_metric_uses_cached_model(monkeypatch):
    class DummySubjectiveBundle:
        @staticmethod
        def get_model():
            return lambda pred_x, ref_x: torch.tensor([4.2])

    monkeypatch.setattr("versa.utterance_metrics.squim.SQUIM_AVAILABLE", True)
    monkeypatch.setattr(
        "versa.utterance_metrics.squim.SQUIM_SUBJECTIVE", DummySubjectiveBundle
    )

    pred, ref = _audio_pair()
    metric = SquimRefMetric()
    scores = metric.compute(pred, ref, metadata={"sample_rate": 16000})

    assert scores == {"torch_squim_mos": pytest.approx(4.2)}


def test_register_squim_metric():
    registry = MetricRegistry()

    register_squim_metric(registry)

    assert registry.get_metric("squim_ref") is SquimRefMetric
    assert registry.get_metric("squim_no_ref") is SquimNoRefMetric
    assert registry.get_metric("squim") is SquimNoRefMetric
    assert registry.get_metadata("squim_ref").requires_reference is True
    assert registry.get_metadata("squim_no_ref").requires_reference is False


def test_pysepm_registration_and_missing_dependency(monkeypatch):
    registry = MetricRegistry()
    register_pysepm_metric(registry)

    assert registry.get_metric("pysepm") is PysepmMetric
    assert registry.get_metric("pysepm_metric") is PysepmMetric
    assert registry.get_metadata("pysepm").requires_reference is True

    monkeypatch.setattr("versa.utterance_metrics.pysepm.pysepm", None)
    with pytest.raises(ImportError, match="pysepm is not installed"):
        PysepmMetric()
