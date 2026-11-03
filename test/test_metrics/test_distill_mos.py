"""Contract tests that do not download optional pretrained assets."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from versa.definition import MetricRegistry
from versa.utterance_metrics import distill_mos


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.waveform = None

    def forward(self, waveform):
        assert not torch.is_grad_enabled()
        self.waveform = waveform
        return torch.tensor([[3.25]], device=waveform.device)


@pytest.mark.parametrize("length", [1, 122879, 122880, 122881, 160000])
def test_preserves_full_waveform_for_upstream_segmentation(length):
    model = RecordingModel()
    result = distill_mos.distill_mos_calculate(model, np.zeros(length), 16000)
    assert result == {"distill_mos": 3.25}
    assert model.waveform.shape == (1, length)
    assert model.waveform.dtype == torch.float32


@pytest.mark.parametrize("sample_rate", [8000, np.int64(8000)])
def test_resampling(sample_rate):
    model = RecordingModel()
    distill_mos.distill_mos_calculate(model, np.zeros(8000), sample_rate)
    assert model.waveform.shape == (1, 16000)


@pytest.mark.parametrize("audio", [[], [[0.0, 1.0]], [float("nan")], [float("inf")]])
def test_invalid_audio(audio):
    with pytest.raises(ValueError):
        distill_mos.distill_mos_calculate(RecordingModel(), audio, 16000)


@pytest.mark.parametrize("sample_rate", [0, -1, True, 16000.5])
def test_invalid_rate(sample_rate):
    with pytest.raises(ValueError):
        distill_mos.distill_mos_calculate(RecordingModel(), [0.0], sample_rate)


def test_setup_and_metric(monkeypatch):
    model = RecordingModel()

    def constructor(**kwargs):
        assert kwargs == {"segmenting_in_forward": True}
        return model

    monkeypatch.setitem(
        sys.modules, "distillmos", SimpleNamespace(ConvTransformerSQAModel=constructor)
    )
    metric = distill_mos.DistillMosMetric()
    assert not model.training
    assert metric.compute(np.zeros(8000), metadata={"sample_rate": 8000}) == {
        "distill_mos": 3.25
    }


def test_missing_backend(monkeypatch):
    monkeypatch.setitem(sys.modules, "distillmos", None)
    with pytest.raises(ModuleNotFoundError, match="install_distill_mos.sh"):
        distill_mos.distill_mos_setup()


def test_registration():
    registry = MetricRegistry()
    distill_mos.register_distill_mos_metric(registry)
    assert registry.get_metric("distillmos") is distill_mos.DistillMosMetric
    assert registry.get_metadata("distill_mos").requires_reference is False
    from versa.metric_registry import load_metric_symbol

    assert load_metric_symbol("DistillMosMetric") is distill_mos.DistillMosMetric
