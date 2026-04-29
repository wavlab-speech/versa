import numpy as np
import pytest
import torch

from versa.definition import MetricRegistry
from versa.sequence_metrics.signal_metric import SignalMetric, register_signal_metric
from versa.utterance_metrics.pysepm import PysepmMetric, register_pysepm_metric
from versa.utterance_metrics.scoreq import (
    ScoreqNrMetric,
    ScoreqRefMetric,
    register_scoreq_metric,
)
from versa.utterance_metrics.sheet_ssqa import (
    SheetSsqaMetric,
    register_sheet_ssqa_metric,
)
from versa.utterance_metrics.squim import (
    SquimNoRefMetric,
    SquimRefMetric,
    register_squim_metric,
)
from versa.utterance_metrics.vad import VadMetric, register_vad_metric
from versa.utterance_metrics.vqscore import VqscoreMetric, register_vqscore_metric


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


def test_vad_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    def dummy_get_speech_ts(pred_x, model, **kwargs):
        calls["pred_x"] = pred_x
        calls["model"] = model
        calls["kwargs"] = kwargs
        return [{"start": 0.1, "end": 0.4}]

    monkeypatch.setattr(
        "versa.utterance_metrics.vad.torch.hub.load",
        lambda **kwargs: ("dummy-model", (dummy_get_speech_ts, None, None, None)),
    )

    pred, _ = _audio_pair()
    metric = VadMetric(
        {
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "max_speech_duration_s": 10,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 40,
        }
    )
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {"vad_info": [{"start": 0.1, "end": 0.4}]}
    assert calls["model"] == "dummy-model"
    assert calls["kwargs"]["sampling_rate"] == 16000
    assert calls["kwargs"]["threshold"] == 0.3
    assert calls["kwargs"]["min_speech_duration_ms"] == 100
    assert calls["kwargs"]["max_speech_duration_s"] == 10
    assert calls["kwargs"]["min_silence_duration_ms"] == 200
    assert calls["kwargs"]["speech_pad_ms"] == 40


def test_register_vad_metric():
    registry = MetricRegistry()

    register_vad_metric(registry)

    assert registry.get_metric("vad") is VadMetric
    assert registry.get_metric("silero_vad") is VadMetric
    assert registry.get_metadata("vad").requires_reference is False


def test_sheet_ssqa_metric_class_returns_existing_key(monkeypatch):
    class DummyInnerModel:
        def to(self, device):
            self.device = device
            return self

    class DummySheetModel:
        def __init__(self):
            self.model = DummyInnerModel()

        def predict(self, wav):
            return 3.25

    monkeypatch.setattr(
        "versa.utterance_metrics.sheet_ssqa.torch.hub.load",
        lambda *args, **kwargs: DummySheetModel(),
    )

    pred, _ = _audio_pair()
    metric = SheetSsqaMetric({"cache_dir": "test-cache", "use_gpu": False})
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {"sheet_ssqa": 3.25}


def test_register_sheet_ssqa_metric():
    registry = MetricRegistry()

    register_sheet_ssqa_metric(registry)

    assert registry.get_metric("sheet_ssqa") is SheetSsqaMetric
    assert registry.get_metric("sheet") is SheetSsqaMetric
    assert registry.get_metadata("sheet_ssqa").requires_reference is False


def test_scoreq_metric_classes_return_existing_keys(monkeypatch):
    calls = []

    class DummyScoreq:
        def __init__(self, data_domain, mode, cache_dir, device):
            self.mode = mode
            calls.append(
                {
                    "data_domain": data_domain,
                    "mode": mode,
                    "cache_dir": cache_dir,
                    "device": device,
                }
            )

        def predict(self, test_path, ref_path):
            assert test_path is not None
            if self.mode == "ref":
                assert ref_path is not None
                return 1.2
            assert ref_path is None
            return 2.4

    monkeypatch.setattr("versa.utterance_metrics.scoreq.Scoreq", DummyScoreq)

    pred, ref = _audio_pair()
    nr_metric = ScoreqNrMetric({"data_domain": "natural", "model_cache": "cache-a"})
    ref_metric = ScoreqRefMetric({"cache_dir": "cache-b"})

    assert nr_metric.compute(pred, metadata={"sample_rate": 16000}) == {
        "scoreq_nr": 2.4
    }
    assert ref_metric.compute(pred, ref, metadata={"sample_rate": 16000}) == {
        "scoreq_ref": 1.2
    }
    assert calls[0] == {
        "data_domain": "natural",
        "mode": "nr",
        "cache_dir": "cache-a",
        "device": "cpu",
    }
    assert calls[1]["mode"] == "ref"
    assert calls[1]["cache_dir"] == "cache-b"


def test_register_scoreq_metric():
    registry = MetricRegistry()

    register_scoreq_metric(registry)

    assert registry.get_metric("scoreq_nr") is ScoreqNrMetric
    assert registry.get_metric("scoreq_ref") is ScoreqRefMetric
    assert registry.get_metric("scoreq") is ScoreqNrMetric
    assert registry.get_metadata("scoreq_nr").requires_reference is False
    assert registry.get_metadata("scoreq_ref").requires_reference is True


def test_scoreq_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.scoreq.Scoreq", None)

    with pytest.raises(ModuleNotFoundError, match="scoreq is not installed"):
        ScoreqNrMetric()


def test_vqscore_metric_class_returns_existing_key(monkeypatch):
    class DummyVqscoreModel:
        device = "cpu"
        input_transform = "none"

        def CNN_1D_encoder(self, sp_input):
            return torch.ones((1, 2, 3))

        def quantizer(self, z, stochastic=False, update=False):
            return z.transpose(2, 1), None, None, None

    monkeypatch.setattr(
        "versa.utterance_metrics.vqscore.vqscore_setup",
        lambda use_gpu=False: DummyVqscoreModel(),
    )

    pred, _ = _audio_pair()
    metric = VqscoreMetric()
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {"vqscore": pytest.approx(1.0, abs=1e-4)}


def test_register_vqscore_metric():
    registry = MetricRegistry()

    register_vqscore_metric(registry)

    assert registry.get_metric("vqscore") is VqscoreMetric
    assert registry.get_metric("vq_score") is VqscoreMetric
    assert registry.get_metadata("vqscore").requires_reference is False
