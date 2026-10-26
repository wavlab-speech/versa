#!/usr/bin/env python3

# Copyright 2026 Jiatong Shi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from versa.definition import MetricFactory, MetricRegistry

PSEUDO_MOS_PATH = (
    Path(__file__).resolve().parents[2]
    / "versa"
    / "utterance_metrics"
    / "pseudo_mos.py"
)
spec = importlib.util.spec_from_file_location("pseudo_mos", PSEUDO_MOS_PATH)
pseudo_mos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pseudo_mos)


class DummyUTMOSv2Model:
    def __init__(self):
        self.cfg = SimpleNamespace()
        self.data_type = None

    def __call__(self, pred_tensor, spec_info, data_type):
        self.data_type = data_type.detach().cpu().numpy()
        return torch.ones((pred_tensor.shape[0], 1), dtype=torch.float32)


class DummyKOMOSModel:
    """Minimal optional KOMOS adapter used to verify model setup."""

    loaded_model_id = None
    loaded_cache_dir = None

    @classmethod
    def from_pretrained(cls, model_id, *, cache_dir):
        """Record the requested model identifier and return a test adapter."""
        cls.loaded_model_id = model_id
        cls.loaded_cache_dir = cache_dir
        return cls()

    def to(self, device):
        """Record the requested execution device."""
        self.device = device
        return self

    def eval(self):
        """Record that setup selected inference mode."""
        self.evaluating = True
        return self


def test_komos_setup_loads_optional_model(monkeypatch, tmp_path):
    """The optional backend loads the published model on the requested device."""
    monkeypatch.setitem(sys.modules, "komos", SimpleNamespace(KOMOS=DummyKOMOSModel))

    predictors, sample_rates = pseudo_mos.pseudo_mos_setup(
        ["komos"], {}, cache_dir=str(tmp_path), use_gpu=False
    )

    assert DummyKOMOSModel.loaded_model_id == "upskyy/komos"
    assert DummyKOMOSModel.loaded_cache_dir == str(tmp_path)
    assert predictors["komos"].device == "cpu"
    assert predictors["komos"].evaluating is True
    assert sample_rates["komos"] == 16000


class RecordingKOMOS:
    """KOMOS adapter that records waveform scoring calls."""

    loaded_models = []

    def __init__(self):
        """Initialize an empty scoring-call log."""
        self.calls = []

    @classmethod
    def from_pretrained(cls, model_id, *, cache_dir):
        """Create and retain an adapter for later call assertions."""
        model = cls()
        cls.loaded_models.append((model_id, model))
        return model

    def to(self, device):
        """Record the configured execution device."""
        self.device = device
        return self

    def eval(self):
        """Return the adapter unchanged for inference-mode chaining."""
        return self

    def score_long_form(self, audio, sr):
        """Record the waveform call and return a deterministic MOS."""
        self.calls.append((audio, sr))
        return 4.2


@pytest.mark.parametrize("duration", [1, 10, 11, 60])
def test_komos_returns_distinct_score_through_registry(monkeypatch, duration):
    """The registered metric preserves input rate and returns a KOMOS score."""
    audio = np.ones(48000 * duration, dtype=np.float32)
    RecordingKOMOS.loaded_models = []
    monkeypatch.setitem(sys.modules, "komos", SimpleNamespace(KOMOS=RecordingKOMOS))

    registry = MetricRegistry()
    pseudo_mos.register_pseudo_mos_metric(registry)
    metric = MetricFactory(registry).create_metric(
        "pseudo_mos", {"predictor_types": ["komos"]}
    )

    scores = metric.compute(
        audio,
        metadata={"sample_rate": 48000},
    )

    assert scores == {"komos": 4.2}
    model_id, predictor = RecordingKOMOS.loaded_models[0]
    assert model_id == "upskyy/komos"
    assert predictor.calls[0][0] is audio
    assert predictor.calls[0][1] == 48000


def test_komos_propagates_long_form_limit_error():
    """Inputs beyond the backend limit fail instead of producing partial MOS."""

    class LimitedKOMOS:
        """Represent a backend rejecting an oversized recording."""

        def score_long_form(self, audio: np.ndarray, sr: int) -> float:
            """Reject the oversized waveform with an actionable backend error."""
            message = "Audio exceeds the 60-second score_long_form() limit."
            raise ValueError(message)

    audio = np.ones(16000 * 61, dtype=np.float32)

    with pytest.raises(ValueError, match="60-second"):
        pseudo_mos.pseudo_mos_metric(
            audio,
            fs=16000,
            predictor_dict={"komos": LimitedKOMOS()},
            predictor_fs={"komos": 16000},
        )


def test_utmosv2_uses_sarulab_dataset_label(monkeypatch):
    monkeypatch.setattr(pseudo_mos, "utmosv2", object())
    monkeypatch.setattr(
        pseudo_mos,
        "process_audio_only_versa",
        lambda pred, cfg: np.ones((2, 3), dtype=np.float32),
        raising=False,
    )

    model = DummyUTMOSv2Model()
    scores = pseudo_mos.pseudo_mos_metric(
        np.ones(160, dtype=np.float32),
        fs=16000,
        predictor_dict={"utmosv2": model},
        predictor_fs={"utmosv2": 16000},
    )

    assert scores["utmosv2"] == 1.0
    assert model.data_type.shape == (1, 10)
    assert model.data_type[0, 1] == 1.0
    assert model.data_type.sum() == 1.0


class RecordingRunPredictor:
    """Stand-in for DNSMOS/PLCMOS `.run(audio, sr)` that records its inputs."""

    def __init__(self, result):
        self.result = result
        self.calls = []

    def run(self, audio, sr):
        self.calls.append((len(audio), sr))
        return self.result


class RecordingSingMOS:
    """Stand-in for a SingMOS model that records the input length."""

    def __init__(self):
        self.lengths = []

    def __call__(self, pred_tensor, length_tensor):
        self.lengths.append(int(length_tensor[0]))
        return torch.ones(1)


def test_dnsmos_does_not_change_rate_for_later_predictors():
    """Predictors after DNSMOS must still resample from the input rate."""
    fs_in = 48000
    t = np.arange(fs_in, dtype=np.float32) / fs_in
    pred = 0.5 * np.sin(2 * np.pi * 220.0 * t).astype(np.float32)

    dnsmos = RecordingRunPredictor({"ovrl_mos": 3.0, "p808_mos": 3.5})
    plcmos = RecordingRunPredictor({"plcmos": 4.0})
    singmos = RecordingSingMOS()
    scores = pseudo_mos.pseudo_mos_metric(
        pred,
        fs=fs_in,
        predictor_dict={"dnsmos": dnsmos, "plcmos": plcmos, "singmos": singmos},
        predictor_fs={"dnsmos": 16000, "plcmos": 16000, "singmos": 16000},
    )

    assert dnsmos.calls == [(16000, 16000)]
    assert plcmos.calls == [(16000, 16000)]
    assert singmos.lengths == [16000]
    assert scores["dns_overall"] == 3.0
    assert scores["plcmos"] == 4.0
