#!/usr/bin/env python3

# Copyright 2026 Jiatong Shi
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

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
