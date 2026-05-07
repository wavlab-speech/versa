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
