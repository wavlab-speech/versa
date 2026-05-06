#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np
import torch

try:
    import torchaudio.functional as F
    from torchaudio.pipelines import SQUIM_OBJECTIVE, SQUIM_SUBJECTIVE
except ImportError:
    logging.warning(
        "Import error. Please install pesq, pystoi, torchaudio for torch squim"
    )
    F = None
    SQUIM_OBJECTIVE = None
    SQUIM_SUBJECTIVE = None

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

SQUIM_AVAILABLE = SQUIM_OBJECTIVE is not None and SQUIM_SUBJECTIVE is not None


def is_squim_available():
    return SQUIM_AVAILABLE


def squim_metric(pred_x, gt_x, fs):
    """
    Reference:
    Kumar et al., "TorchAudio-Squim: Reference-less Speech Quality and
    Intelligibility measures in TorchAudio", ICASSP 2023.
    https://pytorch.org/audio/main/tutorials/squim_tutorial.html

    """
    gt_x = torch.from_numpy(gt_x)
    pred_x = torch.from_numpy(pred_x)

    if fs != 16000:
        gt_x = F.resample(gt_x, fs, 16000)
        pred_x = F.resample(pred_x, fs, 16000)

    gt_x = gt_x.unsqueeze(0).float()
    pred_x = pred_x.unsqueeze(0).float()

    subjective_model = SQUIM_SUBJECTIVE.get_model()
    torch_squim_mos = subjective_model(pred_x, gt_x)

    return {"torch_squim_mos": torch_squim_mos.detach().numpy()[0]}


def squim_metric_no_ref(pred_x, fs):
    """
    Reference:
    Kumar et al., "TorchAudio-Squim: Reference-less Speech Quality and
    Intelligibility measures in TorchAudio", ICASSP 2023.
    https://pytorch.org/audio/main/tutorials/squim_tutorial.html

    """
    pred_x = torch.from_numpy(pred_x)
    if fs != 16000:
        pred_x = F.resample(pred_x, fs, 16000)

    pred_x = pred_x.unsqueeze(0).float()

    objective_model = SQUIM_OBJECTIVE.get_model()
    torch_squim_stoi, torch_squim_pesq, torch_squim_si_sdr = objective_model(pred_x)

    return {
        "torch_squim_stoi": torch_squim_stoi.detach().numpy()[0],
        "torch_squim_pesq": torch_squim_pesq.detach().numpy()[0],
        "torch_squim_si_sdr": torch_squim_si_sdr.detach().numpy()[0],
    }


class SquimMetric(BaseMetric):
    """TorchAudio-SQUIM speech quality metric."""

    def _setup(self):
        if not SQUIM_AVAILABLE:
            raise ImportError(
                "SQUIM is not available. Please install pesq, pystoi, and torchaudio"
            )
        self.mode = self.config.get("mode", "no_ref")
        if self.mode not in {"ref", "no_ref"}:
            raise ValueError(f"Invalid SQUIM mode: {self.mode}")
        if self.mode == "ref":
            self.model = SQUIM_SUBJECTIVE.get_model()
        else:
            self.model = SQUIM_OBJECTIVE.get_model()

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")
        if self.mode == "ref" and references is None:
            raise ValueError("Reference signal must be provided for SQUIM ref mode")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        pred_x = torch.from_numpy(np.asarray(predictions))
        if fs != 16000:
            pred_x = F.resample(pred_x, fs, 16000)
        pred_x = pred_x.unsqueeze(0).float()

        if self.mode == "ref":
            gt_x = torch.from_numpy(np.asarray(references))
            if fs != 16000:
                gt_x = F.resample(gt_x, fs, 16000)
            gt_x = gt_x.unsqueeze(0).float()
            torch_squim_mos = self.model(pred_x, gt_x)
            return {"torch_squim_mos": torch_squim_mos.detach().numpy()[0]}

        torch_squim_stoi, torch_squim_pesq, torch_squim_si_sdr = self.model(pred_x)
        return {
            "torch_squim_stoi": torch_squim_stoi.detach().numpy()[0],
            "torch_squim_pesq": torch_squim_pesq.detach().numpy()[0],
            "torch_squim_si_sdr": torch_squim_si_sdr.detach().numpy()[0],
        }

    def get_metadata(self):
        return _squim_metadata(f"squim_{self.mode}", self.mode)


class SquimRefMetric(SquimMetric):
    """Reference-based TorchAudio-SQUIM MOS metric."""

    def _setup(self):
        self.config = {**self.config, "mode": self.config.get("mode", "ref")}
        super()._setup()


class SquimNoRefMetric(SquimMetric):
    """Reference-less TorchAudio-SQUIM objective metrics."""

    def _setup(self):
        self.config = {**self.config, "mode": self.config.get("mode", "no_ref")}
        super()._setup()


def _squim_metadata(name, mode):
    requires_reference = mode == "ref"
    description = (
        "TorchAudio-SQUIM subjective MOS metric"
        if requires_reference
        else "TorchAudio-SQUIM reference-less PESQ, STOI, and SI-SDR metrics"
    )
    return MetricMetadata(
        name=name,
        category=(
            MetricCategory.DEPENDENT
            if requires_reference
            else MetricCategory.INDEPENDENT
        ),
        metric_type=MetricType.DICT,
        requires_reference=requires_reference,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["torch", "torchaudio"],
        description=description,
        paper_reference="https://arxiv.org/abs/2302.01147",
        implementation_source=(
            "https://pytorch.org/audio/main/tutorials/squim_tutorial.html"
        ),
    )


def register_squim_metric(registry):
    """Register TorchAudio-SQUIM metrics with the registry."""
    registry.register(
        SquimRefMetric,
        _squim_metadata("squim_ref", "ref"),
        aliases=["torch_squim_mos"],
    )
    registry.register(
        SquimNoRefMetric,
        _squim_metadata("squim_no_ref", "no_ref"),
        aliases=["squim", "torch_squim_objective"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = SquimRefMetric()
    scores = metric.compute(a, b, metadata={"sample_rate": 16000})
    print(scores)
