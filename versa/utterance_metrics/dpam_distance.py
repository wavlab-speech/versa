#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for DPAM distance metrics."""

import logging
import urllib.request
import filelock
from pathlib import Path
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class lossnet(nn.Module):
    def __init__(self, nconv=14, nchan=32, dp=0.1, dist_act="no"):
        super(lossnet, self).__init__()
        self.nconv = nconv
        self.dist_act = dist_act
        self.convs = nn.ModuleList()
        self.chan_w = nn.ParameterList()
        for iconv in range(nconv):
            if iconv == 0:
                chin = 1
            else:
                chin = nchan
            if (iconv + 1) % 5 == 0:
                nchan = nchan * 2
            if iconv < nconv - 1:
                conv = [
                    nn.Conv1d(chin, nchan, 3, stride=2, padding=1),
                    nn.BatchNorm1d(nchan),
                    nn.LeakyReLU(),
                ]
                if dp != 0:
                    conv.append(nn.Dropout(p=dp))
            else:
                conv = [
                    nn.Conv1d(chin, nchan, 3, stride=1, padding=1),
                    nn.BatchNorm1d(nchan),
                    nn.LeakyReLU(),
                ]

            self.convs.append(nn.Sequential(*conv))
            self.chan_w.append(nn.Parameter(torch.randn(nchan), requires_grad=True))

        if dist_act == "sig":
            self.act = nn.Sigmoid()
        elif dist_act == "tanh":
            self.act = nn.Tanh()
        elif dist_act == "tshrink":
            self.act = nn.Tanhshrink()
        elif dist_act == "exp":
            self.act = None
        elif dist_act == "no":
            self.act = nn.Identity()
        else:
            self.act = None

    def forward(self, xref, xper):
        device = next(self.parameters()).device
        xref = xref.unsqueeze(1).to(device)
        xper = xper.unsqueeze(1).to(device)
        dist = 0
        for iconv in range(self.nconv):
            xref = self.convs[iconv](xref)
            xper = self.convs[iconv](xper)
            diff = (xref - xper).permute(0, 2, 1)
            wdiff = diff * self.chan_w[iconv]
            wdiff = (
                torch.sum(torch.abs(wdiff), dim=(1, 2)) / diff.shape[1] / diff.shape[2]
            )
            dist = dist + wdiff
        if self.dist_act == "exp":
            dist = torch.exp(torch.clamp(dist, max=20.0)) / (10**5)  # exp(20) ~ 4*10**8
        else:
            dist = self.act(dist)
        return dist


class DpamDistanceMetric(BaseMetric):
    """DPAM distance metric."""

    TARGET_FS = 22050
    MODEL_URL = "https://raw.githubusercontent.com/adrienchaton/PerceptualAudio_Pytorch/refs/heads/master/pretrained/dataset_combined_linear_tshrink.pth"

    def _setup(self):
        """Initialize DPAM-specific components."""
        self.use_gpu = self.config.get("use_gpu", False)
        self.cache_dir = self.config.get("cache_dir", "versa_cache")

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DPAM model: {str(e)}") from e

    def _setup_model(self):
        """Setup the DPAM model."""
        device = "cpu" if not self.use_gpu else "cuda"
        model_path = Path(self.cache_dir) / "dpam" / "dataset_combined_linear.pth"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        with filelock.FileLock(model_path.with_suffix(".lock")):
            if not model_path.exists():
                logger.info(f"Downloading model to {model_path}...")
                urllib.request.urlretrieve(self.MODEL_URL, model_path)
                logger.info("Download complete.")

        # Suppress PyTorch config registration warnings during model loading
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Skipping config registration for"
            )
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

        state = checkpoint["state"]
        prefix = "model_dist."
        state = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        model = lossnet(nconv=14, nchan=16, dp=0, dist_act="tshrink")
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model

    def compute(
        self, predictions: Any, references: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate DPAM distance between two audio samples.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the DPAM distance score.
        """
        pred_x = predictions
        gt_x = references
        fs = metadata.get("sample_rate", 22050) if metadata else 22050

        # Validate inputs
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")
        if gt_x is None:
            raise ValueError("Reference signal must be provided")

        pred_x = np.asarray(pred_x)
        gt_x = np.asarray(gt_x)

        if fs != self.TARGET_FS:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=self.TARGET_FS)
            gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=self.TARGET_FS)

        pred_x = torch.from_numpy(pred_x).unsqueeze(0).float()
        gt_x = torch.from_numpy(gt_x).unsqueeze(0).float()
        dist = self.model(gt_x, pred_x)

        return {"dpam_distance": dist.detach().cpu().numpy().item()}

    def get_metadata(self) -> MetricMetadata:
        """Return DPAM distance metric metadata."""
        return MetricMetadata(
            name="dpam_distance",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["torch", "librosa", "numpy", "filelock"],
            description="DPAM distance between audio samples",
            paper_reference="https://github.com/adrienchaton/PerceptualAudio_Pytorch",
            implementation_source="https://github.com/adrienchaton/PerceptualAudio_Pytorch",
        )


def register_dpam_distance_metric(registry):
    """Register DPAM distance metric with the registry."""
    metric_metadata = MetricMetadata(
        name="dpam_distance",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["torch", "librosa", "numpy", "filelock"],
        description="DPAM distance between audio samples",
        paper_reference="https://github.com/adrienchaton/PerceptualAudio_Pytorch",
        implementation_source="https://github.com/adrienchaton/PerceptualAudio_Pytorch",
    )
    registry.register(
        DpamDistanceMetric,
        metric_metadata,
        aliases=["DpamDistance", "dpam_distance", "dpam"],
    )


if __name__ == "__main__":
    a = np.random.random(22050)
    b = np.random.random(22050)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = DpamDistanceMetric(config)
    metadata = {"sample_rate": 22050}
    score = metric.compute(a, b, metadata=metadata)
    print(f"metrics: {score}")
