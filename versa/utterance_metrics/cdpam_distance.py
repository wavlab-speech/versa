#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for CDPAM distance metrics."""

import logging
from functools import partial
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Handle optional cdpam dependency
try:
    import cdpam

    CDPAM_AVAILABLE = True
except ImportError:
    logger.warning("cdpam is not properly installed. " "Please install cdpam and retry")
    cdpam = None
    CDPAM_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class CdpamNotAvailableError(RuntimeError):
    """Exception raised when cdpam is required but not available."""

    pass


def is_cdpam_available():
    """
    Check if the cdpam package is available.

    Returns:
        bool: True if cdpam is available, False otherwise.
    """
    return CDPAM_AVAILABLE


class CdpamDistanceMetric(BaseMetric):
    """CDPAM distance metric."""

    TARGET_FS = 22050

    def _setup(self):
        """Initialize CDPAM-specific components."""
        if not CDPAM_AVAILABLE:
            raise ImportError(
                "cdpam is not properly installed. " "Please install cdpam and retry"
            )

        self.use_gpu = self.config.get("use_gpu", False)

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize CDPAM model: {str(e)}") from e

    def _setup_model(self):
        """Setup the CDPAM model."""
        device = "cpu" if not self.use_gpu else "cuda"
        # Suppress PyTorch config registration warnings during model loading
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Skipping config registration for"
            )
            _original_torch_load = torch.load
            torch.load = partial(torch.load, weights_only=False)
            model = cdpam.CDPAM(dev=device)
            torch.load = _original_torch_load
        return model

    def compute(
        self, predictions: Any, references: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate CDPAM distance between two audio samples.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the CDPAM distance score.
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

        pred_x = (torch.from_numpy(pred_x).unsqueeze(0) * 32768).round()
        gt_x = (torch.from_numpy(gt_x).unsqueeze(0) * 32768).round()
        dist = self.model.forward(gt_x, pred_x)

        return {"cdpam_distance": dist.detach().cpu().numpy().item()}

    def get_metadata(self) -> MetricMetadata:
        """Return CDPAM distance metric metadata."""
        return MetricMetadata(
            name="cdpam_distance",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["cdpam", "torch", "librosa", "numpy"],
            description="CDPAM distance between audio samples",
            paper_reference="https://github.com/facebookresearch/audiocraft",
            implementation_source="https://github.com/facebookresearch/audiocraft",
        )


def register_cdpam_distance_metric(registry):
    """Register CDPAM distance metric with the registry."""
    metric_metadata = MetricMetadata(
        name="cdpam_distance",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["cdpam", "torch", "librosa", "numpy"],
        description="CDPAM distance between audio samples",
        paper_reference="https://github.com/facebookresearch/audiocraft",
        implementation_source="https://github.com/facebookresearch/audiocraft",
    )
    registry.register(
        CdpamDistanceMetric,
        metric_metadata,
        aliases=["CdpamDistance", "cdpam_distance", "cdpam"],
    )


if __name__ == "__main__":
    a = np.random.random(22050)
    b = np.random.random(22050)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = CdpamDistanceMetric(config)
    metadata = {"sample_rate": 22050}
    score = metric.compute(a, b, metadata=metadata)
    print(f"metrics: {score}")
