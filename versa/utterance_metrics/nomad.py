#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for NOMAD speech quality assessment metrics."""

import logging
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Handle optional nomad dependency
try:
    from nomad_versa import Nomad

    NOMAD_AVAILABLE = True
except ImportError:
    logger.warning(
        "nomad is not installed. Please use `tools/install_nomad.sh` to install"
    )
    Nomad = None
    NOMAD_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class NomadNotAvailableError(RuntimeError):
    """Exception raised when nomad is required but not available."""

    pass


def is_nomad_available():
    """
    Check if the nomad package is available.

    Returns:
        bool: True if nomad is available, False otherwise.
    """
    return NOMAD_AVAILABLE


class NomadMetric(BaseMetric):
    """NOMAD speech quality assessment metric."""

    TARGET_FS = 16000  # NOMAD model's expected sampling rate

    def _setup(self):
        """Initialize NOMAD-specific components."""
        if not NOMAD_AVAILABLE:
            raise ImportError(
                "nomad is not installed. Please use `tools/install_nomad.sh` to install"
            )

        self.use_gpu = self.config.get("use_gpu", False)
        self.cache_dir = self.config.get("model_cache", "versa_cache/nomad_pt-models")

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NOMAD model: {str(e)}") from e

    def _setup_model(self):
        """Setup the NOMAD model."""
        device = "cuda" if self.use_gpu else "cpu"

        if Nomad is None:
            raise ModuleNotFoundError(
                "nomad is not installed. Please use `tools/install_nomad.sh` to install"
            )

        return Nomad(device=device, cache_dir=self.cache_dir)

    def compute(
        self, predictions: Any, references: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate NOMAD score for speech quality assessment.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing NOMAD score.
        """
        pred_x = predictions
        gt_x = references
        fs = metadata.get("sample_rate", 16000) if metadata else 16000

        # Validate inputs
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")
        if gt_x is None:
            raise ValueError("Reference signal must be provided")

        pred_x = np.asarray(pred_x)
        gt_x = np.asarray(gt_x)

        # Resample if necessary (NOMAD only supports 16kHz)
        if fs != self.TARGET_FS:
            gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=self.TARGET_FS)
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=self.TARGET_FS)

        return {
            "nomad": self.model.predict(nmr=gt_x, deg=pred_x),
        }

    def get_metadata(self) -> MetricMetadata:
        """Return NOMAD metric metadata."""
        return MetricMetadata(
            name="nomad",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["nomad_versa", "torch", "librosa", "numpy"],
            description="NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment",
            paper_reference="https://ieeexplore.ieee.org/document/10447047",
            implementation_source="https://github.com/alessandroragano/nomad",
        )


def register_nomad_metric(registry):
    """Register NOMAD metric with the registry."""
    metric_metadata = MetricMetadata(
        name="nomad",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["nomad_versa", "torch", "librosa", "numpy"],
        description="NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment",
        paper_reference="https://ieeexplore.ieee.org/document/10447047",
        implementation_source="https://github.com/alessandroragano/nomad",
    )
    registry.register(
        NomadMetric,
        metric_metadata,
        aliases=["Nomad", "nomad"],
    )
