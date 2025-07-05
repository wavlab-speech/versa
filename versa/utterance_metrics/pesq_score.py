#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for PESQ (Perceptual Evaluation of Speech Quality) metrics."""

import logging
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Handle optional pesq dependency
try:
    from pesq import pesq

    PESQ_AVAILABLE = True
except ImportError:
    logger.warning(
        "pesq is not properly installed. Please install pesq and retry: pip install pesq"
    )
    pesq = None
    PESQ_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class PesqNotAvailableError(RuntimeError):
    """Exception raised when pesq is required but not available."""

    pass


def is_pesq_available():
    """
    Check if the pesq package is available.

    Returns:
        bool: True if pesq is available, False otherwise.
    """
    return PESQ_AVAILABLE


class PesqMetric(BaseMetric):
    """PESQ (Perceptual Evaluation of Speech Quality) metric."""

    def _setup(self):
        """Initialize PESQ-specific components."""
        if not PESQ_AVAILABLE:
            raise ImportError(
                "pesq is not properly installed. Please install pesq and retry: pip install pesq"
            )

    def compute(
        self, predictions: Any, references: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate PESQ score for speech quality assessment.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing PESQ score.
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

        try:
            if fs == 8000:
                pesq_value = pesq(8000, gt_x, pred_x, "nb")
            elif fs < 16000:
                logger.info("not support fs {}, resample to 8khz".format(fs))
                new_gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=8000)
                new_pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=8000)
                pesq_value = pesq(8000, new_gt_x, new_pred_x, "nb")
            elif fs == 16000:
                pesq_value = pesq(16000, gt_x, pred_x, "wb")
            else:
                logger.info("not support fs {}, resample to 16khz".format(fs))
                new_gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
                new_pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
                pesq_value = pesq(16000, new_gt_x, new_pred_x, "wb")
        except BaseException:
            logger.warning(
                "Error from pesq calculation. Please check the audio (likely due to silence)"
            )
            pesq_value = 0.0

        return {"pesq": pesq_value}

    def get_metadata(self) -> MetricMetadata:
        """Return PESQ metric metadata."""
        return MetricMetadata(
            name="pesq",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=False,
            auto_install=False,
            dependencies=["pesq", "librosa", "numpy"],
            description="PESQ: Perceptual Evaluation of Speech Quality",
            paper_reference="https://www.itu.int/rec/T-REC-P.862",
            implementation_source="https://github.com/ludlows/python-pesq",
        )


def register_pesq_metric(registry):
    """Register PESQ metric with the registry."""
    metric_metadata = MetricMetadata(
        name="pesq",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["pesq", "librosa", "numpy"],
        description="PESQ: Perceptual Evaluation of Speech Quality",
        paper_reference="https://www.itu.int/rec/T-REC-P.862",
        implementation_source="https://github.com/ludlows/python-pesq",
    )
    registry.register(
        PesqMetric,
        metric_metadata,
        aliases=["Pesq", "pesq", "perceptual_evaluation_speech_quality"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    metric = PesqMetric()
    scores = metric.compute(a, b, metadata={"sample_rate": 16000})
    print(scores)
