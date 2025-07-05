#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for OWSM Language Identification (LID) metrics."""

import logging
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Handle optional espnet2 dependency
try:
    from espnet2.bin.s2t_inference_language import Speech2Language

    ESPNET2_AVAILABLE = True
except ImportError:
    logger.warning(
        "espnet2 is not properly installed. " "Please install espnet2 and retry"
    )
    Speech2Language = None
    ESPNET2_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class Espnet2NotAvailableError(RuntimeError):
    """Exception raised when espnet2 is required but not available."""

    pass


def is_espnet2_available():
    """
    Check if the espnet2 package is available.

    Returns:
        bool: True if espnet2 is available, False otherwise.
    """
    return ESPNET2_AVAILABLE


class OwsmLidMetric(BaseMetric):
    """OWSM Language Identification (LID) metric."""

    TARGET_FS = 16000  # OWSM model's expected sampling rate

    def _setup(self):
        """Initialize OWSM LID-specific components."""
        if not ESPNET2_AVAILABLE:
            raise ImportError(
                "espnet2 is not properly installed. Please install espnet2 and retry"
            )

        self.model_tag = self.config.get("model_tag", "default")
        self.nbest = self.config.get("nbest", 3)
        self.use_gpu = self.config.get("use_gpu", False)

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OWSM LID model: {str(e)}") from e

    def _setup_model(self):
        """Setup the OWSM LID model."""
        device = "cuda" if self.use_gpu else "cpu"

        if self.model_tag == "default":
            model_tag = "espnet/owsm_v3.1_ebf"
        else:
            model_tag = self.model_tag

        model = Speech2Language.from_pretrained(
            model_tag=model_tag,
            device=device,
            nbest=self.nbest,
        )

        return model

    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate language identification for speech.

        Args:
            predictions: Audio signal to be evaluated.
            references: Not used for LID (single-ended metric).
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing language identification result.
        """
        pred_x = predictions
        fs = metadata.get("sample_rate", 16000) if metadata else 16000

        # Validate inputs
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")

        pred_x = np.asarray(pred_x)

        # Resample if necessary (OWSM only works with 16kHz)
        if fs != self.TARGET_FS:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=self.TARGET_FS)

        result = self.model(pred_x)
        return {"language": result}

    def get_metadata(self) -> MetricMetadata:
        """Return OWSM LID metric metadata."""
        return MetricMetadata(
            name="lid",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.LIST,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["espnet2", "librosa", "numpy"],
            description="OWSM Language Identification (LID) for speech",
            paper_reference="https://arxiv.org/abs/2309.16588",
            implementation_source="https://github.com/espnet/espnet",
        )


def register_owsm_lid_metric(registry):
    """Register OWSM LID metric with the registry."""
    metric_metadata = MetricMetadata(
        name="lid",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.LIST,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["espnet2", "librosa", "numpy"],
        description="OWSM Language Identification (LID) for speech",
        paper_reference="https://arxiv.org/abs/2309.16588",
        implementation_source="https://github.com/espnet/espnet",
    )
    registry.register(
        OwsmLidMetric,
        metric_metadata,
        aliases=["OwsmLid", "lid", "language_id"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    model = OwsmLidMetric()
    print("metrics: {}".format(model.compute(a, None, {"sample_rate": 16000})))
