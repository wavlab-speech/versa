#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for evaluating audio using AudioBox Aesthetics models."""

import json
import logging
import os
from typing import Dict, Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

# Handle optional audiobox_aesthetics dependency
try:
    import audiobox_aesthetics.infer
    import audiobox_aesthetics.utils

    AUDIOBOX_AESTHETICS_AVAILABLE = True
except ImportError:
    logger.warning(
        "audiobox_aesthetics is not properly installed. "
        "Please install with tools/install_audiobox-aesthetics.sh first."
    )
    audiobox_aesthetics = None
    AUDIOBOX_AESTHETICS_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class AudioBoxAestheticsNotAvailableError(RuntimeError):
    """Exception raised when AudioBox Aesthetics is required but not available."""

    pass


def is_audiobox_aesthetics_available():
    """
    Check if the AudioBox Aesthetics package is available.

    Returns:
        bool: True if AudioBox Aesthetics is available, False otherwise.
    """
    return AUDIOBOX_AESTHETICS_AVAILABLE


class AudioBoxAestheticsMetric(BaseMetric):
    """AudioBox Aesthetics metric for audio quality assessment."""

    def _setup(self):
        """Initialize AudioBox Aesthetics-specific components."""
        if not AUDIOBOX_AESTHETICS_AVAILABLE:
            raise ImportError(
                "audiobox_aesthetics is not properly installed. "
                "Please install with tools/install_audiobox-aesthetics.sh first."
            )

        self.model_path = self.config.get("model_path", None)
        self.batch_size = self.config.get("batch_size", 1)
        self.precision = self.config.get("precision", "bf16")
        self.cache_dir = self.config.get("cache_dir", "versa_cache/audiobox")
        self.use_huggingface = self.config.get("use_huggingface", True)
        self.use_gpu = self.config.get("use_gpu", False)

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize AudioBox Aesthetics model: {str(e)}"
            ) from e

    def _setup_model(self):
        """Setup the AudioBox Aesthetics model."""
        device = "cuda" if self.use_gpu else "cpu"

        if self.model_path is None:
            if self.use_huggingface:
                model_path = audiobox_aesthetics.utils.load_model(self.model_path)
            else:
                os.makedirs(self.cache_dir, exist_ok=True)
                model_path = os.path.join(
                    self.cache_dir, audiobox_aesthetics.utils.DEFAULT_CKPT_FNAME
                )
                model_url = audiobox_aesthetics.utils.DEFAULT_S3_URL
                if not os.path.exists(model_path):
                    print(f"Downloading model from {model_url} to {model_path}")
                    audiobox_aesthetics.utils.download_file(model_url, model_path)
        else:
            model_path = self.model_path

        predictor = audiobox_aesthetics.infer.AesWavlmPredictorMultiOutput(
            checkpoint_pth=model_path,
            device=device,
            batch_size=self.batch_size,
            precision=self.precision,
        )
        return predictor

    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate AudioBox Aesthetics scores for audio.

        Args:
            predictions: Audio signal to evaluate.
            references: Not used for this metric.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the AudioBox Aesthetics scores.
        """
        pred_x = predictions
        fs = metadata.get("sample_rate", 16000) if metadata else 16000

        # Validate input
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")

        pred_x = np.asarray(pred_x)

        output = json.loads(self.model.forward_versa([(pred_x, fs)])[0])
        output = {"audiobox_aesthetics_" + k: v for k, v in output.items()}
        return output

    def get_metadata(self) -> MetricMetadata:
        """Return AudioBox Aesthetics metric metadata."""
        return MetricMetadata(
            name="audiobox_aesthetics",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["audiobox_aesthetics", "numpy"],
            description="AudioBox Aesthetics scores for audio quality assessment using WavLM-based models",
            paper_reference="https://github.com/facebookresearch/audiobox-aesthetics",
            implementation_source="https://github.com/facebookresearch/audiobox-aesthetics",
        )


def register_audiobox_aesthetics_metric(registry):
    """Register AudioBox Aesthetics metric with the registry."""
    metric_metadata = MetricMetadata(
        name="audiobox_aesthetics",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["audiobox_aesthetics", "numpy"],
        description="AudioBox Aesthetics scores for audio quality assessment using WavLM-based models",
        paper_reference="https://github.com/facebookresearch/audiobox-aesthetics",
        implementation_source="https://github.com/facebookresearch/audiobox-aesthetics",
    )
    registry.register(
        AudioBoxAestheticsMetric,
        metric_metadata,
        aliases=["AudioBoxAesthetics", "audiobox_aesthetics"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = AudioBoxAestheticsMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, metadata=metadata)
    print(f"metrics: {score}")
