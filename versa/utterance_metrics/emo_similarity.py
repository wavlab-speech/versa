#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for emotion similarity metrics using EMO2VEC."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Handle optional emo2vec dependency
try:
    import emo2vec_versa
    from emo2vec_versa.emo2vec_class import EMO2VEC

    EMO2VEC_AVAILABLE = True
except ImportError:
    logger.info(
        "emo2vec is not installed. Please install the package via "
        "`tools/install_emo2vec.sh`"
    )
    EMO2VEC = None
    EMO2VEC_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class Emo2vecNotAvailableError(RuntimeError):
    """Exception raised when emo2vec is required but not available."""

    pass


def is_emo2vec_available():
    """
    Check if the emo2vec package is available.

    Returns:
        bool: True if emo2vec is available, False otherwise.
    """
    return EMO2VEC_AVAILABLE


class EmotionMetric(BaseMetric):
    """Emotion similarity metric using EMO2VEC."""

    def _setup(self):
        """Initialize Emotion-specific components."""
        if not EMO2VEC_AVAILABLE:
            raise ImportError(
                "emo2vec_versa not found. Please install from tools/installers"
            )

        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path", None)
        self.use_gpu = self.config.get("use_gpu", False)

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Emotion model: {str(e)}") from e

    def _setup_model(self):
        """Setup the Emotion model."""
        if self.model_path is not None:
            model = EMO2VEC(self.model_path, use_gpu=self.use_gpu)
        else:
            if self.model_tag == "default" or self.model_tag == "base":
                model_path = (
                    Path(os.path.abspath(emo2vec_versa.__file__)).parent
                    / "emotion2vec_base.pt"
                )
            else:
                raise ValueError(f"Unknown model_tag for emo2vec: {self.model_tag}")

            # check if model exists
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = EMO2VEC(checkpoint_dir=str(model_path), use_gpu=self.use_gpu)

        return model

    def compute(
        self, predictions: Any, references: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate emotion similarity between two audio samples.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the emotion similarity score.
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

        # NOTE(jiatong): only work for 16000 Hz
        if fs != 16000:
            gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

        embedding_gen = self.model.extract_feature(pred_x, fs=16000)
        embedding_gt = self.model.extract_feature(gt_x, fs=16000)
        similarity = np.dot(embedding_gen, embedding_gt) / (
            np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
        )

        return {"emotion_similarity": float(similarity)}

    def get_metadata(self) -> MetricMetadata:
        """Return Emotion metric metadata."""
        return MetricMetadata(
            name="emotion",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["emo2vec_versa", "librosa", "numpy"],
            description="Emotion similarity between audio samples using EMO2VEC",
            paper_reference="https://github.com/ddlBoJack/emotion2vec",
            implementation_source="https://github.com/ddlBoJack/emotion2vec",
        )


def register_emotion_metric(registry):
    """Register Emotion metric with the registry."""
    metric_metadata = MetricMetadata(
        name="emotion",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["emo2vec_versa", "librosa", "numpy"],
        description="Emotion similarity between audio samples using EMO2VEC",
        paper_reference="https://github.com/ddlBoJack/emotion2vec",
        implementation_source="https://github.com/ddlBoJack/emotion2vec",
    )
    registry.register(
        EmotionMetric,
        metric_metadata,
        aliases=["Emotion", "emotion", "emo2vec_similarity"],
    )


# Legacy functions for backward compatibility
def emo2vec_setup(model_tag="default", model_path=None, use_gpu=False):
    """Set up EMO2VEC model for emotion embedding extraction (legacy function).

    Args:
        model_tag (str, optional): Model tag. Defaults to "default".
        model_path (str, optional): Path to model weights. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        EMO2VEC: The loaded model.

    Raises:
        ImportError: If emo2vec_versa is not installed.
        ValueError: If model_tag is unknown.
        FileNotFoundError: If model file is not found.
    """
    config = {
        "model_tag": model_tag,
        "model_path": model_path,
        "use_gpu": use_gpu,
    }
    metric = EmotionMetric(config)
    return metric.model


def emo_sim(model, pred_x, gt_x, fs):
    """Calculate emotion similarity between two audio samples (legacy function).

    Args:
        model (EMO2VEC): The loaded EMO2VEC model.
        pred_x (np.ndarray): Predicted audio signal.
        gt_x (np.ndarray): Ground truth audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the emotion similarity score.
    """
    config = {"use_gpu": False}
    metric = EmotionMetric(config)
    metric.model = model
    metadata = {"sample_rate": fs}
    return metric.compute(pred_x, gt_x, metadata=metadata)


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = EmotionMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, b, metadata=metadata)
    print(f"metrics: {score}")
