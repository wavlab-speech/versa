#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for discrete speech metrics evaluation."""

import logging
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)

# Handle optional discrete_speech_metrics dependency
try:
    from discrete_speech_metrics import SpeechBERTScore, SpeechBLEU, SpeechTokenDistance

    DISCRETE_SPEECH_AVAILABLE = True
except ImportError:
    logger.warning(
        "discrete_speech_metrics is not properly installed. "
        "Please install discrete_speech_metrics and retry"
    )
    SpeechBERTScore = None
    SpeechBLEU = None
    SpeechTokenDistance = None
    DISCRETE_SPEECH_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class DiscreteSpeechNotAvailableError(RuntimeError):
    """Exception raised when discrete_speech_metrics is required but not available."""

    pass


def is_discrete_speech_available():
    """
    Check if the discrete_speech_metrics package is available.

    Returns:
        bool: True if discrete_speech_metrics is available, False otherwise.
    """
    return DISCRETE_SPEECH_AVAILABLE


class DiscreteSpeechMetric(BaseMetric):
    """Discrete speech metrics for audio evaluation."""

    def _setup(self):
        """Initialize Discrete Speech-specific components."""
        if not DISCRETE_SPEECH_AVAILABLE:
            raise ImportError(
                "discrete_speech_metrics is not properly installed. "
                "Please install discrete_speech_metrics and retry"
            )

        self.use_gpu = self.config.get("use_gpu", False)
        self.sample_rate = self.config.get("sample_rate", 16000)

        # NOTE(jiatong) existing discrete speech metrics only works for 16khz
        # We keep the paper best setting. To use other settings, please conduct the
        # test on your own.

        try:
            self.speech_bert = SpeechBERTScore(
                sr=self.sample_rate,
                model_type="wavlm-large",
                layer=14,
                use_gpu=self.use_gpu,
            )
            self.speech_bleu = SpeechBLEU(
                sr=self.sample_rate,
                model_type="hubert-base",
                vocab=200,
                layer=11,
                n_ngram=2,
                remove_repetition=True,
                use_gpu=self.use_gpu,
            )
            self.speech_token_distance = SpeechTokenDistance(
                sr=self.sample_rate,
                model_type="hubert-base",
                vocab=200,
                layer=6,
                distance_type="jaro-winkler",
                remove_repetition=False,
                use_gpu=self.use_gpu,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize discrete speech metrics: {str(e)}"
            ) from e

    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate discrete speech metrics.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the metric scores.
        """
        pred_x = predictions
        gt_x = references
        fs = (
            metadata.get("sample_rate", self.sample_rate)
            if metadata
            else self.sample_rate
        )

        # Validate inputs
        if pred_x is None or gt_x is None:
            raise ValueError("Both predicted and ground truth signals must be provided")

        pred_x = np.asarray(pred_x)
        gt_x = np.asarray(gt_x)

        scores = {}

        if fs != self.sample_rate:
            gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=self.sample_rate)
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=self.sample_rate)

        # Calculate SpeechBERT score
        try:
            score, _, _ = self.speech_bert.score(gt_x, pred_x)
            scores["speech_bert"] = score
        except Exception as e:
            logger.warning(f"Could not calculate SpeechBERT score: {e}")
            scores["speech_bert"] = 0.0

        # Calculate SpeechBLEU score
        try:
            score = self.speech_bleu.score(gt_x, pred_x)
            scores["speech_bleu"] = score
        except Exception as e:
            logger.warning(f"Could not calculate SpeechBLEU score: {e}")
            scores["speech_bleu"] = 0.0

        # Calculate SpeechTokenDistance score
        try:
            score = self.speech_token_distance.score(gt_x, pred_x)
            scores["speech_token_distance"] = score
        except Exception as e:
            logger.warning(f"Could not calculate SpeechTokenDistance score: {e}")
            scores["speech_token_distance"] = 0.0

        return scores

    def get_metadata(self) -> MetricMetadata:
        """Return Discrete Speech metric metadata."""
        return MetricMetadata(
            name="discrete_speech",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["discrete_speech_metrics", "librosa", "numpy"],
            description="Discrete speech metrics including SpeechBERT, SpeechBLEU, and SpeechTokenDistance for audio evaluation",
            paper_reference="https://github.com/ftshijt/discrete_speech_metrics",
            implementation_source="https://github.com/ftshijt/discrete_speech_metrics",
        )


def register_discrete_speech_metric(registry):
    """Register Discrete Speech metric with the registry."""
    metric_metadata = MetricMetadata(
        name="discrete_speech",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["discrete_speech_metrics", "librosa", "numpy"],
        description="Discrete speech metrics including SpeechBERT, SpeechBLEU, and SpeechTokenDistance for audio evaluation",
        paper_reference="https://github.com/ftshijt/discrete_speech_metrics",
        implementation_source="https://github.com/ftshijt/discrete_speech_metrics",
    )
    registry.register(
        DiscreteSpeechMetric,
        metric_metadata,
        aliases=["DiscreteSpeech", "discrete_speech"],
    )


# Legacy functions for backward compatibility
def discrete_speech_setup(use_gpu=False):
    """Set up discrete speech metrics (legacy function).

    Args:
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        dict: Dictionary containing the initialized metrics.
    """
    config = {"use_gpu": use_gpu}
    metric = DiscreteSpeechMetric(config)
    return {
        "speech_bert": metric.speech_bert,
        "speech_bleu": metric.speech_bleu,
        "speech_token_distance": metric.speech_token_distance,
    }


def discrete_speech_metric(discrete_speech_predictors, pred_x, gt_x, fs):
    """Calculate discrete speech metrics (legacy function).

    Args:
        discrete_speech_predictors (dict): Dictionary of speech metrics.
        pred_x (np.ndarray): Predicted audio signal.
        gt_x (np.ndarray): Ground truth audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the metric scores.
    """
    config = {"use_gpu": False}  # Default config
    metric = DiscreteSpeechMetric(config)
    metric.speech_bert = discrete_speech_predictors["speech_bert"]
    metric.speech_bleu = discrete_speech_predictors["speech_bleu"]
    metric.speech_token_distance = discrete_speech_predictors["speech_token_distance"]
    metadata = {"sample_rate": fs}
    return metric.compute(pred_x, gt_x, metadata=metadata)


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = DiscreteSpeechMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, b, metadata=metadata)
    print(f"metrics: {score}")
