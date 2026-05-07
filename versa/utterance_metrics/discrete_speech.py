#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for discrete speech metrics evaluation."""

import importlib.util
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

DISCRETE_SPEECH_AVAILABLE = (
    importlib.util.find_spec("discrete_speech_metrics") is not None
)
if not DISCRETE_SPEECH_AVAILABLE:
    logger.warning(
        "discrete_speech_metrics is not properly installed. "
        "Please install discrete_speech_metrics and retry"
    )

SpeechBERTScore = None
SpeechBLEU = None
SpeechTokenDistance = None


def _configure_huggingface_cache(cache_dir):
    cache_path = Path(cache_dir).resolve()
    cache_path.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(cache_path))
    os.environ.setdefault("HF_HUB_CACHE", str(cache_path / "hub"))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(cache_path / "transformers"))


def _load_discrete_speech_classes(cache_dir):
    global SpeechBERTScore, SpeechBLEU, SpeechTokenDistance

    if not DISCRETE_SPEECH_AVAILABLE:
        raise ImportError(
            "discrete_speech_metrics is not properly installed. "
            "Please install discrete_speech_metrics and retry"
        )
    _configure_huggingface_cache(cache_dir)
    if SpeechBERTScore is None or SpeechBLEU is None or SpeechTokenDistance is None:
        from discrete_speech_metrics import (
            SpeechBERTScore as _SpeechBERTScore,
            SpeechBLEU as _SpeechBLEU,
            SpeechTokenDistance as _SpeechTokenDistance,
        )

        SpeechBERTScore = _SpeechBERTScore
        SpeechBLEU = _SpeechBLEU
        SpeechTokenDistance = _SpeechTokenDistance
    return SpeechBERTScore, SpeechBLEU, SpeechTokenDistance


from versa.audio_utils import resample_audio
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
        self.cache_dir = self.config.get("cache_dir", "versa_cache/huggingface")
        speech_bert_cls, speech_bleu_cls, speech_token_distance_cls = (
            _load_discrete_speech_classes(self.cache_dir)
        )

        # NOTE(jiatong) existing discrete speech metrics only works for 16khz
        # We keep the paper best setting. To use other settings, please conduct the
        # test on your own.

        try:
            self.speech_bert = speech_bert_cls(
                sr=self.sample_rate,
                model_type="wavlm-large",
                layer=14,
                use_gpu=self.use_gpu,
            )
            self.speech_bleu = speech_bleu_cls(
                sr=self.sample_rate,
                model_type="hubert-base",
                vocab=200,
                layer=11,
                n_ngram=2,
                remove_repetition=True,
                use_gpu=self.use_gpu,
            )
            self.speech_token_distance = speech_token_distance_cls(
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
            gt_x = resample_audio(gt_x, fs, self.sample_rate)
            pred_x = resample_audio(pred_x, fs, self.sample_rate)

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


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = DiscreteSpeechMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, b, metadata=metadata)
    print(f"metrics: {score}")
