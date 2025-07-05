#!/usr/bin/env python3

# Copyright 2025 BoHao Su
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for dimensional emotion prediction metrics using w2v2-how-to."""

import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Handle optional transformers dependency
try:
    from transformers import Wav2Vec2Processor
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning(
        "transformers is not properly installed. "
        "Please install transformers and retry"
    )
    Wav2Vec2Processor = None
    Wav2Vec2Model = None
    Wav2Vec2PreTrainedModel = None
    TRANSFORMERS_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class TransformersNotAvailableError(RuntimeError):
    """Exception raised when transformers is required but not available."""

    pass


def is_transformers_available():
    """
    Check if the transformers package is available.

    Returns:
        bool: True if transformers is available, False otherwise.
    """
    return TRANSFORMERS_AVAILABLE


class RegressionHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(
        self,
        input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)

        return hidden_states, logits


class EmoVadMetric(BaseMetric):
    """Dimensional emotion prediction metric using w2v2-how-to."""

    def _setup(self):
        """Initialize EmoVad-specific components."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers is not properly installed. "
                "Please install transformers and retry"
            )

        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path", None)
        self.model_config = self.config.get("model_config", None)
        self.use_gpu = self.config.get("use_gpu", False)

        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"

        try:
            self.model, self.processor = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize EmoVad model: {str(e)}") from e

    def _setup_model(self):
        """Setup the EmoVad model."""
        if self.model_path is not None and self.model_config is not None:
            model = EmotionModel.from_pretrained(
                pretrained_model_name_or_path=self.model_path, config=self.model_config
            ).to(self.device)
        else:
            if self.model_tag == "default":
                model_tag = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
            else:
                model_tag = self.model_tag
            model = EmotionModel.from_pretrained(model_tag).to(self.device)

        processor = Wav2Vec2Processor.from_pretrained(
            "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
        )

        return model, processor

    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate dimensional emotion (arousal, dominance, valence) of input audio samples.

        Args:
            predictions: Audio signal to evaluate.
            references: Not used for this metric.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the dimensional emotion predictions.
        """
        pred_x = predictions
        fs = metadata.get("sample_rate", 16000) if metadata else 16000

        # Validate input
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")

        pred_x = np.asarray(pred_x)

        # NOTE(jiatong): only work for 16000 Hz
        if fs != 16000:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

        pred_x = self.processor(pred_x, sampling_rate=16000)
        pred_x = pred_x["input_values"][0]
        pred_x = pred_x.reshape(1, -1)
        pred_x = torch.from_numpy(pred_x).to(self.device)

        with torch.no_grad():
            avd_emo = self.model(pred_x)[1].squeeze(0).cpu().numpy()

        arousal, dominance, valence = avd_emo
        arousal = arousal.item()
        dominance = dominance.item()
        valence = valence.item()

        return {
            "arousal_emo_vad": arousal,
            "valence_emo_vad": valence,
            "dominance_emo_vad": dominance,
        }

    def get_metadata(self) -> MetricMetadata:
        """Return EmoVad metric metadata."""
        return MetricMetadata(
            name="emo_vad",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["transformers", "torch", "librosa", "numpy"],
            description="Dimensional emotion prediction (arousal, valence, dominance) using w2v2-how-to",
            paper_reference="https://github.com/audeering/w2v2-how-to",
            implementation_source="https://github.com/audeering/w2v2-how-to",
        )


def register_emo_vad_metric(registry):
    """Register EmoVad metric with the registry."""
    metric_metadata = MetricMetadata(
        name="emo_vad",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["transformers", "torch", "librosa", "numpy"],
        description="Dimensional emotion prediction (arousal, valence, dominance) using w2v2-how-to",
        paper_reference="https://github.com/audeering/w2v2-how-to",
        implementation_source="https://github.com/audeering/w2v2-how-to",
    )
    registry.register(EmoVadMetric, metric_metadata, aliases=["EmoVad", "emo_vad"])


if __name__ == "__main__":
    a = np.random.random(16000)

    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = EmoVadMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, metadata=metadata)
    print(f"metrics: {score}")
