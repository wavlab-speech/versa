#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# flake8: noqa: E501

"""
Speech Properties for Metadata Modeling

This module provides functions for extracting various speech properties
from audio using Qwen2-Audio. The properties are organized into the
following categories:

1. Speaker Characteristics
    - qwen_omni_speaker_count_metric: Number of distinct speakers
    - qwen_omni_speaker_gender_metric: Gender of speaker(s)
    - qwen_omni_speaker_age_metric: Age group of speaker(s)
    - qwen_omni_speech_impairment_metric: Presence and type of speech disorders

2. Voice Properties
    - qwen_omni_voice_pitch_metric: Overall pitch level
    - qwen_omni_pitch_range_metric: Variation in intonation
    - qwen_omni_voice_type_metric: Voice texture characteristics
    - qwen_omni_speech_volume_level_metric: Loudness of speech

3. Speech Content
    - qwen_omni_language_metric: Language(s) being spoken
    - qwen_omni_speech_register_metric: Level of formality in speech
    - qwen_omni_vocabulary_complexity_metric: Sophistication of word choice
    - qwen_omni_speech_purpose_metric: Communicative goal of speech

4. Speech Delivery
    - qwen_omni_speech_emotion_metric: Emotional state conveyed
    - qwen_omni_speech_clarity_metric: Intelligibility of speech
    - qwen_omni_speech_rate_metric: Speed of delivery
    - qwen_omni_speaking_style_metric: Overall presentation manner
    - qwen_omni_laughter_crying_metric: Presence of emotional vocalizations

5. Interaction Patterns
    - qwen_omni_overlapping_speech_metric: Degree of simultaneous speech

6. Recording Environment
    - qwen_omni_speech_background_environment_metric: Setting where recorded
    - qwen_omni_recording_quality_metric: Technical quality of recording
    - qwen_omni_channel_type_metric: Equipment used for recording

Each function follows the same signature pattern:
    qwen_utils: Dictionary containing model, processor, and conversation
    pred_x: Audio signal as numpy array
    fs: Sampling rate in Hz (default 16000)
    custom_prompt: Optional custom prompt to override default

7. Vocal Evaluation
    - qwen_omni_singing_technique_metric: Singing Techniques (styles)

Each function returns a dictionary with a single key-value pair where
the key is the metric name prefixed with "qwen_" and the value is the
model's response.
"""

import copy
import logging
from typing import Dict, Optional, Any

import numpy as np
import torch

from versa.audio_utils import resample_audio

try:
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
except ImportError:
    logging.warning(
        "If qwen2_5_omni is not found with key error, please install the latest version of transformers and retry."
    )
    Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor = None, None

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType
from versa.utterance_metrics.qwen2_audio import DEFAULT_PROMPTS


def qwen_omni_model_setup(
    model_tag: str = "Qwen/Qwen2-Audio-7B-Instruct",
    start_prompt: str = "The following is a conversation with an AI assistant. The assistant is helpful, honest, and harmless.",
    use_gpu: bool = True,
    device_map: Optional[str] = None,
) -> Dict[str, Any]:
    """Set up the Qwen2-Audio model for speech analysis.

    Args:
        model_tag: Model identifier for Qwen2-Audio, defaults to Qwen2-Audio-7B-Instruct
        start_prompt: Initial system prompt for the model conversation

    Returns:
        Dictionary containing model, processor, and conversation starter
    """
    if model_tag == "default":
        model_tag = "Qwen/Qwen2.5-Omni-7B"
    if Qwen2_5OmniForConditionalGeneration is None or Qwen2_5OmniProcessor is None:
        raise RuntimeError(
            "qwen2_5_omni is used for evaluation while transformers is not installed (could be a version issue)."
        )
    target_device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    device_map = device_map or target_device
    processor = Qwen2_5OmniProcessor.from_pretrained(model_tag)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_tag,
        torch_dtype="auto",
        device_map=device_map,
        # attn_implementation="flash_attention_2", NOTE(jiatong): to add
    )
    if device_map in {None, "cpu", "cuda"}:
        model.to(target_device)
    start_conversation = [
        {"role": "system", "content": [{"type": "text", "text": start_prompt}]}
    ]
    return {
        "model": model,
        "processor": processor,
        "start_conversation": start_conversation,
    }


def qwen_omni_base_metric(
    qwen_utils: Dict[str, Any],
    pred_x: np.ndarray,
    fs: int = 16000,
    custom_prompt: Optional[str] = None,
    max_length: int = 500,
) -> str:
    """Calculate the base metric from Qwen2.5-Omni results.

    Args:
        qwen_utils: A utility dict for Qwen2.5-Omni calculation.
            including: Qwen2.5-Omni model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x: Test signal (time,)
        fs: Sampling rate in Hz
        custom_prompt: Custom prompt for the model
        max_length: Maximum length for model generation

    Returns:
        Model's response as a string
    """
    if custom_prompt is None:
        raise ValueError("Custom prompt must be provided for the qwen_omni model.")

    conversation = copy.deepcopy(qwen_utils["start_conversation"])
    processor = qwen_utils["processor"]
    model = qwen_utils["model"]

    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": None},
                {"type": "text", "text": custom_prompt},
            ],
        },
    )

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audio = [resample_audio(pred_x, fs, processor.feature_extractor.sampling_rate)]

    inputs = processor(text=text, audio=audio, return_tensors="pt", padding=True)
    inputs = inputs.to(model.device).to(model.dtype)

    output = model.generate(
        **inputs,
        use_audio_in_video=True,
        return_audio=False,
        thinker_max_new_tokens=max_length,
        thinker_do_sample=False,
    )

    text = processor.batch_decode(
        output, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return text


def create_metric_fn(metric_name: str) -> callable:
    """Factory function to create metric functions.

    Args:
        metric_name: Name of the metric to create a function for

    Returns:
        Function that calculates the specified metric
    """

    def metric_fn(
        qwen_utils: Dict[str, Any],
        pred_x: np.ndarray,
        fs: int = 16000,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """Calculate the specified metric from Qwen2.5-Omni results.

        Args:
            qwen_utils: A utility dict for Qwen2.5-Omni calculation
            pred_x: Test signal (time,)
            fs: Sampling rate in Hz
            custom_prompt: Custom prompt for the model

        Returns:
            Dictionary containing the metric result
        """
        if custom_prompt is None:
            custom_prompt = DEFAULT_PROMPTS.get(metric_name)
            if custom_prompt is None:
                raise ValueError(f"No default prompt found for metric: {metric_name}")

        response = qwen_omni_base_metric(qwen_utils, pred_x, fs, custom_prompt)
        return {f"qwen_omni_{metric_name}": response}

    return metric_fn


# Create metric functions for all categories
# 1. Speaker Characteristics
qwen_omni_speaker_count_metric = create_metric_fn("speaker_count")
qwen_omni_speaker_gender_metric = create_metric_fn("speaker_gender")
qwen_omni_speaker_age_metric = create_metric_fn("speaker_age")
qwen_omni_speech_impairment_metric = create_metric_fn("speech_impairment")

# 2. Voice Properties
qwen_omni_voice_pitch_metric = create_metric_fn("voice_pitch")
qwen_omni_pitch_range_metric = create_metric_fn("pitch_range")
qwen_omni_voice_type_metric = create_metric_fn("voice_type")
qwen_omni_speech_volume_level_metric = create_metric_fn("speech_volume_level")

# 3. Speech Content
qwen_omni_language_metric = create_metric_fn("language")
qwen_omni_speech_register_metric = create_metric_fn("speech_register")
qwen_omni_vocabulary_complexity_metric = create_metric_fn("vocabulary_complexity")
qwen_omni_speech_purpose_metric = create_metric_fn("speech_purpose")

# 4. Speech Delivery
qwen_omni_speech_emotion_metric = create_metric_fn("speech_emotion")
qwen_omni_speech_clarity_metric = create_metric_fn("speech_clarity")
qwen_omni_speech_rate_metric = create_metric_fn("speech_rate")
qwen_omni_speaking_style_metric = create_metric_fn("speaking_style")
qwen_omni_laughter_crying_metric = create_metric_fn("laughter_crying")

# 5. Interaction Patterns
qwen_omni_overlapping_speech_metric = create_metric_fn("overlapping_speech")

# 6. Recording Environment
qwen_omni_speech_background_environment_metric = create_metric_fn(
    "speech_background_environment"
)
qwen_omni_recording_quality_metric = create_metric_fn("recording_quality")
qwen_omni_channel_type_metric = create_metric_fn("channel_type")


qwen_omni_singing_technique_metric = create_metric_fn("singing_technique")


class QwenOmniMetric(BaseMetric):
    """Speech property extraction with Qwen2.5-Omni."""

    metric_name = None

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.start_prompt = self.config.get(
            "start_prompt",
            (
                "The following is a conversation with an AI assistant. "
                "The assistant is helpful, honest, and harmless."
            ),
        )
        self.metric_name = self.config.get("metric_name", self.metric_name)
        if self.metric_name is None:
            raise ValueError("metric_name must be provided")
        self.prompt = self.config.get("prompt")
        self.max_length = self.config.get("max_length", 500)
        self.use_gpu = self.config.get("use_gpu", True)
        self.device_map = self.config.get("device_map")
        self.qwen_utils = qwen_omni_model_setup(
            model_tag=self.model_tag,
            start_prompt=self.start_prompt,
            use_gpu=self.use_gpu,
            device_map=self.device_map,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        prompt = self.prompt or DEFAULT_PROMPTS.get(self.metric_name)
        response = qwen_omni_base_metric(
            self.qwen_utils,
            np.asarray(predictions),
            fs=fs,
            custom_prompt=prompt,
            max_length=self.max_length,
        )
        return {f"qwen_omni_{self.metric_name}": response}

    def get_metadata(self):
        return _qwen_omni_metadata(self.registry_name())

    @classmethod
    def registry_name(cls):
        return f"qwen_omni_{cls.metric_name}" if cls.metric_name else "qwen_omni"


def _qwen_omni_metadata(name):
    return MetricMetadata(
        name=name,
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.STRING,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["transformers", "librosa", "numpy", "torch"],
        description="Speech property extraction with Qwen2.5-Omni",
        paper_reference="https://arxiv.org/abs/2503.20215",
        implementation_source="https://github.com/QwenLM/Qwen2.5-Omni",
    )


def _make_qwen_omni_metric_class(metric_name):
    class _SpecificQwenOmniMetric(QwenOmniMetric):
        pass

    _SpecificQwenOmniMetric.metric_name = metric_name
    class_name = "".join(part.title() for part in metric_name.split("_"))
    _SpecificQwenOmniMetric.__name__ = f"QwenOmni{class_name}Metric"
    return _SpecificQwenOmniMetric


QWEN_OMNI_METRIC_CLASSES = {
    metric_name: _make_qwen_omni_metric_class(metric_name)
    for metric_name in DEFAULT_PROMPTS.keys()
}


def register_qwen_omni_metric(registry):
    """Register Qwen2.5-Omni speech property metrics with the registry."""
    for metric_name, metric_class in QWEN_OMNI_METRIC_CLASSES.items():
        registry_name = f"qwen_omni_{metric_name}"
        registry.register(
            metric_class,
            _qwen_omni_metadata(registry_name),
            aliases=[f"qwen_omni_{metric_name}_metric"],
        )


if __name__ == "__main__":
    a = np.random.random(16000)
    qwen_utils = qwen_omni_model_setup()
    # print("metrics: {}".format(qwen_omni_speaker_age_metric(qwen_utils, a, 16000)))
    print("metrics: {}".format(qwen_omni_speech_emotion_metric(qwen_utils, a, 16000)))
