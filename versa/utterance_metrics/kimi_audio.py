#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Speech Properties for Metadata Modeling (Kimi-based)

This module provides functions for extracting various speech properties
from audio using Kimi-Audio. The properties are organized into the
following categories:

1. Speaker Characteristics
    - kimi_speaker_count_metric: Number of distinct speakers
    - kimi_speaker_gender_metric: Gender of speaker(s)
    - kimi_speaker_age_metric: Age group of speaker(s)
    - kimi_speech_impairment_metric: Presence and type of speech disorders

2. Voice Properties
    - kimi_voice_pitch_metric: Overall pitch level
    - kimi_pitch_range_metric: Variation in intonation
    - kimi_voice_type_metric: Voice texture characteristics
    - kimi_speech_volume_level_metric: Loudness of speech

3. Speech Content
    - kimi_language_metric: Language(s) being spoken
    - kimi_speech_register_metric: Level of formality in speech
    - kimi_vocabulary_complexity_metric: Sophistication of word choice
    - kimi_speech_purpose_metric: Communicative goal of speech

4. Speech Delivery
    - kimi_speech_emotion_metric: Emotional state conveyed
    - kimi_speech_clarity_metric: Intelligibility of speech
    - kimi_speech_rate_metric: Speed of delivery
    - kimi_speaking_style_metric: Overall presentation manner
    - kimi_laughter_crying_metric: Presence of emotional vocalizations

5. Interaction Patterns
    - kimi_overlapping_speech_metric: Degree of simultaneous speech

6. Recording Environment
    - kimi_speech_background_environment_metric: Setting where recorded
    - kimi_recording_quality_metric: Technical quality of recording
    - kimi_channel_type_metric: Equipment used for recording

7. Vocal Evaluation
    - kimi_singing_technique_metric: Singing Techniques (styles)

Each function follows the same signature pattern:
    kimi_utils: Dictionary containing model, processor, and conversation
    pred_x: Audio signal as numpy array
    fs: Sampling rate in Hz (default 16000)
    custom_prompt: Optional custom prompt to override default

Each function returns a dictionary with a single key-value pair where
the key is the metric name prefixed with "kimi_" and the value is the
model's response.
"""

import copy
import logging
import os
from typing import Dict, Optional, Any
import tempfile
import librosa
import numpy as np
import soundfile as sf
import torch

try:
    from kimia_infer.api.kimia import KimiAudio
    from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
except ImportError:
    logging.warning(
        "If KimiAudio is not found with key error, please install the latest version of Kimi-Audio and retry."
    )
    KimiAudio, Glm4Tokenizer = None, None

from qwen2_audio import DEFAULT_PROMPTS
# Default prompts for different metrics

def kimi_model_setup(
    model_tag: str = "MoonshotAI/Kimi-Audio-7B",
    start_prompt: str = "The following is a conversation with an AI assistant. The assistant is helpful, honest, and harmless.",
) -> Dict[str, Any]:
    """Set up the Kimi-Audio model for speech analysis.

    Args:
        model_tag: Model identifier for Kimi-Audio, defaults to MoonshotAI/Kimi-Audio-7B
        start_prompt: Initial system prompt for the model conversation

    Returns:
        Dictionary containing model, processor, and conversation starter
    """
    if model_tag == "default":
        model_tag = "MoonshotAI/Kimi-Audio-7B"
    if KimiAudio is None or Glm4Tokenizer is None:
        raise RuntimeError(
            "Kimi-Audio is used for evaluation while the Kimi library is not installed."
        )
    model = KimiAudio(model_path=model_tag, load_detokenizer=True)
    processor = model.prompt_manager
    sampling_params = {
    "audio_temperature": 0.8,
    "audio_top_k": 10,
    "text_temperature": 0.0,
    "text_top_k": 5,
    "audio_repetition_penalty": 1.0,
    "audio_repetition_window_size": 64,
    "text_repetition_penalty": 1.0,
    "text_repetition_window_size": 16,
}
    start_conversation = [
        {"role": "assistant", "message_type": "text", "content": start_prompt},
    ]
    return {
        "model": model,
        "processor": processor,
        "sampling_params": sampling_params,
        "start_conversation": start_conversation,
    }


def kimi_base_metric(
    kimi_utils: Dict[str, Any],
    pred_x: np.ndarray,
    fs: int = 16000,
    custom_prompt: Optional[str] = None,
    max_length: int = 100,
) -> str:
    """Calculate the base metric from Kimi-Audio results.

    Args:
        kimi_utils: A utility dict for Kimi-Audio calculation containing:
            'model', 'sampling_params', and 'start_conversation'
        pred_x: Test signal (time,)
        fs: Sampling rate in Hz
        custom_prompt: Custom prompt for the model
        max_length: Maximum length for model generation

    Returns:
        Model's response as a string
    """
    if custom_prompt is None:
        raise ValueError("Custom prompt must be provided for the Kimi-Audio model.")

    conversation = copy.deepcopy(kimi_utils["start_conversation"])
    sampling_params = kimi_utils["sampling_params"]
    model = kimi_utils["model"]

    # Resample audio to 16kHz
    y = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    # Create a temporary file to satisfy the library's requirement for a file path
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name

    try:
        sf.write(temp_path, y, 16000)
        conversation.extend(
            [
                {
                    "role": "user",
                    "message_type": "text", 
                    "content": custom_prompt
                },
                {
                    "role": "user",
                    "message_type": "audio",
                    "content": temp_path,
                }
            ]
            
        )
        _, response = model.generate(conversation, **sampling_params, max_new_tokens=max_length, output_type="text")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return response


def create_metric_fn(metric_name: str) -> callable:
    """Factory function to create metric functions.

    Args:
        metric_name: Name of the metric to create a function for

    Returns:
        Function that calculates the specified metric
    """

    def metric_fn(
        kimi_utils: Dict[str, Any],
        pred_x: np.ndarray,
        fs: int = 16000,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """Calculate the specified metric from Kimi-Audio results.

        Args:
            kimi_utils: A utility dict for Kimi-Audio calculation
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

        response = kimi_base_metric(kimi_utils, pred_x, fs, custom_prompt)
        return {f"kimi_{metric_name}": response}

    return metric_fn


# Create metric functions for all categories
# 1. Speaker Characteristics
kimi_speaker_count_metric = create_metric_fn("speaker_count")
kimi_speaker_gender_metric = create_metric_fn("speaker_gender")
kimi_speaker_age_metric = create_metric_fn("speaker_age")
kimi_speech_impairment_metric = create_metric_fn("speech_impairment")

# 2. Voice Properties
kimi_voice_pitch_metric = create_metric_fn("voice_pitch")
kimi_pitch_range_metric = create_metric_fn("pitch_range")
kimi_voice_type_metric = create_metric_fn("voice_type")
kimi_speech_volume_level_metric = create_metric_fn("speech_volume_level")

# 3. Speech Content
kimi_language_metric = create_metric_fn("language")
kimi_speech_register_metric = create_metric_fn("speech_register")
kimi_vocabulary_complexity_metric = create_metric_fn("vocabulary_complexity")
kimi_speech_purpose_metric = create_metric_fn("speech_purpose")

# 4. Speech Delivery
kimi_speech_emotion_metric = create_metric_fn("speech_emotion")
kimi_speech_clarity_metric = create_metric_fn("speech_clarity")
kimi_speech_rate_metric = create_metric_fn("speech_rate")
kimi_speaking_style_metric = create_metric_fn("speaking_style")
kimi_laughter_crying_metric = create_metric_fn("laughter_crying")

# 5. Interaction Patterns
kimi_overlapping_speech_metric = create_metric_fn("overlapping_speech")

# 6. Recording Environment
kimi_speech_background_environment_metric = create_metric_fn(
    "speech_background_environment"
)
kimi_recording_quality_metric = create_metric_fn("recording_quality")
kimi_channel_type_metric = create_metric_fn("channel_type")

# 7. Vocal Evaluation
kimi_singing_technique_metric = create_metric_fn("singing_technique")

if __name__ == "__main__":
    a = np.random.random(16000)
    
    kimi_utils = kimi_model_setup()
    all_metrics = [
        kimi_speaker_count_metric,
        kimi_speaker_gender_metric,
        kimi_speaker_age_metric,
        kimi_speech_impairment_metric,
        kimi_voice_pitch_metric,
        kimi_pitch_range_metric,
        kimi_voice_type_metric,
        kimi_speech_volume_level_metric,
        kimi_language_metric,
        kimi_speech_register_metric,
        kimi_vocabulary_complexity_metric,
        kimi_speech_purpose_metric,
        kimi_speech_emotion_metric,
        kimi_speech_clarity_metric,
        kimi_speech_rate_metric,
        kimi_speaking_style_metric,
        kimi_laughter_crying_metric,
        kimi_overlapping_speech_metric,
        kimi_speech_background_environment_metric,
        kimi_recording_quality_metric,
        kimi_channel_type_metric,
        kimi_singing_technique_metric,
    ]

    for fn in all_metrics:
        print("metrics: {}".format(fn(kimi_utils, a, 16000)))
