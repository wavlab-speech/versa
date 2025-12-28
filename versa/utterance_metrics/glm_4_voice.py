#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Speech Properties for Metadata Modeling (Kimi-based)

This module provides functions for extracting various speech properties
from audio using GLM-4-Voice. The properties are organized into the
following categories:

1. Speaker Characteristics
    - glm_4_voice_speaker_count_metric: Number of distinct speakers
    - glm_4_voice_speaker_gender_metric: Gender of speaker(s)
    - glm_4_voice_speaker_age_metric: Age group of speaker(s)
    - glm_4_voice_speech_impairment_metric: Presence and type of speech disorders

2. Voice Properties
    - glm_4_voice_voice_pitch_metric: Overall pitch level
    - glm_4_voice_pitch_range_metric: Variation in intonation
    - glm_4_voice_voice_type_metric: Voice texture characteristics
    - glm_4_voice_speech_volume_level_metric: Loudness of speech

3. Speech Content
    - glm_4_voice_language_metric: Language(s) being spoken
    - glm_4_voice_speech_register_metric: Level of formality in speech
    - glm_4_voice_vocabulary_complexity_metric: Sophistication of word choice
    - glm_4_voice_speech_purpose_metric: Communicative goal of speech

4. Speech Delivery
    - glm_4_voice_speech_emotion_metric: Emotional state conveyed
    - glm_4_voice_speech_clarity_metric: Intelligibility of speech
    - glm_4_voice_speech_rate_metric: Speed of delivery
    - glm_4_voice_speaking_style_metric: Overall presentation manner
    - glm_4_voice_laughter_crying_metric: Presence of emotional vocalizations

5. Interaction Patterns
    - glm_4_voice_overlapping_speech_metric: Degree of simultaneous speech

6. Recording Environment
    - glm_4_voice_speech_background_environment_metric: Setting where recorded
    - glm_4_voice_recording_quality_metric: Technical quality of recording
    - glm_4_voice_channel_type_metric: Equipment used for recording

7. Vocal Evaluation
    - glm_4_voice_singing_technique_metric: Singing Techniques (styles)

Each function follows the same signature pattern:
    glm_4_voice_utils: Dictionary containing model, processor, and conversation
    pred_x: Audio signal as numpy array
    fs: Sampling rate in Hz (default 16000)
    custom_prompt: Optional custom prompt to override default

Each function returns a dictionary with a single key-value pair where
the key is the metric name prefixed with "glm_4_voice_" and the value is the
model's response.
"""
import sys
import copy
import logging
from typing import Dict, Optional, Any
import os
import librosa
import numpy as np
import torch

try:
   from transformers import AutoModel, AutoTokenizer, WhisperFeatureExtractor, BitsAndBytesConfig
except ImportError:
    logging.warning(
        "If transformers is not found with key error, please follow the installation in the latest version of GLM-4-Voice and retry."
    )
    AutoModel, AutoTokenizer, WhisperFeatureExtractor, BitsAndBytesConfig = None, None, None, None

from qwen2_audio import DEFAULT_PROMPTS
# TO-DO: Remove this line when the GLM-4-Voice is installed in the environment
# This is a workaround to ensure the GLM-4-Voice model can be imported correctly
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../.."))
library_path = os.path.join(project_root, "GLM-4-Voice")

if os.path.exists(library_path):
    if library_path not in sys.path:
        sys.path.insert(0, library_path)
else:
    logging.warning(f"GLM-4-Voice repository not found at {library_path}. Please run: git clone --recurse-submodules https://github.com/THUDM/GLM-4-Voice in the project root.")

try:
    from speech_tokenizer.modeling_whisper import WhisperVQEncoder
    from speech_tokenizer.utils import extract_speech_token
except ImportError:
    logging.error("Failed to import GLM-4-Voice modules. Ensure the repository is cloned correctly.")
    WhisperVQEncoder = None
    extract_speech_token = None

from speech_tokenizer.modeling_whisper import WhisperVQEncoder
from speech_tokenizer.utils import extract_speech_token
# from audio_process import AudioStreamProcessor
# Default prompts for different metrics

class Processor:
    """Preprocessor for GLM-4-Voice audio tokenization.
    """

    def __init__(self, whisper_model: WhisperVQEncoder, feature_extractor: WhisperFeatureExtractor, text_processor: AutoTokenizer):
        self.whisper_model = whisper_model
        self.feature_extractor = feature_extractor
        self.text_tokenizer = text_processor
        
def glm_4_voice_model_setup(
    model_tag: str = "THUDM/glm-4-voice-9b",
    tokenizer_tag: str = "THUDM/glm-4-voice-tokenizer",
    dtype: str = "bfloat16",
    start_prompt: str = "The following is a conversation with an AI assistant. The assistant is helpful, honest, and harmless.",
) -> Dict[str, Any]:
    """Set up the GLM-4-Voice model for speech analysis.

    Args:
        model_tag: Model identifier for GLM-4-Voice, defaults to THUDM/glm-4-voice-9b
        start_prompt: Initial system prompt for the model conversation

    Returns:
        Dictionary containing model, processor, and conversation starter
    """
    if model_tag == "default":
        model_tag = "THUDM/glm-4-voice-9b"
        tokenizer_tag = "THUDM/glm-4-voice-tokenizer"
        
    if  AutoModel is None or  AutoTokenizer is None or BitsAndBytesConfig is None or WhisperFeatureExtractor is None:
        raise RuntimeError(
            "GLM-4-Voice is used for evaluation while the transformers library is not installed."
        )
    
    device_id = torch.cuda.current_device() 
    device = torch.device(f"cuda:{device_id}")
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
    ) if dtype == "int4" else None
    
    model = AutoModel.from_pretrained(
            model_tag,
            trust_remote_code=True,
            quantization_config=bnb_config if bnb_config else None,
            device_map={"": device_id}
        ).eval()
    text_processor = AutoTokenizer.from_pretrained(model_tag, trust_remote_code=True)
    
    whisper_model = WhisperVQEncoder.from_pretrained(tokenizer_tag).eval().to(device)
    feature_extractor = WhisperFeatureExtractor.from_pretrained(tokenizer_tag)
    processor = Processor(whisper_model, feature_extractor, text_processor)
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
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


def glm_4_voice_base_metric(
    glm_4_voice_utils: Dict[str, Any],
    pred_x: np.ndarray,
    fs: int = 16000,
    custom_prompt: Optional[str] = None,
    max_length: int = 1000,
) -> str:
    """Calculate the base metric from GLM-4-Voice results.

    Args:
        glm_4_voice_utils: A utility dict for GLM-4-Voice calculation containing:
            'model', 'sampling_params', and 'start_conversation'
        pred_x: Test signal (time,)
        fs: Sampling rate in Hz
        custom_prompt: Custom prompt for the model
        max_length: Maximum length for model generation

    Returns:
        Model's response as a string
    """
    if custom_prompt is None:
        raise ValueError("Custom prompt must be provided for the GLM-4-Voice model.")

    conversation = copy.deepcopy(glm_4_voice_utils["start_conversation"])
    processor = glm_4_voice_utils["processor"] # The preprocessor is for audio tokenization
    model = glm_4_voice_utils["model"]
    sampling_params = glm_4_voice_utils["sampling_params"]
    temperature = float(sampling_params.get("temperature", 1.0))
    top_p = float(sampling_params.get("top_p", 1.0))

    # Audio preprocess
    audio = torch.from_numpy(pred_x).unsqueeze(0).to(model.device)
    audio_tokens = extract_speech_token(
        processor.whisper_model, processor.feature_extractor, [(audio, fs)]
    )[0]
    
    audio_tokens = "".join([f"<|audio_{x}|>" for x in audio_tokens])
    audio_tokens = "<|begin_of_audio|>" + audio_tokens + "<|end_of_audio|>"
    audio = audio_tokens

    # Text preprocess
    # Need to check the format later
    
    input_text = conversation[0]["content"] + "\n" + custom_prompt 
    # print(f"Input text: {input_text}")
    
    user_input = f"<|system|>\n{input_text}\n<|user|>\n{audio}<|assistant|>streaming_transcription\n"
    
    inputs = processor.text_tokenizer([user_input], return_tensors="pt")
    inputs = inputs.to(model.device)
    
    
    response = model.generate(
        **inputs,
        max_new_tokens=max_length,
        temperature=temperature,
        top_p=top_p,
    )[0]
    text_tokens = []
    audio_offset = processor.text_tokenizer.convert_tokens_to_ids('<|audio_0|>')
    for token_id in response:
        # The model should not generate audio tokens 
        if token_id >= audio_offset:
            continue
        else:
            text_tokens.append(token_id)
    responses = processor.text_tokenizer.decode(text_tokens, ignore_special_tokens=True)
    response = responses.split("<|assistant|>")[-1].split("\n")[-1].replace('<|user|>', '').strip()
    return response


def create_metric_fn(metric_name: str) -> callable:
    """Factory function to create metric functions.

    Args:
        metric_name: Name of the metric to create a function for

    Returns:
        Function that calculates the specified metric
    """

    def metric_fn(
        glm_4_voice_utils: Dict[str, Any],
        pred_x: np.ndarray,
        fs: int = 16000,
        custom_prompt: Optional[str] = None,
    ) -> Dict[str, str]:
        """Calculate the specified metric from GLM-4-Voice results.

        Args:
            glm_4_voice_utils: A utility dict for GLM-4-Voice calculation
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

        response = glm_4_voice_base_metric(glm_4_voice_utils, pred_x, fs, custom_prompt)
        return {f"glm_4_voice_{metric_name}": response}

    return metric_fn


# Create metric functions for all categories
# 1. Speaker Characteristics
glm_4_voice_speaker_count_metric = create_metric_fn("speaker_count")
glm_4_voice_speaker_gender_metric = create_metric_fn("speaker_gender")
glm_4_voice_speaker_age_metric = create_metric_fn("speaker_age")
glm_4_voice_speech_impairment_metric = create_metric_fn("speech_impairment")

# 2. Voice Properties
glm_4_voice_voice_pitch_metric = create_metric_fn("voice_pitch")
glm_4_voice_pitch_range_metric = create_metric_fn("pitch_range")
glm_4_voice_voice_type_metric = create_metric_fn("voice_type")
glm_4_voice_speech_volume_level_metric = create_metric_fn("speech_volume_level")

# 3. Speech Content
glm_4_voice_language_metric = create_metric_fn("language")
glm_4_voice_speech_register_metric = create_metric_fn("speech_register")
glm_4_voice_vocabulary_complexity_metric = create_metric_fn("vocabulary_complexity")
glm_4_voice_speech_purpose_metric = create_metric_fn("speech_purpose")

# 4. Speech Delivery
glm_4_voice_speech_emotion_metric = create_metric_fn("speech_emotion")
glm_4_voice_speech_clarity_metric = create_metric_fn("speech_clarity")
glm_4_voice_speech_rate_metric = create_metric_fn("speech_rate")
glm_4_voice_speaking_style_metric = create_metric_fn("speaking_style")
glm_4_voice_laughter_crying_metric = create_metric_fn("laughter_crying")

# 5. Interaction Patterns
glm_4_voice_overlapping_speech_metric = create_metric_fn("overlapping_speech")

# 6. Recording Environment
glm_4_voice_speech_background_environment_metric = create_metric_fn(
    "speech_background_environment"
)
glm_4_voice_recording_quality_metric = create_metric_fn("recording_quality")
glm_4_voice_channel_type_metric = create_metric_fn("channel_type")

# 7. Vocal Evaluation
glm_4_voice_singing_technique_metric = create_metric_fn("singing_technique")

if __name__ == "__main__":
    a = np.random.random(16000)
    glm_4_voice_utils = glm_4_voice_model_setup()
    all_metrics = [
        glm_4_voice_speaker_count_metric,
        glm_4_voice_speaker_gender_metric,
        glm_4_voice_speaker_age_metric,
        glm_4_voice_speech_impairment_metric,
        glm_4_voice_voice_pitch_metric,
        glm_4_voice_pitch_range_metric,
        glm_4_voice_voice_type_metric,
        glm_4_voice_speech_volume_level_metric,
        glm_4_voice_language_metric,
        glm_4_voice_speech_register_metric,
        glm_4_voice_vocabulary_complexity_metric,
        glm_4_voice_speech_purpose_metric,
        glm_4_voice_speech_emotion_metric,
        glm_4_voice_speech_clarity_metric,
        glm_4_voice_speech_rate_metric,
        glm_4_voice_speaking_style_metric,
        glm_4_voice_laughter_crying_metric,
        glm_4_voice_overlapping_speech_metric,
        glm_4_voice_speech_background_environment_metric,
        glm_4_voice_recording_quality_metric,
        glm_4_voice_channel_type_metric,
        glm_4_voice_singing_technique_metric,
    ]

    for fn in all_metrics:
        print("metrics: {}".format(fn(glm_4_voice_utils, a, 16000)))
