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
from typing import Dict, Optional, Any

import librosa
import numpy as np
import torch

try:
    from kimia_infer.api.kimia import KimiAudio
    from kimia_infer.models.tokenizer.glm4_tokenizer import Glm4Tokenizer
except ImportError:
    logging.warning(
        "If KimiAudio is not found with key error, please install the latest version of Kimi-Audio and retry."
    )
    KimiAudio, Glm4Tokenizer = None, None


# Default prompts for different metrics
DEFAULT_PROMPTS = {
    # Speaker Characteristics
    "speaker_count": """Analyze the audio and determine the number of distinct speakers present.
Provide your answer as a single number between 1-10.
Examples:
- For a monologue: 1
- For an interview with host and guest: 2 
- For a panel discussion with a moderator and three panelists: 4""",
    "speaker_gender": """Identify the perceived gender of the speaker(s).
If multiple speakers, list each speaker with their perceived gender.
Choose from:
- Male
- Female
- Non-binary/unclear
- Multiple speakers with mixed genders""",
    "speaker_age": """Identify the age group of the speaker.
Choose exactly one label from the following categories:
- Child: under 13 years
- Teen: 13-19 years
- Young adult: 20-35 years
- Middle-aged adult: 36-55 years
- Senior: over 55 years""",
    "speech_impairment": """Assess whether there are any noticeable speech impairments or disorders in the speaker's voice.
Choose exactly one category:
- No apparent impairment: typical speech patterns
- Stuttering/disfluency: repetitions, blocks, or prolongations of sounds
- Articulation disorder: difficulty with specific speech sounds
- Voice disorder: abnormal pitch, loudness, or quality
- Fluency disorder: atypical rhythm, rate, or flow of speech
- Foreign accent: non-native pronunciation patterns
- Dysarthria: slurred or unclear speech from muscle weakness
- Apraxia: difficulty with motor planning for speech
- Other impairment: speech pattern that suggests a different disorder""",
    # Voice Properties
    "voice_pitch": """Analyze the voice pitch/tone of the speaker.
Choose exactly one category from the following:
- Very high: significantly higher than average for their perceived gender
- High: noticeably above average pitch 
- Medium: average pitch range
- Low: noticeably below average pitch
- Very low: significantly lower than average for their perceived gender""",
    "pitch_range": """Assess the pitch variation/intonation range in the speaker's voice.
Choose exactly one category:
- Wide range: highly expressive with significant variation between high and low tones
- Moderate range: normal variation in pitch during speech
- Narrow range: minimal pitch variation, relatively monotone delivery
- Monotone: almost no pitch variation""",
    "voice_type": """Identify the dominant voice quality-related characteristic of the speaker.
Choose exactly one category:
- Clear: clean vocal production without noticeable texture issues
- Breathy: voice has audible breath sounds, less vocal cord closure
- Creaky/vocal fry: low-frequency rattling sound, especially at ends of phrases
- Hoarse: rough, raspy quality indicating vocal strain
- Nasal: voice resonates primarily through the nose
- Pressed/tense: strained quality from excessive vocal cord pressure
- Resonant: rich, vibrant voice with good projection
- Whispered: intentionally quiet with minimal vocal cord vibration
- Tremulous: shaky or quivery voice quality""",
    "speech_volume_level": """Assess the overall volume or loudness level of the speaker.
Choose exactly one category:
- Very quiet: barely audible, whispering or very soft-spoken
- Quiet: below average volume, soft-spoken
- Moderate: normal conversational volume
- Loud: above average volume, projecting voice
- Very loud: shouting or extremely high volume
- Variable: significant changes in volume throughout the recording""",
    # Speech Content
    "language": """Identify all languages spoken in the audio.
List languages using their English names.
Choose from common languages:
- English
- Spanish
- Mandarin Chinese
- Hindi
- Arabic
- French
- Russian
- Portuguese
- German
- Japanese
- Other (specify if possible)""",
    "speech_register": """Determine the speech register used by the speaker.
Choose exactly one category:
- Formal register: careful pronunciation, complex grammar, specialized vocabulary
- Standard register: proper grammar and pronunciation for professional or educational contexts
- Consultative register: mixture of formal and casual for everyday professional interactions
- Casual register: relaxed grammar, contractions, colloquialisms for friends/family
- Intimate register: highly familiar language used with close relations
- Technical register: specialized terminology for a specific field or profession
- Slang register: highly informal with group-specific vocabulary""",
    "vocabulary_complexity": """Evaluate the vocabulary complexity level in the speech.
Choose exactly one category:
- Basic: simple, everyday vocabulary, mostly high-frequency words
- General: standard vocabulary for common topics, occasional advanced words
- Advanced: sophisticated vocabulary with specific terminology
- Technical: specialized/domain-specific terminology
- Academic: scholarly vocabulary with abstract concepts""",
    "speech_purpose": """Identify the primary purpose of the speech.
Choose one category:
- Informative: primarily explains or educates
- Persuasive: attempts to convince or change opinions
- Entertainment: primarily aims to amuse or entertain
- Narrative: tells a story or relates events
- Conversational: casual exchange of information
- Instructional: provides specific directions or guidance
- Emotional expression: primarily conveys feelings or emotional state""",
    # Speech Delivery
    "speech_emotion": """Identify the dominant emotion expressed in this speech.
Choose exactly one label from the following categories:
- Neutral: even-toned, matter-of-fact delivery with minimal emotional expression
- Happy: upbeat, positive, enthusiastic tone
- Sad: downcast, melancholic, somber tone
- Angry: irritated, frustrated, hostile tone  
- Fearful: anxious, worried, frightened tone
- Surprised: astonished, shocked tone
- Disgusted: repulsed, revolted tone
- Other: other emotion that cannot be classified by above classes""",
    "speech_clarity": """Rate the overall clarity and intelligibility of the speech.
Choose one category:
- High clarity: perfectly intelligible, professional quality
- Medium clarity: generally understandable with occasional unclear segments
- Low clarity: difficult to understand, frequent unclear segments
- Very low clarity: mostly unintelligible""",
    "speech_rate": """Assess the rate of speech in the audio.
Choose one category:
- Very slow: deliberate, significantly slower than average speech
- Slow: relaxed pace, slower than conversational speech
- Medium: average conversational pace
- Fast: quicker than average conversational speech
- Very fast: rapid delivery, difficult to follow""",
    "speaking_style": """Identify the predominant speaking style of the speaker.
Choose exactly one category:
- Formal: structured, proper, adherence to linguistic conventions
- Professional: clear, efficient communication focused on task/topic
- Casual/conversational: relaxed, everyday speech
- Animated/enthusiastic: highly energetic, expressive speech
- Deliberate: careful, measured delivery
- Dramatic: theatrical, performance-oriented speech
- Authoritative: commanding, confident tone
- Hesitant: uncertain, tentative speech with pauses""",
    "laughter_crying": """Identify if there is laughter, crying, or other emotional vocalizations in the audio.
Choose exactly one category:
- No laughter or crying: speech only
- Contains laughter: audible laughter is present
- Contains crying: audible crying or sobbing is present
- Contains both: both laughter and crying are present
- Contains other emotional sounds: sighs, gasps, etc.
- Contains multiple emotional vocalizations: combination of various emotional sounds""",
    # Interaction Patterns
    "overlapping_speech": """Determine if there is overlapping speech in the audio (people talking simultaneously).
Choose exactly one category:
- No overlap: clean turn-taking with no simultaneous speech
- Minimal overlap: occasional brief instances of overlapping speech
- Moderate overlap: noticeable instances where speakers talk over each other
- Significant overlap: frequent overlapping speech, making it difficult to follow
- Constant overlap: multiple speakers talking simultaneously throughout most of the audio""",
    # Recording Environment
    "speech_background_environment": """Identify the dominant background environment or setting.
Choose one category:
- Quiet indoor: minimal background noise, likely studio environment
- Noisy indoor: indoor setting with noticeable background sounds (cafe, office)
- Outdoor urban: city sounds, traffic
- Outdoor natural: nature sounds, birds, wind, water
- Event/crowd: audience sounds, applause, crowd noise
- Music background: music playing behind speech
- Multiple environments: changes throughout recording""",
    "recording_quality": """Assess the technical quality of the audio recording.
Choose one category:
- Professional: studio-quality, broadcast standard
- Good: clear recording with minimal issues
- Fair: noticeable recording artifacts but generally clear
- Poor: significant recording issues affecting comprehension
- Very poor: severe technical problems making content difficult to understand""",
    "channel_type": """Identify the likely recording channel or device type used to record this audio.
Choose exactly one category:
- Professional microphone: high-quality, full-range audio
- Consumer microphone: decent quality but less clarity than professional
- Smartphone: typical mobile phone recording quality
- Telephone/VoIP: limited frequency range, compression artifacts
- Webcam/computer mic: variable quality, often with computer fan noise
- Headset microphone: close to mouth, may have breathing sounds
- Distant microphone: recorded from a distance, may have room echo
- Radio/broadcast: compressed audio with limited frequency range
- Surveillance/hidden mic: typically lower quality with background noise""",
    # Vocal Evaluation
    "singing_technique": """You are an expert in vocal performance and singing technique.
Given the following audio clip of a singing voice, your task is to identify the predominant singing style used.
Choose one of the following seven styles based on the vocal characteristics:

Breathy: Light, airy voice with noticeable breathiness.
Falsetto: High-pitched, flute-like sound, especially for male voices.
Mixed Voice: A blend of chest and head voice, balanced resonance.
Pharyngeal: Focused, twangy tone with forward placement in the pharynx.
Glissando: Smooth, sliding transitions between notes.
Vibrato: Regular, pulsating pitch variation while sustaining a note.
Control: A neutral, well-supported tone without stylistic effects.

Carefully listen to the tone quality, pitch control, resonance, and transitions in the audio.
Then, output only the predicted singing style from the list above.
""",
}


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
    max_length: int = 1000,
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
    preprocessor = kimi_utils["processor"] # The preprocessor is for audio tokenization
    model = kimi_utils["model"]

    audio = torch.from_numpy(librosa.resample(
        pred_x, orig_sr=fs, target_sr=16000, # Kimi-Audio uses 16kHz as default sampling rate
    )).unsqueeze(0).to(torch.float32).to(model.alm.device)
    
    audio_tokens = preprocessor.audio_tokenizer.tokenize(speech=audio)
    audio_tokens = audio_tokens + preprocessor.kimia_token_offset
    audio_tokens = audio_tokens.squeeze(0).cpu().numpy().tolist()
    
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
                "content": audio,
                "audio_tokens": audio_tokens,
            }
        ]
        
    )

    _, response = model.generate(conversation, **sampling_params, max_new_tokens=max_length, output_type="text")
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
    # print("metrics: {}".format(kimi_speech_emotion_metric(kimi_utils, a, 16000)))
