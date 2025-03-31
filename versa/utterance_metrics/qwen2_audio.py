#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Speech Properties for Metadata Modeling

This module provides functions for extracting various speech properties
from audio using Qwen2-Audio. The properties are organized into the
following categories:

1. Speaker Characteristics
    - qwen2_speaker_count_metric: Number of distinct speakers
    - qwen2_speaker_gender_metric: Gender of speaker(s)
    - qwen2_speaker_age_metric: Age group of speaker(s)
    - qwen2_speech_impairment_metric: Presence and type of speech disorders

2. Voice Properties
    - qwen2_voice_pitch_metric: Overall pitch level
    - qwen2_pitch_range_metric: Variation in intonation
    - qwen2_voice_type_metric: Voice texture characteristics
    - qwen2_speech_volume_level_metric: Loudness of speech

3. Speech Content
    - qwen2_language_metric: Language(s) being spoken
    - qwen2_speech_register_metric: Level of formality in speech
    - qwen2_vocabulary_complexity_metric: Sophistication of word choice
    - qwen2_speech_purpose_metric: Communicative goal of speech

4. Speech Delivery
    - qwen2_speech_emotion_metric: Emotional state conveyed
    - qwen2_speech_clarity_metric: Intelligibility of speech
    - qwen2_speech_rate_metric: Speed of delivery
    - qwen2_speaking_style_metric: Overall presentation manner
    - qwen2_laughter_crying_metric: Presence of emotional vocalizations

5. Interaction Patterns
    - qwen2_overlapping_speech_metric: Degree of simultaneous speech

6. Recording Environment
    - qwen2_speech_background_environment_metric: Setting where recorded
    - qwen2_recording_quality_metric: Technical quality of recording
    - qwen2_channel_type_metric: Equipment used for recording

Each function follows the same signature pattern:
    qwen_utils: Dictionary containing model, processor, and conversation
    pred_x: Audio signal as numpy array
    fs: Sampling rate in Hz (default 16000)
    custom_prompt: Optional custom prompt to override default

Each function returns a dictionary with a single key-value pair where
the key is the metric name prefixed with "qwen_" and the value is the
model's response.
"""

import copy
import logging

import librosa
import numpy as np

try:
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
except ImportError:
    logging.warning(
        "If Qwen2Audio is not found with key error, please install the latest version of transformers and retry."
    )
    Qwen2AudioForConditionalGeneration, AutoProcessor = None, None


DEFAULT_PROMPT = {
    "speaker_age": "What is the age of the speaker? Please answer in 'child', '20s', '30s', '40s', '50s', '60s', '70s', 'senior'.",
    "speech_emotion": "What is the emotion of the speech? Please answer in 'neutral state', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'frustration', 'excited', 'other' only.",
    "speaker_count": "How many speakers are there in the audio? Please answer with a number.",
    "language": "What language is being spoken? Please answer with the name of the language.",
    "speaker_gender": "What is the gender of the speaekr. Please answer with 'male' or 'female' only.",
}


def qwen2_model_setup(
    model_tag="Qwen/Qwen2-Audio-7B-Instruct",
    start_prompt="The following is a conversation with an AI assistant. The assistant is helpful, honest, and harmless.",
):
    if model_tag == "default":
        model_tag = "Qwen/Qwen2-Audio-7B-Instruct"
    if Qwen2AudioForConditionalGeneration is None or AutoProcessor is None:
        raise RuntimeError(
            "Qwen2Audio is used for evaluation while transformers is not installed (could be a version issue)."
        )
    processoor = AutoProcessor.from_pretrained(model_tag)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_tag, device_map="auto"
    )

    start_conversation = [
        {"role": "system", "content": start_prompt},
    ]
    return {
        "model": model,
        "processor": processoor,
        "start_conversation": start_conversation,
    }


def qwen2_base_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None, max_length=500):
    """Calculate the base metric from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model
    Returns:
        ret (dict): dictionary containing the speaker age prediction
    """
    conversation = copy.deepcopy(qwen_utils["start_conversation"])
    processor = qwen_utils["processor"]
    model = qwen_utils["model"]
    if custom_prompt is None:
        raise ValueError("Custom prompt must be provided for the qwen2-audio model.")
    conversation.append(
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": None},
                {"type": "text", "text": custom_prompt},
            ],
        }
    )

    text = processor.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    audio = [
        librosa.resample(
            pred_x, orig_sr=fs, target_sr=processor.feature_extractor.sampling_rate
        )
    ]

    inputs = processor(text=text, audios=audio, return_tensors="pt", padding=True)
    # inputs.input_ids = inputs.input_ids.to(qwen_utils["model"].device)
    for key in inputs.keys():
        inputs[key] = inputs[key].to(qwen_utils["model"].device)

    generate_ids = model.generate(**inputs, max_length=500)
    generate_ids = generate_ids[:, inputs["input_ids"].size(1) :]
    response = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )[0]
    return response


################################################
# Speech Metrics Part I: Speaker Characteristics
################################################


def qwen2_speaker_count_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speaker count from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speaker count prediction
    Returns:
        ret (dict): dictionary containing the speaker count prediction
    """
    if custom_prompt is None:
        custom_prompt = """Analyze the audio and determine the number of distinct speakers present.
Provide your answer as a single number between 1-10.
Examples:
- For a monologue: 1
- For an interview with host and guest: 2 
- For a panel discussion with a moderator and three panelists: 4"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speaker_count": response}


def qwen2_speaker_gender_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Estimate the speaker gender from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)"
        fs (int): sampling rate in Hz"
        custom_prompt (string): custom prompt for model's speaker gender prediction
    Returns:
        ret (dict): ditionary containing
    """

    if custom_prompt is None:
        custom_prompt = """Identify the perceived gender of the speaker(s).
If multiple speakers, list each speaker with their perceived gender.
Choose from:
- Male
- Female
- Non-binary/unclear
- Multiple speakers with mixed genders"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speaker_gender": response}


def qwen2_speaker_age_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speaker age from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's age prediction
    Returns:
        ret (dict): dictionary containing the speaker age prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify the age group of the speaker.
Choose exactly one label from the following categories:
- Child: under 13 years
- Teen: 13-19 years
- Young adult: 20-35 years
- Middle-aged adult: 36-55 years
- Senior: over 55 years"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speaker_age": response}


def qwen2_speech_impairment_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech impairment presence from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech impairment prediction
    Returns:
        ret (dict): dictionary containing the speech impairment prediction
    """
    if custom_prompt is None:
        custom_prompt = """Assess whether there are any noticeable speech impairments or disorders in the speaker's voice.
Choose exactly one category:
- No apparent impairment: typical speech patterns
- Stuttering/disfluency: repetitions, blocks, or prolongations of sounds
- Articulation disorder: difficulty with specific speech sounds
- Voice disorder: abnormal pitch, loudness, or quality
- Fluency disorder: atypical rhythm, rate, or flow of speech
- Foreign accent: non-native pronunciation patterns
- Dysarthria: slurred or unclear speech from muscle weakness
- Apraxia: difficulty with motor planning for speech
- Other impairment: speech pattern that suggests a different disorder"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_impairment": response}


################################################
# Speech Metrics Part II: Voice Properties
################################################


def qwen2_pitch_range_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the pitch range from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's pitch range prediction
    Returns:
        ret (dict): dictionary containing the pitch range prediction
    """
    if custom_prompt is None:
        custom_prompt = """Assess the pitch variation/intonation range in the speaker's voice.
Choose exactly one category:
- Wide range: highly expressive with significant variation between high and low tones
- Moderate range: normal variation in pitch during speech
- Narrow range: minimal pitch variation, relatively monotone delivery
- Monotone: almost no pitch variation"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_pitch_range": response}


def qwen2_voice_pitch_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the voice pitch from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's voice pitch prediction
    Returns:
        ret (dict): dictionary containing the voice pitch prediction
    """
    if custom_prompt is None:
        custom_prompt = """Analyze the voice pitch/tone of the speaker.
Choose exactly one category from the following:
- Very high: significantly higher than average for their perceived gender
- High: noticeably above average pitch 
- Medium: average pitch range
- Low: noticeably below average pitch
- Very low: significantly lower than average for their perceived gender"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_voice_pitch": response}


def qwen2_voice_type_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the voice type from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's voice quality prediction
    Returns:
        ret (dict): dictionary containing the voice quality prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify the dominant voice quality-related characteristic of the speaker.
Choose exactly one category:
- Clear: clean vocal production without noticeable texture issues
- Breathy: voice has audible breath sounds, less vocal cord closure
- Creaky/vocal fry: low-frequency rattling sound, especially at ends of phrases
- Hoarse: rough, raspy quality indicating vocal strain
- Nasal: voice resonates primarily through the nose
- Pressed/tense: strained quality from excessive vocal cord pressure
- Resonant: rich, vibrant voice with good projection
- Whispered: intentionally quiet with minimal vocal cord vibration
- Tremulous: shaky or quivery voice quality"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_voice_type": response}


def qwen2_speech_volume_level_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the volume/loudness level from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's volume level prediction
    Returns:
        ret (dict): dictionary containing the volume level prediction
    """
    if custom_prompt is None:
        custom_prompt = """Assess the overall volume or loudness level of the speaker.
Choose exactly one category:
- Very quiet: barely audible, whispering or very soft-spoken
- Quiet: below average volume, soft-spoken
- Moderate: normal conversational volume
- Loud: above average volume, projecting voice
- Very loud: shouting or extremely high volume
- Variable: significant changes in volume throughout the recording"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_volume_level": response}


###################################################
# Speech Metrics Part III: Speech Content Analysis
###################################################


def qwen2_language_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the language from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's language prediction
    Returns:
        ret (dict): dictionary containing the language prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify all languages spoken in the audio.
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
- Other (specify if possible)"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_language": response}


def qwen2_speech_register_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech register from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech register prediction
    Returns:
        ret (dict): dictionary containing the speech register prediction
    """
    if custom_prompt is None:
        custom_prompt = """Determine the speech register used by the speaker.
Choose exactly one category:
- Formal register: careful pronunciation, complex grammar, specialized vocabulary
- Standard register: proper grammar and pronunciation for professional or educational contexts
- Consultative register: mixture of formal and casual for everyday professional interactions
- Casual register: relaxed grammar, contractions, colloquialisms for friends/family
- Intimate register: highly familiar language used with close relations
- Technical register: specialized terminology for a specific field or profession
- Slang register: highly informal with group-specific vocabulary"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_register": response}


def qwen2_vocabulary_complexity_metric(
    qwen_utils, pred_x, fs=16000, custom_prompt=None
):
    """Calculate the vocabulary complexity from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's vocabulary complexity prediction
    Returns:
        ret (dict): dictionary containing the vocabulary complexity prediction
    """
    if custom_prompt is None:
        custom_prompt = """Evaluate the vocabulary complexity level in the speech.
Choose exactly one category:
- Basic: simple, everyday vocabulary, mostly high-frequency words
- General: standard vocabulary for common topics, occasional advanced words
- Advanced: sophisticated vocabulary with specific terminology
- Technical: specialized/domain-specific terminology
- Academic: scholarly vocabulary with abstract concepts"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_vocabulary_complexity": response}


def qwen2_speech_purpose_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech purpose from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech purpose prediction
    Returns:
        ret (dict): dictionary containing the speech purpose prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify the primary purpose of the speech.
Choose one category:
- Informative: primarily explains or educates
- Persuasive: attempts to convince or change opinions
- Entertainment: primarily aims to amuse or entertain
- Narrative: tells a story or relates events
- Conversational: casual exchange of information
- Instructional: provides specific directions or guidance
- Emotional expression: primarily conveys feelings or emotional state"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_purpose": response}


###################################################
# Speech Metrics Part IV: Speech Delivery Analysis
###################################################


def qwen2_speech_emotion_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speaker age from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech emotion prediction
    Returns:
        ret (dict): dictionary containing the speech emotion prediction
    """
    if custom_prompt is None:
        # IEMOCAP emotion classification prompt
        custom_prompt = """Identify the dominant emotion expressed in this speech.
Choose exactly one label from the following categories:
- Neutral: even-toned, matter-of-fact delivery with minimal emotional expression
- Happy: upbeat, positive, enthusiastic tone
- Sad: downcast, melancholic, somber tone
- Angry: irritated, frustrated, hostile tone  
- Fearful: anxious, worried, frightened tone
- Surprised: astonished, shocked tone
- Disgusted: repulsed, revolted tone
- Other: other emotion that cannot be classified by above classes"""
        response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_emotion": response}


def qwen2_speech_clarity_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech clarity from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech clarity prediction
    Returns:
        ret (dict): dictionary containing the speech clarity prediction
    """
    if custom_prompt is None:
        custom_prompt = """Rate the overall clarity and intelligibility of the speech.
Choose one category:
- High clarity: perfectly intelligible, professional quality
- Medium clarity: generally understandable with occasional unclear segments
- Low clarity: difficult to understand, frequent unclear segments
- Very low clarity: mostly unintelligible"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_clarity": response}


def qwen2_speech_rate_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech rate from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech rate prediction
    Returns:
        ret (dict): dictionary containing the speech rate prediction
    """
    if custom_prompt is None:
        custom_prompt = """Assess the rate of speech in the audio.
Choose one category:
- Very slow: deliberate, significantly slower than average speech
- Slow: relaxed pace, slower than conversational speech
- Medium: average conversational pace
- Fast: quicker than average conversational speech
- Very fast: rapid delivery, difficult to follow"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_rate": response}


def qwen2_speaking_style_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speaking style from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speaking style prediction
    Returns:
        ret (dict): dictionary containing the speaking style prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify the predominant speaking style of the speaker.
Choose exactly one category:
- Formal: structured, proper, adherence to linguistic conventions
- Professional: clear, efficient communication focused on task/topic
- Casual/conversational: relaxed, everyday speech
- Animated/enthusiastic: highly energetic, expressive speech
- Deliberate: careful, measured delivery
- Dramatic: theatrical, performance-oriented speech
- Authoritative: commanding, confident tone
- Hesitant: uncertain, tentative speech with pauses"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speaking_style": response}


def qwen2_laughter_crying_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the presence of laughter/crying from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's laughter/crying prediction
    Returns:
        ret (dict): dictionary containing the laughter/crying prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify if there is laughter, crying, or other emotional vocalizations in the audio.
Choose exactly one category:
- No laughter or crying: speech only
- Contains laughter: audible laughter is present
- Contains crying: audible crying or sobbing is present
- Contains both: both laughter and crying are present
- Contains other emotional sounds: sighs, gasps, etc.
- Contains multiple emotional vocalizations: combination of various emotional sounds"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_laughter_crying": response}


def qwen2_speech_background_environment_metric(
    qwen_utils, pred_x, fs=16000, custom_prompt=None
):
    """Calculate the speech background environment from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's background environment prediction
    Returns:
        ret (dict): dictionary containing the background environment prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify the dominant background environment or setting.
Choose one category:
- Quiet indoor: minimal background noise, likely studio environment
- Noisy indoor: indoor setting with noticeable background sounds (cafe, office)
- Outdoor urban: city sounds, traffic
- Outdoor natural: nature sounds, birds, wind, water
- Event/crowd: audience sounds, applause, crowd noise
- Music background: music playing behind speech
- Multiple environments: changes throughout recording"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speech_background_environment": response}


###################################################
# Speech Metrics Part V: Interaction Patterns
###################################################


def qwen2_overlapping_speech_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the overlapping speech presence from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's overlapping speech prediction
    Returns:
        ret (dict): dictionary containing the overlapping speech prediction
    """
    if custom_prompt is None:
        custom_prompt = """Determine if there is overlapping speech in the audio (people talking simultaneously).
Choose exactly one category:
- No overlap: clean turn-taking with no simultaneous speech
- Minimal overlap: occasional brief instances of overlapping speech
- Moderate overlap: noticeable instances where speakers talk over each other
- Significant overlap: frequent overlapping speech, making it difficult to follow
- Constant overlap: multiple speakers talking simultaneously throughout most of the audio"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_overlapping_speech": response}


############################################
# Audio Metrics
############################################


############################################
# General Metrics
############################################


def qwen2_recording_quality_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the recording quality from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's recording quality prediction
    Returns:
        ret (dict): dictionary containing the recording quality prediction
    """
    if custom_prompt is None:
        custom_prompt = """Assess the technical quality of the audio recording.
Choose one category:
- Professional: studio-quality, broadcast standard
- Good: clear recording with minimal issues
- Fair: noticeable recording artifacts but generally clear
- Poor: significant recording issues affecting comprehension
- Very poor: severe technical problems making content difficult to understand"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_recording_quality": response}


def qwen2_channel_type_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the audio channel type from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's channel type prediction
    Returns:
        ret (dict): dictionary containing the channel type prediction
    """
    if custom_prompt is None:
        custom_prompt = """Identify the likely recording channel or device type used to record this audio.
Choose exactly one category:
- Professional microphone: high-quality, full-range audio
- Consumer microphone: decent quality but less clarity than professional
- Smartphone: typical mobile phone recording quality
- Telephone/VoIP: limited frequency range, compression artifacts
- Webcam/computer mic: variable quality, often with computer fan noise
- Headset microphone: close to mouth, may have breathing sounds
- Distant microphone: recorded from a distance, may have room echo
- Radio/broadcast: compressed audio with limited frequency range
- Surveillance/hidden mic: typically lower quality with background noise"""
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_channel_type": response}


if __name__ == "__main__":
    a = np.random.random(16000)
    qwen_utils = qwen2_model_setup()
    # print("metrics: {}".format(qwen2_speaker_age_metric(qwen_utils, a, 16000)))
    print("metrics: {}".format(qwen2_speech_emotion_metric(qwen_utils, a, 16000)))
