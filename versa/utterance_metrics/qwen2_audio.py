#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import logging
import librosa
import copy

try:
    from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
except ImportError:
    logging.warning(
        "If Qwen2Audio is not found with key error, please install the latest version of transformers and retry."
    )
    Qwen2AudioForConditionalGeneration, AutoProcessor = None, None


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
    model = Qwen2AudioForConditionalGeneration.from_pretrained(model_tag, device_map="auto")


    start_conversation = [
        {"role": "system", "content": start_prompt},
    ]
    return {
        "model": model,
        "processor": processoor,
        "start_conversation": start_conversation
    }


def qwen2_base_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the base metric from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model
    Returns:
        ret (dict): ditionary containing the speaker age prediction
    """
    conversation = copy.deepcopy(qwen_utils["start_conversation"])
    processor = qwen_utils["processor"]
    model = qwen_utils["model"]
    if custom_prompt is None:
        raise ValueError("Custom prompt must be provided for the qwen2-audio model.")
    conversation.append({"role": "user", "content": [
        {"type": "audio", "audio_url": None},
        {"type": "text", "text": custom_prompt}
    ]})
    
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audio = [librosa.resample(pred_x, orig_sr=fs, target_sr=processor.feature_extractor.sampling_rate)]

    inputs = processor(text=text, audios=audio, return_tensors="pt", padding=True)
    inputs.input_ids = inputs.input_ids.to(qwen_utils["model"].device)

    generate_ids = model.generate(**inputs, max_length=256)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
    del conversation
    return response


############################################
# Speech Metrics
############################################

def qwen2_speaker_age_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    
    """Calculate the speaker age from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's age prediction
    Returns:
        ret (dict): ditionary containing the speaker age prediction
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


def qwen2_speech_emotion_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    
    """Calculate the speaker age from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech emotion prediction
    Returns:
        ret (dict): ditionary containing the speech emotion prediction
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


def qwen2_speaker_count_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speaker count from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speaker count prediction
    Returns:
        ret (dict): ditionary containing the speaker count prediction
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


def qwen2_language_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the language from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's language prediction
    Returns:
        ret (dict): ditionary containing the language prediction
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


def qwen2_speech_clarity_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech clarity from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech clarity prediction
    Returns:
        ret (dict): ditionary containing the speech clarity prediction
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
        ret (dict): ditionary containing the speech rate prediction
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


def qwen2_speech_background_environment_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech background environment from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's background environment prediction
    Returns:
        ret (dict): ditionary containing the background environment prediction
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

def qwen2_speech_purpose_metric(qwen_utils, pred_x, fs=16000, custom_prompt=None):
    """Calculate the speech purpose from Qwen2Audio results.

    Args:
        qwen_utils (dict): a utility dict for Qwen2Audio calculation.
            including: Qwen2Audio model ("model"), processor ("processor"), and start conversation ("start_conversation")
        pred_x (np.ndarray): test signal (time,)
        fs (int): sampling rate in Hz
        custom_prompt (string): custom prompt for the model's speech purpose prediction
    Returns:
        ret (dict): ditionary containing the speech purpose prediction
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
        ret (dict): ditionary containing the recording quality prediction
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

if __name__ == "__main__":
    a = np.random.random(16000)
    qwen_utils = qwen2_model_setup()
    # print("metrics: {}".format(qwen2_speaker_age_metric(qwen_utils, a, 16000)))
    print("metrics: {}".format(qwen2_speech_emotion_metric(qwen_utils, a, 16000)))
