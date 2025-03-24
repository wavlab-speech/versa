#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import logging
import librosa

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
    conversation = qwen_utils["start_conversation"]
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
        custom_prompt = "What is the age of the speaker? Please answer in 'child', '20s', '30s', '40s', '50s', '60s', '70s', 'senior'."
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
        custom_prompt = "What is the emotion of the speech? Please answer in 'neutral state', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'frustration', 'excited', 'other' only."
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_Speech_emotion": response}


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
        custom_prompt = "How many speakers are there in the audio? Please answer with a number."
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
        custom_prompt = "What language is being spoken? Please answer with the name of the language."
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
        custom_prompt = "What is the gender of the speaekr. Please answer with 'male' or 'female' only."
    response = qwen2_base_metric(qwen_utils, pred_x, fs, custom_prompt)
    return {"qwen_speaker_gender": response}


############################################
# Audio Metrics
############################################


if __name__ == "__main__":
    a = np.random.random(16000)
    qwen_utils = qwen2_model_setup()
    # print("metrics: {}".format(qwen2_speaker_age_metric(qwen_utils, a, 16000)))
    print("metrics: {}".format(qwen2_speech_emotion_metric(qwen_utils, a, 16000)))
