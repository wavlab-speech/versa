#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

logger = logging.getLogger(__name__)

try:
    import whisper
except ImportError:
    logger.info(
        "Whisper is not properly installed. "
        "Please install following https://github.com/openai/whisper"
    )
    whisper = None

from espnet2.text.cleaner import TextCleaner  # noqa: E402

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def asr_match_setup(
    model_tag="default", beam_size=5, text_cleaner="whisper_basic", use_gpu=True
):
    """Set up ASR matching utilities.

    Args:
        model_tag (str): Whisper model tag.
        beam_size (int): Beam size for decoding.
        text_cleaner (str): Text cleaner type.
        use_gpu (bool): Whether to use GPU.

    Returns:
        dict: WER utilities.
    """
    if model_tag == "default":
        model_tag = "large"
    device = "cuda" if use_gpu else "cpu"
    if whisper is None:
        raise RuntimeError(
            "Whisper WER is used for evaluation while openai-whisper is not installed"
        )
    model = whisper.load_model(model_tag, device=device)
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size}
    return wer_utils


def asr_match_metric(wer_utils, pred_x, gt_x, cache_pred_text=None, fs=16000):
    """Calculate the speaking rate from ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: whisper model ("model"), text cleaner ("cleaner"), and
            beam size ("beam_size")
        pred_x (np.ndarray): test signal (time,)
        gt_x (np.ndarray): ground truth signal (time,)
        cache_pred_text (string): transcription from cache (previous modules)
        fs (int): sampling rate in Hz

    Returns:
        dict: dictionary containing the speaking word rate
    """
    # Process the speech to be evaluated
    if cache_pred_text is not None:
        inf_text = cache_pred_text
    else:
        if fs != TARGET_FS:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            inf_text = wer_utils["model"].transcribe(
                torch.tensor(pred_x).float(), beam_size=wer_utils["beam_size"]
            )["text"]

    # Process the ground truth speech
    if fs != TARGET_FS:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        gt_text = wer_utils["model"].transcribe(
            torch.tensor(gt_x).float(), beam_size=wer_utils["beam_size"]
        )["text"]

    # Calculate the WER
    ref_text = wer_utils["cleaner"](gt_text)
    pred_text = wer_utils["cleaner"](inf_text)

    # Process error rate
    ref_chars = list(ref_text)
    pred_chars = list(pred_text)
    result = {
        "asr_match_delete": 0,
        "asr_match_insert": 0,
        "asr_match_replace": 0,
        "asr_match_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_chars, pred_chars):
        if op == "insert":
            result["asr_match_" + op] += inf_et - inf_st
        else:
            result["asr_match_" + op] += ref_et - ref_st
    total = (
        result["asr_match_delete"]
        + result["asr_match_replace"]
        + result["asr_match_equal"]
    )
    assert total == len(ref_chars), (total, len(ref_chars))
    total = (
        result["asr_match_insert"]
        + result["asr_match_replace"]
        + result["asr_match_equal"]
    )
    assert total == len(pred_chars), (total, len(pred_chars))

    if len(ref_chars) == 0:
        # a fix work around for the asr match when reference is empty
        asr_match_error_rate = 1
    else:
        asr_match_error_rate = (
            result["asr_match_delete"]
            + result["asr_match_insert"]
            + result["asr_match_replace"]
        ) / len(ref_chars)
    return {"asr_match_error_rate": asr_match_error_rate, "whisper_hyp_text": inf_text}


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = asr_match_setup()
    print(f"metrics: {asr_match_metric(wer_utils, a, a, None, 16000)}")
