#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch

try:
    import whisper
except ImportError:
    logger.info(
        "Whisper is not properly installed. Please install following https://github.com/openai/whisper"
    )
    whisper = None

from espnet2.text.cleaner import TextCleaner


TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def speaking_rate_model_setup(
    model_tag="default", beam_size=5, text_cleaner="whisper_basic", use_gpu=True
):
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


def speaking_rate_metric(wer_utils, pred_x, cache_text=None, fs=16000, use_char=False):
    """Calculate the speaking rate from ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: whisper model ("model"), text cleaner ("textcleaner"), and
            beam size ("beam size")
        pred_x (np.ndarray): test signal (time,)
        cache_text (string): transcription from cache (previous modules)
        fs (int): sampling rate in Hz
        use_char (bool): whether to use character-level speaking rate
    Returns:
        ret (dict): ditionary containing the speaking word rate
    """
    if cache_text is not None:
        inf_text = cache_text
    else:
        if fs != TARGET_FS:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            inf_text = wer_utils["model"].transcribe(
                torch.tensor(pred_x).float(), beam_size=wer_utils["beam_size"]
            )["text"]

    if use_char:
        length = len(inf_text)
    else:
        length = len(inf_text.split())
    return {
        "speaking_rate": length / (len(pred_x) / fs),
        "whisper_hyp_text": inf_text,
    }


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = speaking_rate_model_setup()
    print("metrics: {}".format(speaking_rate_metric(wer_utils, a, None, 16000)))
