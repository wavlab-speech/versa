#!/usr/bin/env python3

# Copyright 2025 Haoran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

try:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
except ImportError:
    logging.warning(
        "Faster-whisper is not properly installed. Please install following https://github.com/systran/faster-whisper"
    )
    WhisperModel = None

from espnet2.text.cleaner import TextCleaner

TARGET_FS = 16000


def faster_whisper_wer_setup(
    model_tag="default", beam_size=5, batch_size=1, compute_type="float32" ,text_cleaner="whisper_basic", use_gpu=True
):
    if model_tag == "default":
        model_tag = "large-v3"
    device = "cuda" if use_gpu else "cpu"
    if WhisperModel is None:
        raise RuntimeError(
            "Whisper WER is used for evaluation while faster-whisper is not installed"
        )
    model_whisper = WhisperModel(model_tag, device=device, compute_type=compute_type)
    if batch_size > 1:
        model = BatchedInferencePipeline(model=model_whisper)
    else:
        model = model_whisper
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size, "batch_size": batch_size, "compute_type": compute_type}
    return wer_utils


def faster_whisper_levenshtein_metric(
    wer_utils, pred_x, ref_text, fs=16000, cache_pred_text=None
):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: faster-whisper model ("model"), text cleaner ("textcleaner"), 
            beam size ("beam size") and batch size ("batch_size")
        pred_x (np.ndarray): test signal (time,)
        ref_text (string): reference transcript
        cache_pred_text (string): transcription from cache (previous modules)
        fs (int): sampling rate in Hz
    Returns:
        ret (dict): ditionary containing occurrences of edit operations
    """
    if cache_pred_text is not None:
        inf_text = cache_pred_text
    else:
        if fs != TARGET_FS:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            if wer_utils["batch_size"] > 1:
                pred_x = pred_x.astype(getattr(np, wer_utils["compute_type"]))
                inf_output, _ = wer_utils["model"].transcribe(
                    pred_x, beam_size=wer_utils["beam_size"], batch_size=wer_utils["batch_size"]
                )
                inf_text = "".join(segment.text for segment in inf_output)
            else:
                inf_output, _ = wer_utils["model"].transcribe(
                    pred_x, beam_size=wer_utils["beam_size"]
                )
                inf_text = "".join(segment.text for segment in inf_output)


    ref_text = wer_utils["cleaner"](ref_text).strip()
    pred_text = wer_utils["cleaner"](inf_text).strip()

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "faster_whisper_hyp_text": pred_text,
        "ref_text": ref_text,
        "faster_whisper_wer_delete": 0,
        "faster_whisper_wer_insert": 0,
        "faster_whisper_wer_replace": 0,
        "faster_whisper_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["faster_whisper_wer_" + op] = ret["faster_whisper_wer_" + op] + inf_et - inf_st
        else:
            ret["faster_whisper_wer_" + op] = ret["faster_whisper_wer_" + op] + ref_et - ref_st
    total = (
        ret["faster_whisper_wer_delete"]
        + ret["faster_whisper_wer_replace"]
        + ret["faster_whisper_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["faster_whisper_wer_insert"]
        + ret["faster_whisper_wer_replace"]
        + ret["faster_whisper_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret["faster_whisper_cer_delete"] = 0
    ret["faster_whisper_cer_insert"] = 0
    ret["faster_whisper_cer_replace"] = 0
    ret["faster_whisper_cer_equal"] = 0
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["faster_whisper_cer_" + op] = ret["faster_whisper_cer_" + op] + inf_et - inf_st
        else:
            ret["faster_whisper_cer_" + op] = ret["faster_whisper_cer_" + op] + ref_et - ref_st
    total = (
        ret["faster_whisper_cer_delete"]
        + ret["faster_whisper_cer_replace"]
        + ret["faster_whisper_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["faster_whisper_cer_insert"]
        + ret["faster_whisper_cer_replace"]
        + ret["faster_whisper_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = faster_whisper_wer_setup()
    print(
        "metrics: {}".format(
            faster_whisper_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
