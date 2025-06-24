#!/usr/bin/env python3

# Copyright 2025 Haoran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

try:
    from transformers import Wav2Vec2Processor, HubertForCTC
except ImportError:
    logging.warning(
        "transformers is not properly installed."
    )
    Wav2Vec2Processor = None
    HubertForCTC = None

from espnet2.text.cleaner import TextCleaner

TARGET_FS = 16000


def hubert_wer_setup(
    model_tag="default", text_cleaner="whisper_basic", use_gpu=True
):
    if model_tag == "default":
        model_tag = "facebook/hubert-large-ls960-ft"
    device = "cuda" if use_gpu else "cpu"
    if Wav2Vec2Processor is None and HubertForCTC is None:
        raise RuntimeError(
            "Facebook's hubert WER is used for evaluation while transformers is not installed"
        )
    processor = Wav2Vec2Processor.from_pretrained(model_tag)
    model = HubertForCTC.from_pretrained(model_tag)

    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": model, "processor": processor, "cleaner": textcleaner}
    return wer_utils


def hubert_levenshtein_metric(
    wer_utils, pred_x, ref_text, fs=16000, cache_pred_text=None
):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: hubert asr model ("model"), text cleaner ("textcleaner")
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
            input_values = wer_utils["processor"](pred_x, return_tensors="pt").input_values
            logits = wer_utils["model"](input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            inf_text = wer_utils["processor"].decode(predicted_ids[0])

    ref_text = wer_utils["cleaner"](ref_text).strip()
    pred_text = wer_utils["cleaner"](inf_text).strip()

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "hubert_hyp_text": pred_text,
        "ref_text": ref_text,
        "hubert_wer_delete": 0,
        "hubert_wer_insert": 0,
        "hubert_wer_replace": 0,
        "hubert_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["hubert_wer_" + op] = ret["hubert_wer_" + op] + inf_et - inf_st
        else:
            ret["hubert_wer_" + op] = ret["hubert_wer_" + op] + ref_et - ref_st
    total = (
        ret["hubert_wer_delete"]
        + ret["hubert_wer_replace"]
        + ret["hubert_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["hubert_wer_insert"]
        + ret["hubert_wer_replace"]
        + ret["hubert_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        hubert_cer_delete=0,
        hubert_cer_insert=0,
        hubert_cer_replace=0,
        hubert_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["hubert_cer_" + op] = ret["hubert_cer_" + op] + inf_et - inf_st
        else:
            ret["hubert_cer_" + op] = ret["hubert_cer_" + op] + ref_et - ref_st
    total = (
        ret["hubert_cer_delete"]
        + ret["hubert_cer_replace"]
        + ret["hubert_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["hubert_cer_insert"]
        + ret["hubert_cer_replace"]
        + ret["hubert_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = hubert_wer_setup()
    print(
        "metrics: {}".format(
            hubert_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
