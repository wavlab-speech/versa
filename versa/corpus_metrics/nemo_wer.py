#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    logging.warning(
        "NeMo is not properly installed. Please install following https://github.com/NVIDIA/NeMo"
    )
    nemo_asr= None

from espnet2.text.cleaner import TextCleaner

TARGET_FS = 16000


def nemo_wer_setup(
    model_tag="default", text_cleaner="whisper_basic", use_gpu=True
):
    if model_tag == "default":
        model_tag = "nvidia/stt_en_conformer_transducer_xlarge"
    device = "cuda" if use_gpu else "cpu"
    if nemo_asr is None:
        raise RuntimeError(
            "NeMo WER is used for evaluation while NeMo is not installed"
        )
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_tag)
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": asr_model, "cleaner": textcleaner}
    return wer_utils


def nemo_levenshtein_metric(
    wer_utils, pred_x, ref_text, fs=16000, cache_pred_text=None
):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: nemo asr model ("model"), text cleaner ("textcleaner")
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
            inf_text = wer_utils["model"].transcribe(
                audio=pred_x
            )[0].text

    ref_text = wer_utils["cleaner"](ref_text).strip()
    pred_text = wer_utils["cleaner"](inf_text).strip()

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "nemo_hyp_text": pred_text,
        "ref_text": ref_text,
        "nemo_wer_delete": 0,
        "nemo_wer_insert": 0,
        "nemo_wer_replace": 0,
        "nemo_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["nemo_wer_" + op] = ret["nemo_wer_" + op] + inf_et - inf_st
        else:
            ret["nemo_wer_" + op] = ret["nemo_wer_" + op] + ref_et - ref_st
    total = (
        ret["nemo_wer_delete"]
        + ret["nemo_wer_replace"]
        + ret["nemo_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["nemo_wer_insert"]
        + ret["nemo_wer_replace"]
        + ret["nemo_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        nemo_cer_delete=0,
        nemo_cer_insert=0,
        nemo_cer_replace=0,
        nemo_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["nemo_cer_" + op] = ret["nemo_cer_" + op] + inf_et - inf_st
        else:
            ret["nemo_cer_" + op] = ret["nemo_cer_" + op] + ref_et - ref_st
    total = (
        ret["nemo_cer_delete"]
        + ret["nemo_cer_replace"]
        + ret["nemo_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["nemo_cer_insert"]
        + ret["nemo_cer_replace"]
        + ret["nemo_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = nemo_wer_setup()
    print(
        "metrics: {}".format(
            nemo_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
