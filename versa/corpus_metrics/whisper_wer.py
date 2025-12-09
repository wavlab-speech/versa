#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

try:
    import whisper
except ImportError:
    logging.warning(
        "Whisper is not properly installed. Please install following https://github.com/openai/whisper"
    )
    whisper = None

from espnet2.text.cleaner import TextCleaner
from espnet2.text.phoneme_tokenizer import PhonemeTokenizer

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def whisper_wer_setup(
    model_tag="default", beam_size=5, text_cleaner="whisper_basic", calc_per=False, use_gpu=True
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
    if calc_per is True:
        g2p = {
            "zh": PhonemeTokenizer("pypinyin_g2p_phone_without_prosody"),
            "ja": PhonemeTokenizer("pyopenjtalk"),
            "en": PhonemeTokenizer("g2p_en"),
            # To support additional languages, add corresponding g2p modules here.
        }
        wer_utils.update(g2p=g2p)
    return wer_utils


def whisper_levenshtein_metric(
    wer_utils, pred_x, ref_text, fs=16000, cache_pred_text=None
):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: whisper model ("model"), text cleaner ("textcleaner"), and
            beam size ("beam size")
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
            results = wer_utils["model"].transcribe(
                torch.tensor(pred_x).float(), beam_size=wer_utils["beam_size"]
            )
            inf_text = results["text"]
            lang = results["language"]

    ref_text = wer_utils["cleaner"](ref_text).strip()
    pred_text = wer_utils["cleaner"](inf_text).strip()

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "whisper_hyp_text": pred_text,
        "ref_text": ref_text,
        "whisper_wer_delete": 0,
        "whisper_wer_insert": 0,
        "whisper_wer_replace": 0,
        "whisper_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["whisper_wer_" + op] = ret["whisper_wer_" + op] + inf_et - inf_st
        else:
            ret["whisper_wer_" + op] = ret["whisper_wer_" + op] + ref_et - ref_st
    total = (
        ret["whisper_wer_delete"]
        + ret["whisper_wer_replace"]
        + ret["whisper_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["whisper_wer_insert"]
        + ret["whisper_wer_replace"]
        + ret["whisper_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        whisper_cer_delete=0,
        whisper_cer_insert=0,
        whisper_cer_replace=0,
        whisper_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["whisper_cer_" + op] = ret["whisper_cer_" + op] + inf_et - inf_st
        else:
            ret["whisper_cer_" + op] = ret["whisper_cer_" + op] + ref_et - ref_st
    total = (
        ret["whisper_cer_delete"]
        + ret["whisper_cer_replace"]
        + ret["whisper_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["whisper_cer_insert"]
        + ret["whisper_cer_replace"]
        + ret["whisper_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process per
    if "g2p" in wer_utils.keys():
        assert lang in wer_utils["g2p"].keys(), f"Not support g2p for {lang} language"
        ref_words = wer_utils["g2p"][lang].text2tokens(ref_text)
        pred_words = wer_utils["g2p"][lang].text2tokens(pred_text)

        ref_words = [p for phn in ref_words for p in phn.strip().split('_')]
        pred_words = [p for phn in pred_words for p in phn.strip().split('_')]

        ret.update(
            whisper_per_delete=0,
            whisper_per_insert=0,
            whisper_per_replace=0,
            whisper_per_equal=0,
        )
        for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
            if op == "insert":
                ret["whisper_per_" + op] = ret["whisper_per_" + op] + inf_et - inf_st
            else:
                ret["whisper_per_" + op] = ret["whisper_per_" + op] + ref_et - ref_st
        total = (
            ret["whisper_per_delete"]
            + ret["whisper_per_replace"]
            + ret["whisper_per_equal"]
        )
        assert total == len(ref_words), (total, len(ref_words))
        total = (
            ret["whisper_per_insert"]
            + ret["whisper_per_replace"]
            + ret["whisper_per_equal"]
        )
        assert total == len(pred_words), (total, len(pred_words))

    return ret


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = whisper_wer_setup()
    print(
        "metrics: {}".format(
            whisper_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
