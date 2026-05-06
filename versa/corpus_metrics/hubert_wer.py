#!/usr/bin/env python3

# Copyright 2025 Haoran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np
import torch
from Levenshtein import opcodes

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    from transformers import HubertForCTC, Wav2Vec2Processor
except ImportError:
    logging.warning("transformers is not properly installed.")
    HubertForCTC = None
    Wav2Vec2Processor = None

try:
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    TextCleaner = None

TARGET_FS = 16000


def hubert_wer_setup(
    model_tag="default",
    text_cleaner="whisper_basic",
    use_gpu=True,
    cache_dir="versa_cache/huggingface",
):
    if model_tag == "default":
        model_tag = "facebook/hubert-large-ls960-ft"
    device = "cuda" if use_gpu else "cpu"
    if Wav2Vec2Processor is None or HubertForCTC is None:
        raise RuntimeError(
            "hubert_wer requires transformers. Please install transformers and retry"
        )
    if TextCleaner is None:
        raise ImportError("hubert_wer requires espnet TextCleaner")

    processor = Wav2Vec2Processor.from_pretrained(model_tag, cache_dir=cache_dir)
    model = HubertForCTC.from_pretrained(model_tag, cache_dir=cache_dir).to(device)
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {
        "model": model,
        "processor": processor,
        "cleaner": textcleaner,
        "device": device,
    }
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
            pred_x = resample_audio(pred_x, fs, TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            input_values = wer_utils["processor"](
                pred_x,
                sampling_rate=TARGET_FS,
                return_tensors="pt",
            ).input_values.to(wer_utils["device"])
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
        ret["hubert_wer_delete"] + ret["hubert_wer_replace"] + ret["hubert_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["hubert_wer_insert"] + ret["hubert_wer_replace"] + ret["hubert_wer_equal"]
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
        ret["hubert_cer_delete"] + ret["hubert_cer_replace"] + ret["hubert_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["hubert_cer_insert"] + ret["hubert_cer_replace"] + ret["hubert_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


class HubertWerMetric(BaseMetric):
    """HuBERT CTC ASR-based WER/CER edit counts."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/huggingface")
        self.wer_utils = hubert_wer_setup(
            model_tag=self.model_tag,
            text_cleaner=self.text_cleaner,
            use_gpu=self.use_gpu,
            cache_dir=self.cache_dir,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        metadata = metadata or {}
        ref_text = metadata.get("text")
        if ref_text is None and isinstance(references, str):
            ref_text = references
        if ref_text is None:
            raise ValueError("Reference text must be provided")

        cache_pred_text = metadata.get("hubert_hyp_text")
        general_cache = metadata.get("general_cache")
        if cache_pred_text is None and general_cache:
            cache_pred_text = general_cache.get("hubert_hyp_text")

        fs = metadata.get("sample_rate", 16000)
        return hubert_levenshtein_metric(
            self.wer_utils,
            np.asarray(predictions),
            ref_text,
            fs=fs,
            cache_pred_text=cache_pred_text,
        )

    def get_metadata(self):
        return _hubert_wer_metadata()


def _hubert_wer_metadata():
    return MetricMetadata(
        name="hubert_wer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=True,
        dependencies=["transformers", "espnet2", "Levenshtein", "numpy", "torch"],
        description="Facebook HuBERT-large CTC ASR-based WER and CER edit counts",
        paper_reference="https://arxiv.org/abs/2106.07447",
        implementation_source="https://huggingface.co/facebook/hubert-large-ls960-ft",
    )


def register_hubert_wer_metric(registry):
    """Register HuBERT WER with the registry."""
    registry.register(
        HubertWerMetric,
        _hubert_wer_metadata(),
        aliases=["hubert_asr_wer", "facebook_hubert_wer"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = hubert_wer_setup()
    print(
        "metrics: {}".format(
            hubert_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
