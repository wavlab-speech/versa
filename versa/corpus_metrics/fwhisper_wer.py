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
    from faster_whisper import BatchedInferencePipeline, WhisperModel
except ImportError:
    logging.warning(
        "Faster-whisper is not properly installed. Please install following "
        "https://github.com/systran/faster-whisper"
    )
    BatchedInferencePipeline = None
    WhisperModel = None

try:
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    TextCleaner = None

TARGET_FS = 16000


def fwhisper_wer_setup(
    model_tag="default",
    beam_size=5,
    batch_size=1,
    compute_type="float32",
    text_cleaner="whisper_basic",
    use_gpu=True,
    cache_dir="versa_cache/faster_whisper",
):
    if model_tag == "default":
        model_tag = "large-v3"
    device = "cuda" if use_gpu else "cpu"
    if WhisperModel is None or BatchedInferencePipeline is None:
        raise RuntimeError(
            "faster_whisper_wer requires faster-whisper. "
            "Please install it with tools/install_fwhisper.sh"
        )
    if TextCleaner is None:
        raise ImportError("faster_whisper_wer requires espnet TextCleaner")

    model_whisper = WhisperModel(
        model_tag,
        device=device,
        compute_type=compute_type,
        download_root=cache_dir,
    )
    if batch_size > 1:
        model = BatchedInferencePipeline(model=model_whisper)
    else:
        model = model_whisper
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {
        "model": model,
        "cleaner": textcleaner,
        "beam_size": beam_size,
        "batch_size": batch_size,
        "compute_type": compute_type,
    }
    return wer_utils


def fwhisper_levenshtein_metric(
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
            pred_x = resample_audio(pred_x, fs, TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            if wer_utils["batch_size"] > 1:
                inf_output, _ = wer_utils["model"].transcribe(
                    pred_x,
                    beam_size=wer_utils["beam_size"],
                    batch_size=wer_utils["batch_size"],
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
        "fwhisper_hyp_text": pred_text,
        "ref_text": ref_text,
        "fwhisper_wer_delete": 0,
        "fwhisper_wer_insert": 0,
        "fwhisper_wer_replace": 0,
        "fwhisper_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["fwhisper_wer_" + op] = ret["fwhisper_wer_" + op] + inf_et - inf_st
        else:
            ret["fwhisper_wer_" + op] = ret["fwhisper_wer_" + op] + ref_et - ref_st
    total = (
        ret["fwhisper_wer_delete"]
        + ret["fwhisper_wer_replace"]
        + ret["fwhisper_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["fwhisper_wer_insert"]
        + ret["fwhisper_wer_replace"]
        + ret["fwhisper_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret["fwhisper_cer_delete"] = 0
    ret["fwhisper_cer_insert"] = 0
    ret["fwhisper_cer_replace"] = 0
    ret["fwhisper_cer_equal"] = 0
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["fwhisper_cer_" + op] = ret["fwhisper_cer_" + op] + inf_et - inf_st
        else:
            ret["fwhisper_cer_" + op] = ret["fwhisper_cer_" + op] + ref_et - ref_st
    total = (
        ret["fwhisper_cer_delete"]
        + ret["fwhisper_cer_replace"]
        + ret["fwhisper_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["fwhisper_cer_insert"]
        + ret["fwhisper_cer_replace"]
        + ret["fwhisper_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


class FasterWhisperWerMetric(BaseMetric):
    """Faster-Whisper ASR-based WER/CER edit counts."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.beam_size = self.config.get("beam_size", 5)
        self.batch_size = self.config.get("batch_size", 1)
        self.compute_type = self.config.get("compute_type", "float32")
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/faster_whisper")
        self.wer_utils = fwhisper_wer_setup(
            model_tag=self.model_tag,
            beam_size=self.beam_size,
            batch_size=self.batch_size,
            compute_type=self.compute_type,
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

        cache_pred_text = metadata.get("fwhisper_hyp_text")
        general_cache = metadata.get("general_cache")
        if cache_pred_text is None and general_cache:
            cache_pred_text = general_cache.get("fwhisper_hyp_text")

        fs = metadata.get("sample_rate", 16000)
        return fwhisper_levenshtein_metric(
            self.wer_utils,
            np.asarray(predictions),
            ref_text,
            fs=fs,
            cache_pred_text=cache_pred_text,
        )

    def get_metadata(self):
        return _fwhisper_wer_metadata()


def _fwhisper_wer_metadata():
    return MetricMetadata(
        name="fwhisper_wer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=False,
        dependencies=[
            "faster_whisper",
            "espnet2",
            "Levenshtein",
            "numpy",
            "torch",
        ],
        description="Faster-Whisper ASR-based WER and CER edit counts",
        implementation_source="https://github.com/SYSTRAN/faster-whisper",
    )


def register_fwhisper_wer_metric(registry):
    """Register Faster-Whisper WER with the registry."""
    registry.register(
        FasterWhisperWerMetric,
        _fwhisper_wer_metadata(),
        aliases=["faster_whisper_wer", "fwhisper_asr_wer"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = fwhisper_wer_setup()
    print(
        "metrics: {}".format(
            fwhisper_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
