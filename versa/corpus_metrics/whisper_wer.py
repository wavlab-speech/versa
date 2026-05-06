#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np
import torch
from Levenshtein import opcodes

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    import whisper
except ImportError:
    logging.warning(
        "Whisper is not properly installed. Please install following "
        "https://github.com/openai/whisper"
    )
    whisper = None

try:
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    TextCleaner = None

try:
    from espnet2.text.phoneme_tokenizer import PhonemeTokenizer
except ImportError:
    PhonemeTokenizer = None

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def whisper_wer_setup(
    model_tag="default",
    beam_size=5,
    text_cleaner="whisper_basic",
    calc_per=False,
    use_gpu=True,
    cache_dir="versa_cache/whisper",
):
    if model_tag == "default":
        model_tag = "large"
    device = "cuda" if use_gpu else "cpu"
    if whisper is None:
        raise RuntimeError(
            "Whisper WER is used for evaluation while openai-whisper is not installed"
        )
    if TextCleaner is None:
        raise ImportError("whisper_wer requires espnet TextCleaner. Install espnet")
    if calc_per and PhonemeTokenizer is None:
        raise ImportError(
            "whisper_wer PER requires espnet PhonemeTokenizer. Install espnet"
        )
    model = whisper.load_model(model_tag, device=device, download_root=cache_dir)
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size}
    if calc_per:
        wer_utils["g2p"] = {
            "zh": PhonemeTokenizer("pypinyin_g2p_phone_without_prosody"),
            "ja": PhonemeTokenizer("pyopenjtalk"),
            "en": PhonemeTokenizer("g2p_en"),
        }
    return wer_utils


def _flatten_phonemes(tokens):
    return [phone for token in tokens for phone in token.strip().split("_") if phone]


def whisper_levenshtein_metric(
    wer_utils,
    pred_x,
    ref_text,
    fs=16000,
    cache_pred_text=None,
    cache_pred_language=None,
):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: whisper model ("model"), text cleaner ("textcleaner"), and
            beam size ("beam size")
        pred_x (np.ndarray): test signal (time,)
        ref_text (string): reference transcript
        cache_pred_text (string): transcription from cache (previous modules)
        cache_pred_language (string): language code for cached transcription
        fs (int): sampling rate in Hz
    Returns:
        ret (dict): ditionary containing occurrences of edit operations
    """
    if cache_pred_text is not None:
        inf_text = cache_pred_text
        lang = cache_pred_language
    else:
        if fs != TARGET_FS:
            pred_x = resample_audio(pred_x, fs, TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            results = wer_utils["model"].transcribe(
                torch.tensor(pred_x).float(), beam_size=wer_utils["beam_size"]
            )
            inf_text = results["text"]
            lang = results.get("language")

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
    if "g2p" in wer_utils:
        if lang not in wer_utils["g2p"]:
            raise ValueError(f"Unsupported g2p language for whisper PER: {lang}")

        ref_words = _flatten_phonemes(wer_utils["g2p"][lang].text2tokens(ref_text))
        pred_words = _flatten_phonemes(wer_utils["g2p"][lang].text2tokens(pred_text))
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


class WhisperWerMetric(BaseMetric):
    """Whisper ASR-based WER/CER edit counts."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.beam_size = self.config.get("beam_size", 5)
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.calc_per = self.config.get("calc_per", False)
        self.use_gpu = self.config.get("use_gpu", True)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/whisper")
        self.wer_utils = whisper_wer_setup(
            model_tag=self.model_tag,
            beam_size=self.beam_size,
            text_cleaner=self.text_cleaner,
            calc_per=self.calc_per,
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

        cache_pred_text = metadata.get("whisper_hyp_text")
        general_cache = metadata.get("general_cache")
        if cache_pred_text is None and general_cache:
            cache_pred_text = general_cache.get("whisper_hyp_text")
        cache_pred_language = metadata.get("whisper_language") or metadata.get(
            "language"
        )
        if cache_pred_language is None and general_cache:
            cache_pred_language = general_cache.get("whisper_language")

        fs = metadata.get("sample_rate", 16000)
        return whisper_levenshtein_metric(
            self.wer_utils,
            np.asarray(predictions),
            ref_text,
            fs=fs,
            cache_pred_text=cache_pred_text,
            cache_pred_language=cache_pred_language,
        )

    def get_metadata(self):
        return _whisper_wer_metadata()


def _whisper_wer_metadata():
    return MetricMetadata(
        name="whisper_wer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["whisper", "espnet2", "Levenshtein", "numpy", "torch"],
        description="Whisper ASR-based WER, CER, and optional PER edit counts",
        paper_reference="https://arxiv.org/abs/2212.04356",
        implementation_source="https://github.com/openai/whisper",
    )


def register_whisper_wer_metric(registry):
    """Register Whisper WER with the registry."""
    registry.register(
        WhisperWerMetric,
        _whisper_wer_metadata(),
        aliases=["whisper_asr_wer", "whisper_wer_metric"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = WhisperWerMetric()
    print(metric.compute(a, metadata={"sample_rate": 16000, "text": "test sentence"}))
