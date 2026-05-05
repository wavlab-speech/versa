#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import importlib.util
import logging

import librosa
import numpy as np
import torch
from Levenshtein import opcodes

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def _ensure_torchaudio_legacy_backend_api():
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None


_ensure_torchaudio_legacy_backend_api()

try:
    from espnet2.bin.asr_inference import Speech2Text
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    Speech2Text = None
    TextCleaner = None

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def espnet_wer_setup(
    model_tag="default",
    beam_size=5,
    text_cleaner="whisper_basic",
    use_gpu=True,
    cache_dir=None,
):
    if model_tag == "default":
        model_tag = (
            "espnet/"
            "simpleoier_librispeech_asr_train_asr_conformer7_wavlm_large_raw_en_"
            "bpe5000_sp"
        )
    device = "cuda" if use_gpu else "cpu"
    if Speech2Text is None or TextCleaner is None:
        raise ImportError("espnet_wer requires espnet. Please install espnet and retry")
    if cache_dir is None:
        model = Speech2Text.from_pretrained(
            model_tag=model_tag,
            device=device,
            beam_size=beam_size,
        )
    else:
        try:
            from espnet_model_zoo.downloader import ModelDownloader
        except ImportError:
            raise ImportError(
                "espnet_wer requires espnet_model_zoo. Please install it and retry"
            )
        model_kwargs = ModelDownloader(cachedir=cache_dir).download_and_unpack(
            model_tag
        )
        model = Speech2Text(device=device, beam_size=beam_size, **model_kwargs)
    textcleaner = TextCleaner(text_cleaner)
    if "whisper" in text_cleaner:
        if importlib.util.find_spec("whisper") is None:
            logging.warning(
                "Whipser-based cleaner is used but openai-whisper is not installed"
            )
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size}
    return wer_utils


def espnet_predict(
    model,
    speech,
    fs: int,
    beam_size: int = 5,
):
    """Generate predictions using the espnet model. (from URGENT Challenge)

    Args:
        model (torch.nn.Module): espnet model.
        speech (np.ndarray): speech signal < 120s (time,)
        fs (int): sampling rate in Hz.
        beam_size (int): beam size used in beam search.
    Returns:
        text (str): predicted text
    """
    model.beam_search.beam_size = int(beam_size)

    assert fs == 16000, (fs, 16000)

    # assuming 10 tokens per second
    model.maxlenratio = -min(300, int((len(speech) / TARGET_FS) * 10))

    speech = librosa.util.fix_length(speech, size=(TARGET_FS * CHUNK_SIZE))
    text = model(speech)[0][0]

    return text


def espnet_levenshtein_metric(wer_utils, pred_x, ref_text, fs=16000):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: espnet model ("model"), text cleaner ("textcleaner"), and
            beam size ("beam size")
        pred_x (np.ndarray): test signal (time,)
        ref_text (string): reference transcript
        fs (int): sampling rate in Hz
    Returns:
        ret (dict): ditionary containing occurrences of edit operations
    """
    if fs != TARGET_FS:
        pred_x = resample_audio(pred_x, fs, TARGET_FS)
        fs = TARGET_FS
    with torch.no_grad():
        inf_txt = espnet_predict(
            wer_utils["model"],
            pred_x,
            fs,
            beam_size=wer_utils["beam_size"],
        )

    ref_text = wer_utils["cleaner"](ref_text)
    pred_text = wer_utils["cleaner"](inf_txt)

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "espnet_hyp_text": pred_text,
        "ref_text": ref_text,
        "espnet_wer_delete": 0,
        "espnet_wer_insert": 0,
        "espnet_wer_replace": 0,
        "espnet_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["espnet_wer_" + op] = ret["espnet_wer_" + op] + inf_et - inf_st
        else:
            ret["espnet_wer_" + op] = ret["espnet_wer_" + op] + ref_et - ref_st
    total = (
        ret["espnet_wer_delete"] + ret["espnet_wer_replace"] + ret["espnet_wer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["espnet_wer_insert"] + ret["espnet_wer_replace"] + ret["espnet_wer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        espnet_cer_delete=0,
        espnet_cer_insert=0,
        espnet_cer_replace=0,
        espnet_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["espnet_cer_" + op] = ret["espnet_cer_" + op] + inf_et - inf_st
        else:
            ret["espnet_cer_" + op] = ret["espnet_cer_" + op] + ref_et - ref_st
    total = (
        ret["espnet_cer_delete"] + ret["espnet_cer_replace"] + ret["espnet_cer_equal"]
    )
    assert total == len(ref_words), (total, len(ref_words))
    total = (
        ret["espnet_cer_insert"] + ret["espnet_cer_replace"] + ret["espnet_cer_equal"]
    )
    assert total == len(pred_words), (total, len(pred_words))

    return ret


class EspnetWerMetric(BaseMetric):
    """ESPnet ASR-based WER/CER edit counts."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.beam_size = self.config.get("beam_size", 5)
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/espnet_model_zoo")
        self.wer_utils = espnet_wer_setup(
            model_tag=self.model_tag,
            beam_size=self.beam_size,
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

        fs = metadata.get("sample_rate", 16000)
        return espnet_levenshtein_metric(
            self.wer_utils,
            np.asarray(predictions),
            ref_text,
            fs=fs,
        )

    def get_metadata(self):
        return _espnet_wer_metadata()


def _espnet_wer_metadata():
    return MetricMetadata(
        name="espnet_wer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["espnet2", "Levenshtein", "librosa", "numpy", "torch"],
        description="ESPnet ASR-based WER and CER edit counts",
        paper_reference="https://arxiv.org/pdf/1804.00015",
        implementation_source="https://github.com/espnet/espnet",
    )


def register_espnet_wer_metric(registry):
    """Register ESPnet WER with the registry."""
    registry.register(
        EspnetWerMetric,
        _espnet_wer_metadata(),
        aliases=["espnet_asr_wer", "espnet_wer_metric"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = EspnetWerMetric()
    print(metric.compute(a, metadata={"sample_rate": 16000, "text": "test sentence"}))
