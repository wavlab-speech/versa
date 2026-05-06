#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import numpy as np
import torch

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

logger = logging.getLogger(__name__)

try:
    import whisper
except ImportError:
    logger.info(
        "Whisper is not properly installed. Please install following "
        "https://github.com/openai/whisper"
    )
    whisper = None

try:
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    logger.info("ESPnet is not properly installed. Please install espnet and retry")
    TextCleaner = None

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def speaking_rate_model_setup(
    model_tag="default", beam_size=5, text_cleaner="whisper_basic", use_gpu=True
):
    if model_tag == "default":
        model_tag = "large"
    device = "cuda" if use_gpu else "cpu"
    if whisper is None:
        raise ImportError(
            "speaking_rate requires openai-whisper. "
            "Please install following https://github.com/openai/whisper"
        )
    if TextCleaner is None:
        raise ImportError(
            "speaking_rate requires espnet TextCleaner. Please install espnet"
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
            pred_x = resample_audio(pred_x, fs, TARGET_FS)
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


class SpeakingRateMetric(BaseMetric):
    """Speaking word or character rate estimated from Whisper ASR output."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.beam_size = self.config.get("beam_size", 5)
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.use_char = self.config.get("use_char", False)
        self.wer_utils = speaking_rate_model_setup(
            model_tag=self.model_tag,
            beam_size=self.beam_size,
            text_cleaner=self.text_cleaner,
            use_gpu=self.use_gpu,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        metadata = metadata or {}
        cache_text = metadata.get("whisper_hyp_text")
        general_cache = metadata.get("general_cache")
        if cache_text is None and general_cache:
            cache_text = general_cache.get("whisper_hyp_text")

        fs = metadata.get("sample_rate", 16000)
        pred_x = np.asarray(predictions)
        return speaking_rate_metric(
            self.wer_utils,
            pred_x,
            cache_text=cache_text,
            fs=fs,
            use_char=self.use_char,
        )

    def get_metadata(self):
        return _speaking_rate_metadata()


def _speaking_rate_metadata():
    return MetricMetadata(
        name="speaking_rate",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["whisper", "espnet2", "librosa", "torch", "numpy"],
        description="Speaking word or character rate estimated from Whisper ASR",
        paper_reference="https://github.com/openai/whisper",
        implementation_source="https://github.com/openai/whisper",
    )


def register_speaking_rate_metric(registry):
    """Register speaking_rate with the registry."""
    registry.register(
        SpeakingRateMetric,
        _speaking_rate_metadata(),
        aliases=["speaking_rate_metric", "swr"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = SpeakingRateMetric()
    print("metrics: {}".format(metric.compute(a, metadata={"sample_rate": 16000})))
