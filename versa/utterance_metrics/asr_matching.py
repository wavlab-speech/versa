#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, Optional, Union, Any

import numpy as np
import torch
from Levenshtein import opcodes

logger = logging.getLogger(__name__)

# Handle optional whisper dependency
try:
    import whisper

    WHISPER_AVAILABLE = True
except ImportError:
    logger.warning(
        "Whisper is not properly installed. "
        "Please install following https://github.com/openai/whisper"
    )
    whisper = None
    WHISPER_AVAILABLE = False

try:
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    TextCleaner = None
from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType

# Constants
TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


class WhisperNotAvailableError(RuntimeError):
    """Exception raised when Whisper is required but not available."""

    pass


def is_whisper_available():
    """
    Check if the Whisper package is available.

    Returns:
        bool: True if Whisper is available, False otherwise.
    """
    return WHISPER_AVAILABLE


class ASRMatchMetric(BaseMetric):
    """ASR-oriented Mismatch Error Rate (ASR-Match) metric using Whisper."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.beam_size = self.config.get("beam_size", 5)
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/whisper")
        self.wer_utils = asr_match_setup(
            self.model_tag,
            self.beam_size,
            self.text_cleaner,
            use_gpu=self.use_gpu,
            cache_dir=self.cache_dir,
        )

    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        pred_x = predictions
        gt_x = references
        fs = 16000
        cache_pred_text = None
        if metadata is not None:
            fs = metadata.get("sample_rate", 16000)
            cache_pred_text = metadata.get("cache_pred_text", None)
        # Validate inputs
        if pred_x is None or gt_x is None:
            raise ValueError("Both predicted and ground truth signals must be provided")
        pred_x = np.asarray(pred_x)
        gt_x = np.asarray(gt_x)
        return asr_match_metric(self.wer_utils, pred_x, gt_x, cache_pred_text, fs)

    def get_metadata(self) -> MetricMetadata:
        return MetricMetadata(
            name="asr_match",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["whisper", "espnet2", "Levenshtein", "librosa", "torch"],
            description="ASR-oriented Mismatch Error Rate (ASR-Match) using Whisper for reference-based speech evaluation.",
            paper_reference=None,
            implementation_source="https://github.com/ftshijt/versa",
        )


def register_asr_match_metric(registry):
    """Register ASR-Match metric with the registry."""
    metric_metadata = MetricMetadata(
        name="asr_match",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["whisper", "espnet2", "Levenshtein", "librosa", "torch"],
        description="ASR-oriented Mismatch Error Rate (ASR-Match) using Whisper for reference-based speech evaluation.",
        paper_reference=None,
        implementation_source="https://github.com/ftshijt/versa",
    )
    registry.register(
        ASRMatchMetric, metric_metadata, aliases=["ASRMatch", "asr_match_error_rate"]
    )


def asr_match_setup(
    model_tag="default",
    beam_size=5,
    text_cleaner="whisper_basic",
    use_gpu=True,
    cache_dir="versa_cache/whisper",
):
    """Legacy function API for setting up ASR-Match."""
    if not WHISPER_AVAILABLE:
        raise ImportError(
            "Whisper is not properly installed. Please install following https://github.com/openai/whisper"
        )
    if TextCleaner is None:
        raise ImportError("asr_match requires espnet TextCleaner. Install espnet")
    if model_tag == "default":
        model_tag = "large"
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    try:
        model = whisper.load_model(model_tag, device=device, download_root=cache_dir)
        cleaner = TextCleaner(text_cleaner)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Whisper model: {str(e)}") from e
    return {
        "model": model,
        "cleaner": cleaner,
        "beam_size": beam_size,
        "device": device,
    }


def asr_match_metric(wer_utils, pred_x, gt_x, cache_pred_text=None, fs=16000):
    """Legacy function API for computing ASR-Match."""
    pred_x = np.asarray(pred_x)
    gt_x = np.asarray(gt_x)
    model = wer_utils["model"]
    cleaner = wer_utils["cleaner"]
    beam_size = wer_utils.get("beam_size", 5)

    if cache_pred_text is not None:
        inf_text = cache_pred_text
    else:
        try:
            if fs != TARGET_FS:
                pred_x = resample_audio(pred_x, fs, TARGET_FS)
            with torch.no_grad():
                transcription = model.transcribe(
                    torch.tensor(pred_x).float(), beam_size=beam_size
                )
                inf_text = transcription["text"]
        except Exception as e:
            raise RuntimeError(
                f"Failed to transcribe predicted signal: {str(e)}"
            ) from e

    try:
        if fs != TARGET_FS:
            gt_x = resample_audio(gt_x, fs, TARGET_FS)
        with torch.no_grad():
            transcription = model.transcribe(
                torch.tensor(gt_x).float(), beam_size=beam_size
            )
            gt_text = transcription["text"]
    except Exception as e:
        raise RuntimeError(f"Failed to transcribe ground truth signal: {str(e)}") from e

    ref_text = cleaner(gt_text)
    pred_text = cleaner(inf_text)
    ref_chars = list(ref_text)
    pred_chars = list(pred_text)
    result = {
        "asr_match_delete": 0,
        "asr_match_insert": 0,
        "asr_match_replace": 0,
        "asr_match_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_chars, pred_chars):
        if op == "insert":
            result["asr_match_" + op] += inf_et - inf_st
        else:
            result["asr_match_" + op] += ref_et - ref_st

    if len(ref_chars) == 0:
        asr_match_error_rate = 1.0
        logger.warning("Reference text is empty, setting error rate to 1.0")
    else:
        asr_match_error_rate = (
            result["asr_match_delete"]
            + result["asr_match_insert"]
            + result["asr_match_replace"]
        ) / len(ref_chars)

    return {
        "asr_match_error_rate": asr_match_error_rate,
        "whisper_hyp_text": inf_text,
        "ref_text_length": len(ref_chars),
        "pred_text_length": len(pred_chars),
        "match_details": result,
    }


if __name__ == "__main__":
    # Example usage for the class-based metric
    try:
        # Generate random test audio (1 second at 16kHz)
        test_audio = np.random.random(TARGET_FS)
        # Set up ASR matching metric
        config = {
            "model_tag": "tiny",
            "beam_size": 1,
            "text_cleaner": "whisper_basic",
            "use_gpu": torch.cuda.is_available(),
        }
        metric = ASRMatchMetric(config)
        # Calculate metrics
        metrics = metric.compute(
            test_audio, test_audio, metadata={"sample_rate": TARGET_FS}
        )
        # Print results
        print(f"ASR Match Error Rate: {metrics['asr_match_error_rate']:.4f}")
        print(f"Transcription: '{metrics['whisper_hyp_text']}'")
    except WhisperNotAvailableError:
        print("This script requires the Whisper package. Please install it first.")
    except Exception as e:
        print(f"Error running ASR match: {str(e)}")
