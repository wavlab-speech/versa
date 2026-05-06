#!/usr/bin/env python3

# Copyright 2025 Haoran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
from Levenshtein import opcodes

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

try:
    import nemo.collections.asr as nemo_asr
except ImportError:
    logging.warning(
        "NeMo is not properly installed. Please install following "
        "https://github.com/NVIDIA/NeMo"
    )
    nemo_asr = None

try:
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    TextCleaner = None

TARGET_FS = 16000


def nemo_wer_setup(model_tag="default", text_cleaner="whisper_basic", use_gpu=True):
    if model_tag == "default":
        model_tag = "nvidia/stt_en_conformer_transducer_xlarge"
    device = "cuda" if use_gpu else "cpu"
    if nemo_asr is None:
        raise RuntimeError(
            "nemo_wer requires NeMo. Please install tools/install_nemo.sh"
        )
    if TextCleaner is None:
        raise ImportError("nemo_wer requires espnet TextCleaner")

    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_tag)
    asr_model = asr_model.to(device)
    textcleaner = TextCleaner(text_cleaner)
    wer_utils = {"model": asr_model, "cleaner": textcleaner, "device": device}
    return wer_utils


def _extract_nemo_text(transcription):
    if isinstance(transcription, str):
        return transcription
    if hasattr(transcription, "text"):
        return transcription.text
    if isinstance(transcription, (list, tuple)) and transcription:
        return _extract_nemo_text(transcription[0])
    return str(transcription)


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
            pred_x = resample_audio(pred_x, fs, TARGET_FS)
            fs = TARGET_FS
        with torch.no_grad():
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp_path = tmp.name
                sf.write(tmp_path, pred_x, fs)
                transcription = wer_utils["model"].transcribe([tmp_path])
                inf_text = _extract_nemo_text(transcription)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.unlink(tmp_path)

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
    total = ret["nemo_wer_delete"] + ret["nemo_wer_replace"] + ret["nemo_wer_equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret["nemo_wer_insert"] + ret["nemo_wer_replace"] + ret["nemo_wer_equal"]
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
    total = ret["nemo_cer_delete"] + ret["nemo_cer_replace"] + ret["nemo_cer_equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret["nemo_cer_insert"] + ret["nemo_cer_replace"] + ret["nemo_cer_equal"]
    assert total == len(pred_words), (total, len(pred_words))

    return ret


class NemoWerMetric(BaseMetric):
    """NVIDIA NeMo ASR-based WER/CER edit counts."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.wer_utils = nemo_wer_setup(
            model_tag=self.model_tag,
            text_cleaner=self.text_cleaner,
            use_gpu=self.use_gpu,
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

        cache_pred_text = metadata.get("nemo_hyp_text")
        general_cache = metadata.get("general_cache")
        if cache_pred_text is None and general_cache:
            cache_pred_text = general_cache.get("nemo_hyp_text")

        fs = metadata.get("sample_rate", 16000)
        return nemo_levenshtein_metric(
            self.wer_utils,
            np.asarray(predictions),
            ref_text,
            fs=fs,
            cache_pred_text=cache_pred_text,
        )

    def get_metadata(self):
        return _nemo_wer_metadata()


def _nemo_wer_metadata():
    return MetricMetadata(
        name="nemo_wer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["nemo", "espnet2", "Levenshtein", "numpy", "soundfile", "torch"],
        description="NVIDIA NeMo Conformer-Transducer ASR-based WER and CER edit counts",
        paper_reference="https://arxiv.org/abs/2005.08100",
        implementation_source=(
            "https://huggingface.co/nvidia/stt_en_conformer-transducer-xlarge"
        ),
    )


def register_nemo_wer_metric(registry):
    """Register NeMo WER with the registry."""
    registry.register(
        NemoWerMetric,
        _nemo_wer_metadata(),
        aliases=["nemo_asr_wer"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    wer_utils = nemo_wer_setup()
    print(
        "metrics: {}".format(
            nemo_levenshtein_metric(wer_utils, a, "test a sentence.", 16000)
        )
    )
