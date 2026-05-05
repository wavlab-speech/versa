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

try:
    from espnet2.bin.s2t_inference import Speech2Text
    from espnet2.text.cleaner import TextCleaner
except ImportError:
    Speech2Text = None
    TextCleaner = None

TARGET_FS = 16000
CHUNK_SIZE = 30  # seconds


def owsm_wer_setup(
    model_tag="default",
    beam_size=5,
    text_cleaner="whisper_basic",
    use_gpu=True,
    cache_dir=None,
):
    if model_tag == "default":
        model_tag = "espnet/owsm_v3.1_ebf"
    device = "cuda" if use_gpu else "cpu"
    if Speech2Text is None or TextCleaner is None:
        raise ImportError("owsm_wer requires espnet. Please install espnet and retry")
    if cache_dir is None:
        model = Speech2Text.from_pretrained(
            model_tag=model_tag,
            device=device,
            task_sym="<asr>",
            beam_size=beam_size,
            predict_time=False,
        )
    else:
        try:
            from espnet_model_zoo.downloader import ModelDownloader
        except ImportError:
            raise ImportError(
                "owsm_wer requires espnet_model_zoo. Please install it and retry"
            )
        model_kwargs = ModelDownloader(cachedir=cache_dir).download_and_unpack(
            model_tag
        )
        model = Speech2Text(
            device=device,
            task_sym="<asr>",
            beam_size=beam_size,
            predict_time=False,
            **model_kwargs,
        )
    textcleaner = TextCleaner(text_cleaner)
    if "whisper" in text_cleaner:
        if importlib.util.find_spec("whisper") is None:
            logging.warning(
                "Whipser-based cleaner is used but openai-whisper is not installed"
            )
    wer_utils = {"model": model, "cleaner": textcleaner, "beam_size": beam_size}
    return wer_utils


# Copied from Whisper utils
def format_timestamp(
    seconds: float, always_include_hours: bool = False, decimal_marker: str = "."
):
    assert seconds >= 0, "non-negative timestamp expected"
    milliseconds = round(seconds * 1000.0)

    hours = milliseconds // 3_600_000
    milliseconds -= hours * 3_600_000

    minutes = milliseconds // 60_000
    milliseconds -= minutes * 60_000

    seconds = milliseconds // 1_000
    milliseconds -= seconds * 1_000

    hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
    return (
        f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    )


def owsm_predict(
    model,
    speech,
    fs: int,
    src_lang: str = "none",
    beam_size: int = 5,
    long_form: bool = False,
    text_prev: str = "",
):
    """Generate predictions using the OWSM model. (from URGENT Challenge)

    Args:
        model (torch.nn.Module): OWSM model.
        speech (np.ndarray): speech signal < 120s (time,)
        fs (int): sampling rate in Hz.
        src_lang (str): source language in ISO 639-2 Code.
        beam_size (int): beam size used in beam search.
        long_form (bool): perform long-form decoding for audios longer than 30s.
            If an exception happens, it will fall back to standard decoding on the
            initial 30s.
        text_prev (str): generation will be conditioned on this prompt if provided.
    Returns:
        text (str): predicted text
    """
    task_sym = "<asr>"
    model.beam_search.beam_size = int(beam_size)

    assert fs == 16000, (fs, 16000)

    # Detect language using the first 30s of speech
    if src_lang == "none":
        # default 30 seconds chunk for owsm training
        src_lang = model(
            librosa.util.fix_length(speech, size=(TARGET_FS * CHUNK_SIZE))
        )[0][0].strip()[1:-1]
    lang_sym = f"<{src_lang}>"

    # ASR or ST
    if long_form:  # speech will be padded in decode_long()
        try:
            model.maxlenratio = -300
            utts = model.decode_long(
                speech,
                condition_on_prev_text=False,
                init_text=text_prev,
                end_time_threshold="<29.00>",
                lang_sym=lang_sym,
                task_sym=task_sym,
            )

            text = []
            for t1, t2, res in utts:
                text.append(
                    f"[{format_timestamp(seconds=t1)} --> "
                    f"{format_timestamp(seconds=t2)}] {res}"
                )
            text = "\n".join(text)

            return text
        except Exception:
            print(
                "An exception occurred in long-form decoding. "
                "Fall back to standard decoding (only first 30s)"
            )

    # assuming 10 tokens per second
    model.maxlenratio = -min(300, int((len(speech) / TARGET_FS) * 10))

    speech = librosa.util.fix_length(speech, size=(TARGET_FS * CHUNK_SIZE))
    text = model(speech, text_prev, lang_sym=lang_sym, task_sym=task_sym)[0][-2]

    return text


def owsm_levenshtein_metric(wer_utils, pred_x, ref_text, fs=16000):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        wer_utils (dict): a utility dict for WER calculation.
            including: owsm model ("model"), text cleaner ("textcleaner"), and
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
        inf_txt = owsm_predict(
            wer_utils["model"],
            pred_x,
            fs,
            src_lang="eng",
            beam_size=wer_utils["beam_size"],
            long_form=len(pred_x) > CHUNK_SIZE * fs,
        )

    ref_text = wer_utils["cleaner"](ref_text).strip()
    pred_text = wer_utils["cleaner"](inf_txt).strip()

    # process wer
    ref_words = ref_text.strip().split()
    pred_words = pred_text.strip().split()
    ret = {
        "owsm_hyp_text": pred_text,
        "ref_text": ref_text,
        "owsm_wer_delete": 0,
        "owsm_wer_insert": 0,
        "owsm_wer_replace": 0,
        "owsm_wer_equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["owsm_wer_" + op] = ret["owsm_wer_" + op] + inf_et - inf_st
        else:
            ret["owsm_wer_" + op] = ret["owsm_wer_" + op] + ref_et - ref_st
    total = ret["owsm_wer_delete"] + ret["owsm_wer_replace"] + ret["owsm_wer_equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret["owsm_wer_insert"] + ret["owsm_wer_replace"] + ret["owsm_wer_equal"]
    assert total == len(pred_words), (total, len(pred_words))

    # process cer
    ref_words = [c for c in ref_text]
    pred_words = [c for c in pred_text]
    ret.update(
        owsm_cer_delete=0,
        owsm_cer_insert=0,
        owsm_cer_replace=0,
        owsm_cer_equal=0,
    )
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, pred_words):
        if op == "insert":
            ret["owsm_cer_" + op] = ret["owsm_cer_" + op] + inf_et - inf_st
        else:
            ret["owsm_cer_" + op] = ret["owsm_cer_" + op] + ref_et - ref_st
    total = ret["owsm_cer_delete"] + ret["owsm_cer_replace"] + ret["owsm_cer_equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret["owsm_cer_insert"] + ret["owsm_cer_replace"] + ret["owsm_cer_equal"]
    assert total == len(pred_words), (total, len(pred_words))

    return ret


class OwsmWerMetric(BaseMetric):
    """OWSM ASR-based WER/CER edit counts."""

    def _setup(self):
        self.model_tag = self.config.get("model_tag", "default")
        self.beam_size = self.config.get("beam_size", 5)
        self.text_cleaner = self.config.get("text_cleaner", "whisper_basic")
        self.use_gpu = self.config.get("use_gpu", True)
        self.cache_dir = self.config.get("cache_dir", "versa_cache/espnet_model_zoo")
        self.wer_utils = owsm_wer_setup(
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
        return owsm_levenshtein_metric(
            self.wer_utils,
            np.asarray(predictions),
            ref_text,
            fs=fs,
        )

    def get_metadata(self):
        return _owsm_wer_metadata()


def _owsm_wer_metadata():
    return MetricMetadata(
        name="owsm_wer",
        category=MetricCategory.NON_MATCH,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=True,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["espnet2", "Levenshtein", "librosa", "numpy", "torch"],
        description="OWSM ASR-based WER and CER edit counts",
        paper_reference="https://arxiv.org/abs/2309.13876",
        implementation_source="https://github.com/espnet/espnet",
    )


def register_owsm_wer_metric(registry):
    """Register OWSM WER with the registry."""
    registry.register(
        OwsmWerMetric,
        _owsm_wer_metadata(),
        aliases=["owsm_asr_wer", "owsm_wer_metric"],
    )


if __name__ == "__main__":
    a = np.random.random(16000)
    metric = OwsmWerMetric()
    print(metric.compute(a, metadata={"sample_rate": 16000, "text": "test sentence"}))
