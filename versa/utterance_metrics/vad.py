#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import librosa
import numpy as np
import torch

from versa.audio_utils import resample_audio
from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType


def vad_model_setup(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=float("inf"),
    min_silence_duration_ms=100,
    speech_pad_ms=30,
    trust_repo=True,
    force_reload=False,
):

    hub_kwargs = {
        "repo_or_dir": "snakers4/silero-vad",
        "model": "silero_vad",
        "force_reload": force_reload,
    }
    if trust_repo is not None:
        hub_kwargs["trust_repo"] = trust_repo
    model, utils = torch.hub.load(**hub_kwargs)
    get_speech_ts, _, _, _, *_ = utils
    return {
        "module": model,
        "util": get_speech_ts,
        "threshold": threshold,
        "min_speech_duration_ms": min_speech_duration_ms,
        "max_speech_duration_s": max_speech_duration_s,
        "min_silence_duration_ms": min_silence_duration_ms,
        "speech_pad_ms": speech_pad_ms,
    }


def vad_metric(model_info, pred_x, fs):
    model = model_info["module"]
    get_speech_ts = model_info["util"]
    # NOTE(jiatong): only work for 16000 Hz
    if fs > 16000:
        pred_x = resample_audio(pred_x, fs, 16000)
        fs = 16000
    elif fs < 16000:
        pred_x = resample_audio(pred_x, fs, 8000)
        fs = 8000

    speech_timestamps = get_speech_ts(
        pred_x,
        model,
        sampling_rate=fs,
        return_seconds=True,
        threshold=model_info["threshold"],
        min_speech_duration_ms=model_info["min_speech_duration_ms"],
        max_speech_duration_s=model_info["max_speech_duration_s"],
        min_silence_duration_ms=model_info["min_silence_duration_ms"],
        speech_pad_ms=model_info["speech_pad_ms"],
    )
    return {"vad_info": speech_timestamps}


class VadMetric(BaseMetric):
    """Voice activity detection using Silero VAD."""

    def _setup(self):
        self.threshold = self.config.get("threshold", 0.5)
        self.min_speech_duration_ms = self.config.get("min_speech_duration_ms", 250)
        self.max_speech_duration_s = self.config.get(
            "max_speech_duration_s", float("inf")
        )
        self.min_silence_duration_ms = self.config.get("min_silence_duration_ms", 100)
        self.speech_pad_ms = self.config.get("speech_pad_ms", 30)
        self.trust_repo = self.config.get("trust_repo", True)
        self.force_reload = self.config.get("force_reload", False)
        self.model_info = vad_model_setup(
            threshold=self.threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=self.max_speech_duration_s,
            min_silence_duration_ms=self.min_silence_duration_ms,
            speech_pad_ms=self.speech_pad_ms,
            trust_repo=self.trust_repo,
            force_reload=self.force_reload,
        )

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        return vad_metric(self.model_info, np.asarray(predictions), fs)

    def get_metadata(self):
        return _vad_metadata()


def _vad_metadata():
    return MetricMetadata(
        name="vad",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["torch", "librosa", "numpy"],
        description="Voice activity detection timestamps from Silero VAD",
        paper_reference="https://arxiv.org/abs/2111.14467",
        implementation_source="https://github.com/snakers4/silero-vad",
    )


def register_vad_metric(registry):
    """Register VAD with the registry."""
    registry.register(
        VadMetric,
        _vad_metadata(),
        aliases=["vad_metric", "silero_vad"],
    )


if __name__ == "__main__":
    torch.hub.download_url_to_file(
        "https://models.silero.ai/vad_models/en.wav", "en_example.wav"
    )
    a, fs = librosa.load("en_example.wav", sr=None)
    metric = VadMetric()
    print("metrics: {}".format(metric.compute(a, metadata={"sample_rate": fs})))
