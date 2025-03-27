#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import librosa
import numpy as np
import torch


def vad_model_setup(
    threshold=0.5,
    min_speech_duration_ms=250,
    max_speech_duration_s=float("inf"),
    min_silence_duration_ms=100,
    speech_pad_ms=30,
):

    model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
    (get_speech_ts, _, _, _, *_) = utils
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
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
        fs = 16000
    elif fs < 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=8000)
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


if __name__ == "__main__":
    torch.hub.download_url_to_file(
        "https://models.silero.ai/vad_models/en.wav", "en_example.wav"
    )
    a, fs = librosa.load("en_example.wav", sr=None)
    model_info = vad_model_setup()
    print("metrics: {}".format(vad_metric(model_info, a, 16000)))
