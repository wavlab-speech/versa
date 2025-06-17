#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import librosa
import numpy as np
from espnet2.bin.universa_inference import UniversaInference


def universa_model_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = UniversaInference(
            model_file=model_path, train_config=model_config, device=device
        )
    else:
        if model_tag == "default":
            model_tag = "espnet/universa-wavlm_base_urgent24_multi-metric_noref"
        model = UniversaInference.from_pretrained(model_tag=model_tag, device=device)
    return model


def audio_preprocess(audio, fs):
    if fs != 16000:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=16000)
    audio = audio.astype(np.float32)
    audio = torch.from_numpy(audio).unsqueeze(0).float()
    audio_lengths = torch.tensor([len(audio[0])])
    return audio, audio_lengths


def universa_metric(model, pred_x, gt_x=None, text=None, fs=16000):
    # NOTE(jiatong): only work for 16000 Hz
    if gt_x is not None:
        gt_x, gt_length = audio_preprocess(gt_x, fs)
    pred_x, pred_length = audio_preprocess(pred_x, fs)

    universa_metrics = model(
        pred_x, pred_length, ref_audio=gt_x, ref_audio_length=gt_length, ref_text=text
    )

    # post process
    result = {}
    for key in universa_metrics.keys():
        if key == "encoded_feat":
            continue  # skip detailed representation extraction
        result["universa_{}".format(key)] = universa_metrics[0][0]
    return result


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    model = universa_model_setup()
    print("metrics: {}".format(universa_metric(model, a, b, 16000)))
