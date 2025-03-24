#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import numpy as np
from espnet2.bin.enh_inference import SeparateSpeech

from versa.sequence_metrics.signal_metric import signal_metric


def se_snr_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        model = SeparateSpeech.from_pretrained(
            model_file=model_path,
            train_config=model_config,
            normalize_output_wav=True,
            device=device,
        )
    else:
        if model_tag == "default":
            model_tag = "wyz/tfgridnet_for_urgent24"
        model = SeparateSpeech.from_pretrained(
            model_tag=model_tag, normalize_output_wav=True, device=device
        )
    return model


def se_snr(model, pred_x, fs):
    enhanced_x = model(pred_x[None, :], fs=fs)[0]
    signal_metrics = signal_metric(pred_x, enhanced_x)
    updated_metrics = {f"se_{key}": value for key, value in signal_metrics.items()}
    updated_metrics.pop("se_sir")
    return updated_metrics


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    model = se_snr_setup()
    print("metrics: {}".format(se_snr(model, a, 16000)))
