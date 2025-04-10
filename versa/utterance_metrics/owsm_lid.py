#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os

import librosa
import numpy as np
from espnet2.bin.s2t_inference_language import Speech2Language


def owsm_lid_model_setup(model_tag="default", nbest=3, use_gpu=False):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_tag == "default":
        model_tag = "espnet/owsm_v3.1_ebf"
    model = Speech2Language.from_pretrained(
        model_tag=model_tag,
        device=device,
        nbest=nbest,
    )

    return model


def language_id(model, pred_x, fs):
    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    result = model(pred_x)
    return {"language": result}


if __name__ == "__main__":
    a = np.random.random(16000)
    model = owsm_lid_model_setup()
    print("metrics: {}".format(language_id(model, a, 16000)))
