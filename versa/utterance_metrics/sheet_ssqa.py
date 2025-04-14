#!/usr/bin/env python3

# Copyright 2024 Wen-Chin Huang
# MIT License (https://opensource.org/licenses/MIT)
# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import librosa
import numpy as np
import torch


def sheet_ssqa_setup(
    model_tag="default",
    model_path=None,
    model_config=None,
    cache_dir="versa_cache",
    use_gpu=False,
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if model_path is not None and model_config is not None:
        raise NotImplementedError(
            "Pending implementation for customized setup (Jiatong)"
        )
    else:
        if model_tag == "default":
            model_tag = "unilight/sheet:v0.1.0"
        torch.hub.set_dir(cache_dir)
        model = torch.hub.load(
            "unilight/sheet:v0.1.0", "default", trust_repo=True, force_reload=False
        )

    model.model.to(device)
    return model


def sheet_ssqa(model, pred_x, fs, use_gpu=False):
    # NOTE(jiatong): current model only work for 16000 Hz
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
    pred_x = torch.tensor(pred_x).float()
    if use_gpu:
        pred_x = pred_x.to("cuda")
    return {"sheet_ssqa": model.predict(wav=pred_x)}


if __name__ == "__main__":
    a = np.random.random(16000)
    model = sheet_ssqa_setup()
    print("metrics: {}".format(sheet_ssqa(model, a, 16000)))
