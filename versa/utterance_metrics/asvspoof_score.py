#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Author: You (Neil) Zhang

## This is to evalute the generated speech with a deepfake detection model
# We include the AASIST model trained on ASVspoof 2019 LA dataset to
# output the confidence score of whether the speech input is a deepfake
# Please refer to https://github.com/clovaai/aasist for more details


import json
import os
import sys

import librosa
import numpy as np
import torch

sys.path.append("./tools/checkpoints/aasist")
from models.AASIST import Model as AASIST


def deepfake_detection_model_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"
    if model_path is not None and model_config is not None:
        with open(model_config, "r") as f_json:
            config = json.loads(f_json.read())
            model = AASIST(config["model_config"]).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        if model_tag == "default":
            model_root = "./tools/checkpoints/aasist"
            model_config = os.path.join(model_root, "config/AASIST.conf")
            model_path = os.path.join(model_root, "models/weights/AASIST.pth")

            with open(model_config, "r") as f_json:
                config = json.loads(f_json.read())
                model = AASIST(config["model_config"]).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            raise NotImplementedError
    model.device = device
    return model


def asvspoof_metric(model, pred_x, fs):
    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    pred_x = torch.from_numpy(pred_x).unsqueeze(0).float().to(model.device)
    model.eval()
    with torch.no_grad():
        _, output = model(pred_x)
    output = torch.softmax(output, dim=1)
    output = output.squeeze(0).cpu().numpy()
    return {"asvspoof_score": output[1]}


if __name__ == "__main__":
    a = np.random.random(16000)
    model = deepfake_detection_model_setup(use_gpu=False)
    print("metrics: {}".format(asvspoof_metric(model, a, 16000)))
