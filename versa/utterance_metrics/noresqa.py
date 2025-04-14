#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import logging
import os
import sys

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

from urllib.request import urlretrieve

import torch.nn as nn

base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../tools/Noresqa")
)
sys.path.insert(0, base_path)


try:
    import fairseq
except ImportError:
    logger.info(
        "fairseq is not installed. Please use `tools/install_fairseq.sh` to install"
    )

try:
    from model import NORESQA
    from utils import (
        feats_loading,
        model_prediction_noresqa,
        model_prediction_noresqa_mos,
    )

except ImportError:
    logger.info(
        "noresqa is not installed. Please use `tools/install_noresqa.sh` to install"
    )
    Noresqa = None


def noresqa_model_setup(model_tag="default", metric_type=0, cache_dir="versa_cache/noresqa_model", use_gpu=False):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if model_tag == "default":

        if not os.path.isdir(cache_dir):
            print("Creating checkpoints directory")
            os.makedirs(cache_dir)

        url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        w2v_path = os.path.join(cache_dir, "wav2vec_small.pt")
        if not os.path.isfile(w2v_path):
            print("Downloading wav2vec 2.0 started")
            urlretrieve(url_w2v, w2v_path)
            print("wav2vec 2.0 download completed")

        model = NORESQA(
            output=40, output2=40, metric_type=metric_type, config_path=w2v_path
        )

        if metric_type == 0:
            model_checkpoint_path = "{}/models/model_noresqa.pth".format(base_path)
            state = torch.load(model_checkpoint_path, map_location="cpu")["state_base"]

        elif metric_type == 1:
            model_checkpoint_path = "{}/models/model_noresqa_mos.pth".format(base_path)
            state = torch.load(model_checkpoint_path, map_location="cpu")["state_dict"]

        pretrained_dict = {}
        for k, v in state.items():
            if "module" in k:
                pretrained_dict[k.replace("module.", "")] = v
            else:
                pretrained_dict[k] = v
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(pretrained_dict)

        # change device as needed
        model.to(device)
        model.device = device
        model.eval()

        sfmax = nn.Softmax(dim=1)

    else:
        raise NotImplementedError

    return model


def noresqa_metric(model, gt_x, pred_x, fs, metric_type=1):
    # NOTE(hyejin): only work for 16000 Hz
    gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
    pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)
    nmr_feat, test_feat = feats_loading(pred_x, gt_x, noresqa_or_noresqaMOS=metric_type)
    test_feat = torch.from_numpy(test_feat).float().to(model.device).unsqueeze(0)
    nmr_feat = torch.from_numpy(nmr_feat).float().to(model.device).unsqueeze(0)

    with torch.no_grad():
        if metric_type == 0:
            noresqa_pout, noresqa_qout = model_prediction_noresqa(
                test_feat, nmr_feat, model
            )
            return {"noresqa_score": noresqa_pout}
        elif metric_type == 1:
            mos_score = model_prediction_noresqa_mos(test_feat, nmr_feat, model)
            return {"noresqa_score": mos_score}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    model = noresqa_model_setup(use_gpu=True)
    print("metrics: {}".format(noresqa_metric(model, a, b, 16000)))
