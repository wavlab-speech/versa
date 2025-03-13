#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch

try:
    from scoreq_versa import Scoreq
except ImportError:
    logger.info(
        "scoreq is not installed. Please use `tools/install_scoreq.sh` to install"
    )
    Scoreq = None


def scoreq_nr_setup(
    data_domain="synthetic",
    cache_dir="versa_cache/scoreq_pt-models",
    use_gpu=False,
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if Scoreq is None:
        raise ModuleNotFoundError(
            "scoreq is not installed. Please use `tools/install_scoreq.sh` to install"
        )

    return Scoreq(
        data_domain=data_domain, mode="nr", cache_dir=cache_dir, device=device
    )


def scoreq_ref_setup(
    data_domain="synthetic",
    cache_dir="./scoreq_pt-models",
    use_gpu=False,
):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if Scoreq is None:
        raise ModuleNotFoundError(
            "scoreq is not installed. Please use `tools/install_scoreq.sh` to install"
        )

    return Scoreq(
        data_domain=data_domain, mode="ref", cache_dir=cache_dir, device=device
    )


def scoreq_nr(model, pred_x, fs):
    # NOTE(jiatong): current model only have 16k options
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {"scoreq_nr": model.predict(test_path=pred_x, ref_path=None)}


def scoreq_ref(model, pred_x, gt_x, fs):
    # NOTE(jiatong): current model only have 16k options
    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {"scoreq_ref": model.predict(test_path=pred_x, ref_path=gt_x)}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    nr_model = scoreq_nr_setup(use_gpu=True)
    ref_model = scoreq_ref_setup(use_gpu=True)
    fs = 16000
    metric_nr = scoreq_nr(nr_model, a, fs)
    metric_ref = scoreq_ref(ref_model, a, b, fs)
    print(metric_nr)
    print(metric_ref)
