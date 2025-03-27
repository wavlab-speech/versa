#!/usr/bin/env python3

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch

try:
    from nomad_versa import Nomad
except ImportError:
    logger.info(
        "nomad is not installed. Please use `tools/install_nomad.sh` to install"
    )
    Nomad = None


def nomad_setup(use_gpu=False, cache_dir="./nomad_pt-models"):
    if use_gpu:
        device = "cuda"
    else:
        device = "cpu"

    if Nomad is None:
        raise ModuleNotFoundError(
            "nomad is not installed. Please use `tools/install_nomad.sh` to install"
        )

    return Nomad(device=device)


def nomad(model, pred_x, gt_x, fs):
    """
    Reference:
    A. Ragano, J. Skoglund and A. Hines,
    "NOMAD: Unsupervised Learning of Perceptual Embeddings For Speech Enhancement and Non-Matching Reference Audio Quality Assessment,"
    ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Seoul, Korea, Republic of, 2024, pp. 1011-1015
    Codebase:
    https://github.com/alessandroragano/nomad

    """

    # NOTE(hyejin): current model only have 16k options
    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    return {
        "nomad": model.predict(nmr=gt_x, deg=pred_x),
    }


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    nomad_model = nomad_setup(use_gpu=True)
    fs = 16000
    nomad_score = nomad(nomad_model, a, b, fs)
    print(nomad_score)
