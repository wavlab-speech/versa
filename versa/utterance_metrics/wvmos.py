#!/usr/bin/env python3

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

logger = logging.getLogger(__name__)

import librosa
import numpy as np
import torch

try:
    from wvmos import get_wvmos
except ImportError:
    logger.info(
        "WVMOS is not installed. Please use `tools/install_wvmos.sh` to install"
    )
    get_wvmos = None


def wvmos_setup(use_gpu=False):


    if get_wvmos is None:
        raise ModuleNotFoundError(
            "WVMOS is not installed. Please use `tools/install_wvmos.sh` to install"
        )
    
    model = get_wvmos(cuda=use_gpu)

    return model


def wvmos_calculate(model, pred_x, gen_sr):
    """
    Reference:
    https://github.com/AndreevP/wvmos/tree/main
    
    """

    # If gen_sr is not 16000, resample the audio using librosa:
    # This check is also performed in model.processor
    if gen_sr != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=gen_sr, target_sr=16000)
 
    x = model.processor(pred_x, return_tensors="pt", padding=True, sampling_rate=16000).input_values
    
    with torch.no_grad():
        if model.cuda_flag:
            x = x.cuda()
        res = model.forward(x).mean()
    return {
        "wvmos": res.cpu().item()
    }




