#!/usr/bin/env python3

# Copyright 2025 Jionghao Han
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# This file includes code adapted from the MultiGauss project:
#   https://github.com/fcumlin/MultiGauss
#   Copyright (c) 2025 Fredrik Cumlin
#   Licensed under the MIT License


r"""
Notes from the MultiGauss project (Fredrik Cumlin):
The model operates at 16 kHz sample rate and on signals of 10 s duration, hence,
all audio is resampled to 16 kHz and repeated or cropped to 10 s before
processing. Note that the sample rate implies that no energy with frequencies
above 8 kHz are seen by the model.
"""

import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
import librosa
import numpy as np
import torch
import torchaudio

MULTIGAUSS_DIR = (
    Path(__file__).parent.parent.parent / "tools" / "checkpoints" / "multigauss"
)
print(f"MULTIGAUSS_DIR: {MULTIGAUSS_DIR}")
try:
    import gin

    sys.path.append(str(MULTIGAUSS_DIR))
    import model as model_lib
    from train import TrainingLoop
except ImportError:
    raise ImportError(
        "MultiGauss is not set up. Please install the package via "
        "`tools/install_multigauss.sh`"
    )


def _repeat_and_crop_to_length(
    waveform: torch.Tensor,
    target_length: int = 160_000,
) -> torch.Tensor:
    """Repeates or crops the waveform to give it the target length."""
    current_length = waveform.shape[-1]
    if current_length < target_length:
        num_repeats = target_length // current_length + 1
        waveform = waveform.repeat(1, num_repeats)
    return waveform[:, :target_length]


def multigauss_model_setup(
    model_tag="probabilistic", cache_dir="versa_cache", use_gpu=False
):
    """Setup multigauss model.

    Args:
        model_tag (str): Model tag. Defaults to "probabilistic". Can be "probabilistic" or "non_probabilistic".
        cache_dir (str): Cache directory. Defaults to "versa_cache".
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        models: The loaded models.
    """
    device = "cuda" if use_gpu else "cpu"
    model_folder = MULTIGAUSS_DIR / "runs" / model_tag
    print(f"Loading model from {model_folder}")
    gin.clear_config()
    gin.external_configurable(TrainingLoop)
    gin.parse_config_file(model_folder / "config.gin", skip_unknown=True)
    ssl_model_layer = gin.query_parameter("TrainingLoop.ssl_layer")
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
    ssl_model = bundle.get_model(
        dl_kwargs=dict(model_dir=str(Path(cache_dir) / "torchaudio"))
    ).to(device=device)
    ssl_model.eval()
    ssl_model_extract = lambda x: ssl_model.extract_features(x)[0][ssl_model_layer]
    multigauss_model = model_lib.ProjectionHead(in_shape=(1920, 499))
    state_dict = torch.load(
        model_folder / "model_best_state_dict.pt",
        map_location=device,
        weights_only=True,
    )
    multigauss_model.load_state_dict(state_dict)
    multigauss_model = multigauss_model.to(device=device)
    multigauss_model.eval()
    return {
        "ssl_model_extract": ssl_model_extract,
        "multigauss_model": multigauss_model,
        "device": device,
    }


def multigauss_metric(models, pred_x, fs):
    """Calculate multigauss score for audio.

    Args:
        models (dict): The loaded models.
        pred_x (np.ndarray): Audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the multigauss score.
    """
    pred_x = torch.from_numpy(pred_x).float()
    if fs != 16000:
        pred_x = torchaudio.transforms.Resample(orig_freq=fs, new_freq=16000)(pred_x)
    pred_x = _repeat_and_crop_to_length(
        pred_x,
        target_length=160_000,  # Training was done with 10 s of audio (16 kHz).
    )

    with torch.no_grad():
        feature = (
            models["ssl_model_extract"](pred_x.to(device=models["device"])).squeeze().T
        )
        mean_prediction, covariance_prediction = models["multigauss_model"](
            feature.unsqueeze(0)
        )
    return {
        "multigauss_mos": mean_prediction[0][0].item(),
        "multigauss_noi": mean_prediction[0][1].item(),
        "multigauss_col": mean_prediction[0][2].item(),
        "multigauss_dis": mean_prediction[0][3].item(),
        "multigauss_loud": mean_prediction[0][4].item(),
        "multigauss_covariance": covariance_prediction[0].cpu().numpy(), # ["mos", "noi", "col", "dis", "loud"]
    }


if __name__ == "__main__":
    a = np.random.random(16000)
    model = multigauss_model_setup(use_gpu=True if torch.cuda.is_available() else False)
    print(f"MultiGauss metrics: {multigauss_metric(model, a, 16000)}")
