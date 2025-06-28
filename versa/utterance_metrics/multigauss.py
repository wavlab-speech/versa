#!/usr/bin/env python3

# Copyright 2025 Jionghao Han
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
#
# This file includes code adapted from the MultiGauss project:
#   https://github.com/fcumlin/MultiGauss
#   Copyright (c) 2025 Fredrik Cumlin
#   Licensed under the MIT License

import logging

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
import librosa
import numpy as np
import torch
import torchaudio


try:
    import gin
    sys.path.append(str(Path(__file__).parent.parent.parent / "tools/checkpoints/multigauss"))
    import model as model_lib
    from train import TrainingLoop
except ImportError:
    raise ImportError(
        "MultiGauss is not set up. Please install the package via "
        "`tools/install_multigauss.sh`"
    )


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
    model_folder = Path(f"./tools/checkpoints/multigauss/runs/{model_tag}")
    print(f"Loading model from {model_folder}")
    gin.clear_config()
    gin.external_configurable(TrainingLoop)
    gin.parse_config_file(model_folder / "config.gin", skip_unknown=True)
    ssl_model_layer = gin.query_parameter("TrainingLoop.ssl_layer")
    bundle = torchaudio.pipelines.WAV2VEC2_XLSR_2B
    ssl_model = bundle.get_model(dl_kwargs=dict(model_dir=str(Path(cache_dir) / "torchaudio"))).to(device=device)
    ssl_model.eval()
    ssl_model_extract = lambda x: ssl_model.extract_features(x)[0][ssl_model_layer]
    multigauss_model = model_lib.ProjectionHead()
    state_dict = torch.load(
        model_folder / "model_best_state_dict.pt",
        map_location=device,
        weights_only=True
    )
    multigauss_model.load_state_dict(state_dict)
    multigauss_model.eval()
    return {
        "ssl_model_extract": ssl_model_extract,
        "multigauss_model": multigauss_model,
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
    if fs != 16000:
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    with torch.no_grad():
        feature = models["ssl_model_extract"](pred_x).squeeze().T
        print(f"{feature.shape=}")
        mean_prediction, covariance_prediction = models["multigauss_model"](feature.unsqueeze(0))
    return {
        "multigauss_mean": mean_prediction,
        "multigauss_covariance": covariance_prediction,
    }

if __name__ == "__main__":
    a = np.random.random(16000)
    model = multigauss_model_setup(use_gpu=True if torch.cuda.is_available() else False)
    print(f"MultiGauss metrics: {multigauss_metric(model, a, 16000)}")
