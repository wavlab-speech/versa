#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for emotion similarity metrics using EMO2VEC."""

import logging
import os
from pathlib import Path

import librosa
import numpy as np

logger = logging.getLogger(__name__)

try:
    import emo2vec_versa
    from emo2vec_versa.emo2vec_class import EMO2VEC
except ImportError:
    logger.info(
        "emo2vec is not installed. Please install the package via "
        "`tools/install_emo2vec.sh`"
    )
    EMO2VEC = None


def emo2vec_setup(model_tag="default", model_path=None, use_gpu=False):
    """Set up EMO2VEC model for emotion embedding extraction.

    Args:
        model_tag (str, optional): Model tag. Defaults to "default".
        model_path (str, optional): Path to model weights. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        EMO2VEC: The loaded model.

    Raises:
        ImportError: If emo2vec_versa is not installed.
        ValueError: If model_tag is unknown.
        FileNotFoundError: If model file is not found.
    """
    if EMO2VEC is None:
        raise ImportError(
            "emo2vec_versa not found. Please install from tools/installers"
        )

    if model_path is not None:
        model = EMO2VEC(model_path, use_gpu=use_gpu)
    else:
        if model_tag == "default" or model_tag == "base":
            model_path = (
                Path(os.path.abspath(emo2vec_versa.__file__)).parent
                / "emotion2vec_base.pt"
            )
        else:
            raise ValueError(f"Unknown model_tag for emo2vec: {model_tag}")

        # check if model exists
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = EMO2VEC(checkpoint_dir=str(model_path), use_gpu=use_gpu)
    return model


def emo_sim(model, pred_x, gt_x, fs):
    """Calculate emotion similarity between two audio samples.

    Args:
        model (EMO2VEC): The loaded EMO2VEC model.
        pred_x (np.ndarray): Predicted audio signal.
        gt_x (np.ndarray): Ground truth audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the emotion similarity score.
    """
    # NOTE(jiatong): only work for 16000 Hz
    if fs != 16000:
        gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=16000)
        pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

    embedding_gen = model.extract_feature(pred_x, fs=16000)
    embedding_gt = model.extract_feature(gt_x, fs=16000)
    similarity = np.dot(embedding_gen, embedding_gt) / (
        np.linalg.norm(embedding_gen) * np.linalg.norm(embedding_gt)
    )
    return {"emotion_similarity": similarity}


if __name__ == "__main__":
    a = np.random.random(16000)
    b = np.random.random(16000)
    model = emo2vec_setup()
    print(f"metrics: {emo_sim(model, a, b, 16000)}")
