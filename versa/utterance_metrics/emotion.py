#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

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
        "emo2vec is not installed. Please install the package via `tools/install_emo2vec.sh`"
    )
    EMO2VEC = None


def emo2vec_setup(model_tag="default", model_path=None, use_gpu=False):
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
            raise ValueError("Unknown model_tag for emo2vec: {}".format(model_tag))

        # check if model exists
        if not model_path.exists():
            raise FileNotFoundError("Model file not found: {}".format(model_path))

        model = EMO2VEC(checkpoint_dir=str(model_path), use_gpu=use_gpu)
    return model


def emo_sim(model, pred_x, gt_x, fs):
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
    print("metrics: {}".format(emo_sim(model, a, b, 16000)))
