#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for evaluating audio using AudioBox Aesthetics models."""

import json
import os

import numpy as np

try:
    import audiobox_aesthetics.infer
    import audiobox_aesthetics.utils
except ImportError:
    audiobox_aesthetics = None


def audiobox_aesthetics_setup(
    model_path=None,
    batch_size=1,
    precision="bf16",
    cache_dir="versa_cache/audiobox",
    use_huggingface=True,
    use_gpu=False,
):
    """Set up the AudioBox Aesthetics model for inference.

    Args:
        model_path (str, optional): Path to model weights. Defaults to None.
        batch_size (int, optional): Batch size for inference. Defaults to 1.
        precision (str, optional): Precision for inference. Defaults to "bf16".
        cache_dir (str, optional): Directory to cache model. Defaults to "versa_cache/audiobox".
        use_huggingface (bool, optional): Whether to use Hugging Face. Defaults to True.
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        AesWavlmPredictorMultiOutput: The loaded model.

    Raises:
        ImportError: If audiobox_aesthetics is not installed.
    """
    if audiobox_aesthetics is None:
        raise ImportError(
            "Please install with tools/install_audiobox-aesthetics.sh first."
        )

    device = "cuda" if use_gpu else "cpu"

    if model_path is None:
        if use_huggingface:
            model_path = audiobox_aesthetics.utils.load_model(model_path)
        else:
            os.makedirs(cache_dir, exist_ok=True)
            model_path = os.path.join(
                cache_dir, audiobox_aesthetics.utils.DEFAULT_CKPT_FNAME
            )
            model_url = audiobox_aesthetics.utils.DEFAULT_S3_URL
            if not os.path.exists(model_path):
                print(f"Downloading model from {model_url} to {model_path}")
                audiobox_aesthetics.utils.download_file(model_url, model_path)

    predictor = audiobox_aesthetics.infer.AesWavlmPredictorMultiOutput(
        checkpoint_pth=model_path,
        device=device,
        batch_size=batch_size,
        precision=precision,
    )
    return predictor


def audiobox_aesthetics_score(model, pred_x, fs):
    """Calculate AudioBox Aesthetics scores for audio.

    Args:
        model (AesWavlmPredictorMultiOutput): The loaded model.
        pred_x (np.ndarray): Audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the AudioBox Aesthetics scores.
    """
    output = json.loads(model.forward_versa([(pred_x, fs)])[0])
    output = {"audiobox_aesthetics_" + k: v for k, v in output.items()}
    return output


if __name__ == "__main__":
    a = np.random.random(16000)
    model = audiobox_aesthetics_setup()
    print(f"metrics: {audiobox_aesthetics_score(model, a, 16000)}")
