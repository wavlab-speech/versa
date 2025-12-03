#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import librosa
import soundfile as sf


def arecho_model_setup(model_tag="default", use_gpu=False):
    """
    Setup ARECHO model for inference.

    Args:
        model_tag (str): Model tag to use. Options:
            - "default": espnet/arecho_base_v0
            - "base_v0": espnet/arecho_base_v0
            - "scale_v0": espnet/arecho_scale_v0
            - "base_v0.1-large-decoder": espnet/arecho_base_v0.1-large-decoder
            - "scale_v0.1-larger-decoder": espnet/arecho_scale_v0.1-larger-decoder
        use_gpu (bool): Whether to use GPU for inference

    Returns:
        UniversaInference: Loaded ARECHO model for inference
    """
    try:
        from espnet2.bin.universa_inference import UniversaInference
    except ImportError:
        raise ImportError(
            "Please install espnet and espnet_model_zoo to use ARECHO metric: "
            "pip install espnet espnet_model_zoo"
        )

    # Map model tags to actual model names
    model_mapping = {
        "default": "espnet/arecho_base_v0",
        "base_v0": "espnet/arecho_base_v0",
        "scale_v0": "espnet/arecho_scale_v0",
        "base_v0.1-large-decoder": "espnet/arecho_base_v0.1-large-decoder",
        "scale_v0.1-larger-decoder": "espnet/arecho_scale_v0.1-larger-decoder",
    }

    if model_tag not in model_mapping:
        raise ValueError(
            f"Unknown model_tag: {model_tag}. Available options: {list(model_mapping.keys())}"
        )

    model_name = model_mapping[model_tag]

    # Set device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"

    # Load the model
    model = UniversaInference.from_pretrained(model_name, device=device)

    return model


def audio_preprocess(audio, fs, target_fs=16000):
    """
    Preprocess audio for ARECHO model.

    Args:
        audio (np.ndarray): Input audio signal
        fs (int): Original sampling rate
        target_fs (int): Target sampling rate (default: 16000)

    Returns:
        tuple: (processed_audio_tensor, audio_lengths_tensor)
    """
    # Resample to target sampling rate
    if fs != target_fs:
        audio = librosa.resample(audio, orig_sr=fs, target_sr=target_fs)

    # Convert to float32
    audio = audio.astype(np.float32)

    # Convert to tensor and add batch dimension
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    audio_lengths = torch.tensor([len(audio)])

    return audio_tensor, audio_lengths


def arecho_metric(model, pred_x, fs):
    """
    Compute ARECHO metrics for audio evaluation.

    Args:
        model: ARECHO model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x

    Returns:
        dict: Dictionary containing ARECHO metric scores
    """
    # Preprocess the prediction audio
    pred_audio, pred_lengths = audio_preprocess(pred_x, fs)

    # Move to same device as model
    if next(model.model.parameters()).is_cuda:
        pred_audio = pred_audio.cuda()
        pred_lengths = pred_lengths.cuda()

    ref_audio = torch.zeros(1, 8000, dtype=torch.float32)
    ref_lengths = torch.tensor([8000])
    if next(model.model.parameters()).is_cuda:
        ref_audio = ref_audio.cuda()
        ref_lengths = ref_lengths.cuda()

    # Run inference
    results = model(
        pred_audio.float(),
        pred_lengths,
        ref_audio=ref_audio.float(),
        ref_audio_lengths=ref_lengths,
    )

    # Convert results to dictionary format
    if isinstance(results, dict):
        # If results is already a dictionary, use it directly
        arecho_scores = {"arecho_" + key: value[0] for key, value in results.items()}
    else:
        # If results is a tensor or other format, convert to dictionary
        arecho_scores = {"arecho_score": float(results.cpu().numpy())}

    return arecho_scores


def arecho_noref_metric(model, pred_x, fs):
    """
    Compute ARECHO metrics without reference (uses placeholder reference).

    Args:
        model: ARECHO model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x

    Returns:
        dict: Dictionary containing ARECHO metric scores
    """
    return arecho_metric(model, pred_x, fs)


if __name__ == "__main__":
    # Test the implementation
    print("Testing ARECHO metric implementation...")

    # Generate test audio
    fs = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Test no-reference model
    print("Testing no-reference model...")
    model = arecho_model_setup(model_tag="base_v0", use_gpu=False)
    scores = arecho_noref_metric(model, test_audio, fs)
    print(f"No-reference scores: {scores}")
