#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import librosa
import soundfile as sf


def universa_model_setup(model_tag="default", use_gpu=False):
    """
    Setup Universa model for inference.

    Args:
        model_tag (str): Model tag to use. Options:
            - "default": espnet/universa-wavlm_base_urgent24_multi-metric_noref
            - "audioref": espnet/universa-wavlm_base_urgent24_multi-metric_audioref
            - "textref": espnet/universa-wavlm_base_urgent24_multi-metric_textref
            - "fullref": espnet/universa-wavlm_base_urgent24_multi-metric_fullref
        use_gpu (bool): Whether to use GPU for inference

    Returns:
        UniversaInference: Loaded model for inference
    """
    try:
        from espnet2.bin.universa_inference import UniversaInference
    except ImportError:
        raise ImportError(
            "Please install espnet and espnet_model_zoo to use Universa metric: "
            "pip install espnet espnet_model_zoo"
        )

    # Map model tags to actual model names
    model_mapping = {
        "default": "espnet/universa-wavlm_base_urgent24_multi-metric_noref",
        "noref": "espnet/universa-wavlm_base_urgent24_multi-metric_noref",
        "audioref": "espnet/universa-wavlm_base_urgent24_multi-metric_audioref",
        "textref": "espnet/universa-wavlm_base_urgent24_multi-metric_textref",
        "fullref": "espnet/universa-wavlm_base_urgent24_multi-metric_fullref",
    }

    if model_tag not in model_mapping:
        raise ValueError(
            f"Unknown model_tag: {model_tag}. Available options: {list(model_mapping.keys())}"
        )

    model_name = model_mapping[model_tag]

    # Load the model
    model = UniversaInference.from_pretrained(model_name)

    # Set device
    if use_gpu and torch.cuda.is_available():
        model = model.cuda()

    return model


def audio_preprocess(audio, fs, target_fs=16000):
    """
    Preprocess audio for Universa model.

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


def universa_metric(model, pred_x, fs, gt_x=None, ref_text=None):
    """
    Compute Universa metrics for audio evaluation.

    Args:
        model: Universa model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x
        gt_x (np.ndarray, optional): Reference audio signal
        ref_text (str, optional): Reference text

    Returns:
        dict: Dictionary containing Universa metric scores
    """
    # Preprocess the prediction audio
    pred_audio, pred_lengths = audio_preprocess(pred_x, fs)

    # Move to same device as model
    if next(model.model.parameters()).is_cuda:
        pred_audio = pred_audio.cuda()
        pred_lengths = pred_lengths.cuda()

    # Prepare reference audio if provided
    ref_audio = None
    ref_lengths = None
    if gt_x is not None:
        ref_audio, ref_lengths = audio_preprocess(gt_x, fs)
        if next(model.model.parameters()).is_cuda:
            ref_audio = ref_audio.cuda()
            ref_lengths = ref_lengths.cuda()

    # Run inference based on available references
    if ref_audio is not None and ref_text is not None:
        # Both audio and text reference
        results = model(
            pred_audio.float(),
            pred_lengths,
            ref_audio=ref_audio.float(),
            ref_audio_lengths=ref_lengths,
            ref_text=ref_text,
        )
    elif ref_audio is not None:
        # Only audio reference
        results = model(
            pred_audio.float(),
            pred_lengths,
            ref_audio=ref_audio.float(),
            ref_audio_lengths=ref_lengths,
        )
    elif ref_text is not None:
        # Only text reference
        results = model(pred_audio.float(), pred_lengths, ref_text=ref_text)
    else:
        # No reference
        results = model(pred_audio.float(), pred_lengths)

    # Convert results to dictionary format
    if isinstance(results, dict):
        # If results is already a dictionary, use it directly
        universa_scores = {
            "universa_" + key: value[0][0] for key, value in results.items()
        }
    else:
        # If results is a tensor or other format, convert to dictionary
        # This might need adjustment based on actual Universa output format
        universa_scores = {"universa_score": float(results.cpu().numpy())}

    return universa_scores


def universa_noref_metric(model, pred_x, fs):
    """
    Compute Universa metrics without reference.

    Args:
        model: Universa model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x

    Returns:
        dict: Dictionary containing Universa metric scores
    """
    return universa_metric(model, pred_x, fs)


def universa_audioref_metric(model, pred_x, fs, gt_x):
    """
    Compute Universa metrics with audio reference.

    Args:
        model: Universa model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x
        gt_x (np.ndarray): Reference audio signal

    Returns:
        dict: Dictionary containing Universa metric scores
    """
    return universa_metric(model, pred_x, fs, gt_x=gt_x)


def universa_textref_metric(model, pred_x, fs, ref_text):
    """
    Compute Universa metrics with text reference.

    Args:
        model: Universa model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x
        ref_text (str): Reference text

    Returns:
        dict: Dictionary containing Universa metric scores
    """
    return universa_metric(model, pred_x, fs, ref_text=ref_text)


def universa_fullref_metric(model, pred_x, fs, gt_x, ref_text):
    """
    Compute Universa metrics with both audio and text reference.

    Args:
        model: Universa model for inference
        pred_x (np.ndarray): Audio signal to be evaluated
        fs (int): Sampling rate of pred_x
        gt_x (np.ndarray): Reference audio signal
        ref_text (str): Reference text

    Returns:
        dict: Dictionary containing Universa metric scores
    """
    return universa_metric(model, pred_x, fs, gt_x=gt_x, ref_text=ref_text)


if __name__ == "__main__":
    # Test the implementation
    print("Testing Universa metric implementation...")

    # Generate test audio
    fs = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(fs * duration))
    test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    try:
        # Test no-reference model
        print("Testing no-reference model...")
        model = universa_model_setup(model_tag="noref", use_gpu=False)
        scores = universa_noref_metric(model, test_audio, fs)
        print(f"No-reference scores: {scores}")

        # Test with audio reference
        print("Testing audio reference model...")
        model_audioref = universa_model_setup(model_tag="audioref", use_gpu=False)
        scores_audioref = universa_audioref_metric(
            model_audioref, test_audio, fs, test_audio
        )
        print(f"Audio reference scores: {scores_audioref}")

        # Test with text reference
        print("Testing text reference model...")
        model_textref = universa_model_setup(model_tag="textref", use_gpu=False)
        scores_textref = universa_textref_metric(
            model_textref, test_audio, fs, "test text"
        )
        print(f"Text reference scores: {scores_textref}")

        print("All tests completed successfully!")

    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if espnet is not installed.")
