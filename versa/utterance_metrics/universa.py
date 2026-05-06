#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
# Mainly adapted from ESPnet-SE (https://github.com/espnet/espnet.git)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import numpy as np
import torch
import soundfile

from versa.audio_utils import resample_audio


def _ensure_torchaudio_legacy_backend_api():
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda *args, **kwargs: None


_ensure_torchaudio_legacy_backend_api()

try:
    from espnet2.bin.universa_inference import UniversaInference
except ImportError:
    UniversaInference = None

from versa.definition import BaseMetric, MetricCategory, MetricMetadata, MetricType

# Global model instances to avoid reloading
_universa_models = {}


def get_universa_model(model_type="noref", cache_dir=None):
    """
    Get or load Universa model instance.

    Args:
        model_type (str): One of "noref", "audioref", "textref", "fullref"

    Returns:
        UniversaInference: Loaded model instance
    """
    model_mapping = {
        "noref": "espnet/universa-wavlm_base_urgent24_multi-metric_noref",
        "audioref": "espnet/universa-wavlm_base_urgent24_multi-metric_audioref",
        "textref": "espnet/universa-wavlm_base_urgent24_multi-metric_textref",
        "fullref": "espnet/universa-wavlm_base_urgent24_multi-metric_fullref",
    }

    if UniversaInference is None:
        raise ImportError("universa requires espnet. Please install espnet and retry")

    cache_key = (model_type, cache_dir)
    if cache_key not in _universa_models:
        if model_type not in model_mapping:
            raise ValueError(
                f"Unknown model_type: {model_type}. "
                f"Choose from {list(model_mapping.keys())}"
            )

        print(f"Loading Universa model: {model_mapping[model_type]}")
        if cache_dir is None:
            _universa_models[cache_key] = UniversaInference.from_pretrained(
                model_mapping[model_type]
            )
        else:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "universa requires espnet_model_zoo. Please install it and retry"
                )
            model_kwargs = ModelDownloader(cachedir=cache_dir).download_and_unpack(
                model_mapping[model_type]
            )
            _universa_models[cache_key] = UniversaInference(**model_kwargs)

    return _universa_models[cache_key]


def audio_preprocess(audio_data, original_sr=None, target_sr=16000):
    """
    Preprocess audio data for Universa inference.

    Args:
        audio_data: numpy array or file path
        original_sr: original sample rate (if audio_data is numpy array)
        target_sr: target sample rate

    Returns:
        tuple: (audio_tensor, audio_lengths_tensor)
    """
    if isinstance(audio_data, str):
        # File path
        audio, sr = soundfile.read(audio_data)
    else:
        # Numpy array
        audio = audio_data
        sr = original_sr or target_sr

    # Ensure audio is 1D
    if audio.ndim > 1:
        audio = audio.mean(axis=1) if audio.shape[1] > 1 else audio[:, 0]

    # Resample if needed
    if sr != target_sr:
        audio = resample_audio(audio, sr, target_sr)

    # Convert to float32 and create tensor
    audio = audio.astype(np.float32)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0)
    audio_lengths = torch.tensor([len(audio_tensor[0])])

    return audio_tensor, audio_lengths


def universa_metric_noref(audio_data, original_sr=None, cache_dir=None):
    """
    Universa no-reference quality assessment.

    Args:
        audio_data: numpy array or file path
        original_sr: original sample rate (if audio_data is numpy array)

    Returns:
        dict: Universa quality metrics with float values and 'universa_' prefix
    """
    model = get_universa_model("noref", cache_dir=cache_dir)
    audio, audio_lengths = audio_preprocess(audio_data, original_sr)

    with torch.no_grad():
        result = model(audio.float(), audio_lengths)

    # Convert to float values with universa_ prefix
    formatted_result = {}
    for key, value in result.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            formatted_result[f"universa_{key}"] = float(
                value.item() if hasattr(value, "item") else value.flatten()[0]
            )
        else:
            formatted_result[f"universa_{key}"] = float(value)

    return formatted_result


def universa_metric_audioref(
    audio_data, ref_audio_data, original_sr=None, ref_sr=None, cache_dir=None
):
    """
    Universa inference with audio reference.

    Args:
        audio_data: numpy array or file path (test audio)
        ref_audio_data: numpy array or file path (reference audio)
        original_sr: original sample rate for test audio
        ref_sr: original sample rate for reference audio

    Returns:
        dict: Universa quality metrics with float values and 'universa_' prefix
    """
    model = get_universa_model("audioref", cache_dir=cache_dir)
    audio, audio_lengths = audio_preprocess(audio_data, original_sr)
    ref_audio, ref_audio_lengths = audio_preprocess(ref_audio_data, ref_sr)

    with torch.no_grad():
        result = model(
            audio.float(),
            audio_lengths,
            ref_audio=ref_audio.float(),
            ref_audio_lengths=ref_audio_lengths,
        )

    # Convert to float values with universa_ prefix
    formatted_result = {}
    for key, value in result.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            formatted_result[f"universa_{key}"] = float(
                value.item() if hasattr(value, "item") else value.flatten()[0]
            )
        else:
            formatted_result[f"universa_{key}"] = float(value)

    return formatted_result


def universa_metric_textref(audio_data, ref_text, original_sr=None, cache_dir=None):
    """
    Universa inference with text reference.

    Args:
        audio_data: numpy array or file path
        ref_text: reference text string
        original_sr: original sample rate (if audio_data is numpy array)

    Returns:
        dict: Universa quality metrics with float values and 'universa_' prefix
    """
    model = get_universa_model("textref", cache_dir=cache_dir)
    audio, audio_lengths = audio_preprocess(audio_data, original_sr)

    with torch.no_grad():
        result = model(audio.float(), audio_lengths, ref_text=ref_text)

    # Convert to float values with universa_ prefix
    formatted_result = {}
    for key, value in result.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            formatted_result[f"universa_{key}"] = float(
                value.item() if hasattr(value, "item") else value.flatten()[0]
            )
        else:
            formatted_result[f"universa_{key}"] = float(value)

    return formatted_result


def universa_metric_fullref(
    audio_data,
    ref_audio_data,
    ref_text,
    original_sr=None,
    ref_sr=None,
    cache_dir=None,
):
    """
    Universa inference with both audio and text reference.

    Args:
        audio_data: numpy array or file path (test audio)
        ref_audio_data: numpy array or file path (reference audio)
        ref_text: reference text string
        original_sr: original sample rate for test audio
        ref_sr: original sample rate for reference audio

    Returns:
        dict: Universa quality metrics with float values and 'universa_' prefix
    """
    model = get_universa_model("fullref", cache_dir=cache_dir)
    audio, audio_lengths = audio_preprocess(audio_data, original_sr)
    ref_audio, ref_audio_lengths = audio_preprocess(ref_audio_data, ref_sr)

    with torch.no_grad():
        result = model(
            audio.float(),
            audio_lengths,
            ref_audio=ref_audio.float(),
            ref_audio_lengths=ref_audio_lengths,
            ref_text=ref_text,
        )

    # Convert to float values with universa_ prefix
    formatted_result = {}
    for key, value in result.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            formatted_result[f"universa_{key}"] = float(
                value.item() if hasattr(value, "item") else value.flatten()[0]
            )
        else:
            formatted_result[f"universa_{key}"] = float(value)

    return formatted_result


def universa_metric(
    audio_data,
    ref_audio=None,
    ref_text=None,
    original_sr=16000,
    ref_sr=None,
    cache_dir=None,
):
    """
    Universal Universa metric function that automatically selects the appropriate model
    based on available references.

    Args:
        audio_data: numpy array or file path (test audio)
        ref_audio: numpy array or file path (reference audio, optional)
        ref_text: reference text string (optional)
        original_sr: original sample rate for test audio
        ref_sr: original sample rate for reference audio

    Returns:
        dict: Universa quality metrics
    """
    if ref_audio is not None and ref_text is not None:
        # Full reference (both audio and text)
        return universa_metric_fullref(
            audio_data, ref_audio, ref_text, original_sr, ref_sr, cache_dir=cache_dir
        )
    elif ref_audio is not None:
        # Audio reference only
        return universa_metric_audioref(
            audio_data, ref_audio, original_sr, ref_sr, cache_dir=cache_dir
        )
    elif ref_text is not None:
        # Text reference only
        return universa_metric_textref(
            audio_data, ref_text, original_sr, cache_dir=cache_dir
        )
    else:
        # No reference
        return universa_metric_noref(audio_data, original_sr, cache_dir=cache_dir)


class UniversaMetric(BaseMetric):
    """Uni-VERSA speech assessment metric."""

    def _setup(self):
        self.model_type = self.config.get(
            "model_type", self.config.get("model_tag", "auto")
        )
        if self.model_type == "default":
            self.model_type = "noref"
        self.cache_dir = self.config.get("cache_dir")

    def compute(self, predictions, references=None, metadata=None):
        if predictions is None:
            raise ValueError("Predicted signal must be provided")

        metadata = metadata or {}
        fs = metadata.get("sample_rate", 16000)
        ref_sr = metadata.get("reference_sample_rate", fs)
        ref_text = metadata.get("text")
        if isinstance(references, str):
            ref_text = references
            ref_audio = None
        else:
            ref_audio = references

        model_type = self.model_type
        if model_type == "noref":
            return universa_metric_noref(predictions, fs, cache_dir=self.cache_dir)
        if model_type == "audioref":
            if ref_audio is None:
                raise ValueError("Audio reference must be provided")
            return universa_metric_audioref(
                predictions, ref_audio, fs, ref_sr, cache_dir=self.cache_dir
            )
        if model_type == "textref":
            if ref_text is None:
                raise ValueError("Text reference must be provided")
            return universa_metric_textref(
                predictions, ref_text, fs, cache_dir=self.cache_dir
            )
        if model_type == "fullref":
            if ref_audio is None:
                raise ValueError("Audio reference must be provided")
            if ref_text is None:
                raise ValueError("Text reference must be provided")
            return universa_metric_fullref(
                predictions,
                ref_audio,
                ref_text,
                fs,
                ref_sr,
                cache_dir=self.cache_dir,
            )

        metric_kwargs = {
            "ref_audio": ref_audio,
            "ref_text": ref_text,
            "original_sr": fs,
            "ref_sr": ref_sr,
        }
        if self.cache_dir is not None:
            metric_kwargs["cache_dir"] = self.cache_dir
        return universa_metric(predictions, **metric_kwargs)

    def get_metadata(self):
        return _universa_metadata()


def _universa_metadata():
    return MetricMetadata(
        name="universa",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.DICT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["espnet2", "torch", "librosa", "numpy", "soundfile"],
        description="Uni-VERSA speech assessment metrics",
        paper_reference="https://arxiv.org/abs/2505.20741",
        implementation_source=(
            "https://huggingface.co/collections/espnet/"
            "universa-6834e7c0a28225bffb6e2526"
        ),
    )


def register_universa_metric(registry):
    """Register Uni-VERSA with the registry."""
    registry.register(
        UniversaMetric,
        _universa_metadata(),
        aliases=[
            "uni_versa",
            "universal_speech_assessment",
            "universa_noref",
            "universa_audioref",
            "universa_textref",
            "universa_fullref",
        ],
    )


# Debug code
if __name__ == "__main__":
    # Generate test audio
    test_audio = np.random.random(16000)
    ref_audio = np.random.random(16000)
    ref_text = "This is a test reference text"

    print("=== Universa Metrics Tests ===")

    # Test no-reference
    try:
        print("\n1. Testing no-reference Universa...")
        noref_result = universa_metric_noref(test_audio, 16000)
        print("No-ref result:", noref_result)
    except Exception as e:
        print(f"No-ref test failed: {e}")

    # Test with audio reference
    try:
        print("\n2. Testing audio-reference Universa...")
        audioref_result = universa_metric_audioref(test_audio, ref_audio, 16000, 16000)
        print("Audio-ref result:", audioref_result)
    except Exception as e:
        print(f"Audio-ref test failed: {e}")

    # Test with text reference
    try:
        print("\n3. Testing text-reference Universa...")
        textref_result = universa_metric_textref(test_audio, ref_text, 16000)
        print("Text-ref result:", textref_result)
    except Exception as e:
        print(f"Text-ref test failed: {e}")

    # Test with full reference
    try:
        print("\n4. Testing full-reference Universa...")
        fullref_result = universa_metric_fullref(
            test_audio, ref_audio, ref_text, 16000, 16000
        )
        print("Full-ref result:", fullref_result)
    except Exception as e:
        print(f"Full-ref test failed: {e}")

    # Test universal function
    try:
        print("\n5. Testing universal Universa function...")

        # Auto-select no-ref
        auto_noref = universa_metric(test_audio, original_sr=16000)
        print("Auto no-ref:", auto_noref)

        # Auto-select audio-ref
        auto_audioref = universa_metric(
            test_audio, ref_audio=ref_audio, original_sr=16000, ref_sr=16000
        )
        print("Auto audio-ref:", auto_audioref)

        # Auto-select text-ref
        auto_textref = universa_metric(test_audio, ref_text=ref_text, original_sr=16000)
        print("Auto text-ref:", auto_textref)

        # Auto-select full-ref
        auto_fullref = universa_metric(
            test_audio,
            ref_audio=ref_audio,
            ref_text=ref_text,
            original_sr=16000,
            ref_sr=16000,
        )
        print("Auto full-ref:", auto_fullref)

    except Exception as e:
        print(f"Universal function test failed: {e}")
