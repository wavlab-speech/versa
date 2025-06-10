import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.discrete_speech import (
    discrete_speech_setup,
    discrete_speech_metric,
)

# Reuse the same helper functions from your STOI test
def generate_fixed_wav(
    filename, duration=1.0, sample_rate=16000, base_freq=150, envelope_func=None
):
    """
    Generate a deterministic WAV file with a modulated sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if envelope_func is None:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    else:
        envelope = envelope_func(t)
    audio = envelope * np.sin(2 * np.pi * base_freq * t)
    amplitude = np.iinfo(np.int16).max
    data = (audio * amplitude).astype(np.int16)
    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def load_wav_as_array(wav_path, sample_rate=16000):
    """
    Load a WAV file and convert it to a NumPy array scaled to [-1, 1].
    """
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        audio_data = wf.readframes(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    return audio_array / np.iinfo(np.int16).max


@pytest.fixture(scope="session")
def fixed_audio_wav(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio.wav"
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=16000, base_freq=150)
    return audio_file


@pytest.fixture(scope="session")
def fixed_ground_truth_wav(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    gt_file = tmp_dir / "fixed_ground_truth.wav"
    # Use a different base frequency for ground truth (e.g. 300 Hz) to simulate a mismatch.
    generate_fixed_wav(gt_file, duration=1.0, sample_rate=16000, base_freq=300)
    return gt_file


@pytest.fixture(scope="session")
def fixed_audio(fixed_audio_wav):
    return load_wav_as_array(fixed_audio_wav)


@pytest.fixture(scope="session")
def fixed_ground_truth(fixed_ground_truth_wav):
    return load_wav_as_array(fixed_ground_truth_wav)


@pytest.fixture(scope="session")
def discrete_speech_predictors():
    """Set up discrete speech predictors once per test session."""
    return discrete_speech_setup(use_gpu=False)


# -------------------------------
# Discrete Speech Metric Tests
# -------------------------------
def test_discrete_speech_metric_identical(fixed_audio, discrete_speech_predictors):
    """
    When comparing an audio signal with itself, the discrete speech scores should be high.
    """
    scores = discrete_speech_metric(discrete_speech_predictors, fixed_audio, fixed_audio, 16000)
    
    # Check that all expected metrics are present
    assert "speech_bert" in scores
    assert "speech_bleu" in scores
    assert "speech_token_distance" in scores
    
    # For identical signals, scores should be relatively high
    # Note: Perfect scores (1.0) are not always expected for discrete speech metrics
    assert scores["speech_bert"] > 0.9, f"Expected SpeechBERT score > 0.5 for identical signals, got {scores['speech_bert']}"
    assert scores["speech_bleu"] > 0.9, f"Expected SpeechBLEU score > 0.3 for identical signals, got {scores['speech_bleu']}"
    assert scores["speech_token_distance"] > 0.9, f"Expected SpeechTokenDistance score > 0.3 for identical signals, got {scores['speech_token_distance']}"


def test_discrete_speech_metric_different(fixed_audio, fixed_ground_truth, discrete_speech_predictors):
    """
    When comparing two different fixed signals, the discrete speech scores should be lower than identical signals.
    """
    # Get scores for identical signals first
    identical_scores = discrete_speech_metric(discrete_speech_predictors, fixed_audio, fixed_audio, 16000)
    
    # Get scores for different signals
    different_scores = discrete_speech_metric(discrete_speech_predictors, fixed_audio, fixed_ground_truth, 16000)
    
    # Check that all expected metrics are present
    assert "speech_bert" in different_scores
    assert "speech_bleu" in different_scores  
    assert "speech_token_distance" in different_scores
    
    # Different signals should have lower scores than identical signals
    assert different_scores["speech_bert"] <= identical_scores["speech_bert"], \
        f"Expected SpeechBERT score for different signals ({different_scores['speech_bert']}) to be <= identical signals ({identical_scores['speech_bert']})"
    
    assert different_scores["speech_bleu"] <= identical_scores["speech_bleu"], \
        f"Expected SpeechBLEU score for different signals ({different_scores['speech_bleu']}) to be <= identical signals ({identical_scores['speech_bleu']})"
    
    assert different_scores["speech_token_distance"] <= identical_scores["speech_token_distance"], \
        f"Expected SpeechTokenDistance score for different signals ({different_scores['speech_token_distance']}) to be <= identical signals ({identical_scores['speech_token_distance']})"
