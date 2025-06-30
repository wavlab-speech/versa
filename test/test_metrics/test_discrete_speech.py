import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.discrete_speech import (
    DiscreteSpeechMetric,
    is_discrete_speech_available,
)


# -------------------------------
# Helper: Generate a fixed WAV file
# -------------------------------
def generate_fixed_wav(
    filename, duration=1.0, sample_rate=16000, base_freq=150, envelope_func=None
):
    """
    Generate a deterministic WAV file with a modulated sine wave.

    Parameters:
      - filename: Path (str or Path) to write the WAV file.
      - duration: Duration of the audio in seconds.
      - sample_rate: Number of samples per second.
      - base_freq: Frequency (in Hz) of the sine wave.
      - envelope_func: Optional function to generate a custom amplitude envelope.
                       If None, a default sine-based envelope is used.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    # Use default envelope if none is provided.
    if envelope_func is None:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    else:
        envelope = envelope_func(t)
    audio = envelope * np.sin(2 * np.pi * base_freq * t)

    # Scale to 16-bit PCM.
    amplitude = np.iinfo(np.int16).max
    data = (audio * amplitude).astype(np.int16)

    # Write the WAV file.
    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(1)  # Mono audio.
        wf.setsampwidth(2)  # 16 bits per sample.
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


# -------------------------------
# Session-Scoped Fixtures to Create WAV Files
# -------------------------------
@pytest.fixture(scope="session")
def fixed_audio_wav(tmp_path_factory):
    """
    Create a fixed WAV file to be used as test audio.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio.wav"
    # Generate an audio file with a 150 Hz sine wave.
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=16000, base_freq=150)
    return audio_file


@pytest.fixture(scope="session")
def fixed_ground_truth_wav(tmp_path_factory):
    """
    Create a fixed WAV file to be used as ground truth.
    This one uses a different base frequency (e.g., 300 Hz) so that the test
    intentionally simulates a mismatch.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    gt_file = tmp_dir / "fixed_ground_truth.wav"
    # Generate a ground truth file with a 300 Hz sine wave.
    generate_fixed_wav(gt_file, duration=1.0, sample_rate=16000, base_freq=300)
    return gt_file


# -------------------------------
# Fixtures to Load WAV Files into NumPy Arrays
# -------------------------------
def load_wav_as_array(wav_path, sample_rate=16000):
    """
    Load a WAV file and convert it into a NumPy array of floats scaled to [-1, 1].
    """
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        audio_data = wf.readframes(frames)
    # Convert from 16-bit PCM.
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    return audio_array / np.iinfo(np.int16).max


@pytest.fixture(scope="session")
def fixed_audio(fixed_audio_wav):
    """
    Load the fixed audio file as a NumPy array.
    """
    return load_wav_as_array(fixed_audio_wav)


@pytest.fixture(scope="session")
def fixed_ground_truth(fixed_ground_truth_wav):
    """
    Load the fixed ground truth file as a NumPy array.
    """
    return load_wav_as_array(fixed_ground_truth_wav)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(
    not is_discrete_speech_available(), reason="Discrete Speech Metrics not available"
)
@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
    ],
)
def test_utterance_discrete_speech_identical(use_gpu, fixed_audio):
    """
    Test the Discrete Speech metric using identical audio signals.
    When comparing an audio signal with itself, the discrete speech scores should be high.
    """
    config = {"use_gpu": use_gpu}

    metric = DiscreteSpeechMetric(config)
    metadata = {"sample_rate": 16000}
    result = metric.compute(fixed_audio, fixed_audio, metadata=metadata)

    # Check that all expected metrics are present
    assert "speech_bert" in result, "Result should contain 'speech_bert' key"
    assert "speech_bleu" in result, "Result should contain 'speech_bleu' key"
    assert (
        "speech_token_distance" in result
    ), "Result should contain 'speech_token_distance' key"

    # For identical signals, scores should be relatively high
    # Note: Perfect scores (1.0) are not always expected for discrete speech metrics
    assert (
        result["speech_bert"] > 0.9
    ), f"Expected SpeechBERT score > 0.9 for identical signals, got {result['speech_bert']}"
    assert (
        result["speech_bleu"] > 0.9
    ), f"Expected SpeechBLEU score > 0.9 for identical signals, got {result['speech_bleu']}"
    assert (
        result["speech_token_distance"] > 0.9
    ), f"Expected SpeechTokenDistance score > 0.9 for identical signals, got {result['speech_token_distance']}"


@pytest.mark.skipif(
    not is_discrete_speech_available(), reason="Discrete Speech Metrics not available"
)
@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
    ],
)
def test_utterance_discrete_speech_different(use_gpu, fixed_audio, fixed_ground_truth):
    """
    Test the Discrete Speech metric using different audio signals.
    When comparing two different fixed signals, the discrete speech scores should be lower than identical signals.
    """
    config = {"use_gpu": use_gpu}

    metric = DiscreteSpeechMetric(config)
    metadata = {"sample_rate": 16000}

    # Get scores for identical signals first
    identical_result = metric.compute(fixed_audio, fixed_audio, metadata=metadata)

    # Get scores for different signals
    different_result = metric.compute(
        fixed_audio, fixed_ground_truth, metadata=metadata
    )

    # Check that all expected metrics are present
    assert "speech_bert" in different_result, "Result should contain 'speech_bert' key"
    assert "speech_bleu" in different_result, "Result should contain 'speech_bleu' key"
    assert (
        "speech_token_distance" in different_result
    ), "Result should contain 'speech_token_distance' key"

    # Different signals should have lower scores than identical signals
    assert (
        different_result["speech_bert"] <= identical_result["speech_bert"]
    ), f"Expected SpeechBERT score for different signals ({different_result['speech_bert']}) to be <= identical signals ({identical_result['speech_bert']})"
    assert (
        different_result["speech_bleu"] <= identical_result["speech_bleu"]
    ), f"Expected SpeechBLEU score for different signals ({different_result['speech_bleu']}) to be <= identical signals ({identical_result['speech_bleu']})"
    assert (
        different_result["speech_token_distance"]
        <= identical_result["speech_token_distance"]
    ), f"Expected SpeechTokenDistance score for different signals ({different_result['speech_token_distance']}) to be <= identical signals ({identical_result['speech_token_distance']})"


@pytest.mark.skipif(
    not is_discrete_speech_available(), reason="Discrete Speech Metrics not available"
)
def test_discrete_speech_metric_metadata():
    """Test that the Discrete Speech metric has correct metadata."""
    config = {"use_gpu": False}
    metric = DiscreteSpeechMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "discrete_speech"
    assert metadata.category.value == "dependent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is True
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert "discrete_speech_metrics" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


@pytest.mark.skipif(
    not is_discrete_speech_available(), reason="Discrete Speech Metrics not available"
)
def test_discrete_speech_metric_different_sample_rates():
    """Test that the Discrete Speech metric handles different sample rates correctly."""
    config = {"use_gpu": False}
    metric = DiscreteSpeechMetric(config)

    # Test with 44.1kHz audio (should be resampled to 16kHz)
    audio_44k = np.random.random(44100)
    metadata_44k = {"sample_rate": 44100}
    result_44k = metric.compute(audio_44k, audio_44k, metadata=metadata_44k)

    # Test with 16kHz audio (no resampling needed)
    audio_16k = np.random.random(16000)
    metadata_16k = {"sample_rate": 16000}
    result_16k = metric.compute(audio_16k, audio_16k, metadata=metadata_16k)

    # Both should return valid scores with expected keys
    expected_keys = ["speech_bert", "speech_bleu", "speech_token_distance"]

    for key in expected_keys:
        assert key in result_44k, f"44kHz result should contain '{key}' key"
        assert key in result_16k, f"16kHz result should contain '{key}' key"
        assert isinstance(
            result_44k[key], (int, float)
        ), f"Score {key} should be numeric"
        assert isinstance(
            result_16k[key], (int, float)
        ), f"Score {key} should be numeric"


@pytest.mark.skipif(
    not is_discrete_speech_available(), reason="Discrete Speech Metrics not available"
)
def test_discrete_speech_metric_invalid_input():
    """Test that the Discrete Speech metric handles invalid inputs correctly."""
    config = {"use_gpu": False}
    metric = DiscreteSpeechMetric(config)

    # Test with None input
    with pytest.raises(
        ValueError, match="Both predicted and ground truth signals must be provided"
    ):
        metric.compute(None, np.random.random(16000), metadata={"sample_rate": 16000})

    with pytest.raises(
        ValueError, match="Both predicted and ground truth signals must be provided"
    ):
        metric.compute(np.random.random(16000), None, metadata={"sample_rate": 16000})


@pytest.mark.skipif(
    not is_discrete_speech_available(), reason="Discrete Speech Metrics not available"
)
def test_discrete_speech_metric_config_options():
    """Test that the Discrete Speech metric handles different configuration options."""
    # Test with GPU disabled
    config_cpu = {"use_gpu": False}
    metric_cpu = DiscreteSpeechMetric(config_cpu)

    # Test with different sample rate
    config_custom_sr = {"use_gpu": False, "sample_rate": 22050}
    metric_custom_sr = DiscreteSpeechMetric(config_custom_sr)

    # All should work without errors
    audio = np.random.random(16000)
    metadata = {"sample_rate": 16000}

    result_cpu = metric_cpu.compute(audio, audio, metadata=metadata)
    result_custom_sr = metric_custom_sr.compute(audio, audio, metadata=metadata)

    # All should return the same structure
    expected_keys = ["speech_bert", "speech_bleu", "speech_token_distance"]

    for key in expected_keys:
        assert key in result_cpu
        assert key in result_custom_sr


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav, fixed_ground_truth_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_ground_truth_wav).exists()
