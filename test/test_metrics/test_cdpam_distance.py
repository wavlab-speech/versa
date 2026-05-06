import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.cdpam_distance import (
    CdpamDistanceMetric,
    is_cdpam_available,
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
    Create a ground truth WAV file to be used as reference audio.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    gt_file = tmp_dir / "fixed_ground_truth.wav"
    # Use a different base frequency for ground truth (e.g. 300 Hz) to simulate a mismatch.
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
    Load the ground truth audio file as a NumPy array.
    """
    return load_wav_as_array(fixed_ground_truth_wav)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
    ],
)
def test_utterance_cdpam_distance(use_gpu, fixed_audio, fixed_ground_truth):
    """
    Test the CDPAM distance metric using the fixed audio files.
    The test uses deterministic data so that the result is always reproducible.
    """
    config = {"use_gpu": use_gpu}

    metric = CdpamDistanceMetric(config)
    metadata = {"sample_rate": 16000}
    result = metric.compute(fixed_audio, fixed_ground_truth, metadata=metadata)

    # Check that the result contains the expected key
    assert "cdpam_distance" in result, "Result should contain 'cdpam_distance' key"

    # Check that the result is a float
    cdpam_dist = result["cdpam_distance"]
    assert isinstance(cdpam_dist, float), "cdpam_distance should be a float"

    # Check that the distance score is reasonable (should be non-negative)
    assert cdpam_dist >= 0.0, f"CDPAM distance should be non-negative, got {cdpam_dist}"


@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
def test_cdpam_distance_metric_metadata():
    """Test that the CDPAM distance metric has correct metadata."""
    config = {"use_gpu": False}
    metric = CdpamDistanceMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "cdpam_distance"
    assert metadata.category.value == "dependent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is True
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert "cdpam" in metadata.dependencies
    assert "torch" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
def test_cdpam_distance_metric_different_sample_rates():
    """Test that the CDPAM distance metric handles different sample rates correctly."""
    config = {"use_gpu": False}
    metric = CdpamDistanceMetric(config)

    # Test with 44.1kHz audio (should be resampled to 22.05kHz)
    audio_44k_1 = np.random.random(44100)
    audio_44k_2 = np.random.random(44100)
    metadata_44k = {"sample_rate": 44100}
    result_44k = metric.compute(audio_44k_1, audio_44k_2, metadata=metadata_44k)

    # Test with 22.05kHz audio (no resampling needed)
    audio_22k_1 = np.random.random(22050)
    audio_22k_2 = np.random.random(22050)
    metadata_22k = {"sample_rate": 22050}
    result_22k = metric.compute(audio_22k_1, audio_22k_2, metadata=metadata_22k)

    # Both should return valid scores with expected keys
    assert (
        "cdpam_distance" in result_44k
    ), "44kHz result should contain 'cdpam_distance' key"
    assert (
        "cdpam_distance" in result_22k
    ), "22kHz result should contain 'cdpam_distance' key"

    # Both should return float values
    assert isinstance(result_44k["cdpam_distance"], float)
    assert isinstance(result_22k["cdpam_distance"], float)


@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
def test_cdpam_distance_metric_invalid_input():
    """Test that the CDPAM distance metric handles invalid inputs correctly."""
    config = {"use_gpu": False}
    metric = CdpamDistanceMetric(config)

    # Test with None predictions
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, np.random.random(22050), metadata={"sample_rate": 22050})

    # Test with None references
    with pytest.raises(ValueError, match="Reference signal must be provided"):
        metric.compute(np.random.random(22050), None, metadata={"sample_rate": 22050})


@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
def test_cdpam_distance_metric_config_options():
    """Test that the CDPAM distance metric handles different configuration options."""
    # Test with GPU disabled
    config_cpu = {"use_gpu": False}
    metric_cpu = CdpamDistanceMetric(config_cpu)

    # All should work without errors
    audio1 = np.random.random(22050)
    audio2 = np.random.random(22050)
    metadata = {"sample_rate": 22050}

    result_cpu = metric_cpu.compute(audio1, audio2, metadata=metadata)

    # Should return the same structure
    assert "cdpam_distance" in result_cpu
    assert isinstance(result_cpu["cdpam_distance"], float)


@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
def test_cdpam_distance_metric_identical_signals():
    """Test that the CDPAM distance metric gives zero distance for identical signals."""
    config = {"use_gpu": False}
    metric = CdpamDistanceMetric(config)
    metadata = {"sample_rate": 22050}

    # Test with identical signals
    audio = np.random.random(22050)
    result = metric.compute(audio, audio, metadata=metadata)

    # Results should be 0.0 for identical signals
    assert (
        result["cdpam_distance"] == 0.0
    ), "Identical signals should have zero distance"


@pytest.mark.skipif(not is_cdpam_available(), reason="CDPAM not available")
def test_cdpam_distance_metric_consistent_results():
    """Test that the CDPAM distance metric gives consistent results for the same inputs."""
    config = {"use_gpu": False}
    metric = CdpamDistanceMetric(config)
    metadata = {"sample_rate": 22050}

    # Test with fixed signals
    audio1 = np.random.random(22050)
    audio2 = np.random.random(22050)
    result1 = metric.compute(audio1, audio2, metadata=metadata)
    result2 = metric.compute(audio1, audio2, metadata=metadata)

    # Results should be identical for the same inputs
    np.testing.assert_almost_equal(
        result1["cdpam_distance"],
        result2["cdpam_distance"],
        decimal=6,
        err_msg="Results should be identical for the same inputs",
    )


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav, fixed_ground_truth_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_ground_truth_wav).exists()
