import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.emo_vad import EmoVadMetric, is_transformers_available


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


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
    ],
)
def test_utterance_emo_vad(use_gpu, fixed_audio):
    """
    Test the EmoVad metric using the fixed audio.
    The test uses deterministic data so that the result is always reproducible.
    """
    config = {"use_gpu": use_gpu}

    metric = EmoVadMetric(config)
    metadata = {"sample_rate": 16000}
    result = metric.compute(fixed_audio, metadata=metadata)

    # Check that the result contains the expected key
    assert "arousal_emo_vad" in result, "Result should contain 'arousal_emo_vad' key"
    assert "valence_emo_vad" in result, "Result should contain 'valence_emo_vad' key"
    assert (
        "dominance_emo_vad" in result
    ), "Result should contain 'dominance_emo_vad' key"

    # Check that the result is a numpy array with 3 values (arousal, valence, dominance)
    arousal = result["arousal_emo_vad"]
    valence = result["valence_emo_vad"]
    dominance = result["dominance_emo_vad"]
    assert isinstance(arousal, float), "arousal_emo_vad should be a float"
    assert isinstance(valence, float), "valence_emo_vad should be a float"
    assert isinstance(dominance, float), "dominance_emo_vad should be a float"

    # Check that all values are numeric and reasonable (emotion scores are typically between 0 and 1)
    assert (
        0.0 <= arousal <= 1.0
    ), f"Arousal score should be between 0 and 1, got {arousal}"
    assert (
        0.0 <= valence <= 1.0
    ), f"Valence score should be between 0 and 1, got {valence}"
    assert (
        0.0 <= dominance <= 1.0
    ), f"Dominance score should be between 0 and 1, got {dominance}"


@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_emo_vad_metric_metadata():
    """Test that the EmoVad metric has correct metadata."""
    config = {"use_gpu": False}
    metric = EmoVadMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "emo_vad"
    assert metadata.category.value == "independent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is False
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert "transformers" in metadata.dependencies
    assert "torch" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_emo_vad_metric_different_sample_rates():
    """Test that the EmoVad metric handles different sample rates correctly."""
    config = {"use_gpu": False}
    metric = EmoVadMetric(config)

    # Test with 44.1kHz audio (should be resampled to 16kHz)
    audio_44k = np.random.random(44100)
    metadata_44k = {"sample_rate": 44100}
    result_44k = metric.compute(audio_44k, metadata=metadata_44k)

    # Test with 16kHz audio (no resampling needed)
    audio_16k = np.random.random(16000)
    metadata_16k = {"sample_rate": 16000}
    result_16k = metric.compute(audio_16k, metadata=metadata_16k)

    # Both should return valid scores with expected keys
    assert (
        "arousal_emo_vad" in result_44k
    ), "44kHz result should contain 'arousal_emo_vad' key"
    assert (
        "valence_emo_vad" in result_44k
    ), "44kHz result should contain 'valence_emo_vad' key"
    assert (
        "dominance_emo_vad" in result_44k
    ), "44kHz result should contain 'dominance_emo_vad' key"
    assert (
        "arousal_emo_vad" in result_16k
    ), "16kHz result should contain 'arousal_emo_vad' key"
    assert (
        "valence_emo_vad" in result_16k
    ), "16kHz result should contain 'valence_emo_vad' key"
    assert (
        "dominance_emo_vad" in result_16k
    ), "16kHz result should contain 'dominance_emo_vad' key"

    # Both should return numpy arrays with 3 values
    assert (
        type(result_44k["arousal_emo_vad"]) == float
    ), "arousal_emo_vad should be a float"
    assert (
        type(result_44k["valence_emo_vad"]) == float
    ), "valence_emo_vad should be a float"
    assert (
        type(result_44k["dominance_emo_vad"]) == float
    ), "dominance_emo_vad should be a float"
    assert (
        type(result_16k["arousal_emo_vad"]) == float
    ), "arousal_emo_vad should be a float"
    assert (
        type(result_16k["valence_emo_vad"]) == float
    ), "valence_emo_vad should be a float"
    assert (
        type(result_16k["dominance_emo_vad"]) == float
    ), "dominance_emo_vad should be a float"


@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_emo_vad_metric_invalid_input():
    """Test that the EmoVad metric handles invalid inputs correctly."""
    config = {"use_gpu": False}
    metric = EmoVadMetric(config)

    # Test with None input
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, metadata={"sample_rate": 16000})


@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_emo_vad_metric_config_options():
    """Test that the EmoVad metric handles different configuration options."""
    # Test with GPU disabled
    config_cpu = {"use_gpu": False}
    metric_cpu = EmoVadMetric(config_cpu)

    # Test with different model tag
    config_custom_model = {
        "use_gpu": False,
        "model_tag": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
    }
    metric_custom_model = EmoVadMetric(config_custom_model)

    # All should work without errors
    audio = np.random.random(16000)
    metadata = {"sample_rate": 16000}

    result_cpu = metric_cpu.compute(audio, metadata=metadata)
    result_custom_model = metric_custom_model.compute(audio, metadata=metadata)

    # All should return the same structure
    assert "arousal_emo_vad" in result_cpu
    assert "valence_emo_vad" in result_cpu
    assert "dominance_emo_vad" in result_cpu
    assert "arousal_emo_vad" in result_custom_model
    assert "valence_emo_vad" in result_custom_model
    assert "dominance_emo_vad" in result_custom_model
    assert (
        type(result_cpu["arousal_emo_vad"]) == float
    ), "arousal_emo_vad should be a float"
    assert (
        type(result_cpu["valence_emo_vad"]) == float
    ), "valence_emo_vad should be a float"
    assert (
        type(result_cpu["dominance_emo_vad"]) == float
    ), "dominance_emo_vad should be a float"


@pytest.mark.skipif(
    not is_transformers_available(), reason="Transformers not available"
)
def test_emo_vad_metric_identical_signals():
    """Test that the EmoVad metric gives consistent results for identical signals."""
    config = {"use_gpu": False}
    metric = EmoVadMetric(config)
    metadata = {"sample_rate": 16000}

    # Test with identical signals
    audio = np.random.random(16000)
    result1 = metric.compute(audio, metadata=metadata)
    result2 = metric.compute(audio, metadata=metadata)

    # Results should be identical for the same input
    np.testing.assert_array_almost_equal(
        result1["arousal_emo_vad"],
        result2["arousal_emo_vad"],
        decimal=6,
        err_msg="Results should be identical for the same input",
    )


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
