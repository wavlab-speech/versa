import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.emo_similarity import EmotionMetric, is_emo2vec_available


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
def fixed_audio_wav_2(tmp_path_factory):
    """
    Create a second fixed WAV file to be used as reference audio.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio_2.wav"
    # Generate an audio file with a 200 Hz sine wave.
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=16000, base_freq=200)
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


@pytest.fixture(scope="session")
def fixed_audio_2(fixed_audio_wav_2):
    """
    Load the second fixed audio file as a NumPy array.
    """
    return load_wav_as_array(fixed_audio_wav_2)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
@pytest.mark.parametrize(
    "use_gpu",
    [
        False,
    ],
)
def test_utterance_emotion(use_gpu, fixed_audio, fixed_audio_2):
    """
    Test the Emotion metric using the fixed audio files.
    The test uses deterministic data so that the result is always reproducible.
    """
    config = {"use_gpu": use_gpu}

    metric = EmotionMetric(config)
    metadata = {"sample_rate": 16000}
    result = metric.compute(fixed_audio, fixed_audio_2, metadata=metadata)

    # Check that the result contains the expected key
    assert (
        "emotion_similarity" in result
    ), "Result should contain 'emotion_similarity' key"

    # Check that the result is a float
    emotion_sim = result["emotion_similarity"]
    assert isinstance(emotion_sim, float), "emotion_similarity should be a float"

    # Check that the similarity score is reasonable (between -1 and 1 for cosine similarity)
    assert (
        -1.0 <= emotion_sim <= 1.0
    ), f"Emotion similarity should be between -1 and 1, got {emotion_sim}"


@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
def test_emotion_metric_metadata():
    """Test that the Emotion metric has correct metadata."""
    config = {"use_gpu": False}
    metric = EmotionMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "emotion"
    assert metadata.category.value == "dependent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is True
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert "emo2vec_versa" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
def test_emotion_metric_different_sample_rates():
    """Test that the Emotion metric handles different sample rates correctly."""
    config = {"use_gpu": False}
    metric = EmotionMetric(config)

    # Test with 44.1kHz audio (should be resampled to 16kHz)
    audio_44k_1 = np.random.random(44100)
    audio_44k_2 = np.random.random(44100)
    metadata_44k = {"sample_rate": 44100}
    result_44k = metric.compute(audio_44k_1, audio_44k_2, metadata=metadata_44k)

    # Test with 16kHz audio (no resampling needed)
    audio_16k_1 = np.random.random(16000)
    audio_16k_2 = np.random.random(16000)
    metadata_16k = {"sample_rate": 16000}
    result_16k = metric.compute(audio_16k_1, audio_16k_2, metadata=metadata_16k)

    # Both should return valid scores with expected keys
    assert (
        "emotion_similarity" in result_44k
    ), "44kHz result should contain 'emotion_similarity' key"
    assert (
        "emotion_similarity" in result_16k
    ), "16kHz result should contain 'emotion_similarity' key"

    # Both should return float values
    assert isinstance(result_44k["emotion_similarity"], float)
    assert isinstance(result_16k["emotion_similarity"], float)


@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
def test_emotion_metric_invalid_input():
    """Test that the Emotion metric handles invalid inputs correctly."""
    config = {"use_gpu": False}
    metric = EmotionMetric(config)

    # Test with None predictions
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, np.random.random(16000), metadata={"sample_rate": 16000})

    # Test with None references
    with pytest.raises(ValueError, match="Reference signal must be provided"):
        metric.compute(np.random.random(16000), None, metadata={"sample_rate": 16000})


@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
def test_emotion_metric_config_options():
    """Test that the Emotion metric handles different configuration options."""
    # Test with GPU disabled
    config_cpu = {"use_gpu": False}
    metric_cpu = EmotionMetric(config_cpu)

    # Test with different model tag
    config_custom_model = {"use_gpu": False, "model_tag": "base"}
    metric_custom_model = EmotionMetric(config_custom_model)

    # All should work without errors
    audio1 = np.random.random(16000)
    audio2 = np.random.random(16000)
    metadata = {"sample_rate": 16000}

    result_cpu = metric_cpu.compute(audio1, audio2, metadata=metadata)
    result_custom_model = metric_custom_model.compute(audio1, audio2, metadata=metadata)

    # All should return the same structure
    assert "emotion_similarity" in result_cpu
    assert "emotion_similarity" in result_custom_model
    assert isinstance(result_cpu["emotion_similarity"], float)
    assert isinstance(result_custom_model["emotion_similarity"], float)


@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
def test_emotion_metric_identical_signals():
    """Test that the Emotion metric gives high similarity for identical signals."""
    config = {"use_gpu": False}
    metric = EmotionMetric(config)
    metadata = {"sample_rate": 16000}

    # Test with identical signals
    audio = np.random.random(16000)
    result = metric.compute(audio, audio, metadata=metadata)

    # Results should be very close to 1.0 for identical signals
    assert (
        result["emotion_similarity"] > 0.99
    ), "Identical signals should have very high similarity"


@pytest.mark.skipif(not is_emo2vec_available(), reason="Emo2vec not available")
def test_emotion_metric_consistent_results():
    """Test that the Emotion metric gives consistent results for the same inputs."""
    config = {"use_gpu": False}
    metric = EmotionMetric(config)
    metadata = {"sample_rate": 16000}

    # Test with fixed signals
    audio1 = np.random.random(16000)
    audio2 = np.random.random(16000)
    result1 = metric.compute(audio1, audio2, metadata=metadata)
    result2 = metric.compute(audio1, audio2, metadata=metadata)

    # Results should be identical for the same inputs
    np.testing.assert_almost_equal(
        result1["emotion_similarity"],
        result2["emotion_similarity"],
        decimal=6,
        err_msg="Results should be identical for the same inputs",
    )


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav, fixed_audio_wav_2):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_audio_wav_2).exists()
