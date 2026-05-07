import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.chroma_alignment import ChromaAlignmentMetric


# -------------------------------
# Helper: Generate a fixed WAV file
# -------------------------------
def generate_fixed_wav(
    filename, duration=1.0, sample_rate=22050, base_freq=440, envelope_func=None
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
    # Generate an audio file with a 440 Hz sine wave (A4 note).
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=22050, base_freq=440)
    return audio_file


@pytest.fixture(scope="session")
def fixed_ground_truth_wav(tmp_path_factory):
    """
    Create a fixed WAV file to be used as ground truth.
    This one uses a different duration but same frequency to test DTW alignment.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    gt_file = tmp_dir / "fixed_ground_truth.wav"
    # Generate a ground truth file with a 440 Hz sine wave but different duration.
    generate_fixed_wav(gt_file, duration=1.2, sample_rate=22050, base_freq=440)
    return gt_file


@pytest.fixture(scope="session")
def different_pitch_wav(tmp_path_factory):
    """
    Create a WAV file with a different pitch for testing distance metrics.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    diff_file = tmp_dir / "different_pitch.wav"
    # Generate a file with a 554.37 Hz sine wave (C#5 note).
    generate_fixed_wav(diff_file, duration=1.0, sample_rate=22050, base_freq=554.37)
    return diff_file


# -------------------------------
# Fixtures to Load WAV Files into NumPy Arrays
# -------------------------------
def load_wav_as_array(wav_path, sample_rate=22050):
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


@pytest.fixture(scope="session")
def different_pitch_audio(different_pitch_wav):
    """
    Load the different pitch audio file as a NumPy array.
    """
    return load_wav_as_array(different_pitch_wav)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.parametrize(
    "scale_factor,feature_types,distance_metrics",
    [
        (100.0, ["stft"], ["cosine"]),
        (50.0, ["stft", "cqt"], ["cosine", "euclidean"]),
        (200.0, ["stft", "cqt", "cens"], ["cosine"]),
    ],
)
def test_utterance_chroma_alignment(
    scale_factor, feature_types, distance_metrics, fixed_audio, fixed_ground_truth
):
    """
    Test the Chroma Alignment metric using the fixed audio and ground truth.
    The test uses deterministic data so that the result is always reproducible.
    """
    config = {
        "scale_factor": scale_factor,
        "feature_types": feature_types,
        "distance_metrics": distance_metrics,
        "normalize": True,
        "normalize_by_path": True,
    }

    metric = ChromaAlignmentMetric(config)
    metadata = {"sample_rate": 22050}
    result = metric.compute(fixed_audio, fixed_ground_truth, metadata=metadata)

    # Check that the result contains the expected keys
    for feat_type in feature_types:
        for dist_metric in distance_metrics:
            key = f"chroma_{feat_type}_{dist_metric}_dtw"
            assert key in result, f"Result should contain '{key}' key"
            assert isinstance(
                result[key], (int, float)
            ), f"Score {key} should be numeric"
            assert result[key] >= 0, f"Score {key} should be non-negative"

    # Check for additional scaled variants
    if "stft" in feature_types and "cosine" in distance_metrics:
        assert "chroma_stft_cosine_dtw_raw" in result
        assert "chroma_stft_cosine_dtw_log" in result


def test_chroma_alignment_metric_metadata():
    """Test that the Chroma Alignment metric has correct metadata."""
    config = {"scale_factor": 100.0}
    metric = ChromaAlignmentMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "chroma_alignment"
    assert metadata.category.value == "dependent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is True
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is False
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies
    assert "scipy" in metadata.dependencies


def test_chroma_alignment_metric_different_pitches(fixed_audio, different_pitch_audio):
    """Test that the Chroma Alignment metric gives higher distances for different pitches."""
    config = {"scale_factor": 100.0}
    metric = ChromaAlignmentMetric(config)
    metadata = {"sample_rate": 22050}

    # Test with same pitch (should give lower distance)
    result_same = metric.compute(fixed_audio, fixed_audio, metadata=metadata)

    # Test with different pitch (should give higher distance)
    result_different = metric.compute(
        fixed_audio, different_pitch_audio, metadata=metadata
    )

    # The distance should be higher for different pitches
    for key in result_same:
        if key in result_different and not key.endswith("_log"):
            # Log-scaled metric works differently, so skip it
            assert (
                result_different[key] >= result_same[key]
            ), f"Distance should be higher for different pitches in {key}"


def test_chroma_alignment_metric_invalid_input():
    """Test that the Chroma Alignment metric handles invalid inputs correctly."""
    config = {"scale_factor": 100.0}
    metric = ChromaAlignmentMetric(config)

    # Test with None input
    with pytest.raises(
        ValueError, match="Both predicted and ground truth signals must be provided"
    ):
        metric.compute(None, np.random.random(22050), metadata={"sample_rate": 22050})

    with pytest.raises(
        ValueError, match="Both predicted and ground truth signals must be provided"
    ):
        metric.compute(np.random.random(22050), None, metadata={"sample_rate": 22050})


def test_chroma_alignment_metric_config_options():
    """Test that the Chroma Alignment metric handles different configuration options."""
    # Test with different scale factors
    config_small_scale = {
        "scale_factor": 50.0,
        "feature_types": ["stft"],
        "distance_metrics": ["cosine"],
    }
    metric_small = ChromaAlignmentMetric(config_small_scale)

    config_large_scale = {
        "scale_factor": 200.0,
        "feature_types": ["stft"],
        "distance_metrics": ["cosine"],
    }
    metric_large = ChromaAlignmentMetric(config_large_scale)

    # Test with normalization options
    config_no_norm = {
        "normalize": False,
        "feature_types": ["stft"],
        "distance_metrics": ["cosine"],
    }
    metric_no_norm = ChromaAlignmentMetric(config_no_norm)

    # All should work without errors
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 22050))
    audio2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 22050))
    metadata = {"sample_rate": 22050}
    result_small = metric_small.compute(audio, audio2, metadata=metadata)
    result_large = metric_large.compute(audio, audio2, metadata=metadata)
    result_no_norm = metric_no_norm.compute(audio, audio2, metadata=metadata)

    # All should return the same structure
    assert "chroma_stft_cosine_dtw" in result_small
    assert "chroma_stft_cosine_dtw" in result_large
    assert "chroma_stft_cosine_dtw" in result_no_norm

    # Scale factor should affect the magnitude
    assert (
        result_large["chroma_stft_cosine_dtw"] > result_small["chroma_stft_cosine_dtw"]
    )


def test_chroma_alignment_metric_alignment_paths():
    """Test that the Chroma Alignment metric can return alignment paths when requested."""
    config = {
        "scale_factor": 100.0,
        "feature_types": ["stft"],
        "distance_metrics": ["cosine"],
        "return_alignment": True,
    }

    metric = ChromaAlignmentMetric(config)
    metadata = {"sample_rate": 22050}
    audio = np.random.random(22050)

    result = metric.compute(audio, audio, metadata=metadata)

    # Should contain alignments when requested
    assert "alignments" in result
    assert "chroma_stft_cosine_dtw" in result["alignments"]


def test_chroma_alignment_metric_multidimensional_input():
    """Test that the Chroma Alignment metric handles multidimensional input correctly."""
    config = {
        "scale_factor": 100.0,
        "feature_types": ["stft"],
        "distance_metrics": ["cosine"],
    }
    metric = ChromaAlignmentMetric(config)
    metadata = {"sample_rate": 22050}

    # Test with 2D input (should be flattened)
    audio_2d = np.random.random((22050, 1))
    result_2d = metric.compute(audio_2d, audio_2d, metadata=metadata)

    # Test with 1D input
    audio_1d = np.random.random(22050)
    result_1d = metric.compute(audio_1d, audio_1d, metadata=metadata)

    # Both should work and give similar results (not exactly the same due to randomness)
    assert "chroma_stft_cosine_dtw" in result_2d
    assert "chroma_stft_cosine_dtw" in result_1d


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(
    fixed_audio_wav, fixed_ground_truth_wav, different_pitch_wav
):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_ground_truth_wav).exists()
    assert Path(different_pitch_wav).exists()
