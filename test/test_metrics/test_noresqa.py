import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from packaging.version import parse as V

from versa.utterance_metrics.noresqa import NoresqaMetric, is_noresqa_available


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


@pytest.fixture(scope="session")
def fixed_audio_8k_wav(tmp_path_factory):
    """
    Create a fixed WAV file with 8kHz sample rate to test resampling.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio_8k.wav"
    # Generate an audio file with a 150 Hz sine wave at 8kHz.
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=8000, base_freq=150)
    return audio_file


@pytest.fixture(scope="session")
def fixed_ground_truth_8k_wav(tmp_path_factory):
    """
    Create a fixed WAV file with 8kHz sample rate to test resampling.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    gt_file = tmp_dir / "fixed_ground_truth_8k.wav"
    # Generate a ground truth file with a 300 Hz sine wave at 8kHz.
    generate_fixed_wav(gt_file, duration=1.0, sample_rate=8000, base_freq=300)
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


@pytest.fixture(scope="session")
def fixed_audio_8k(fixed_audio_8k_wav):
    """
    Load the fixed 8kHz audio file as a NumPy array.
    """
    return load_wav_as_array(fixed_audio_8k_wav, sample_rate=8000)


@pytest.fixture(scope="session")
def fixed_ground_truth_8k(fixed_ground_truth_8k_wav):
    """
    Load the fixed 8kHz ground truth file as a NumPy array.
    """
    return load_wav_as_array(fixed_ground_truth_8k_wav, sample_rate=8000)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(
    not is_noresqa_available(),
    reason="noresqa is not available",
)
@pytest.mark.parametrize(
    "metric_type,model_tag,use_gpu",
    [
        (1, "default", False),  # NORESQA-MOS
        (0, "default", False),  # NORESQA-score
    ],
)
def test_noresqa_metric_basic(
    metric_type, model_tag, use_gpu, fixed_audio, fixed_ground_truth
):
    """
    Test the NORESQA metric with basic configuration.
    """
    config = {
        "metric_type": metric_type,
        "model_tag": model_tag,
        "use_gpu": use_gpu,
        "cache_dir": "test_cache/noresqa_model",
    }

    metric = NoresqaMetric(config)
    result = metric.compute(
        fixed_audio, fixed_ground_truth, metadata={"sample_rate": 16000}
    )

    # Check that result contains noresqa_score field
    if metric_type == 0:
        assert "noresqa_score" in result
        assert isinstance(result["noresqa_score"], (int, float, np.number))
        assert not np.isnan(result["noresqa_score"])
        assert not np.isinf(result["noresqa_score"])
    elif metric_type == 1:
        assert "noresqa_mos" in result
        assert isinstance(result["noresqa_mos"], (int, float, np.number))
        assert not np.isnan(result["noresqa_mos"])
        assert not np.isinf(result["noresqa_mos"])


@pytest.mark.skipif(
    not is_noresqa_available(),
    reason="noresqa is not available",
)
def test_noresqa_metric_resampling(fixed_audio_8k, fixed_ground_truth_8k):
    """
    Test the NORESQA metric with audio that needs resampling.
    """
    config = {
        "metric_type": 1,  # NORESQA-MOS
        "model_tag": "default",
        "use_gpu": False,
        "cache_dir": "test_cache/noresqa_model",
    }

    metric = NoresqaMetric(config)
    result = metric.compute(
        fixed_audio_8k, fixed_ground_truth_8k, metadata={"sample_rate": 8000}
    )

    # Check that result contains noresqa_score field
    assert "noresqa_mos" in result
    assert isinstance(result["noresqa_mos"], (int, float, np.number))
    assert not np.isnan(result["noresqa_mos"])
    assert not np.isinf(result["noresqa_mos"])


@pytest.mark.skipif(
    not is_noresqa_available(),
    reason="noresqa is not available",
)
def test_noresqa_metric_invalid_input():
    """
    Test the NORESQA metric with invalid input.
    """
    config = {
        "metric_type": 1,
        "model_tag": "default",
        "use_gpu": False,
        "cache_dir": "test_cache/noresqa_model",
    }

    metric = NoresqaMetric(config)

    # Test with None predictions
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, np.random.random(16000), metadata={"sample_rate": 16000})

    # Test with None references
    with pytest.raises(ValueError, match="Reference signal must be provided"):
        metric.compute(np.random.random(16000), None, metadata={"sample_rate": 16000})


@pytest.mark.skipif(
    not is_noresqa_available(),
    reason="noresqa is not available",
)
@pytest.mark.parametrize("metric_type", [0, 1])
def test_noresqa_metric_metadata(metric_type):
    """
    Test the NORESQA metric metadata.
    """
    config = {
        "metric_type": metric_type,
        "model_tag": "default",
        "use_gpu": False,
        "cache_dir": "test_cache/noresqa_model",
    }

    metric = NoresqaMetric(config)
    metadata = metric.get_metadata()

    expected_name = "noresqa_mos" if metric_type == 1 else "noresqa_score"
    assert metadata.name == expected_name
    assert metadata.category.value == "dependent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is True
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert metadata.auto_install is False
    assert "fairseq" in metadata.dependencies
    assert "torch" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


def test_noresqa_metric_not_available():
    """
    Test the NORESQA metric when noresqa is not available.
    """
    # This test should be skipped if noresqa is available
    if is_noresqa_available():
        pytest.skip("noresqa is available, skipping this test")

    config = {
        "metric_type": 1,
        "model_tag": "default",
        "use_gpu": False,
        "cache_dir": "test_cache/noresqa_model",
    }

    with pytest.raises(ImportError, match="noresqa is not installed"):
        NoresqaMetric(config)


@pytest.mark.skipif(
    not is_noresqa_available(),
    reason="noresqa is not available",
)
def test_noresqa_metric_invalid_metric_type():
    """
    Test the NORESQA metric with invalid metric_type.
    """
    config = {
        "metric_type": 2,  # Invalid metric type
        "model_tag": "default",
        "use_gpu": False,
        "cache_dir": "test_cache/noresqa_model",
    }

    with pytest.raises(RuntimeError, match="Invalid metric_type"):
        NoresqaMetric(config)


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(
    fixed_audio_wav,
    fixed_ground_truth_wav,
    fixed_audio_8k_wav,
    fixed_ground_truth_8k_wav,
):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_ground_truth_wav).exists()
    assert Path(fixed_audio_8k_wav).exists()
    assert Path(fixed_ground_truth_8k_wav).exists()
