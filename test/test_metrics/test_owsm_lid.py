import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from packaging.version import parse as V

from versa.utterance_metrics.owsm_lid import OwsmLidMetric, is_espnet2_available


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
def fixed_audio_8k_wav(tmp_path_factory):
    """
    Create a fixed WAV file with 8kHz sample rate to test resampling.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio_8k.wav"
    # Generate an audio file with a 150 Hz sine wave at 8kHz.
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=8000, base_freq=150)
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
def fixed_audio_8k(fixed_audio_8k_wav):
    """
    Load the fixed 8kHz audio file as a NumPy array.
    """
    return load_wav_as_array(fixed_audio_8k_wav, sample_rate=8000)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(
    not is_espnet2_available(),
    reason="espnet2 is not available",
)
@pytest.mark.parametrize(
    "model_tag,nbest,use_gpu",
    [
        ("default", 3, False),
        ("default", 5, False),
        ("espnet/owsm_v3.1_ebf", 3, False),
    ],
)
def test_owsm_lid_metric_basic(model_tag, nbest, use_gpu, fixed_audio):
    """
    Test the OWSM LID metric with basic configuration.
    """
    config = {
        "model_tag": model_tag,
        "nbest": nbest,
        "use_gpu": use_gpu,
    }

    metric = OwsmLidMetric(config)
    result = metric.compute(fixed_audio, metadata={"sample_rate": 16000})

    # Check that result contains language field
    assert "language" in result
    assert isinstance(result["language"][0][0], str)
    assert len(result["language"]) > 0


@pytest.mark.skipif(
    not is_espnet2_available(),
    reason="espnet2 is not available",
)
def test_owsm_lid_metric_resampling(fixed_audio_8k):
    """
    Test the OWSM LID metric with audio that needs resampling.
    """
    config = {
        "model_tag": "default",
        "nbest": 3,
        "use_gpu": False,
    }

    metric = OwsmLidMetric(config)
    result = metric.compute(fixed_audio_8k, metadata={"sample_rate": 8000})

    # Check that result contains language field
    assert "language" in result
    assert isinstance(result["language"][0][0], str)
    assert len(result["language"]) > 0


@pytest.mark.skipif(
    not is_espnet2_available(),
    reason="espnet2 is not available",
)
def test_owsm_lid_metric_invalid_input():
    """
    Test the OWSM LID metric with invalid input.
    """
    config = {
        "model_tag": "default",
        "nbest": 3,
        "use_gpu": False,
    }

    metric = OwsmLidMetric(config)

    # Test with None input
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, metadata={"sample_rate": 16000})


@pytest.mark.skipif(
    not is_espnet2_available(),
    reason="espnet2 is not available",
)
def test_owsm_lid_metric_metadata():
    """
    Test the OWSM LID metric metadata.
    """
    config = {
        "model_tag": "default",
        "nbest": 3,
        "use_gpu": False,
    }

    metric = OwsmLidMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "lid"
    assert metadata.category.value == "independent"
    assert metadata.metric_type.value == "list"
    assert metadata.requires_reference is False
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert metadata.auto_install is False
    assert "espnet2" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


def test_owsm_lid_metric_espnet2_not_available():
    """
    Test the OWSM LID metric when espnet2 is not available.
    """
    # This test should be skipped if espnet2 is available
    if is_espnet2_available():
        pytest.skip("espnet2 is available, skipping this test")

    config = {
        "model_tag": "default",
        "nbest": 3,
        "use_gpu": False,
    }

    with pytest.raises(ImportError, match="espnet2 is not properly installed"):
        OwsmLidMetric(config)


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav, fixed_audio_8k_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_audio_8k_wav).exists()
