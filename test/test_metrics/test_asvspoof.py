import wave
from pathlib import Path

import numpy as np
import pytest
import torch

from versa.utterance_metrics.asvspoof_score import ASVSpoofMetric, is_aasist_available


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
@pytest.mark.skipif(not is_aasist_available(), reason="AASIST not available")
@pytest.mark.parametrize(
    "model_tag,use_gpu",
    [
        ("default", False),
    ],
)
def test_utterance_asvspoof(model_tag, use_gpu, fixed_audio):
    """
    Test the ASVspoof metric using the fixed audio.
    The test uses deterministic data so that the result is always reproducible.
    """
    config = {
        "model_tag": model_tag,
        "use_gpu": use_gpu
    }
    
    metric = ASVSpoofMetric(config)
    metadata = {"sample_rate": 16000}
    result = metric.compute(fixed_audio, metadata=metadata)
    
    asvspoof_score = result["asvspoof_score"]
    
    # Check that the score is a valid probability (between 0 and 1)
    assert 0.0 <= asvspoof_score <= 1.0, f"ASVspoof score {asvspoof_score} is not between 0 and 1"
    
    # Check that the result contains the expected key
    assert "asvspoof_score" in result, "Result should contain 'asvspoof_score' key"


@pytest.mark.skipif(not is_aasist_available(), reason="AASIST not available")
def test_asvspoof_metric_metadata():
    """Test that the ASVspoof metric has correct metadata."""
    config = {"use_gpu": False}
    metric = ASVSpoofMetric(config)
    metadata = metric.get_metadata()
    
    assert metadata.name == "asvspoof"
    assert metadata.category.value == "independent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is False
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert "torch" in metadata.dependencies
    assert "librosa" in metadata.dependencies
    assert "numpy" in metadata.dependencies


@pytest.mark.skipif(not is_aasist_available(), reason="AASIST not available")
def test_asvspoof_metric_resampling():
    """Test that the ASVspoof metric handles different sample rates correctly."""
    config = {"use_gpu": False}
    metric = ASVSpoofMetric(config)
    
    # Test with 44.1kHz audio (should be resampled to 16kHz)
    audio_44k = np.random.random(44100)
    metadata_44k = {"sample_rate": 44100}
    result_44k = metric.compute(audio_44k, metadata=metadata_44k)
    
    # Test with 16kHz audio (no resampling needed)
    audio_16k = np.random.random(16000)
    metadata_16k = {"sample_rate": 16000}
    result_16k = metric.compute(audio_16k, metadata=metadata_16k)
    
    # Both should return valid scores
    assert 0.0 <= result_44k["asvspoof_score"] <= 1.0
    assert 0.0 <= result_16k["asvspoof_score"] <= 1.0


@pytest.mark.skipif(not is_aasist_available(), reason="AASIST not available")
def test_asvspoof_metric_invalid_input():
    """Test that the ASVspoof metric handles invalid inputs correctly."""
    config = {"use_gpu": False}
    metric = ASVSpoofMetric(config)
    
    # Test with None input
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, metadata={"sample_rate": 16000})


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists() 