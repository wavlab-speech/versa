import wave
from pathlib import Path

import numpy as np
import pytest
import torch
from packaging.version import parse as V

from versa.utterance_metrics.pam import PamMetric, PAM, is_pam_available


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
def fixed_audio_44k_wav(tmp_path_factory):
    """
    Create a fixed WAV file with 44.1kHz sample rate to test resampling.
    """
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio_44k.wav"
    # Generate an audio file with a 150 Hz sine wave at 44.1kHz.
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=44100, base_freq=150)
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
def fixed_audio_44k(fixed_audio_44k_wav):
    """
    Load the fixed 44.1kHz audio file as a NumPy array.
    """
    return load_wav_as_array(fixed_audio_44k_wav, sample_rate=44100)


# -------------------------------
# Test Functions
# -------------------------------
@pytest.mark.skipif(
    not is_pam_available(),
    reason="PAM dependencies are not available",
)
@pytest.mark.parametrize(
    "repro,use_gpu",
    [
        (True, False),
        (False, False),
    ],
)
def test_pam_metric_basic(repro, use_gpu, fixed_audio):
    """
    Test the PAM metric with basic configuration.
    """
    config = {
        "repro": repro,
        "use_gpu": use_gpu,
        "cache_dir": "test_cache/pam",
        "text_model": "gpt2",
        "text_len": 77,
        "transformer_embed_dim": 768,
        "audioenc_name": "HTSAT",
        "out_emb": 768,
        "sampling_rate": 44100,
        "duration": 7,
        "fmin": 50,
        "fmax": 8000,
        "n_fft": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "window_size": 1024,
        "d_proj": 1024,
        "temperature": 0.003,
        "num_classes": 527,
        "batch_size": 1024,
        "demo": False,
    }

    metric = PamMetric(config)
    result = metric.compute(fixed_audio, metadata={"sample_rate": 16000})

    # Check that result contains pam_score field
    assert "pam_score" in result
    assert isinstance(result["pam_score"], (int, float, np.number))
    assert not np.isnan(result["pam_score"])
    assert not np.isinf(result["pam_score"])
    # PAM score should be between 0 and 1
    assert 0.0 <= result["pam_score"] <= 1.0


@pytest.mark.skipif(
    not is_pam_available(),
    reason="PAM dependencies are not available",
)
def test_pam_metric_resampling(fixed_audio_44k):
    """
    Test the PAM metric with audio that needs resampling.
    """
    config = {
        "repro": True,
        "use_gpu": False,
        "cache_dir": "test_cache/pam",
        "text_model": "gpt2",
        "text_len": 77,
        "transformer_embed_dim": 768,
        "audioenc_name": "HTSAT",
        "out_emb": 768,
        "sampling_rate": 44100,
        "duration": 7,
        "fmin": 50,
        "fmax": 8000,
        "n_fft": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "window_size": 1024,
        "d_proj": 1024,
        "temperature": 0.003,
        "num_classes": 527,
        "batch_size": 1024,
        "demo": False,
    }

    metric = PamMetric(config)
    result = metric.compute(fixed_audio_44k, metadata={"sample_rate": 44100})

    # Check that result contains pam_score field
    assert "pam_score" in result
    assert isinstance(result["pam_score"], (int, float, np.number))
    assert not np.isnan(result["pam_score"])
    assert not np.isinf(result["pam_score"])
    # PAM score should be between 0 and 1
    assert 0.0 <= result["pam_score"] <= 1.0


@pytest.mark.skipif(
    not is_pam_available(),
    reason="PAM dependencies are not available",
)
def test_pam_metric_invalid_input():
    """
    Test the PAM metric with invalid input.
    """
    config = {
        "repro": True,
        "use_gpu": False,
        "cache_dir": "test_cache/pam",
        "text_model": "gpt2",
        "text_len": 77,
        "transformer_embed_dim": 768,
        "audioenc_name": "HTSAT",
        "out_emb": 768,
        "sampling_rate": 44100,
        "duration": 7,
        "fmin": 50,
        "fmax": 8000,
        "n_fft": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "window_size": 1024,
        "d_proj": 1024,
        "temperature": 0.003,
        "num_classes": 527,
        "batch_size": 1024,
        "demo": False,
    }

    metric = PamMetric(config)

    # Test with None input
    with pytest.raises(ValueError, match="Predicted signal must be provided"):
        metric.compute(None, metadata={"sample_rate": 16000})


@pytest.mark.skipif(
    not is_pam_available(),
    reason="PAM dependencies are not available",
)
def test_pam_metric_metadata():
    """
    Test the PAM metric metadata.
    """
    config = {
        "repro": True,
        "use_gpu": False,
        "cache_dir": "test_cache/pam",
        "text_model": "gpt2",
        "text_len": 77,
        "transformer_embed_dim": 768,
        "audioenc_name": "HTSAT",
        "out_emb": 768,
        "sampling_rate": 44100,
        "duration": 7,
        "fmin": 50,
        "fmax": 8000,
        "n_fft": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "window_size": 1024,
        "d_proj": 1024,
        "temperature": 0.003,
        "num_classes": 527,
        "batch_size": 1024,
        "demo": False,
    }

    metric = PamMetric(config)
    metadata = metric.get_metadata()

    assert metadata.name == "pam"
    assert metadata.category.value == "independent"
    assert metadata.metric_type.value == "float"
    assert metadata.requires_reference is False
    assert metadata.requires_text is False
    assert metadata.gpu_compatible is True
    assert metadata.auto_install is False
    assert "torch" in metadata.dependencies
    assert "torchaudio" in metadata.dependencies
    assert "transformers" in metadata.dependencies
    assert "huggingface_hub" in metadata.dependencies
    assert "numpy" in metadata.dependencies


def test_pam_metric_not_available():
    """
    Test the PAM metric when PAM dependencies are not available.
    """
    # This test should be skipped if PAM is available
    if is_pam_available():
        pytest.skip("PAM dependencies are available, skipping this test")

    config = {
        "repro": True,
        "use_gpu": False,
        "cache_dir": "test_cache/pam",
        "text_model": "gpt2",
        "text_len": 77,
        "transformer_embed_dim": 768,
        "audioenc_name": "HTSAT",
        "out_emb": 768,
        "sampling_rate": 44100,
        "duration": 7,
        "fmin": 50,
        "fmax": 8000,
        "n_fft": 1024,
        "hop_size": 320,
        "mel_bins": 64,
        "window_size": 1024,
        "d_proj": 1024,
        "temperature": 0.003,
        "num_classes": 527,
        "batch_size": 1024,
        "demo": False,
    }

    with pytest.raises(RuntimeError, match="Failed to initialize PAM model"):
        PamMetric(config)


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav, fixed_audio_44k_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_audio_44k_wav).exists()
