#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unit tests for NOMAD metric."""

import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from versa.utterance_metrics.nomad import (
    NomadMetric,
    is_nomad_available,
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
# Mock NOMAD Model Fixture
# -------------------------------
@pytest.fixture
def mock_nomad_model():
    """Create a mock NOMAD model for testing."""
    model = Mock()
    model.predict.return_value = 0.5  # Mock prediction value
    return model


# -------------------------------
# Test NOMAD Metric Class
# -------------------------------
class TestNomadMetric:
    """Test the NomadMetric class."""

    def test_initialization_without_nomad(self):
        """Test that initialization fails without nomad dependency."""
        with patch("versa.utterance_metrics.nomad.NOMAD_AVAILABLE", False):
            config = {"use_gpu": False, "model_cache": "test_cache"}
            with pytest.raises(ImportError, match="nomad is not installed"):
                NomadMetric(config)

    @patch("versa.utterance_metrics.nomad.Nomad")
    def test_initialization_success(self, mock_nomad_class):
        """Test successful initialization of NomadMetric."""
        # Mock the NOMAD class
        mock_model = Mock()
        mock_nomad_class.return_value = mock_model

        config = {
            "use_gpu": False,
            "model_cache": "test_cache",
        }

        metric = NomadMetric(config)
        assert metric.model is not None
        mock_nomad_class.assert_called_once_with(device="cpu", cache_dir="test_cache")

    def test_compute_with_none_predictions(self):
        """Test that compute raises error with None predictions."""
        with patch("versa.utterance_metrics.nomad.Nomad") as mock_nomad_class:
            mock_model = Mock()
            mock_nomad_class.return_value = mock_model

            config = {"use_gpu": False, "model_cache": "test_cache"}
            metric = NomadMetric(config)

            with pytest.raises(ValueError, match="Predicted signal must be provided"):
                metric.compute(None, np.random.random(16000))

    def test_compute_with_none_references(self):
        """Test that compute raises error with None references."""
        with patch("versa.utterance_metrics.nomad.Nomad") as mock_nomad_class:
            mock_model = Mock()
            mock_nomad_class.return_value = mock_model

            config = {"use_gpu": False, "model_cache": "test_cache"}
            metric = NomadMetric(config)

            with pytest.raises(ValueError, match="Reference signal must be provided"):
                metric.compute(np.random.random(16000), None)

    @patch("versa.utterance_metrics.nomad.librosa.resample")
    def test_compute_success(self, mock_resample, mock_nomad_model):
        """Test successful computation of NOMAD score."""
        # Mock the resample function
        mock_resample.side_effect = lambda x, orig_sr, target_sr: x

        config = {"use_gpu": False, "model_cache": "test_cache"}
        metric = NomadMetric(config)
        metric.model = mock_nomad_model

        audio = np.random.random(16000)
        gt_audio = np.random.random(16000)
        metadata = {"sample_rate": 16000}

        result = metric.compute(audio, gt_audio, metadata=metadata)

        assert "nomad" in result
        assert result["nomad"] == 0.5
        mock_nomad_model.predict.assert_called_once()

    @patch("versa.utterance_metrics.nomad.librosa.resample")
    def test_compute_with_resampling(self, mock_resample, mock_nomad_model):
        """Test computation with resampling."""
        # Mock the resample function
        mock_resample.side_effect = lambda x, orig_sr, target_sr: x

        config = {"use_gpu": False, "model_cache": "test_cache"}
        metric = NomadMetric(config)
        metric.model = mock_nomad_model

        audio = np.random.random(8000)  # Different sample rate
        gt_audio = np.random.random(8000)
        metadata = {"sample_rate": 8000}

        result = metric.compute(audio, gt_audio, metadata=metadata)

        assert "nomad" in result
        # Verify resampling was called
        assert mock_resample.call_count == 2

    def test_get_metadata(self):
        """Test that get_metadata returns correct metadata."""
        with patch("versa.utterance_metrics.nomad.Nomad") as mock_nomad_class:
            mock_model = Mock()
            mock_nomad_class.return_value = mock_model

            config = {"use_gpu": False, "model_cache": "test_cache"}
            metric = NomadMetric(config)

            metadata = metric.get_metadata()
            assert metadata.name == "nomad"
            assert metadata.category.value == "dependent"
            assert metadata.metric_type.value == "float"
            assert metadata.requires_reference
            assert not metadata.requires_text
            assert metadata.gpu_compatible


# -------------------------------
# Test Utility Functions
# -------------------------------
class TestUtilityFunctions:
    """Test utility functions."""

    @patch("versa.utterance_metrics.nomad.NOMAD_AVAILABLE", True)
    def test_is_nomad_available_true(self):
        """Test is_nomad_available when NOMAD is available."""
        assert is_nomad_available() is True

    @patch("versa.utterance_metrics.nomad.NOMAD_AVAILABLE", False)
    def test_is_nomad_available_false(self):
        """Test is_nomad_available when NOMAD is not available."""
        assert is_nomad_available() is False


# -------------------------------
# Integration Tests
# -------------------------------
@pytest.mark.integration
class TestNomadIntegration:
    """Integration tests for NOMAD metric."""

    @pytest.mark.parametrize(
        "sample_rate,use_gpu",
        [
            (16000, False),
            (22050, False),
            (48000, False),
        ],
    )
    def test_nomad_with_different_sample_rates(
        self, sample_rate, use_gpu, fixed_audio, fixed_ground_truth
    ):
        """Test NOMAD with different sample rates."""
        # Skip if NOMAD dependencies are not available
        if not is_nomad_available():
            pytest.skip("NOMAD dependencies not available")

        # This test would require a real NOMAD model file
        # For now, we'll just test the basic structure
        config = {
            "use_gpu": use_gpu,
            "model_cache": "test_cache",
        }

        # Test that the metric can be instantiated (without actual model loading)
        with patch("versa.utterance_metrics.nomad.Nomad") as mock_nomad_class:
            mock_model = Mock()
            mock_nomad_class.return_value = mock_model

            metric = NomadMetric(config)
            assert metric.model is not None


# -------------------------------
# Example Test Function Using the Reused WAV Files
# -------------------------------
@pytest.mark.parametrize(
    "use_gpu,cache_dir",
    [
        (False, "test_cache"),
        (True, "test_cache"),
    ],
)
def test_utterance_nomad(use_gpu, cache_dir, fixed_audio, fixed_ground_truth):
    """
    Test the NOMAD metric using the fixed audio and ground truth.
    The test uses deterministic data so that the result is always reproducible.
    """
    with patch("versa.utterance_metrics.nomad.Nomad") as mock_nomad_class:
        mock_model = Mock()
        mock_model.predict.return_value = 0.5
        mock_nomad_class.return_value = mock_model

        # Use the new class-based API
        config = {"use_gpu": use_gpu, "model_cache": cache_dir}
        metric = NomadMetric(config)
        metadata = {"sample_rate": 16000}
        result = metric.compute(fixed_audio, fixed_ground_truth, metadata=metadata)
        nomad_score = result["nomad"]

        # We expect the score to be 0.5 based on our mock
        assert nomad_score == pytest.approx(
            0.5, rel=1e-3, abs=1e-6
        ), "value from nomad_score {} is mismatch from the defined one {}".format(
            nomad_score, 0.5
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


# -------------------------------
# Test Registration Function
# -------------------------------
def test_register_nomad_metric():
    """Test the registration function."""
    from versa.utterance_metrics.nomad import register_nomad_metric

    # Mock registry
    mock_registry = Mock()

    # Register the metric
    register_nomad_metric(mock_registry)

    # Verify registration was called
    mock_registry.register.assert_called_once()

    # Verify the call arguments
    call_args = mock_registry.register.call_args
    assert call_args[0][0] == NomadMetric  # First argument should be the class
    assert call_args[0][1].name == "nomad"  # Second argument should be metadata
