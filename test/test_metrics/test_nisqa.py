#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Unit tests for NISQA metric."""

import wave
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from versa.utterance_metrics.nisqa import (
    NisqaMetric,
    nisqa_metric,
    nisqa_model_setup,
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
# Mock NISQA Model Fixture
# -------------------------------
@pytest.fixture
def mock_nisqa_model():
    """Create a mock NISQA model for testing."""
    model = Mock()
    model.device = "cpu"
    model.args = {"model": "NISQA"}
    return model


# -------------------------------
# Test NISQA Metric Class
# -------------------------------
class TestNisqaMetric:
    """Test the NisqaMetric class."""

    def test_initialization_without_model_path(self):
        """Test that initialization fails without model path."""
        config = {"use_gpu": False}
        with pytest.raises(ValueError, match="NISQA model path must be provided"):
            NisqaMetric(config)

    @patch("versa.utterance_metrics.nisqa.torch.load")
    @patch("versa.utterance_metrics.nisqa.NL.NISQA")
    def test_initialization_success(self, mock_nisqa_class, mock_torch_load):
        """Test successful initialization of NisqaMetric."""
        # Mock the checkpoint
        mock_checkpoint = {
            "args": {
                "model": "NISQA",
                "ms_seg_length": 15,
                "ms_n_mels": 48,
                "cnn_model": "resnet",
                "cnn_c_out_1": 32,
                "cnn_c_out_2": 32,
                "cnn_c_out_3": 32,
                "cnn_kernel_size": 3,
                "cnn_dropout": 0.1,
                "cnn_pool_1": 2,
                "cnn_pool_2": 2,
                "cnn_pool_3": 2,
                "cnn_fc_out_h": 128,
                "td": "lstm",
                "td_sa_d_model": 128,
                "td_sa_nhead": 8,
                "td_sa_pos_enc": "sin",
                "td_sa_num_layers": 2,
                "td_sa_h": 128,
                "td_sa_dropout": 0.1,
                "td_lstm_h": 128,
                "td_lstm_num_layers": 2,
                "td_lstm_dropout": 0.1,
                "td_lstm_bidirectional": True,
                "td_2": "lstm",
                "td_2_sa_d_model": 128,
                "td_2_sa_nhead": 8,
                "td_2_sa_pos_enc": "sin",
                "td_2_sa_num_layers": 2,
                "td_2_sa_h": 128,
                "td_2_sa_dropout": 0.1,
                "td_2_lstm_h": 128,
                "td_2_lstm_num_layers": 2,
                "td_2_lstm_dropout": 0.1,
                "td_2_lstm_bidirectional": True,
                "pool": "att",
                "pool_att_h": 128,
                "pool_att_dropout": 0.1,
            },
            "model_state_dict": {},
        }
        mock_torch_load.return_value = mock_checkpoint

        # Mock the NISQA model
        mock_model = Mock()
        mock_model.load_state_dict.return_value = ([], [])  # No missing/unexpected keys
        mock_nisqa_class.return_value = mock_model

        config = {
            "nisqa_model_path": "./tools/NISQA/weights/nisqa.tar",
            "use_gpu": False,
        }

        metric = NisqaMetric(config)
        assert metric.model is not None
        assert metric.model.device == "cpu"

    def test_compute_with_none_predictions(self):
        """Test that compute raises error with None predictions."""
        config = {
            "nisqa_model_path": "./tools/NISQA/weights/nisqa.tar",
            "use_gpu": False,
        }
        metric = NisqaMetric(config)

        with pytest.raises(ValueError, match="Predicted signal must be provided"):
            metric.compute(None)

    @patch("versa.utterance_metrics.nisqa.NL.versa_eval_mos")
    def test_compute_success(self, mock_eval_mos, mock_nisqa_model):
        """Test successful computation of NISQA scores."""
        # Mock the evaluation function
        mock_eval_mos.return_value = {
            "mos_pred": [[0.5]],
            "noi_pred": [[1.0]],
            "dis_pred": [[2.0]],
            "col_pred": [[1.5]],
            "loud_pred": [[1.2]],
        }

        config = {
            "nisqa_model_path": "./tools/NISQA/weights/nisqa.tar",
            "use_gpu": False,
        }
        metric = NisqaMetric(config)
        metric.model = mock_nisqa_model

        audio = np.random.random(16000)
        metadata = {"sample_rate": 16000}

        result = metric.compute(audio, metadata=metadata)

        assert "nisqa_mos_pred" in result
        assert "nisqa_noi_pred" in result
        assert "nisqa_dis_pred" in result
        assert "nisqa_col_pred" in result
        assert "nisqa_loud_pred" in result
        assert result["nisqa_mos_pred"] == 0.5

    def test_get_metadata(self):
        """Test that get_metadata returns correct metadata."""
        config = {
            "nisqa_model_path": "./tools/NISQA/weights/nisqa.tar",
            "use_gpu": False,
        }
        metric = NisqaMetric(config)

        metadata = metric.get_metadata()
        assert metadata.name == "nisqa"
        assert metadata.category.value == "independent"
        assert metadata.metric_type.value == "float"
        assert not metadata.requires_reference
        assert not metadata.requires_text
        assert metadata.gpu_compatible


# -------------------------------
# Integration Tests
# -------------------------------
@pytest.mark.integration
class TestNisqaIntegration:
    """Integration tests for NISQA metric."""

    @pytest.mark.parametrize(
        "sample_rate,use_gpu",
        [
            (16000, False),
            (22050, False),
            (48000, False),
        ],
    )
    def test_nisqa_with_different_sample_rates(self, sample_rate, use_gpu, fixed_audio):
        """Test NISQA with different sample rates."""
        # Skip if NISQA dependencies are not available
        try:
            import versa.utterance_metrics.nisqa_utils.nisqa_lib
        except ImportError:
            pytest.skip("NISQA dependencies not available")

        # This test would require a real NISQA model file
        # For now, we'll just test the basic structure
        config = {
            "use_gpu": use_gpu,
        }

        # Test that the metric can be instantiated (without actual model loading)
        with pytest.raises(ValueError, match="NISQA model path must be provided"):
            NisqaMetric(config)


# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()


# -------------------------------
# Test Registration Function
# -------------------------------
def test_register_nisqa_metric():
    """Test the registration function."""
    from versa.utterance_metrics.nisqa import register_nisqa_metric

    # Mock registry
    mock_registry = Mock()

    # Register the metric
    register_nisqa_metric(mock_registry)

    # Verify registration was called
    mock_registry.register.assert_called_once()

    # Verify the call arguments
    call_args = mock_registry.register.call_args
    assert call_args[0][0] == NisqaMetric  # First argument should be the class
    assert call_args[0][1].name == "nisqa"  # Second argument should be metadata
