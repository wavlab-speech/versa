import wave
from pathlib import Path

import numpy as np
import pytest

from versa.utterance_metrics.emo_vad import dim_emo_pred, w2v2_emo_dim_setup

# Assume the fixed WAV file fixtures and helper function are defined as in the dimentionsal emotion prediction test.
# For example:


def generate_fixed_wav(
    filename, duration=1.0, sample_rate=16000, base_freq=150, envelope_func=None
):
    """
    Generate a deterministic WAV file with a modulated sine wave.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    if envelope_func is None:
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.5 * t)
    else:
        envelope = envelope_func(t)
    audio = envelope * np.sin(2 * np.pi * base_freq * t)
    amplitude = np.iinfo(np.int16).max
    data = (audio * amplitude).astype(np.int16)
    with wave.open(str(filename), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(data.tobytes())


def load_wav_as_array(wav_path, sample_rate=16000):
    """
    Load a WAV file and convert it to a NumPy array scaled to [-1, 1].
    """
    with wave.open(str(wav_path), "rb") as wf:
        frames = wf.getnframes()
        audio_data = wf.readframes(frames)
    audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    return audio_array / np.iinfo(np.int16).max


@pytest.fixture(scope="session")
def fixed_audio_wav(tmp_path_factory):
    tmp_dir = tmp_path_factory.mktemp("audio_data")
    audio_file = tmp_dir / "fixed_audio.wav"
    generate_fixed_wav(audio_file, duration=1.0, sample_rate=16000, base_freq=150)
    return audio_file


@pytest.fixture(scope="session")
def fixed_audio(fixed_audio_wav):
    return load_wav_as_array(fixed_audio_wav)


# -------------------------------
# emo_vad Metric Definition and Tests
# -------------------------------
def test_emo_vad_metric_identical(fixed_audio):
    """
    When comparing an audio signal with itself, the STOI score should be 1.0.
    """
    emo_utils = w2v2_emo_dim_setup()
    scores = dim_emo_pred(emo_utils, fixed_audio, 16000)
    assert scores["aro_val_dom_emo"] == pytest.approx(
        np.array([0.3982302, 0.43092448, 0.41154572], dtype=np.float32),
        rel=1e-3,
        abs=1e-6,
    ), f"Expected aro_val_dom_emo of [0.3982302, 0.43092448, 0.41154572] for identical signals, got {scores['aro_val_dom_emo']}"
