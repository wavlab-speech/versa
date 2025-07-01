#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
# Copyright 2025 Jionghao Han
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
import wave
import numpy as np
import pytest

from versa.utterance_metrics.pseudo_mos import pseudo_mos_setup, pseudo_mos_metric


def generate_fixed_wav(
    filename, duration=1.0, sample_rate=16000, base_freq=150, envelope_func=None
):
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
# DNSMOS PRO Unit Test
# -------------------------------
def test_dnsmos_pro_metric_identical(fixed_audio):
    """
    Test that DNSMOS Pro returns valid scores for a known synthetic audio input.
    """
    predictor_dict, predictor_fs = pseudo_mos_setup(
        [
            "dnsmos_pro_bvcc",
            "dnsmos_pro_nisqa",
            "dnsmos_pro_vcc2018",
        ],
        predictor_args={},
    )
    scores = pseudo_mos_metric(
        fixed_audio, fs=16000, predictor_dict=predictor_dict, predictor_fs=predictor_fs
    )

    required_keys = {"dnsmos_pro_bvcc", "dnsmos_pro_nisqa", "dnsmos_pro_vcc2018"}
    assert required_keys.issubset(
        scores.keys()
    ), f"Missing expected DNSMOS keys. Got: {list(scores.keys())}"

    for key in required_keys:
        assert isinstance(scores[key], float), f"{key} should be a float"
        assert (
            1.0 <= scores[key] <= 5.0
        ), f"{key} score {scores[key]} out of valid MOS range [1, 5]"

    assert scores["dnsmos_pro_bvcc"] == pytest.approx(
        1.8937609195709229, rel=1e-3, abs=1e-6
    ), f"Expected dnsmos_pro_bvcc of 1.0 for identical signals, got {scores['dnsmos_pro_bvcc']}"

    assert scores["dnsmos_pro_nisqa"] == pytest.approx(
        1.7998836040496826, rel=1e-3, abs=1e-6
    ), f"Expected dnsmos_pro_nisqa of 1.0 for identical signals, got {scores['dnsmos_pro_nisqa']}"

    assert scores["dnsmos_pro_vcc2018"] == pytest.approx(
        2.113027334213257, rel=1e-3, abs=1e-6
    ), f"Expected dnsmos_pro_vcc2018 of 1.0 for identical signals, got {scores['dnsmos_pro_vcc2018']}"
