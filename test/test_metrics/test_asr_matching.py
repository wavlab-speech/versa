import numpy as np
import wave
from pathlib import Path
from packaging.version import parse as V
from versa.utterance_metrics.asr_matching import asr_match_setup, asr_match_metric
import torch

# -------------------------------
# Helper: Generate a fixed WAV file
# -------------------------------
def generate_fixed_wav(filename, duration=1.0, sample_rate=16000, base_freq=150, envelope_func=None):
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
    with wave.open(str(filename), 'w') as wf:
        wf.setnchannels(1)        # Mono audio.
        wf.setsampwidth(2)        # 16 bits per sample.
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
    with wave.open(str(wav_path), 'rb') as wf:
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
# Example Test Function Using the Reused WAV Files
# -------------------------------
@pytest.mark.parametrize(
    "model_tag,beam_size,text_cleaner,cache_text",
    [
        ("tiny", 1, "whisper_basic", None),
        ("tiny", 2, "whisper_en", None),
        ("tiny", 1, "whisper_en", "already_text"),
    ],
)
def test_utterance_asr_matching(model_tag, beam_size, text_cleaner, cache_text, fixed_audio, fixed_ground_truth):
    """
    Test the ASR matching metric using the fixed audio and ground truth.
    The test uses deterministic data so that the result is always reproducible.
    """
    wer_utils = asr_match_setup(model_tag, beam_size, text_cleaner, use_gpu=False)
    result = asr_match_metric(wer_utils, fixed_audio, fixed_ground_truth, cache_text, 16000)
    asr_match_error_rate = result["asr_match_error_rate"]
    
    # We expect the error rate to be 1.0 based on the intentional differences in the signals.
    assert asr_match_error_rate == pytest.approx(
        1.0, rel=1e-3, abs=1e-6
    ), ("value from asr_match_error_rate {} is mismatch from the defined one {}"
         .format(asr_match_error_rate, 1))

# -------------------------------
# Additional Example Test to Verify the File Creation (Optional)
# -------------------------------
def test_fixed_wav_files_exist(fixed_audio_wav, fixed_ground_truth_wav):
    """
    Verify that the fixed WAV files were created.
    """
    assert Path(fixed_audio_wav).exists()
    assert Path(fixed_ground_truth_wav).exists()

