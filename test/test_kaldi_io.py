"""Verify Kaldi audio compatibility through VERSA's OmniIO-backed loader."""

import io
import shlex
import wave

import numpy as np
import pytest
from omniio import kaldi as kaldi_io

from versa.scorer_shared import _utterance_input_signatures, audio_loader_setup
from versa.utils_shared import load_audio, wav_normalize


def _wav_bytes(samples, sample_rate):
    """Build an independent PCM WaveHolder fixture using the WAV standard."""
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
        writer.setsampwidth(2)
        writer.setframerate(sample_rate)
        writer.writeframes(samples.astype("<i2").tobytes())
    return buffer.getvalue()


@pytest.mark.parametrize("channels", [1, 2])
@pytest.mark.parametrize("layout", ["waveholder", "extended"])
def test_existing_archive_offsets_are_lazy_and_preserve_audio(
    tmp_path, monkeypatch, channels, layout
):
    """Read native RIFF and extended AUDIO entries without a writer dependency."""
    samples = np.array([-30000, -123, 0, 123, 30000], dtype=np.int16)
    if channels == 2:
        samples = np.column_stack((samples, -samples))
    sample_rate = 16000
    payload = _wav_bytes(samples, sample_rate)
    if layout == "extended":
        # Extended archives frame the encoded audio with a length-prefixed AUDIO tag.
        payload = b"AUDIO" + b"\x04" + len(payload).to_bytes(4, "little") + payload
    archive = tmp_path / "existing.ark"
    scp = tmp_path / "wav.scp"
    with archive.open("wb") as writer, scp.open("w") as index:
        for key in ("first", "second"):
            writer.write((key + " ").encode())
            index.write(f"{key} {archive}:{writer.tell()}\n")
            writer.write(payload)

    mapping = audio_loader_setup(str(scp), "kaldi")

    def unexpected_read(*args, **kwargs):
        """Reject waveform reads while the scorer plans resume identities."""
        raise AssertionError("resume planning decoded archive audio")

    with monkeypatch.context() as patch:
        patch.setattr(type(mapping), "__getitem__", unexpected_read)
        assert list(mapping) == ["first", "second"]
        assert "first" in mapping
        for policy in ("path", "content"):
            signatures = _utterance_input_signatures(mapping, None, None, policy)
            assert set(signatures) == {"first", "second"}

    for key in mapping:
        rate, actual = load_audio(mapping[key], "kaldi")
        assert rate == sample_rate
        # AUDIO decoding returns floats on the PCM scale; WaveHolder returns int16.
        divisor = 32768.0 if layout == "extended" else 32767.0
        expected = (samples[:, 0] if channels == 2 else samples) / divisor
        np.testing.assert_array_equal(wav_normalize(actual), expected)


@pytest.mark.parametrize("extended", [False, True])
def test_omniio_writer_archives_load_through_versa(tmp_path, extended):
    """Round-trip both audio archive layouts through sequential and lazy readers."""
    samples = np.array([-30000, -20, 0, 20, 30000], dtype=np.int16)
    archive, scp = tmp_path / "audio.ark", tmp_path / "audio.scp"
    options = {"write_function": "soundfile"} if extended else {}
    with kaldi_io.WriteHelper(f"ark,scp:{archive},{scp}", **options) as writer:
        writer["one"] = (16000, samples)
        writer["two"] = (8000, samples[::-1])
    mapping = audio_loader_setup(str(scp), "kaldi")
    with kaldi_io.ReadHelper(f"ark:{archive}") as reader:
        sequential = dict(reader)
    for key, expected_rate, expected in (
        ("one", 16000, samples),
        ("two", 8000, samples[::-1]),
    ):
        rate, actual = load_audio(mapping[key], "kaldi")
        assert rate == expected_rate == sequential[key][0]
        expected_normalized = expected / (32768.0 if extended else 32767.0)
        np.testing.assert_array_equal(wav_normalize(actual), expected_normalized)
        np.testing.assert_array_equal(
            wav_normalize(sequential[key][1]), expected_normalized
        )


def test_wav_scp_paths_and_command_pipes(tmp_path):
    """Keep direct WAV files and shell pipe entries usable with --io kaldi."""
    samples = np.array([-30000, 0, 30000], dtype=np.int16)
    wav = tmp_path / "audio with spaces.wav"
    wav.write_bytes(_wav_bytes(samples, 22050))
    scp = tmp_path / "wav.scp"
    scp.write_text(f"file {wav}\npipe cat {shlex.quote(str(wav))} |\n")
    mapping = audio_loader_setup(str(scp), "kaldi")
    for key in ("file", "pipe"):
        rate, actual = load_audio(mapping[key], "kaldi")
        assert rate == 22050
        np.testing.assert_array_equal(actual, samples)
