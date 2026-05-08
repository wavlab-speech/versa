import numpy as np
import pytest

from versa.corpus_metrics import clap_score


class FakeCLAPScore:
    sample_rate = 16000

    def get_text_embeddings(self, text_data):
        return np.array(
            [[1.0, 0.0] if text == "low tone" else [0.0, 1.0] for text in text_data]
        )

    def get_audio_embeddings(self, audio_data, sr):
        return np.array(
            [[1.0, 0.0] if np.mean(audio) < 0.0 else [0.0, 1.0] for audio in audio_data]
        )

    def calculate_clap_score(self, text_embds, audio_embds, batch_size):
        text_norm = text_embds / np.linalg.norm(text_embds, axis=-1, keepdims=True)
        audio_norm = audio_embds / np.linalg.norm(audio_embds, axis=-1, keepdims=True)
        return np.mean(np.sum(text_norm * audio_norm, axis=-1)), 0.0


def test_clap_score_setup_requires_dependency(monkeypatch):
    monkeypatch.setattr(clap_score, "CLAPScore", None)
    with pytest.raises(ModuleNotFoundError, match="frechet_audio_distance"):
        clap_score.clap_score_setup()


def test_clap_score_scoring_matches_text_by_key(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    first = audio_dir / "b.wav"
    second = audio_dir / "a.wav"

    import soundfile as sf

    sf.write(first, -np.ones(16000, dtype=np.float32) * 0.1, 16000)
    sf.write(second, np.ones(16000, dtype=np.float32) * 0.1, 16000)

    clap_info = {
        "module": FakeCLAPScore(),
        "cache_dir": str(tmp_path / "cache"),
        "cache_embeddings": False,
        "io": "dir",
    }
    scores = clap_score.clap_score_scoring(
        str(audio_dir),
        clap_info,
        text_info={"a.wav": "high tone", "b.wav": "low tone"},
    )

    assert scores["clap_score"] == pytest.approx(1.0)


def test_clap_score_scoring_requires_text(tmp_path):
    clap_info = {
        "module": FakeCLAPScore(),
        "cache_dir": str(tmp_path / "cache"),
        "cache_embeddings": False,
        "io": "dir",
    }
    with pytest.raises(ValueError, match="requires text"):
        clap_score.clap_score_scoring(str(tmp_path), clap_info, text_info=None)
