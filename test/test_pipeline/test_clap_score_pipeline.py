import numpy as np
import yaml

from versa.corpus_metrics import clap_score
from versa.definition import MetricCategory
from versa.scorer_shared import VersaScorer, audio_loader_setup


class FakeCLAPScore:
    sample_rate = 16000

    def __init__(self, **kwargs):
        pass

    def get_text_embeddings(self, text_data):
        return np.ones((len(text_data), 2), dtype=np.float32)

    def get_audio_embeddings(self, audio_data, sr):
        return np.ones((len(audio_data), 2), dtype=np.float32)

    def calculate_clap_score(self, text_embds, audio_embds, batch_size):
        return 1.0, 0.0


def test_clap_score_pipeline_registration(monkeypatch, tmp_path):
    import soundfile as sf

    monkeypatch.setattr(clap_score, "CLAPScore", FakeCLAPScore)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    sf.write(audio_dir / "test.wav", np.ones(16000, dtype=np.float32) * 0.1, 16000)

    with open("egs/separate_metrics/clap_score.yaml", "r", encoding="utf-8") as f:
        score_config = yaml.safe_load(f)
    score_config[0]["cache_dir"] = str(tmp_path / "cache")

    scorer = VersaScorer()
    score_modules = scorer.load_metrics(
        score_config,
        use_gt=False,
        use_gt_text=True,
        use_gpu=False,
    )
    score_info = scorer.score_corpus(
        audio_loader_setup(str(audio_dir), "dir"),
        score_modules.filter_by_category(MetricCategory.DISTRIBUTIONAL),
        text_info={"test.wav": "speech audio"},
    )

    assert score_info["clap_score"] == 1.0
