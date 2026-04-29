import pytest
import torch

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, find_files
from versa.sequence_metrics.signal_metric import register_signal_metric
from versa.utterance_metrics.scoreq import register_scoreq_metric
from versa.utterance_metrics.sheet_ssqa import register_sheet_ssqa_metric
from versa.utterance_metrics.squim import register_squim_metric
from versa.utterance_metrics.stoi import register_stoi_metric
from versa.utterance_metrics.vad import register_vad_metric
from versa.utterance_metrics.vqscore import register_vqscore_metric


def _sample_files():
    gen_files = find_files("test/test_samples/test2")
    gt_files = find_files("test/test_samples/test1")
    return gen_files, gt_files


def test_stoi_and_signal_pipeline_with_registry():
    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_stoi_metric(registry)
    register_signal_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "stoi"}, {"name": "signal_metric"}],
        use_gt=True,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        gt_files=gt_files,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert "stoi" in score_info[0]
    assert "sdr" in score_info[0]
    assert "ci_sdr" in score_info[0]


def test_squim_pipeline_with_registry_and_mocked_models(monkeypatch):
    class DummyObjectiveBundle:
        @staticmethod
        def get_model():
            return lambda pred_x: (
                torch.tensor([0.6]),
                torch.tensor([1.2]),
                torch.tensor([-3.4]),
            )

    class DummySubjectiveBundle:
        @staticmethod
        def get_model():
            return lambda pred_x, ref_x: torch.tensor([4.2])

    monkeypatch.setattr("versa.utterance_metrics.squim.SQUIM_AVAILABLE", True)
    monkeypatch.setattr(
        "versa.utterance_metrics.squim.SQUIM_OBJECTIVE", DummyObjectiveBundle
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.squim.SQUIM_SUBJECTIVE", DummySubjectiveBundle
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_squim_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "squim_no_ref"}, {"name": "squim_ref"}],
        use_gt=True,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        gt_files=gt_files,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert score_info[0]["torch_squim_stoi"] == 0.6
    assert score_info[0]["torch_squim_pesq"] == 1.2
    assert score_info[0]["torch_squim_si_sdr"] == -3.4
    assert score_info[0]["torch_squim_mos"] == 4.2


def test_vad_pipeline_with_registry_and_mocked_model(monkeypatch):
    def dummy_get_speech_ts(pred_x, model, **kwargs):
        return [{"start": 0.1, "end": 0.2}]

    monkeypatch.setattr(
        "versa.utterance_metrics.vad.torch.hub.load",
        lambda **kwargs: ("dummy-model", (dummy_get_speech_ts, None, None, None)),
    )

    gen_files, _ = _sample_files()
    registry = MetricRegistry()
    register_vad_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "vad"}],
        use_gt=False,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert score_info[0]["vad_info"] == [{"start": 0.1, "end": 0.2}]


def test_sheet_ssqa_pipeline_with_registry_and_mocked_model(monkeypatch):
    class DummyInnerModel:
        def to(self, device):
            return self

    class DummySheetModel:
        def __init__(self):
            self.model = DummyInnerModel()

        def predict(self, wav):
            return 3.25

    monkeypatch.setattr(
        "versa.utterance_metrics.sheet_ssqa.torch.hub.load",
        lambda *args, **kwargs: DummySheetModel(),
    )

    gen_files, _ = _sample_files()
    registry = MetricRegistry()
    register_sheet_ssqa_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "sheet_ssqa"}],
        use_gt=False,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert score_info[0]["sheet_ssqa"] == 3.25


def test_scoreq_pipeline_with_registry_and_mocked_model(monkeypatch):
    class DummyScoreq:
        def __init__(self, data_domain, mode, cache_dir, device):
            self.mode = mode

        def predict(self, test_path, ref_path):
            if self.mode == "ref":
                return 1.2
            return 2.4

    monkeypatch.setattr("versa.utterance_metrics.scoreq.Scoreq", DummyScoreq)

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_scoreq_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "scoreq_nr"}, {"name": "scoreq_ref"}],
        use_gt=True,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        gt_files=gt_files,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert score_info[0]["scoreq_nr"] == 2.4
    assert score_info[0]["scoreq_ref"] == 1.2


def test_vqscore_pipeline_with_registry_and_mocked_model(monkeypatch):
    class DummyVqscoreModel:
        device = "cpu"
        input_transform = "none"

        def CNN_1D_encoder(self, sp_input):
            return torch.ones((1, 2, 3))

        def quantizer(self, z, stochastic=False, update=False):
            return z.transpose(2, 1), None, None, None

    monkeypatch.setattr(
        "versa.utterance_metrics.vqscore.vqscore_setup",
        lambda use_gpu=False: DummyVqscoreModel(),
    )

    gen_files, _ = _sample_files()
    registry = MetricRegistry()
    register_vqscore_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "vqscore"}],
        use_gt=False,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert score_info[0]["vqscore"] == pytest.approx(1.0, abs=1e-4)
