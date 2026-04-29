import torch

from versa.definition import MetricRegistry
from versa.scorer_shared import VersaScorer, find_files
from versa.sequence_metrics.signal_metric import register_signal_metric
from versa.utterance_metrics.squim import register_squim_metric
from versa.utterance_metrics.stoi import register_stoi_metric


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
