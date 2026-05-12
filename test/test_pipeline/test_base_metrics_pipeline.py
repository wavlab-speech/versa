import json

import pytest
import torch

from versa.corpus_metrics.espnet_wer import register_espnet_wer_metric
from versa.corpus_metrics.owsm_wer import register_owsm_wer_metric
from versa.corpus_metrics.whisper_wer import register_whisper_wer_metric
from versa.definition import (
    BaseMetric,
    MetricCategory,
    MetricMetadata,
    MetricRegistry,
    MetricType,
)
from versa.scorer_shared import VersaScorer, find_files
from versa.sequence_metrics.mcd_f0 import register_mcd_f0_metric
from versa.sequence_metrics.signal_metric import register_signal_metric
from versa.sequence_metrics.warpq import register_warpq_metric
from versa.utterance_metrics.log_wmse import register_log_wmse_metric
from versa.utterance_metrics.pseudo_mos import register_pseudo_mos_metric
from versa.utterance_metrics.qwen2_audio import register_qwen2_audio_metric
from versa.utterance_metrics.qwen_omni import register_qwen_omni_metric
from versa.utterance_metrics.scoreq import register_scoreq_metric
from versa.utterance_metrics.se_snr import register_se_snr_metric
from versa.utterance_metrics.sheet_ssqa import register_sheet_ssqa_metric
from versa.utterance_metrics.singer import register_singer_metric
from versa.utterance_metrics.speaker import register_speaker_metric
from versa.utterance_metrics.squim import register_squim_metric
from versa.utterance_metrics.stoi import register_stoi_metric
from versa.utterance_metrics.vad import register_vad_metric
from versa.utterance_metrics.universa import register_universa_metric
from versa.utterance_metrics.visqol_score import register_visqol_metric
from versa.utterance_metrics.vqscore import register_vqscore_metric


def _sample_files():
    gen_files = find_files("test/test_samples/test2")
    gt_files = find_files("test/test_samples/test1")
    return gen_files, gt_files


class CountingMetric(BaseMetric):
    calls = 0

    def _setup(self):
        pass

    def compute(self, predictions, references=None, metadata=None):
        CountingMetric.calls += 1
        return 1.0

    def get_metadata(self):
        return MetricMetadata(
            name="counting",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=False,
            auto_install=False,
            dependencies=[],
            description="Test metric that counts compute calls.",
        )


def test_score_utterances_resume_skips_completed_keys(tmp_path):
    gen_files, _ = _sample_files()
    completed_key = next(iter(gen_files))
    output_file = tmp_path / "scores.jsonl"
    output_file.write_text(
        json.dumps({"key": completed_key, "counting": 3.0}) + "\n",
        encoding="utf-8",
    )

    registry = MetricRegistry()
    registry.register(CountingMetric, CountingMetric().get_metadata())
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics([{"name": "counting"}], use_gt=False)

    CountingMetric.calls = 0
    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        output_file=str(output_file),
        io="soundfile",
        resume=True,
    )

    assert score_info[0] == {"key": completed_key, "counting": 3.0}
    assert CountingMetric.calls == max(len(gen_files) - 1, 0)
    assert len(score_info) == len(gen_files)


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


def test_mcd_f0_pipeline_with_registry_and_mocked_metric(monkeypatch):
    monkeypatch.setattr(
        "versa.sequence_metrics.mcd_f0._ensure_mcd_f0_dependencies", lambda: None
    )
    monkeypatch.setattr(
        "versa.sequence_metrics.mcd_f0.mcd_f0",
        lambda pred_x, gt_x, fs, f0min, f0max, **kwargs: {
            "mcd": 1.2,
            "f0rmse": 3.4,
            "f0corr": 0.5,
        },
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_mcd_f0_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "mcd_f0"}],
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
    assert score_info[0]["mcd"] == 1.2
    assert score_info[0]["f0rmse"] == 3.4
    assert score_info[0]["f0corr"] == 0.5


def test_warpq_pipeline_with_registry_and_mocked_metric(monkeypatch):
    class DummyWarpqModel:
        args = {"sr": 16000}

    monkeypatch.setattr(
        "versa.sequence_metrics.warpq.warpq_setup",
        lambda **kwargs: DummyWarpqModel(),
    )
    monkeypatch.setattr(
        "versa.sequence_metrics.warpq.warpq",
        lambda model, pred_x, gt_x, fs=8000: {"warpq": 3.8},
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_warpq_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "warpq"}],
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
    assert score_info[0]["warpq"] == 3.8


def test_se_snr_pipeline_with_registry_and_mocked_metric(monkeypatch):
    class DummySeModel:
        pass

    monkeypatch.setattr(
        "versa.utterance_metrics.se_snr.se_snr_setup",
        lambda **kwargs: DummySeModel(),
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.se_snr.se_snr",
        lambda model, pred_x, fs: {
            "se_sdr": 1.0,
            "se_sar": 2.0,
            "se_si_snr": 3.0,
            "se_ci_sdr": 4.0,
        },
    )

    gen_files, _ = _sample_files()
    registry = MetricRegistry()
    register_se_snr_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "se_snr"}],
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
    assert score_info[0]["se_sdr"] == 1.0
    assert score_info[0]["se_sar"] == 2.0
    assert score_info[0]["se_si_snr"] == 3.0
    assert score_info[0]["se_ci_sdr"] == 4.0


def test_speaker_pipeline_with_registry_and_mocked_metric(monkeypatch):
    class DummySpeakerModel:
        pass

    monkeypatch.setattr(
        "versa.utterance_metrics.speaker.speaker_model_setup",
        lambda **kwargs: DummySpeakerModel(),
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.speaker.speaker_metric",
        lambda model, pred_x, gt_x, fs: {"spk_similarity": 0.75},
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_speaker_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "speaker"}],
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
    assert score_info[0]["spk_similarity"] == 0.75


def test_singer_pipeline_with_registry_and_mocked_metric(monkeypatch):
    class DummySingerModel:
        pass

    monkeypatch.setattr(
        "versa.utterance_metrics.singer.singer_model_setup",
        lambda **kwargs: DummySingerModel(),
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.singer.singer_metric",
        lambda model, pred_x, gt_x, fs, target_sr=44100: {"singer_similarity": 0.5},
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_singer_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "singer"}],
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
    assert score_info[0]["singer_similarity"] == 0.5


def test_log_wmse_pipeline_with_registry_and_mocked_model(monkeypatch):
    class DummyLogWMSE:
        def __init__(self, **kwargs):
            pass

        def __call__(self, unproc_x, proc_x, gt_x):
            return torch.tensor([0.33])

    monkeypatch.setattr("versa.utterance_metrics.log_wmse.LogWMSE", DummyLogWMSE)

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_log_wmse_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "log_wmse"}],
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
    assert score_info[0]["log_wmse"] == pytest.approx(0.33)


def test_pseudo_mos_pipeline_with_registry_and_mocked_metric(monkeypatch):
    monkeypatch.setattr(
        "versa.utterance_metrics.pseudo_mos.pseudo_mos_setup",
        lambda predictor_types, predictor_args, cache_dir, use_gpu: (
            {"utmos": object()},
            {"utmos": 16000},
        ),
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.pseudo_mos.pseudo_mos_metric",
        lambda pred, fs, predictor_dict, predictor_fs, use_gpu=False: {"utmos": 4.2},
    )

    gen_files, _ = _sample_files()
    registry = MetricRegistry()
    register_pseudo_mos_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "pseudo_mos", "predictor_types": ["utmos"]}],
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
    assert score_info[0]["utmos"] == 4.2


def test_universa_pipeline_with_registry_and_mocked_metric(monkeypatch):
    def dummy_universa_metric(
        audio_data, ref_audio=None, ref_text=None, original_sr=16000, ref_sr=None
    ):
        return {"universa_mos": 3.5}

    monkeypatch.setattr(
        "versa.utterance_metrics.universa.universa_metric",
        dummy_universa_metric,
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_universa_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "universa"}],
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
    assert score_info[0]["universa_mos"] == 3.5


def test_qwen_metric_pipelines_with_registry_and_mocked_models(monkeypatch):
    monkeypatch.setattr(
        "versa.utterance_metrics.qwen2_audio.qwen2_model_setup",
        lambda **kwargs: {"model": "qwen2"},
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.qwen2_audio.qwen2_base_metric",
        lambda qwen_utils, pred_x, fs=16000, custom_prompt=None, max_length=1000: (
            "young adult"
        ),
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.qwen_omni.qwen_omni_model_setup",
        lambda **kwargs: {"model": "omni"},
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.qwen_omni.qwen_omni_base_metric",
        lambda qwen_utils, pred_x, fs=16000, custom_prompt=None, max_length=500: (
            "happy"
        ),
    )

    gen_files, _ = _sample_files()
    registry = MetricRegistry()
    register_qwen2_audio_metric(registry)
    register_qwen_omni_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [
            {"name": "qwen2_audio_speaker_age"},
            {"name": "qwen_omni_speech_emotion"},
        ],
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
    assert score_info[0]["qwen_speaker_age"] == "young adult"
    assert score_info[0]["qwen_omni_speech_emotion"] == "happy"


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


def test_visqol_pipeline_with_registry_and_mocked_model(monkeypatch):
    class DummySimilarityResult:
        moslqo = 4.1

    class DummyApi:
        def Measure(self, gt_x, pred_x):
            return DummySimilarityResult()

    monkeypatch.setattr(
        "versa.utterance_metrics.visqol_score.visqol_setup",
        lambda model: (DummyApi(), 16000),
    )

    gen_files, gt_files = _sample_files()
    registry = MetricRegistry()
    register_visqol_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "visqol", "model": "speech"}],
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
    assert score_info[0]["visqol"] == 4.1


def test_wer_pipeline_with_registry_and_mocked_models(monkeypatch):
    monkeypatch.setattr(
        "versa.corpus_metrics.espnet_wer.espnet_wer_setup",
        lambda **kwargs: {"model": "espnet"},
    )
    monkeypatch.setattr(
        "versa.corpus_metrics.owsm_wer.owsm_wer_setup",
        lambda **kwargs: {"model": "owsm"},
    )
    monkeypatch.setattr(
        "versa.corpus_metrics.whisper_wer.whisper_wer_setup",
        lambda **kwargs: {"model": "whisper"},
    )
    monkeypatch.setattr(
        "versa.corpus_metrics.espnet_wer.espnet_levenshtein_metric",
        lambda wer_utils, pred_x, ref_text, fs=16000: {
            "espnet_hyp_text": ref_text,
            "espnet_wer_equal": 1,
        },
    )
    monkeypatch.setattr(
        "versa.corpus_metrics.owsm_wer.owsm_levenshtein_metric",
        lambda wer_utils, pred_x, ref_text, fs=16000: {
            "owsm_hyp_text": ref_text,
            "owsm_wer_equal": 1,
        },
    )
    monkeypatch.setattr(
        "versa.corpus_metrics.whisper_wer.whisper_levenshtein_metric",
        lambda wer_utils, pred_x, ref_text, fs=16000, cache_pred_text=None: {
            "whisper_hyp_text": cache_pred_text or ref_text,
            "whisper_wer_equal": 1,
        },
    )

    gen_files, _ = _sample_files()
    text_info = {key: "hello world" for key in gen_files}
    registry = MetricRegistry()
    register_espnet_wer_metric(registry)
    register_owsm_wer_metric(registry)
    register_whisper_wer_metric(registry)
    scorer = VersaScorer(registry)
    metric_suite = scorer.load_metrics(
        [{"name": "espnet_wer"}, {"name": "owsm_wer"}, {"name": "whisper_wer"}],
        use_gt=False,
        use_gt_text=True,
        use_gpu=False,
    )

    score_info = scorer.score_utterances(
        gen_files,
        metric_suite,
        text_info=text_info,
        output_file=None,
        io="soundfile",
    )

    assert score_info
    assert score_info[0]["espnet_hyp_text"] == "hello world"
    assert score_info[0]["owsm_hyp_text"] == "hello world"
    assert score_info[0]["whisper_hyp_text"] == "hello world"
