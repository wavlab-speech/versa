import numpy as np
import pytest
import torch

from versa.corpus_metrics.espnet_wer import (
    EspnetWerMetric,
    register_espnet_wer_metric,
)
from versa.corpus_metrics.fwhisper_wer import (
    FasterWhisperWerMetric,
    register_fwhisper_wer_metric,
)
from versa.corpus_metrics.hubert_wer import HubertWerMetric, register_hubert_wer_metric
from versa.corpus_metrics.nemo_wer import NemoWerMetric, register_nemo_wer_metric
from versa.corpus_metrics.owsm_wer import OwsmWerMetric, register_owsm_wer_metric
from versa.corpus_metrics.whisper_wer import (
    WhisperWerMetric,
    register_whisper_wer_metric,
)
from versa.definition import MetricRegistry
from versa.sequence_metrics.mcd_f0 import McdF0Metric, register_mcd_f0_metric
from versa.sequence_metrics.signal_metric import SignalMetric, register_signal_metric
from versa.sequence_metrics.warpq import WarpqMetric, register_warpq_metric
from versa.utterance_metrics.log_wmse import LogWmseMetric, register_log_wmse_metric
from versa.utterance_metrics.pseudo_mos import (
    PseudoMosMetric,
    register_pseudo_mos_metric,
)
from versa.utterance_metrics.pysepm import PysepmMetric, register_pysepm_metric
from versa.utterance_metrics.qwen2_audio import (
    QWEN2_AUDIO_METRIC_CLASSES,
    register_qwen2_audio_metric,
)
from versa.utterance_metrics.qwen_omni import (
    QWEN_OMNI_METRIC_CLASSES,
    register_qwen_omni_metric,
)
from versa.utterance_metrics.scoreq import (
    ScoreqNrMetric,
    ScoreqRefMetric,
    register_scoreq_metric,
)
from versa.utterance_metrics.se_snr import SeSnrMetric, register_se_snr_metric
from versa.utterance_metrics.sheet_ssqa import (
    SheetSsqaMetric,
    register_sheet_ssqa_metric,
)
from versa.utterance_metrics.singer import SingerMetric, register_singer_metric
from versa.utterance_metrics.speaker import SpeakerMetric, register_speaker_metric
from versa.utterance_metrics.squim import (
    SquimNoRefMetric,
    SquimRefMetric,
    register_squim_metric,
)
from versa.utterance_metrics.vad import VadMetric, register_vad_metric
from versa.utterance_metrics.universa import UniversaMetric, register_universa_metric
from versa.utterance_metrics.visqol_score import VisqolMetric, register_visqol_metric
from versa.utterance_metrics.vqscore import VqscoreMetric, register_vqscore_metric


def _audio_pair(length=16000):
    t = np.linspace(0, 1, length, endpoint=False)
    pred = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    ref = 0.5 * np.sin(2 * np.pi * 221 * t).astype(np.float32)
    return pred, ref


def test_signal_metric_class_returns_existing_keys():
    pred, ref = _audio_pair()
    metric = SignalMetric()

    scores = metric.compute(pred, ref)

    assert set(scores) == {"sdr", "sir", "sar", "si_snr", "ci_sdr"}
    assert all(isinstance(value, (float, np.floating)) for value in scores.values())


def test_mcd_f0_metric_class_returns_existing_keys(monkeypatch):
    calls = {}

    def dummy_mcd_f0(pred_x, gt_x, fs, f0min, f0max, **kwargs):
        calls["fs"] = fs
        calls["f0min"] = f0min
        calls["f0max"] = f0max
        calls["kwargs"] = kwargs
        return {"mcd": 1.2, "f0rmse": 3.4, "f0corr": 0.5}

    monkeypatch.setattr(
        "versa.sequence_metrics.mcd_f0._ensure_mcd_f0_dependencies", lambda: None
    )
    monkeypatch.setattr("versa.sequence_metrics.mcd_f0.mcd_f0", dummy_mcd_f0)

    pred, ref = _audio_pair()
    metric = McdF0Metric({"f0min": 50, "f0max": 700, "dtw": True})
    scores = metric.compute(pred, ref, metadata={"sample_rate": 22050})

    assert scores == {"mcd": 1.2, "f0rmse": 3.4, "f0corr": 0.5}
    assert calls["fs"] == 22050
    assert calls["f0min"] == 50
    assert calls["f0max"] == 700
    assert calls["kwargs"]["dtw"] is True


def test_register_mcd_f0_metric():
    registry = MetricRegistry()

    register_mcd_f0_metric(registry)

    assert registry.get_metric("mcd_f0") is McdF0Metric
    assert registry.get_metric("mcd") is McdF0Metric
    assert registry.get_metadata("mcd_f0").requires_reference is True


def test_mcd_f0_missing_dependency(monkeypatch):
    monkeypatch.setattr(
        "versa.sequence_metrics.mcd_f0._ensure_mcd_f0_dependencies",
        lambda: (_ for _ in ()).throw(ImportError("mcd_f0 requires pysptk")),
    )

    with pytest.raises(ImportError, match="mcd_f0 requires"):
        McdF0Metric()


def test_warpq_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    class DummyWarpqModel:
        args = {"sr": 16000}

    def dummy_warpq_setup(**kwargs):
        calls["setup"] = kwargs
        return DummyWarpqModel()

    def dummy_warpq(model, pred_x, gt_x, fs=8000):
        calls["compute_fs"] = fs
        return {"warpq": 3.8}

    monkeypatch.setattr("versa.sequence_metrics.warpq.warpq_setup", dummy_warpq_setup)
    monkeypatch.setattr("versa.sequence_metrics.warpq.warpq", dummy_warpq)

    pred, ref = _audio_pair()
    metric = WarpqMetric({"fs": 16000, "n_mfcc": 20, "apply_vad": True})
    scores = metric.compute(pred, ref, metadata={"sample_rate": 22050})

    assert scores == {"warpq": 3.8}
    assert calls["setup"]["fs"] == 16000
    assert calls["setup"]["n_mfcc"] == 20
    assert calls["setup"]["apply_vad"] is True
    assert calls["compute_fs"] == 22050


def test_register_warpq_metric():
    registry = MetricRegistry()

    register_warpq_metric(registry)

    assert registry.get_metric("warpq") is WarpqMetric
    assert registry.get_metric("warp_q") is WarpqMetric
    assert registry.get_metadata("warpq").requires_reference is True


def test_warpq_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.sequence_metrics.warpq.warpqMetric", None)

    with pytest.raises(ImportError, match="Please install WARP-Q"):
        WarpqMetric()


def test_warpq_resamples_with_keyword_sample_rates(monkeypatch):
    import versa.sequence_metrics.warpq as warpq_module

    calls = []

    class DummyWarpqModel:
        args = {"sr": 8000}

        def evaluate_versa(self, gt_x, pred_x):
            calls.append(("evaluate", gt_x.shape[0], pred_x.shape[0]))
            return 2.5

    def dummy_resample(audio, orig_sr, target_sr):
        calls.append(("resample", orig_sr, target_sr))
        return audio[:2]

    monkeypatch.setattr(
        warpq_module,
        "resample_audio",
        dummy_resample,
    )

    scores = warpq_module.warpq(DummyWarpqModel(), np.arange(4), np.arange(4), fs=16000)

    assert scores == {"warpq": 2.5}
    assert calls == [
        ("resample", 16000, 8000),
        ("resample", 16000, 8000),
        ("evaluate", 2, 2),
    ]


def test_espnet_wer_metric_class_uses_reference_text(monkeypatch):
    calls = {}

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return {"model": "dummy", "beam_size": kwargs["beam_size"]}

    monkeypatch.setattr(
        "versa.corpus_metrics.espnet_wer.espnet_wer_setup",
        dummy_setup,
    )

    def dummy_metric(wer_utils, pred_x, ref_text, fs=16000):
        calls["wer_utils"] = wer_utils
        calls["ref_text"] = ref_text
        calls["fs"] = fs
        return {"espnet_hyp_text": "hello", "espnet_wer_equal": 1}

    monkeypatch.setattr(
        "versa.corpus_metrics.espnet_wer.espnet_levenshtein_metric",
        dummy_metric,
    )

    pred, _ = _audio_pair()
    metric = EspnetWerMetric({"beam_size": 7})
    scores = metric.compute(pred, metadata={"sample_rate": 22050, "text": "hello"})

    assert scores == {"espnet_hyp_text": "hello", "espnet_wer_equal": 1}
    assert calls["setup"]["cache_dir"] == "versa_cache/espnet_model_zoo"
    assert calls["wer_utils"]["beam_size"] == 7
    assert calls["ref_text"] == "hello"
    assert calls["fs"] == 22050


def test_owsm_wer_metric_class_uses_reference_text(monkeypatch):
    calls = {}

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return {"model": "dummy", "beam_size": kwargs["beam_size"]}

    monkeypatch.setattr(
        "versa.corpus_metrics.owsm_wer.owsm_wer_setup",
        dummy_setup,
    )

    def dummy_metric(wer_utils, pred_x, ref_text, fs=16000):
        calls["ref_text"] = ref_text
        calls["fs"] = fs
        return {"owsm_hyp_text": "hello", "owsm_wer_equal": 1}

    monkeypatch.setattr(
        "versa.corpus_metrics.owsm_wer.owsm_levenshtein_metric",
        dummy_metric,
    )

    pred, _ = _audio_pair()
    metric = OwsmWerMetric()
    scores = metric.compute(pred, references="hello", metadata={"sample_rate": 16000})

    assert scores == {"owsm_hyp_text": "hello", "owsm_wer_equal": 1}
    assert calls["setup"]["cache_dir"] == "versa_cache/espnet_model_zoo"
    assert calls["ref_text"] == "hello"
    assert calls["fs"] == 16000


def test_whisper_wer_metric_class_uses_cached_text(monkeypatch):
    calls = {}

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return {"model": "dummy", "beam_size": kwargs["beam_size"]}

    monkeypatch.setattr(
        "versa.corpus_metrics.whisper_wer.whisper_wer_setup",
        dummy_setup,
    )

    def dummy_metric(
        wer_utils,
        pred_x,
        ref_text,
        fs=16000,
        cache_pred_text=None,
        cache_pred_language=None,
    ):
        calls["ref_text"] = ref_text
        calls["cache_pred_text"] = cache_pred_text
        calls["cache_pred_language"] = cache_pred_language
        return {"whisper_hyp_text": cache_pred_text, "whisper_wer_equal": 1}

    monkeypatch.setattr(
        "versa.corpus_metrics.whisper_wer.whisper_levenshtein_metric",
        dummy_metric,
    )

    pred, _ = _audio_pair()
    metric = WhisperWerMetric()
    scores = metric.compute(
        pred,
        metadata={
            "sample_rate": 16000,
            "text": "hello",
            "general_cache": {
                "whisper_hyp_text": "cached hello",
                "whisper_language": "en",
            },
        },
    )

    assert scores == {"whisper_hyp_text": "cached hello", "whisper_wer_equal": 1}
    assert calls["setup"]["cache_dir"] == "versa_cache/whisper"
    assert calls["setup"]["calc_per"] is False
    assert calls["ref_text"] == "hello"
    assert calls["cache_pred_text"] == "cached hello"
    assert calls["cache_pred_language"] == "en"


def test_whisper_per_metric_counts_cached_text(monkeypatch):
    import versa.corpus_metrics.whisper_wer as whisper_wer_module

    class DummyCleaner:
        def __init__(self, cleaner_name):
            self.cleaner_name = cleaner_name

        def __call__(self, text):
            return text

    class DummyTokenizer:
        def __init__(self, tokenizer_name):
            self.tokenizer_name = tokenizer_name

        def text2tokens(self, text):
            return text.split()

    monkeypatch.setattr(whisper_wer_module, "TextCleaner", DummyCleaner)
    monkeypatch.setattr(whisper_wer_module, "PhonemeTokenizer", DummyTokenizer)
    monkeypatch.setattr(
        whisper_wer_module,
        "whisper",
        type(
            "DummyWhisper",
            (),
            {"load_model": staticmethod(lambda *args, **kwargs: object())},
        ),
    )

    pred, _ = _audio_pair()
    metric = WhisperWerMetric({"calc_per": True})
    scores = metric.compute(
        pred,
        metadata={
            "sample_rate": 16000,
            "text": "HH_AH L_OW",
            "whisper_hyp_text": "HH_AH L_OW Z",
            "language": "en",
        },
    )

    assert scores["whisper_per_equal"] == 4
    assert scores["whisper_per_insert"] == 1
    assert scores["whisper_per_delete"] == 0
    assert scores["whisper_per_replace"] == 0


@pytest.mark.parametrize(
    "module_name,class_name,setup_name,metric_name,hyp_key",
    [
        (
            "versa.corpus_metrics.fwhisper_wer",
            FasterWhisperWerMetric,
            "fwhisper_wer_setup",
            "fwhisper_levenshtein_metric",
            "fwhisper_hyp_text",
        ),
        (
            "versa.corpus_metrics.nemo_wer",
            NemoWerMetric,
            "nemo_wer_setup",
            "nemo_levenshtein_metric",
            "nemo_hyp_text",
        ),
        (
            "versa.corpus_metrics.hubert_wer",
            HubertWerMetric,
            "hubert_wer_setup",
            "hubert_levenshtein_metric",
            "hubert_hyp_text",
        ),
    ],
)
def test_new_wer_metric_classes_use_cached_text(
    monkeypatch, module_name, class_name, setup_name, metric_name, hyp_key
):
    calls = {}

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return {"model": "dummy"}

    monkeypatch.setattr(f"{module_name}.{setup_name}", dummy_setup)

    def dummy_metric(wer_utils, pred_x, ref_text, fs=16000, cache_pred_text=None):
        calls["ref_text"] = ref_text
        calls["fs"] = fs
        calls["cache_pred_text"] = cache_pred_text
        return {hyp_key: cache_pred_text, hyp_key.replace("_hyp_text", "_wer_equal"): 1}

    monkeypatch.setattr(f"{module_name}.{metric_name}", dummy_metric)

    pred, _ = _audio_pair()
    metric = class_name({"model_tag": "tiny-test"})
    scores = metric.compute(
        pred,
        metadata={
            "sample_rate": 22050,
            "text": "hello",
            "general_cache": {hyp_key: "cached hello"},
        },
    )

    assert scores[hyp_key] == "cached hello"
    assert calls["setup"]["model_tag"] == "tiny-test"
    assert calls["ref_text"] == "hello"
    assert calls["fs"] == 22050
    assert calls["cache_pred_text"] == "cached hello"


def test_register_wer_metrics():
    registry = MetricRegistry()

    register_espnet_wer_metric(registry)
    register_owsm_wer_metric(registry)
    register_whisper_wer_metric(registry)
    register_fwhisper_wer_metric(registry)
    register_nemo_wer_metric(registry)
    register_hubert_wer_metric(registry)

    assert registry.get_metric("espnet_wer") is EspnetWerMetric
    assert registry.get_metric("owsm_wer") is OwsmWerMetric
    assert registry.get_metric("whisper_wer") is WhisperWerMetric
    assert registry.get_metric("fwhisper_wer") is FasterWhisperWerMetric
    assert registry.get_metric("faster_whisper_wer") is FasterWhisperWerMetric
    assert registry.get_metric("nemo_wer") is NemoWerMetric
    assert registry.get_metric("hubert_wer") is HubertWerMetric
    assert registry.get_metadata("espnet_wer").requires_text is True
    assert registry.get_metadata("owsm_wer").requires_text is True
    assert registry.get_metadata("whisper_wer").requires_text is True
    assert registry.get_metadata("fwhisper_wer").requires_text is True
    assert registry.get_metadata("nemo_wer").requires_text is True
    assert registry.get_metadata("hubert_wer").requires_text is True


def test_whisper_wer_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.corpus_metrics.whisper_wer.whisper", None)

    with pytest.raises(RuntimeError, match="openai-whisper is not installed"):
        WhisperWerMetric()


def test_register_signal_metric():
    registry = MetricRegistry()

    register_signal_metric(registry)

    assert registry.get_metric("signal_metric") is SignalMetric
    assert registry.get_metric("signal") is SignalMetric
    assert registry.get_metadata("signal_metric").requires_reference is True


def test_se_snr_metric_class_returns_existing_keys(monkeypatch):
    calls = {}

    class DummySeModel:
        pass

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return DummySeModel()

    def dummy_se_snr(model, pred_x, fs):
        calls["model"] = model
        calls["pred_x"] = pred_x
        calls["fs"] = fs
        return {
            "se_sdr": 1.0,
            "se_sar": 2.0,
            "se_si_snr": 3.0,
            "se_ci_sdr": 4.0,
        }

    monkeypatch.setattr("versa.utterance_metrics.se_snr.se_snr_setup", dummy_setup)
    monkeypatch.setattr("versa.utterance_metrics.se_snr.se_snr", dummy_se_snr)

    pred, _ = _audio_pair()
    metric = SeSnrMetric({"model_tag": "test-tag", "use_gpu": True})
    scores = metric.compute(pred, metadata={"sample_rate": 22050})

    assert scores == {
        "se_sdr": 1.0,
        "se_sar": 2.0,
        "se_si_snr": 3.0,
        "se_ci_sdr": 4.0,
    }
    assert calls["setup"]["cache_dir"] == "versa_cache/espnet_model_zoo"
    assert calls["setup"]["model_tag"] == "test-tag"
    assert calls["setup"]["use_gpu"] is True
    assert calls["fs"] == 22050


def test_register_se_snr_metric():
    registry = MetricRegistry()

    register_se_snr_metric(registry)

    assert registry.get_metric("se_snr") is SeSnrMetric
    assert registry.get_metric("se_snr_metric") is SeSnrMetric
    assert registry.get_metadata("se_snr").requires_reference is False


def test_se_snr_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.se_snr.SeparateSpeech", None)

    with pytest.raises(ImportError, match="se_snr requires espnet"):
        SeSnrMetric()


def test_speaker_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    class DummySpeakerModel:
        pass

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return DummySpeakerModel()

    def dummy_speaker_metric(model, pred_x, gt_x, fs):
        calls["model"] = model
        calls["fs"] = fs
        return {"spk_similarity": 0.75}

    monkeypatch.setattr(
        "versa.utterance_metrics.speaker.speaker_model_setup", dummy_setup
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.speaker.speaker_metric", dummy_speaker_metric
    )

    pred, ref = _audio_pair()
    metric = SpeakerMetric({"model_tag": "test-speaker", "use_gpu": True})
    scores = metric.compute(pred, ref, metadata={"sample_rate": 22050})

    assert scores == {"spk_similarity": 0.75}
    assert calls["setup"]["cache_dir"] == "versa_cache/espnet_model_zoo"
    assert calls["setup"]["model_tag"] == "test-speaker"
    assert calls["setup"]["use_gpu"] is True
    assert calls["fs"] == 22050


def test_register_speaker_metric():
    registry = MetricRegistry()

    register_speaker_metric(registry)

    assert registry.get_metric("speaker") is SpeakerMetric
    assert registry.get_metric("spk_similarity") is SpeakerMetric
    assert registry.get_metadata("speaker").requires_reference is True


def test_speaker_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.speaker.Speech2Embedding", None)

    with pytest.raises(ImportError, match="speaker requires espnet"):
        SpeakerMetric()


def test_singer_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    class DummySingerModel:
        pass

    def dummy_setup(**kwargs):
        calls["setup"] = kwargs
        return DummySingerModel()

    def dummy_singer_metric(model, pred_x, gt_x, fs, target_sr=44100):
        calls["model"] = model
        calls["fs"] = fs
        calls["target_sr"] = target_sr
        return {"singer_similarity": 0.5}

    monkeypatch.setattr(
        "versa.utterance_metrics.singer.singer_model_setup", dummy_setup
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.singer.singer_metric", dummy_singer_metric
    )

    pred, ref = _audio_pair()
    metric = SingerMetric({"model_name": "contrastive", "target_sr": 48000})
    scores = metric.compute(pred, ref, metadata={"sample_rate": 22050})

    assert scores == {"singer_similarity": 0.5}
    assert calls["setup"]["model_name"] == "contrastive"
    assert calls["fs"] == 22050
    assert calls["target_sr"] == 48000


def test_register_singer_metric():
    registry = MetricRegistry()

    register_singer_metric(registry)

    assert registry.get_metric("singer") is SingerMetric
    assert registry.get_metric("singer_similarity") is SingerMetric
    assert registry.get_metadata("singer").requires_reference is True


def test_singer_missing_dependency(monkeypatch):
    monkeypatch.setattr(
        "versa.utterance_metrics.singer.singer_model_setup",
        lambda **kwargs: (_ for _ in ()).throw(
            ImportError("Please run `install_ssl-singer-identity.sh` in tools.")
        ),
    )

    with pytest.raises(ImportError, match="install_ssl-singer-identity"):
        SingerMetric()


def test_log_wmse_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    class DummyLogWMSE:
        def __init__(self, **kwargs):
            calls["setup"] = kwargs

        def __call__(self, unproc_x, proc_x, gt_x):
            calls["unproc_shape"] = tuple(unproc_x.shape)
            calls["proc_shape"] = tuple(proc_x.shape)
            calls["gt_shape"] = tuple(gt_x.shape)
            return torch.tensor([0.33])

    monkeypatch.setattr("versa.utterance_metrics.log_wmse.LogWMSE", DummyLogWMSE)

    pred, ref = _audio_pair()
    unprocessed = pred * 0.5
    metric = LogWmseMetric({"audio_length": 2.0, "sample_rate": 48000})
    scores = metric.compute(
        pred,
        ref,
        metadata={"sample_rate": 16000, "unprocessed": unprocessed},
    )

    assert scores == {"log_wmse": pytest.approx(0.33)}
    assert calls["setup"]["audio_length"] == 2.0
    assert calls["setup"]["sample_rate"] == 48000
    assert calls["unproc_shape"] == (1, 1, pred.shape[0])
    assert calls["proc_shape"] == (1, 1, 1, pred.shape[0])
    assert calls["gt_shape"] == (1, 1, 1, ref.shape[0])


def test_register_log_wmse_metric():
    registry = MetricRegistry()

    register_log_wmse_metric(registry)

    assert registry.get_metric("log_wmse") is LogWmseMetric
    assert registry.get_metric("log-wmse") is LogWmseMetric
    assert registry.get_metadata("log_wmse").requires_reference is True


def test_log_wmse_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.log_wmse.LogWMSE", None)

    with pytest.raises(ImportError, match="torch-log-wmse"):
        LogWmseMetric()


def test_pseudo_mos_metric_class_returns_existing_keys(monkeypatch):
    calls = {}

    def dummy_setup(predictor_types, predictor_args, cache_dir, use_gpu):
        calls["setup"] = {
            "predictor_types": predictor_types,
            "predictor_args": predictor_args,
            "cache_dir": cache_dir,
            "use_gpu": use_gpu,
        }
        return {"utmos": object()}, {"utmos": 16000}

    def dummy_metric(pred, fs, predictor_dict, predictor_fs, use_gpu=False):
        calls["fs"] = fs
        calls["use_gpu"] = use_gpu
        return {"utmos": 4.2}

    monkeypatch.setattr(
        "versa.utterance_metrics.pseudo_mos.pseudo_mos_setup", dummy_setup
    )
    monkeypatch.setattr(
        "versa.utterance_metrics.pseudo_mos.pseudo_mos_metric", dummy_metric
    )

    pred, _ = _audio_pair()
    metric = PseudoMosMetric(
        {
            "predictor_types": ["utmos"],
            "predictor_args": {"utmos": {"fs": 16000}},
            "cache_dir": "cache",
            "use_gpu": True,
        }
    )
    scores = metric.compute(pred, metadata={"sample_rate": 22050})

    assert scores == {"utmos": 4.2}
    assert calls["setup"]["predictor_types"] == ["utmos"]
    assert calls["setup"]["cache_dir"] == "cache"
    assert calls["setup"]["use_gpu"] is True
    assert calls["fs"] == 22050
    assert calls["use_gpu"] is True


def test_register_pseudo_mos_metric():
    registry = MetricRegistry()

    register_pseudo_mos_metric(registry)

    assert registry.get_metric("pseudo_mos") is PseudoMosMetric
    assert registry.get_metric("utmos") is PseudoMosMetric
    assert registry.get_metadata("pseudo_mos").requires_reference is False


def test_universa_metric_class_auto_selects_references(monkeypatch):
    calls = {}

    def dummy_universa_metric(
        audio_data, ref_audio=None, ref_text=None, original_sr=16000, ref_sr=None
    ):
        calls["ref_audio"] = ref_audio
        calls["ref_text"] = ref_text
        calls["original_sr"] = original_sr
        calls["ref_sr"] = ref_sr
        return {"universa_mos": 3.5}

    monkeypatch.setattr(
        "versa.utterance_metrics.universa.universa_metric", dummy_universa_metric
    )

    pred, ref = _audio_pair()
    metric = UniversaMetric()
    scores = metric.compute(
        pred,
        ref,
        metadata={"sample_rate": 22050, "text": "hello"},
    )

    assert scores == {"universa_mos": 3.5}
    assert calls["ref_audio"] is ref
    assert calls["ref_text"] == "hello"
    assert calls["original_sr"] == 22050


def test_register_universa_metric():
    registry = MetricRegistry()

    register_universa_metric(registry)

    assert registry.get_metric("universa") is UniversaMetric
    assert registry.get_metric("uni_versa") is UniversaMetric
    assert registry.get_metadata("universa").requires_reference is False


def test_universa_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.universa.UniversaInference", None)

    with pytest.raises(ImportError, match="universa requires espnet"):
        UniversaMetric({"model_type": "noref"}).compute(np.zeros(16000))


def test_qwen2_audio_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "versa.utterance_metrics.qwen2_audio.qwen2_model_setup",
        lambda **kwargs: {"model": "dummy"},
    )

    def dummy_base_metric(
        qwen_utils, pred_x, fs=16000, custom_prompt=None, max_length=1000
    ):
        calls["fs"] = fs
        calls["custom_prompt"] = custom_prompt
        calls["max_length"] = max_length
        return "young adult"

    monkeypatch.setattr(
        "versa.utterance_metrics.qwen2_audio.qwen2_base_metric",
        dummy_base_metric,
    )

    pred, _ = _audio_pair()
    metric_class = QWEN2_AUDIO_METRIC_CLASSES["speaker_age"]
    metric = metric_class({"prompt": "Age?", "max_length": 77})
    scores = metric.compute(pred, metadata={"sample_rate": 22050})

    assert scores == {"qwen_speaker_age": "young adult"}
    assert calls["fs"] == 22050
    assert calls["custom_prompt"] == "Age?"
    assert calls["max_length"] == 77


def test_register_qwen2_audio_metric():
    registry = MetricRegistry()

    register_qwen2_audio_metric(registry)

    metric_class = QWEN2_AUDIO_METRIC_CLASSES["speaker_age"]
    assert registry.get_metric("qwen2_audio_speaker_age") is metric_class
    assert registry.get_metric("qwen2_speaker_age_metric") is metric_class
    assert registry.get_metadata("qwen2_audio_speaker_age").requires_reference is False


def test_qwen_omni_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "versa.utterance_metrics.qwen_omni.qwen_omni_model_setup",
        lambda **kwargs: {"model": "dummy"},
    )

    def dummy_base_metric(
        qwen_utils, pred_x, fs=16000, custom_prompt=None, max_length=500
    ):
        calls["fs"] = fs
        calls["custom_prompt"] = custom_prompt
        calls["max_length"] = max_length
        return "happy"

    monkeypatch.setattr(
        "versa.utterance_metrics.qwen_omni.qwen_omni_base_metric",
        dummy_base_metric,
    )

    pred, _ = _audio_pair()
    metric_class = QWEN_OMNI_METRIC_CLASSES["speech_emotion"]
    metric = metric_class({"prompt": "Emotion?", "max_length": 88})
    scores = metric.compute(pred, metadata={"sample_rate": 22050})

    assert scores == {"qwen_omni_speech_emotion": "happy"}
    assert calls["fs"] == 22050
    assert calls["custom_prompt"] == "Emotion?"
    assert calls["max_length"] == 88


def test_register_qwen_omni_metric():
    registry = MetricRegistry()

    register_qwen_omni_metric(registry)

    metric_class = QWEN_OMNI_METRIC_CLASSES["speech_emotion"]
    assert registry.get_metric("qwen_omni_speech_emotion") is metric_class
    assert registry.get_metric("qwen_omni_speech_emotion_metric") is metric_class
    assert registry.get_metadata("qwen_omni_speech_emotion").requires_reference is False


def test_squim_no_ref_metric_uses_cached_model(monkeypatch):
    class DummyObjectiveBundle:
        @staticmethod
        def get_model():
            return lambda pred_x: (
                torch.tensor([0.6]),
                torch.tensor([1.2]),
                torch.tensor([-3.4]),
            )

    monkeypatch.setattr("versa.utterance_metrics.squim.SQUIM_AVAILABLE", True)
    monkeypatch.setattr(
        "versa.utterance_metrics.squim.SQUIM_OBJECTIVE", DummyObjectiveBundle
    )

    pred, _ = _audio_pair()
    metric = SquimNoRefMetric()
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {
        "torch_squim_stoi": pytest.approx(0.6),
        "torch_squim_pesq": pytest.approx(1.2),
        "torch_squim_si_sdr": pytest.approx(-3.4),
    }


def test_squim_ref_metric_uses_cached_model(monkeypatch):
    class DummySubjectiveBundle:
        @staticmethod
        def get_model():
            return lambda pred_x, ref_x: torch.tensor([4.2])

    monkeypatch.setattr("versa.utterance_metrics.squim.SQUIM_AVAILABLE", True)
    monkeypatch.setattr(
        "versa.utterance_metrics.squim.SQUIM_SUBJECTIVE", DummySubjectiveBundle
    )

    pred, ref = _audio_pair()
    metric = SquimRefMetric()
    scores = metric.compute(pred, ref, metadata={"sample_rate": 16000})

    assert scores == {"torch_squim_mos": pytest.approx(4.2)}


def test_register_squim_metric():
    registry = MetricRegistry()

    register_squim_metric(registry)

    assert registry.get_metric("squim_ref") is SquimRefMetric
    assert registry.get_metric("squim_no_ref") is SquimNoRefMetric
    assert registry.get_metric("squim") is SquimNoRefMetric
    assert registry.get_metadata("squim_ref").requires_reference is True
    assert registry.get_metadata("squim_no_ref").requires_reference is False


def test_pysepm_registration_and_missing_dependency(monkeypatch):
    registry = MetricRegistry()
    register_pysepm_metric(registry)

    assert registry.get_metric("pysepm") is PysepmMetric
    assert registry.get_metric("pysepm_metric") is PysepmMetric
    assert registry.get_metadata("pysepm").requires_reference is True

    monkeypatch.setattr("versa.utterance_metrics.pysepm.pysepm", None)
    with pytest.raises(ImportError, match="pysepm is not installed"):
        PysepmMetric()


def test_vad_metric_class_returns_existing_key(monkeypatch):
    calls = {}

    def dummy_get_speech_ts(pred_x, model, **kwargs):
        calls["pred_x"] = pred_x
        calls["model"] = model
        calls["kwargs"] = kwargs
        return [{"start": 0.1, "end": 0.4}]

    monkeypatch.setattr(
        "versa.utterance_metrics.vad.torch.hub.load",
        lambda **kwargs: ("dummy-model", (dummy_get_speech_ts, None, None, None)),
    )

    pred, _ = _audio_pair()
    metric = VadMetric(
        {
            "threshold": 0.3,
            "min_speech_duration_ms": 100,
            "max_speech_duration_s": 10,
            "min_silence_duration_ms": 200,
            "speech_pad_ms": 40,
        }
    )
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {"vad_info": [{"start": 0.1, "end": 0.4}]}
    assert calls["model"] == "dummy-model"
    assert calls["kwargs"]["sampling_rate"] == 16000
    assert calls["kwargs"]["threshold"] == 0.3
    assert calls["kwargs"]["min_speech_duration_ms"] == 100
    assert calls["kwargs"]["max_speech_duration_s"] == 10
    assert calls["kwargs"]["min_silence_duration_ms"] == 200
    assert calls["kwargs"]["speech_pad_ms"] == 40


def test_register_vad_metric():
    registry = MetricRegistry()

    register_vad_metric(registry)

    assert registry.get_metric("vad") is VadMetric
    assert registry.get_metric("silero_vad") is VadMetric
    assert registry.get_metadata("vad").requires_reference is False


def test_sheet_ssqa_metric_class_returns_existing_key(monkeypatch):
    class DummyInnerModel:
        def to(self, device):
            self.device = device
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

    pred, _ = _audio_pair()
    metric = SheetSsqaMetric({"cache_dir": "test-cache", "use_gpu": False})
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {"sheet_ssqa": 3.25}


def test_register_sheet_ssqa_metric():
    registry = MetricRegistry()

    register_sheet_ssqa_metric(registry)

    assert registry.get_metric("sheet_ssqa") is SheetSsqaMetric
    assert registry.get_metric("sheet") is SheetSsqaMetric
    assert registry.get_metadata("sheet_ssqa").requires_reference is False


def test_scoreq_metric_classes_return_existing_keys(monkeypatch):
    calls = []

    class DummyScoreq:
        def __init__(self, data_domain, mode, cache_dir, device):
            self.mode = mode
            calls.append(
                {
                    "data_domain": data_domain,
                    "mode": mode,
                    "cache_dir": cache_dir,
                    "device": device,
                }
            )

        def predict(self, test_path, ref_path):
            assert test_path is not None
            if self.mode == "ref":
                assert ref_path is not None
                return 1.2
            assert ref_path is None
            return 2.4

    monkeypatch.setattr("versa.utterance_metrics.scoreq.Scoreq", DummyScoreq)

    pred, ref = _audio_pair()
    nr_metric = ScoreqNrMetric({"data_domain": "natural", "model_cache": "cache-a"})
    ref_metric = ScoreqRefMetric({"cache_dir": "cache-b"})

    assert nr_metric.compute(pred, metadata={"sample_rate": 16000}) == {
        "scoreq_nr": 2.4
    }
    assert ref_metric.compute(pred, ref, metadata={"sample_rate": 16000}) == {
        "scoreq_ref": 1.2
    }
    assert calls[0] == {
        "data_domain": "natural",
        "mode": "nr",
        "cache_dir": "cache-a",
        "device": "cpu",
    }
    assert calls[1]["mode"] == "ref"
    assert calls[1]["cache_dir"] == "cache-b"


def test_register_scoreq_metric():
    registry = MetricRegistry()

    register_scoreq_metric(registry)

    assert registry.get_metric("scoreq_nr") is ScoreqNrMetric
    assert registry.get_metric("scoreq_ref") is ScoreqRefMetric
    assert registry.get_metric("scoreq") is ScoreqNrMetric
    assert registry.get_metadata("scoreq_nr").requires_reference is False
    assert registry.get_metadata("scoreq_ref").requires_reference is True


def test_scoreq_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.scoreq.Scoreq", None)

    with pytest.raises(ModuleNotFoundError, match="scoreq is not installed"):
        ScoreqNrMetric()


def test_vqscore_metric_class_returns_existing_key(monkeypatch):
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

    pred, _ = _audio_pair()
    metric = VqscoreMetric()
    scores = metric.compute(pred, metadata={"sample_rate": 16000})

    assert scores == {"vqscore": pytest.approx(1.0, abs=1e-4)}


def test_register_vqscore_metric():
    registry = MetricRegistry()

    register_vqscore_metric(registry)

    assert registry.get_metric("vqscore") is VqscoreMetric
    assert registry.get_metric("vq_score") is VqscoreMetric
    assert registry.get_metadata("vqscore").requires_reference is False


def test_visqol_metric_class_returns_existing_key(monkeypatch):
    class DummySimilarityResult:
        moslqo = 4.1

    class DummyApi:
        def Measure(self, gt_x, pred_x):
            return DummySimilarityResult()

    monkeypatch.setattr(
        "versa.utterance_metrics.visqol_score.visqol_setup",
        lambda model: (DummyApi(), 16000),
    )

    pred, ref = _audio_pair()
    metric = VisqolMetric({"model": "speech"})
    scores = metric.compute(pred, ref, metadata={"sample_rate": 16000})

    assert scores == {"visqol": 4.1}


def test_register_visqol_metric():
    registry = MetricRegistry()

    register_visqol_metric(registry)

    assert registry.get_metric("visqol") is VisqolMetric
    assert registry.get_metric("VISQOL") is VisqolMetric
    assert registry.get_metadata("visqol").requires_reference is True


def test_visqol_missing_dependency(monkeypatch):
    monkeypatch.setattr("versa.utterance_metrics.visqol_score.visqol_lib_py", None)
    monkeypatch.setattr("versa.utterance_metrics.visqol_score.visqol_config_pb2", None)

    with pytest.raises(ImportError, match="visqol is not installed"):
        VisqolMetric()
