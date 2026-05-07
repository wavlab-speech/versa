import importlib
import logging
import os
from pathlib import Path

__version__ = "0.0.1"  # noqa: F401

logger = logging.getLogger(__name__)

os.environ.setdefault(
    "NUMBA_CACHE_DIR", str(Path.cwd() / "versa_cache" / "numba_cache")
)


def _optional_metric_import(module_name, names, install_hint=None):
    """Import optional metric symbols without making package import fail."""
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        if install_hint:
            logger.info(install_hint)
        else:
            logger.info("Optional metric module %s is not available", module_name)
        return
    except RuntimeError:
        logger.info("Issues detected in %s; please check the environment.", module_name)
        return

    for name in names:
        globals()[name] = getattr(module, name)


_optional_metric_import(
    "versa.sequence_metrics.mcd_f0",
    ("McdF0Metric", "register_mcd_f0_metric"),
)
# from versa.sequence_metrics.signal_metric import SignalMetric, register_signal_metric
_optional_metric_import(
    "versa.sequence_metrics.warpq",
    ("WarpqMetric", "register_warpq_metric"),
)

_optional_metric_import(
    "versa.utterance_metrics.discrete_speech",
    ("DiscreteSpeechMetric", "register_discrete_speech_metric"),
    (
        "Please pip install "
        "git+https://github.com/ftshijt/DiscreteSpeechMetrics.git and retry"
    ),
)

_optional_metric_import(
    "versa.utterance_metrics.pseudo_mos",
    ("PseudoMosMetric", "register_pseudo_mos_metric"),
)

_optional_metric_import(
    "versa.utterance_metrics.pesq_score",
    ("PesqMetric", "register_pesq_metric"),
    "Please install pesq with `pip install pesq` and retry",
)

# try:
#     from versa.utterance_metrics.stoi import StoiMetric, register_stoi_metric
# except ImportError:
#     logging.info("Please install pystoi with `pip install pystoi` and retry")
_optional_metric_import(
    "versa.utterance_metrics.stoi",
    ("StoiMetric", "EstoiMetric", "register_stoi_metric"),
    "Please install pystoi with `pip install pystoi` and retry",
)

_optional_metric_import(
    "versa.utterance_metrics.speaker",
    ("SpeakerMetric", "register_speaker_metric"),
)

_optional_metric_import(
    "versa.utterance_metrics.singer",
    ("SingerMetric", "register_singer_metric"),
    "Please install singer_identity following tools/install_ssl-singer-identity.sh",
)

_optional_metric_import(
    "versa.utterance_metrics.visqol_score",
    ("VisqolMetric", "register_visqol_metric"),
    "Please install visqol following https://github.com/google/visqol and retry",
)

_optional_metric_import(
    "versa.corpus_metrics.espnet_wer",
    ("EspnetWerMetric", "register_espnet_wer_metric"),
)
_optional_metric_import(
    "versa.corpus_metrics.clap_score",
    ("ClapScoreMetric", "register_clap_score_metric"),
    "Please install frechet-audio-distance following tools/install_clap_score.sh",
)
# from versa.corpus_metrics.fad import FadMetric, register_fad_metric
_optional_metric_import(
    "versa.corpus_metrics.owsm_wer",
    ("OwsmWerMetric", "register_owsm_wer_metric"),
)
_optional_metric_import(
    "versa.corpus_metrics.whisper_wer",
    ("WhisperWerMetric", "register_whisper_wer_metric"),
)
_optional_metric_import(
    "versa.corpus_metrics.fwhisper_wer",
    ("FasterWhisperWerMetric", "register_fwhisper_wer_metric"),
    "Please install faster-whisper following tools/install_fwhisper.sh",
)
_optional_metric_import(
    "versa.corpus_metrics.nemo_wer",
    ("NemoWerMetric", "register_nemo_wer_metric"),
    "Please install NeMo following tools/install_nemo.sh",
)
_optional_metric_import(
    "versa.corpus_metrics.hubert_wer",
    ("HubertWerMetric", "register_hubert_wer_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.asr_matching",
    ("ASRMatchMetric", "register_asr_match_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.audiobox_aesthetics_score",
    ("AudioBoxAestheticsMetric", "register_audiobox_aesthetics_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.emo_similarity",
    ("Emo2vecMetric", "register_emo2vec_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.nomad",
    ("NomadMetric", "register_nomad_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.noresqa",
    ("NoresqaMetric", "register_noresqa_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.owsm_lid",
    ("OwsmLidMetric", "register_owsm_lid_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.log_wmse",
    ("LogWmseMetric", "register_log_wmse_metric"),
    "Please install torch-log-wmse and retry",
)
_optional_metric_import(
    "versa.utterance_metrics.universa",
    ("UniversaMetric", "register_universa_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.arecho",
    ("ArechoMetric", "register_arecho_metric"),
)

# from versa.utterance_metrics.pysepm import PysepmMetric, register_pysepm_metric
_optional_metric_import(
    "versa.utterance_metrics.pysepm",
    ("PysepmMetric", "register_pysepm_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.qwen2_audio",
    ("Qwen2AudioMetric", "register_qwen2_audio_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.qwen_omni",
    ("QwenOmniMetric", "register_qwen_omni_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.scoreq",
    (
        "ScoreqMetric",
        "ScoreqNrMetric",
        "ScoreqRefMetric",
        "register_scoreq_metric",
    ),
)
_optional_metric_import(
    "versa.utterance_metrics.se_snr",
    ("SeSnrMetric", "register_se_snr_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.sheet_ssqa",
    ("SheetSsqaMetric", "register_sheet_ssqa_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.speaking_rate",
    ("SpeakingRateMetric", "register_speaking_rate_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.squim",
    ("SquimMetric", "SquimRefMetric", "SquimNoRefMetric", "register_squim_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.srmr",
    ("SRMRMetric", "register_srmr_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.chroma_alignment",
    ("ChromaAlignmentMetric", "register_chroma_alignment_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.dpam_distance",
    ("DpamDistanceMetric", "register_dpam_distance_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.cdpam_distance",
    ("CdpamDistanceMetric", "register_cdpam_distance_metric"),
)

_optional_metric_import(
    "versa.utterance_metrics.vqscore",
    ("VqscoreMetric", "register_vqscore_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.vad",
    ("VadMetric", "register_vad_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.nisqa",
    ("NisqaMetric", "register_nisqa_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.pam",
    ("PamMetric", "register_pam_metric"),
)
_optional_metric_import(
    "versa.sequence_metrics.signal_metric",
    ("SignalMetric", "register_signal_metric"),
)
_optional_metric_import(
    "versa.utterance_metrics.sigmos",
    ("SigmosMetric", "register_sigmos_metric"),
    "Please install SigMOS dependencies and retry",
)
_optional_metric_import(
    "versa.utterance_metrics.wvmos",
    ("WvmosMetric", "register_wvmos_metric"),
    "Please install WVMOS following tools/install_wvmos.sh",
)
