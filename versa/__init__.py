import importlib
import logging

__version__ = "0.0.1"  # noqa: F401

logger = logging.getLogger(__name__)


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


# from versa.sequence_metrics.mcd_f0 import McdF0Metric, register_mcd_f0_metric
# from versa.sequence_metrics.signal_metric import SignalMetric, register_signal_metric

_optional_metric_import(
    "versa.utterance_metrics.discrete_speech",
    ("DiscreteSpeechMetric", "register_discrete_speech_metric"),
    (
        "Please pip install "
        "git+https://github.com/ftshijt/DiscreteSpeechMetrics.git and retry"
    ),
)

# from versa.utterance_metrics.pseudo_mos import (
#     PseudoMosMetric,
#     register_pseudo_mos_metric,
# )

# try:
#     from versa.utterance_metrics.pesq_score import PesqMetric, register_pesq_metric
# except ImportError:
#     logging.info("Please install pesq with `pip install pesq` and retry")

# try:
#     from versa.utterance_metrics.stoi import StoiMetric, register_stoi_metric
# except ImportError:
#     logging.info("Please install pystoi with `pip install pystoi` and retry")
_optional_metric_import(
    "versa.utterance_metrics.stoi",
    ("StoiMetric", "EstoiMetric", "register_stoi_metric"),
    "Please install pystoi with `pip install pystoi` and retry",
)

# try:
#     from versa.utterance_metrics.speaker import SpeakerMetric, register_speaker_metric
# except ImportError:
#     logging.info("Please install espnet with `pip install espnet` and retry")

# try:
#     from versa.utterance_metrics.singer import SingerMetric, register_singer_metric
# except ImportError:
#     logging.info("Please install ...")

# try:
#     from versa.utterance_metrics.visqol_score import (
#         VisqolMetric,
#         register_visqol_metric,
#     )
# except ImportError:
#     logging.info(
#         "Please install visqol follow https://github.com/google/visqol and retry"
#     )

# from versa.corpus_metrics.espnet_wer import (
#     EspnetWerMetric,
#     register_espnet_wer_metric,
# )
# from versa.corpus_metrics.fad import FadMetric, register_fad_metric
# from versa.corpus_metrics.owsm_wer import OwsmWerMetric, register_owsm_wer_metric
# from versa.corpus_metrics.whisper_wer import (
#     WhisperWerMetric,
#     register_whisper_wer_metric
# )
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

# from versa.utterance_metrics.pysepm import PysepmMetric, register_pysepm_metric
_optional_metric_import(
    "versa.utterance_metrics.pysepm",
    ("PysepmMetric", "register_pysepm_metric"),
)
# from versa.utterance_metrics.qwen2_audio import (
#     Qwen2ChannelTypeMetric,
#     Qwen2LanguageMetric,
#     Qwen2LaughterCryingMetric,
#     Qwen2ModelSetup,
#     Qwen2OverlappingSpeechMetric,
#     Qwen2PitchRangeMetric,
#     Qwen2RecordingQualityMetric,
#     Qwen2SpeakerAgeMetric,
#     Qwen2SpeakerCountMetric,
#     Qwen2SpeakerGenderMetric,
#     Qwen2SpeakingStyleMetric,
#     Qwen2SpeechBackgroundEnvironmentMetric,
#     Qwen2SpeechClarityMetric,
#     Qwen2SpeechEmotionMetric,
#     Qwen2SpeechImpairmentMetric,
#     Qwen2SpeechPurposeMetric,
#     Qwen2SpeechRateMetric,
#     Qwen2SpeechRegisterMetric,
#     Qwen2SpeechVolumeLevelMetric,
#     Qwen2VocabularyComplexityMetric,
#     Qwen2VoicePitchMetric,
#     Qwen2VoiceTypeMetric,
#     Qwen2SingingTechniqueMetric,
# )
# from versa.utterance_metrics.qwen_omni import (
#     QwenOmniMetric,
#     register_qwen_omni_metric
# )
# from versa.utterance_metrics.scoreq import (
#     ScoreqMetric,
#     register_scoreq_metric
# )
# from versa.utterance_metrics.se_snr import SeSnrMetric, register_se_snr_metric
# from versa.utterance_metrics.sheet_ssqa import (
#     SheetSsqaMetric,
#     register_sheet_ssqa_metric,
# )
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

# from versa.utterance_metrics.vqscore import VqscoreMetric, register_vqscore_metric
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
