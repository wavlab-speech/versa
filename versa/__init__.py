import logging

__version__ = "0.0.1"  # noqa: F401

# from versa.sequence_metrics.mcd_f0 import McdF0Metric, register_mcd_f0_metric
# from versa.sequence_metrics.signal_metric import SignalMetric, register_signal_metric

try:
    from versa.utterance_metrics.discrete_speech import (
        DiscreteSpeechMetric,
        register_discrete_speech_metric,
    )
except ImportError:
    logging.info(
        "Please pip install git+https://github.com/ftshijt/DiscreteSpeechMetrics.git and retry"
    )
except RuntimeError:
    logging.info(
        "Issues detected in discrete speech metrics, please double check the environment."
    )

# from versa.utterance_metrics.pseudo_mos import PseudoMosMetric, register_pseudo_mos_metric

# try:
#     from versa.utterance_metrics.pesq_score import PesqMetric, register_pesq_metric
# except ImportError:
#     logging.info("Please install pesq with `pip install pesq` and retry")

# try:
#     from versa.utterance_metrics.stoi import StoiMetric, register_stoi_metric
# except ImportError:
#     logging.info("Please install pystoi with `pip install pystoi` and retry")

# try:
#     from versa.utterance_metrics.speaker import SpeakerMetric, register_speaker_metric
# except ImportError:
#     logging.info("Please install espnet with `pip install espnet` and retry")

# try:
#     from versa.utterance_metrics.singer import SingerMetric, register_singer_metric
# except ImportError:
#     logging.info("Please install ...")

# try:
#     from versa.utterance_metrics.visqol_score import VisqolMetric, register_visqol_metric
# except ImportError:
#     logging.info(
#         "Please install visqol follow https://github.com/google/visqol and retry"
#     )

# from versa.corpus_metrics.espnet_wer import EspnetWerMetric, register_espnet_wer_metric
# from versa.corpus_metrics.fad import FadMetric, register_fad_metric
# from versa.corpus_metrics.owsm_wer import OwsmWerMetric, register_owsm_wer_metric
# from versa.corpus_metrics.whisper_wer import (
#     WhisperWerMetric,
#     register_whisper_wer_metric
# )
from versa.utterance_metrics.asr_matching import (
    ASRMatchMetric,
    register_asr_match_metric,
)
from versa.utterance_metrics.audiobox_aesthetics_score import (
    AudioBoxAestheticsMetric,
    register_audiobox_aesthetics_metric,
)
from versa.utterance_metrics.emo_similarity import (
    Emo2vecMetric,
    register_emo2vec_metric,
)
from versa.utterance_metrics.nomad import NomadMetric, register_nomad_metric
from versa.utterance_metrics.noresqa import NoresqaMetric, register_noresqa_metric
from versa.utterance_metrics.owsm_lid import OwsmLidMetric, register_owsm_lid_metric

# from versa.utterance_metrics.pysepm import PysepmMetric, register_pysepm_metric
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
# from versa.utterance_metrics.sheet_ssqa import SheetSsqaMetric, register_sheet_ssqa_metric
# from versa.utterance_metrics.speaking_rate import (
#     SpeakingRateMetric,
#     register_speaking_rate_metric
# )
# from versa.utterance_metrics.squim import SquimMetric, register_squim_metric
from versa.utterance_metrics.srmr import SRMRMetric, register_srmr_metric
from versa.utterance_metrics.chroma_alignment import (
    ChromaAlignmentMetric,
    register_chroma_alignment_metric,
)
from versa.utterance_metrics.dpam_distance import (
    DpamDistanceMetric,
    register_dpam_distance_metric,
)
from versa.utterance_metrics.cdpam_distance import (
    CdpamDistanceMetric,
    register_cdpam_distance_metric,
)

# from versa.utterance_metrics.vqscore import VqscoreMetric, register_vqscore_metric
from versa.utterance_metrics.nisqa import NisqaMetric, register_nisqa_metric
