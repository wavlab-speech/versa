import logging

from versa.sequence_metrics.mcd_f0 import mcd_f0
from versa.sequence_metrics.signal_metric import signal_metric

try:
    from versa.utterance_metrics.discrete_speech import (
        discrete_speech_metric,
        discrete_speech_setup,
    )
except ImportError:
    logging.info(
        "Please pip install git+https://github.com/ftshijt/DiscreteSpeechMetrics.git and retry"
    )
except RuntimeError:
    logging.info(
        "Issues detected in discrete speech metrics, please double check the environment."
    )

from versa.utterance_metrics.pseudo_mos import pseudo_mos_metric, pseudo_mos_setup

try:
    from versa.utterance_metrics.pesq_score import pesq_metric
except ImportError:
    logging.info("Please install pesq with `pip install pesq` and retry")

try:
    from versa.utterance_metrics.stoi import stoi_metric
except ImportError:
    logging.info("Please install pystoi with `pip install pystoi` and retry")

try:
    from versa.utterance_metrics.speaker import speaker_metric, speaker_model_setup
except ImportError:
    logging.info("Please install espnet with `pip install espnet` and retry")

try:
    from versa.utterance_metrics.visqol_score import visqol_metric, visqol_setup
except ImportError:
    logging.info(
        "Please install visqol follow https://github.com/google/visqol and retry"
    )

from versa.corpus_metrics.espnet_wer import espnet_levenshtein_metric, espnet_wer_setup
from versa.corpus_metrics.owsm_wer import owsm_levenshtein_metric, owsm_wer_setup
from versa.corpus_metrics.whisper_wer import (
    whisper_levenshtein_metric,
    whisper_wer_setup,
)
from versa.utterance_metrics.scoreq import (
    scoreq_nr,
    scoreq_nr_setup,
    scoreq_ref,
    scoreq_ref_setup,
)
from versa.utterance_metrics.nomad import nomad_setup, nomad
from versa.utterance_metrics.emotion import emo2vec_setup, emo_sim
from versa.utterance_metrics.owsm_lid import owsm_lid_model_setup, language_id
from versa.utterance_metrics.sheet_ssqa import sheet_ssqa, sheet_ssqa_setup
from versa.utterance_metrics.squim import squim_metric, squim_metric_no_ref
from versa.utterance_metrics.se_snr import se_snr, se_snr_setup
from versa.utterance_metrics.pysepm import pysepm_metric
from versa.utterance_metrics.srmr import srmr_metric
from versa.corpus_metrics.fad import fad_scoring, fad_setup
from versa.utterance_metrics.noresqa import noresqa_model_setup, noresqa_metric
from versa.utterance_metrics.speaking_rate import (
    speaking_rate_metric,
    speaking_rate_model_setup,
)
from versa.utterance_metrics.asr_matching import asr_match_metric, asr_match_setup
from versa.utterance_metrics.audiobox_aesthetics_score import (
    audiobox_aesthetics_setup,
    audiobox_aesthetics_score,
)
from versa.utterance_metrics.qwen2_audio import qwen2_model_setup, qwen2_speaker_age_metric, qwen2_speaker_count_metric, qwen2_speaker_gender_metric, qwen2_speech_emotion_metric, qwen2_language_metric