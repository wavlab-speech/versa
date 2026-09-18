"""Helpers for loading optional metric modules."""

import importlib
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from versa.definition import MetricRegistry


@dataclass(frozen=True)
class MetricModuleSpec:
    """Identify an optional module, its exported symbols, and installation guidance."""

    module_name: str
    symbols: tuple
    install_hint: Optional[str] = None


METRIC_MODULES = (
    MetricModuleSpec(
        "versa.sequence_metrics.mcd_f0",
        ("McdF0Metric", "register_mcd_f0_metric"),
    ),
    MetricModuleSpec(
        "versa.sequence_metrics.warpq",
        ("WarpqMetric", "register_warpq_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.discrete_speech",
        ("DiscreteSpeechMetric", "register_discrete_speech_metric"),
        (
            "Please pip install "
            "git+https://github.com/ftshijt/DiscreteSpeechMetrics.git and retry"
        ),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.pseudo_mos",
        ("PseudoMosMetric", "register_pseudo_mos_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.pesq_score",
        ("PesqMetric", "register_pesq_metric"),
        "Please install pesq with `pip install pesq` and retry",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.stoi",
        ("StoiMetric", "EstoiMetric", "register_stoi_metric"),
        "Please install pystoi with `pip install pystoi` and retry",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.speaker",
        ("SpeakerMetric", "register_speaker_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.singer",
        ("SingerMetric", "register_singer_metric"),
        "Please install singer_identity following tools/install_ssl-singer-identity.sh",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.visqol_score",
        ("VisqolMetric", "register_visqol_metric"),
        "Please install visqol following https://github.com/google/visqol and retry",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.espnet_wer",
        ("EspnetWerMetric", "register_espnet_wer_metric"),
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.clap_score",
        ("ClapScoreMetric", "register_clap_score_metric"),
        "Please install frechet-audio-distance following tools/install_clap_score.sh",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.fad",
        ("FadMetric", "register_fad_metric"),
        "Please install FADTK following tools/install_fadtk.sh",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.individual_fad",
        ("IndividualFadMetric", "register_individual_fad_metric"),
        "Please install FADTK following tools/install_fadtk.sh",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.kid",
        ("KidMetric", "register_kid_metric"),
        "Please install FADTK following tools/install_fadtk.sh",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.owsm_wer",
        ("OwsmWerMetric", "register_owsm_wer_metric"),
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.whisper_wer",
        ("WhisperWerMetric", "register_whisper_wer_metric"),
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.fwhisper_wer",
        ("FasterWhisperWerMetric", "register_fwhisper_wer_metric"),
        "Please install faster-whisper following tools/install_fwhisper.sh",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.nemo_wer",
        ("NemoWerMetric", "register_nemo_wer_metric"),
        "Please install NeMo following tools/install_nemo.sh",
    ),
    MetricModuleSpec(
        "versa.corpus_metrics.hubert_wer",
        ("HubertWerMetric", "register_hubert_wer_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.asr_matching",
        ("ASRMatchMetric", "register_asr_match_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.audiobox_aesthetics_score",
        ("AudioBoxAestheticsMetric", "register_audiobox_aesthetics_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.asvspoof_score",
        ("ASVSpoofMetric", "register_asvspoof_metric"),
        "Please install AASIST following tools/install_asvspoof.sh",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.emo_similarity",
        ("Emo2vecMetric", "register_emo2vec_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.emo_vad",
        ("EmoVadMetric", "register_emo_vad_metric"),
        "Please install transformers and retry",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.nomad",
        ("NomadMetric", "register_nomad_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.noresqa",
        ("NoresqaMetric", "register_noresqa_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.owsm_lid",
        ("OwsmLidMetric", "register_owsm_lid_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.log_wmse",
        ("LogWmseMetric", "register_log_wmse_metric"),
        "Please install torch-log-wmse and retry",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.universa",
        ("UniversaMetric", "register_universa_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.arecho",
        ("ArechoMetric", "register_arecho_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.pysepm",
        ("PysepmMetric", "register_pysepm_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.qwen2_audio",
        ("Qwen2AudioMetric", "register_qwen2_audio_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.qwen_omni",
        ("QwenOmniMetric", "register_qwen_omni_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.scoreq",
        (
            "ScoreqMetric",
            "ScoreqNrMetric",
            "ScoreqRefMetric",
            "register_scoreq_metric",
        ),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.se_snr",
        ("SeSnrMetric", "register_se_snr_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.sheet_ssqa",
        ("SheetSsqaMetric", "register_sheet_ssqa_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.speaking_rate",
        ("SpeakingRateMetric", "register_speaking_rate_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.squim",
        ("SquimMetric", "SquimRefMetric", "SquimNoRefMetric", "register_squim_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.srmr",
        ("SRMRMetric", "register_srmr_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.chroma_alignment",
        ("ChromaAlignmentMetric", "register_chroma_alignment_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.dpam_distance",
        ("DpamDistanceMetric", "register_dpam_distance_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.cdpam_distance",
        ("CdpamDistanceMetric", "register_cdpam_distance_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.vqscore",
        ("VqscoreMetric", "register_vqscore_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.multigauss",
        ("MultiGaussMetric", "register_multigauss_metric"),
        "Please install MultiGauss following tools/install_multigauss.sh",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.songeval",
        ("SongEvalMetric", "register_songeval_metric"),
        "Please install SongEval dependencies following tools/install_songeval.sh",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.mapss",
        ("MapssMetric", "register_mapss_metric"),
        "Please install MAPSS following tools/install_mapss.sh",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.vad",
        ("VadMetric", "register_vad_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.nisqa",
        ("NisqaMetric", "register_nisqa_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.pam",
        ("PamMetric", "register_pam_metric"),
    ),
    MetricModuleSpec(
        "versa.sequence_metrics.signal_metric",
        ("SignalMetric", "register_signal_metric"),
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.sigmos",
        ("SigmosMetric", "register_sigmos_metric"),
        "Please install SigMOS dependencies and retry",
    ),
    MetricModuleSpec(
        "versa.utterance_metrics.wvmos",
        ("WvmosMetric", "register_wvmos_metric"),
        "Please install WVMOS following tools/install_wvmos.sh",
    ),
)

_SYMBOL_TO_SPEC = {symbol: spec for spec in METRIC_MODULES for symbol in spec.symbols}


def metric_symbol_names() -> Iterable[str]:
    """Return metric symbols that can be loaded lazily from optional modules."""
    return _SYMBOL_TO_SPEC.keys()


def load_metric_symbol(symbol: str, logger: Optional[logging.Logger] = None) -> Any:
    """Load one optional metric symbol by name."""
    spec = _SYMBOL_TO_SPEC.get(symbol)
    if spec is None:
        raise AttributeError(symbol)
    module = importlib.import_module(spec.module_name)
    return getattr(module, symbol)


def load_metric_symbols(
    target_globals: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """Import all optional metric modules and return their exported symbols."""
    loaded: Dict[str, Any] = {}
    for spec in METRIC_MODULES:
        module = _try_import_metric_module(spec, logger=logger)
        if module is None:
            continue
        for symbol in spec.symbols:
            loaded[symbol] = getattr(module, symbol)
    if target_globals is not None:
        target_globals.update(loaded)
    return loaded


def create_populated_registry(
    logger: Optional[logging.Logger] = None,
) -> MetricRegistry:
    """Create a registry populated by all importable metric modules."""
    registry = MetricRegistry()
    for name, symbol in load_metric_symbols(logger=logger).items():
        if not name.startswith("register_") or not name.endswith("_metric"):
            continue
        if callable(symbol):
            try:
                symbol(registry)
            except Exception as exc:
                (logger or logging.getLogger(__name__)).warning(
                    "Failed to register metric via %s: %s", name, exc
                )
    return registry


def register_metric_for_config(
    registry: MetricRegistry,
    metric_name: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Register only the selected backend, preserving source-discovered aliases.

    Raise ValueError for unknown names and ImportError with installation guidance
    for unavailable backends. Already registered concrete metrics are untouched.
    """
    if _has_concrete_metric(registry, metric_name):
        return
    spec = _metric_specs_by_name().get(metric_name)
    if spec is None:
        raise ValueError(f"No runtime module registered for metric {metric_name!r}")
    try:
        module = importlib.import_module(spec.module_name)
    except ImportError as exc:
        hint = spec.install_hint or (
            "Install the selected metric's dependencies listed by --describe-metric "
            f"{metric_name} and retry"
        )
        raise ImportError(
            f"Cannot load metric {metric_name!r} from {spec.module_name}: {exc}. {hint}"
        ) from exc
    for symbol in spec.symbols:
        if symbol.startswith("register_") and symbol.endswith("_metric"):
            getattr(module, symbol)(registry)
    if not _has_concrete_metric(registry, metric_name):
        raise ValueError(
            f"Module {spec.module_name} did not register metric {metric_name!r}"
        )


@lru_cache(maxsize=1)
def _metric_specs_by_name() -> Dict[str, MetricModuleSpec]:
    """Index canonical names and aliases from packaged source without backend imports."""
    from versa.metric_discovery import _discover_module_metadata

    package_root = Path(__file__).resolve().parent
    specs = {}
    for spec in METRIC_MODULES:
        path = package_root.joinpath(*spec.module_name.split(".")[1:]).with_suffix(
            ".py"
        )
        for metadata, aliases in _discover_module_metadata(path):
            for name in [metadata.name, *aliases]:
                if name in specs and specs[name] != spec:
                    raise ValueError(
                        f"Metric name {name!r} belongs to multiple modules"
                    )
                specs[name] = spec
    return specs


def _has_concrete_metric(registry: MetricRegistry, metric_name: str) -> bool:
    """Check that a registered class exposes runtime scoring and metadata methods."""
    metric_class = registry.get_metric(metric_name)
    if metric_class is None:
        return False
    return hasattr(metric_class, "compute") and hasattr(metric_class, "get_metadata")


def _try_import_metric_module(
    spec: MetricModuleSpec, logger: Optional[logging.Logger] = None
) -> Optional[Any]:
    """Import an optional metric module, logging failures and returning None."""
    log = logger or logging.getLogger(__name__)
    try:
        return importlib.import_module(spec.module_name)
    except ImportError:
        if spec.install_hint:
            log.info(spec.install_hint)
        else:
            log.info("Optional metric module %s is not available", spec.module_name)
    except Exception as exc:
        log.info("Issues detected in %s: %s", spec.module_name, exc)
    return None
