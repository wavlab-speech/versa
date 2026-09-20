#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Shared scoring, result persistence, resume, and multi-source orchestration."""
import gc
import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor

import kaldiio
import soundfile as sf
import yaml
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Any, Union
from tqdm import tqdm

from versa.completion import (
    ERROR_BACKEND_SETUP,
    ERROR_CONFIGURATION,
    ERROR_INFERENCE,
    INPUT_IDENTITY_PATH,
    LEGACY_RECOMPUTE,
    RunStatus,
    STATUS_FAILED,
    STATUS_SUCCESS,
    classify_outcome,
    ensure_completion,
    input_signature,
    merge_rows,
    metric_signature,
    metric_signatures,
    pending_metrics,
    record_metric_status,
)
from versa.result_summary import compute_summary
from versa.audio_utils import resample_audio
from versa.definition import (
    BaseMetric,
    GPUMetric,
    MetricRegistry,
    MetricFactory,
    MetricSuite,
    MetricCategory,
    MetricType,
    MetricMetadata,
)
from versa.utils_shared import (
    check_all_same,
    check_minimum_length,
    default_numpy_serializer,
    find_files,
    load_audio,
    wav_normalize,
)

_worker_metric_suite = None


def _initialize_score_worker(metric_specs):
    """Create process-local metric instances for utterance scoring."""
    global _worker_metric_suite
    _worker_metric_suite = MetricSuite(
        {name: metric_class(config) for name, metric_class, config in metric_specs}
    )


def _score_utterance_worker(utterance):
    """Load and score one utterance without writing output files.

    The parent process owns resume state, so the worker computes only the
    pending metrics it was given and returns a fresh row for merging."""
    key, gen_file, gt_file, text, io, pending, inputs = utterance
    scorer = VersaScorer(MetricRegistry())

    gen_sr, gen_wav = load_audio(gen_file, io)
    gen_wav = wav_normalize(gen_wav)
    metric_names = _worker_metric_suite.metrics.keys()
    if not scorer._validate_audio(gen_wav, gen_sr, key, "generated", metric_names):
        return None

    gt_wav, gt_sr = None, None
    if gt_file is not None:
        gt_sr, gt_wav = load_audio(gt_file, io)
        gt_wav = wav_normalize(gt_wav)
        if not scorer._validate_audio(gt_wav, gt_sr, key, "ground truth", metric_names):
            return None

    gen_wav, gt_wav, gen_sr = scorer._align_sample_rates(gen_wav, gt_wav, gen_sr, gt_sr)
    processor = ScoreProcessor(
        _worker_metric_suite,
        signatures=metric_signatures(_worker_metric_suite.metrics),
    )
    return processor.process_batch(
        [UtteranceTask(key, gen_wav, gt_wav, gen_sr, text, pending, None, inputs)]
    )[0]


def audio_loader_setup(audio, io):
    # get ready compute embeddings
    """Build an utterance mapping from a Kaldi SCP, soundfile SCP, or directory.

    Kaldi entries load lazily. Soundfile SCP values are paths; command pipes
    raise ValueError and require the Kaldi interface instead."""
    if io == "kaldi":
        audio_files = kaldiio.load_scp(audio)
    elif io == "dir":
        audio_files = find_files(audio)
    elif io == "soundfile":
        audio_files = {}
        with open(audio) as f:
            for line in f.readlines():
                key, value = line.strip().split(maxsplit=1)
                if value.endswith("|"):
                    raise ValueError(
                        "Not supported wav.scp format. Set IO interface to kaldi"
                    )
                audio_files[key] = value
    return audio_files


def _create_populated_registry() -> MetricRegistry:
    """Create a registry populated with metric metadata."""
    from versa.metric_discovery import create_metric_discovery_registry

    return create_metric_discovery_registry(include_runtime_imports=False)


def load_score_modules(
    score_config: List[Dict[str, Any]],
    use_gt: bool = True,
    use_gt_text: bool = False,
    use_gpu: bool = False,
) -> MetricSuite:
    """Legacy wrapper for loading utterance-level scoring modules."""
    assert score_config, "no scoring function is provided"
    scorer = VersaScorer(_create_populated_registry())
    score_config = [
        config
        for config in score_config
        if not (
            scorer.registry.get_metadata(config["name"])
            and scorer.registry.get_metadata(config["name"]).category
            == MetricCategory.DISTRIBUTIONAL
        )
    ]
    return scorer.load_metrics(
        score_config,
        use_gt=use_gt,
        use_gt_text=use_gt_text,
        use_gpu=use_gpu,
    )


def _metric_cache_namespace(metric_name, metric_config=None):
    """Return a collision-safe shared namespace for a registered metric."""
    name = str(metric_name).lower()
    if name in {"speaker", "spk_similarity", "speaker_similarity"}:
        from versa.utterance_metrics.speaker import resolve_speaker_backend

        config = metric_config or {}
        backend = resolve_speaker_backend(
            model_tag=config.get("model_tag", "default"),
            backend=config.get("backend"),
            model_path=config.get("model_path"),
            model_config=config.get("model_config"),
        )
        return "huggingface" if backend == "huggingface" else "espnet_model_zoo"
    if name.startswith(("qwen2_audio_", "qwen_omni_")) or name in {
        "hubert_wer",
        "pam",
    }:
        return "huggingface"
    if name in {"asr_matching", "speaking_rate", "whisper_wer"}:
        return "whisper"
    if name in {
        "arecho",
        "espnet_wer",
        "owsm_lid",
        "owsm_wer",
        "se_snr",
        "universa",
    }:
        return "espnet_model_zoo"
    if name in {
        "multigauss",
        "pseudo_mos",
        "sheet_ssqa",
        "squim_no_ref",
        "squim_ref",
        "vad",
    }:
        return "torch"
    return name


def configure_metric_cache_dirs(score_config, cache_folder=None):
    """Apply a shared cache root without overriding metric-specific settings.

    Metrics backed by the same model hub receive a shared namespace. Other
    metrics receive their own directory so unrelated intermediate files cannot
    collide. Configurations that already declare ``cache_dir`` remain
    authoritative.
    """
    if cache_folder is None:
        return score_config

    cache_root = Path(cache_folder).expanduser()
    return [
        {
            **config,
            "cache_dir": config.get(
                "cache_dir",
                str(cache_root / _metric_cache_namespace(config["name"], config)),
            ),
        }
        for config in score_config
    ]


def configure_shared_cache_environment(cache_folder=None):
    """Point common model hubs at a single shareable cache root."""
    if cache_folder is None:
        return None

    cache_root = Path(cache_folder).expanduser().resolve()
    hf_cache = cache_root / "huggingface"
    torch_cache = cache_root / "torch"
    cache_root.mkdir(parents=True, exist_ok=True)

    os.environ["VERSA_CACHE_DIR"] = str(cache_root)
    os.environ["VERSA_HF_CACHE_DIR"] = str(hf_cache)
    os.environ["HF_HOME"] = str(hf_cache)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_DATASETS_CACHE"] = str(hf_cache / "datasets")
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["NEMO_CACHE_DIR"] = str(cache_root / "nemo")
    os.environ["XDG_CACHE_HOME"] = str(cache_root)
    return cache_root


def list_scoring(
    gen_files: Dict[str, str],
    score_modules: MetricSuite,
    gt_files: Optional[Dict[str, str]] = None,
    text_info: Optional[Dict[str, str]] = None,
    output_file: Optional[str] = None,
    io: str = "kaldi",
    batch_size: int = 1,
    resume: bool = False,
) -> List[Dict[str, Any]]:
    """Legacy wrapper for scoring a list of utterances."""
    scorer = VersaScorer(_create_populated_registry())
    return scorer.score_utterances(
        gen_files,
        score_modules,
        gt_files=gt_files,
        text_info=text_info,
        output_file=output_file,
        io=io,
        batch_size=batch_size,
        resume=resume,
    )


def load_summary(score_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Legacy alias for summary computation."""
    return compute_summary(score_info)


def _load_existing_jsonl_scores(
    output_file: Optional[str],
) -> Dict[str, Dict[str, Any]]:
    """Load previously written utterance scores keyed by utterance id."""
    if not output_file or not os.path.exists(output_file):
        return {}

    existing_scores = {}
    logger = logging.getLogger(__name__)
    with open(output_file, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                score = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring invalid JSON line %d in resume file %s",
                    line_number,
                    output_file,
                )
                continue

            key = score.get("key")
            if key is None:
                logger.warning(
                    "Ignoring resume line %d in %s because it has no key",
                    line_number,
                    output_file,
                )
                continue

            existing_scores[key] = score

    return existing_scores


def _input_reference(mapping: Any, key: str) -> Any:
    """Describe one mapped input without loading its audio.

    A Kaldi mapping loads the array when it is indexed, so its stored archive
    entry is used instead. Directory and soundfile mappings already hold paths.
    Because an archive entry is not a readable file on its own, content identity
    degrades to that entry for Kaldi inputs."""
    if mapping is None or key not in mapping:
        return None
    lazy_entries = getattr(mapping, "_dict", None)
    if isinstance(lazy_entries, dict):
        return lazy_entries[key]
    return mapping[key]


def _utterance_input_signatures(
    gen_files: Dict[str, str],
    gt_files: Optional[Dict[str, str]],
    text_info: Optional[Dict[str, str]],
    policy: str = INPUT_IDENTITY_PATH,
) -> Dict[str, str]:
    """Compute the input identity of every utterance under one policy."""
    return {
        key: input_signature(
            [_input_reference(gen_files, key), _input_reference(gt_files, key)],
            policy,
            text_info.get(key) if text_info else None,
        )
        for key in gen_files
    }


def _plan_utterance_work(
    keys: Any,
    existing_scores: Dict[str, Dict[str, Any]],
    signatures: Dict[str, str],
    input_signatures: Dict[str, str],
    legacy_resume: str = LEGACY_RECOMPUTE,
) -> tuple:
    """Split utterances into pending metric work and fully completed rows.

    Returns the pending metric names per utterance and the set of keys whose
    stored row already covers every configured metric. A key without a stored
    row is always pending, even when no metric is configured."""
    pending_by_key = {}
    completed_keys = set()
    for key in keys:
        existing = existing_scores.get(key)
        pending = pending_metrics(
            existing, signatures, input_signatures.get(key), legacy_resume
        )
        if pending or existing is None:
            pending_by_key[key] = pending
        else:
            completed_keys.add(key)
    return pending_by_key, completed_keys


def _subset_suite(metric_suite: MetricSuite, pending: Optional[Any]) -> MetricSuite:
    """Restrict a suite to the metrics an utterance still needs."""
    if pending is None:
        return metric_suite
    requested = set(pending)
    return MetricSuite(
        {
            name: metric
            for name, metric in metric_suite.metrics.items()
            if name in requested
        }
    )


def _score_with_status(
    metric_suite: MetricSuite,
    key: str,
    predictions: Any,
    references: Any,
    metadata: Dict[str, Any],
    signatures: Dict[str, str],
    pending: Optional[Any] = None,
    inputs: Optional[str] = None,
    existing: Optional[Dict[str, Any]] = None,
    run_status: Optional[RunStatus] = None,
) -> Dict[str, Any]:
    """Compute the pending metrics of one item and record their outcomes.

    Every attempted metric contributes a completion entry naming its status and
    identity. A failed or abstaining metric keeps a null value so reports still
    discover the field. The result is merged into ``existing`` so previously
    successful metrics survive a partial rerun."""
    utt_score = {"key": key}
    ensure_completion(utt_score, inputs)

    outcomes = _subset_suite(metric_suite, pending).compute_all_detailed(
        predictions=predictions, references=references, metadata=metadata
    )
    for metric_name, (metric_results, error) in outcomes.items():
        if isinstance(metric_results, dict):
            fields = list(metric_results)
            utt_score.update(metric_results)
        else:
            fields = [metric_name]
            utt_score[metric_name] = metric_results
        status, category = classify_outcome(metric_results, error)
        entry = record_metric_status(
            utt_score,
            metric_name,
            signatures.get(metric_name),
            status,
            fields=fields,
            error=error,
            error_category=category,
        )
        if run_status is not None:
            run_status.record_metric_entry(entry)

    if run_status is not None:
        run_status.scored_utterances += 1
    return merge_rows(existing, utt_score)


def _pending_files(
    gen_files: Dict[str, str],
    existing_scores: Dict[str, Dict[str, Any]],
    signatures: Dict[str, str],
    input_signatures: Dict[str, str],
    legacy_resume: str = LEGACY_RECOMPUTE,
) -> Dict[str, str]:
    """Select the input mapping restricted to utterances with pending work."""
    _, completed_keys = _plan_utterance_work(
        gen_files, existing_scores, signatures, input_signatures, legacy_resume
    )
    return {key: path for key, path in gen_files.items() if key not in completed_keys}


def _record_skip(run_status: Optional[RunStatus]) -> None:
    """Count an utterance dropped before scoring when tracking run status."""
    if run_status is not None:
        run_status.record_skipped_utterance()


def _repair_resume_file(
    output_file: Optional[str],
    existing_scores: Dict[str, Dict[str, Any]],
) -> None:
    """Atomically rewrite a resume file as one record per stored key.

    Repeated keys collapse to their last record and a truncated final record is
    dropped, so the file the run appends to is already well formed. Every
    readable row is kept, including partially completed utterances and keys
    outside the current inputs: a resumed run that is interrupted again must not
    lose the work it had already stored."""
    if not output_file or not os.path.exists(output_file):
        return

    _write_jsonl_scores(output_file, list(existing_scores.values()))


def _finalize_resume_file(
    output_file: Optional[str],
    existing_scores: Dict[str, Dict[str, Any]],
    updated_rows: List[Dict[str, Any]],
) -> None:
    """Replace a resumed result file with one record per utterance.

    Recomputed utterances were appended next to the stored records they
    supersede; this pass keeps only the newer record while preserving the
    original position of known keys and the order of new ones. Interrupting a
    resumed run before this pass leaves those superseded records in the file,
    where the next resume reads the newer one; aggregate such a file only after
    a run has finished."""
    if not output_file or not os.path.exists(output_file):
        return

    ordered = dict(existing_scores)
    for row in updated_rows:
        key = row.get("key")
        if key is not None:
            ordered[key] = row
    _write_jsonl_scores(output_file, list(ordered.values()))


def _ensure_append_starts_on_new_line(output_file: str) -> None:
    """Make sure resumed JSONL appends cannot merge with a partial final line."""
    if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
        return

    with open(output_file, "rb") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) == b"\n":
            return

    with open(output_file, "a", encoding="utf-8") as f:
        f.write("\n")


def _write_jsonl_scores(
    output_file: Optional[str],
    score_info: List[Dict[str, Any]],
) -> None:
    """Write utterance scores as JSONL in the current utterance order.

    The file is replaced atomically so an interruption during a metric-oriented
    checkpoint cannot leave a truncated final record behind."""
    if not output_file:
        return

    temporary = f"{output_file}.tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        for utt_score in score_info:
            printable_result = json.dumps(utt_score, default=default_numpy_serializer)
            f.write(f"{printable_result}\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, output_file)


def _write_jsonl_score(file_handle, utt_score: Dict[str, Any]) -> None:
    """Write one utterance score and flush it for resume checkpointing."""
    printable_result = json.dumps(utt_score, default=default_numpy_serializer)
    file_handle.write(f"{printable_result}\n")
    file_handle.flush()


def _validate_multi_source_file_sets(gen_source_files, gt_source_files):
    """Validate ordered source mappings and return stable mixture key order."""
    if len(gen_source_files) < 2 or len(gt_source_files) < 2:
        raise ValueError(
            "Multi-source scoring requires at least two generated and two "
            "reference source mappings"
        )
    if len(gen_source_files) != len(gt_source_files):
        raise ValueError(
            "Generated and reference source mapping counts must match; received "
            f"{len(gen_source_files)} and {len(gt_source_files)}"
        )

    keys = list(gen_source_files[0])
    if not keys:
        raise ValueError("Multi-source scoring received no mixture keys")
    expected = set(keys)
    mappings = [
        *(f"generated source {index}" for index in range(len(gen_source_files))),
        *(f"reference source {index}" for index in range(len(gt_source_files))),
    ]
    for label, source_files in zip(mappings, [*gen_source_files, *gt_source_files]):
        actual = set(source_files)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            details = []
            if missing:
                details.append("missing keys: " + ", ".join(missing[:5]))
            if extra:
                details.append("unexpected keys: " + ", ".join(extra[:5]))
            raise ValueError(f"{label} does not match source 0 ({'; '.join(details)})")
    return keys


def _release_metric_resources() -> None:
    """Best-effort cleanup after unloading model-backed metrics."""
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class UtteranceTask(NamedTuple):
    """One prepared utterance and the resume state that scoping it requires.

    ``pending`` names the metrics to compute; None computes the whole suite.
    ``existing`` is the previously stored row to merge into, and ``inputs`` is
    the input identity recorded with the resulting completion envelope."""

    key: str
    gen_wav: Any
    gt_wav: Any
    sample_rate: int
    text: Optional[str] = None
    pending: Optional[Any] = None
    existing: Optional[Dict[str, Any]] = None
    inputs: Optional[str] = None


class ScoreProcessor:
    """Handles batch processing and caching of scores."""

    def __init__(
        self,
        metric_suite: MetricSuite,
        output_file: Optional[str] = None,
        resume: bool = False,
        signatures: Optional[Dict[str, str]] = None,
        run_status: Optional[RunStatus] = None,
    ):
        """Bind a metric suite and optionally open its result file.

        Resume appends, repairing a missing final newline first. Otherwise the
        file is truncated. The processor owns the handle and must be closed.
        ``signatures`` supplies the metric identities recorded in each
        completion envelope; it defaults to the identities of the bound suite."""
        self.metric_suite = metric_suite
        self.output_file = output_file
        self.signatures = (
            signatures
            if signatures is not None
            else metric_signatures(metric_suite.metrics)
        )
        self.run_status = run_status
        self.logger = logging.getLogger(self.__class__.__name__)

        if output_file:
            mode = "a" if resume else "w"
            if resume:
                _ensure_append_starts_on_new_line(output_file)
            self.file_handle = open(output_file, mode, encoding="utf-8")
        else:
            self.file_handle = None

    def process_batch(self, cache_info: List[tuple]) -> List[Dict[str, Any]]:
        """Score a batch of cached utterances and record per-metric outcomes.

        Each metric is recorded as success, failed, or abstained together with
        the identity of the evaluation, so a later resume recomputes only the
        metrics that did not complete. Rows are merged into any stored record
        for the same utterance before being written."""
        batch_score_info = []
        for utt_info in cache_info:
            task = UtteranceTask(*utt_info)
            metadata = {
                "key": task.key,
                "sample_rate": task.sample_rate,
                "text": task.text,
                "general_cache": {"whisper_hyp_text": None},
            }
            utt_score = _score_with_status(
                self.metric_suite,
                task.key,
                task.gen_wav,
                task.gt_wav,
                metadata,
                self.signatures,
                pending=task.pending,
                inputs=task.inputs,
                existing=task.existing,
                run_status=self.run_status,
            )
            batch_score_info.append(utt_score)

            if self.file_handle:
                printable_result = json.dumps(
                    utt_score, default=default_numpy_serializer
                )
                self.file_handle.write(f"{printable_result}\n")
                self.file_handle.flush()

        return batch_score_info

    def close(self):
        """Close file handle if open."""
        if self.file_handle:
            self.file_handle.close()


class VersaScorer:
    """Main scorer class that orchestrates the scoring process."""

    def __init__(self, registry: MetricRegistry = None):
        """Bind a registry, or create metadata-only discovery, and prepare its factory."""
        self.registry = registry or self._create_default_registry()
        self.factory = MetricFactory(self.registry)
        self.logger = logging.getLogger(self.__class__.__name__)

    def _create_default_registry(self) -> MetricRegistry:
        """Create and populate the default metric registry."""
        return _create_populated_registry()

    def load_metrics(
        self,
        score_config: List[Dict[str, Any]],
        use_gt: bool = True,
        use_gt_text: bool = False,
        use_gpu: bool = False,
        run_status: Optional[RunStatus] = None,
    ) -> MetricSuite:
        """Load and configure metrics based on configuration.

        Metrics that cannot be used are reported rather than silently dropped:
        an unmet input requirement is a configuration failure and a backend
        that raises while being constructed is a setup failure."""
        metrics = {}
        failures = {}

        for config in score_config:
            metric_name = config["name"]

            try:
                # Check if metric requires ground truth
                metadata = self.registry.get_metadata(metric_name)
                if metadata and metadata.requires_reference and not use_gt:
                    self.logger.warning(
                        f"Cannot use {metric_name} because no ground truth is provided"
                    )
                    failures[metric_name] = ERROR_CONFIGURATION
                    continue

                if metadata and metadata.requires_text and not use_gt_text:
                    self.logger.warning(
                        f"Cannot use {metric_name} because no ground truth text is provided"
                    )
                    failures[metric_name] = ERROR_CONFIGURATION
                    continue

                from versa.metric_registry import register_metric_for_config

                register_metric_for_config(
                    self.registry, metric_name, logger=self.logger
                )

                # Create metric instance
                metric_config = {**config, "use_gpu": use_gpu}
                metric = self.factory.create_metric(metric_name, metric_config)
                metrics[metric_name] = metric

                self.logger.info(f"Loaded {metric_name} successfully")

            except Exception as e:
                self.logger.error(
                    f"Failed to load metric {metric_name}: {e}", exc_info=True
                )
                failures[metric_name] = ERROR_BACKEND_SETUP
                continue

        if run_status is not None:
            run_status.record_load(len(score_config), len(metrics), failures)
        return MetricSuite(metrics)

    def score_utterances(
        self,
        gen_files: Dict[str, str],
        metric_suite: MetricSuite,
        gt_files: Optional[Dict[str, str]] = None,
        text_info: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None,
        io: str = "kaldi",
        batch_size: int = 1,
        resume: bool = False,
        num_workers: int = 1,
        legacy_resume: str = LEGACY_RECOMPUTE,
        input_identity: str = INPUT_IDENTITY_PATH,
        run_status: Optional[RunStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Score individual utterances, recording per-metric completion.

        With ``resume``, an utterance is skipped only when every configured
        metric is recorded as successful or abstained under the same identity;
        otherwise its missing and failed metrics are recomputed and merged into
        the stored row. ``legacy_resume`` selects how rows written before the
        completion contract are treated, and ``input_identity`` selects whether
        changed audio is detected by path or by content."""

        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        metric_suite = MetricSuite(
            {
                name: metric
                for name, metric in metric_suite.metrics.items()
                if metric.get_metadata().category != MetricCategory.DISTRIBUTIONAL
            }
        )
        if num_workers > 1 and any(
            getattr(metric, "config", {}).get("use_gpu", False)
            for metric in metric_suite.metrics.values()
        ):
            raise ValueError(
                "Local multiprocessing is CPU-only; use num_workers=1 with GPU metrics"
            )
        signatures = metric_signatures(metric_suite.metrics)
        existing_scores = _load_existing_jsonl_scores(output_file) if resume else {}
        input_signatures = _utterance_input_signatures(
            gen_files, gt_files, text_info, input_identity
        )
        pending_by_key, completed_keys = _plan_utterance_work(
            gen_files, existing_scores, signatures, input_signatures, legacy_resume
        )
        if run_status is not None:
            run_status.total_utterances += len(gen_files)
        if resume and output_file:
            self.logger.info(
                "Resume enabled: %d of %d utterances in %s already completed "
                "every configured metric",
                len(completed_keys),
                len(gen_files),
                output_file,
            )
        elif resume:
            self.logger.warning(
                "Resume requested without output_file; scoring normally"
            )

        if num_workers > 1:
            return self._score_utterances_parallel(
                gen_files,
                metric_suite,
                gt_files=gt_files,
                text_info=text_info,
                output_file=output_file,
                io=io,
                existing_scores=existing_scores,
                completed_keys=completed_keys,
                pending_by_key=pending_by_key,
                input_signatures=input_signatures,
                signatures=signatures,
                resume=resume,
                num_workers=num_workers,
                run_status=run_status,
            )

        if resume and output_file:
            _repair_resume_file(output_file, existing_scores)
        processor = ScoreProcessor(
            metric_suite,
            output_file,
            resume=resume,
            signatures=signatures,
            run_status=run_status,
        )
        score_info = [
            existing_scores[key] for key in gen_files if key in completed_keys
        ]
        if run_status is not None:
            for key in gen_files:
                if key in completed_keys:
                    run_status.record_row(existing_scores[key], resumed=True)
        cache_info = []

        try:
            for key in tqdm(gen_files.keys()):
                if key in completed_keys:
                    self.logger.debug("Skipping completed utterance %s", key)
                    continue

                # Step1: Load and validate generated audio
                gen_sr, gen_wav = load_audio(gen_files[key], io)
                gen_wav = wav_normalize(gen_wav)

                if not self._validate_audio(
                    gen_wav,
                    gen_sr,
                    key,
                    "generated",
                    metric_suite.metrics.keys(),
                ):
                    _record_skip(run_status)
                    continue

                # Step2: Load and validate ground truth audio
                gt_wav, gt_sr = None, None
                if gt_files is not None:
                    if key not in gt_files:
                        self.logger.warning(
                            f"Ground truth not found for key {key}, skipping"
                        )
                        _record_skip(run_status)
                        continue

                    gt_sr, gt_wav = load_audio(gt_files[key], io)
                    gt_wav = wav_normalize(gt_wav)

                    if not self._validate_audio(
                        gt_wav,
                        gt_sr,
                        key,
                        "ground truth",
                        metric_suite.metrics.keys(),
                    ):
                        _record_skip(run_status)
                        continue

                # Step3: Load text information
                text = text_info.get(key) if text_info else None
                if text_info and key not in text_info:
                    self.logger.warning(f"Text not found for key {key}, skipping")
                    _record_skip(run_status)
                    continue

                # Step4: Resample if needed
                gen_wav, gt_wav, gen_sr = self._align_sample_rates(
                    gen_wav, gt_wav, gen_sr, gt_sr
                )

                # Step5: Cache for batch processing
                utterance_info = UtteranceTask(
                    key,
                    gen_wav,
                    gt_wav,
                    gen_sr,
                    text,
                    pending_by_key.get(key),
                    existing_scores.get(key),
                    input_signatures.get(key),
                )
                cache_info.append(utterance_info)

                if len(cache_info) >= batch_size:
                    score_info.extend(processor.process_batch(cache_info))
                    cache_info = []

            # Process remaining items
            if cache_info:
                score_info.extend(processor.process_batch(cache_info))

        finally:
            processor.close()

        if resume and output_file:
            _finalize_resume_file(output_file, existing_scores, score_info)
        self.logger.info(f"Scoring completed. Results saved to {output_file}")
        return score_info

    def score_multi_source_utterances(
        self,
        gen_source_files: List[Dict[str, Any]],
        metric_suite: MetricSuite,
        gt_source_files: List[Dict[str, Any]],
        output_file: Optional[str] = None,
        io: str = "soundfile",
        resume: bool = False,
        legacy_resume: str = LEGACY_RECOMPUTE,
        input_identity: str = INPUT_IDENTITY_PATH,
        run_status: Optional[RunStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Score explicitly ordered source pairs for each mixture.

        Each item in ``gen_source_files`` and ``gt_source_files`` is one
        source-specific SCP mapping. List order defines the source assignment.
        This path is intentionally separate from single-utterance scoring so a
        metric such as MAPSS cannot silently infer or permute source pairs.

        Resume follows the same completion contract as utterance scoring: a
        mixture is skipped only when every configured metric completed under
        the same identity, and the identity of all ordered sources contributes
        to the recorded input signature.
        """

        keys = _validate_multi_source_file_sets(gen_source_files, gt_source_files)
        if not metric_suite.metrics:
            raise ValueError("No multi-source scoring function is available")
        if any(
            not metric.get_metadata().requires_multiple_sources
            for metric in metric_suite.metrics.values()
        ):
            raise ValueError(
                "Multi-source scoring only accepts metrics declared with "
                "requires_multiple_sources=True"
            )

        signatures = metric_signatures(metric_suite.metrics)
        existing_scores = _load_existing_jsonl_scores(output_file) if resume else {}
        input_signatures = {
            key: input_signature(
                [
                    _input_reference(source_files, key)
                    for source_files in [*gen_source_files, *gt_source_files]
                ],
                input_identity,
            )
            for key in keys
        }
        pending_by_key, completed_keys = _plan_utterance_work(
            keys, existing_scores, signatures, input_signatures, legacy_resume
        )
        score_by_key = {key: existing_scores[key] for key in completed_keys}
        if run_status is not None:
            run_status.total_utterances += len(keys)
            for key in keys:
                if key in completed_keys:
                    run_status.record_row(existing_scores[key], resumed=True)

        file_handle = None
        if output_file:
            if resume:
                _repair_resume_file(output_file, existing_scores)
                _ensure_append_starts_on_new_line(output_file)
            file_handle = open(output_file, "a" if resume else "w", encoding="utf-8")

        try:
            for key in tqdm(keys):
                if key in completed_keys:
                    continue

                predictions = []
                references = []
                valid = True
                for source_index, source_files in enumerate(gen_source_files):
                    sr, waveform = load_audio(source_files[key], io)
                    waveform = wav_normalize(waveform)
                    if not self._validate_audio(
                        waveform,
                        sr,
                        key,
                        f"generated source {source_index}",
                        metric_suite.metrics.keys(),
                    ):
                        valid = False
                        break
                    predictions.append(resample_audio(waveform, sr, 16000))
                if not valid:
                    _record_skip(run_status)
                    continue

                for source_index, source_files in enumerate(gt_source_files):
                    sr, waveform = load_audio(source_files[key], io)
                    waveform = wav_normalize(waveform)
                    if not self._validate_audio(
                        waveform,
                        sr,
                        key,
                        f"reference source {source_index}",
                        metric_suite.metrics.keys(),
                    ):
                        valid = False
                        break
                    references.append(resample_audio(waveform, sr, 16000))
                if not valid:
                    _record_skip(run_status)
                    continue

                metadata = {
                    "key": key,
                    "sample_rate": 16000,
                }
                utt_score = _score_with_status(
                    metric_suite,
                    key,
                    predictions,
                    references,
                    metadata,
                    signatures,
                    pending=pending_by_key.get(key),
                    inputs=input_signatures.get(key),
                    existing=existing_scores.get(key),
                    run_status=run_status,
                )

                score_by_key[key] = utt_score
                if file_handle:
                    _write_jsonl_score(file_handle, utt_score)
        finally:
            if file_handle:
                file_handle.close()

        score_info = [score_by_key[key] for key in keys if key in score_by_key]
        if resume and output_file:
            _finalize_resume_file(output_file, existing_scores, score_info)
        return score_info

    def _score_utterances_parallel(
        self,
        gen_files: Dict[str, str],
        metric_suite: MetricSuite,
        gt_files: Optional[Dict[str, str]],
        text_info: Optional[Dict[str, str]],
        output_file: Optional[str],
        io: str,
        existing_scores: Dict[str, Dict[str, Any]],
        completed_keys: set,
        pending_by_key: Dict[str, List[str]],
        input_signatures: Dict[str, str],
        signatures: Dict[str, str],
        resume: bool,
        num_workers: int,
        run_status: Optional[RunStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Score utterances in process-local metric suites.

        Workers receive the pending metrics of their utterance and return a
        fresh row; this process merges it into the stored record so resume
        behaves identically in serial and multiprocessing execution."""
        jobs = []
        for key in gen_files:
            if key in completed_keys:
                continue
            if gt_files is not None and key not in gt_files:
                self.logger.warning("Ground truth not found for key %s, skipping", key)
                _record_skip(run_status)
                continue
            if text_info is not None and key not in text_info:
                self.logger.warning("Text not found for key %s, skipping", key)
                _record_skip(run_status)
                continue
            jobs.append(
                (
                    key,
                    gen_files[key],
                    gt_files[key] if gt_files is not None else None,
                    text_info.get(key) if text_info is not None else None,
                    io,
                    pending_by_key.get(key),
                    input_signatures.get(key),
                )
            )

        metric_specs = [
            (name, metric.__class__, metric.config)
            for name, metric in metric_suite.metrics.items()
        ]
        new_scores = {}
        file_handle = None
        if output_file:
            if resume:
                _repair_resume_file(output_file, existing_scores)
                _ensure_append_starts_on_new_line(output_file)
            file_handle = open(output_file, "a" if resume else "w", encoding="utf-8")

        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_initialize_score_worker,
                initargs=(metric_specs,),
            ) as executor:
                results = executor.map(_score_utterance_worker, jobs)
                for result in tqdm(results, total=len(jobs)):
                    if result is None:
                        _record_skip(run_status)
                        continue
                    key = result["key"]
                    if run_status is not None:
                        run_status.record_row(result)
                    merged = merge_rows(existing_scores.get(key), result)
                    new_scores[key] = merged
                    if file_handle:
                        _write_jsonl_score(file_handle, merged)
        finally:
            if file_handle:
                file_handle.close()

        if run_status is not None:
            for key in gen_files:
                if key in completed_keys:
                    run_status.record_row(existing_scores[key], resumed=True)
        score_info = [
            new_scores.get(key, existing_scores.get(key))
            for key in gen_files
            if key in new_scores or key in completed_keys
        ]
        if resume and output_file:
            _finalize_resume_file(output_file, existing_scores, score_info)
        self.logger.info("Scoring completed. Results saved to %s", output_file)
        return score_info

    def score_utterances_by_metric(
        self,
        gen_files: Dict[str, str],
        score_config: List[Dict[str, Any]],
        gt_files: Optional[Dict[str, str]] = None,
        text_info: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None,
        io: str = "kaldi",
        batch_size: int = 1,
        resume: bool = False,
        use_gpu: bool = False,
        legacy_resume: str = LEGACY_RECOMPUTE,
        input_identity: str = INPUT_IDENTITY_PATH,
        run_status: Optional[RunStatus] = None,
    ) -> List[Dict[str, Any]]:
        """Score all utterances one metric at a time to lower peak model memory.

        Resume is evaluated per metric: a metric pass visits only the
        utterances that have not already completed that metric under the same
        identity, so an interrupted run does not reload and recompute the
        metrics it finished."""

        use_gt = gt_files is not None
        use_gt_text = text_info is not None
        existing_scores = _load_existing_jsonl_scores(output_file) if resume else {}
        input_signatures = _utterance_input_signatures(
            gen_files, gt_files, text_info, input_identity
        )
        score_by_key = {
            key: dict(existing_scores.get(key, {"key": key})) for key in gen_files
        }
        scored_keys = set()
        if run_status is not None:
            run_status.total_utterances += len(gen_files)

        for config in score_config:
            metric_name = config["name"]
            metadata = self.registry.get_metadata(metric_name)
            if metadata and metadata.category == MetricCategory.DISTRIBUTIONAL:
                self.logger.info("Skipping %s for utterance-level scoring", metric_name)
                continue

            # Plan from the configured identity first so a metric whose work is
            # already complete never loads its backend.
            planned = {metric_name: metric_signature(metric_name, config)}
            if not _pending_files(
                gen_files, score_by_key, planned, input_signatures, legacy_resume
            ):
                self.logger.info(
                    "Metric %s already completed every utterance; skipping",
                    metric_name,
                )
                if run_status is not None:
                    run_status.record_load(1, 0)
                continue

            metric_suite = self.load_metrics(
                [config],
                use_gt=use_gt,
                use_gt_text=use_gt_text,
                use_gpu=use_gpu,
                run_status=run_status,
            )
            metric_suite = MetricSuite(
                {
                    name: metric
                    for name, metric in metric_suite.metrics.items()
                    if metric.get_metadata().category != MetricCategory.DISTRIBUTIONAL
                }
            )

            if len(metric_suite.metrics) == 0:
                self.logger.info("Skipping %s for utterance-level scoring", metric_name)
                _release_metric_resources()
                continue

            # Re-plan with the loaded identities, which a metric may extend
            # beyond its configuration.
            pending_files = _pending_files(
                gen_files,
                score_by_key,
                metric_signatures(metric_suite.metrics),
                input_signatures,
                legacy_resume,
            )

            try:
                if not pending_files:
                    self.logger.info(
                        "Metric %s already completed every utterance; skipping",
                        metric_name,
                    )
                    continue

                self.logger.info("Scoring utterances with metric %s", metric_name)
                metric_status = RunStatus() if run_status is not None else None
                metric_scores = self.score_utterances(
                    pending_files,
                    metric_suite,
                    gt_files=gt_files,
                    text_info=text_info,
                    output_file=None,
                    io=io,
                    batch_size=batch_size,
                    resume=False,
                    input_identity=input_identity,
                    run_status=metric_status,
                )
                if run_status is not None:
                    run_status.merge_metric_counts(metric_status)
                for utt_score in metric_scores:
                    key = utt_score.get("key")
                    if key is None:
                        continue
                    scored_keys.add(key)
                    score_by_key[key] = merge_rows(score_by_key.get(key), utt_score)
            finally:
                del metric_suite
                _release_metric_resources()

            _write_jsonl_scores(
                output_file,
                [score_by_key[key] for key in gen_files if key in score_by_key],
            )

        if run_status is not None:
            run_status.scored_utterances += len(scored_keys)
            run_status.resumed_utterances += len(
                [key for key in gen_files if key not in scored_keys]
            )
        score_info = [score_by_key[key] for key in gen_files if key in score_by_key]
        self.logger.info(
            f"Metric-oriented scoring completed. Results saved to {output_file}"
        )
        return score_info

    def score_corpus(
        self,
        gen_files: Dict[str, str],
        metric_suite: MetricSuite,
        base_files: Optional[Dict[str, str]] = None,
        text_info: Optional[Dict[str, str]] = None,
        output_file: Optional[str] = None,
        run_status: Optional[RunStatus] = None,
    ) -> Dict[str, Any]:
        """Score at corpus level (e.g., FAD, KID).

        A corpus metric that raises is counted as a failure in ``run_status``
        so a run with an unusable distributional metric is not reported as
        complete."""

        score_info = {}

        # Filter for distributional metrics
        distributional_metrics = metric_suite.filter_by_category(
            MetricCategory.DISTRIBUTIONAL
        )

        for name, metric in distributional_metrics.metrics.items():
            try:
                metadata = {"baseline_files": base_files, "text_info": text_info}

                score_result = metric.compute(
                    predictions=gen_files, references=base_files, metadata=metadata
                )
                if isinstance(score_result, dict):
                    score_info.update(score_result)
                else:
                    score_info.update({name: score_result})
                if run_status is not None:
                    run_status.record_metric_entry({"status": STATUS_SUCCESS})

            except Exception as e:
                self.logger.error(
                    f"Error computing corpus metric {name}: {e}", exc_info=True
                )
                if run_status is not None:
                    run_status.record_metric_entry(
                        {"status": STATUS_FAILED, "error_category": ERROR_INFERENCE}
                    )

        if output_file:
            with open(output_file, "w") as f:
                yaml.dump(score_info, f)

        return score_info

    def _validate_audio(
        self,
        wav: Any,
        sr: int,
        key: str,
        audio_type: str,
        metric_names: Optional[Any] = None,
    ) -> bool:
        """Validate audio data."""
        # Length check
        if not check_minimum_length(wav.shape[0] / sr, list(metric_names or [])):
            self.logger.warning(
                f"Audio {key} ({audio_type}, length {wav.shape[0] / sr}) is too short, skipping"
            )
            return False

        # Check for silent audio
        if check_all_same(wav):
            self.logger.warning(
                f"Audio {key} ({audio_type}) has only the same value, skipping"
            )
            return False

        return True

    def _align_sample_rates(
        self, gen_wav: Any, gt_wav: Any, gen_sr: int, gt_sr: Optional[int]
    ) -> tuple:
        """Align sample rates between generated and ground truth audio."""
        if gt_sr is None:
            return gen_wav, gt_wav, gen_sr

        if gen_sr > gt_sr:
            self.logger.warning("Resampling generated audio to match ground truth")
            gen_wav = resample_audio(gen_wav, gen_sr, gt_sr)
            gen_sr = gt_sr
        elif gen_sr < gt_sr:
            self.logger.warning(
                "Resampling ground truth audio to match generated audio"
            )
            gt_wav = resample_audio(gt_wav, gt_sr, gen_sr)

        return gen_wav, gt_wav, gen_sr
