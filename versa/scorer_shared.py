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
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

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
    """Load and score one utterance without writing output files."""
    key, gen_file, gt_file, text, io = utterance
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
    return ScoreProcessor(_worker_metric_suite).process_batch(
        [(key, gen_wav, gt_wav, gen_sr, text)]
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
    """Write utterance scores as JSONL in the current utterance order."""
    if not output_file:
        return

    with open(output_file, "w", encoding="utf-8") as f:
        for utt_score in score_info:
            printable_result = json.dumps(utt_score, default=default_numpy_serializer)
            f.write(f"{printable_result}\n")


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


class ScoreProcessor:
    """Handles batch processing and caching of scores."""

    def __init__(
        self,
        metric_suite: MetricSuite,
        output_file: Optional[str] = None,
        resume: bool = False,
    ):
        """Bind a metric suite and optionally open its result file.

        Resume appends, repairing a missing final newline first. Otherwise the
        file is truncated. The processor owns the handle and must be closed."""
        self.metric_suite = metric_suite
        self.output_file = output_file
        self.logger = logging.getLogger(self.__class__.__name__)

        if output_file:
            mode = "a" if resume else "w"
            if resume:
                _ensure_append_starts_on_new_line(output_file)
            self.file_handle = open(output_file, mode, encoding="utf-8")
        else:
            self.file_handle = None

    def process_batch(self, cache_info: List[tuple]) -> List[Dict[str, Any]]:
        """Process a batch of cached utterance information."""
        batch_score_info = []
        for utt_info in cache_info:
            key, gen_wav, gt_wav, gen_sr, text = utt_info
            utt_score = {"key": key}

            try:
                # Prepare metadata for metric computation
                metadata = {
                    "key": key,
                    "sample_rate": gen_sr,
                    "text": text,
                    "general_cache": {"whisper_hyp_text": None},
                }

                # Compute all metrics
                scores = self.metric_suite.compute_all(
                    predictions=gen_wav, references=gt_wav, metadata=metadata
                )

                # Flatten the metric results
                for metric_name, metric_results in scores.items():
                    if isinstance(metric_results, dict):
                        utt_score.update(metric_results)
                    else:
                        utt_score[metric_name] = metric_results

            except Exception as e:
                self.logger.error(f"Error processing file: {key} with error {e}")

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
    ) -> MetricSuite:
        """Load and configure metrics based on configuration."""
        metrics = {}

        for config in score_config:
            metric_name = config["name"]

            try:
                # Check if metric requires ground truth
                metadata = self.registry.get_metadata(metric_name)
                if metadata and metadata.requires_reference and not use_gt:
                    self.logger.warning(
                        f"Cannot use {metric_name} because no ground truth is provided"
                    )
                    continue

                if metadata and metadata.requires_text and not use_gt_text:
                    self.logger.warning(
                        f"Cannot use {metric_name} because no ground truth text is provided"
                    )
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
                self.logger.error(f"Failed to load metric {metric_name}: {e}")
                continue

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
    ) -> List[Dict[str, Any]]:
        """Score individual utterances."""

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
        existing_scores = _load_existing_jsonl_scores(output_file) if resume else {}
        completed_keys = set(existing_scores).intersection(gen_files)
        if resume and output_file:
            self.logger.info(
                "Resume enabled: found %d completed utterances in %s",
                len(completed_keys),
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
                resume=resume,
                num_workers=num_workers,
            )

        processor = ScoreProcessor(metric_suite, output_file, resume=resume)
        score_info = [
            existing_scores[key] for key in gen_files if key in existing_scores
        ]
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
                    continue

                # Step2: Load and validate ground truth audio
                gt_wav, gt_sr = None, None
                if gt_files is not None:
                    if key not in gt_files:
                        self.logger.warning(
                            f"Ground truth not found for key {key}, skipping"
                        )
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
                        continue

                # Step3: Load text information
                text = text_info.get(key) if text_info else None
                if text_info and key not in text_info:
                    self.logger.warning(f"Text not found for key {key}, skipping")
                    continue

                # Step4: Resample if needed
                gen_wav, gt_wav, gen_sr = self._align_sample_rates(
                    gen_wav, gt_wav, gen_sr, gt_sr
                )

                # Step5: Cache for batch processing
                utterance_info = (key, gen_wav, gt_wav, gen_sr, text)
                cache_info.append(utterance_info)

                if len(cache_info) >= batch_size:
                    score_info.extend(processor.process_batch(cache_info))
                    cache_info = []

            # Process remaining items
            if cache_info:
                score_info.extend(processor.process_batch(cache_info))

        finally:
            processor.close()

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
    ) -> List[Dict[str, Any]]:
        """Score explicitly ordered source pairs for each mixture.

        Each item in ``gen_source_files`` and ``gt_source_files`` is one
        source-specific SCP mapping. List order defines the source assignment.
        This path is intentionally separate from single-utterance scoring so a
        metric such as MAPSS cannot silently infer or permute source pairs.
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

        existing_scores = _load_existing_jsonl_scores(output_file) if resume else {}
        completed_keys = set(existing_scores).intersection(keys)
        score_by_key = {
            key: existing_scores[key] for key in keys if key in existing_scores
        }

        file_handle = None
        if output_file:
            mode = "a" if resume else "w"
            if resume:
                _ensure_append_starts_on_new_line(output_file)
            file_handle = open(output_file, mode, encoding="utf-8")

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
                    continue

                metadata = {
                    "key": key,
                    "sample_rate": 16000,
                }
                utt_score = {"key": key}
                scores = metric_suite.compute_all(predictions, references, metadata)
                for metric_name, metric_results in scores.items():
                    if isinstance(metric_results, dict):
                        utt_score.update(metric_results)
                    else:
                        utt_score[metric_name] = metric_results

                score_by_key[key] = utt_score
                if file_handle:
                    _write_jsonl_score(file_handle, utt_score)
        finally:
            if file_handle:
                file_handle.close()

        return [score_by_key[key] for key in keys if key in score_by_key]

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
        resume: bool,
        num_workers: int,
    ) -> List[Dict[str, Any]]:
        """Score utterances in process-local metric suites."""
        jobs = []
        for key in gen_files:
            if key in completed_keys:
                continue
            if gt_files is not None and key not in gt_files:
                self.logger.warning("Ground truth not found for key %s, skipping", key)
                continue
            if text_info is not None and key not in text_info:
                self.logger.warning("Text not found for key %s, skipping", key)
                continue
            jobs.append(
                (
                    key,
                    gen_files[key],
                    gt_files[key] if gt_files is not None else None,
                    text_info.get(key) if text_info is not None else None,
                    io,
                )
            )

        metric_specs = [
            (name, metric.__class__, metric.config)
            for name, metric in metric_suite.metrics.items()
        ]
        new_scores = {}
        file_handle = None
        if output_file:
            mode = "a" if resume else "w"
            if resume:
                _ensure_append_starts_on_new_line(output_file)
            file_handle = open(output_file, mode, encoding="utf-8")

        try:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_initialize_score_worker,
                initargs=(metric_specs,),
            ) as executor:
                results = executor.map(_score_utterance_worker, jobs)
                for result in tqdm(results, total=len(jobs)):
                    if result is not None:
                        new_scores[result["key"]] = result
                        if file_handle:
                            _write_jsonl_score(file_handle, result)
        finally:
            if file_handle:
                file_handle.close()

        score_info = [
            existing_scores.get(key, new_scores.get(key))
            for key in gen_files
            if key in existing_scores or key in new_scores
        ]
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
    ) -> List[Dict[str, Any]]:
        """Score all utterances one metric at a time to lower peak model memory."""

        use_gt = gt_files is not None
        use_gt_text = text_info is not None
        existing_scores = _load_existing_jsonl_scores(output_file) if resume else {}
        score_by_key = {
            key: dict(existing_scores.get(key, {"key": key})) for key in gen_files
        }

        for config in score_config:
            metric_name = config["name"]
            metadata = self.registry.get_metadata(metric_name)
            if metadata and metadata.category == MetricCategory.DISTRIBUTIONAL:
                self.logger.info("Skipping %s for utterance-level scoring", metric_name)
                continue

            metric_suite = self.load_metrics(
                [config],
                use_gt=use_gt,
                use_gt_text=use_gt_text,
                use_gpu=use_gpu,
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

            try:
                self.logger.info("Scoring utterances with metric %s", metric_name)
                metric_scores = self.score_utterances(
                    gen_files,
                    metric_suite,
                    gt_files=gt_files,
                    text_info=text_info,
                    output_file=None,
                    io=io,
                    batch_size=batch_size,
                    resume=False,
                )
                for utt_score in metric_scores:
                    key = utt_score.get("key")
                    if key is None:
                        continue
                    score_by_key.setdefault(key, {"key": key}).update(utt_score)
            finally:
                del metric_suite
                _release_metric_resources()

            _write_jsonl_scores(
                output_file,
                [score_by_key[key] for key in gen_files if key in score_by_key],
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
    ) -> Dict[str, Any]:
        """Score at corpus level (e.g., FAD, KID)."""

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

            except Exception as e:
                self.logger.error(f"Error computing corpus metric {name}: {e}")

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
