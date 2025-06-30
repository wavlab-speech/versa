#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import json
import kaldiio
import librosa
import soundfile as sf
import yaml
from typing import Dict, List, Optional, Any, Union
from tqdm import tqdm

from versa.definition import (
    BaseMetric,
    GPUMetric, 
    MetricRegistry,
    MetricFactory,
    MetricSuite,
    MetricCategory,
    MetricType,
    MetricMetadata
)
from versa.metrics import STR_METRIC, NUM_METRIC
from versa.utils_shared import (
    check_all_same,
    check_minimum_length,
    default_numpy_serializer,
    find_files,
    load_audio,
    wav_normalize,
)


def audio_loader_setup(audio, io):
    # get ready compute embeddings
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


class ScoreProcessor:
    """Handles batch processing and caching of scores."""
    
    def __init__(self, metric_suite: MetricSuite, output_file: Optional[str] = None):
        self.metric_suite = metric_suite
        self.output_file = output_file
        self.logger = logging.getLogger(self.__class__.__name__)
        
        if output_file:
            self.file_handle = open(output_file, "w", encoding="utf-8")
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
                    "general_cache": {"whisper_hyp_text": None}
                }
                
                # Compute all metrics
                scores = self.metric_suite.compute_all(
                    predictions=gen_wav,
                    references=gt_wav,
                    metadata=metadata
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
                printable_result = json.dumps(utt_score, default=default_numpy_serializer)
                self.file_handle.write(f"{printable_result}\n")
        
        return batch_score_info
    
    def close(self):
        """Close file handle if open."""
        if self.file_handle:
            self.file_handle.close()


class VersaScorer:
    """Main scorer class that orchestrates the scoring process."""
    
    def __init__(self, registry: MetricRegistry = None):
        self.registry = registry or self._create_default_registry()
        self.factory = MetricFactory(self.registry)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _create_default_registry(self) -> MetricRegistry:
        """Create and populate the default metric registry."""
        registry = MetricRegistry()
        # This would be populated by importing all metric modules
        # and having them auto-register themselves
        return registry
    
    def load_metrics(self, score_config: List[Dict[str, Any]], 
                    use_gt: bool = True, use_gt_text: bool = False, 
                    use_gpu: bool = False) -> MetricSuite:
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
                
                # Create metric instance
                metric_config = {**config, "use_gpu": use_gpu}
                metric = self.factory.create_metric(metric_name, metric_config)
                metrics[metric_name] = metric
                
                self.logger.info(f"Loaded {metric_name} successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load metric {metric_name}: {e}")
                continue
        
        return MetricSuite(metrics)
    
    def score_utterances(self, gen_files: Dict[str, str], 
                        metric_suite: MetricSuite,
                        gt_files: Optional[Dict[str, str]] = None,
                        text_info: Optional[Dict[str, str]] = None,
                        output_file: Optional[str] = None,
                        io: str = "kaldi",
                        batch_size: int = 1) -> List[Dict[str, Any]]:
        """Score individual utterances."""
        
        processor = ScoreProcessor(metric_suite, output_file)
        score_info = []
        cache_info = []
        
        try:
            for key in tqdm(gen_files.keys()):
                # Step1: Load and validate generated audio
                gen_sr, gen_wav = load_audio(gen_files[key], io)
                gen_wav = wav_normalize(gen_wav)
                
                if not self._validate_audio(gen_wav, gen_sr, key, "generated"):
                    continue
                
                # Step2: Load and validate ground truth audio
                gt_wav, gt_sr = None, None
                if gt_files is not None:
                    if key not in gt_files:
                        self.logger.warning(f"Ground truth not found for key {key}, skipping")
                        continue
                    
                    gt_sr, gt_wav = load_audio(gt_files[key], io)
                    gt_wav = wav_normalize(gt_wav)
                    
                    if not self._validate_audio(gt_wav, gt_sr, key, "ground truth"):
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
    
    def score_corpus(self, gen_files: Dict[str, str],
                    metric_suite: MetricSuite,
                    base_files: Optional[Dict[str, str]] = None,
                    text_info: Optional[Dict[str, str]] = None,
                    output_file: Optional[str] = None) -> Dict[str, Any]:
        """Score at corpus level (e.g., FAD, KID)."""
        
        score_info = {}
        
        # Filter for distributional metrics
        distributional_metrics = metric_suite.filter_by_category(
            MetricCategory.DISTRIBUTIONAL
        )
        
        for name, metric in distributional_metrics.metrics.items():
            try:
                metadata = {
                    "baseline_files": base_files,
                    "text_info": text_info
                }
                
                score_result = metric.compute(
                    predictions=gen_files,
                    references=base_files,
                    metadata=metadata
                )
                score_info.update({name: score_result})
                
            except Exception as e:
                self.logger.error(f"Error computing corpus metric {name}: {e}")
        
        if output_file:
            with open(output_file, "w") as f:
                yaml.dump(score_info, f)
        
        return score_info
    
    def _validate_audio(self, wav: Any, sr: int, key: str, audio_type: str) -> bool:
        """Validate audio data."""
        # Length check
        if not check_minimum_length(wav.shape[0] / sr, []):  # Metric names would be passed here
            self.logger.warning(
                f"Audio {key} ({audio_type}, length {wav.shape[0] / sr}) is too short, skipping"
            )
            return False
        
        # Check for silent audio
        if check_all_same(wav):
            self.logger.warning(f"Audio {key} ({audio_type}) has only the same value, skipping")
            return False
        
        return True
    
    def _align_sample_rates(self, gen_wav: Any, gt_wav: Any, 
                           gen_sr: int, gt_sr: Optional[int]) -> tuple:
        """Align sample rates between generated and ground truth audio."""
        if gt_sr is None:
            return gen_wav, gt_wav, gen_sr
        
        if gen_sr > gt_sr:
            self.logger.warning("Resampling generated audio to match ground truth")
            gen_wav = librosa.resample(gen_wav, orig_sr=gen_sr, target_sr=gt_sr)
            gen_sr = gt_sr
        elif gen_sr < gt_sr:
            self.logger.warning("Resampling ground truth audio to match generated audio")
            gt_wav = librosa.resample(gt_wav, orig_sr=gt_sr, target_sr=gen_sr)
        
        return gen_wav, gt_wav, gen_sr


def compute_summary(score_info: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics from individual scores."""
    if not score_info:
        return {}
    
    summary = {}
    for key in score_info[0].keys():
        if key not in NUM_METRIC:
            continue
        
        values = [score[key] for score in score_info if key in score and score[key] is not None]
        if not values:
            continue
        
        summary[key] = sum(values)
        if "_wer" not in key and "_cer" not in key:
            summary[key] /= len(values)
    
    return summary
