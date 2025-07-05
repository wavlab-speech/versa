#!/usr/bin/env python3

# Copyright 2024 Adapted from signal_metric example
# Chroma-based distance estimation with dynamic programming alignment
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, Any, Optional, Union, Tuple, List

import librosa
import numpy as np
from scipy.spatial.distance import cosine, euclidean

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType

logger = logging.getLogger(__name__)


def calculate_chroma_features(audio, sr=22050, feature_type="stft", **kwargs):
    """
    Calculate chroma features using different librosa methods.

    Args:
        audio: Input audio signal
        sr: Sample rate
        feature_type: 'stft', 'cqt', 'vqt', or 'cens'
        **kwargs: Additional parameters for librosa functions

    Returns:
        Chroma feature matrix (12 x time_frames)
    """
    if feature_type == "stft":
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, **kwargs)
    elif feature_type == "cqt":
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr, **kwargs)
    elif feature_type == "vqt":
        chroma = librosa.feature.chroma_vqt(y=audio, sr=sr, **kwargs)
    elif feature_type == "cens":
        chroma = librosa.feature.chroma_cens(y=audio, sr=sr, **kwargs)
    else:
        raise ValueError(f"Unsupported feature_type: {feature_type}")

    return chroma


def dtw_distance(
    x, y, distance_metric="cosine", scale_factor=100.0, normalize_by_path=True
):
    """
    Dynamic Time Warping distance between two feature sequences.

    Args:
        x: First feature sequence (features x time1)
        y: Second feature sequence (features x time2)
        distance_metric: 'cosine', 'euclidean', or callable
        scale_factor: Multiplicative factor to scale up the distance
        normalize_by_path: Whether to normalize by path length

    Returns:
        DTW distance and alignment path
    """
    n, m = x.shape[1], y.shape[1]

    # Distance function
    if distance_metric == "cosine":
        dist_func = lambda a, b: cosine(a, b)
    elif distance_metric == "euclidean":
        dist_func = lambda a, b: euclidean(a, b)
    elif callable(distance_metric):
        dist_func = distance_metric
    else:
        raise ValueError(f"Unsupported distance_metric: {distance_metric}")

    # Initialize DTW matrix
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0

    # Fill DTW matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = dist_func(x[:, i - 1], y[:, j - 1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i - 1, j],  # insertion
                dtw_matrix[i, j - 1],  # deletion
                dtw_matrix[i - 1, j - 1],  # match
            )

    # Backtrack to find alignment path
    path = []
    i, j = n, m
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        if (
            dtw_matrix[i - 1, j - 1] <= dtw_matrix[i - 1, j]
            and dtw_matrix[i - 1, j - 1] <= dtw_matrix[i, j - 1]
        ):
            i, j = i - 1, j - 1
        elif dtw_matrix[i - 1, j] <= dtw_matrix[i, j - 1]:
            i = i - 1
        else:
            j = j - 1

    path.reverse()

    # Calculate final distance with scaling options
    final_distance = dtw_matrix[n, m]

    if normalize_by_path:
        final_distance = final_distance / len(path)

    # Apply scaling factor
    final_distance *= scale_factor

    return final_distance, path


def calculate_chroma_distance(
    pred_x,
    gt_x,
    sr=22050,
    feature_type="stft",
    distance_metric="cosine",
    normalize=True,
    scale_factor=100.0,
    normalize_by_path=True,
    **chroma_kwargs,
):
    """
    Calculate chroma-based distance with DTW alignment.

    Args:
        pred_x: Predicted audio signal
        gt_x: Ground truth audio signal
        sr: Sample rate
        feature_type: Chroma feature type ('stft', 'cqt', 'vqt', 'cens')
        distance_metric: Distance metric for DTW
        normalize: Whether to normalize chroma features
        scale_factor: Multiplicative scaling factor for distance
        normalize_by_path: Whether to normalize by DTW path length
        **chroma_kwargs: Additional parameters for chroma extraction

    Returns:
        DTW distance between chroma features
    """
    # Extract chroma features
    chroma_pred = calculate_chroma_features(
        pred_x, sr=sr, feature_type=feature_type, **chroma_kwargs
    )
    chroma_gt = calculate_chroma_features(
        gt_x, sr=sr, feature_type=feature_type, **chroma_kwargs
    )

    # Normalize features if requested
    if normalize:
        chroma_pred = librosa.util.normalize(chroma_pred, axis=0)
        chroma_gt = librosa.util.normalize(chroma_gt, axis=0)

    # Calculate DTW distance
    dtw_dist, alignment_path = dtw_distance(
        chroma_pred,
        chroma_gt,
        distance_metric,
        scale_factor=scale_factor,
        normalize_by_path=normalize_by_path,
    )

    return dtw_dist, alignment_path


class ChromaAlignmentMetric(BaseMetric):
    """Chroma-based distance estimation with dynamic programming alignment."""

    def _setup(self):
        """Initialize Chroma Alignment-specific components."""
        self.sample_rate = self.config.get("sample_rate", 22050)
        self.feature_types = self.config.get("feature_types", ["stft", "cqt", "cens"])
        self.distance_metrics = self.config.get(
            "distance_metrics", ["cosine", "euclidean"]
        )
        self.scale_factor = self.config.get("scale_factor", 100.0)
        self.normalize = self.config.get("normalize", True)
        self.normalize_by_path = self.config.get("normalize_by_path", True)
        self.return_alignment = self.config.get("return_alignment", False)
        self.chroma_kwargs = self.config.get("chroma_kwargs", {})

    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate chroma-based distance metrics.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing chroma distance metrics.
        """
        pred_x = predictions
        gt_x = references
        sr = (
            metadata.get("sample_rate", self.sample_rate)
            if metadata
            else self.sample_rate
        )

        # Validate inputs
        if pred_x is None or gt_x is None:
            raise ValueError("Both predicted and ground truth signals must be provided")

        pred_x = np.asarray(pred_x)
        gt_x = np.asarray(gt_x)

        # Ensure 1D arrays
        if pred_x.ndim > 1:
            pred_x = pred_x.flatten()
        if gt_x.ndim > 1:
            gt_x = gt_x.flatten()

        results = {}
        alignments = {} if self.return_alignment else None

        # Calculate metrics for different feature types and distance metrics
        for feat_type in self.feature_types:
            for dist_metric in self.distance_metrics:
                try:
                    dtw_dist, alignment = calculate_chroma_distance(
                        pred_x,
                        gt_x,
                        sr=sr,
                        feature_type=feat_type,
                        distance_metric=dist_metric,
                        scale_factor=self.scale_factor,
                        normalize=self.normalize,
                        normalize_by_path=self.normalize_by_path,
                        **self.chroma_kwargs,
                    )

                    metric_name = f"chroma_{feat_type}_{dist_metric}_dtw"
                    results[metric_name] = dtw_dist

                    if self.return_alignment and alignments is not None:
                        alignments[metric_name] = alignment

                except Exception as e:
                    logger.warning(
                        f"Could not calculate {feat_type} with {dist_metric}: {e}"
                    )
                    continue

        # Add additional scaled variants
        try:
            # Raw DTW distance (no path normalization, higher scale)
            dtw_dist_raw, _ = calculate_chroma_distance(
                pred_x,
                gt_x,
                sr=sr,
                feature_type="stft",
                distance_metric="cosine",
                scale_factor=1000.0,
                normalize_by_path=True,
                normalize=self.normalize,
                **self.chroma_kwargs,
            )
            results["chroma_stft_cosine_dtw_raw"] = dtw_dist_raw

            # Log-scaled distance
            dtw_dist_base, _ = calculate_chroma_distance(
                pred_x,
                gt_x,
                sr=sr,
                feature_type="stft",
                distance_metric="cosine",
                scale_factor=1.0,
                normalize_by_path=True,
                normalize=self.normalize,
                **self.chroma_kwargs,
            )
            results["chroma_stft_cosine_dtw_log"] = (
                -np.log10(dtw_dist_base + 1e-10) * 10
            )

        except Exception as e:
            logger.warning(f"Could not calculate additional scaled metrics: {e}")

        if self.return_alignment and alignments is not None:
            results["alignments"] = alignments

        return results

    def get_metadata(self) -> MetricMetadata:
        """Return Chroma Alignment metric metadata."""
        return MetricMetadata(
            name="chroma_alignment",
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=False,
            auto_install=False,
            dependencies=["librosa", "numpy", "scipy"],
            description="Chroma-based distance estimation with dynamic programming alignment for audio similarity assessment",
            paper_reference="https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html",
            implementation_source="https://github.com/librosa/librosa",
        )


def register_chroma_alignment_metric(registry):
    """Register Chroma Alignment metric with the registry."""
    metric_metadata = MetricMetadata(
        name="chroma_alignment",
        category=MetricCategory.DEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=True,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["librosa", "numpy", "scipy"],
        description="Chroma-based distance estimation with dynamic programming alignment for audio similarity assessment",
        paper_reference="https://librosa.org/doc/latest/generated/librosa.feature.chroma_stft.html",
        implementation_source="https://github.com/librosa/librosa",
    )
    registry.register(
        ChromaAlignmentMetric,
        metric_metadata,
        aliases=["ChromaAlignment", "chroma_alignment"],
    )


if __name__ == "__main__":
    # Create test signals with different lengths
    sr = 22050
    duration1 = 2.0  # 2 seconds
    duration2 = 2.5  # 2.5 seconds

    # Generate test signals (sine waves with different frequencies)
    t1 = np.linspace(0, duration1, int(sr * duration1))
    t2 = np.linspace(0, duration2, int(sr * duration2))

    pred_signal = np.sin(2 * np.pi * 440 * t1)  # A4 note
    gt_signal = np.sin(2 * np.pi * 440 * t2)  # Same note, different length

    # Test the new class-based metric
    config = {"scale_factor": 100.0}
    metric = ChromaAlignmentMetric(config)
    metadata = {"sample_rate": sr}
    score = metric.compute(pred_signal, gt_signal, metadata=metadata)
    print(f"metrics: {score}")
