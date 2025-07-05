#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Module for NORESQA speech quality assessment metrics."""

import logging
import os
import sys
import warnings
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# Handle optional dependencies
try:
    import fairseq

    FAIRSEQ_AVAILABLE = True
except ImportError:
    logger.warning(
        "fairseq is not installed. Please use `tools/install_fairseq.sh` to install"
    )
    fairseq = None
    FAIRSEQ_AVAILABLE = False

# Setup NORESQA path
base_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../tools/Noresqa")
)
sys.path.insert(0, base_path)

try:
    from model import NORESQA
    from utils import (
        feats_loading,
        model_prediction_noresqa,
        model_prediction_noresqa_mos,
    )

    NORESQA_AVAILABLE = True
except ImportError:
    logger.warning(
        "noresqa is not installed. Please use `tools/install_noresqa.sh` to install"
    )
    NORESQA = None
    feats_loading = None
    model_prediction_noresqa = None
    model_prediction_noresqa_mos = None
    NORESQA_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class NoresqaNotAvailableError(RuntimeError):
    """Exception raised when noresqa is required but not available."""

    pass


def is_noresqa_available():
    """
    Check if the noresqa package is available.

    Returns:
        bool: True if noresqa is available, False otherwise.
    """
    return NORESQA_AVAILABLE and FAIRSEQ_AVAILABLE


class NoresqaMetric(BaseMetric):
    """NORESQA speech quality assessment metric."""

    TARGET_FS = 16000  # NORESQA model's expected sampling rate

    def _setup(self):
        """Initialize NORESQA-specific components."""
        if not NORESQA_AVAILABLE:
            raise ImportError(
                "noresqa is not installed. Please use `tools/install_noresqa.sh` to install"
            )
        if not FAIRSEQ_AVAILABLE:
            raise ImportError(
                "fairseq is not installed. Please use `tools/install_fairseq.sh` to install"
            )

        self.model_tag = self.config.get("model_tag", "default")
        self.metric_type = self.config.get(
            "metric_type", 1
        )  # 0: NORESQA-score, 1: NORESQA-MOS
        self.cache_dir = self.config.get("cache_dir", "versa_cache/noresqa_model")
        self.use_gpu = self.config.get("use_gpu", False)

        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize NORESQA model: {str(e)}") from e

    def _setup_model(self):
        """Setup the NORESQA model."""
        device = "cuda" if self.use_gpu else "cpu"

        if self.model_tag == "default":
            if not os.path.isdir(self.cache_dir):
                logger.info("Creating checkpoints directory")
                os.makedirs(self.cache_dir)

            url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
            w2v_path = os.path.join(self.cache_dir, "wav2vec_small.pt")
            if not os.path.isfile(w2v_path):
                logger.info("Downloading wav2vec 2.0 started")
                urlretrieve(url_w2v, w2v_path)
                logger.info("wav2vec 2.0 download completed")

            model = NORESQA(
                output=40,
                output2=40,
                metric_type=self.metric_type,
                config_path=w2v_path,
            )

            if self.metric_type == 0:
                model_checkpoint_path = "{}/models/model_noresqa.pth".format(base_path)
                # Suppress PyTorch config registration warnings during model loading
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Skipping config registration for"
                    )
                    state = torch.load(model_checkpoint_path, map_location="cpu")[
                        "state_base"
                    ]
            elif self.metric_type == 1:
                model_checkpoint_path = "{}/models/model_noresqa_mos.pth".format(
                    base_path
                )
                # Suppress PyTorch config registration warnings during model loading
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message="Skipping config registration for"
                    )
                    state = torch.load(model_checkpoint_path, map_location="cpu")[
                        "state_dict"
                    ]

            pretrained_dict = {}
            for k, v in state.items():
                if "module" in k:
                    pretrained_dict[k.replace("module.", "")] = v
                else:
                    pretrained_dict[k] = v
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.load_state_dict(pretrained_dict)

            # change device as needed
            model.to(device)
            model.device = device
            model.eval()

        else:
            raise NotImplementedError(f"Model tag '{self.model_tag}' not implemented")

        return model

    def compute(
        self, predictions: Any, references: Any, metadata: Dict[str, Any] = None
    ) -> Dict[str, Union[float, str]]:
        """Calculate NORESQA score for speech quality assessment.

        Args:
            predictions: Predicted audio signal.
            references: Ground truth audio signal.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing NORESQA score.
        """
        pred_x = predictions
        gt_x = references
        fs = metadata.get("sample_rate", 16000) if metadata else 16000

        # Validate inputs
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")
        if gt_x is None:
            raise ValueError("Reference signal must be provided")

        pred_x = np.asarray(pred_x)
        gt_x = np.asarray(gt_x)

        # Resample to 16kHz (NORESQA only works with 16kHz)
        if fs != self.TARGET_FS:
            gt_x = librosa.resample(gt_x, orig_sr=fs, target_sr=self.TARGET_FS)
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=self.TARGET_FS)

        nmr_feat, test_feat = feats_loading(
            pred_x, gt_x, noresqa_or_noresqaMOS=self.metric_type
        )
        test_feat = (
            torch.from_numpy(test_feat).float().to(self.model.device).unsqueeze(0)
        )
        nmr_feat = torch.from_numpy(nmr_feat).float().to(self.model.device).unsqueeze(0)

        with torch.no_grad():
            if self.metric_type == 0:
                noresqa_pout, noresqa_qout = model_prediction_noresqa(
                    test_feat, nmr_feat, self.model
                )
                return {"noresqa_score": noresqa_pout}
            elif self.metric_type == 1:
                mos_score = model_prediction_noresqa_mos(
                    test_feat, nmr_feat, self.model
                )
                return {"noresqa_score": mos_score}
            else:
                raise ValueError(f"Invalid metric_type: {self.metric_type}")

    def get_metadata(self) -> MetricMetadata:
        """Return NORESQA metric metadata."""
        metric_name = "noresqa_mos" if self.metric_type == 1 else "noresqa_score"
        description = "NORESQA-MOS" if self.metric_type == 1 else "NORESQA-score"

        return MetricMetadata(
            name=metric_name,
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["fairseq", "torch", "librosa", "numpy"],
            description=f"{description}: Non-matching reference based speech quality assessment",
            paper_reference="https://arxiv.org/abs/2104.09411",
            implementation_source="https://github.com/facebookresearch/NORESQA",
        )


def register_noresqa_metric(registry):
    """Register NORESQA metric with the registry."""
    # Register both metric types
    for metric_type, metric_name in [(0, "noresqa_score"), (1, "noresqa_mos")]:
        description = "NORESQA-MOS" if metric_type == 1 else "NORESQA-score"

        metric_metadata = MetricMetadata(
            name=metric_name,
            category=MetricCategory.DEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=True,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["fairseq", "torch", "librosa", "numpy"],
            description=f"{description}: Non-matching reference based speech quality assessment",
            paper_reference="https://arxiv.org/abs/2104.09411",
            implementation_source="https://github.com/facebookresearch/NORESQA",
        )
        registry.register(
            NoresqaMetric,
            metric_metadata,
            aliases=[f"Noresqa{metric_type}", metric_name],
        )
