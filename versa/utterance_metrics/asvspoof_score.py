#!/usr/bin/env python3

# Copyright 2024 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Author: You (Neil) Zhang

"""
This is to evaluate the generated speech with a deepfake detection model.
We include the AASIST model trained on ASVspoof 2019 LA dataset to
output the confidence score of whether the speech input is a deepfake.
Please refer to https://github.com/clovaai/aasist for more details.
"""

import json
import logging
import os
import sys
from typing import Dict, Any, Optional, Union

import librosa
import numpy as np
import torch

logger = logging.getLogger(__name__)

# Handle optional AASIST dependency
try:
    sys.path.append("./tools/checkpoints/aasist")
    from models.AASIST import Model as AASIST  # noqa: E402
    AASIST_AVAILABLE = True
except ImportError:
    logger.warning(
        "AASIST is not properly installed. "
        "Please install following https://github.com/clovaai/aasist"
    )
    AASIST = None
    AASIST_AVAILABLE = False

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class AASISTNotAvailableError(RuntimeError):
    """Exception raised when AASIST is required but not available."""
    pass


def is_aasist_available():
    """
    Check if the AASIST package is available.

    Returns:
        bool: True if AASIST is available, False otherwise.
    """
    return AASIST_AVAILABLE


class ASVSpoofMetric(BaseMetric):
    """ASVspoof deepfake detection metric using AASIST model."""

    def _setup(self):
        """Initialize ASVspoof-specific components."""
        if not AASIST_AVAILABLE:
            raise ImportError(
                "AASIST is not properly installed. Please install following https://github.com/clovaai/aasist"
            )
        
        self.model_tag = self.config.get("model_tag", "default")
        self.model_path = self.config.get("model_path", None)
        self.model_config = self.config.get("model_config", None)
        self.use_gpu = self.config.get("use_gpu", False)
        
        self.device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
        
        try:
            self.model = self._setup_model()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize AASIST model: {str(e)}") from e

    def _setup_model(self):
        """Setup the AASIST model."""
        if self.model_path is not None and self.model_config is not None:
            with open(self.model_config, "r") as f_json:
                config = json.loads(f_json.read())
                model = AASIST(config["model_config"]).to(self.device)
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        else:
            if self.model_tag == "default":
                model_root = "./tools/checkpoints/aasist"
                model_config = os.path.join(model_root, "config/AASIST.conf")
                model_path = os.path.join(model_root, "models/weights/AASIST.pth")

                with open(model_config, "r") as f_json:
                    config = json.loads(f_json.read())
                    model = AASIST(config["model_config"]).to(self.device)
                    model.load_state_dict(torch.load(model_path, map_location=self.device))
            else:
                raise NotImplementedError(f"Model tag '{self.model_tag}' not implemented")
        
        model.device = self.device
        return model

    def compute(self, predictions: Any, references: Any = None, 
                metadata: Dict[str, Any] = None) -> Dict[str, Union[float, str]]:
        """Calculate ASVspoof score for audio.

        Args:
            predictions: Audio signal to evaluate.
            references: Not used for this metric.
            metadata: Optional metadata containing sample_rate.

        Returns:
            dict: Dictionary containing the ASVspoof score.
        """
        pred_x = predictions
        fs = metadata.get("sample_rate", 16000) if metadata else 16000
        
        # Validate input
        if pred_x is None:
            raise ValueError("Predicted signal must be provided")
        
        pred_x = np.asarray(pred_x)
        
        # NOTE(jiatong): only work for 16000 Hz
        if fs != 16000:
            pred_x = librosa.resample(pred_x, orig_sr=fs, target_sr=16000)

        pred_x = torch.from_numpy(pred_x).unsqueeze(0).float().to(self.device)
        self.model.eval()
        with torch.no_grad():
            _, output = self.model(pred_x)
        output = torch.softmax(output, dim=1)
        output = output.squeeze(0).cpu().numpy()
        
        return {"asvspoof_score": output[1]}

    def get_metadata(self) -> MetricMetadata:
        """Return ASVspoof metric metadata."""
        return MetricMetadata(
            name="asvspoof",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=True,
            auto_install=False,
            dependencies=["torch", "librosa", "numpy"],
            description="ASVspoof deepfake detection score using AASIST model for speech authenticity assessment",
            paper_reference="https://github.com/clovaai/aasist",
            implementation_source="https://github.com/clovaai/aasist"
        )


def register_asvspoof_metric(registry):
    """Register ASVspoof metric with the registry."""
    metric_metadata = MetricMetadata(
        name="asvspoof",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=True,
        auto_install=False,
        dependencies=["torch", "librosa", "numpy"],
        description="ASVspoof deepfake detection score using AASIST model for speech authenticity assessment",
        paper_reference="https://github.com/clovaai/aasist",
        implementation_source="https://github.com/clovaai/aasist"
    )
    registry.register(ASVSpoofMetric, metric_metadata, aliases=["ASVSpoof", "asvspoof_score"])


# Legacy functions for backward compatibility
def deepfake_detection_model_setup(
    model_tag="default", model_path=None, model_config=None, use_gpu=False
):
    """Setup deepfake detection model (legacy function).

    Args:
        model_tag (str): Model tag. Defaults to "default".
        model_path (str, optional): Path to model weights. Defaults to None.
        model_config (str, optional): Path to model config. Defaults to None.
        use_gpu (bool, optional): Whether to use GPU. Defaults to False.

    Returns:
        AASIST: The loaded model.
    """
    config = {
        "model_tag": model_tag,
        "model_path": model_path,
        "model_config": model_config,
        "use_gpu": use_gpu
    }
    metric = ASVSpoofMetric(config)
    return metric.model


def asvspoof_metric(model, pred_x, fs):
    """Calculate ASVspoof score for audio (legacy function).

    Args:
        model (AASIST): The loaded deepfake detection model.
        pred_x (np.ndarray): Audio signal.
        fs (int): Sampling rate.

    Returns:
        dict: Dictionary containing the ASVspoof score.
    """
    config = {"use_gpu": hasattr(model, 'device') and model.device == 'cuda'}
    metric = ASVSpoofMetric(config)
    metric.model = model
    metadata = {"sample_rate": fs}
    return metric.compute(pred_x, metadata=metadata)


if __name__ == "__main__":
    a = np.random.random(16000)
    
    # Test the new class-based metric
    config = {"use_gpu": False}
    metric = ASVSpoofMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, metadata=metadata)
    print(f"metrics: {score}") 