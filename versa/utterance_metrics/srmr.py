#!/usr/bin/env python3
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

import numpy as np

try:
    from srmrpy import srmr  # Import the srmr package for speech quality metrics
except ImportError:
    logger.info("srmr is not installed. Please use `tools/install_srmr.sh` to install")
    srmr = None

from versa.definition import BaseMetric, MetricMetadata, MetricCategory, MetricType


class SRMRMetric(BaseMetric):
    """Speech-to-Reverberation Modulation energy Ratio (SRMR) metric."""
    
    def _setup(self):
        """Initialize SRMR-specific components."""
        if srmr is None:
            raise ImportError(
                "srmr is not installed. Please use `tools/install_srmr.sh` to install"
            )
        
        # Set default parameters from config
        self.n_cochlear_filters = self.config.get("n_cochlear_filters", 23)
        self.low_freq = self.config.get("low_freq", 125)
        self.min_cf = self.config.get("min_cf", 4)
        self.max_cf = self.config.get("max_cf", 128)
        self.fast = self.config.get("fast", True)
        self.norm = self.config.get("norm", False)
    
    def compute(self, predictions: Any, references: Any = None, 
                metadata: Dict[str, Any] = None) -> Dict[str, float]:
        """Compute the SRMR score."""
        pred_x = predictions
        sample_rate = metadata.get("sample_rate", 16000) if metadata else 16000
        
        srmr_score = srmr(
            pred_x,
            sample_rate,
            n_cochlear_filters=self.n_cochlear_filters,
            low_freq=self.low_freq,
            min_cf=self.min_cf,
            max_cf=self.max_cf,
            fast=self.fast,
            norm=self.norm,
        )

        return {
            "srmr": srmr_score,
        }
    
    def get_metadata(self) -> MetricMetadata:
        """Return SRMR metric metadata."""
        return MetricMetadata(
            name="srmr",
            category=MetricCategory.INDEPENDENT,
            metric_type=MetricType.FLOAT,
            requires_reference=False,
            requires_text=False,
            gpu_compatible=False,
            auto_install=False,
            dependencies=["srmrpy"],
            description="Speech-to-Reverberation Modulation energy Ratio (SRMR) for speech quality assessment",
            paper_reference="http://www.individual.utoronto.ca/falkt/falk/pdf/FalkChan_TASLP2010.pdf",
            implementation_source="https://github.com/shimhz/SRMRpy.git"
        )


# Auto-registration function
def register_srmr_metric(registry):
    """Register SRMR metric with the registry."""
    metric_metadata = MetricMetadata(
        name="srmr",
        category=MetricCategory.INDEPENDENT,
        metric_type=MetricType.FLOAT,
        requires_reference=False,
        requires_text=False,
        gpu_compatible=False,
        auto_install=False,
        dependencies=["srmrpy"],
        description="Speech-to-Reverberation Modulation energy Ratio (SRMR) for speech quality assessment",
        paper_reference="http://www.individual.utoronto.ca/falkt/falk/pdf/FalkChan_TASLP2010.pdf",
        implementation_source="https://github.com/shimhz/SRMRpy.git"
    )
    registry.register(SRMRMetric, metric_metadata, aliases=["SRMR"])


if __name__ == "__main__":
    a = np.random.random(16000)
    
    # Test the new class-based metric
    config = {
        "n_cochlear_filters": 23,
        "low_freq": 125,
        "min_cf": 4,
        "max_cf": 128,
        "fast": True,
        "norm": False
    }
    metric = SRMRMetric(config)
    metadata = {"sample_rate": 16000}
    score = metric.compute(a, metadata=metadata)
    print("SRMR", score)
