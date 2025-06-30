#!/usr/bin/env python3

# Copyright 2025 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging


class MetricCategory(Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"
    NON_MATCH = "non_match"
    DISTRIBUTIONAL = "distributional"


class MetricType(Enum):
    STRING = "string"
    FLOAT = "float"
    INT = "int"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"
    TUPLE = "tuple"
    ARRAY = "array"
    TIME = "time"


@dataclass
class MetricMetadata:
    name: str
    category: MetricCategory
    metric_type: MetricType
    requires_reference: bool
    requires_text: bool
    gpu_compatible: bool
    auto_install: bool
    dependencies: List[str]
    description: str
    paper_reference: Optional[str] = None
    implementation_source: Optional[str] = None


class MetricRegistry:
    """Centralized registry for all metrics with automatic discovery."""

    def __init__(self):
        self._metrics: Dict[str, type] = {}
        self._metadata: Dict[str, MetricMetadata] = {}
        self._aliases: Dict[str, str] = {}

    def register(
        self, metric_class: type, metadata: MetricMetadata, aliases: List[str] = None
    ):
        """Register a metric with its metadata."""
        self._metrics[metadata.name] = metric_class
        self._metadata[metadata.name] = metadata

        # Register aliases
        if aliases:
            for alias in aliases:
                self._aliases[alias] = metadata.name

    def get_metric(self, name: str) -> type:
        """Get metric class by name or alias."""
        real_name = self._aliases.get(name, name)
        return self._metrics.get(real_name)

    def get_metadata(self, name: str) -> MetricMetadata:
        """Get metric metadata by name or alias."""
        real_name = self._aliases.get(name, name)
        return self._metadata.get(real_name)

    def list_metrics(
        self, category: MetricCategory = None, metric_type: MetricType = None
    ) -> List[str]:
        """List available metrics with optional filtering."""
        metrics = []
        for name, metadata in self._metadata.items():
            if category and metadata.category != category:
                continue
            if metric_type and metadata.metric_type != metric_type:
                continue
            metrics.append(name)
        return sorted(metrics)


class BaseMetric(ABC):
    """Abstract base class for all metrics."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._setup()

    @abstractmethod
    def _setup(self):
        """Initialize metric-specific components."""
        pass

    @abstractmethod
    def compute(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Any:
        """Compute the metric score."""
        pass

    @abstractmethod
    def get_metadata(self) -> MetricMetadata:
        """Return metric metadata."""
        pass

    def validate_inputs(self, predictions: Any, references: Any = None) -> bool:
        """Validate input data before computation."""
        return True

    def preprocess(self, data: Any) -> Any:
        """Preprocess data before metric computation."""
        return data

    def postprocess(self, scores: Any) -> Any:
        """Postprocess scores after computation."""
        return scores


class GPUMetric(BaseMetric):
    """Base class for GPU-compatible metrics."""

    def __init__(self, config: Dict[str, Any] = None, device: str = "cuda"):
        self.device = device
        super().__init__(config)

    def to_device(self, data: Any) -> Any:
        """Move data to specified device."""
        if hasattr(data, "to"):
            return data.to(self.device)
        return data


class MetricFactory:
    """Factory for creating metric instances with dependency management."""

    def __init__(self, registry: MetricRegistry):
        self.registry = registry
        self._dependency_cache = {}

    def create_metric(self, name: str, config: Dict[str, Any] = None) -> BaseMetric:
        """Create a metric instance with proper dependency resolution."""
        metadata = self.registry.get_metadata(name)
        metric_class = self.registry.get_metric(name)

        if not metric_class:
            raise ValueError(f"Metric '{name}' not found in registry")

        # Check and install dependencies
        self._ensure_dependencies(metadata.dependencies)

        return metric_class(config)

    def create_metric_suite(
        self, metric_names: List[str], config: Dict[str, Any] = None
    ) -> "MetricSuite":
        """Create a suite of metrics."""
        metrics = {}
        for name in metric_names:
            metrics[name] = self.create_metric(name, config.get(name, {}))
        return MetricSuite(metrics)

    def _ensure_dependencies(self, dependencies: List[str]):
        """Ensure all dependencies are available."""
        for dep in dependencies:
            if dep not in self._dependency_cache:
                try:
                    __import__(dep)
                    self._dependency_cache[dep] = True
                except ImportError:
                    self.logger.warning(f"Dependency '{dep}' not available")
                    self._dependency_cache[dep] = False


class MetricSuite:
    """Container for multiple metrics with batch processing capabilities."""

    def __init__(self, metrics: Dict[str, BaseMetric]):
        self.metrics = metrics
        self.logger = logging.getLogger(self.__class__.__name__)

    def compute_all(
        self, predictions: Any, references: Any = None, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Compute all metrics in the suite."""
        results = {}
        for name, metric in self.metrics.items():
            try:
                results[name] = metric.compute(predictions, references, metadata)
            except Exception as e:
                self.logger.error(f"Error computing metric '{name}': {e}")
                results[name] = None
        return results

    def compute_parallel(
        self,
        predictions: Any,
        references: Any = None,
        metadata: Dict[str, Any] = None,
        n_workers: int = 4,
    ) -> Dict[str, Any]:
        """Compute metrics in parallel."""
        # Implementation for parallel metric computation
        pass

    def filter_by_category(self, category: MetricCategory) -> "MetricSuite":
        """Filter metrics by category."""
        filtered_metrics = {
            name: metric
            for name, metric in self.metrics.items()
            if metric.get_metadata().category == category
        }
        return MetricSuite(filtered_metrics)
