"""Regression coverage for backend-free discovery and isolated metric selection."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from versa.definition import MetricRegistry
from versa.metric_discovery import create_metric_discovery_registry
from versa import metric_registry


@pytest.fixture(scope="module")
def source_registry():
    """Parse source once for the complete name-to-module regression matrix."""
    return create_metric_discovery_registry()


def test_lightweight_imports_in_fresh_process():
    """Reject backend and model dependencies even when installed on the test host."""
    code = """
import sys
from importlib.abc import MetaPathFinder
class BlockBackends(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchaudio', 'transformers'} or fullname.startswith(('versa.utterance_metrics.', 'versa.corpus_metrics.', 'versa.sequence_metrics.')):
            raise AssertionError('Unexpected backend import: ' + fullname)
sys.meta_path.insert(0, BlockBackends())
import versa
from versa.metric_discovery import create_metric_discovery_registry
import versa.reporting
from versa.metric_registry import _metric_specs_by_name
registry = create_metric_discovery_registry()
assert registry.list_metrics()
assert _metric_specs_by_name()
assert 'PesqMetric' in dir(versa)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    )


def test_all_discovered_names_have_runtime_owners():
    """Keep discovery, aliases, and generated Qwen names aligned with runtime specs."""
    registry = create_metric_discovery_registry()
    specs = metric_registry._metric_specs_by_name()
    names = set(registry.list_metrics()) | set(registry.list_aliases())
    assert names <= specs.keys()
    assert any(name.startswith("qwen_omni_") for name in names)
    assert any(name.startswith("qwen2_audio_") for name in names)


@pytest.mark.parametrize("name", sorted(metric_registry._metric_specs_by_name()))
def test_selected_name_imports_only_its_owner(name, monkeypatch, source_registry):
    """Resolve every canonical name and alias without probing unrelated modules."""
    source = source_registry
    metadata = source.get_metadata(name)
    spec = metric_registry._metric_specs_by_name()[name]
    calls = []
    concrete = type(
        "Concrete", (), {"compute": lambda: None, "get_metadata": lambda: None}
    )

    def register(registry):
        """Model a backend registering its canonical metric and aliases."""
        registry.register(concrete, metadata, source.get_aliases(name))

    def import_module(module_name):
        """Fail if selection touches a backend other than the declared owner."""
        calls.append(module_name)
        assert module_name == spec.module_name
        return SimpleNamespace(**{symbol: register for symbol in spec.symbols})

    monkeypatch.setattr(metric_registry.importlib, "import_module", import_module)
    registry = MetricRegistry()
    metric_registry.register_metric_for_config(registry, name)
    metric_registry.register_metric_for_config(registry, name)
    assert registry.get_metric(name) is concrete
    assert calls == [spec.module_name]


def test_missing_backend_reports_selected_metric_without_fallback(monkeypatch):
    """Preserve the dependency failure and installation hint without bulk imports."""
    calls = []

    def missing(module_name):
        """Simulate a selected backend with an unavailable optional dependency."""
        calls.append(module_name)
        raise ModuleNotFoundError("No module named pesq")

    monkeypatch.setattr(metric_registry.importlib, "import_module", missing)
    with pytest.raises(ImportError, match="pip install pesq") as error:
        metric_registry.register_metric_for_config(MetricRegistry(), "pesq")
    assert isinstance(error.value.__cause__, ModuleNotFoundError)
    assert calls == ["versa.utterance_metrics.pesq_score"]


def test_unknown_metric_does_not_import_backends(monkeypatch):
    """Reject unknown metric names before any runtime import."""

    def unexpected(name):
        """Make accidental fallback imports immediately visible."""
        pytest.fail(f"Unexpected import: {name}")

    monkeypatch.setattr(metric_registry.importlib, "import_module", unexpected)
    with pytest.raises(ValueError, match="No runtime module"):
        metric_registry.register_metric_for_config(MetricRegistry(), "unknown_metric")
