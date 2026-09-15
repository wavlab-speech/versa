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


def test_stoi_discovery_and_runtime_registration_in_fresh_process():
    """Discover and load both STOI variants and aliases without unrelated backends."""
    code = """
import sys
from importlib.abc import MetaPathFinder
from types import ModuleType

class BlockOtherBackends(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('versa.utterance_metrics.', 'versa.corpus_metrics.', 'versa.sequence_metrics.')) and fullname != 'versa.utterance_metrics.stoi':
            raise AssertionError('Unexpected backend import: ' + fullname)

sys.meta_path.insert(0, BlockOtherBackends())
from versa.metric_discovery import create_metric_discovery_registry
from versa.metric_registry import _metric_specs_by_name

names = {
    'stoi': ('stoi', False), 'STOI': ('stoi', False),
    'stoi_metric': ('stoi', False), 'estoi': ('estoi', True),
    'ESTOI': ('estoi', True), 'estoi_metric': ('estoi', True),
}
registry = create_metric_discovery_registry()
for name, (canonical, extended) in names.items():
    metadata = registry.get_metadata(name)
    assert metadata is not None, name
    assert metadata.name == canonical
    assert metadata.requires_reference
    assert metadata.dependencies == ['pystoi', 'numpy']
    assert _metric_specs_by_name()[name].module_name == 'versa.utterance_metrics.stoi'
assert 'versa.utterance_metrics.stoi' not in sys.modules
assert 'pystoi' not in sys.modules

# Substitute only the numerical backend; exercise real registration and scorer loading.
backend = ModuleType('pystoi')
calls = []
def stoi(reference, prediction, sample_rate, extended=False):
    calls.append((sample_rate, extended, len(reference), len(prediction)))
    return 0.75
backend.stoi = stoi
sys.modules['pystoi'] = backend
from versa.scorer_shared import VersaScorer
for name, (canonical, extended) in names.items():
    scorer = VersaScorer(create_metric_discovery_registry())
    suite = scorer.load_metrics([{'name': name}], use_gt=True)
    assert name in suite.metrics, name
    metric = suite.metrics[name]
    assert metric.get_metadata() == scorer.registry.get_metadata(name)
    result = metric.compute([1., 2., 3.], [1., 2.], {'sample_rate': 8000})
    assert result == {canonical: 0.75}
    assert calls[-1] == (8000, extended, 2, 2)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "expression, expected_name",
    [
        ('_stoi_metadata("stoi", False)', "stoi"),
        ('_stoi_metadata(name="estoi", extended=True)', "estoi"),
        ("_stoi_metadata(self.output_key, self.extended)", None),
        ('_stoi_metadata("stoi", extended=choose_mode())', None),
        ('_stoi_metadata("stoi", unknown=False)', None),
        ("_stoi_metadata(**options)", None),
    ],
)
def test_stoi_metadata_helper_accepts_only_literal_arguments(expression, expected_name):
    """Handle positional and keyword literals without evaluating runtime expressions."""
    import ast

    from versa.metric_discovery import _metadata_from_known_helper

    metadata = _metadata_from_known_helper(ast.parse(expression, mode="eval").body)
    assert (metadata.name if metadata else None) == expected_name
