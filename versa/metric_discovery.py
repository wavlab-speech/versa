#!/usr/bin/env python3

# Copyright 2026 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Helpers for discovering metrics from the command line."""

import ast
import contextlib
import io
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml

from versa.definition import MetricCategory, MetricRegistry, MetricType
from versa.metric_metadata import (
    _qwen2_audio_metadata,
    _qwen2_audio_aliases,
    _qwen_omni_metadata,
    _qwen_omni_aliases,
    _squim_metadata,
    _scoreq_metadata,
    _stoi_metadata,
)

_MCD_F0_DEFAULTS = {
    "name": "mcd_f0",
    "f0min": 40,
    "f0max": 800,
    "mcep_shift": 5,
    "mcep_fftl": 1024,
    "mcep_dim": 39,
    "mcep_alpha": 0.466,
    "seq_mismatch_tolerance": 0.1,
    "power_threshold": -20,
    "dtw": True,
}

_PSEUDO_MOS_UTMOS = {
    "name": "pseudo_mos",
    "predictor_types": ["utmos"],
    "predictor_args": {"utmos": {"fs": 16000}},
}

_RECOMMENDED_CONFIGS: Dict[str, Dict[str, List[Dict[str, Any]]]] = {
    "tts": {
        "cpu": [
            _MCD_F0_DEFAULTS,
            {"name": "pesq"},
            {"name": "stoi"},
            _PSEUDO_MOS_UTMOS,
            {"name": "speaking_rate"},
        ],
        "gpu": [
            _MCD_F0_DEFAULTS,
            {"name": "discrete_speech"},
            _PSEUDO_MOS_UTMOS,
            {"name": "whisper_wer", "model_tag": "default", "beam_size": 1},
            {"name": "speaker", "model_tag": "default"},
            {"name": "asr_match", "model_tag": "default", "beam_size": 1},
            {"name": "audiobox_aesthetics", "batch_size": 1},
        ],
    },
    "codec": {
        "cpu": [
            _MCD_F0_DEFAULTS,
            {"name": "signal_metric"},
            {"name": "pesq"},
            {"name": "stoi"},
            _PSEUDO_MOS_UTMOS,
        ],
        "gpu": [
            _MCD_F0_DEFAULTS,
            {"name": "signal_metric"},
            {"name": "pesq"},
            {"name": "stoi"},
            {"name": "discrete_speech"},
            _PSEUDO_MOS_UTMOS,
            {"name": "speaker", "model_tag": "default"},
        ],
    },
    "se": {
        "cpu": [
            {"name": "signal_metric"},
            {"name": "pesq"},
            {"name": "stoi"},
            {"name": "srmr"},
            {"name": "pysepm"},
        ],
        "gpu": [
            {"name": "signal_metric"},
            {"name": "pesq"},
            {"name": "stoi"},
            {"name": "squim_ref"},
            {"name": "squim_no_ref"},
            {"name": "se_snr"},
        ],
    },
    "svs": {
        "cpu": [
            _MCD_F0_DEFAULTS,
            _PSEUDO_MOS_UTMOS,
            {"name": "speaker", "model_tag": "default"},
        ],
        "gpu": [
            _MCD_F0_DEFAULTS,
            _PSEUDO_MOS_UTMOS,
            {"name": "singer"},
            {"name": "speaker", "model_tag": "default"},
            {"name": "qwen_omni_singing_technique"},
        ],
    },
}

_TASK_ALIASES = {
    "speech-enhancement": "se",
    "speech_enhancement": "se",
    "enhancement": "se",
    "singing": "svs",
}


def create_metric_discovery_registry(
    *, include_runtime_imports: bool = False
) -> MetricRegistry:
    """Create the registry used by discovery commands."""
    if include_runtime_imports:
        with _quiet_metric_imports():
            from versa.metric_registry import create_populated_registry

            registry = create_populated_registry(logger=logging.getLogger(__name__))
    else:
        registry = MetricRegistry()
    _add_source_discovered_metrics(registry)
    return registry


def parse_metric_category(value: Optional[str]) -> Optional[MetricCategory]:
    """Parse a category filter from the CLI."""
    if value is None:
        return None
    try:
        return MetricCategory(value)
    except ValueError as exc:
        choices = ", ".join(category.value for category in MetricCategory)
        raise ValueError(
            f"Unknown metric category '{value}'. Choices: {choices}"
        ) from exc


def parse_metric_type(value: Optional[str]) -> Optional[MetricType]:
    """Parse a metric type filter from the CLI."""
    if value is None:
        return None
    try:
        return MetricType(value)
    except ValueError as exc:
        choices = ", ".join(metric_type.value for metric_type in MetricType)
        raise ValueError(f"Unknown metric type '{value}'. Choices: {choices}") from exc


def format_metric_list(
    registry: MetricRegistry,
    category: Optional[MetricCategory] = None,
    metric_type: Optional[MetricType] = None,
) -> str:
    """Format registered metrics as a compact table."""
    metrics = registry.list_metrics(category=category, metric_type=metric_type)
    rows = []
    for name in metrics:
        metadata = registry.get_metadata(name)
        rows.append(
            [
                metadata.name,
                metadata.category.value,
                metadata.metric_type.value,
                "yes" if metadata.requires_reference else "no",
                "yes" if metadata.requires_text else "no",
                "yes" if metadata.gpu_compatible else "no",
                "yes" if metadata.auto_install else "no",
            ]
        )

    headers = ["name", "category", "type", "ref", "text", "gpu", "auto"]
    table = _format_table(headers, rows)
    return f"{table}\n\n{len(rows)} metric(s) available."


def describe_metric(registry: MetricRegistry, metric_name: str) -> str:
    """Describe one registered metric."""
    metadata = registry.get_metadata(metric_name)
    if metadata is None:
        suggestions = _suggest_metric_names(registry, metric_name)
        message = f"Metric '{metric_name}' was not found."
        if suggestions:
            message += "\nDid you mean: " + ", ".join(suggestions) + "?"
        raise ValueError(message)

    aliases = [
        alias for alias in registry.get_aliases(metric_name) if alias != metadata.name
    ]
    lines = [
        f"name: {metadata.name}",
        f"category: {metadata.category.value}",
        f"type: {metadata.metric_type.value}",
        f"requires_reference: {str(metadata.requires_reference).lower()}",
        "requires_multiple_sources: " + str(metadata.requires_multiple_sources).lower(),
        f"requires_text: {str(metadata.requires_text).lower()}",
        f"gpu_compatible: {str(metadata.gpu_compatible).lower()}",
        f"auto_install: {str(metadata.auto_install).lower()}",
        "dependencies: " + (", ".join(metadata.dependencies) or "none"),
    ]
    if aliases:
        lines.append("aliases: " + ", ".join(aliases))
    lines.append(f"description: {metadata.description}")
    if metadata.paper_reference:
        lines.append(f"paper_reference: {metadata.paper_reference}")
    if metadata.implementation_source:
        lines.append(f"implementation_source: {metadata.implementation_source}")

    return "\n".join(lines)


def recommend_config(task: str, device: str) -> str:
    """Return a recommended score config as YAML."""
    task_key = _TASK_ALIASES.get(task.lower(), task.lower())
    device_key = device.lower()

    if task_key not in _RECOMMENDED_CONFIGS:
        choices = ", ".join(sorted(_RECOMMENDED_CONFIGS))
        raise ValueError(f"Unknown task '{task}'. Choices: {choices}")
    if device_key not in _RECOMMENDED_CONFIGS[task_key]:
        choices = ", ".join(sorted(_RECOMMENDED_CONFIGS[task_key]))
        raise ValueError(f"Unknown device '{device}'. Choices: {choices}")

    config = _RECOMMENDED_CONFIGS[task_key][device_key]
    header = [
        f"# Recommended VERSA score config for task={task_key}, device={device_key}",
        "# Save this YAML and pass it with --score_config.",
    ]
    body = yaml.safe_dump(config, sort_keys=False)
    return "\n".join(header) + "\n" + body


def supported_recommendation_tasks() -> Iterable[str]:
    """Return tasks with built-in recommendations."""
    return sorted(_RECOMMENDED_CONFIGS)


@contextlib.contextmanager
def _quiet_metric_imports():
    """Temporarily suppress logging and stderr during optional metric imports."""
    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            yield
        finally:
            logging.disable(previous_disable_level)


def _format_table(headers: List[str], rows: List[List[str]]) -> str:
    """Render headers and rows as a left-aligned table with computed column widths."""
    widths = [
        max(len(str(row[index])) for row in [headers] + rows)
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(
                str(value).ljust(widths[index]) for index, value in enumerate(row)
            )
        )
    return "\n".join(lines)


def _add_source_discovered_metrics(registry: MetricRegistry) -> None:
    """Register source-derived placeholders for metrics not already registered."""
    package_root = Path(__file__).resolve().parent
    for path in package_root.rglob("*.py"):
        if path.name.startswith("__"):
            continue
        for metadata, aliases in _discover_module_metadata(path):
            if registry.get_metadata(metadata.name):
                continue
            registry.register(_DiscoveredMetric, metadata, aliases=aliases)


def _discover_module_metadata(path: Path) -> List[Tuple[Any, List[str]]]:
    """Extract metadata and aliases from Python source without importing it.

    Recognize literal constructors, local metadata helpers, and supported
    prompt families. Unparseable source yields an empty list."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    assigned_metadata = {}
    function_metadata = {}
    discovered = _discover_prompt_metrics(path, tree)

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                metadata = _metadata_from_call(child.value)
                if metadata is not None:
                    function_metadata[node.name] = metadata
                    discovered.append((metadata, []))
                    break

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        metadata = _metadata_from_call(node.value)
        if metadata is None:
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                assigned_metadata[target.id] = metadata
        discovered.append((metadata, []))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not _is_register_call(node.func):
            continue
        metadata = _metadata_from_register_call(
            node, assigned_metadata, function_metadata
        )
        if metadata is None:
            continue
        aliases = _aliases_from_register_call(node)
        discovered.append((metadata, aliases))

    unique = {}
    for metadata, aliases in discovered:
        unique[metadata.name] = (metadata, aliases)
    return list(unique.values())


def _discover_prompt_metrics(path: Path, tree: ast.AST) -> List[Tuple[Any, List[str]]]:
    """Expand Qwen prompt names into metadata and aliases using source ASTs."""
    prompt_names = _default_prompt_names(tree)
    if not prompt_names and path.name == "qwen_omni.py":
        prompt_names = _default_prompt_names(
            ast.parse((path.parent / "qwen2_audio.py").read_text(encoding="utf-8"))
        )
    if not prompt_names:
        return []

    families = {
        "qwen2_audio.py": ("qwen2_audio", _qwen2_audio_metadata, _qwen2_audio_aliases),
        "qwen_omni.py": ("qwen_omni", _qwen_omni_metadata, _qwen_omni_aliases),
    }
    if path.name not in families:
        return []
    prefix, metadata, aliases = families[path.name]
    return [(metadata(f"{prefix}_{name}"), aliases(name)) for name in prompt_names]


def _default_prompt_names(tree: ast.AST) -> List[str]:
    """Read literal string keys from the source assignment to DEFAULT_PROMPTS."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "DEFAULT_PROMPTS"
            for target in node.targets
        ):
            continue
        if isinstance(node.value, ast.Dict):
            return [
                key.value
                for key in node.value.keys
                if isinstance(key, ast.Constant) and isinstance(key.value, str)
            ]
    return []


def _metadata_from_register_call(
    node: ast.Call,
    assigned_metadata: Dict[str, Any],
    function_metadata: Dict[str, Any],
) -> Optional[Any]:
    """Resolve the metadata argument of a registry call from known source values."""
    if len(node.args) < 2:
        return None
    metadata_arg = node.args[1]
    if isinstance(metadata_arg, ast.Name):
        return assigned_metadata.get(metadata_arg.id)
    if isinstance(metadata_arg, ast.Call) and isinstance(metadata_arg.func, ast.Name):
        return function_metadata.get(
            metadata_arg.func.id
        ) or _metadata_from_known_helper(metadata_arg)
    return _metadata_from_call(metadata_arg)


def _metadata_from_known_helper(node: ast.Call) -> Optional[Any]:
    """Evaluate only the supported dependency-light metadata helpers."""
    if not isinstance(node.func, ast.Name):
        return None
    helpers = {
        "_squim_metadata": _squim_metadata,
        "_scoreq_metadata": _scoreq_metadata,
        "_stoi_metadata": _stoi_metadata,
    }
    helper = helpers.get(node.func.id)
    if helper is None:
        return None
    try:
        args = [ast.literal_eval(arg) for arg in node.args]
        kwargs = {
            keyword.arg: ast.literal_eval(keyword.value) for keyword in node.keywords
        }
        if None in kwargs:  # Do not evaluate expanded keyword dictionaries.
            return None
        return helper(*args, **kwargs)
    except (TypeError, ValueError):
        return None


def _metadata_from_call(node: ast.AST) -> Optional[Any]:
    """Build metadata from literal constructor arguments, or return None."""
    from versa.definition import MetricMetadata

    if not isinstance(node, ast.Call) or not _is_name(node.func, "MetricMetadata"):
        return None

    values = {}
    positional_fields = [
        "name",
        "category",
        "metric_type",
        "requires_reference",
        "requires_text",
        "gpu_compatible",
        "auto_install",
        "dependencies",
        "description",
        "paper_reference",
        "implementation_source",
        "requires_multiple_sources",
    ]
    for field, value in zip(positional_fields, node.args):
        values[field] = _literal_metric_value(value)
    for keyword in node.keywords:
        if keyword.arg:
            values[keyword.arg] = _literal_metric_value(keyword.value)

    required_fields = positional_fields[:9]
    if any(field not in values for field in required_fields):
        return None
    if any(values[field] is None for field in required_fields):
        return None

    try:
        return MetricMetadata(**values)
    except (TypeError, ValueError):
        return None


def _literal_metric_value(node: ast.AST) -> Any:
    """Decode literal AST values and metric enums without executing source code."""
    from versa.definition import MetricCategory, MetricType

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        if node.value.id == "MetricCategory":
            return MetricCategory[node.attr]
        if node.value.id == "MetricType":
            return MetricType[node.attr]
    if isinstance(node, ast.List):
        return [_literal_metric_value(item) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_metric_value(item) for item in node.elts)
    if isinstance(node, ast.Dict):
        return {
            _literal_metric_value(key): _literal_metric_value(value)
            for key, value in zip(node.keys, node.values)
        }
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _aliases_from_register_call(node: ast.Call) -> List[str]:
    """Extract positional or keyword aliases, returning an empty list if unknown."""
    alias_node = None
    if len(node.args) >= 3:
        alias_node = node.args[2]
    for keyword in node.keywords:
        if keyword.arg == "aliases":
            alias_node = keyword.value
            break
    aliases = _literal_metric_value(alias_node) if alias_node is not None else []
    return aliases if isinstance(aliases, list) else []


def _is_register_call(node: ast.AST) -> bool:
    """Recognize an attribute access named ``register`` in the source AST."""
    return isinstance(node, ast.Attribute) and node.attr == "register"


def _is_name(node: ast.AST, name: str) -> bool:
    """Check whether an AST node is a simple reference to the specified name."""
    return isinstance(node, ast.Name) and node.id == name


class _DiscoveredMetric:
    """Placeholder metric class for metadata-only discovery."""


def _suggest_metric_names(registry: MetricRegistry, metric_name: str) -> List[str]:
    """Return up to five sorted names or aliases containing the query, ignoring case."""
    query = metric_name.lower()
    names = set(registry.list_metrics()) | set(registry.list_aliases())
    return sorted(name for name in names if query in name.lower())[:5]
