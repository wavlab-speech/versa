"""Load, validate, and query the packaged VERSA Prompt Bank.

Protocol data is read through :mod:`importlib.resources` so the bank works from
a source checkout and from an installed wheel alike. Loading never imports a
model backend or a VERSA metric module.
"""

import difflib
from collections.abc import Hashable
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Tuple

import yaml

from versa.prompt_bank.schema import (
    BANK_SCHEMA_VERSION,
    BankValidationError,
    DEFAULT_LISTED_STATUSES,
    ErrorCollector,
    PROTOCOL_STATUSES,
    Protocol,
    ProtocolNotFoundError,
    parse_protocol,
)

DATA_PACKAGE = "versa.prompt_bank.data"
MANIFEST_NAME = "manifest.yaml"

_CACHED_BANK = None


class StrictLoader(yaml.SafeLoader):
    """A safe YAML loader that rejects duplicate mapping keys.

    Protocol YAML defines executable evaluation behavior, so a repeated key must
    fail loudly instead of silently discarding the earlier definition.
    """


def _construct_unique_mapping(loader, node, deep=False):
    """Build a mapping, raising on any key that the node declares twice."""
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, Hashable):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found unhashable key",
                key_node.start_mark,
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found duplicate key {!r}".format(key),
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


StrictLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def strict_load(text):
    """Parse one YAML document, rejecting duplicate mapping keys."""
    return yaml.load(text, Loader=StrictLoader)


@dataclass(frozen=True)
class PromptBank:
    """An immutable, validated view of every bundled protocol."""

    schema_version: int
    protocols: Tuple[Protocol, ...]
    by_id: Mapping[str, Protocol]
    sources: Tuple[str, ...]

    def get(self, protocol_id):
        """Return one protocol by exact ID, suggesting near matches otherwise."""
        try:
            return self.by_id[protocol_id]
        except KeyError:
            suggestions = difflib.get_close_matches(
                str(protocol_id), sorted(self.by_id), n=3
            )
            hint = " Did you mean: {}?".format(", ".join(suggestions))
            raise ProtocolNotFoundError(
                "unknown protocol {!r}.{}".format(
                    protocol_id, hint if suggestions else ""
                )
            )


def _resource_text(name):
    """Return one packaged data file as text, supporting Python 3.8 upward."""
    try:
        from importlib.resources import files
    except ImportError:  # Python 3.8
        from importlib.resources import read_text

        return read_text(DATA_PACKAGE, name, encoding="utf-8")
    return files(DATA_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def _resource_names():
    """Return the names of the packaged data files, sorted for determinism."""
    try:
        from importlib.resources import files
    except ImportError:  # Python 3.8
        from importlib.resources import contents

        return sorted(contents(DATA_PACKAGE))
    return sorted(entry.name for entry in files(DATA_PACKAGE).iterdir())


def _read_documents():
    """Return ``(name, text)`` pairs for the manifest and every indexed file."""
    documents = [(MANIFEST_NAME, _resource_text(MANIFEST_NAME))]
    present = {name for name in _resource_names() if name.endswith(".yaml")}
    errors = ErrorCollector()
    try:
        manifest = strict_load(documents[0][1])
    except yaml.YAMLError as error:
        raise BankValidationError(
            ["{}: is not valid YAML: {}".format(MANIFEST_NAME, error)]
        )
    for name in _manifest_files(manifest, present, errors):
        documents.append((name, _resource_text(name)))
    errors.raise_if_any()
    return documents


def _manifest_files(manifest, present, errors):
    """Validate the manifest against the packaged files and return its index."""
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "files"}:
        errors.add(MANIFEST_NAME, None, "manifest needs schema_version and files")
        return []
    if manifest["schema_version"] != BANK_SCHEMA_VERSION:
        errors.add(
            MANIFEST_NAME,
            None,
            "manifest schema_version must be {}".format(BANK_SCHEMA_VERSION),
        )
    names = manifest["files"]
    if (
        not isinstance(names, list)
        or not names
        or any(not isinstance(name, str) for name in names)
    ):
        errors.add(MANIFEST_NAME, None, "files must be a non-empty list of names")
        return []
    if len(set(names)) != len(names):
        errors.add(MANIFEST_NAME, None, "files must not repeat an entry")
        return []
    indexed = set(names)
    if MANIFEST_NAME in indexed:
        errors.add(MANIFEST_NAME, None, "the manifest must not index itself")
        return []
    missing = sorted(indexed - present)
    if missing:
        errors.add(MANIFEST_NAME, None, "indexed files are missing: {}".format(missing))
    unindexed = sorted(present - indexed - {MANIFEST_NAME})
    if unindexed:
        errors.add(
            MANIFEST_NAME, None, "protocol files are not indexed: {}".format(unindexed)
        )
    return [] if len(errors) else names


def build_bank(documents):
    """Validate ``(name, text)`` YAML documents and return a :class:`PromptBank`.

    The first document is the manifest, which has already been checked against
    the available files. Every remaining document holds protocol records.
    """
    errors = ErrorCollector()
    protocols = []
    sources = []
    for name, text in documents[1:]:
        sources.append(name)
        document = _parse_document(name, text, errors)
        if document is None:
            continue
        for entry in document:
            protocol = parse_protocol(entry, name, errors)
            if protocol is not None:
                protocols.append(protocol)
    _check_uniqueness(protocols, errors)
    errors.raise_if_any()
    protocols = tuple(sorted(protocols, key=lambda protocol: protocol.id))
    return PromptBank(
        schema_version=BANK_SCHEMA_VERSION,
        protocols=protocols,
        by_id=MappingProxyType({protocol.id: protocol for protocol in protocols}),
        sources=tuple(sources),
    )


def _parse_document(name, text, errors):
    """Return the protocol list of one YAML file, or ``None`` when malformed."""
    try:
        document = strict_load(text)
    except yaml.YAMLError as error:
        errors.add(name, None, "is not valid YAML: {}".format(error))
        return None
    if not isinstance(document, dict) or set(document) != {
        "schema_version",
        "protocols",
    }:
        errors.add(name, None, "needs exactly schema_version and protocols")
        return None
    if document["schema_version"] != BANK_SCHEMA_VERSION:
        errors.add(name, None, "schema_version must be {}".format(BANK_SCHEMA_VERSION))
        return None
    protocols = document["protocols"]
    if not isinstance(protocols, list) or not protocols:
        errors.add(name, None, "protocols must be a non-empty list")
        return None
    return protocols


def _check_uniqueness(protocols, errors):
    """Reject duplicate protocol IDs and duplicate declared output keys."""
    seen_ids = {}
    seen_keys = {}
    for protocol in protocols:
        first = seen_ids.get(protocol.id)
        if first is not None:
            errors.add(
                protocol.source_file,
                protocol.id,
                "duplicates the ID already defined in {}".format(first),
            )
        else:
            seen_ids[protocol.id] = protocol.source_file
        output_key = protocol.response_contract.output_key
        if output_key is None:
            continue
        owner = seen_keys.get(output_key)
        if owner is not None:
            errors.add(
                protocol.source_file,
                protocol.id,
                "output_key {!r} is already declared by {}".format(output_key, owner),
            )
        else:
            seen_keys[output_key] = protocol.id


def load_bank(force_reload=False):
    """Return the cached bundled bank, loading and validating it on first use."""
    global _CACHED_BANK
    if _CACHED_BANK is None or force_reload:
        _CACHED_BANK = build_bank(_read_documents())
    return _CACHED_BANK


def validate_bank():
    """Validate every bundled record, reporting all errors in one exception."""
    return load_bank(force_reload=True)


def get_protocol(protocol_id):
    """Return one immutable protocol record by its exact versioned ID."""
    return load_bank().get(protocol_id)


def list_protocols(
    domain=None,
    task=None,
    audio_inputs=None,
    response_mode=None,
    model_family=None,
    runner=None,
    status=None,
):
    """Return matching protocols sorted by ID, without importing a backend.

    The default status view lists ``experimental`` and ``stable`` records;
    ``draft`` and ``deprecated`` records are returned only when ``status`` names
    them explicitly.
    """
    statuses = (
        DEFAULT_LISTED_STATUSES
        if status is None
        else frozenset({status} if isinstance(status, str) else status)
    )
    unknown = sorted(statuses - set(PROTOCOL_STATUSES))
    if unknown:
        raise ValueError(
            "unknown lifecycle status {}; known statuses are {}".format(
                unknown, list(PROTOCOL_STATUSES)
            )
        )
    selected = []
    for protocol in load_bank().protocols:
        if protocol.status not in statuses:
            continue
        if domain is not None and protocol.domain != domain:
            continue
        if task is not None and protocol.task != task:
            continue
        if (
            audio_inputs is not None
            and protocol.input_contract.audio_inputs != audio_inputs
        ):
            continue
        if (
            response_mode is not None
            and protocol.response_contract.mode != response_mode
        ):
            continue
        if model_family is not None and not any(
            entry.family == model_family and entry.status != "unsupported"
            for entry in protocol.model_compatibility
        ):
            continue
        if runner is not None and not protocol.supports_runner(runner):
            continue
        selected.append(protocol)
    return tuple(selected)
