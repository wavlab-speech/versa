"""Typed records, validation, and digests for VERSA Prompt Bank protocols.

The module imports only the standard library. Protocol records can therefore be
parsed and validated without importing a model backend, an audio stack, or any
VERSA metric module.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Optional, Tuple

BANK_SCHEMA_VERSION = 1

PROTOCOL_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+\.v[1-9][0-9]*$")
PLACEHOLDER_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_PLACEHOLDER_TOKEN_PATTERN = re.compile(r"\{\{|\}\}|\{[^{}]*\}")

PLACEHOLDER_ALLOWLIST = frozenset(
    {
        "labels",
        "target_instruction",
        "reference_text",
        "candidate_a_name",
        "candidate_b_name",
        "rubric_items",
    }
)
# Supplied by the renderer from the response contract, never from user context.
RENDERER_PLACEHOLDERS = frozenset({"labels"})
# Placeholders whose availability is declared by the input contract.
INPUT_PLACEHOLDERS = {
    "target_instruction": "requires_target_instruction",
    "reference_text": "requires_reference_text",
}
# Optional rendering context with deterministic defaults.
OPTIONAL_PLACEHOLDER_DEFAULTS = {
    "candidate_a_name": "Candidate A",
    "candidate_b_name": "Candidate B",
}
# Placeholders whose context value is a sequence of short strings.
SEQUENCE_PLACEHOLDERS = frozenset({"rubric_items"})

PROTOCOL_STATUSES = ("draft", "experimental", "stable", "deprecated")
DEFAULT_LISTED_STATUSES = frozenset({"experimental", "stable"})
DOMAINS = ("speech", "general_audio", "music", "interaction")
RESPONSE_MODES = ("closed_label", "integer", "scalar", "json", "pairwise")
RENDER_MODES = ("zero_shot", "few_shot_text", "pairwise")
MODEL_STATUSES = ("expected", "tested", "unsupported")
RUNNER_STATUSES = ("planned", "supported", "unsupported")
METRIC_LINK_ROLES = ("companion", "diagnostic", "calibration")
JSON_FIELD_TYPES = ("string", "integer", "number", "boolean", "array")
NUMERIC_JSON_TYPES = ("integer", "number")
MIN_FEW_SHOT_EXAMPLES = 2


class PromptBankError(Exception):
    """Base class for every Prompt Bank failure."""


class ProtocolNotFoundError(PromptBankError, KeyError):
    """Raised when a protocol ID is not present in the bank."""


class BankValidationError(PromptBankError):
    """Raised with every validation message collected from the bundled bank."""

    def __init__(self, messages):
        """Store the ordered validation messages and build a single summary."""
        self.messages = tuple(messages)
        count = len(self.messages)
        joined = "\n".join("  - " + message for message in self.messages)
        super().__init__(
            "prompt bank validation failed with {} error(s):\n{}".format(count, joined)
        )


class RenderError(PromptBankError):
    """Raised when a protocol cannot be rendered with the requested mode/context."""


class ErrorCollector:
    """Accumulate validation messages labelled with source file and protocol ID."""

    def __init__(self):
        """Start with no recorded messages."""
        self.messages = []

    def add(self, source, protocol_id, message):
        """Record one message for a source file and optional protocol ID."""
        location = (
            source if protocol_id is None else "{}[{}]".format(source, protocol_id)
        )
        self.messages.append("{}: {}".format(location, message))

    def __len__(self):
        """Return the number of recorded messages."""
        return len(self.messages)

    def raise_if_any(self):
        """Raise :class:`BankValidationError` when any message was recorded."""
        if self.messages:
            raise BankValidationError(self.messages)


@dataclass(frozen=True)
class InputContract:
    """Audio and text inputs a protocol requires from the calling runner."""

    audio_inputs: int
    requires_target_instruction: bool = False
    requires_reference_audio: bool = False
    requires_reference_text: bool = False

    def required_context_keys(self):
        """Return the text context keys a caller must supply, sorted by name."""
        return tuple(
            sorted(
                name
                for name, attribute in INPUT_PLACEHOLDERS.items()
                if getattr(self, attribute)
            )
        )


@dataclass(frozen=True)
class JsonField:
    """One typed field of a structured JSON response contract."""

    name: str
    type: str
    required: bool = True
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    items: Optional[str] = None


@dataclass(frozen=True)
class ScalarAnchor:
    """One labelled anchor point of a scalar rating scale."""

    value: float
    description: str


@dataclass(frozen=True)
class ResponseContract:
    """The accepted response shape, including abstention and reportable fields."""

    mode: str
    labels: Tuple[str, ...] = ()
    preference_labels: Tuple[str, ...] = ()
    allow_abstain: bool = False
    abstain_label: Optional[str] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    anchors: Tuple[ScalarAnchor, ...] = ()
    fields: Tuple[JsonField, ...] = ()
    primary_numeric_field: Optional[str] = None
    output_key: Optional[str] = None

    def choice_labels(self):
        """Return the closed or pairwise labels a response may use."""
        return self.preference_labels if self.mode == "pairwise" else self.labels


@dataclass(frozen=True)
class FewShotExample:
    """One text-only demonstration of audible evidence and its expected output."""

    input: str
    output: str


@dataclass(frozen=True)
class FewShotSpec:
    """Text-only demonstrations and the target instruction rendered after them."""

    examples: Tuple[FewShotExample, ...]
    template: str


@dataclass(frozen=True)
class ProtocolModes:
    """Mode bodies a protocol provides; ``zero_shot`` is always present."""

    zero_shot: str
    few_shot_text: Optional[FewShotSpec] = None
    pairwise: Optional[str] = None

    def available(self):
        """Return the render modes this protocol defines, in canonical order."""
        modes = ["zero_shot"]
        if self.few_shot_text is not None:
            modes.append("few_shot_text")
        if self.pairwise is not None:
            modes.append("pairwise")
        return tuple(modes)


@dataclass(frozen=True)
class ModelCompatibility:
    """A model family's expected or verified behavior for one protocol."""

    family: str
    status: str
    model_id: Optional[str] = None
    revision: Optional[str] = None
    validation: Optional[str] = None
    rendering_notes: Optional[str] = None


@dataclass(frozen=True)
class RunnerCompatibility:
    """Whether a VERSA runner can supply this protocol's inputs and modes."""

    runner: str
    status: str
    modes: Tuple[str, ...] = ()


@dataclass(frozen=True)
class MetricLink:
    """A traditional VERSA metric linked to a protocol for interpretation."""

    metric: str
    role: str


@dataclass(frozen=True)
class Provenance:
    """Where a protocol came from and why it is written the way it is."""

    sources: Tuple[str, ...] = ()
    rationale: Optional[str] = None


@dataclass(frozen=True)
class Protocol:
    """One immutable, versioned measurement protocol from the Prompt Bank."""

    id: str
    version: int
    title: str
    status: str
    domain: str
    task: str
    description: str
    input_contract: InputContract
    response_contract: ResponseContract
    modes: ProtocolModes
    provenance: Provenance = Provenance()
    model_compatibility: Tuple[ModelCompatibility, ...] = ()
    runner_compatibility: Tuple[RunnerCompatibility, ...] = ()
    metric_links: Tuple[MetricLink, ...] = ()
    source_file: str = ""
    digest: str = ""

    def available_modes(self):
        """Return the render modes this protocol defines, in canonical order."""
        return self.modes.available()

    def supports_runner(self, runner, mode=None):
        """Report whether a runner is declared supported, optionally for one mode."""
        for entry in self.runner_compatibility:
            if entry.runner != runner:
                continue
            if entry.status != "supported":
                return False
            return mode is None or mode in entry.modes
        return False


class DuplicateJsonKeyError(ValueError):
    """Raised when a JSON response repeats a key instead of declaring it once."""


class NonFiniteJsonError(ValueError):
    """Raised when a JSON response carries NaN or Infinity."""


def is_finite_number(value):
    """Report whether a value is a real, finite number and not a boolean."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _reject_json_constant(name):
    """Reject the JSON extensions NaN, Infinity, and -Infinity."""
    raise NonFiniteJsonError("must not contain {}".format(name))


def _reject_duplicate_keys(pairs):
    """Build a JSON object, rejecting a repeated key instead of keeping the last."""
    payload = {}
    for key, value in pairs:
        if key in payload:
            raise DuplicateJsonKeyError("must not repeat the key {!r}".format(key))
        payload[key] = value
    return payload


def loads_strict_json(value):
    """Parse JSON, rejecting non-finite constants and repeated object keys."""
    return json.loads(
        value,
        parse_constant=_reject_json_constant,
        object_pairs_hook=_reject_duplicate_keys,
    )


def scan_placeholders(text):
    """Return (placeholder names, error messages) for one protocol text body.

    Only allow-listed simple names are accepted. Attribute access, indexing,
    conversion flags, and format specifications are rejected, and literal braces
    must be escaped as ``{{`` or ``}}``.
    """
    names = []
    errors = []
    position = 0
    for match in _PLACEHOLDER_TOKEN_PATTERN.finditer(text):
        if _has_bare_brace(text[position : match.start()]):
            errors.append("unescaped brace outside a placeholder")
        position = match.end()
        token = match.group()
        if token in ("{{", "}}"):
            continue
        name = token[1:-1]
        if not PLACEHOLDER_NAME_PATTERN.match(name):
            errors.append(
                "placeholder {!r} must be a simple name without attribute access, "
                "indexing, conversion flags, or format specifications".format(token)
            )
        elif name not in PLACEHOLDER_ALLOWLIST:
            errors.append(
                "placeholder {!r} is not in the allow-list {}".format(
                    name, sorted(PLACEHOLDER_ALLOWLIST)
                )
            )
        else:
            names.append(name)
    if _has_bare_brace(text[position:]):
        errors.append("unescaped brace outside a placeholder")
    return names, errors


def _has_bare_brace(segment):
    """Report whether a text segment contains an unescaped brace character."""
    return "{" in segment or "}" in segment


def substitute(text, values):
    """Replace allow-listed placeholders and unescape literal brace pairs."""

    def replace(match):
        """Return the replacement for one placeholder or escaped brace token."""
        token = match.group()
        if token == "{{":
            return "{"
        if token == "}}":
            return "}"
        return values[token[1:-1]]

    return _PLACEHOLDER_TOKEN_PATTERN.sub(replace, text)


def canonical_json(payload):
    """Serialize a payload deterministically for digesting."""
    return json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )


def compute_digest(payload):
    """Return the SHA-256 digest of the canonical JSON form of a payload."""
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def digest_text(text):
    """Return the SHA-256 digest of an exact rendered prompt string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


_PROTOCOL_KEYS = frozenset(
    {
        "id",
        "version",
        "title",
        "status",
        "domain",
        "task",
        "description",
        "input_contract",
        "response_contract",
        "protocol",
        "provenance",
        "model_compatibility",
        "runner_compatibility",
        "metric_links",
    }
)
_INPUT_CONTRACT_KEYS = frozenset(
    {
        "audio_inputs",
        "requires_target_instruction",
        "requires_reference_audio",
        "requires_reference_text",
    }
)
_RESPONSE_CONTRACT_KEYS = {
    "closed_label": frozenset({"mode", "labels", "allow_abstain", "abstain_label"}),
    "integer": frozenset(
        {"mode", "minimum", "maximum", "allow_abstain", "abstain_label"}
    ),
    "scalar": frozenset(
        {"mode", "minimum", "maximum", "anchors", "allow_abstain", "abstain_label"}
    ),
    "json": frozenset({"mode", "fields", "primary_numeric_field", "output_key"}),
    "pairwise": frozenset(
        {"mode", "preference_labels", "allow_abstain", "abstain_label"}
    ),
}
_JSON_FIELD_KEYS = frozenset({"type", "required", "minimum", "maximum", "items"})
_AUDIO_PATH_PATTERN = re.compile(r"\.(wav|flac|mp3|ogg|opus|m4a)\b", re.IGNORECASE)
_FIELD_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_OUTPUT_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_protocol(mapping, source, errors):
    """Validate one protocol mapping and return a :class:`Protocol` or ``None``.

    Every detected problem is appended to ``errors``; ``None`` is returned when
    the record cannot be represented, so a single run reports all failures.
    """
    if not isinstance(mapping, dict):
        errors.add(source, None, "each protocol entry must be a mapping")
        return None
    protocol_id = mapping.get("id")
    if not isinstance(protocol_id, str) or not PROTOCOL_ID_PATTERN.match(protocol_id):
        errors.add(
            source,
            protocol_id if isinstance(protocol_id, str) else None,
            "id must match {}".format(PROTOCOL_ID_PATTERN.pattern),
        )
        return None
    before = len(errors)
    unknown = sorted(set(mapping) - _PROTOCOL_KEYS)
    if unknown:
        errors.add(source, protocol_id, "unknown protocol fields: {}".format(unknown))

    version = _integer(mapping, "version", source, protocol_id, errors)
    if version is not None and "v{}".format(version) != protocol_id.rsplit(".", 1)[-1]:
        errors.add(
            source,
            protocol_id,
            "version {} does not match the trailing ID version".format(version),
        )
    title = _text(mapping, "title", source, protocol_id, errors)
    description = _text(mapping, "description", source, protocol_id, errors)
    task = _text(mapping, "task", source, protocol_id, errors)
    status = _choice(mapping, "status", PROTOCOL_STATUSES, source, protocol_id, errors)
    domain = _choice(mapping, "domain", DOMAINS, source, protocol_id, errors)
    input_contract = _parse_input_contract(mapping, source, protocol_id, errors)
    response_contract = _parse_response_contract(
        mapping, version, source, protocol_id, errors
    )
    modes = _parse_modes(mapping, response_contract, source, protocol_id, errors)
    if input_contract is not None and response_contract is not None:
        _validate_contract_pair(
            input_contract, response_contract, modes, source, protocol_id, errors
        )
        if modes is not None:
            _validate_placeholders(
                input_contract, response_contract, modes, source, protocol_id, errors
            )
    provenance = _parse_provenance(mapping, source, protocol_id, errors)
    model_compatibility = _parse_model_compatibility(
        mapping, source, protocol_id, errors
    )
    runner_compatibility = _parse_runner_compatibility(
        mapping, modes, source, protocol_id, errors
    )
    metric_links = _parse_metric_links(mapping, source, protocol_id, errors)
    if len(errors) != before:
        return None
    digest = compute_digest(
        digest_payload(
            protocol_id,
            version,
            input_contract,
            response_contract,
            modes,
            model_compatibility,
        )
    )
    return Protocol(
        id=protocol_id,
        version=version,
        title=title,
        status=status,
        domain=domain,
        task=task,
        description=description,
        input_contract=input_contract,
        response_contract=response_contract,
        modes=modes,
        provenance=provenance,
        model_compatibility=model_compatibility,
        runner_compatibility=runner_compatibility,
        metric_links=metric_links,
        source_file=source,
        digest=digest,
    )


def digest_payload(
    protocol_id, version, input_contract, response_contract, modes, model_compatibility
):
    """Build the evaluation-bearing payload used for the protocol digest.

    Descriptions, citations, titles, status, metric links, and runner
    declarations are excluded: they cannot change rendered text, accepted
    inputs, or parsed outputs. Model entries contribute only rendering notes.
    """
    return {
        "id": protocol_id,
        "version": version,
        "input_contract": _asdict(input_contract),
        "response_contract": _asdict(response_contract),
        "modes": _asdict(modes),
        "rendering_notes": [
            {"family": entry.family, "notes": entry.rendering_notes}
            for entry in model_compatibility
            if entry.rendering_notes
        ],
    }


def _asdict(record):
    """Convert a nested frozen record into plain JSON-serializable containers."""
    if isinstance(record, tuple):
        return [_asdict(item) for item in record]
    if hasattr(record, "__dataclass_fields__"):
        return {
            name: _asdict(getattr(record, name)) for name in record.__dataclass_fields__
        }
    return record


def _text(mapping, key, source, protocol_id, errors, required=True):
    """Return a non-empty string field, recording an error when it is invalid."""
    value = mapping.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        errors.add(source, protocol_id, "{} must be a non-empty string".format(key))
        return None
    return value


def _integer(mapping, key, source, protocol_id, errors):
    """Return an integer field, recording an error for missing or bool values."""
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        errors.add(source, protocol_id, "{} must be an integer".format(key))
        return None
    return value


def _choice(mapping, key, allowed, source, protocol_id, errors):
    """Return a field constrained to a fixed set of allowed string values."""
    value = mapping.get(key)
    if value not in allowed:
        errors.add(
            source, protocol_id, "{} must be one of {}".format(key, list(allowed))
        )
        return None
    return value


def _flag(mapping, key, source, protocol_id, errors):
    """Return a boolean field defaulting to False, rejecting non-boolean values."""
    value = mapping.get(key, False)
    if not isinstance(value, bool):
        errors.add(source, protocol_id, "{} must be true or false".format(key))
        return False
    return value


def _string_tuple(mapping, key, source, protocol_id, errors, minimum_items):
    """Return a tuple of unique non-empty strings from a list field."""
    value = mapping.get(key)
    if not isinstance(value, list) or len(value) < minimum_items:
        errors.add(
            source,
            protocol_id,
            "{} must be a list of at least {} entries".format(key, minimum_items),
        )
        return ()
    if any(not isinstance(item, str) or not item.strip() for item in value):
        errors.add(
            source, protocol_id, "{} entries must be non-empty strings".format(key)
        )
        return ()
    if len(set(value)) != len(value):
        errors.add(source, protocol_id, "{} entries must be unique".format(key))
        return ()
    return tuple(value)


def _parse_input_contract(mapping, source, protocol_id, errors):
    """Parse and validate the ``input_contract`` block."""
    block = mapping.get("input_contract")
    if not isinstance(block, dict):
        errors.add(source, protocol_id, "input_contract must be a mapping")
        return None
    unknown = sorted(set(block) - _INPUT_CONTRACT_KEYS)
    if unknown:
        errors.add(
            source, protocol_id, "unknown input_contract fields: {}".format(unknown)
        )
    audio_inputs = block.get("audio_inputs")
    if audio_inputs not in (1, 2) or isinstance(audio_inputs, bool):
        errors.add(source, protocol_id, "input_contract.audio_inputs must be 1 or 2")
        return None
    return InputContract(
        audio_inputs=audio_inputs,
        requires_target_instruction=_flag(
            block, "requires_target_instruction", source, protocol_id, errors
        ),
        requires_reference_audio=_flag(
            block, "requires_reference_audio", source, protocol_id, errors
        ),
        requires_reference_text=_flag(
            block, "requires_reference_text", source, protocol_id, errors
        ),
    )


def _parse_response_contract(mapping, version, source, protocol_id, errors):
    """Parse and validate the ``response_contract`` block for its declared mode."""
    block = mapping.get("response_contract")
    if not isinstance(block, dict):
        errors.add(source, protocol_id, "response_contract must be a mapping")
        return None
    mode = _choice(block, "mode", RESPONSE_MODES, source, protocol_id, errors)
    if mode is None:
        return None
    unknown = sorted(set(block) - _RESPONSE_CONTRACT_KEYS[mode])
    if unknown:
        errors.add(
            source,
            protocol_id,
            "fields {} are not valid for response mode {!r}".format(unknown, mode),
        )
    allow_abstain = _flag(block, "allow_abstain", source, protocol_id, errors)
    abstain_label = _text(
        block, "abstain_label", source, protocol_id, errors, required=allow_abstain
    )
    labels = preference_labels = ()
    minimum = maximum = None
    anchors = fields = ()
    primary_numeric_field = output_key = None
    if mode == "closed_label":
        labels = _string_tuple(block, "labels", source, protocol_id, errors, 2)
    elif mode == "pairwise":
        preference_labels = _string_tuple(
            block, "preference_labels", source, protocol_id, errors, 2
        )
    elif mode in ("integer", "scalar"):
        minimum, maximum = _parse_range(block, mode, source, protocol_id, errors)
        if mode == "scalar":
            anchors = _parse_anchors(
                block, minimum, maximum, source, protocol_id, errors
            )
    else:
        fields = _parse_json_fields(block, source, protocol_id, errors)
        primary_numeric_field, output_key = _parse_primary_field(
            block, fields, version, source, protocol_id, errors
        )
    if allow_abstain and abstain_label is not None:
        if abstain_label in labels or abstain_label in preference_labels:
            errors.add(
                source,
                protocol_id,
                "abstain_label {!r} must differ from the response labels".format(
                    abstain_label
                ),
            )
    elif not allow_abstain and block.get("abstain_label") is not None:
        errors.add(source, protocol_id, "abstain_label requires allow_abstain: true")
    if mode == "json":
        _validate_json_abstention(fields, source, protocol_id, errors)
        allow_abstain = any(
            field.name == "abstain" and field.required for field in fields
        )
        abstain_label = None
    return ResponseContract(
        mode=mode,
        labels=labels,
        preference_labels=preference_labels,
        allow_abstain=allow_abstain,
        abstain_label=abstain_label,
        minimum=minimum,
        maximum=maximum,
        anchors=anchors,
        fields=fields,
        primary_numeric_field=primary_numeric_field,
        output_key=output_key,
    )


def _parse_range(block, mode, source, protocol_id, errors):
    """Return the inclusive numeric range required by integer and scalar modes."""
    bounds = []
    for key in ("minimum", "maximum"):
        value = block.get(key)
        valid = isinstance(value, int) and not isinstance(value, bool)
        if mode == "scalar":
            valid = valid or isinstance(value, float)
        valid = valid and is_finite_number(value)
        if not valid:
            errors.add(
                source,
                protocol_id,
                "response_contract.{} must be a finite {} for mode {!r}".format(
                    key, "number" if mode == "scalar" else "integer", mode
                ),
            )
            bounds.append(None)
        else:
            bounds.append(value)
    minimum, maximum = bounds
    if minimum is not None and maximum is not None and minimum >= maximum:
        errors.add(source, protocol_id, "response_contract.minimum must be < maximum")
    return minimum, maximum


def _parse_anchors(block, minimum, maximum, source, protocol_id, errors):
    """Return the ordered scale anchors a scalar contract must declare."""
    entries = block.get("anchors")
    if not isinstance(entries, list) or len(entries) < 2:
        errors.add(
            source,
            protocol_id,
            "scalar contracts require at least two anchors",
        )
        return ()
    anchors = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"value", "description"}:
            errors.add(
                source, protocol_id, "each anchor needs exactly value and description"
            )
            return ()
        value = entry["value"]
        description = entry["description"]
        if not is_finite_number(value):
            errors.add(source, protocol_id, "anchor value must be a finite number")
            return ()
        if not isinstance(description, str) or not description.strip():
            errors.add(source, protocol_id, "anchor description must be a string")
            return ()
        if minimum is not None and maximum is not None:
            if not minimum <= value <= maximum:
                errors.add(
                    source,
                    protocol_id,
                    "anchor value {} is outside the declared range".format(value),
                )
                return ()
        anchors.append(ScalarAnchor(value=float(value), description=description))
    if len({anchor.value for anchor in anchors}) != len(anchors):
        errors.add(source, protocol_id, "anchor values must be unique")
        return ()
    return tuple(sorted(anchors, key=lambda anchor: anchor.value))


def _parse_json_fields(block, source, protocol_id, errors):
    """Return the ordered typed fields of a structured JSON contract."""
    declared = block.get("fields")
    if not isinstance(declared, dict) or not declared:
        errors.add(source, protocol_id, "json contracts require a non-empty fields map")
        return ()
    fields = []
    for name, spec in declared.items():
        if not isinstance(name, str) or not _FIELD_NAME_PATTERN.match(name):
            errors.add(source, protocol_id, "field name {!r} is not valid".format(name))
            return ()
        if not isinstance(spec, dict):
            errors.add(source, protocol_id, "field {!r} must be a mapping".format(name))
            return ()
        unknown = sorted(set(spec) - _JSON_FIELD_KEYS)
        if unknown:
            errors.add(
                source,
                protocol_id,
                "field {!r} has unknown keys {}".format(name, unknown),
            )
            return ()
        field_type = spec.get("type")
        if field_type not in JSON_FIELD_TYPES:
            errors.add(
                source,
                protocol_id,
                "field {!r} type must be one of {}".format(
                    name, list(JSON_FIELD_TYPES)
                ),
            )
            return ()
        required = spec.get("required", True)
        if not isinstance(required, bool):
            errors.add(
                source, protocol_id, "field {!r} required must be boolean".format(name)
            )
            return ()
        items = spec.get("items")
        if field_type == "array" and items != "string":
            errors.add(
                source,
                protocol_id,
                "array field {!r} must declare items: string".format(name),
            )
            return ()
        if field_type != "array" and items is not None:
            errors.add(
                source, protocol_id, "field {!r} may not declare items".format(name)
            )
            return ()
        bounds = []
        for key in ("minimum", "maximum"):
            value = spec.get(key)
            if value is None:
                bounds.append(None)
                continue
            if field_type not in NUMERIC_JSON_TYPES or isinstance(value, bool):
                errors.add(
                    source,
                    protocol_id,
                    "field {!r} may not declare {}".format(name, key),
                )
                return ()
            if not is_finite_number(value):
                errors.add(
                    source,
                    protocol_id,
                    "field {!r} {} must be a finite number".format(name, key),
                )
                return ()
            bounds.append(value)
        if bounds[0] is not None and bounds[1] is not None and bounds[0] >= bounds[1]:
            errors.add(
                source, protocol_id, "field {!r} minimum must be < maximum".format(name)
            )
            return ()
        fields.append(
            JsonField(
                name=name,
                type=field_type,
                required=required,
                minimum=bounds[0],
                maximum=bounds[1],
                items=items,
            )
        )
    return tuple(fields)


def _parse_primary_field(block, fields, version, source, protocol_id, errors):
    """Validate the optional declared primary numeric field and its output key."""
    primary = block.get("primary_numeric_field")
    output_key = block.get("output_key")
    if primary is None:
        if output_key is not None:
            errors.add(
                source,
                protocol_id,
                "output_key requires an explicit primary_numeric_field",
            )
        return None, None
    if not isinstance(primary, str):
        errors.add(source, protocol_id, "primary_numeric_field must be a string")
        return None, None
    named = {field.name: field for field in fields}
    field = named.get(primary)
    if field is None:
        errors.add(
            source,
            protocol_id,
            "primary_numeric_field {!r} is not declared".format(primary),
        )
        return None, None
    if field.type not in NUMERIC_JSON_TYPES:
        errors.add(source, protocol_id, "primary_numeric_field must be numeric")
        return None, None
    if not field.required or field.minimum is None or field.maximum is None:
        errors.add(
            source,
            protocol_id,
            "primary_numeric_field must be required and declare an explicit range",
        )
        return None, None
    if not isinstance(output_key, str) or not _OUTPUT_KEY_PATTERN.match(output_key):
        errors.add(
            source,
            protocol_id,
            "output_key must be a lowercase identifier when a primary field is declared",
        )
        return None, None
    suffix = "_v{}".format(version)
    if version is not None and not output_key.endswith(suffix):
        errors.add(
            source,
            protocol_id,
            "output_key {!r} must end in {!r} so successive protocol versions can "
            "report side by side".format(output_key, suffix),
        )
        return None, None
    return primary, output_key


def _validate_json_abstention(fields, source, protocol_id, errors):
    """Require a boolean ``abstain`` field in every structured JSON contract."""
    named = {field.name: field for field in fields}
    field = named.get("abstain")
    if field is None or field.type != "boolean" or not field.required:
        errors.add(
            source,
            protocol_id,
            "json contracts require a required boolean 'abstain' field",
        )


def _parse_modes(mapping, response_contract, source, protocol_id, errors):
    """Parse the ``protocol`` block holding one body per supported render mode."""
    block = mapping.get("protocol")
    if not isinstance(block, dict):
        errors.add(source, protocol_id, "protocol must be a mapping of render modes")
        return None
    unknown = sorted(set(block) - set(RENDER_MODES))
    if unknown:
        errors.add(source, protocol_id, "unknown protocol modes: {}".format(unknown))
    zero_shot = _text(block, "zero_shot", source, protocol_id, errors)
    if zero_shot is None:
        return None
    pairwise = _text(block, "pairwise", source, protocol_id, errors, required=False)
    few_shot = None
    if "few_shot_text" in block:
        few_shot = _parse_few_shot(
            block["few_shot_text"], response_contract, source, protocol_id, errors
        )
    return ProtocolModes(zero_shot=zero_shot, few_shot_text=few_shot, pairwise=pairwise)


def _parse_few_shot(block, response_contract, source, protocol_id, errors):
    """Parse text-only demonstrations and the instruction rendered after them."""
    if not isinstance(block, dict) or set(block) != {"examples", "template"}:
        errors.add(
            source, protocol_id, "few_shot_text needs exactly examples and template"
        )
        return None
    template = _text(block, "template", source, protocol_id, errors)
    entries = block.get("examples")
    if not isinstance(entries, list) or len(entries) < MIN_FEW_SHOT_EXAMPLES:
        errors.add(
            source,
            protocol_id,
            "few_shot_text requires at least {} examples".format(MIN_FEW_SHOT_EXAMPLES),
        )
        return None
    examples = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"input", "output"}:
            errors.add(
                source, protocol_id, "each example needs exactly input and output"
            )
            return None
        text = entry["input"]
        output = entry["output"]
        if not isinstance(text, str) or not text.strip():
            errors.add(source, protocol_id, "example input must be a non-empty string")
            return None
        if not isinstance(output, (str, int, float, bool)):
            errors.add(source, protocol_id, "example output must be a scalar value")
            return None
        output = output if isinstance(output, str) else json.dumps(output)
        for value in (text, output):
            if _AUDIO_PATH_PATTERN.search(value):
                errors.add(
                    source,
                    protocol_id,
                    "text-only examples must not reference audio files",
                )
                return None
        if response_contract is not None:
            for message in validate_response(output, response_contract):
                errors.add(
                    source,
                    protocol_id,
                    "example output {!r}: {}".format(output, message),
                )
        examples.append(FewShotExample(input=text, output=output))
    if template is None:
        return None
    return FewShotSpec(examples=tuple(examples), template=template)


def validate_response(text, contract):
    """Return the reasons a response string does not satisfy a response contract.

    An empty list means the response is acceptable. The check is deliberately
    strict: it never extracts a value from surrounding prose.
    """
    value = text.strip()
    if contract.allow_abstain and contract.mode != "json":
        if value == contract.abstain_label:
            return []
    if contract.mode in ("closed_label", "pairwise"):
        allowed = contract.choice_labels()
        if value not in allowed:
            return ["must be exactly one of {}".format(list(allowed))]
        return []
    if contract.mode in ("integer", "scalar") and None in (
        contract.minimum,
        contract.maximum,
    ):
        return ["declares an incomplete numeric range"]
    if contract.mode == "integer":
        try:
            number = int(value)
        except ValueError:
            return ["must be a single integer"]
        if not contract.minimum <= number <= contract.maximum:
            return [
                "must be between {} and {}".format(contract.minimum, contract.maximum)
            ]
        return []
    if contract.mode == "scalar":
        try:
            number = float(value)
        except ValueError:
            return ["must be a single number"]
        if not math.isfinite(number):
            return ["must be a finite number"]
        if not contract.minimum <= number <= contract.maximum:
            return [
                "must be between {} and {}".format(contract.minimum, contract.maximum)
            ]
        return []
    return _validate_json_response(value, contract)


def _validate_json_response(value, contract):
    """Return the reasons a response is not valid under a JSON contract."""
    try:
        payload = loads_strict_json(value)
    except (DuplicateJsonKeyError, NonFiniteJsonError) as error:
        return [str(error)]
    except ValueError:
        return ["must be valid JSON"]
    if not isinstance(payload, dict):
        return ["must be a JSON object"]
    messages = []
    declared = {field.name: field for field in contract.fields}
    unknown = sorted(set(payload) - set(declared))
    if unknown:
        messages.append("has undeclared fields {}".format(unknown))
    for field in contract.fields:
        if field.name not in payload:
            if field.required:
                messages.append("is missing required field {!r}".format(field.name))
            continue
        messages.extend(_validate_json_value(payload[field.name], field))
    return messages


def _validate_json_value(value, field):
    """Return the reasons one JSON value does not match its declared field."""
    if field.type == "boolean":
        return (
            []
            if isinstance(value, bool)
            else ["field {!r} must be a boolean".format(field.name)]
        )
    if field.type == "string":
        return (
            []
            if isinstance(value, str)
            else ["field {!r} must be a string".format(field.name)]
        )
    if field.type == "array":
        if not isinstance(value, list) or any(
            not isinstance(item, str) for item in value
        ):
            return ["field {!r} must be an array of strings".format(field.name)]
        return []
    if not is_finite_number(value):
        return ["field {!r} must be a finite number".format(field.name)]
    if field.type == "integer" and not isinstance(value, int):
        return ["field {!r} must be an integer".format(field.name)]
    if field.minimum is not None and value < field.minimum:
        return ["field {!r} is below the declared minimum".format(field.name)]
    if field.maximum is not None and value > field.maximum:
        return ["field {!r} is above the declared maximum".format(field.name)]
    return []


def mode_bodies(modes, mode):
    """Return the instruction bodies rendered for one mode, in canonical order.

    Few-shot rendering keeps the zero-shot body so the rubric a model is judged
    against is identical in both modes; only the demonstrations and the closing
    target instruction are added.
    """
    if mode == "few_shot_text":
        return (modes.zero_shot, modes.few_shot_text.template)
    if mode == "pairwise":
        return (modes.pairwise,)
    return (modes.zero_shot,)


def _validate_contract_pair(
    input_contract, response_contract, modes, source, protocol_id, errors
):
    """Check the rules that couple input, response, and available render modes."""
    pairwise_contract = response_contract.mode == "pairwise"
    if pairwise_contract and input_contract.audio_inputs != 2:
        errors.add(source, protocol_id, "pairwise contracts require two audio inputs")
    if input_contract.audio_inputs == 2 and not pairwise_contract:
        errors.add(
            source, protocol_id, "two-audio protocols must use the pairwise contract"
        )
    if modes is None:
        return
    if pairwise_contract and modes.pairwise is None:
        errors.add(source, protocol_id, "pairwise contracts require a pairwise body")
    if modes.pairwise is not None and not pairwise_contract:
        errors.add(
            source,
            protocol_id,
            "a pairwise body requires the pairwise response contract",
        )


def _validate_placeholders(
    input_contract, response_contract, modes, source, protocol_id, errors
):
    """Check every mode body's placeholders against the declared contracts."""
    names_by_body = {}
    for mode in modes.available():
        for body in mode_bodies(modes, mode):
            if body in names_by_body:
                continue
            names, messages = scan_placeholders(body)
            for message in messages:
                errors.add(source, protocol_id, "{}: {}".format(mode, message))
            names_by_body[body] = set(names)
            _validate_body_placeholders(
                names_by_body[body],
                input_contract,
                response_contract,
                mode,
                source,
                protocol_id,
                errors,
            )
    required = set(input_contract.required_context_keys())
    for mode in modes.available():
        used = set()
        for body in mode_bodies(modes, mode):
            used |= names_by_body.get(body, set())
        missing = sorted(required - used)
        if missing:
            errors.add(
                source,
                protocol_id,
                "{}: rendering must use the required context {}".format(mode, missing),
            )


def _validate_body_placeholders(
    used, input_contract, response_contract, mode, source, protocol_id, errors
):
    """Check that one body's placeholders are supported by the contracts."""
    if "labels" in used and not response_contract.choice_labels():
        errors.add(
            source,
            protocol_id,
            "{}: {{labels}} requires a closed_label or pairwise contract".format(mode),
        )
    for name, attribute in sorted(INPUT_PLACEHOLDERS.items()):
        if name in used and not getattr(input_contract, attribute):
            errors.add(
                source,
                protocol_id,
                "{}: {{{}}} requires input_contract.{}".format(mode, name, attribute),
            )
    if input_contract.audio_inputs != 2:
        for name in sorted(OPTIONAL_PLACEHOLDER_DEFAULTS):
            if name in used:
                errors.add(
                    source,
                    protocol_id,
                    "{}: {{{}}} requires two audio inputs".format(mode, name),
                )


def _parse_provenance(mapping, source, protocol_id, errors):
    """Parse the optional provenance block of sources and authoring rationale."""
    block = mapping.get("provenance")
    if block is None:
        return Provenance()
    if not isinstance(block, dict) or set(block) - {"sources", "rationale"}:
        errors.add(
            source, protocol_id, "provenance may only hold sources and rationale"
        )
        return Provenance()
    sources = block.get("sources", [])
    if not isinstance(sources, list) or any(
        not isinstance(item, str) or not item.strip() for item in sources
    ):
        errors.add(source, protocol_id, "provenance.sources must be a list of strings")
        return Provenance()
    rationale = _text(block, "rationale", source, protocol_id, errors, required=False)
    return Provenance(sources=tuple(sources), rationale=rationale)


def _parse_model_compatibility(mapping, source, protocol_id, errors):
    """Parse model-family expectations, requiring evidence for tested entries."""
    entries = mapping.get("model_compatibility", [])
    if not isinstance(entries, list) or not entries:
        errors.add(source, protocol_id, "model_compatibility must be a non-empty list")
        return ()
    allowed = {
        "family",
        "status",
        "model_id",
        "revision",
        "validation",
        "rendering_notes",
    }
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - allowed:
            errors.add(source, protocol_id, "invalid model_compatibility entry")
            return ()
        family = _text(entry, "family", source, protocol_id, errors)
        status = _choice(entry, "status", MODEL_STATUSES, source, protocol_id, errors)
        if family is None or status is None:
            return ()
        model_id = _text(entry, "model_id", source, protocol_id, errors, required=False)
        revision = _text(entry, "revision", source, protocol_id, errors, required=False)
        validation = _text(
            entry, "validation", source, protocol_id, errors, required=False
        )
        if status == "tested" and not (model_id and revision and validation):
            errors.add(
                source,
                protocol_id,
                "model {!r} marked tested needs model_id, revision, and validation".format(
                    family
                ),
            )
            return ()
        parsed.append(
            ModelCompatibility(
                family=family,
                status=status,
                model_id=model_id,
                revision=revision,
                validation=validation,
                rendering_notes=_text(
                    entry,
                    "rendering_notes",
                    source,
                    protocol_id,
                    errors,
                    required=False,
                ),
            )
        )
    if len({entry.family for entry in parsed}) != len(parsed):
        errors.add(source, protocol_id, "model_compatibility families must be unique")
        return ()
    return tuple(parsed)


def _parse_runner_compatibility(mapping, modes, source, protocol_id, errors):
    """Parse which VERSA runners can execute the protocol and in which modes."""
    entries = mapping.get("runner_compatibility", [])
    if not isinstance(entries, list) or not entries:
        errors.add(source, protocol_id, "runner_compatibility must be a non-empty list")
        return ()
    available = set(modes.available()) if modes is not None else set(RENDER_MODES)
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"runner", "status", "modes"}:
            errors.add(source, protocol_id, "invalid runner_compatibility entry")
            return ()
        runner = _text(entry, "runner", source, protocol_id, errors)
        status = _choice(entry, "status", RUNNER_STATUSES, source, protocol_id, errors)
        if runner is None or status is None:
            return ()
        declared = entry.get("modes", [])
        if not isinstance(declared, list) or any(
            not isinstance(item, str) or item not in available for item in declared
        ):
            errors.add(
                source,
                protocol_id,
                "runner {!r} declares modes outside {}".format(
                    runner, sorted(available)
                ),
            )
            return ()
        if status != "unsupported" and not declared:
            errors.add(
                source, protocol_id, "runner {!r} must declare its modes".format(runner)
            )
            return ()
        parsed.append(
            RunnerCompatibility(runner=runner, status=status, modes=tuple(declared))
        )
    if len({entry.runner for entry in parsed}) != len(parsed):
        errors.add(source, protocol_id, "runner_compatibility runners must be unique")
        return ()
    return tuple(parsed)


def _parse_metric_links(mapping, source, protocol_id, errors):
    """Parse optional links to traditional VERSA metrics used as context."""
    entries = mapping.get("metric_links", [])
    if not isinstance(entries, list):
        errors.add(source, protocol_id, "metric_links must be a list")
        return ()
    parsed = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"metric", "role"}:
            errors.add(source, protocol_id, "each metric link needs metric and role")
            return ()
        metric = _text(entry, "metric", source, protocol_id, errors)
        role = _choice(entry, "role", METRIC_LINK_ROLES, source, protocol_id, errors)
        if metric is None or role is None:
            return ()
        parsed.append(MetricLink(metric=metric, role=role))
    if len({entry.metric for entry in parsed}) != len(parsed):
        errors.add(source, protocol_id, "metric_links must reference unique metrics")
        return ()
    return tuple(parsed)
