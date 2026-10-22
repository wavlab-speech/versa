"""Deterministic text rendering for VERSA Prompt Bank protocols.

The renderer owns whitespace, section order, label order, demonstration order,
context delimiters, and structured-output templates. It never loads audio,
imports a model, or treats context values as templates. Free-text context is
enclosed in tags named after its placeholder and is otherwise reproduced
verbatim, so caller text cannot read as part of the protocol's instructions.
"""

from dataclasses import dataclass
from typing import Optional

from versa.prompt_bank.schema import (
    BANK_SCHEMA_VERSION,
    compute_digest,
    DELIMITED_PLACEHOLDERS,
    digest_text,
    LABEL_PLACEHOLDERS,
    MAX_LABEL_PLACEHOLDER_LENGTH,
    mode_bodies,
    OPTIONAL_PLACEHOLDER_DEFAULTS,
    RENDER_MODES,
    RENDERER_PLACEHOLDERS,
    RENDERER_VERSION,
    SEQUENCE_PLACEHOLDERS,
    Protocol,
    RenderError,
    ResponseContract,
    scan_placeholders,
    sorted_keys,
    substitute,
)


@dataclass(frozen=True)
class RenderedPrompt:
    """One rendered protocol text together with its reproducibility identity.

    ``protocol_digest`` identifies the protocol, including the renderer version;
    two renderings with different captions or instructions share it.
    ``context_digest`` identifies the caller context after validation and
    defaults, and ``rendered_digest`` identifies this exact text, so a result
    record can prove which prompt produced it.
    """

    text: str
    protocol_id: str
    protocol_version: int
    mode: str
    protocol_digest: str
    rendered_digest: str
    context_digest: str
    renderer_version: int = RENDERER_VERSION
    bank_schema_version: int = BANK_SCHEMA_VERSION
    response_schema: Optional[ResponseContract] = None


def render_protocol(protocol, mode="zero_shot", context=None):
    """Render one protocol in the requested mode and return a `RenderedPrompt`.

    ``protocol`` is a protocol record or a versioned protocol ID. Unknown modes,
    unknown context keys, and missing required context raise
    :class:`~versa.prompt_bank.schema.RenderError` before any text is produced.
    """
    protocol = _resolve(protocol)
    if mode not in RENDER_MODES:
        raise RenderError(
            "unknown render mode {!r}; supported modes are {}".format(
                mode, list(RENDER_MODES)
            )
        )
    available = protocol.available_modes()
    if mode not in available:
        raise RenderError(
            "protocol {} does not define mode {!r}; it defines {}".format(
                protocol.id, mode, list(available)
            )
        )
    bodies = mode_bodies(protocol.modes, mode)
    context_values = _resolve_context(protocol, mode, bodies, context or {})
    values = _substitution_values(protocol, context_values)
    # Protocol text is canonicalized before substitution, so context values are
    # reproduced exactly instead of having their own blank lines collapsed.
    sections = [substitute(_canonical_section(bodies[0]), values)]
    if mode == "few_shot_text":
        sections.append(
            _canonical_section(_examples_section(protocol.modes.few_shot_text.examples))
        )
    for body in bodies[1:]:
        sections.append(substitute(_canonical_section(body), values))
    sections.append(
        _canonical_section(response_instructions(protocol.response_contract))
    )
    rendered = "\n\n".join(section for section in sections if section) + "\n"
    return RenderedPrompt(
        text=rendered,
        protocol_id=protocol.id,
        protocol_version=protocol.version,
        mode=mode,
        protocol_digest=protocol.digest,
        rendered_digest=digest_text(rendered),
        context_digest=compute_digest(context_values),
        renderer_version=RENDERER_VERSION,
        bank_schema_version=BANK_SCHEMA_VERSION,
        response_schema=protocol.response_contract,
    )


def _resolve(protocol):
    """Return a protocol record, looking an ID up in the bundled bank."""
    if isinstance(protocol, Protocol):
        return protocol
    if isinstance(protocol, str):
        from versa.prompt_bank.loader import get_protocol

        return get_protocol(protocol)
    raise RenderError("protocol must be a Protocol record or a protocol ID string")


def _resolve_context(protocol, mode, bodies, context):
    """Validate caller context strictly and return the normalized caller values.

    The result covers every caller placeholder the rendering uses, with
    documented defaults applied, and is what ``context_digest`` identifies.
    """
    if not isinstance(context, dict):
        raise RenderError("context must be a mapping")
    used = []
    for body in bodies:
        used.extend(scan_placeholders(body)[0])
    caller_keys = [
        name for name in dict.fromkeys(used) if name not in RENDERER_PLACEHOLDERS
    ]
    unknown = sorted_keys(set(context) - set(caller_keys))
    if unknown:
        raise RenderError(
            "unknown context keys {} for {} in mode {!r}; this rendering accepts {}".format(
                unknown, protocol.id, mode, sorted(caller_keys)
            )
        )
    return {name: _context_value(protocol, mode, name, context) for name in caller_keys}


def _substitution_values(protocol, context_values):
    """Return the text substituted for every placeholder of one rendering."""
    values = {"labels": ", ".join(protocol.response_contract.choice_labels())}
    for name, value in context_values.items():
        if name in SEQUENCE_PLACEHOLDERS:
            values[name] = "\n".join("- " + item for item in value)
        elif name in DELIMITED_PLACEHOLDERS:
            values[name] = "<{0}>\n{1}\n</{0}>".format(name, value)
        else:
            values[name] = value
    return values


def _context_value(protocol, mode, name, context):
    """Return one validated context value, applying documented defaults."""
    if name not in context:
        if name in OPTIONAL_PLACEHOLDER_DEFAULTS:
            return OPTIONAL_PLACEHOLDER_DEFAULTS[name]
        raise RenderError(
            "context key {!r} is required to render {} in mode {!r}".format(
                name, protocol.id, mode
            )
        )
    value = context[name]
    if name in SEQUENCE_PLACEHOLDERS:
        if (
            not isinstance(value, (list, tuple))
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            raise RenderError(
                "context key {!r} must be a non-empty sequence of strings".format(name)
            )
        items = [_normalize_newlines(item).strip() for item in value]
        if any(len(item.splitlines()) > 1 for item in items):
            raise RenderError(
                "context key {!r} items must each be a single line".format(name)
            )
        return items
    if not isinstance(value, str) or not value.strip():
        raise RenderError("context key {!r} must be a non-empty string".format(name))
    value = _normalize_newlines(value).strip()
    if name in LABEL_PLACEHOLDERS and (
        len(value.splitlines()) > 1 or len(value) > MAX_LABEL_PLACEHOLDER_LENGTH
    ):
        raise RenderError(
            "context key {!r} must be a single-line label of at most {} "
            "characters".format(name, MAX_LABEL_PLACEHOLDER_LENGTH)
        )
    if name in DELIMITED_PLACEHOLDERS:
        lowered = value.lower()
        for tag in ("<{}>".format(name), "</{}>".format(name)):
            if tag in lowered:
                raise RenderError(
                    "context key {!r} must not contain the delimiter {}".format(
                        name, tag
                    )
                )
    return value


def _normalize_newlines(value):
    """Convert Windows and old Mac line endings to ``\\n``."""
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _examples_section(examples):
    """Render text-only demonstrations in their authored order."""
    lines = ["Examples:", ""]
    for example in examples:
        lines.append("Input: {}".format(example.input))
        lines.append("Output: {}".format(example.output))
        lines.append("")
    return "\n".join(lines)


def response_instructions(contract):
    """Generate the response instructions implied by a response contract."""
    if contract.mode in ("closed_label", "pairwise"):
        lines = [
            "Respond with exactly one label from: {}.".format(
                ", ".join(contract.choice_labels())
            )
        ]
    elif contract.mode == "integer":
        lines = [
            "Respond with a single integer between {} and {}.".format(
                contract.minimum, contract.maximum
            )
        ]
    elif contract.mode == "scalar":
        lines = [
            "Respond with a single number between {} and {}.".format(
                contract.minimum, contract.maximum
            ),
            "Scale anchors:",
        ]
        lines.extend(
            "- {}: {}".format(_number(anchor.value), anchor.description)
            for anchor in contract.anchors
        )
    else:
        return "\n".join(
            [
                "Return valid JSON only, with no other text and no keys beyond these:",
                _json_template(contract),
                _json_requirement_note(contract),
            ]
        )
    if contract.allow_abstain:
        reason = (
            "any of those labels"
            if contract.mode in ("closed_label", "pairwise")
            else "an answer"
        )
        lines.append(
            'Respond with "{}" when the audio does not support {}.'.format(
                contract.abstain_label, reason
            )
        )
    lines.append("Do not add any other text.")
    return "\n".join(lines)


def _json_template(contract):
    """Build the compact JSON response template in declared field order."""
    parts = [
        '"{}": {}'.format(field.name, _json_placeholder(field))
        for field in contract.fields
    ]
    return "{" + ", ".join(parts) + "}"


def _json_requirement_note(contract):
    """State which declared keys are required, so optional keys are unambiguous."""
    optional = [field.name for field in contract.fields if not field.required]
    if not optional:
        return "Every key is required."
    return "Every key is required except {}, which may be omitted.".format(
        ", ".join('"{}"'.format(name) for name in optional)
    )


def _json_placeholder(field):
    """Return the placeholder shown for one declared JSON field.

    One-sided bounds are rendered as stated bounds rather than dropped, so the
    instructions describe the same contract the validator enforces.
    """
    if field.type == "boolean":
        return "<true|false>"
    if field.type == "string":
        return "<string>"
    if field.type == "array":
        return "[<string>]"
    if field.minimum is not None and field.maximum is not None:
        return "<{} {}-{}>".format(
            field.type, _number(field.minimum), _number(field.maximum)
        )
    if field.minimum is not None:
        return "<{}, at least {}>".format(field.type, _number(field.minimum))
    if field.maximum is not None:
        return "<{}, at most {}>".format(field.type, _number(field.maximum))
    return "<{}>".format(field.type)


def _number(value):
    """Format a bound without a trailing ``.0`` for whole numbers."""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _canonical_section(section):
    """Strip trailing spaces and collapse repeated blank lines in protocol text.

    Only protocol and renderer text passes through here; caller context is
    substituted afterwards and kept verbatim.
    """
    lines = []
    for line in section.strip().splitlines():
        line = line.rstrip()
        if not line and (not lines or not lines[-1]):
            continue
        lines.append(line)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)
