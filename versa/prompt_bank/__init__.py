"""VERSA Prompt Bank: versioned, renderable audio evaluation protocols.

The public API is deliberately small and backend-free. Importing this package
loads protocol data definitions only; it never imports Torch, Transformers, or a
VERSA metric module.

Example:
    >>> from versa.prompt_bank import get_protocol, render_protocol
    >>> rendered = render_protocol(get_protocol("speech.emotion.v1"))
    >>> rendered.protocol_id
    'speech.emotion.v1'

All bundled protocols are ``experimental``: they load, validate, list, and
render, but no metric executes them yet, and they do not carry human-grounded
validation evidence. ``generation.pairwise_alignment.v1`` renders and stays
unexecutable until a runner supplies two audio inputs.
"""

from versa.prompt_bank.loader import (
    PromptBank,
    get_protocol,
    list_protocols,
    load_bank,
    validate_bank,
)
from versa.prompt_bank.renderer import RenderedPrompt, render_protocol
from versa.prompt_bank.schema import (
    BANK_SCHEMA_VERSION,
    BankValidationError,
    InputContract,
    JsonField,
    MetricLink,
    ModelCompatibility,
    ParsedResponse,
    PromptBankError,
    Protocol,
    ProtocolNotFoundError,
    RENDERER_VERSION,
    RenderError,
    RESPONSE_STATUSES,
    ResponseContract,
    RunnerCompatibility,
    parse_response,
    validate_response,
)

__all__ = [
    "BANK_SCHEMA_VERSION",
    "RENDERER_VERSION",
    "RESPONSE_STATUSES",
    "BankValidationError",
    "InputContract",
    "JsonField",
    "MetricLink",
    "ModelCompatibility",
    "ParsedResponse",
    "PromptBank",
    "PromptBankError",
    "Protocol",
    "ProtocolNotFoundError",
    "RenderError",
    "RenderedPrompt",
    "ResponseContract",
    "RunnerCompatibility",
    "get_protocol",
    "list_protocols",
    "load_bank",
    "parse_response",
    "render_protocol",
    "validate_bank",
    "validate_response",
]
