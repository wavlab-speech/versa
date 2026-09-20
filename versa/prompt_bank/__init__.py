"""VERSA Prompt Bank: versioned, renderable audio evaluation protocols.

The public API is deliberately small and backend-free. Importing this package
loads protocol data definitions only; it never imports Torch, Transformers, or a
VERSA metric module.

Example:
    >>> from versa.prompt_bank import get_protocol, render_protocol
    >>> rendered = render_protocol(get_protocol("speech.emotion.v1"))
    >>> rendered.protocol_id
    'speech.emotion.v1'

All bundled protocols are ``experimental``: they are executable, but they do not
yet carry human-grounded validation evidence.
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
    PromptBankError,
    Protocol,
    ProtocolNotFoundError,
    RenderError,
    ResponseContract,
    RunnerCompatibility,
    validate_response,
)

__all__ = [
    "BANK_SCHEMA_VERSION",
    "BankValidationError",
    "InputContract",
    "JsonField",
    "MetricLink",
    "ModelCompatibility",
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
    "render_protocol",
    "validate_bank",
    "validate_response",
]
