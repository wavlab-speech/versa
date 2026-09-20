# Prompt Bank

The Prompt Bank is a versioned library of audio evaluation *protocols*. A
protocol is a measurement specification: a task, an input contract, a response
contract, a rendering, provenance, and a version. It is not an agent framework,
a model host, or a scorer.

This release is the foundation only. Protocols load, validate, list, and render;
**no metric runs them yet**. Wiring `prompt_id` through the existing Qwen
wrappers is a separate, additive change that does not alter any current metric
name, alias, default prompt, or output key.

## Installation and import

The bank ships inside the `versa` wheel and needs no extra dependency beyond
PyYAML, which VERSA already requires. Loading it never imports Torch,
Transformers, or a metric backend.

```python
from versa.prompt_bank import get_protocol, list_protocols, render_protocol

protocol = get_protocol("speech.emotion.v1")
rendered = render_protocol(protocol, mode="zero_shot")
print(rendered.text)
```

## Bundled protocols

All eight protocols are `experimental`: they are complete and renderable, but
they carry no human-grounded validation evidence yet. Do not report them as
validated metrics.

| Protocol ID | Audio inputs | Required context | Response contract | Modes |
| --- | --- | --- | --- | --- |
| `speech.emotion.v1` | 1 | — | closed label, 8 labels + `unclear` | zero-shot, few-shot text |
| `speech.recording_quality.v1` | 1 | — | closed label, ordered 5-point scale | zero-shot, few-shot text |
| `speech.speaker_count.v1` | 1 | — | integer 0–10 | zero-shot |
| `speech.overlap.v1` | 1 | — | closed label, ordered 5-point scale | zero-shot, few-shot text |
| `audio.caption_accuracy.v1` | 1 | `reference_text` | JSON, reports `prompt_audio_caption_accuracy` | zero-shot |
| `generation.prompt_alignment.v1` | 1 | `target_instruction` | JSON, reports `prompt_generation_prompt_alignment` | zero-shot |
| `generation.pairwise_alignment.v1` | 2 | `target_instruction` | pairwise preference `a`/`b`/`tie` + `unclear` | zero-shot, pairwise |
| `interaction.turn_taking.v1` | 1 | — | JSON, reports `prompt_interaction_turn_taking` | zero-shot |

`generation.pairwise_alignment.v1` renders but cannot execute: no current VERSA
runner supplies two audio inputs. Its `runner_compatibility` says so explicitly,
and that is intentional — model capability and runner capability are declared
separately.

`interaction.turn_taking.v1` is a qualitative assessment made from a single
recording without event timestamps. It is not a measurement of response latency
or of interruption counts.

## Public API

| Function | Behavior |
| --- | --- |
| `get_protocol(protocol_id)` | Return one immutable protocol. Unknown IDs raise `ProtocolNotFoundError` (a `KeyError`) with close-match suggestions. |
| `list_protocols(...)` | Filter by `domain`, `task`, `audio_inputs`, `response_mode`, `model_family`, `runner`, and `status`; results are sorted by ID. The default view shows `experimental` and `stable` records only. |
| `render_protocol(protocol_or_id, mode, context)` | Return a `RenderedPrompt`. |
| `validate_bank()` | Re-read and validate every bundled record, reporting all errors in one `BankValidationError`. |
| `validate_response(text, contract)` | Return the reasons a response does not satisfy a contract; an empty list means it does. |

A `RenderedPrompt` carries `text`, `protocol_id`, `protocol_version`, `mode`,
`protocol_digest`, `bank_schema_version`, and `response_schema`. Record those
identity fields with any score you keep: a score without its protocol is not
reproducible.

Protocol records are frozen dataclasses holding tuples, so a caller cannot
mutate cached bank state, at any nesting depth.

## Rendering modes

| Mode | Meaning |
| --- | --- |
| `zero_shot` | The instruction body plus the generated response instructions. |
| `few_shot_text` | Text-only demonstrations, then the target instruction, then the response instructions. |
| `pairwise` | The two-candidate comparison body. Rendering it does not make it executable. |

The renderer owns whitespace, section order, label order, demonstration order,
and structured-output templates. It generates the response instructions from the
response contract, so a protocol never hand-writes its own JSON template. Every
rendering ends with exactly one newline and is covered by an exact snapshot test
in `test/prompt_bank_snapshots.json`.

Few-shot demonstrations are text-only by design. They describe audible evidence;
they are not audio examples, and benchmark results for `zero_shot` and
`few_shot_text` must be reported separately, never blended.

## Context rules

Context is validated strictly, before any text is produced:

- Required keys come from the input contract: `target_instruction` when
  `requires_target_instruction` is set, `reference_text` when
  `requires_reference_text` is set. A blank value is rejected.
- `candidate_a_name` and `candidate_b_name` are optional, default to
  `Candidate A` and `Candidate B`, and exist only for two-audio protocols.
- Unknown keys are rejected, so a misspelling cannot silently change an
  evaluation.
- `labels` is supplied by the renderer from the response contract and must not
  be passed by a caller.
- Context values are inserted verbatim. They are never re-interpreted as
  templates, so a brace inside a caption or instruction is safe.

```python
rendered = render_protocol(
    "audio.caption_accuracy.v1",
    mode="zero_shot",
    context={"reference_text": "A dog barks twice, then a door closes."},
)
```

## Versioning, digests, and lifecycle

A protocol ID is an immutable measurement identifier ending in `.vN`, and `N`
must match the record's `version`. Any change that can alter rendered text,
accepted inputs, or parsed outputs requires a new version — never an edit in
place. Editorial changes to titles, descriptions, citations, and rationale may
keep the ID.

`protocol_digest` is the SHA-256 of a canonical JSON form (UTF-8, recursively
sorted keys, order-preserving arrays, compact separators) of the
evaluation-bearing fields only: ID, version, input contract, response contract,
mode bodies, and model-specific rendering notes. Descriptions, citations, metric
links, status, and runner declarations are excluded. `test/test_prompt_bank.py`
pins the eight digests so a YAML or JSON library change cannot silently alter
them.

The bank schema version is `1`; it describes the serialization format and does
not replace protocol versioning.

| Status | Meaning |
| --- | --- |
| `draft` | Incomplete or under review. Hidden from the default listing. |
| `experimental` | Complete and renderable, without human-grounded validation. |
| `stable` | Rendering frozen within the version, validation evidence documented. |
| `deprecated` | Retrievable by exact ID for reproducibility, hidden from the default listing. |

## Adding or changing a protocol

1. Edit or add a YAML record under `versa/prompt_bank/data/`, and list any new
   file in `manifest.yaml`. Validation rejects a missing file, a repeated entry,
   and an unindexed protocol file.
2. Start a new protocol at `status: experimental` and never at `stable`.
3. Run `python -m pytest test/test_prompt_bank.py -q`. Rendering and digest
   snapshots will fail if the change is visible to a model.
4. If the change is intentional, bump the protocol version, regenerate the
   snapshots with `python test/test_prompt_bank.py`, update the digest fixtures,
   and review the diff as evaluation logic rather than as text.

## Current limitations

- No metric executes a protocol yet; `prompt_id` support in the Qwen wrappers is
  the next increment.
- Response parsing, provenance envelopes, and report integration are not part of
  this release. `validate_response` checks a response against a contract but
  does not parse results into scores.
- Audio few-shot examples, dynamic rubrics, and multi-stage judge pipelines are
  out of scope until their licensing, rendering, and validation questions are
  settled.
- A protocol is not a benchmark. Promotion to `stable` requires documented
  agreement and bias analysis, not prompt coverage.

See [prompt_bank_implementation_plan.md](prompt_bank_implementation_plan.md) for
the full design, the increment plan, and the decisions this implementation
locked in.
