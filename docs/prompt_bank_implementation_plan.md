# VERSA Prompt Bank Implementation Plan

Design version: 2 (increment A implemented; judging and backend scope added).
Last reconciled: 2026-09-20.

Status: increment A (bank foundation) is implemented in `versa/prompt_bank/`
with eight experimental protocols, foundation tests, and package data; see
[prompt_bank.md](prompt_bank.md) for the user-facing guide. Increments B and C
below remain plans. The design decisions that increment A locked in are recorded
in [Locked decisions](#locked-decisions-increment-a-reconciliation); where this
document and that section disagree, that section is authoritative.
[Audio judging as a supported workflow](#audio-judging-as-a-supported-workflow)
and [Judge backends](#judge-backends-local-weights-and-hosted-apis) define the
judged evaluation path and the local/hosted adapter contract; the integrations
they scope land separately. This plan is
the detailed protocol design referenced by action IDs P0-P4 of the development
action register maintained with the maintainers' roadmap notes.

## Purpose

Add a small, research-aligned Prompt Bank to VERSA for reusable audio evaluation
protocols. The Prompt Bank is not an agent framework, model host, or audio
generation system. It is a versioned library of prompt specifications that can
be rendered by existing audio-language-model metrics, starting with the
Qwen2-Audio and Qwen2.5-Omni integrations already in VERSA.

The first release must make prompt-only and few-shot audio evaluation usable
without changing the existing scoring workflow. It must also preserve every
current Qwen metric name, default prompt, raw `prompt` override, output key,
and configuration example.

## Current Repository Anchor

The implementation should build on the existing narrow path rather than
introduce a parallel evaluation framework.

| Existing component | Current behavior | Prompt Bank role |
| --- | --- | --- |
| `versa/utterance_metrics/qwen2_audio.py` | Owns `DEFAULT_PROMPTS`, creates one registry metric per prompt, accepts `prompt` | Resolve a named bank protocol into the existing `prompt` path. |
| `versa/utterance_metrics/qwen_omni.py` | Reuses Qwen2-Audio `DEFAULT_PROMPTS` | Use the same resolver and protocol IDs. |
| `versa/metric_registry.py` | Lazily resolves metric modules so importing `versa` pulls in no model backend | The bank keeps the same property: `versa.prompt_bank` imports no metric or model module. |
| `versa/metric_discovery.py` | Statically discovers Qwen prompt names by parsing the `DEFAULT_PROMPTS` assignment with `ast` | Keep this working in v0; migrate discovery only after the bank is stable. |
| `versa/scorer_shared.py` | Instantiates registry metrics from YAML dictionaries | Pass `prompt_id`, `prompt_context`, and eventually protocol settings as ordinary metric config. |
| `versa/config_validation.py` | Validates metric configuration against constructor signatures | Extend validation when Prompt Bank-specific fields become public. |
| `docs/contributing.md` | Defines the metric, docs, example, and test conventions | Add a short Prompt Bank contributor section after the vertical slice is complete. |

## Product Boundary

### In scope

- Named, versioned prompt protocols for one or two audio inputs plus optional
  text instructions and reference text.
- Prompt-only evaluation with structured, label, scalar, or pairwise outputs.
- Text-only few-shot demonstrations that specify a rubric and output format.
- A small model compatibility catalogue that records tested model families and
  model-specific rendering notes.
- Optional links from protocols to traditional VERSA metrics for calibration
  and interpretation.
- Direct execution through current Qwen2-Audio and Qwen2.5-Omni metric
  wrappers.

### Explicitly out of scope for v0 and v1

- Generating or editing audio.
- Training reward models or judge models.
- A generic remote API client, model serving layer, or provider credentials
  *inside the bank*. `versa.prompt_bank` never imports a provider SDK, endpoint,
  or credential. Named judge adapters for local weights and hosted APIs are in
  scope as separate optional integrations; see
  [Judge backends](#judge-backends-local-weights-and-hosted-apis).
- Autonomous voice-agent orchestration, tool execution, or real-time sessions.
- Shipping third-party audio few-shot examples. They create dataset licensing,
  storage, preprocessing, and model-format obligations that are not needed for
  the first useful release.
- Replacing traditional VERSA metrics with judge outputs.

## Design Principles

1. A prompt is a measurement protocol, not just an instruction string. Each
   record carries a task, input contract, response contract, version, and
   provenance.
2. Prompt-only evaluation is first-class. A reference signal or a traditional
   metric may be linked, but neither is required unless the protocol says so.
3. Few-shot is a protocol mode, not a special metric implementation. The
   renderer selects zero-shot or text-only few-shot examples from the same
   specification.
4. Preserve compatibility. `prompt` always overrides `prompt_id`; absence of
   both keeps today’s `DEFAULT_PROMPTS` behavior.
5. Start with deterministic local operations: load, validate, list, and render.
   Do not couple the initial bank to heavyweight model loading or network calls.
6. Separate evidence extraction from judgment in future multi-stage pipelines.
   Audio-language models should state observable evidence before a text model
   turns it into a final decision.
7. Store the model name, prompt-bank version, protocol ID, rendering mode, and
   raw response with results whenever possible. A score without its protocol is
   not reproducible.

## Versioning and Reproducibility Contract

Protocol IDs are immutable measurement identifiers. Every bundled protocol ID
must end in `.vN`, and the suffix must match the integer `version` field. Any
change that can alter rendered text, accepted inputs, parsed outputs, scoring,
or model-specific rendering requires a new protocol version. Editorial changes
to descriptions, citations, and rationale may retain the ID only when they do
not affect rendering or validation.

The loader computes a deterministic `protocol_digest` from the canonicalized
evaluation-bearing fields: ID, version, input contract, response contract,
protocol modes, and rendering-affecting compatibility notes. `RenderedPrompt`
and structured result provenance carry this digest. Results should also record
the installed VERSA package version and bank schema version when available.
This protects reproducibility when comparing results produced from different
checkouts or package builds.

Canonicalization uses UTF-8 JSON with recursively sorted object keys,
order-preserving arrays, `ensure_ascii=False`, and compact separators before a
SHA-256 digest is calculated. Descriptions, citations, and other non-evaluation
metadata are excluded. Add a digest fixture test so Python/YAML library changes
cannot silently alter the algorithm.

The initial bank schema version is `1`. Schema-version changes describe the
serialization format and do not replace protocol versioning.

## Release Shape

The work is intentionally divided into three releasable increments. Complete
the first increment before committing to the second.

| Increment | Outcome | New runtime dependencies | Main risk |
| --- | --- | --- | --- |
| A. Bank foundation | Load, validate, list, and render a compact set of protocols. No scorer integration yet. | None beyond existing PyYAML. | Schema grows prematurely. |
| B. Qwen vertical slice | `prompt_id` runs through existing Qwen metric classes and YAML configs. | Existing optional Qwen dependencies only. | Backward compatibility with Qwen prompt discovery. |
| C. Evaluation protocols | Structured parsing, prompt result provenance, few-shot mode, pairwise protocols, and reporting. | None required for parsing. | Treating noisy judge outputs as calibrated scores. |
| D. Judge backends | One adapter contract; the same protocol runs on local weights and on hosted APIs (Gemini, Qwen3.5-Omni, Qwen3.8-Omni-Flash). | Optional per-provider extras only. | Provider coupling leaking into the bank; mutable hosted endpoints. |

Multi-stage audio/text judge pipelines are a future design track. They should
not block A-C.

## Locked decisions (increment A reconciliation)

These decisions resolve the ambiguities this plan left open, and they match the
shipped implementation. Later sections describe intent; this section describes
what the code does.

### Terminology and scope

- `protocol` is the public record term and `Prompt Bank` the feature name.
  `prompt` stays reserved for raw rendered text and the legacy metric override.
- The eight initial IDs ship as `experimental`. No protocol is `stable`, and
  promotion requires documented human-grounded validation, not prompt coverage.
- Increment A changes no Qwen, scorer, discovery, or legacy prompt code.

### Rendering

- `few_shot_text` renders the zero-shot body, then the demonstrations, then the
  `template` body as a closing instruction. Demonstrations never replace the
  rubric, so a mode comparison changes the demonstrations only and each
  protocol's few-shot template is a closer rather than a restatement. Required
  context is checked across a mode's whole rendering rather than per body.
  Benchmarking still reports the two modes separately.
- Every protocol must define `zero_shot`, including a pairwise protocol, whose
  `pairwise` body adds candidate naming. A `pairwise` body requires the pairwise
  response contract and two audio inputs, and two audio inputs require the
  pairwise contract.
- Placeholders are simple names from a fixed allow-list: `labels`,
  `target_instruction`, `reference_text`, `candidate_a_name`,
  `candidate_b_name`, and `rubric_items`. Attribute access, indexing, conversion
  flags, and format specifications are rejected, and a literal brace must be
  escaped.
- `{labels}` is supplied by the renderer from the response contract and is
  rejected as caller context. It is only valid in a `closed_label` or `pairwise`
  protocol.
- `candidate_a_name` and `candidate_b_name` are optional context with the fixed
  defaults `Candidate A` and `Candidate B`, and are valid only for two-audio
  protocols. `rubric_items` is accepted as a sequence of strings rendered as a
  dash list; no bundled protocol uses it yet.
- A required text input must appear in every renderable body of its protocol.
  Declaring an input that no body renders would silently ignore caller context.
- Context values are substituted verbatim and never re-interpreted, so a brace
  inside a caption or instruction is safe. Unknown context keys, blank values,
  and missing required keys raise `RenderError` before any text is produced.
- Response instructions are generated from the response contract, including the
  compact JSON template in declared field order. Protocols do not hand-write
  JSON templates. Abstention wording differs by contract mode: label contracts
  say "any of those labels", numeric contracts say "an answer". One-sided bounds
  render as stated bounds (`<integer, at least 0>`) and optional keys are named,
  so the instructions and the validator describe the same contract.
- Canonical text: each section is stripped, per-line trailing whitespace is
  removed, blank-line runs collapse to one, sections join with one blank line,
  and the prompt ends with exactly one newline. Authored YAML paragraphs are
  single lines so substitution cannot produce ragged wrapping.

### Schema

- A JSON contract declares abstention with a required boolean `abstain` field.
  `abstain_label` is rejected there; label, integer, scalar, and pairwise
  contracts require it whenever `allow_abstain` is true, and it must differ from
  every response label.
- `output_key` requires an explicit `primary_numeric_field`; that field must be
  numeric, required, and carry an explicit range. Output keys are unique across
  the bank and must end in `_v<version>`. Protocol IDs and output keys use
  different spellings of the same version: protocol `speech.example.v2` reports
  under `prompt_example_score_v2`, so it can run beside a deprecated
  `speech.example.v1` reporting under `prompt_example_score_v1`. Migration: a new
  protocol version introduces a new output key, and consumers of the old key keep
  reading the old protocol's results. No score
  key is inferred from arbitrary JSON.
- Array fields carry `items: string` in v0.
- `id` matches `^[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)+\.v[1-9][0-9]*$`, is globally
  unique, and its trailing version matches the `version` field.
- `response_contract.mode` determines the mandatory fields: `labels` for
  `closed_label`, an integer range for `integer`, a numeric range and anchors
  for `scalar`, typed `fields` for `json`, and `preference_labels` for
  `pairwise`.
- A model entry's `rendering_notes` documents how that family needs a protocol
  presented, for example the order two audio inputs arrive in. The renderer does
  not read it; it participates in the digest because changing it changes how the
  protocol must be presented. It is compatibility documentation, not rendering
  logic.
- Unknown keys are rejected at every level (protocol, input contract, response
  contract, JSON field, provenance, compatibility entry) so a typo fails loudly.
- Numeric bounds must be finite. A `NaN` bound would make every range comparison
  pass. Integers are unbounded in Python and always finite; only floats are
  checked, so a large integer literal is validated rather than overflowing.
- Protocol YAML is parsed with a loader that rejects duplicate mapping keys and
  reports file, line, and column, and that rejects unhashable keys the same way.
  YAML here is executable evaluation logic. Malformed nested values reach the
  validation report rather than raising `TypeError`, including mixed-type keys
  that cannot be ordered against each other.
- Few-shot demonstrations must be text-only: at least two of them, no audio file
  references, and every output must pass `validate_response` against the
  protocol's own contract. Demonstration text is rendered literally and is never
  substituted, so JSON examples containing braces are allowed.
- `model_compatibility` describes the model family and `runner_compatibility`
  describes VERSA execution; they are separate, and a `tested` model entry
  requires a model ID, a revision, and a validation record. All bundled runner
  entries are `planned`, except the pairwise protocol's entries, which are
  `unsupported`. Increment B promotes the one-audio Qwen entries.
- `metric_links` are parsed and validated for shape and role only. The loader
  stays backend-independent, so no registry lookup runs in increment A; optional
  registry validation is deferred to the increment that displays those links.

### Digest and validation reporting

- `protocol_digest` is the SHA-256 of canonical JSON (UTF-8, recursively sorted
  keys, order-preserving arrays, `ensure_ascii=False`, compact separators) over
  exactly: `id`, `version`, the input contract, the response contract, the mode
  bodies, and model entries' `rendering_notes`. Titles, descriptions, status,
  provenance, metric links, and runner declarations are excluded. The bank
  schema version is carried beside the digest rather than inside it.
- `RenderedPrompt` also carries `rendered_digest`, the SHA-256 of the exact
  rendered text. Two renderings of one protocol with different captions share a
  protocol digest by design, so the rendered digest is what identifies an
  evaluation.
- Response validation is strict: non-finite numbers (`NaN`, `Infinity`, and
  overflow such as `1e400`) and repeated JSON keys are rejected rather than
  silently resolved to the last value.
- Validation collects every error in one `BankValidationError`, each message
  prefixed with its source file and protocol ID.
- `manifest.yaml` is authoritative: a missing indexed file, a repeated entry, a
  self-reference, and an unindexed protocol file are all rejected.

### Decisions recorded now for increment B

- Qwen prompt precedence is raw `prompt` (using `prompt is not None`), then the
  bank protocol, then the legacy `DEFAULT_PROMPTS` default. Preserving an empty
  raw prompt is an intentional behavior correction from `self.prompt or ...`; it
  will be tested and documented rather than described as byte-for-byte
  compatible.
- Protocol-selected input requirements are validated dynamically, before model
  setup, so an invalid protocol configuration never downloads a checkpoint.
  Transcript, caption, generation instruction, and lyrics stay distinct roles.
- Per-utterance context needs a small validated key-to-context mapping or
  manifest adapter, specified before the first evaluation pilot; missing and
  duplicate keys are rejected. A static `prompt_context` covers shared
  instructions only.
- Rendering a pairwise protocol does not make a one-audio runner able to execute
  it. Runner support stays separate from model capability.
- Likelihood or logits scoring is not part of this foundation. It is added only
  with a versioned scoring contract in the increment that needs it.
- Turn-taking output is experimental qualitative assessment. Validated
  interaction timing requires event or timestamp evidence this protocol does not
  collect.

## Proposed Repository Layout

Create the following files only as their corresponding increment begins.

```text
versa/
  prompt_bank/
    __init__.py               # Increment A, shipped
    loader.py                 # Increment A, shipped
    schema.py                 # Increment A, shipped
    renderer.py               # Increment A, shipped
    models.py                 # Increment C, not created yet
    data/
      __init__.py             # Makes resources accessible on Python 3.8
      manifest.yaml           # Ordered, authoritative file index
      speech_understanding.yaml
      generation_alignment.yaml
      interaction.yaml
      audio_quality.yaml      # Increment C, only when justified
docs/
  prompt_bank.md              # Increment A, shipped
  prompt_bank_implementation_plan.md
  prompt_bank_survey.md       # Research notes and bibliography, separate PR
egs/
  separate_metrics/
    qwen2_audio_prompt_bank.yaml   # Increment B
    qwen_omni_prompt_bank.yaml     # Increment B
test/
  test_prompt_bank.py             # Increment A, shipped
  prompt_bank_snapshots.json      # Increment A, exact rendering snapshots
  test_metrics/
    test_qwen_prompt_bank.py      # Increment B
```

Use `importlib.resources` to locate packaged YAML data. Do not derive data
paths from the current working directory. Add the YAML files to package data in
`setup.py` or `pyproject.toml` if the current packaging configuration does not
already include them. VERSA supports Python 3.8, so use an API available there
(or a small version-gated compatibility helper); do not assume
`importlib.resources.files()` is universally available.

## Public API

The foundation API should be small and stable:

```python
from versa.prompt_bank import (
    get_protocol,
    list_protocols,
    render_protocol,
    validate_bank,
)

protocol = get_protocol("speech.emotion.v1")
rendered = render_protocol(
    protocol,
    mode="few_shot_text",
    context={},
)
```

Required behavior:

- `get_protocol(protocol_id)` returns an immutable or safely copied protocol.
  It raises `KeyError` with nearby ID suggestions for unknown IDs.
- `list_protocols(...)` filters by domain, task, input contract, response mode,
  model family, and stability status without importing a model backend.
  Its default view includes `experimental` and `stable`, excludes `draft` and
  `deprecated`, and returns records sorted by ID.
- `render_protocol(protocol_or_id, mode, context)` returns a `RenderedPrompt`
  object with `text`, `protocol_id`, `protocol_version`, `mode`, and optional
  `response_schema`, plus `protocol_digest`, `rendered_digest`, and
  `bank_schema_version`.
- `validate_bank()` validates every bundled record and reports all errors in one
  exception, including source filename and protocol ID.

Avoid exposing a public `Prompt` class. The public concept is a protocol, not a
bare string.

## Protocol Schema

Use YAML for authoring and a typed Python representation for runtime. A JSON
Schema file is not necessary in increment A; dataclass validation is sufficient
and easier to keep compatible with the project’s Python support. Add JSON
Schema only when external contributions or editors need it.

Every protocol requires these fields:

```yaml
id: speech.emotion.v1
version: 1
title: Dominant speech emotion
status: experimental           # draft | experimental | stable | deprecated
domain: speech                 # speech | general_audio | music | interaction
task: emotion_classification
description: Identify the dominant perceived emotion in spoken audio.
input_contract:
  audio_inputs: 1              # 1 | 2
  requires_target_instruction: false
  requires_reference_audio: false
  requires_reference_text: false
response_contract:
  mode: closed_label           # closed_label | integer | scalar | json | pairwise
  labels: [neutral, happy, sad, angry, fearful, surprised, disgusted, other]
  allow_abstain: true
  abstain_label: unclear
protocol:
  zero_shot: |
    Listen to the audio and identify its dominant perceived emotion.
    Base the answer on audible prosody and vocal delivery, not assumed identity
    or the literal meaning of words. Reply with exactly one label from:
    {labels}.
  few_shot_text:
    examples:
      - input: "Audible evidence: level pitch, steady pace, no strong affect."
        output: neutral
      - input: "Audible evidence: raised intensity, clipped delivery, tense tone."
        output: angry
    template: |
      Use the examples to calibrate the labels. Then evaluate the target audio.
      Reply with exactly one label from: {labels}.
provenance:
  sources:
    - https://arxiv.org/abs/2407.10759
  rationale: Adapted from the existing VERSA Qwen emotion prompt.
model_compatibility:
  - family: qwen2_audio
    status: expected
  - family: qwen2_5_omni
    status: expected
runner_compatibility:
  - runner: qwen2_audio
    status: planned
    modes: [zero_shot, few_shot_text]
  - runner: qwen_omni
    status: planned
    modes: [zero_shot, few_shot_text]
metric_links:
  - metric: emo_vad
    role: companion            # companion | diagnostic | calibration
```

Each content YAML file has one deterministic top-level shape:

```yaml
schema_version: 1
protocols:
  - id: speech.emotion.v1
    # ...complete protocol record...
```

`manifest.yaml` contains `schema_version` and an ordered `files` list. It is
the authoritative package index: validation rejects missing files, duplicate
entries, and unindexed protocol YAML files. Protocol ordering within files is
authoring-only; public listing uses a documented deterministic sort, by ID by
default.

Lifecycle status has a narrow meaning:

- `draft`: incomplete or under contributor review; excluded from default
  listings and runner integration.
- `experimental`: complete and renderable, but without sufficient
  human-grounded validation, and not yet executed by any metric. All eight
  initial protocols start here.
- `stable`: rendering is frozen within the version and validation evidence is
  documented. Promotion follows Phase 4 criteria.
- `deprecated`: still retrievable and renderable by exact ID for
  reproducibility, excluded from default listings, and carries a replacement
  ID or a rationale when no replacement exists.

Binary decisions use `closed_label` with exactly two non-abstention labels;
they do not need a separate response mode.

Constrained number and structured-score contracts use these shapes:

```yaml
# Integer-only response, for example speaker count.
response_contract:
  mode: integer
  minimum: 0
  maximum: 10
  allow_abstain: true
  abstain_label: unclear

# Structured score with one explicitly reportable field.
response_contract:
  mode: json
  fields:
    score: {type: integer, minimum: 1, maximum: 5, required: true}
    evidence: {type: array, items: string, required: true}
    confidence: {type: number, minimum: 0, maximum: 1, required: true}
    abstain: {type: boolean, required: true}
  primary_numeric_field: score
  output_key: prompt_generation_prompt_alignment_v1
```

Every validation rule for this schema is stated once, under
[Locked decisions](#locked-decisions-increment-a-reconciliation): ID format and
uniqueness, the mandatory fields each response mode implies, abstention,
`output_key` versioning, finite bounds, placeholders, unknown keys, demonstration
content, and the separation of `model_compatibility` from `runner_compatibility`.
The example above illustrates those rules; it does not restate them.

`few_shot_audio` is deliberately absent from this schema. Revisit it after a
licensed example package and a model-specific multi-audio contract are agreed.

## Protocol Taxonomy and Initial Contents

Do not begin with 45 protocols. Build a compact corpus that exercises every
important contract. The initial set should have eight protocols, each reviewed
as a protocol rather than a prompt string.

| ID | Domain | Mode | Why it belongs in v0 |
| --- | --- | --- | --- |
| `speech.emotion.v1` | speech | closed label, zero/few-shot | Migrates an existing Qwen default and validates labels/examples. |
| `speech.recording_quality.v1` | speech | closed label, zero/few-shot | Connects to DNSMOS, UTMOS, NISQA, and SRMR as companions. |
| `speech.speaker_count.v1` | speech | integer | Tests constrained integer output without a reference. |
| `speech.overlap.v1` | speech | closed label | Covers interaction signals with one audio input. |
| `audio.caption_accuracy.v1` | general audio | JSON | Covers caption judging and hallucination-aware response fields. |
| `generation.prompt_alignment.v1` | general audio | JSON with declared score field | Makes prompt-to-audio evaluation a first-class prompt-only task. |
| `generation.pairwise_alignment.v1` | general audio | pairwise | Covers two-audio inputs, order randomization, tie, and abstention. |
| `interaction.turn_taking.v1` | interaction | JSON with declared score field | A small bridge to voice-agent evaluation without building an agent stack. |

Recommended initial protocol files:

- `speech_understanding.yaml`: the four speech entries, including recording
  quality for the initial release.
- `generation_alignment.yaml`: caption accuracy and the two generation entries.
- `interaction.yaml`: turn taking.

Keep `audio_quality.yaml` until increment C unless it has enough protocols to
justify the extra file.

The initial input contracts are explicit:

- Emotion, recording quality, speaker count, overlap, caption accuracy, and
  turn taking consume one audio input. Caption accuracy additionally requires
  `reference_text` containing the caption being judged.
- Prompt alignment consumes one audio input and requires
  `target_instruction` containing the generation request.
- Pairwise alignment consumes two audio inputs and requires
  `target_instruction`; candidate display names are optional rendering context,
  while candidate order and seed belong to execution metadata.

## Rendering Contract

The renderer must build a plain text string that existing Qwen wrappers can
consume. It does not interpret audio or call models.

### Rendering modes

| Mode | v0 | Meaning |
| --- | --- | --- |
| `zero_shot` | yes | Render the zero-shot instruction and output contract. |
| `few_shot_text` | yes | Render the zero-shot body, then text-only demonstrations, then the closing target instruction. The rubric is identical to `zero_shot` by construction. |
| `pairwise` | yes | Render comparison instructions for audio A and audio B. |
| `rubric` | later in C | Render a fixed scalar rubric with anchors. |
| `dynamic_rubric` | deferred | Requires a planning stage and a judge pipeline. |
| `few_shot_audio` | deferred | Requires licensed audio assets and multi-audio rendering. |

For structured responses, append an explicit compact JSON template. Example:

```text
Return valid JSON only:
{"score": <integer 1-5>, "evidence": ["short audible observation"],
 "confidence": <number 0-1>, "abstain": <true|false>}
```

Avoid asking models to reveal hidden reasoning. Request short observable
evidence, which is useful for audit and can be independently checked.

### Canonical rendering rules

Rendered prompt text is executable evaluation logic and receives exact snapshot
coverage: tests compare the complete string for every bundled protocol and
supported mode, so an accidental prompt change is visible in review. The
rendering rules themselves — section order, whitespace, label and demonstration
ordering, contract-generated response templates, and verbatim context
substitution — are stated once under
[Locked decisions](#locked-decisions-increment-a-reconciliation).

## Qwen Integration: Increment B

The Qwen metric wrappers already accept `prompt`. Add the following optional
configuration fields to `Qwen2AudioMetric` and `QwenOmniMetric`:

```yaml
- name: qwen2_audio_speech_emotion
  prompt_id: speech.emotion.v1
  prompt_mode: few_shot_text
  prompt_context:
    target_instruction: "Classify the speaker's delivered affect."
```

Resolution precedence must be exact:

1. `prompt`: when the key is present and its value is not `None`, use the raw
   supplied prompt unchanged, including an empty string. Emit a warning only if
   a conflicting `prompt_id` is also present.
2. `prompt_id`: load and render the named protocol using `prompt_mode` and
   `prompt_context`.
3. Neither: use the existing `DEFAULT_PROMPTS[self.metric_name]` string
   unchanged.

Implementation responsibilities:

- Add a private `_resolve_prompt()` helper in each Qwen metric class or shared
  utility, returning `RenderedPrompt` or a minimal equivalent.
- Read Prompt Bank configuration and call `_resolve_prompt()` at the beginning
  of `_setup()`, before `qwen2_model_setup()` or `qwen_omni_model_setup()`.
  Invalid IDs, modes, context, and runner contracts must fail before checkpoint
  or processor initialization.
- Keep `DEFAULT_PROMPTS` in `qwen2_audio.py` untouched in increment B. This
  protects static metric discovery and all existing registry aliases.
- Permit any bank protocol whose `input_contract` and
  `runner_compatibility` entry are compatible with the wrapper. Qwen wrapper v0
  supports exactly one predicted audio input, no reference-audio transport,
  and required text only through `prompt_context`. Reject incompatible
  protocols with an actionable error; do not silently concatenate audio or
  omit required context.
- Keep current output keys (`qwen_<metric_name>` and
  `qwen_omni_<metric_name>`). Store protocol provenance separately only after
  the scorer has a stable metadata/output convention.
- Add `prompt_id`, `prompt_mode`, and `prompt_context` to config validation’s
  allowed parameters. Validate the bank ID and mode before model setup, so a
  typo does not download a checkpoint first.
- Use one shared resolver for both wrappers unless their rendering behavior
  actually differs. Warning category and message text are part of focused test
  coverage to prevent duplicate or inconsistent behavior.

Two new config examples should use existing named metrics, not create a generic
`prompt_bank` metric:

```yaml
# egs/separate_metrics/qwen2_audio_prompt_bank.yaml
- name: qwen2_audio_speech_emotion
  prompt_id: speech.emotion.v1
  prompt_mode: few_shot_text

- name: qwen2_audio_recording_quality
  prompt_id: speech.recording_quality.v1
  prompt_mode: zero_shot
```

This keeps VERSA’s model loading and registry logic intact while proving the
bank is useful.

## Audio judging as a supported workflow

Audio judging is a first-class VERSA evaluation workflow, not a side effect of a
few protocols. `audio.caption_accuracy.v1`, `generation.prompt_alignment.v1`,
`generation.pairwise_alignment.v1`, and `interaction.turn_taking.v1` are judge
protocols today, and most planned evaluation jobs depend on this path.

### The judged evaluation shape

```text
one or two audio inputs
  + optional reference text (caption, transcript, lyrics)
  + optional target instruction
  + the protocol's rubric, labels, and response contract
    -> judge model (local weights or hosted API)
      -> structured result: score or label or preference,
         short audible evidence, confidence, explicit abstention
```

The protocol owns the rubric and the response contract. The runner owns input
resolution and execution. The backend owns model access. None of these layers
may silently substitute for another: a rubric change is a protocol version, a
model change is provenance, and neither is a scoring decision.

Judges return text or structured payloads, never calibrated numbers. A judged
value becomes a reportable metric only through a declared
`primary_numeric_field` and its versioned `output_key`.

### Result envelope for a judged utterance

Every judged result records the following beside the score. This list is the
acceptance contract for the increment that implements parsing and reporting; it
is deliberately wider than the protocol digest, because the digest identifies
the protocol, not the evaluation.

| Group | Fields |
| --- | --- |
| Protocol identity | protocol ID, protocol version, protocol digest, bank schema version, prompt mode |
| Prompt identity | rendered prompt digest, and a digest of the resolved context mapping |
| Judge identity | model family, requested model ID, resolved model version reported by the provider, weights revision for local models, adapter name and version |
| Inference settings | temperature, top_p, max output tokens, seed when supported, whether provider-native structured output was used, retry count |
| Audio handling | input sample rate, channel handling, duration, any truncation or re-encoding the adapter applied |
| Outcome | status (scored, abstained, parse_failed, backend_error, invalid_input), raw response subject to the retention policy, parsed payload, normalization flag |

Two renderings of one protocol with different captions share a protocol digest
by design. The rendered prompt digest and the context digest are what make a
judged result reproducible, so neither is optional.

### Abstention, parse failure, and aggregation

Abstention is a valid measurement outcome and must never be scored:

- A response with `abstain: true`, or the abstention label of a label, integer,
  scalar, or preference contract, produces **no numeric value**. The sentinel
  score the prompts request alongside `abstain: true` exists only to keep the
  response shape fixed; parsing must discard it, and a reader must never see it
  as a genuine lowest-quality judgment.
- A response that fails contract validation produces no numeric value either.
  The raw text is retained under the retention policy and the row is counted as
  `parse_failed`.
- Aggregation reports explicit denominators per judged metric: requested,
  attempted, scored, abstained, parse_failed, and backend_error. The mean is
  taken over `scored` rows only.
- Abstention rate and parse-failure rate are reported beside every judged
  metric. A high abstention rate invalidates a comparison even when the mean of
  the remaining rows looks reasonable.
- No imputation. A failed or abstained utterance is never replaced by a default,
  a neutral value, or the mean.
- Preference protocols report win, loss, and tie counts with the candidate order
  and seed retained. Preference labels are not averaged into a scalar without a
  documented aggregation method and a position-bias check.

This reconciles with the shared result-summary policy: judged provenance fields
are metadata and must stay outside generic numeric means.

## Judge backends: local weights and hosted APIs

The same protocol must be usable against local weights and against hosted,
commercial APIs. A protocol is a measurement specification; where the model runs
is provenance and cost, not meaning. The bank itself stays provider-free: no
provider SDK, credential, endpoint, or HTTP client may be imported by
`versa.prompt_bank`, and the foundation's import-isolation test keeps that true.

### Adapter contract

A judge backend is a small adapter that lives with the metric runners, not in
the bank. It exposes four operations:

```text
describe()    -> backend identity: family, requested model ID, resolved model
                 version, weights revision or endpoint, adapter version,
                 provider SDK version
constraints() -> declared limits: max audio inputs, max audio seconds, max
                 request bytes, accepted containers and codecs, sample-rate and
                 channel policy, whether provider-native structured output is
                 supported, whether log-probabilities are available
judge(request)-> raw response text, optional provider-parsed structured payload,
                 provider metadata (resolved model version, response ID, finish
                 reason, usage counts), latency, and an error category on failure
close()       -> release sessions, files, or loaded weights
```

`request` carries the `RenderedPrompt` (text, protocol identity, response
schema), the ordered audio inputs, and the inference settings. An adapter never
edits prompt text, never chooses a protocol, and never decides a score.

Execution order is fixed, and every step before the last one is local:

1. Resolve the protocol, mode, and context; fail on any bank-level error.
2. Check runner and backend compatibility, including audio-input count.
3. Check the audio inputs against `constraints()`: duration, size, container,
   channels, sample rate.
4. Render the prompt and compute its digest.
5. Only then load weights or issue the request.

A configuration, context, or constraint failure must therefore never download a
checkpoint, upload audio, or spend a token. Failures map onto the existing
error taxonomy: configuration error, backend setup failure, invalid input,
inference error, or valid abstention.

### Provider dependencies and consent

- Every provider SDK is an optional extra in `pyproject.toml`
  (for example `versa[gemini]`), never a core dependency, and is imported inside
  the adapter at setup time.
- Credentials come from the environment or an explicit config field. They are
  never written to results, logs, or provenance records.
- A hosted adapter sends audio off the machine. It requires explicit opt-in in
  the metric configuration, and its documentation states what is transmitted and
  points to the provider's retention terms. Local-only users must be able to run
  every bundled protocol without any hosted adapter installed.
- Offline CI never contacts a provider: hosted adapters are exercised with
  recorded fixtures and mocks, and live runs are manual and recorded.

### Model and version identification

Hosted endpoints are mutable, so identity needs more than a model name:

- Record the requested model ID exactly as configured, and the resolved model
  version the provider reports in its response when one is available.
- For local weights, record the repository ID and the exact commit SHA.
- Record the adapter version and provider SDK version; a client-side change can
  alter request construction.
- `model_compatibility: tested` already requires a model ID, a revision, and a
  validation record. For a hosted model, the validation record must also carry
  the run date, because the endpoint behind the name can change without notice.
  Treat hosted evidence as dated, and re-verify before citing it in a release.

### Audio-input constraints

Adapters declare limits and validate against them before sending anything. The
values below come from each provider's published documentation at the time of
writing and must be re-read at implementation time rather than trusted here.

| Target | Availability | Documented audio handling |
| --- | --- | --- |
| Qwen2-Audio, Qwen2.5-Omni | Local weights, already in VERSA | Existing wrappers resample to the processor rate without channel mixing; one audio input. |
| [Gemini](https://ai.google.dev/gemini-api/docs/audio) | Hosted API | Documents `audio/wav`, `mp3`, `aiff`, `aac`, `ogg`, `flac`, `mpeg`, `m4a`, `l16`, `opus`, `alaw`, `mulaw`, and `webm`; up to 9.5 hours of audio per prompt; audio billed at 32 tokens per second; inline requests capped at 20 MB total, with the Files API above that; audio downsampled to 16 Kbps and multi-channel audio combined to a single channel; JSON-schema structured output supported; `gemini-3.8-flash` documented for audio understanding. |
| [Qwen3.5-Omni](https://qwen.ai/blog?id=qwen3.5-omni) | Offline API and Realtime API; Plus, Flash, and Light instruct variants | 256k-token context; more than 10 hours of audio input; ASR across 113 languages and dialects. |
| [Qwen3.8-Omni-Flash](https://qwen.ai/blog?id=qwen3.8-omni-flash) | Hosted API on the Qianwen AI Platform, plus a realtime API | 1M-token context; text, image, audio, and video input; substantially lower audio input pricing than Qwen3.5-Omni-Plus. |

Two rules follow from those differences:

- The adapter, not the protocol, owns transport-level audio handling. If a
  provider downmixes or re-encodes, that behavior is recorded in the result
  envelope so a channel-sensitive or bandwidth-sensitive protocol is not
  silently evaluated on altered audio.
- Batch evaluation uses the offline or standard request path. Realtime and
  streaming interfaces are out of scope for metric scoring; they belong to a
  separate interaction-evaluation track if one is ever pursued.

### Structured-response handling

- When `constraints().supports_structured_output` is true, the adapter may send
  the response contract as a provider-native schema. The rendered JSON template
  stays in the prompt regardless, so the two transports produce comparable text.
- Whether native structured output was used is recorded per result. Results
  produced with and without it are comparable only when that flag is reported.
- Both paths are validated by the same `validate_response` contract check. There
  is one validator, and it rejects non-finite numbers, repeated keys, undeclared
  fields, and out-of-range values.
- Permitted normalization before validation is limited to stripping surrounding
  whitespace and a fenced code block. Any normalization is flagged. Repairing
  malformed JSON, extracting a number from prose, or retrying until a response
  parses are all forbidden; a retry that changes sampling is a new observation
  and is recorded as such.

### Integration scope and acceptance criteria

Each backend lands as its own change. The acceptance criteria are identical, so
a new provider is a known quantity rather than a negotiation:

1. No new mandatory dependency; `versa.prompt_bank` import isolation unchanged;
   the core CI lane passes with the provider package absent.
2. Preflight proven: a test shows that an invalid protocol, mode, context, or
   audio input fails before any network call or weight load.
3. Identity proven: a mocked run asserts every field of the result envelope is
   populated, including the resolved model version when the provider returns it.
4. Contract coverage: at least one protocol per applicable response mode runs
   end to end against recorded fixtures, including a malformed response and an
   abstention, neither of which produces a score.
5. Repeatability characterized, not assumed: fixed decoding settings, and a
   recorded run-to-run agreement rate on a small fixed sample. Hosted judges are
   not deterministic and must not be described as such.
6. Consent and egress documented; opt-in configuration enforced.
7. Cost and latency recorded per utterance on the smoke set.
8. Cross-backend parity for at least one protocol against the existing local
   Qwen path on the same files, reporting agreement, abstention rate, and
   parse-failure rate. Parity evidence does not by itself promote a protocol to
   `stable`; that still needs human-grounded validation.

| Action | Scope | Depends on |
| --- | --- | --- |
| J1 | Judge backend adapter contract, result envelope, and conformance tests; wrap the existing local Qwen path as the reference adapter | P2, P3 |
| J2 | Gemini hosted adapter behind `versa[gemini]` | J1 |
| J3 | Qwen3.5-Omni hosted adapter (offline API), Plus/Flash/Light selectable | J1 |
| J4 | Qwen3.8-Omni-Flash hosted adapter | J1, J3 |
| J5 | Cross-backend protocol parity report on a fixed public sample | J2, J3 or J4 |

None of these blocks the bank or the local Qwen slice. A provider that becomes
unavailable is recorded as a release gap rather than worked around.

## Results and Parsing: Increment C

The current Qwen metrics return strings and VERSA summaries aggregate numeric
fields. Do not change that behavior in increment B. Increment C should add a
small opt-in structured result path.

Proposed result envelope for new structured protocols:

```json
{
  "raw_response": "{...}",
  "parsed": {"score": 4, "evidence": ["clear speech"], "confidence": 0.8},
  "parse_status": "ok",
  "protocol_id": "generation.prompt_alignment.v1",
  "protocol_version": 1,
  "protocol_digest": "sha256:...",
  "bank_schema_version": 1,
  "prompt_mode": "zero_shot",
  "model_id": "...",
  "model_revision": "..."
}
```

Rules:

- Parse only protocol-declared response modes. Raw text remains the source of
  truth when parsing fails.
- Apply only documented deterministic normalization. Trim surrounding Unicode
  whitespace. For closed labels, allow case-folding only when it maps to one
  unique declared label. Numeric responses must be a complete numeric token;
  do not extract a number from prose. JSON must be a single JSON object; do not
  repair malformed JSON or silently strip Markdown fences.
- Record applied normalization beside `parsed` whenever the normalized form
  differs from `raw_response`.
- Return `parse_status: invalid_json`, `invalid_label`, `out_of_range`, or
  `unsupported` instead of guessing a value.
- Flatten one declared numeric field into a stable score key only when the
  protocol explicitly names both `primary_numeric_field` and `output_key`, for
  example
  `prompt_generation_prompt_alignment_v1`.
- Validate bundled `output_key` values for global uniqueness and reserve the
  `prompt_` prefix for Prompt Bank structured results.
- Keep raw evidence and protocol metadata in JSONL result records. Numeric
  summary behavior stays unchanged for legacy strings.
- Add pairwise randomization metadata (`candidate_order`, `seed`) before using
  pairwise results in a benchmark report.

## Traditional Metric Linkage

Metric links are interpretive metadata, not an automatic blended score.

| Prompt protocol area | Existing metrics to link | Role |
| --- | --- | --- |
| Recording quality | `dns_*`, `utmos`, `nisqa`, `srmr`, `sigmos` | companion and calibration |
| Speech clarity | `stoi`, `pesq`, `whisper_wer`, `asr_match` | diagnostic |
| Speaker preservation | `speaker`, `speaker_similarity` where applicable | diagnostic |
| Prompt-to-audio alignment | `clap_score` | companion |
| Emotion / delivery | `emo_vad`, `speaking_rate` | companion |
| Overlap / turn-taking | no stable direct metric in current registry | prompt-only, future baseline |

Increment C reporting should group results by `metric_links` but never imply
that a companion metric validates the judge. Calibration is an explicit
experiment: collect human labels, calculate agreement and rank correlation,
then record the calibration dataset, judge model, and protocol version.

## Safety, Privacy, and Retention

Protocol wording and examples must not ask a judge to infer identity, protected
traits, medical conditions, or unverifiable intent from audio. When a task
legitimately concerns an acoustically observable property, request bounded
audible evidence and permit abstention. Contributor review should flag
demographic stereotypes and labels whose meaning is not operationally defined.

Raw audio, reference text, prompt context, and model evidence may contain
personal or confidential information. The Prompt Bank itself performs no
uploading or persistence. Increment C must make raw-response and evidence
retention explicit in the scorer/reporting design, preserve existing VERSA
defaults, and document redaction or omission options before claiming support
for sensitive evaluation workflows.

## Few-Shot Protocol Requirements

Text-only examples serve two jobs: output-format calibration and rubric
calibration. They must not pretend to be audio examples.

- Examples must describe audible evidence rather than demographic assumptions,
  medical diagnoses, or unverifiable intent.
- Include boundary cases when a label scale has ordered categories, and keep the
  demonstrations consistent with the rubric they illustrate. A demonstration
  that contradicts the label guide changes the scoring standard, not just the
  examples.
- Keep examples short enough that they do not dominate the target prompt.
- Benchmarking must compare zero-shot and few-shot separately. Never report a
  blended result as a single protocol score.

The mechanical rules — minimum count, text-only content, and validation of every
example output against the protocol's own contract — are enforced by the loader
and stated under
[Locked decisions](#locked-decisions-increment-a-reconciliation).

## Future Multi-Stage Judge Track

Do not implement this in the first Prompt Bank pull request, but preserve the
schema direction so it can be added without a redesign.

```text
text LLM planner
  -> chooses observable checks from the task and rubric
audio LLM evidence extractor
  -> answers targeted audio questions with short evidence and uncertainty
text LLM aggregator
  -> applies the rubric to evidence and emits structured judgment
```

Future pipeline record fields: `stages`, `stage_inputs`, `stage_outputs`,
`model_spec`, `failure_policy`, and `evidence_retention`. The first supported
pipeline should be caption/evidence then text judge, since it is inspectable and
can be tested with mock stage outputs. Dynamic rubric planning, tool calling,
and voice-agent trajectories remain subsequent work.

## Research and Protocol Review Checklist

The survey is a source for protocol requirements, not a collection of papers to
name in code. Capture the following in `docs/prompt_bank_survey.md` in a
separate documentation pass:

- Audio evaluation and judging: AudioBench, AIR-Bench, IFEval-Audio, ISA-Bench,
  SSEU-Bench, AnyAudio-Judge, ParaPairAudioBench, AQAScore, DEAF,
  HalluAudio/SVHalluc, and AudioTrust.
- Audio generation: DCASE Sound Scene Synthesis, EmergentTTS-Eval,
  SpeechJudge, SpeechQualityLLM, CMI-RewardBench, NVBench, and AudioCapBench.
- Voice-agent evaluation: VoiceAgentBench plus interaction/turn-taking work.
- Text evaluation transfer: G-Eval, Prometheus, MT-Bench/Chatbot Arena,
  JudgmentBench, Autorubric, prompt-sensitivity work, and judge-bias studies.

For each source, record publication date, task, input modalities, output form,
human reference signal, judge model, reported failure modes, transferable
protocol idea, and direct URL. Do not present unverified model/version claims as
facts; links should point to the paper or official project page.

## Test Plan

### Increment A tests — implemented

`test/test_prompt_bank.py` is the authoritative list. It covers package-resource
loading, manifest rules, every schema rule, strict context, response validation,
nested immutability, exact rendering snapshots with pinned digests, and
backend-free import isolation. Installed-wheel resource loading is checked by
`ci/check_installed_wheel.py` on every supported Python version, so a missing
package-data rule fails before release rather than after it.

### Increment B tests (`test/test_metrics/test_qwen_prompt_bank.py`)

- Mock Qwen model setup and base inference; do not download a checkpoint.
- Existing no-config path uses the exact legacy `DEFAULT_PROMPTS` string.
- Raw `prompt` overrides `prompt_id`, including when `prompt` is an empty
  string; a conflicting ID emits exactly one tested warning.
- `prompt_id` renders the requested zero-shot and few-shot protocol.
- Unknown IDs, missing context, invalid modes, and runner incompatibility fail
  before model setup.
- A two-audio pairwise protocol is rejected by the one-audio Qwen wrapper.
- Both Qwen metric families retain their existing output keys and aliases.
- Metric discovery continues to list all legacy Qwen metrics without importing
  YAML data or model backends.

### Increment C tests

- Valid and malformed JSON result parsing.
- Label, range, tie, and abstain validation.
- Raw response preservation on parser failure.
- Result provenance is present in JSONL records.
- Summary only aggregates explicitly declared numeric fields.
- Pairwise candidate order and seed are retained.
- Declared numeric output keys are unique and only the declared primary field
  is included in summaries.

Run focused checks first:

```bash
python -m pytest test/test_prompt_bank.py -q
python -m pytest test/test_metrics/test_qwen_prompt_bank.py -q
python -m pytest test/test_general.py -q
python -m black --check versa test
python -m flake8 versa test --count --select=E9,F63,F7,F82 --show-source --statistics
```

Run a manually controlled real-model smoke test only after the mocked tests
pass, using one short public or locally permitted audio file. Record the exact
model revision and output in the pull request, but do not make it a default CI
test.

## Implementation Sequence

### Phases 0 and 1: contract and foundation — done

Both phases are complete and shipped. The decisions Phase 0 had to reach are
recorded in [Locked decisions](#locked-decisions-increment-a-reconciliation),
which is the authoritative statement of the contract; the increment table in
[Release Shape](#release-shape) is the authoritative statement of scope.

Phase 1 delivered `versa/prompt_bank/{schema,loader,renderer}.py`, the packaged
manifest and three protocol files, `docs/prompt_bank.md`,
`test/test_prompt_bank.py` with `test/prompt_bank_snapshots.json`, package-data
rules in `pyproject.toml`, and installed-wheel coverage in
`ci/check_installed_wheel.py`. Its exit criterion was verified two ways: an
isolated interpreter renders a protocol while importing no Torch, Transformers,
or metric module, and the same rendering works from a wheel installed outside
the checkout.

### Phase 2: Integrate the existing Qwen wrappers

1. Add prompt resolution config fields to `Qwen2AudioMetric` and
   `QwenOmniMetric`.
2. Preserve the exact three-level precedence using `prompt is not None`: raw
   prompt, bank protocol, legacy default.
3. Resolve and validate before model setup so invalid protocol configuration
   never initializes or downloads a model.
4. Promote only the implemented one-audio Qwen runner entries from `planned`
   to `supported`; keep pairwise execution unsupported.
5. Add two config examples and mocked metric tests.
6. Add a short supported-metrics note linking the Qwen family to Prompt Bank.

Exit criterion: a `prompt_id` in an existing Qwen YAML config yields the
rendered text at model inference, while all existing Qwen config tests and
discovery commands still pass unchanged.

### Phase 3: Make results evaluable and auditable

1. Add opt-in response parsing for closed-label, integer, scalar, JSON, and
   pairwise contracts.
2. Decide the output envelope and reporting exposure with care to avoid breaking
   legacy JSONL consumers.
3. Store protocol digest, schema version, exact model ID/revision, prompt mode,
   and candidate-order metadata beside raw and parsed responses.
4. Add metric-link display in reports, clearly marked as companion context.
5. Add explicit retention/redaction controls for raw responses and evidence.
6. Add pairwise randomization and abstention handling before publishing any
   pairwise benchmark results.

Exit criterion: one JSON score protocol produces an auditable numeric field
plus raw evidence in JSONL, with focused parser and report tests.

### Phase 4: Research expansion, only after validation data exists

1. Build a human-annotated validation subset for each new protocol family.
2. Compare zero-shot, few-shot, and prompt variants separately.
3. Measure agreement, ranking correlation, positional bias, verbosity bias,
   language bias, and abstention behavior.
4. Add model-spec entries for each tested judge family and version.
5. Promote protocols from `experimental` to `stable` only with documented
   validation results.

Exit criterion: an expanded bank is governed by evidence, not just prompt
coverage.

## Non-Goals and Decision Gates

- Do not externalize all legacy Qwen prompts in the first change. The safe first
  integration is additive. Move legacy defaults into the bank only after the
  bank passes compatibility snapshots and discovery no longer scrapes Python
  source.
- Do not add a generic `prompt_bank` metric name in v0. It would force a new
  model-selection and output-key design before the bank demonstrates value.
- Do not expose audio few-shot examples until their license, storage format,
  resampling behavior, and multi-audio model rendering are specified.
- Do not blend judge and traditional metric values into a composite number
  without a named dataset, fitting method, and reproducible calibration record.
- Do not claim a prompt is a benchmark. A protocol becomes a validated metric
  only after human-grounded reliability analysis.

## Definition of Done by Pull Request

### Foundation pull request (Increment A) — met

Every criterion was met and is enforced by the tests and CI lanes listed under
[Test Plan](#test-plan). Do not restate it here; use the increment table in
[Release Shape](#release-shape) for scope and
[Locked decisions](#locked-decisions-increment-a-reconciliation) for the rules.

### Qwen vertical-slice pull request (Increment B)

The second implementation pull request is complete when all of the following
are true:

- Existing Qwen metric configs gain optional `prompt_id`, `prompt_mode`, and
  `prompt_context`, with `prompt is not None` precedence and legacy fallback
  intact.
- Two new Qwen configuration examples demonstrate bank use.
- Mocked tests cover both wrappers, early failure before model setup, runner
  compatibility, exact legacy fallback, empty-string override, and no
  regression in metric discovery.
- No legacy metric name, alias, output key, or default prompt text changes.

## Suggested Next Session Task

Phase 1 is complete. The next task is Phase 2: add `prompt_id`, `prompt_mode`,
and `prompt_context` to `Qwen2AudioMetric` and `QwenOmniMetric` with the
precedence and pre-setup validation recorded in
[Locked decisions](#locked-decisions-increment-a-reconciliation), promote the
one-audio Qwen runner entries from `planned` to `supported`, and add the two
configuration examples and mocked tests. Keep pairwise execution unsupported and
leave response parsing to Phase 3.
