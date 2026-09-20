"""Foundation tests for the VERSA Prompt Bank.

The bundled bank is validated, filtered, and rendered without importing a model
backend or a metric module. Loading the same data from an installed wheel is
checked separately by ``ci/check_installed_wheel.py`` in the packaging lane.

Run ``python test/test_prompt_bank.py`` to rewrite the rendering snapshot after
an intentional protocol change; review the resulting diff as evaluation logic.
"""

import copy
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from versa.prompt_bank import (
    BankValidationError,
    ProtocolNotFoundError,
    RenderError,
    get_protocol,
    list_protocols,
    load_bank,
    render_protocol,
    validate_bank,
    validate_response,
)
from versa.prompt_bank.loader import MANIFEST_NAME, build_bank, _manifest_files
from versa.prompt_bank.schema import (
    PLACEHOLDER_ALLOWLIST,
    ErrorCollector,
    digest_text,
)

SNAPSHOT_PATH = Path(__file__).with_name("prompt_bank_snapshots.json")

# Digest fixtures guard the canonicalization algorithm against silent changes
# in the YAML or JSON libraries. Update them only with a protocol version bump.
EXPECTED_DIGESTS = {
    "audio.caption_accuracy.v1": (
        "916a8fbe1a6e2d078ac475f27e0c5cff603c2da64c83b6c0a031dbd4097ab7ca"
    ),
    "generation.pairwise_alignment.v1": (
        "5594813455716d4fb08fd94da6bc04dc96da9b351ccbd3d02c1117df2c23e722"
    ),
    "generation.prompt_alignment.v1": (
        "77184dfcf7c3e4366b9b6aca417a240fac0df25d0be0f21115ad9da83bb0352c"
    ),
    "interaction.turn_taking.v1": (
        "ed55824ef524e405968e18ea10aca4bfc3894692a3ab6acf681ade14d392827c"
    ),
    "speech.emotion.v1": (
        "bac1a2788d830e87fd4e8ef750c77d015ce90d7d731b43e7c9493c8779d2bf51"
    ),
    "speech.overlap.v1": (
        "12629c5e725c4c1ddf5d04168fba5ca712823c07b54c0a75971e959a4f5c3365"
    ),
    "speech.recording_quality.v1": (
        "c15d9e578652bd156b68d0507311902a6708470666a7d5ac0048980dce5b5d57"
    ),
    "speech.speaker_count.v1": (
        "9eef89948de3950681aeb5249a1be538fac6c6d157f723ac909ab10303a7c2fc"
    ),
}

# Context required to render the protocols that declare text inputs.
RENDER_CONTEXTS = {
    "audio.caption_accuracy.v1": {
        "reference_text": "A dog barks twice, then a door closes."
    },
    "generation.prompt_alignment.v1": {
        "target_instruction": "Generate a calm piano melody that fades out."
    },
    "generation.pairwise_alignment.v1": {
        "target_instruction": "Generate a calm piano melody that fades out."
    },
}


def _base_protocol():
    """Return a minimal valid protocol mapping that tests mutate locally."""
    return {
        "id": "speech.example.v1",
        "version": 1,
        "title": "Example protocol",
        "status": "experimental",
        "domain": "speech",
        "task": "example_task",
        "description": "A synthetic protocol used only by the foundation tests.",
        "input_contract": {"audio_inputs": 1},
        "response_contract": {
            "mode": "closed_label",
            "labels": ["yes", "no"],
            "allow_abstain": True,
            "abstain_label": "unclear",
        },
        "protocol": {"zero_shot": "Answer the question about the audio.\n"},
        "model_compatibility": [{"family": "qwen2_audio", "status": "expected"}],
        "runner_compatibility": [
            {"runner": "qwen2_audio", "status": "planned", "modes": ["zero_shot"]}
        ],
    }


def _documents(*protocols, **kwargs):
    """Build in-memory bank documents, mimicking an already-checked manifest."""
    source = kwargs.pop("source", "synthetic.yaml")
    schema_version = kwargs.pop("schema_version", 1)
    assert not kwargs, kwargs
    document = {"schema_version": schema_version, "protocols": list(protocols)}
    return [
        (MANIFEST_NAME, yaml.safe_dump({"schema_version": 1, "files": [source]})),
        (source, yaml.safe_dump(document, sort_keys=False)),
    ]


def _build(*protocols, **kwargs):
    """Validate synthetic protocol mappings and return the resulting bank."""
    return build_bank(_documents(*protocols, **kwargs))


def _errors(*protocols, **kwargs):
    """Return the validation messages raised by synthetic protocol mappings."""
    with pytest.raises(BankValidationError) as failure:
        _build(*protocols, **kwargs)
    return failure.value.messages


def _rendered_snapshots():
    """Render every bundled protocol and mode into a snapshot mapping."""
    snapshots = {}
    for protocol in load_bank().protocols:
        for mode in protocol.available_modes():
            rendered = render_protocol(
                protocol, mode=mode, context=RENDER_CONTEXTS.get(protocol.id)
            )
            snapshots["{}::{}".format(protocol.id, mode)] = {
                "digest": rendered.protocol_digest,
                "rendered_digest": rendered.rendered_digest,
                "bank_schema_version": rendered.bank_schema_version,
                "text": rendered.text,
            }
    return snapshots


def test_bundled_bank_validates_from_package_resources():
    """Every packaged file loads, and all eight initial protocols are present."""
    bank = validate_bank()
    assert bank.schema_version == 1
    assert sorted(bank.sources) == [
        "generation_alignment.yaml",
        "interaction.yaml",
        "speech_understanding.yaml",
    ]
    assert [protocol.id for protocol in bank.protocols] == sorted(EXPECTED_DIGESTS)
    assert {protocol.status for protocol in bank.protocols} == {"experimental"}


def test_protocol_identity_and_digest_fixtures():
    """IDs, trailing versions, and canonical digests stay byte-stable."""
    for protocol in load_bank().protocols:
        assert protocol.id.endswith(".v{}".format(protocol.version))
        assert protocol.digest == EXPECTED_DIGESTS[protocol.id]
        assert protocol.source_file.endswith(".yaml")


def test_declared_output_keys_are_unique_and_explicit():
    """Only JSON contracts declare a reportable field, and each key is unique."""
    keys = []
    for protocol in load_bank().protocols:
        contract = protocol.response_contract
        if contract.output_key is None:
            assert contract.primary_numeric_field is None
            continue
        assert contract.mode == "json"
        keys.append(contract.output_key)
    assert sorted(keys) == [
        "prompt_audio_caption_accuracy_v1",
        "prompt_generation_prompt_alignment_v1",
        "prompt_interaction_turn_taking_v1",
    ]


def test_get_protocol_reports_near_matches():
    """An unknown ID raises a KeyError subclass that suggests close names."""
    with pytest.raises(ProtocolNotFoundError) as failure:
        get_protocol("speech.emotion.v2")
    assert "speech.emotion.v1" in str(failure.value)
    assert isinstance(failure.value, KeyError)


@pytest.mark.parametrize(
    "filters,expected",
    [
        ({"domain": "speech"}, 4),
        ({"domain": "interaction"}, 1),
        ({"task": "emotion_classification"}, 1),
        ({"audio_inputs": 2}, 1),
        ({"response_mode": "json"}, 3),
        ({"response_mode": "integer"}, 1),
        ({"model_family": "qwen2_5_omni"}, 8),
        ({"model_family": "granite_speech"}, 0),
        ({"runner": "qwen2_audio"}, 0),
        ({"status": "draft"}, 0),
    ],
)
def test_list_protocols_filters(filters, expected):
    """Filtering is exact, additive, and independent of model backends."""
    selected = list_protocols(**filters)
    assert len(selected) == expected
    assert list(selected) == sorted(selected, key=lambda protocol: protocol.id)


def test_list_protocols_hides_draft_and_deprecated_by_default():
    """Draft and deprecated records are returned only when named explicitly."""
    draft = _base_protocol()
    draft["status"] = "draft"
    retired = copy.deepcopy(_base_protocol())
    retired["id"] = "speech.retired.v1"
    retired["status"] = "deprecated"
    bank = _build(draft, retired)
    assert len(bank.protocols) == 2
    listed = [
        protocol
        for protocol in bank.protocols
        if protocol.status in ("experimental", "stable")
    ]
    assert listed == []
    assert list_protocols(status=["draft", "deprecated"]) == ()


def test_list_protocols_rejects_an_unknown_status():
    """A misspelled lifecycle status fails instead of returning nothing."""
    with pytest.raises(ValueError, match="unknown lifecycle status"):
        list_protocols(status="retired")


def test_rendering_matches_committed_snapshots():
    """Rendered text for every protocol and mode matches the reviewed snapshot."""
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    actual = _rendered_snapshots()
    assert sorted(actual) == sorted(expected)
    for name in sorted(expected):
        assert actual[name] == expected[name], name


def test_rendered_text_is_canonical():
    """Every rendering ends with one newline and has no trailing whitespace."""
    for name, snapshot in _rendered_snapshots().items():
        text = snapshot["text"]
        assert text.endswith("\n") and not text.endswith("\n\n"), name
        assert "\n\n\n" not in text, name
        assert all(line == line.rstrip() for line in text.splitlines()), name
        for placeholder in PLACEHOLDER_ALLOWLIST:
            assert "{{{}}}".format(placeholder) not in text, name


def test_optional_context_defaults_and_overrides():
    """Candidate names default deterministically and accept caller overrides."""
    instruction = {"target_instruction": "Generate rain on a tin roof."}
    default = render_protocol(
        "generation.pairwise_alignment.v1", mode="pairwise", context=instruction
    )
    assert "Candidate A" in default.text and "Candidate B" in default.text
    named = render_protocol(
        "generation.pairwise_alignment.v1",
        mode="pairwise",
        context=dict(instruction, candidate_a_name="baseline"),
    )
    assert "baseline" in named.text and "Candidate B" in named.text
    assert named.protocol_digest == default.protocol_digest


def test_context_values_are_not_templates():
    """A brace in a context value is inserted verbatim, never re-interpreted."""
    rendered = render_protocol(
        "generation.prompt_alignment.v1",
        context={"target_instruction": "Say {labels} out loud."},
    )
    assert "Say {labels} out loud." in rendered.text


@pytest.mark.parametrize(
    "protocol_id,mode,context,message",
    [
        ("speech.emotion.v1", "rubric", None, "unknown render mode"),
        ("speech.speaker_count.v1", "few_shot_text", None, "does not define mode"),
        ("audio.caption_accuracy.v1", "zero_shot", None, "is required to render"),
        (
            "audio.caption_accuracy.v1",
            "zero_shot",
            {"reference_text": "a caption", "extra": "x"},
            "unknown context keys",
        ),
        (
            "audio.caption_accuracy.v1",
            "zero_shot",
            {"reference_text": "   "},
            "must be a non-empty string",
        ),
        ("speech.emotion.v1", "zero_shot", {"labels": "a, b"}, "unknown context keys"),
    ],
)
def test_render_rejects_invalid_requests(protocol_id, mode, context, message):
    """Mode and context problems fail loudly before any text is produced."""
    with pytest.raises(RenderError, match=message):
        render_protocol(protocol_id, mode=mode, context=context)


def test_render_requires_a_protocol_or_id():
    """A protocol object or a versioned ID string is required."""
    with pytest.raises(RenderError, match="Protocol record or a protocol ID"):
        render_protocol(42)


def test_returned_protocols_cannot_mutate_bank_state():
    """Protocol records and their nested containers are immutable."""
    protocol = get_protocol("speech.emotion.v1")
    with pytest.raises(dataclasses.FrozenInstanceError):
        protocol.status = "stable"
    with pytest.raises(AttributeError):
        protocol.response_contract.labels.append("excited")
    with pytest.raises(TypeError):
        protocol.modes.few_shot_text.examples[0] = None
    with pytest.raises(TypeError):
        load_bank().by_id["speech.emotion.v1"] = None
    assert get_protocol("speech.emotion.v1") is protocol
    assert get_protocol("speech.emotion.v1").status == "experimental"


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"id": "Speech.Emotion.V1"}, "id must match"),
        ({"id": "speech.example.v2"}, "does not match the trailing ID version"),
        ({"status": "released"}, "status must be one of"),
        ({"domain": "podcast"}, "domain must be one of"),
        ({"title": ""}, "title must be a non-empty string"),
        ({"notes": "extra"}, "unknown protocol fields"),
        ({"input_contract": {"audio_inputs": 3}}, "audio_inputs must be 1 or 2"),
        (
            {"input_contract": {"audio_inputs": 1, "requires_caption": True}},
            "unknown input_contract fields",
        ),
        (
            {"response_contract": {"mode": "closed_label", "labels": ["yes"]}},
            "labels must be a list of at least 2 entries",
        ),
        (
            {
                "response_contract": {
                    "mode": "closed_label",
                    "labels": ["yes", "yes"],
                }
            },
            "labels entries must be unique",
        ),
        (
            {
                "response_contract": {
                    "mode": "closed_label",
                    "labels": ["yes", "no"],
                    "minimum": 1,
                }
            },
            "not valid for response mode",
        ),
        (
            {
                "response_contract": {
                    "mode": "closed_label",
                    "labels": ["yes", "no"],
                    "allow_abstain": True,
                    "abstain_label": "yes",
                }
            },
            "must differ from the response labels",
        ),
        (
            {
                "response_contract": {
                    "mode": "closed_label",
                    "labels": ["yes", "no"],
                    "abstain_label": "unclear",
                }
            },
            "abstain_label requires allow_abstain",
        ),
        (
            {"response_contract": {"mode": "integer", "minimum": 5, "maximum": 1}},
            "minimum must be < maximum",
        ),
        (
            {"response_contract": {"mode": "scalar", "minimum": 1, "maximum": 5}},
            "at least two anchors",
        ),
        (
            {
                "response_contract": {
                    "mode": "json",
                    "fields": {"score": {"type": "integer"}},
                }
            },
            "required boolean 'abstain' field",
        ),
        (
            {
                "response_contract": {
                    "mode": "json",
                    "fields": {
                        "note": {"type": "string"},
                        "abstain": {"type": "boolean"},
                    },
                    "primary_numeric_field": "note",
                    "output_key": "prompt_note",
                }
            },
            "primary_numeric_field must be numeric",
        ),
        (
            {
                "response_contract": {
                    "mode": "json",
                    "fields": {
                        "score": {"type": "integer", "minimum": 1, "maximum": 5},
                        "abstain": {"type": "boolean"},
                    },
                    "primary_numeric_field": "score",
                }
            },
            "output_key must be a lowercase identifier",
        ),
        (
            {"protocol": {"few_shot_text": {"examples": [], "template": "x"}}},
            "zero_shot must be a non-empty string",
        ),
        (
            {"protocol": {"zero_shot": "Answer.", "rubric": "later"}},
            "unknown protocol modes",
        ),
        (
            {"protocol": {"zero_shot": "Answer about {speaker_age}."}},
            "is not in the allow-list",
        ),
        (
            {"protocol": {"zero_shot": "Answer using {labels.upper}."}},
            "must be a simple name",
        ),
        (
            {"protocol": {"zero_shot": "Answer using {labels!r}."}},
            "must be a simple name",
        ),
        (
            {"protocol": {"zero_shot": 'Return {"score": 1}.'}},
            "must be a simple name",
        ),
        (
            {"protocol": {"zero_shot": "Answer briefly } and stop."}},
            "unescaped brace",
        ),
        (
            {"protocol": {"zero_shot": "Answer about {target_instruction}."}},
            "requires input_contract.requires_target_instruction",
        ),
        (
            {"protocol": {"zero_shot": "Compare {candidate_a_name}."}},
            "requires two audio inputs",
        ),
        (
            {"model_compatibility": [{"family": "qwen2_audio", "status": "tested"}]},
            "needs model_id, revision, and validation",
        ),
        (
            {
                "runner_compatibility": [
                    {
                        "runner": "qwen2_audio",
                        "status": "supported",
                        "modes": ["few_shot_text"],
                    }
                ]
            },
            "declares modes outside",
        ),
        (
            {"metric_links": [{"metric": "dnsmos", "role": "reference"}]},
            "role must be one of",
        ),
    ],
)
def test_invalid_protocol_records_are_rejected(mutation, message):
    """Each schema rule fails with a message naming the source and protocol."""
    protocol = _base_protocol()
    protocol.update(mutation)
    messages = _errors(protocol)
    assert any(message in entry for entry in messages), messages
    assert all(entry.startswith("synthetic.yaml") for entry in messages), messages


def test_validation_reports_every_error_at_once():
    """One run reports all problems instead of stopping at the first."""
    protocol = _base_protocol()
    protocol["status"] = "released"
    protocol["domain"] = "podcast"
    protocol["title"] = ""
    messages = _errors(protocol)
    assert len(messages) == 3


def test_required_context_must_appear_in_every_render_mode():
    """A declared text input that no mode renders would be silently ignored."""
    protocol = _base_protocol()
    protocol["input_contract"] = {
        "audio_inputs": 1,
        "requires_reference_text": True,
    }
    protocol["protocol"] = {"zero_shot": "Judge the audio."}
    messages = _errors(protocol)
    assert any("must use the required context" in entry for entry in messages)


def test_few_shot_rendering_keeps_the_zero_shot_rubric():
    """Both modes judge against the same rubric, so only demonstrations differ."""
    zero_shot = render_protocol("speech.recording_quality.v1", mode="zero_shot")
    few_shot = render_protocol("speech.recording_quality.v1", mode="few_shot_text")
    guide = "- fair: audible defects that do not prevent comfortable listening"
    assert guide in zero_shot.text and guide in few_shot.text
    assert few_shot.text.startswith(zero_shot.text.split("\n\n")[0])
    assert "Examples:" in few_shot.text
    for label in get_protocol("speech.recording_quality.v1").response_contract.labels:
        assert "- {}:".format(label) in few_shot.text
    assert zero_shot.protocol_digest == few_shot.protocol_digest
    assert zero_shot.rendered_digest != few_shot.rendered_digest


def test_rendered_digest_separates_identical_protocols():
    """The protocol digest is shared; the rendered digest identifies the text."""
    first = render_protocol(
        "audio.caption_accuracy.v1", context={"reference_text": "A dog barks."}
    )
    second = render_protocol(
        "audio.caption_accuracy.v1", context={"reference_text": "A door closes."}
    )
    assert first.protocol_digest == second.protocol_digest
    assert first.rendered_digest != second.rendered_digest
    assert first.rendered_digest == digest_text(first.text)


def test_pairwise_contracts_require_two_audio_inputs_and_a_body():
    """Pairwise records must declare two inputs, and two inputs imply pairwise."""
    single = _base_protocol()
    single["response_contract"] = {
        "mode": "pairwise",
        "preference_labels": ["a", "b", "tie"],
    }
    messages = _errors(single)
    assert any("require two audio inputs" in entry for entry in messages)
    assert any("require a pairwise body" in entry for entry in messages)

    two_audio = _base_protocol()
    two_audio["input_contract"] = {"audio_inputs": 2}
    messages = _errors(two_audio)
    assert any("must use the pairwise contract" in entry for entry in messages)


@pytest.mark.parametrize(
    "examples,message",
    [
        (
            [{"input": "Audible evidence: one bark.", "output": "yes"}],
            "at least 2 examples",
        ),
        (
            [
                {"input": "Audible evidence: one bark.", "output": "maybe"},
                {"input": "Audible evidence: silence.", "output": "no"},
            ],
            "must be exactly one of",
        ),
        (
            [
                {"input": "Listen to sample_01.wav.", "output": "yes"},
                {"input": "Audible evidence: silence.", "output": "no"},
            ],
            "must not reference audio files",
        ),
    ],
)
def test_few_shot_examples_are_validated(examples, message):
    """Demonstrations stay text-only and every output satisfies the contract."""
    protocol = _base_protocol()
    protocol["protocol"]["few_shot_text"] = {
        "examples": examples,
        "template": "Answer the question about the audio.",
    }
    messages = _errors(protocol)
    assert any(message in entry for entry in messages), messages


def _json_protocol():
    """Return a synthetic JSON-contract protocol with few-shot demonstrations."""
    protocol = _base_protocol()
    protocol["response_contract"] = {
        "mode": "json",
        "fields": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "note": {"type": "string", "required": False},
            "abstain": {"type": "boolean"},
        },
        "primary_numeric_field": "score",
        "output_key": "prompt_example_score_v1",
    }
    protocol["protocol"] = {
        "zero_shot": "Rate the audio.",
        "few_shot_text": {
            "examples": [
                {
                    "input": "Audible evidence: clean speech.",
                    "output": '{"score": 5, "abstain": false}',
                },
                {
                    "input": "Audible evidence: heavy distortion.",
                    "output": '{"score": 2, "note": "clipping", "abstain": false}',
                },
            ],
            "template": "Rate the audio you are given on the same scale.",
        },
    }
    return protocol


def test_json_protocols_support_few_shot_demonstrations():
    """JSON example outputs are literal text, so their braces are not templates."""
    bank = _build(_json_protocol())
    protocol = bank.protocols[0]
    assert protocol.available_modes() == ("zero_shot", "few_shot_text")
    rendered = render_protocol(protocol, mode="few_shot_text")
    assert 'Output: {"score": 5, "abstain": false}' in rendered.text
    assert '{"score": <integer 1-5>, "note": <string>, "abstain": <true|false>}' in (
        rendered.text
    )


def test_json_instructions_describe_the_validated_contract():
    """One-sided bounds and optional keys are stated, not silently dropped."""
    protocol = _json_protocol()
    protocol["protocol"] = {"zero_shot": "Rate the audio."}
    protocol["response_contract"] = {
        "mode": "json",
        "fields": {
            "score": {"type": "integer", "minimum": 0},
            "ratio": {"type": "number", "maximum": 1},
            "free": {"type": "number", "required": False},
            "abstain": {"type": "boolean"},
        },
    }
    rendered = render_protocol(_build(protocol).protocols[0])
    assert '"score": <integer, at least 0>' in rendered.text
    assert '"ratio": <number, at most 1>' in rendered.text
    assert '"free": <number>' in rendered.text
    assert 'Every key is required except "free", which may be omitted.' in rendered.text
    contract = _build(protocol).protocols[0].response_contract
    assert validate_response('{"score": -1, "ratio": 0.5, "abstain": false}', contract)
    assert (
        validate_response('{"score": 3, "ratio": 0.5, "abstain": false}', contract)
        == []
    )


def test_duplicate_ids_and_output_keys_are_rejected():
    """Global uniqueness covers protocol IDs and declared output keys."""
    first = _base_protocol()
    second = copy.deepcopy(first)
    messages = _errors(first, second)
    assert any("duplicates the ID" in entry for entry in messages)

    scored = _base_protocol()
    scored["response_contract"] = {
        "mode": "json",
        "fields": {
            "score": {"type": "integer", "minimum": 1, "maximum": 5},
            "abstain": {"type": "boolean"},
        },
        "primary_numeric_field": "score",
        "output_key": "prompt_example_score_v1",
    }
    other = copy.deepcopy(scored)
    other["id"] = "speech.other_example.v1"
    messages = _errors(scored, other)
    assert any("is already declared by" in entry for entry in messages)


def test_malformed_documents_are_rejected():
    """Broken YAML and unexpected document shapes fail with the file name."""
    with pytest.raises(BankValidationError, match="is not valid YAML"):
        build_bank([(MANIFEST_NAME, ""), ("broken.yaml", "protocols: [\n")])
    with pytest.raises(BankValidationError, match="needs exactly schema_version"):
        build_bank([(MANIFEST_NAME, ""), ("odd.yaml", "protocols: []\n")])
    with pytest.raises(BankValidationError, match="schema_version must be 1"):
        _build(_base_protocol(), schema_version=2)
    with pytest.raises(BankValidationError, match="protocols must be a non-empty list"):
        _build()


@pytest.mark.parametrize(
    "manifest,present,message",
    [
        ({"schema_version": 1}, {"a.yaml"}, "needs schema_version and files"),
        (
            {"schema_version": 2, "files": ["a.yaml"]},
            {"a.yaml"},
            "schema_version must be 1",
        ),
        (
            {"schema_version": 1, "files": ["a.yaml", "a.yaml"]},
            {"a.yaml"},
            "must not repeat an entry",
        ),
        (
            {"schema_version": 1, "files": ["a.yaml"]},
            {"a.yaml", "b.yaml"},
            "protocol files are not indexed",
        ),
        (
            {"schema_version": 1, "files": ["a.yaml", "b.yaml"]},
            {"a.yaml"},
            "indexed files are missing",
        ),
        (
            {"schema_version": 1, "files": [MANIFEST_NAME]},
            {MANIFEST_NAME},
            "must not index itself",
        ),
    ],
)
def test_manifest_is_the_authoritative_index(manifest, present, message):
    """The manifest must match the packaged files exactly."""
    errors = ErrorCollector()
    _manifest_files(manifest, present, errors)
    assert any(message in entry for entry in errors.messages), errors.messages


@pytest.mark.parametrize(
    "protocol_id,response,expected",
    [
        ("speech.emotion.v1", "angry", True),
        ("speech.emotion.v1", "unclear", True),
        ("speech.emotion.v1", "The speaker sounds angry.", False),
        ("speech.emotion.v1", "excited", False),
        ("speech.speaker_count.v1", "3", True),
        ("speech.speaker_count.v1", "11", False),
        ("speech.speaker_count.v1", "about three", False),
        ("generation.pairwise_alignment.v1", "tie", True),
        ("generation.pairwise_alignment.v1", "candidate a", False),
        (
            "interaction.turn_taking.v1",
            '{"score": 4, "evidence": ["one pause"], "disruptions": [],'
            ' "confidence": 0.7, "abstain": false}',
            True,
        ),
        ("interaction.turn_taking.v1", '{"score": 4}', False),
        (
            "interaction.turn_taking.v1",
            '{"score": 9, "evidence": [], "disruptions": [],'
            ' "confidence": 0.7, "abstain": false}',
            False,
        ),
        ("interaction.turn_taking.v1", "Score: 4 out of 5", False),
    ],
)
def test_validate_response_never_extracts_values_from_prose(
    protocol_id, response, expected
):
    """Responses are accepted only when they match the declared contract exactly."""
    contract = get_protocol(protocol_id).response_contract
    assert (validate_response(response, contract) == []) is expected


@pytest.mark.parametrize(
    "response,message",
    [
        (
            '{"score": 4, "evidence": [], "disruptions": [],'
            ' "confidence": NaN, "abstain": false}',
            "must not contain NaN",
        ),
        (
            '{"score": 4, "evidence": [], "disruptions": [],'
            ' "confidence": Infinity, "abstain": false}',
            "must not contain Infinity",
        ),
        (
            '{"score": 4, "evidence": [], "disruptions": [],'
            ' "confidence": 1e400, "abstain": false}',
            "must be a finite number",
        ),
        (
            '{"score": 1, "score": 5, "evidence": [], "disruptions": [],'
            ' "confidence": 0.5, "abstain": false}',
            "must not repeat the key 'score'",
        ),
    ],
)
def test_json_responses_reject_non_finite_and_duplicate_keys(response, message):
    """An ambiguous or non-finite response is rejected, never silently resolved."""
    contract = get_protocol("interaction.turn_taking.v1").response_contract
    assert any(
        message in entry for entry in validate_response(response, contract)
    ), validate_response(response, contract)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_scalar_and_integer_responses_reject_non_finite_text(value):
    """Python's float parser accepts nan and inf; the contract must not."""
    integer_contract = get_protocol("speech.speaker_count.v1").response_contract
    assert validate_response(value, integer_contract)
    protocol = _base_protocol()
    protocol["response_contract"] = {
        "mode": "scalar",
        "minimum": 1,
        "maximum": 5,
        "anchors": [
            {"value": 1, "description": "lowest"},
            {"value": 5, "description": "highest"},
        ],
    }
    protocol["protocol"] = {"zero_shot": "Rate the audio."}
    contract = _build(protocol).protocols[0].response_contract
    assert validate_response(value, contract)


@pytest.mark.parametrize(
    "contract,message",
    [
        (
            {"mode": "integer", "minimum": ".nan", "maximum": 10},
            "must be a finite integer",
        ),
        (
            {
                "mode": "scalar",
                "minimum": 1,
                "maximum": ".inf",
                "anchors": [
                    {"value": 1, "description": "lowest"},
                    {"value": 2, "description": "higher"},
                ],
            },
            "must be a finite number",
        ),
        (
            {
                "mode": "json",
                "fields": {
                    "score": {"type": "integer", "minimum": ".nan", "maximum": 5},
                    "abstain": {"type": "boolean"},
                },
            },
            "must be a finite number",
        ),
    ],
)
def test_non_finite_schema_bounds_are_rejected(contract, message):
    """A NaN bound would make every range comparison silently pass."""
    protocol = _base_protocol()
    protocol["response_contract"] = yaml.safe_load(yaml.safe_dump(contract))
    protocol["protocol"] = {"zero_shot": "Rate the audio."}
    messages = _errors(protocol)
    assert any(message in entry for entry in messages), messages


def test_duplicate_yaml_keys_are_rejected():
    """A repeated key would discard evaluation logic before validation sees it."""
    document = """
schema_version: 1
protocols:
  - id: speech.example.v1
    version: 999
    version: 1
    title: Example protocol
    status: experimental
    domain: speech
    task: example_task
    description: A synthetic protocol.
    input_contract: {audio_inputs: 1}
    response_contract: {mode: closed_label, labels: [yes, no]}
    protocol: {zero_shot: "Answer the question about the audio."}
    model_compatibility: [{family: qwen2_audio, status: expected}]
    runner_compatibility: [{runner: qwen2_audio, status: planned, modes: [zero_shot]}]
"""
    with pytest.raises(BankValidationError, match="duplicate key"):
        build_bank([(MANIFEST_NAME, ""), ("duplicated.yaml", document)])


@pytest.mark.parametrize(
    "mutation",
    [
        {
            "response_contract": {
                "mode": "json",
                "fields": {
                    "score": {"type": "integer", "minimum": 1, "maximum": 5},
                    "abstain": {"type": "boolean"},
                },
                "primary_numeric_field": ["score"],
                "output_key": "prompt_example_score_v1",
            }
        },
        {
            "runner_compatibility": [
                {"runner": "qwen2_audio", "status": "planned", "modes": [[]]}
            ]
        },
        {"status": ["experimental"]},
        {"metric_links": [{"metric": ["dnsmos"], "role": "companion"}]},
        {"model_compatibility": [{"family": {"a": 1}, "status": "expected"}]},
    ],
)
def test_malformed_nested_values_are_reported_not_raised(mutation):
    """Unhashable and mistyped values must not escape the validation report."""
    protocol = _base_protocol()
    protocol.update(mutation)
    messages = _errors(protocol)
    assert all(entry.startswith("synthetic.yaml") for entry in messages), messages


def test_output_keys_are_versioned_so_protocol_versions_coexist():
    """A deprecated .v1 and its .v2 successor report under distinct keys."""
    first = _json_protocol()
    first["id"] = "speech.scored.v1"
    first["status"] = "deprecated"
    second = copy.deepcopy(first)
    second["id"] = "speech.scored.v2"
    second["version"] = 2
    second["status"] = "experimental"
    second["response_contract"]["output_key"] = "prompt_example_score_v2"
    bank = _build(first, second)
    assert [protocol.id for protocol in bank.protocols] == [
        "speech.scored.v1",
        "speech.scored.v2",
    ]
    assert [protocol.response_contract.output_key for protocol in bank.protocols] == [
        "prompt_example_score_v1",
        "prompt_example_score_v2",
    ]
    assert bank.get("speech.scored.v1").digest != bank.get("speech.scored.v2").digest

    mismatched = copy.deepcopy(second)
    mismatched["response_contract"]["output_key"] = "prompt_example_score_v1"
    messages = _errors(mismatched)
    assert any("must end in '_v2'" in entry for entry in messages), messages


def test_bank_loads_without_model_or_metric_imports():
    """A fresh isolated interpreter renders a protocol with no heavy imports."""
    root = str(Path(__file__).resolve().parents[1])
    code = (
        "import json, sys;"
        "sys.path.insert(0, {root!r});"
        "import versa.prompt_bank as bank;"
        "rendered = bank.render_protocol('speech.emotion.v1');"
        "prefixes = ('torch', 'transformers', 'librosa',"
        " 'versa.utterance_metrics', 'versa.corpus_metrics', 'versa.sequence_metrics');"
        "print(json.dumps({{'digest': rendered.protocol_digest,"
        " 'loaded': sorted(n for n in sys.modules if n.startswith(prefixes)),"
        " 'count': len(bank.list_protocols())}}))"
    ).format(root=root)
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    payload = json.loads(result.stdout)
    assert payload["loaded"] == []
    assert payload["count"] == len(EXPECTED_DIGESTS)
    assert payload["digest"] == EXPECTED_DIGESTS["speech.emotion.v1"]


if __name__ == "__main__":
    SNAPSHOT_PATH.write_text(
        json.dumps(_rendered_snapshots(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("wrote {}".format(SNAPSHOT_PATH))
