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
from versa.prompt_bank.schema import PLACEHOLDER_ALLOWLIST, ErrorCollector

SNAPSHOT_PATH = Path(__file__).with_name("prompt_bank_snapshots.json")

# Digest fixtures guard the canonicalization algorithm against silent changes
# in the YAML or JSON libraries. Update them only with a protocol version bump.
EXPECTED_DIGESTS = {
    "audio.caption_accuracy.v1": (
        "c6307772d83c99c1e77c620be592efef337d6c12ec3644b1ae5360c60c2e66af"
    ),
    "generation.pairwise_alignment.v1": (
        "5594813455716d4fb08fd94da6bc04dc96da9b351ccbd3d02c1117df2c23e722"
    ),
    "generation.prompt_alignment.v1": (
        "5294f6cd5e4aa7cff7d61dcf978c86a25d9ca71154f541c8b41fdaddb6eb4edb"
    ),
    "interaction.turn_taking.v1": (
        "678164bf619f6ceec41c4f261f2e104f31c18ef655258bdd8feb9c274dd499aa"
    ),
    "speech.emotion.v1": (
        "2bc9c79b1824eb4acce7d653c9af3ff62a55308f0c4330c2c1cfa2411863bc69"
    ),
    "speech.overlap.v1": (
        "58e82a9d6e4076aa52a77c38ef1d32768f0aa52995e524685726a487a5a46749"
    ),
    "speech.recording_quality.v1": (
        "d7cea33d0e9872150985fce0bb64def2be35c694563dc380cbfc4b8bf1b56655"
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
        "prompt_audio_caption_accuracy",
        "prompt_generation_prompt_alignment",
        "prompt_interaction_turn_taking",
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


def test_required_context_must_appear_in_every_mode_body():
    """A declared text input that no body uses would be silently ignored."""
    protocol = _base_protocol()
    protocol["input_contract"] = {
        "audio_inputs": 1,
        "requires_reference_text": True,
    }
    protocol["protocol"] = {
        "zero_shot": "Judge the caption {reference_text}.",
        "few_shot_text": {
            "examples": [
                {"input": "Audible evidence: one bark.", "output": "yes"},
                {"input": "Audible evidence: silence.", "output": "no"},
            ],
            "template": "Judge the audio.",
        },
    }
    messages = _errors(protocol)
    assert any("body must use the required context" in entry for entry in messages)


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
        (
            [
                {"input": "Audible evidence: {labels}.", "output": "yes"},
                {"input": "Audible evidence: silence.", "output": "no"},
            ],
            "must not contain braces",
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
        "output_key": "prompt_example_score",
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
