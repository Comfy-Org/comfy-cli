from __future__ import annotations

import copy

import pytest

from comfy_cli import workflow_ops

SUBGRAPH_ID = "12345678-1234-4123-8123-123456789abc"
NESTED_ID = "abcdefab-cdef-4abc-8def-abcdefabcdef"
OTHER_NESTED_ID = "fedcbafe-dcba-4fed-8cba-fedcbafedcba"


def _definition(value: int = 1) -> dict:
    return {
        "id": SUBGRAPH_ID,
        "name": "One",
        "inputs": [],
        "outputs": [],
        "nodes": [{"id": 10, "type": "Inner", "widgets_values": [value]}],
        "links": [],
    }


def test_define_subgraph_emits_cmp_payload_and_inserts_definition():
    workflow = {"nodes": [], "links": []}
    definition = _definition()
    snapshot = copy.deepcopy(definition)

    result, op = workflow_ops.define_subgraph(workflow, definition, actor="agent", base_version=4)

    assert definition == snapshot
    assert op == {
        "op": "define_subgraph",
        "op_id": op["op_id"],
        "actor": "agent",
        "base_version": 4,
        "stamp": [4, "agent"],
        "subgraph_id": SUBGRAPH_ID,
        "subgraph_definition": definition,
    }
    assert result["definitions"]["subgraphs"] == [definition]


@pytest.mark.parametrize(
    ("definition", "match"),
    [
        ([], "JSON object"),
        ({"id": 7, "nodes": [], "links": []}, "non-empty string id"),
        ({"id": "not-a-uuid", "nodes": [], "links": []}, "valid UUID"),
        ({"id": SUBGRAPH_ID, "nodes": {}, "links": []}, "nodes and links must be arrays"),
        (
            {
                "id": SUBGRAPH_ID,
                "nodes": [],
                "links": [],
                "definitions": {"subgraphs": [{"id": "not-a-uuid", "nodes": [], "links": []}]},
            },
            "definitions.subgraphs\\[0\\].*valid UUID",
        ),
    ],
)
def test_define_subgraph_rejects_malformed_input_before_mutation(definition, match):
    workflow = {"nodes": [], "links": []}
    before = copy.deepcopy(workflow)

    with pytest.raises(ValueError, match=match):
        workflow_ops.define_subgraph(workflow, definition)

    assert workflow == before


def test_define_subgraph_can_assign_an_explicit_new_id():
    definition = _definition()
    definition.pop("id")

    _, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, definition, subgraph_id=SUBGRAPH_ID)

    assert op["subgraph_id"] == SUBGRAPH_ID
    assert op["subgraph_definition"]["id"] == SUBGRAPH_ID


def test_define_subgraph_rejects_existing_id_and_different_definition():
    workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": [_definition()]}}

    with pytest.raises(ValueError, match="already exists"):
        workflow_ops.define_subgraph(workflow, _definition(2))


def test_define_subgraph_preserves_nested_definitions_inside_single_parent_op():
    nested = {"id": NESTED_ID, "nodes": [], "links": []}
    definition = {**_definition(), "definitions": {"subgraphs": [nested]}}

    result, op = workflow_ops.define_subgraph({"nodes": [], "links": []}, definition)

    assert op["subgraph_definition"]["definitions"] == {"subgraphs": [nested]}
    assert result["definitions"]["subgraphs"] == [definition]
    assert op["op"] == "define_subgraph"


@pytest.mark.parametrize(
    "definition",
    [
        {
            **_definition(),
            "definitions": {"subgraphs": [{"id": SUBGRAPH_ID, "nodes": [], "links": []}]},
        },
        {
            **_definition(),
            "definitions": {
                "subgraphs": [
                    {
                        "id": NESTED_ID,
                        "nodes": [],
                        "links": [],
                        "definitions": {"subgraphs": [{"id": OTHER_NESTED_ID, "nodes": [], "links": []}]},
                    },
                    {"id": OTHER_NESTED_ID, "nodes": [], "links": []},
                ]
            },
        },
    ],
    ids=["ancestor", "across-branches"],
)
def test_define_subgraph_rejects_duplicate_ids_across_entire_definition_tree(definition):
    workflow = {"nodes": [], "links": []}
    before = copy.deepcopy(workflow)

    with pytest.raises(ValueError, match="duplicates subgraph definition id"):
        workflow_ops.define_subgraph(workflow, definition)

    assert workflow == before


def test_define_subgraph_rejects_id_already_nested_in_workflow():
    existing = {
        "id": OTHER_NESTED_ID,
        "nodes": [],
        "links": [],
        "definitions": {"subgraphs": [{"id": NESTED_ID, "nodes": [], "links": []}]},
    }
    workflow = {"nodes": [], "links": [], "definitions": {"subgraphs": [existing]}}
    before = copy.deepcopy(workflow)
    definition = {
        **_definition(),
        "definitions": {"subgraphs": [{"id": NESTED_ID, "nodes": [], "links": []}]},
    }

    with pytest.raises(ValueError, match="duplicates subgraph definition id"):
        workflow_ops.define_subgraph(workflow, definition)

    assert workflow == before


def test_apply_define_subgraph_rejects_malformed_nested_definition_atomically():
    workflow = {"nodes": [], "links": []}
    before = copy.deepcopy(workflow)
    definition = {
        **_definition(),
        "definitions": {"subgraphs": [{"id": NESTED_ID, "nodes": {}, "links": []}]},
    }
    op = {
        "op": "define_subgraph",
        "op_id": "a" * 32,
        "actor": "peer",
        "base_version": 0,
        "stamp": [0, "peer"],
        "subgraph_id": SUBGRAPH_ID,
        "subgraph_definition": definition,
    }

    with pytest.raises(ValueError, match="malformed_op:.*definitions.subgraphs\\[0\\].*nodes and links"):
        workflow_ops.apply_op(workflow, op, None)

    assert workflow == before


def test_apply_define_subgraph_exact_replay_is_idempotent_and_conflict_is_rejected():
    workflow = {"nodes": [], "links": []}
    _, op = workflow_ops.define_subgraph(workflow, _definition())
    before = copy.deepcopy(workflow)

    workflow_ops.apply_op(workflow, op, None)
    assert workflow == before

    conflicting = {**op, "op_id": "f" * 32, "subgraph_definition": _definition(2)}
    with pytest.raises(ValueError, match="definition_conflict:.*already exists with different content"):
        workflow_ops.apply_op(workflow, conflicting, None)
    assert workflow == before
