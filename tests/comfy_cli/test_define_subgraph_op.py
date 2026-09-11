from __future__ import annotations

import copy

import pytest

from comfy_cli import workflow_ops

SUBGRAPH_ID = "12345678-1234-4123-8123-123456789abc"


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
        ({"id": SUBGRAPH_ID, "nodes": {}, "links": []}, "nodes and links must be arrays"),
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


def test_apply_define_subgraph_exact_replay_is_idempotent_and_conflict_is_rejected():
    workflow = {"nodes": [], "links": []}
    _, op = workflow_ops.define_subgraph(workflow, _definition())
    before = copy.deepcopy(workflow)

    workflow_ops.apply_op(workflow, op, None)
    assert workflow == before

    conflicting = {**op, "op_id": "f" * 32, "subgraph_definition": _definition(2)}
    with pytest.raises(ValueError, match="already exists with different content"):
        workflow_ops.apply_op(workflow, conflicting, None)
