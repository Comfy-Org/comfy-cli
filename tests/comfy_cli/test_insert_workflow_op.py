from __future__ import annotations

import copy

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph


def _template() -> dict:
    return {
        "nodes": [
            {"id": 100, "type": "Src", "outputs": [{"name": "o", "links": [200]}]},
            {"id": 101, "type": "def-2", "inputs": [{"name": "a", "link": 200}]},
        ],
        "links": [[200, 100, 0, 101, 0, "X"]],
        "definitions": {"subgraphs": [{"id": "def-2", "nodes": [{"id": 5, "type": "Inner"}], "links": []}]},
    }


def test_remap_workflow_ids_matches_cmp_fixture_and_does_not_mutate_input():
    template = _template()
    snapshot = copy.deepcopy(template)

    out = workflow_ops.remap_workflow_ids(template, node_id_start=8, link_id_start=8)

    assert template == snapshot
    assert [node["id"] for node in out["nodes"]] == [8, 9]
    assert out["links"] == [[8, 8, 0, 9, 0, "X"]]
    assert out["nodes"][0]["outputs"][0]["links"] == [8]
    assert out["nodes"][1]["inputs"] == [{"name": "a", "link": 8}]
    assert out["definitions"]["subgraphs"][0]["nodes"][0]["id"] == 5


def test_insert_workflow_emits_one_op_with_nested_definitions_preserved():
    live = {"last_node_id": 7, "last_link_id": 7, "nodes": [], "links": []}

    result, op = workflow_ops.insert_workflow(live, _template(), actor="agent", base_version=4)

    assert op["op"] == "insert_workflow"
    assert op["actor"] == "agent"
    assert op["stamp"] == [4, "agent"]
    assert [node["id"] for node in op["workflow"]["nodes"]] == [8, 9]
    assert op["workflow"]["definitions"] == _template()["definitions"]
    assert result["definitions"] == _template()["definitions"]
    assert result["_applied_ops"] == [op["op_id"]]


def test_insert_workflow_is_not_batchable():
    with pytest.raises(workflow_ops.NotBatchableError) as exc:
        workflow_ops.apply_specs(
            {"nodes": [], "links": []},
            Graph.from_object_info({}),
            [{"op": "insert_workflow", "workflow": _template()}],
        )
    assert exc.value.code == "workflow_insert_workflow_not_batchable"


def test_insert_workflow_accepts_empty_collections_without_partial_failure():
    live = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}

    result, op = workflow_ops.insert_workflow(live, {"nodes": [], "links": [], "definitions": {"subgraphs": []}})

    assert result["nodes"] == []
    assert result["links"] == []
    assert result["_applied_ops"] == [op["op_id"]]


def test_insert_workflow_rejects_dangling_link_before_mutation():
    live = {"nodes": [{"id": 1, "type": "Live"}], "links": []}
    before = copy.deepcopy(live)
    template = {"nodes": [{"id": 10, "type": "New"}], "links": [[20, 10, 0, 999, 0, "X"]]}

    with pytest.raises(ValueError, match="dangling_link_endpoint"):
        workflow_ops.insert_workflow(live, template)

    assert live == before


@pytest.mark.parametrize(
    "template",
    [
        {"nodes": ["not-an-object"], "links": []},
        {"nodes": [{"type": "MissingId"}], "links": []},
        {"nodes": [{"id": 1, "type": "A"}], "links": [[2, 1]]},
    ],
)
def test_insert_workflow_rejects_malformed_members_before_mutation(template):
    live = {"nodes": [], "links": []}
    before = copy.deepcopy(live)

    with pytest.raises(ValueError, match="malformed_op|invalid_node_payload"):
        workflow_ops.insert_workflow(live, template)

    assert live == before


def test_insert_workflow_definition_collision_is_idempotent_or_conflict():
    definition = {"id": "def-1", "nodes": [], "links": []}
    live = {"nodes": [], "links": [], "definitions": {"subgraphs": [copy.deepcopy(definition)]}}

    workflow_ops.insert_workflow(
        live, {"nodes": [], "links": [], "definitions": {"subgraphs": [copy.deepcopy(definition)]}}
    )
    assert live["definitions"]["subgraphs"] == [definition]

    before = copy.deepcopy(live)
    conflicting = {"id": "def-1", "nodes": [{"id": 1, "type": "Changed"}], "links": []}
    with pytest.raises(ValueError, match="definition_conflict"):
        workflow_ops.insert_workflow(live, {"nodes": [], "links": [], "definitions": {"subgraphs": [conflicting]}})
    assert live == before


def test_non_definition_op_rejects_definitions_field():
    live = {"nodes": [], "links": []}
    op = workflow_ops._new_op(
        "add_node",
        "test",
        0,
        node_id=1,
        node={"id": 1, "type": "A"},
        definitions={"subgraphs": []},
    )

    with pytest.raises(ValueError, match="malformed_op.*definitions"):
        workflow_ops.apply_op(live, op, None)

    assert live == {"nodes": [], "links": [], "_applied_ops": []}
