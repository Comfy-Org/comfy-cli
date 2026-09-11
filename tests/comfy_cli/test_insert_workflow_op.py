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
