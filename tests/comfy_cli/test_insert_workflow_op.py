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
        "groups": [],
        "definitions": {"subgraphs": [{"id": "def-2", "nodes": [{"id": 5, "type": "Inner"}], "links": []}]},
    }


def test_insert_workflow_emits_input_payload_without_remapping_ids():
    live = {"last_node_id": 7, "last_link_id": 7, "nodes": [], "links": []}
    template = _template()

    result, op = workflow_ops.insert_workflow(live, template, actor="agent", base_version=4)

    assert op["op"] == "insert_workflow"
    assert op["actor"] == "agent"
    assert op["stamp"] == [4, "agent"]
    assert op["workflow"] == template
    assert result is live


def test_insert_workflow_emits_semantically_invalid_json_for_server_validation():
    live = {"nodes": [], "links": []}
    template = {
        "nodes": [{"id": 10}],
        "links": [[20, 10, 0, 999, 0]],
        "groups": [],
        "definitions": {"subgraphs": [{}]},
    }

    _, op = workflow_ops.insert_workflow(live, template)

    assert op["workflow"] == template


@pytest.mark.parametrize("missing", ["nodes", "links", "groups"])
def test_insert_workflow_rejects_each_missing_required_collection(missing):
    template = _template()
    del template[missing]

    with pytest.raises(ValueError, match=rf"insert_workflow.*missing required field.*{missing}"):
        workflow_ops.insert_workflow({"nodes": [], "links": []}, template)


def test_insert_workflow_accepts_required_collections_without_definitions():
    template = {"nodes": [], "links": [], "groups": []}

    _, op = workflow_ops.insert_workflow({"nodes": [], "links": []}, template)

    assert op["workflow"] == template


@pytest.mark.parametrize("field", ["nodes", "links", "groups"])
def test_insert_workflow_rejects_non_array_required_collection(field):
    template = _template()
    template[field] = {}

    with pytest.raises(ValueError, match=rf"insert_workflow field {field} must be an array"):
        workflow_ops.insert_workflow({"nodes": [], "links": []}, template)


def test_insert_workflow_rejects_non_object_definitions():
    template = _template()
    template["definitions"] = []

    with pytest.raises(ValueError, match="insert_workflow field definitions must be an object"):
        workflow_ops.insert_workflow({"nodes": [], "links": []}, template)


def test_insert_workflow_does_not_mutate_local_workflow():
    live = {"nodes": [{"id": 100, "type": "Existing"}], "links": [], "last_node_id": 100}
    before = copy.deepcopy(live)

    result, _ = workflow_ops.insert_workflow(live, _template())

    assert result is live
    assert live == before


def test_insert_workflow_is_not_batchable():
    with pytest.raises(workflow_ops.NotBatchableError) as exc:
        workflow_ops.apply_specs(
            {"nodes": [], "links": []},
            Graph.from_object_info({}),
            [{"op": "insert_workflow", "workflow": _template()}],
        )
    assert exc.value.code == "workflow_insert_workflow_not_batchable"


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
