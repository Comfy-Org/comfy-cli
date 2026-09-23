from __future__ import annotations

import copy

import pytest

from comfy_cli import layout, workflow_ops
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


def test_insert_workflow_rebases_far_off_template_near_existing_graph():
    """Regression for the staging bug: a template whose nodes carry their own
    baked-in absolute `pos` (e.g. authored/exported far from the origin) must
    land beside the target graph's existing nodes, not at its original
    coordinates -- otherwise the insert produces two disconnected clusters on
    canvas."""
    live = {
        "nodes": [
            {"id": 1, "type": "A", "pos": [40, 60], "size": [210, 100]},
            {"id": 2, "type": "B", "pos": [400, 1060], "size": [210, 100]},
        ],
        "links": [],
    }
    template = {
        "nodes": [
            {"id": 100, "type": "Src", "pos": [0, 7330], "size": [210, 100]},
            {"id": 101, "type": "Dst", "pos": [300, 7430], "size": [210, 100]},
        ],
        "links": [[200, 100, 0, 101, 0, "X"]],
        "groups": [],
    }

    _, op = workflow_ops.insert_workflow(live, template)
    inserted = op["workflow"]["nodes"]

    existing_box = layout._bbox(live["nodes"])
    inserted_box = layout._bbox(inserted)

    # The block lands beside the existing graph's bounding box, not thousands
    # of pixels away at its own original coordinates.
    assert inserted_box[0] >= existing_box[2]
    assert abs(inserted_box[1] - existing_box[1]) < 50
    assert inserted[0]["pos"][1] < 2000

    # Internal relative layout is preserved: a uniform translation, not a
    # per-node reshuffle.
    orig_dx = template["nodes"][1]["pos"][0] - template["nodes"][0]["pos"][0]
    orig_dy = template["nodes"][1]["pos"][1] - template["nodes"][0]["pos"][1]
    new_dx = inserted[1]["pos"][0] - inserted[0]["pos"][0]
    new_dy = inserted[1]["pos"][1] - inserted[0]["pos"][1]
    assert new_dx == pytest.approx(orig_dx)
    assert new_dy == pytest.approx(orig_dy)

    # the source template dict passed in is untouched (insert_workflow copies)
    assert template["nodes"][0]["pos"] == [0, 7330]


def test_insert_workflow_leaves_positions_untouched_when_graph_is_empty():
    """Nothing to be beside: an empty target graph is a no-op for placement,
    matching `cascade_pos`'s own empty-graph behaviour."""
    template = {
        "nodes": [{"id": 100, "type": "Src", "pos": [0, 7330], "size": [210, 100]}],
        "links": [],
        "groups": [],
    }

    _, op = workflow_ops.insert_workflow({"nodes": [], "links": []}, template)

    assert op["workflow"]["nodes"][0]["pos"] == [0, 7330]


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


def test_insert_workflow_rejects_missing_nodes():
    template = _template()
    del template["nodes"]

    with pytest.raises(ValueError, match=r"insert_workflow.*missing required field.*nodes"):
        workflow_ops.insert_workflow({"nodes": [], "links": []}, template)


def test_insert_workflow_accepts_nodes_only_and_preserves_omitted_optional_collections():
    template = {"nodes": [{"id": 100, "type": "Src"}]}

    _, op = workflow_ops.insert_workflow({"nodes": [], "links": []}, template)

    assert op["workflow"] == template
    assert "links" not in op["workflow"]
    assert "groups" not in op["workflow"]
    assert op["workflow"]["nodes"][0]["id"] == 100


def test_insert_workflow_accepts_required_collections_without_definitions():
    template = {"nodes": [], "links": [], "groups": []}

    _, op = workflow_ops.insert_workflow({"nodes": [], "links": []}, template)

    assert op["workflow"] == template


@pytest.mark.parametrize("field", ["nodes", "links", "groups"])
def test_insert_workflow_rejects_non_array_collection(field):
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
