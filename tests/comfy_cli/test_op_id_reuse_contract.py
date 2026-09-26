import copy

import pytest

from comfy_cli import workflow_ops as ops


class _Graph:
    def widget_order_for_node(self, _class_type, _values):
        return ["steps"]


def _workflow():
    return {"nodes": [{"id": 7, "type": "KSampler", "widgets_values": [20], "inputs": [], "outputs": []}], "links": []}


def _op(value=25):
    return {
        "op": "set_widget",
        "op_id": "same0000000000000000000000000000",
        "actor": "human:a",
        "base_version": 5,
        "stamp": [5, "human:a"],
        "node_id": 7,
        "widget": "steps",
        "value": value,
    }


def test_changed_payload_reusing_op_id_rejects_without_mutation():
    workflow = _workflow()
    ops.apply_op(workflow, _op(25), _Graph())
    before = copy.deepcopy(workflow)
    with pytest.raises(ValueError, match="^op_id_reuse:"):
        ops.apply_op(workflow, _op(30), _Graph())
    assert workflow == before


def test_canonical_digest_matches_cmp_a8_vector_and_key_order_is_immaterial():
    workflow = _workflow()
    first = _op()
    ops.apply_op(workflow, first, _Graph())
    assert (
        workflow["_applied_op_digests"][first["op_id"]]
        == "d395ba92dd991c07c0f790b974b3d4f34e3b29c71a26621634d75de31040a431"
    )
    reordered = dict(reversed(list(first.items())))
    assert ops.apply_op(workflow, reordered, _Graph()) is workflow


def test_legacy_list_only_replay_returns_before_digesting_payload():
    workflow = _workflow()
    workflow["_applied_ops"] = [_op()["op_id"]]
    replay = _op()
    replay["value"] = object()
    assert ops.apply_op(workflow, replay, _Graph()) is workflow


def test_canonical_ignores_replay_digest_metadata():
    workflow = _workflow()
    applied = copy.deepcopy(workflow)
    ops.apply_op(applied, _op(), _Graph())
    applied["nodes"] = copy.deepcopy(workflow["nodes"])
    assert ops.canonical(applied) == ops.canonical(workflow)
