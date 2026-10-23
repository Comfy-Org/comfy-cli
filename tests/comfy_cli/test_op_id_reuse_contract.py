"""``op_id`` is bound to one canonical payload (op-vocabulary Amendment v1.6 / CMP A8).

The graph here is a real :class:`~comfy_cli.cql.engine.Graph` built from a
one-node ``object_info``, not a hand-rolled stub. ``apply_op`` -> ``set_widget``
goes through the catalog (``graph.node`` for the schema, ``widget_order_for_node``
for the positional slot), so a stub that answers only the widget order cannot
reach the write at all.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from comfy_cli import workflow_ops as ops
from comfy_cli.cql.engine import Graph

_OBJECT_INFO: dict[str, Any] = {
    "KSampler": {
        "input": {"required": {"steps": ["INT", {"default": 20, "min": 1, "max": 10000}]}},
        "input_order": {"required": ["steps"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "sampling",
        "display_name": "KSampler",
        "python_module": "nodes",
    },
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


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


def test_changed_payload_reusing_op_id_rejects_without_mutation(graph):
    workflow = _workflow()
    ops.apply_op(workflow, _op(25), graph)
    before = copy.deepcopy(workflow)
    with pytest.raises(ValueError, match="^op_id_reuse:"):
        ops.apply_op(workflow, _op(30), graph)
    assert workflow == before


def test_canonical_digest_matches_cmp_a8_vector_and_key_order_is_immaterial(graph):
    workflow = _workflow()
    first = _op()
    ops.apply_op(workflow, first, graph)
    assert (
        workflow["_applied_op_digests"][first["op_id"]]
        == "d395ba92dd991c07c0f790b974b3d4f34e3b29c71a26621634d75de31040a431"
    )
    reordered = dict(reversed(list(first.items())))
    assert ops.apply_op(workflow, reordered, graph) is workflow


def test_legacy_list_only_replay_returns_before_digesting_payload(graph):
    workflow = _workflow()
    workflow["_applied_ops"] = [_op()["op_id"]]
    replay = _op()
    replay["value"] = object()
    assert ops.apply_op(workflow, replay, graph) is workflow


def test_canonical_ignores_replay_digest_metadata(graph):
    workflow = _workflow()
    applied = copy.deepcopy(workflow)
    ops.apply_op(applied, _op(), graph)
    applied["nodes"] = copy.deepcopy(workflow["nodes"])
    assert ops.canonical(applied) == ops.canonical(workflow)
