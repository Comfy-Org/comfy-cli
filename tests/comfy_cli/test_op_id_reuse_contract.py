"""``op_id`` is bound to one canonical payload (op-vocabulary Amendment v1.10 / CMP A8).

The graph here is a real :class:`~comfy_cli.cql.engine.Graph` built from a
one-node ``object_info``, not a hand-rolled stub. ``apply_op`` -> ``set_widget``
goes through the catalog (``graph.node`` for the schema, ``widget_order_for_node``
for the positional slot), so a stub that answers only the widget order cannot
reach the write at all.
"""

from __future__ import annotations

import copy
import json
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


# --- Amendment v1.10: what the canonical form has to agree with -------------


def test_numbers_are_rendered_the_way_ecmascript_renders_them():
    """The counterpart implementation (CMP A8) canonicalizes with
    ``JSON.stringify``, so the digest only matches cross-implementation if the
    numbers are spelled ECMAScript's way. Python's own float repr disagrees on
    exactly the shapes a FLOAT widget produces: an integral float, negative
    zero, and an exponent.
    """
    assert ops._canonical_op({"v": 20.0}) == '{"v":20}'
    assert ops._canonical_op({"v": -0.0}) == '{"v":0}'
    assert ops._canonical_op({"v": 1e-7}) == '{"v":1e-7}'
    assert ops._canonical_op({"v": 1e-6}) == '{"v":0.000001}'
    assert ops._canonical_op({"v": 1e20}) == '{"v":100000000000000000000}'
    assert ops._canonical_op({"v": 1e21}) == '{"v":1e+21}'
    assert ops._canonical_op({"v": 1.5}) == '{"v":1.5}'
    assert ops._canonical_op({"v": 5e-324}) == '{"v":5e-324}'


def test_a_float_widget_relayed_through_js_digests_the_same(graph):
    """A FLOAT op whose value is integral comes back from a JS relay as an int.
    Both spellings have to digest identically, or the identical redelivery is
    rejected as ``op_id_reuse``.
    """
    workflow = _workflow()
    as_float = _op(20.0)
    ops.apply_op(workflow, as_float, graph)
    relayed = _op(20)  # what `JSON.parse(JSON.stringify(op))` yields
    assert ops._op_digest(relayed) == ops._op_digest(as_float)
    assert ops.apply_op(workflow, relayed, graph) is workflow


def test_non_finite_numbers_canonicalize_as_null_like_json_stringify(graph):
    """``json.load`` accepts NaN/Infinity, so a document can hold one and an op
    captures it in ``old`` or in the host's ``host_widgets_values``. The earlier
    ``allow_nan=False`` made every such write raise; ``JSON.stringify`` renders
    them as ``null``, and matching that is both cross-implementation correct and
    what un-breaks overwriting a NaN widget.
    """
    assert ops._canonical_op({"v": float("nan")}) == '{"v":null}'
    assert ops._canonical_op({"v": float("inf")}) == '{"v":null}'
    assert ops._canonical_op({"v": [float("-inf")]}) == '{"v":[null]}'

    workflow = _workflow()
    workflow["nodes"][0]["widgets_values"] = [float("nan")]
    workflow, op = ops.set_widget(workflow, graph, 7, "steps", 30)
    assert workflow["nodes"][0]["widgets_values"] == [30]
    assert workflow["_applied_op_digests"][op["op_id"]] == ops._op_digest(op)


def test_a_lone_surrogate_digests_instead_of_dying_on_the_utf8_encode():
    """``json.loads('"\\ud800"')`` produces a lone surrogate, which cannot be
    UTF-8 encoded at all. ES2019 well-formed stringify escapes it; so do we, so
    the digest stays total instead of raising a bare ``UnicodeEncodeError``
    outside the ``malformed_op:`` contract callers classify errors by.
    """
    op = {"v": json.loads('"\\ud800"')}
    assert ops._canonical_op(op) == '{"v":"\\ud800"}'
    assert len(ops._op_digest(op)) == 64


def test_ordinary_large_payloads_are_not_rejected_by_the_canonicalization_bounds():
    """The bounds exist to stop a runaway payload, not to cap a real edit. The
    first pair (4,096 entries / 262,144 cost units) rejected edits that used to
    succeed: ``clear``/``reset_doc`` enumerate every node id, ``delete_node``
    on a hub node enumerates every incident link, and a prompt value runs past
    256K characters.
    """
    assert len(ops._op_digest({"removed_nodes": list(range(20_000))})) == 64
    assert len(ops._op_digest({"removed_links": [[i, 1, 0, 2, 0, "T"] for i in range(10_000)]})) == 64
    assert len(ops._op_digest({"value": "x" * 1_000_000})) == 64


def test_canonicalization_still_refuses_a_runaway_payload():
    nested: Any = "leaf"
    for _ in range(ops._MAX_OP_PAYLOAD_DEPTH + 2):
        nested = [nested]
    with pytest.raises(ValueError, match="^payload_too_deep:"):
        ops._op_digest({"v": nested})
    with pytest.raises(ValueError, match="^malformed_op: array exceeds"):
        ops._op_digest({"v": [0] * (ops._MAX_OP_COLLECTION_ENTRIES + 1)})
    with pytest.raises(ValueError, match="cost budget$"):
        ops._op_digest({"v": ["x" * 4096] * 16_384})
    with pytest.raises(ValueError, match="^malformed_op: op payload is not canonical JSON:"):
        ops._op_digest({"v": object()})
