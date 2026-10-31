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


def test_object_keys_sort_by_code_point_not_by_utf16_code_unit():
    """§8.1's caveat applies to the key sort too: for ASCII, JS UTF-16 ``<``
    agrees with code-point order, but above the Basic Multilingual Plane it does
    not. ``Object.keys(...).sort()`` puts U+1F600 before U+FF00; code-point
    order is the other way round, and code-point order is what the amendment
    pins. Only a dict widget whose own keys are non-ASCII can reach this, since
    every key an op envelope carries is ASCII.
    """
    assert ops._canonical_op({"\U0001f600": 1, "\uff00": 2}) == '{"\uff00":2,"\U0001f600":1}'


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


def test_the_bounds_mirror_the_shared_op_package_values():
    """These are MIRRORED from the shared op package, not chosen here.

    ``comfy-multi-player``'s ``src/limits.ts`` at
    ``221b2b2e470c79afaface24b1edcb53cdafc8922`` fixes ``MAX_PAYLOAD_DEPTH`` at
    64, ``MAX_COLLECTION_ENTRIES`` at 4096, ``MAX_OP_COST`` at 262144 and
    ``MAX_OPS_PER_BATCH`` at 1024, and
    its A8 canonicalizer charges the whole envelope on EVERY apply, before the
    dedupe gate (``applyOps`` -> ``opDigest`` -> ``canonicalJson`` ->
    ``opBoundsRefusal``). So a looser bound here cannot rescue an oversized
    edit; it only lets comfy-cli accept a payload the shared package then
    refuses as ``malformed_op``. Widen it there and move this pin — never here
    alone.
    """
    assert ops._MAX_OP_PAYLOAD_DEPTH == 64
    assert ops._MAX_OP_COLLECTION_ENTRIES == 4096
    assert ops._MAX_OP_COST == 262_144
    # The fourth bound in the same `src/limits.ts` block bounds the BATCH rather
    # than one op's payload, so it is enforced at `apply_specs`/`replace_ops`
    # instead of in the canonicalizer and is pinned, with its boundary cases, in
    # `test_max_ops_per_batch.py`. Asserted here too so that reading this list
    # cannot suggest only three of the four are mirrored.
    assert ops._MAX_OPS_PER_BATCH == 1024


def test_ordinary_large_payloads_are_not_rejected_by_the_canonicalization_bounds():
    """The bounds exist to stop a runaway payload, not to cap a real edit, and
    at the shared values they still clear every real edit by two orders of
    magnitude. The widest op the frozen vocabulary mints from an actual
    workflow fixture in this repo measures ~1.6K cost units and 13 entries.

    The three edits cited as needing a looser bound are exercised here at sizes
    far past anything observed: a ``clear`` enumerating a thousand node ids, a
    hub ``delete_node`` enumerating a thousand incident links, and a 60K-char
    prompt value whose op also captures an equal-sized ``old``.
    """
    assert len(ops._op_digest({"op": "clear", "removed_nodes": list(range(1_000))})) == 64
    assert len(ops._op_digest({"op": "delete_node", "node_id": 7, "removed_links": list(range(1_000))})) == 64
    assert len(ops._op_digest({"op": "set_widget", "value": "x" * 60_000, "old": "y" * 60_000})) == 64


def test_the_shared_bounds_bind_only_past_any_observed_workflow():
    """Where the mirrored bounds actually bite, pinned so the next reader does
    not have to re-derive it before deciding whether an edit is really capped.

    ``removed_nodes``/``removed_links`` are flat lists, so the ENTRY cap is what
    binds them, at 4,097 — a workflow with more than 4,096 nodes, or a single
    node carrying more than 4,096 incident links. The largest workflow in this
    repo's fixtures has 11 nodes. A widget string is charged by length, so the
    COST budget binds it, and an op that captures an equal-sized ``old`` halves
    the headroom: 131,064 characters for the bare value/old pair, a little less
    once the envelope's own fields are charged. That is a 128 KiB string in one
    widget.
    """
    assert len(ops._op_digest({"removed_nodes": [0] * ops._MAX_OP_COLLECTION_ENTRIES})) == 64
    with pytest.raises(ValueError, match="^malformed_op: array exceeds"):
        ops._op_digest({"removed_nodes": [0] * (ops._MAX_OP_COLLECTION_ENTRIES + 1)})

    assert len(ops._op_digest({"value": "x" * 131_064, "old": "y" * 131_064})) == 64
    with pytest.raises(ValueError, match="cost budget$"):
        ops._op_digest({"value": "x" * 131_065, "old": "y" * 131_065})


def test_canonicalization_still_refuses_a_runaway_payload():
    nested: Any = "leaf"
    for _ in range(ops._MAX_OP_PAYLOAD_DEPTH + 2):
        nested = [nested]
    with pytest.raises(ValueError, match="^payload_too_deep:"):
        ops._op_digest({"v": nested})
    with pytest.raises(ValueError, match="^malformed_op: array exceeds"):
        ops._op_digest({"v": [0] * (ops._MAX_OP_COLLECTION_ENTRIES + 1)})
    # Cost-only overflow: 64 entries is well inside the entry cap, so this
    # reaches the cost branch rather than tripping the entry branch first.
    with pytest.raises(ValueError, match="cost budget$"):
        ops._op_digest({"v": ["x" * 4096] * 64})
    with pytest.raises(ValueError, match="^malformed_op: op payload is not canonical JSON:"):
        ops._op_digest({"v": object()})


# --- Amendment v1.10: rejecting before mutation -----------------------------


def test_a_rejected_first_seen_op_leaves_no_ledger_behind(graph):
    """ "Rejects before mutation" has to include the ledger itself: a document
    that arrived without ``_applied_ops`` must not come back carrying an empty
    one because canonicalization refused the op.
    """
    workflow = _workflow()
    before = copy.deepcopy(workflow)
    deep = _op()
    deep["value"] = [["x"] * 1]
    for _ in range(ops._MAX_OP_PAYLOAD_DEPTH + 2):
        deep["value"] = [deep["value"]]
    with pytest.raises(ValueError, match="^payload_too_deep:"):
        ops.apply_op(workflow, deep, graph)
    assert workflow == before
    assert "_applied_ops" not in workflow


@pytest.mark.parametrize(
    ("ledger", "message"),
    [
        (None, "^malformed_op: _applied_op_digests must be an object"),
        ([], "^malformed_op: _applied_op_digests must be an object"),
        ("{}", "^malformed_op: _applied_op_digests must be an object"),
    ],
)
def test_a_malformed_digest_ledger_is_refused_before_the_handler_runs(graph, ledger, message):
    """The ledger is stripped on write but never checked on load. Indexing it in
    ``apply_op``'s tail raised ``TypeError`` after the handler had already
    written the document and appended to ``_applied_ops`` — outside the
    rollback, so the op stayed half-committed.
    """
    workflow = _workflow()
    workflow["_applied_op_digests"] = ledger
    before = copy.deepcopy(workflow)
    with pytest.raises(ValueError, match=message):
        ops.apply_op(workflow, _op(), graph)
    assert workflow == before


def test_a_truthy_non_dict_ledger_is_refused_on_the_reuse_branch_too(graph):
    workflow = _workflow()
    applied = _op()
    ops.apply_op(workflow, applied, graph)
    workflow["_applied_op_digests"] = "not a mapping"
    with pytest.raises(ValueError, match="^malformed_op: _applied_op_digests must be an object"):
        ops.apply_op(workflow, applied, graph)


@pytest.mark.parametrize("op_id", [["a"], {"a": 1}, 7, None])
def test_a_non_string_op_id_is_refused_before_the_handler_runs(graph, op_id):
    workflow = _workflow()
    before = copy.deepcopy(workflow)
    bad = _op()
    bad["op_id"] = op_id
    with pytest.raises(ValueError, match="^malformed_op: op_id must be a string"):
        ops.apply_op(workflow, bad, graph)
    assert workflow == before


def test_a_malformed_applied_ops_list_is_refused_before_the_handler_runs(graph):
    workflow = _workflow()
    workflow["_applied_ops"] = {"not": "a list"}
    before = copy.deepcopy(workflow)
    with pytest.raises(ValueError, match="^malformed_op: _applied_ops must be an array"):
        ops.apply_op(workflow, _op(), graph)
    assert workflow == before


# --- Amendment v1.10: the enriched envelope ---------------------------------


def _primitive_canvas(value="old prompt") -> dict:
    """A frontend-only ``PrimitiveNode``: no catalog entry, so ``set_widget``
    takes the ``legacy_primitive`` path that enriches the op with the host's
    positional array AFTER ``apply_op`` has already recorded a digest.
    """
    return {
        "nodes": [
            {
                "id": 51,
                "type": "PrimitiveNode",
                "outputs": [{"name": "STRING", "type": "STRING", "links": [], "widget": {"name": "text"}}],
                "widgets_values": [value],
            }
        ],
        "links": [],
    }


def test_the_ledger_binds_the_enriched_envelope_the_caller_broadcasts(graph):
    """``_set_widget_impl`` finishes the op after ``apply_op`` returns, because
    ``host_widgets_values`` can only be read back out of the document the write
    produced. If the ledger kept the pre-enrichment digest, a redelivery of the
    payload that was actually broadcast would be rejected as ``op_id_reuse``.
    """
    workflow, op = ops.set_widget(_primitive_canvas(), graph, 51, "value", "a lighthouse at dusk")
    assert op["promoted"]["host_widgets_values"] == ["a lighthouse at dusk"]
    assert workflow["_applied_op_digests"][op["op_id"]] == ops._op_digest(op)
    # The broadcast envelope round-trips as the documented no-op...
    assert ops.apply_op(workflow, copy.deepcopy(op), graph) is workflow
    # ...and the same id with a changed payload is still refused.
    changed = copy.deepcopy(op)
    changed["value"] = "a lighthouse at dawn"
    with pytest.raises(ValueError, match="^op_id_reuse:"):
        ops.apply_op(workflow, changed, graph)


def test_a_nan_host_widget_no_longer_fails_an_op_that_already_applied(graph):
    """The enriched digest runs after the write has committed, so raising there
    reported failure for an op the document had already accepted: the caller
    never got it back to broadcast and ``set_widget``'s ``ValueError`` fallback
    could attempt a second write.
    """
    workflow, op = ops.set_widget(_primitive_canvas(float("nan")), graph, 51, "value", "lighthouse")
    assert workflow["nodes"][0]["widgets_values"] == ["lighthouse"]
    assert workflow["_applied_op_digests"][op["op_id"]] == ops._op_digest(op)


def test_rebinding_a_digest_is_inert_when_the_entry_is_not_this_calls_own():
    """The three cases ``_rebind_op_digest`` must leave alone: no ledger at all
    (the op deferred, so nothing recorded it), an entry belonging to a different
    payload (a duplicate), and an uncanonical enriched envelope (the document is
    already written, so raising is worse than keeping the older digest).
    """
    op = _op()
    applied = ops._op_digest(op)
    enriched = dict(op, promoted={"host_widgets_values": ["x"]})

    no_ledger: dict = {}
    ops._rebind_op_digest(no_ledger, enriched, applied)
    assert no_ledger == {}

    someone_elses = {"_applied_op_digests": {op["op_id"]: "a" * 64}}
    ops._rebind_op_digest(someone_elses, enriched, applied)
    assert someone_elses["_applied_op_digests"][op["op_id"]] == "a" * 64

    uncanonical = dict(enriched, promoted={"host_widgets_values": [object()]})
    ours = {"_applied_op_digests": {op["op_id"]: applied}}
    ops._rebind_op_digest(ours, uncanonical, applied)
    assert ours["_applied_op_digests"][op["op_id"]] == applied

    ours = {"_applied_op_digests": {op["op_id"]: applied}}
    ops._rebind_op_digest(ours, enriched, applied)
    assert ours["_applied_op_digests"][op["op_id"]] == ops._op_digest(enriched)
