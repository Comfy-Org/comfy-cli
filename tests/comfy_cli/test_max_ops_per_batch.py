"""A batch is bounded at ``_MAX_OPS_PER_BATCH``, mirrored from the shared package.

``comfy-multi-player``'s ``src/limits.ts`` has FOUR bounds on untrusted ops. The
three payload ones (depth, collection entries, cost) are pinned in
``test_op_id_reuse_contract.py`` beside the canonicalizer they guard. This file
owns the fourth, which is about the BATCH rather than about one op's payload, and
is therefore enforced somewhere else entirely: at the two surfaces that build or
dispatch a whole batch, ``apply_specs`` and ``replace_ops``.

Why a local bound at all, when the shared package already has one: without it
comfy-cli builds a batch of any length and the refusal lands downstream *after*
comfy-cli has already written the document or already handed a caller the batch
as the legal translation of its graph. That is the mint-then-refuse shape — the
producer commits, the shared applier rejects the whole batch — that the payload
bounds exist to close one level down.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
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

_FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _nodes(count: int) -> dict:
    """A frontend-format graph of ``count`` unlinked nodes.

    ``replace_ops`` mints one ``add_node`` per node and one ``connect`` per link,
    so an unlinked graph makes the op count equal to the node count and the
    boundary exact.
    """
    return {
        "nodes": [
            {"id": i + 1, "type": "KSampler", "widgets_values": [20], "inputs": [], "outputs": []} for i in range(count)
        ],
        "links": [],
    }


def _add_node_specs(count: int) -> list[dict]:
    return [{"op": "add_node", "class_type": "KSampler"} for _ in range(count)]


_DYNAMIC_GROUP_FIXTURE = json.loads((_FIXTURES / "dynamic_group.json").read_text(encoding="utf-8"))


@pytest.fixture
def graph_with_group() -> Graph:
    """A catalog carrying a DynamicGroup class, so a batch can be built whose op
    count is larger than its spec count."""
    return Graph.from_object_info(copy.deepcopy(_DYNAMIC_GROUP_FIXTURE["object_info"]))


def _group_specs(pairs: int) -> list[dict]:
    """``pairs`` × (one ``add_node``, one three-row ``set_dynamic_group_rows``).

    Each pair is 2 specs and 11 ops — 1 for the node, then the row count plus
    three row fields (``lora_name``, ``strength``, ``enabled``) × 3 rows. A
    group may only be written once per batch, hence one fresh node per write.
    """
    rows = [{"lora_name": "A.safetensors", "strength": 1.0} for _ in range(3)]
    specs: list[dict] = []
    for i in range(pairs):
        specs.append({"op": "add_node", "class_type": "DevToolsNodeWithDynamicGroup", "as": f"n{i}"})
        specs.append({"op": "set_dynamic_group_rows", "node": f"$n{i}", "group": "loras", "rows": rows})
    return specs


# --- the constant is mirrored, not chosen here ------------------------------


def test_the_batch_bound_mirrors_the_shared_op_package_value():
    """``MAX_OPS_PER_BATCH`` is 1024 in ``comfy-multi-player``'s ``src/limits.ts``
    at ``221b2b2e470c79afaface24b1edcb53cdafc8922`` — cited by SHA rather than by
    branch, because the whole point is that the two implementations agree.

    The shared package enforces it in two places, both of which refuse the batch
    WHOLE rather than truncating or partially applying it: ``applyOps`` checks it
    before any op is processed and returns a ``malformed_op`` rejection for every
    op in the batch, and ``inspectOps`` raises ``malformed_op`` from the storage
    preflight. So comfy-cli cannot buy anything by allowing a longer batch — it
    can only move where the batch dies, to after comfy-cli has committed to it.
    Widen it there and move this pin; never widen it here alone.
    """
    assert ops._MAX_OPS_PER_BATCH == 1024


# --- the boundary: N applies, N+1 is refused --------------------------------


def test_apply_specs_accepts_a_batch_of_exactly_the_limit(graph):
    workflow, minted, _aliases = ops.apply_specs({"nodes": [], "links": []}, graph, _add_node_specs(1024))
    assert len(minted) == ops._MAX_OPS_PER_BATCH
    assert len(workflow["nodes"]) == ops._MAX_OPS_PER_BATCH


def test_apply_specs_refuses_one_op_past_the_limit(graph):
    workflow = {"nodes": [], "links": []}
    with pytest.raises(ValueError, match=r"^malformed_op: batch of 1025 specs exceeds the 1024-op limit"):
        ops.apply_specs(workflow, graph, _add_node_specs(1025))
    # "Rejected before any op was processed", the same promise the shared
    # package's message makes: the document the caller handed in is untouched,
    # including by the position planner that runs on every spec.
    assert workflow == {"nodes": [], "links": []}


def test_apply_specs_counts_the_ops_the_specs_expand_into_not_the_specs(graph_with_group):
    """One spec is not one op, so the entry check on ``len(specs)`` is only the
    cheap half of the bound.

    ``set_dynamic_group_rows`` expands into a row-count write plus one
    ``set_widget`` per row field, so 200 specs here mint 1,100 ops: comfortably
    past the entry check and comfortably past the bound. Without the second
    check this batch applies locally and is handed to a caller, and ``applyOps``
    then refuses every op in it as ``malformed_op`` — the mint-then-refuse shape
    this bound exists to close. Red on the parent, which returns 1,100 ops.
    """
    with pytest.raises(ValueError, match=r"^malformed_op: batch mints 1100 ops, more than the 1024-op limit") as exc:
        ops.apply_specs({"nodes": [], "links": []}, graph_with_group, _group_specs(100))
    # Refused like every other mid-batch failure, so a summary-receipt caller
    # reads the same structured position and the same "nothing was written".
    assert exc.value.applied_count == 0


def test_a_batch_whose_expansion_stays_within_the_bound_still_applies(graph_with_group):
    """The complement of the test above: the second check bounds the expansion,
    it does not refuse an expanding batch as such."""
    _workflow, minted, _aliases = ops.apply_specs({"nodes": [], "links": []}, graph_with_group, _group_specs(90))
    assert len(minted) == 990


def test_replace_ops_mints_a_batch_of_exactly_the_limit():
    assert len(ops.replace_ops({"nodes": [], "links": []}, _nodes(1024))) == ops._MAX_OPS_PER_BATCH


def test_replace_ops_refuses_to_mint_one_op_past_the_limit():
    """Refused as ``NotExpressibleError``, not as a bare ``ValueError``.

    That is what the two production callers already catch, and the distinction is
    behavioural rather than cosmetic: ``templates fetch --emit-ops`` treats it as
    non-fatal and reports ``ops_skipped`` beside a file it has still written,
    while ``generate``'s emit path turns it into an ``EmitError``. A bare
    ``ValueError`` would escape both and crash a fetch that had otherwise
    succeeded.
    """
    with pytest.raises(ops.NotExpressibleError, match=r"needs 1025 ops, more than the 1024-op batch limit"):
        ops.replace_ops({"nodes": [], "links": []}, _nodes(1025))


def test_the_old_document_counts_toward_the_batch_too():
    """``replace_ops`` emits a ``delete_node`` for every node already in ``old``
    before the adds, so the bound binds on the SUM. Counting only ``new`` would
    let a 600-node replacement of a 600-node canvas mint 1,200 ops.
    """
    assert len(ops.replace_ops(_nodes(512), _nodes(512))) == ops._MAX_OPS_PER_BATCH
    with pytest.raises(ops.NotExpressibleError, match=r"needs 1200 ops"):
        ops.replace_ops(_nodes(600), _nodes(600))


# --- the per-op payload bounds, on a batch nothing applies ------------------


def test_replace_ops_refuses_an_op_whose_own_payload_is_out_of_bounds():
    """The batch length is only the OUTER bound.

    ``apply_specs``' ops each go through ``apply_op``, which canonicalizes them
    and so enforces §20.3's per-op bounds (depth 64, 4,096 entries, 262,144
    cost). ``replace_ops`` builds its ops and hands them straight to a caller,
    so nothing checked those — and ``templates fetch --emit-ops`` presented the
    result as the legal translation of the graph while the shared applier
    refused it whole as ``malformed_op``. Same mint-then-refuse shape, one level
    down.
    """
    # A `widgets_values` past the entry cap: copied verbatim into `add_node`.
    oversized = {
        "nodes": [
            {
                "id": 1,
                "type": "KSampler",
                "widgets_values": [0] * (ops._MAX_OP_COLLECTION_ENTRIES + 1),
                "inputs": [],
                "outputs": [],
            }
        ],
        "links": [],
    }
    with pytest.raises(ops.NotExpressibleError, match=r"op #0 \(add_node\) is not a legal op payload: .*entries"):
        ops.replace_ops({"nodes": [], "links": []}, oversized)

    # And by cost, from one very long widget string.
    costly = {
        "nodes": [
            {
                "id": 1,
                "type": "KSampler",
                "widgets_values": ["x" * (ops._MAX_OP_COST + 1)],
                "inputs": [],
                "outputs": [],
            }
        ],
        "links": [],
    }
    with pytest.raises(ops.NotExpressibleError, match=r"not a legal op payload: .*cost budget"):
        ops.replace_ops({"nodes": [], "links": []}, costly)


def test_the_per_op_check_leaves_an_ordinary_batch_alone():
    """It refuses a payload the shared applier would refuse, and nothing else:
    every op a real graph mints is orders of magnitude inside the bounds."""
    minted = ops.replace_ops(_nodes(4), _nodes(8))
    assert len(minted) == 12
    assert all(len(ops._op_digest(op)) == 64 for op in minted)


# --- where it actually bites, measured rather than assumed ------------------


def test_the_batch_bound_binds_only_far_past_any_real_workflow():
    """Pinned so the next reader does not have to re-derive the headroom before
    deciding whether the bound caps a real edit.

    Measured over all 735 workflows in ``Comfy-Org/workflow_templates`` (the
    gallery ``templates fetch --emit-ops`` reads), the longest batch
    ``replace_ops`` actually mints is **50 ops** — 4.9% of the bound — from
    ``templates-1_input-multiple_styles_prompt``. 447 of those templates carry a
    subgraph definition, a canvas group or a reroute point, so ``replace_ops``
    refuses them as inexpressible before size is even considered; across all 735
    the absolute upper bound on nodes+links is 212, still a fifth of the bound.
    The largest expressible workflow in this repo's own fixtures mints 3 ops.

    What that measurement does NOT cover is the ``old`` side. ``replace_ops``
    emits a ``delete_node`` per node already in the document being replaced, and
    that document is user data bounded by nothing here — see
    ``test_the_old_document_counts_toward_the_batch_too``. Both production
    callers degrade rather than fail on it.

    This test pins the fixture end of the ``new``-side measurement, which is the
    half that can regress in CI.
    """
    fixture = json.loads((_FIXTURES / "seedream_5_0_pro_t2i_ui.json").read_text(encoding="utf-8"))
    minted = ops.replace_ops({"nodes": [], "links": []}, fixture)
    assert len(minted) * 100 < ops._MAX_OPS_PER_BATCH, (
        f"the widest real fixture batch is {len(minted)} ops; if it is approaching "
        f"{ops._MAX_OPS_PER_BATCH} the bound needs widening in comfy-multi-player first"
    )


def test_no_production_surface_builds_a_batch_near_the_bound(graph):
    """The one shape in the frozen vocabulary that scales with document size and
    is NOT capped by this bound: a standalone ``clear`` or ``reset_doc``, which
    carries every node id inside ONE op's payload rather than as one op each.
    Those are bounded by ``_MAX_OP_COLLECTION_ENTRIES`` (4,096 entries) instead,
    and they are standalone-only — ``BATCHABLE_OPS`` excludes them — so there is
    no shape where a large document turns into a long batch behind the caller's
    back.
    """
    assert "clear" not in ops.BATCHABLE_OPS
    assert "reset_doc" not in ops.BATCHABLE_OPS
    _workflow, op = ops.clear(_nodes(2048))
    assert len(op["removed_nodes"]) == 2048
