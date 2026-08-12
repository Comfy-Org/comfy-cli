"""Enforce docs/op-vocabulary-v1.md against the code it freezes.

The doc is the normative contract for the structured-edit op vocabulary; these
tests pin it two ways so neither the doc nor the code can drift silently:

  * the doc's frozen-kinds table == ``workflow_ops.FROZEN_OPS`` == the kinds
    ``apply_op`` actually replays (minus the explicitly deferred ones);
  * the doc's ``Batchable`` column == the kinds ``apply_specs`` actually
    dispatches;
  * the ``$``-alias sugar and the batch-``clear`` rejection behave exactly as
    the doc rules.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path
from typing import Any

import pytest

from comfy_cli import error_codes, workflow_ops
from comfy_cli.cql.engine import Graph

DOC = Path(__file__).resolve().parents[2] / "docs" / "op-vocabulary-v1.md"


# ---------------------------------------------------------------------------
# doc parsing — the frozen-kinds table is the machine-readable surface
# ---------------------------------------------------------------------------


def _parse_frozen_table() -> dict[str, bool]:
    """Parse the doc's frozen-kinds markdown table into {kind: batchable}.

    The table is identified by a header row containing both ``Kind`` and
    ``Batchable``; the ``Batchable`` cell must be exactly ``yes`` or ``no``.
    """
    assert DOC.is_file(), f"contract doc missing: {DOC}"
    lines = DOC.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not (stripped.startswith("|") and "Kind" in stripped and "Batchable" in stripped):
            continue
        header = [c.strip() for c in stripped.strip("|").split("|")]
        kind_col = header.index("Kind")
        batch_col = header.index("Batchable")
        table: dict[str, bool] = {}
        for row in lines[i + 2 :]:  # skip the |---| separator
            row = row.strip()
            if not row.startswith("|"):
                break
            cells = [c.strip() for c in row.strip("|").split("|")]
            kind = cells[kind_col].strip("`")
            batchable = cells[batch_col]
            assert batchable in ("yes", "no"), f"Batchable cell for {kind!r} must be exactly yes/no, got {batchable!r}"
            table[kind] = batchable == "yes"
        assert table, "frozen-kinds table has a header but no rows"
        return table
    raise AssertionError("no markdown table with `Kind` and `Batchable` columns found in the doc")


# ---------------------------------------------------------------------------
# probes — what the code actually accepts, discovered behaviorally
# ---------------------------------------------------------------------------


def _apply_op_accepts(kind: str) -> bool:
    """True iff ``apply_op`` dispatches ``kind`` (vs rejecting it as unknown).

    A malformed-but-dispatched probe op fails with KeyError etc. — that still
    counts as accepted; only the explicit ``unknown op`` rejection counts as no.
    """
    wf: dict[str, Any] = {"nodes": [], "links": []}
    op = {"op": kind, "op_id": uuid.uuid4().hex, "actor": "probe", "base_version": 0, "stamp": [0, "probe"]}
    try:
        workflow_ops.apply_op(wf, op, None)
    except ValueError as e:
        if "unknown op" in str(e):
            return False
    except Exception:
        pass  # dispatched into a handler, probe op just lacks that kind's fields
    return True


def _apply_specs_accepts(kind: str) -> bool:
    """True iff ``apply_specs`` dispatches ``kind`` inside a batch."""
    try:
        workflow_ops.apply_specs({"nodes": [], "links": []}, _graph(), [{"op": kind}])
    except workflow_ops.NotBatchableError:
        return False
    except (ValueError, KeyError) as e:
        return "unknown op" not in str(e)
    return True


# ---------------------------------------------------------------------------
# a two-node catalog: one MODEL producer, one MODEL consumer
# ---------------------------------------------------------------------------


def _object_info() -> dict[str, Any]:
    return {
        "TinyLoader": {
            "input": {"required": {"ckpt_name": [["a.safetensors", "b.safetensors"]]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL"],
            "output_name": ["MODEL"],
            "category": "loaders",
            "display_name": "Tiny Loader",
            "python_module": "nodes",
        },
        "TinySink": {
            "input": {"required": {"model": "MODEL"}},
            "input_order": {"required": ["model"]},
            "output": [],
            "output_name": [],
            "category": "test",
            "display_name": "Tiny Sink",
            "python_module": "nodes",
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _connect_specs(from_ref: str, to_ref: str) -> list[dict]:
    return [
        {"op": "add_node", "class_type": "TinyLoader", "as": "up"},
        {"op": "add_node", "class_type": "TinySink", "as": "sink"},
        {"op": "connect", "from": from_ref, "to": to_ref},
    ]


# ---------------------------------------------------------------------------
# 1. the doc's frozen kinds == FROZEN_OPS == apply_op's dispatch table
# ---------------------------------------------------------------------------


def test_doc_lists_exactly_the_apply_op_kinds():
    table = _parse_frozen_table()
    assert set(table) == set(workflow_ops.FROZEN_OPS), (
        f"doc table {sorted(table)} != FROZEN_OPS {sorted(workflow_ops.FROZEN_OPS)}"
    )
    # The probe must be able to tell acceptance from rejection at all.
    assert not _apply_op_accepts("definitely_not_an_op")
    accepted = {k for k in workflow_ops.FROZEN_OPS if _apply_op_accepts(k)}
    expected = set(workflow_ops.FROZEN_OPS) - set(workflow_ops.DEFERRED_OPS)
    assert accepted == expected, (
        f"apply_op accepts {sorted(accepted)} but the frozen vocabulary (minus deferred "
        f"{sorted(workflow_ops.DEFERRED_OPS)}) is {sorted(expected)}"
    )
    # Deferred kinds are frozen in the doc but must NOT be replayable yet.
    for kind in workflow_ops.DEFERRED_OPS:
        assert kind in workflow_ops.FROZEN_OPS
        assert not _apply_op_accepts(kind), f"deferred op {kind!r} is implemented; un-defer it in the contract"


# ---------------------------------------------------------------------------
# 2. the doc's Batchable column == apply_specs' dispatch table
# ---------------------------------------------------------------------------


def test_batchability_matches_apply_specs():
    table = _parse_frozen_table()
    doc_batchable = {k for k, batchable in table.items() if batchable}
    assert doc_batchable == set(workflow_ops.BATCHABLE_OPS)
    probed = {k for k in workflow_ops.FROZEN_OPS if _apply_specs_accepts(k)}
    assert probed == doc_batchable, (
        f"apply_specs dispatches {sorted(probed)} but the doc marks {sorted(doc_batchable)} batchable"
    )


# ---------------------------------------------------------------------------
# 3./4./5. alias rules: bare and $-prefixed resolve identically; ${ rejects
# ---------------------------------------------------------------------------


def test_dollar_prefixed_alias_resolves():
    specs = _connect_specs("$up.MODEL", "$sink.model")
    specs.append({"op": "set_widget", "node": "$up", "widget": "ckpt_name", "value": "b.safetensors"})
    wf, ops, aliases = workflow_ops.apply_specs({"nodes": [], "links": []}, _graph(), specs)
    links = wf["links"]
    assert len(links) == 1
    assert links[0][1] == aliases["up"] and links[0][3] == aliases["sink"]
    set_op = next(op for op in ops if op["op"] == "set_widget")
    assert set_op["node_id"] == aliases["up"]
    assert set_op["value"] == "b.safetensors"


def test_bare_alias_still_resolves():
    wf, _ops, aliases = workflow_ops.apply_specs(
        {"nodes": [], "links": []}, _graph(), _connect_specs("up.MODEL", "sink.model")
    )
    links = wf["links"]
    assert len(links) == 1
    assert links[0][1] == aliases["up"] and links[0][3] == aliases["sink"]


def test_dollar_brace_rejected():
    with pytest.raises(ValueError, match="recipe param"):
        workflow_ops.apply_specs({"nodes": [], "links": []}, _graph(), _connect_specs("${up}.MODEL", "$sink.model"))


# ---------------------------------------------------------------------------
# 6. op_id format (doc section 8.2): LWW-load-bearing, so its shape is contract
# ---------------------------------------------------------------------------


def test_op_id_format_is_frozen():
    """op_id is the final LWW tiebreaker (doc sections 8.1/8.2): exactly 32
    lowercase hex chars, no dashes, on every minted op kind — its lexicographic
    comparison decides conflict outcomes, not just deduplication."""
    op_id_re = re.compile(r"^[0-9a-f]{32}$")
    g = _graph()
    wf: dict[str, Any] = {"nodes": [], "links": []}
    wf, add_op = workflow_ops.add_node(wf, g, "TinyLoader")
    wf, add2_op = workflow_ops.add_node(wf, g, "TinySink")
    wf, conn_op = workflow_ops.connect(wf, g, add_op["node_id"], "MODEL", add2_op["node_id"], "model")
    wf, set_op = workflow_ops.set_widget(wf, g, add_op["node_id"], "ckpt_name", "a.safetensors")
    wf, del_op = workflow_ops.delete_node(wf, g, add2_op["node_id"])
    wf, clear_op = workflow_ops.clear(wf)
    for op in (add_op, add2_op, conn_op, set_op, del_op, clear_op):
        assert op_id_re.match(op["op_id"]), f"{op['op']} minted op_id {op['op_id']!r}, not 32 lowercase hex chars"
        assert op["stamp"] == [op["base_version"], op["actor"]]


# ---------------------------------------------------------------------------
# 7. clear in a batch: registered code, hint names the standalone command
# ---------------------------------------------------------------------------


def test_clear_rejected_in_batch_with_registered_hint():
    with pytest.raises(workflow_ops.NotBatchableError) as ei:
        workflow_ops.apply_specs(
            {"nodes": [], "links": []},
            _graph(),
            [{"op": "add_node", "class_type": "TinyLoader"}, {"op": "clear"}],
        )
    err = ei.value
    assert error_codes.is_registered(err.code), f"{err.code!r} is not in error_codes.REGISTRY"
    registered = error_codes.get(err.code)
    assert registered is not None and "comfy workflow clear" in (registered.hint or "")
    assert "comfy workflow clear" in err.hint
    # Atomicity is part of the message contract: the caller must learn nothing landed.
    assert "no changes were applied" in str(err).lower()
