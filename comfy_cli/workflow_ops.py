"""CRDT-ready structured edit operations over frontend-format ComfyUI graphs.

This is the op-model the agent (and a human via the CLI) uses to mutate a
workflow. Every primitive returns ``(workflow, op)`` where ``op`` is a
self-describing, replayable, conflict-free operation. The same op stream feeds
both a single-writer file edit (locally) and a CRDT/merge consumer (cloud) — the
CLI never merges; it only emits ops that *can* merge.

Design (settled by the identity spike):

* **Identity is leaderless & collision-free.** New node/link ids are random
  53-bit integers (``mint_id``): no shared counter, no coordination, and still
  ``int``-typed so the API converter (which gates link ids on ``isinstance(int)``)
  and an int-keyed frontend keep working. ``last_node_id``/``last_link_id`` are
  kept only as advisory high-water marks, never as allocators.
* **Widgets are name-addressed, never index-addressed.** ``set_widget`` carries
  the widget *name*; ``apply_op`` resolves name → ``widgets_values`` index against
  the live schema at apply time, so an op survives widget-layout drift.
* **Ops are idempotent** (deduped by ``op_id``) and carry a causal ``stamp``
  ``[base_version, actor]`` for deterministic last-writer-wins tie-breaking.
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any

# New ids live in [2**40, 2**53): always large (never collides with small
# frontend counter ids), always inside JS Number.MAX_SAFE_INTEGER.
_ID_FLOOR = 1 << 40


def mint_id() -> int:
    """A leaderless, collision-free, int-typed identity for a node or link."""
    return _ID_FLOOR | random.getrandbits(52)


def _new_op(kind: str, actor: str, base_version: int, **fields: Any) -> dict[str, Any]:
    return {
        "op": kind,
        "op_id": uuid.uuid4().hex,
        "actor": actor,
        "base_version": base_version,
        "stamp": [base_version, actor],
        **fields,
    }


def _find(workflow: dict, node_id: Any) -> dict | None:
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and n.get("id") == node_id:
            return n
    return None


def _require(workflow: dict, node_id: Any) -> dict:
    n = _find(workflow, node_id)
    if n is None:
        raise ValueError(f"node {node_id} not found in workflow")
    return n


# ---------------------------------------------------------------------------
# primitives — each returns (workflow, op); the op is applied via apply_op so
# apply(base, op) == primitive(base) holds by construction (P1 fidelity).
# ---------------------------------------------------------------------------


def add_node(
    workflow: dict,
    graph,
    class_type: str,
    *,
    pos: list | None = None,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    m = graph.node(class_type)
    if m is None:
        raise ValueError(f"unknown node type {class_type!r}")
    node = _build_node(mint_id(), class_type, m, graph, pos)
    op = _new_op(
        "add_node",
        actor,
        base_version,
        node_id=node["id"],
        class_type=class_type,
        pos=node["pos"],
        node=node,
    )
    return apply_op(workflow, op, graph), op


def set_widget(
    workflow: dict,
    graph,
    node_id: Any,
    widget: str,
    value: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    node = _require(workflow, node_id)
    class_type = node.get("type", "")
    idx = _widget_index(graph, class_type, widget)  # raises on unknown widget name
    widgets = node.get("widgets_values") or []
    old = widgets[idx] if idx < len(widgets) else None
    warnings = _validate_widget(graph, class_type, widget, value)  # raises on shape mismatch
    op = _new_op(
        "set_widget",
        actor,
        base_version,
        node_id=node_id,
        widget=widget,
        value=value,
        old=old,
    )
    if warnings:
        op["warnings"] = warnings
    return apply_op(workflow, op, graph), op


def connect(
    workflow: dict,
    graph,
    from_node: Any,
    from_slot: Any,
    to_node: Any,
    to_slot: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    src = _require(workflow, from_node)
    dst = _require(workflow, to_node)
    out_idx, link_type = _resolve_output_slot(src, graph, from_slot)
    in_idx, grow = _resolve_input_target(dst, graph, to_slot, link_type)
    # Type-check concrete slots: an output only connects to an input of the same
    # type (or a wildcard "*"). Autogrow slots are minted with the source type,
    # so they need no check. Without this, a mis-wire silently clobbers a link.
    if in_idx is not None:
        dst_type = (dst.get("inputs") or [])[in_idx].get("type")
        if link_type and dst_type and link_type != dst_type and "*" not in (link_type, dst_type):
            raise ValueError(
                f"type mismatch: {link_type} output of node {from_node} cannot connect to "
                f"{dst_type} input {(dst.get('inputs') or [])[in_idx].get('name')!r} of node {to_node}"
            )
    op = _new_op(
        "connect",
        actor,
        base_version,
        link_id=mint_id(),
        from_node=from_node,
        from_slot=out_idx,
        to_node=to_node,
        to_slot=in_idx,
        link_type=link_type,
    )
    if grow is not None:
        op["grow"] = grow  # autogrow: apply appends this input slot, then wires it
    return apply_op(workflow, op, graph), op


def delete_node(
    workflow: dict,
    graph,
    node_id: Any,
    *,
    actor: str = "cli",
    base_version: int = 0,
) -> tuple[dict, dict]:
    _require(workflow, node_id)
    removed = [ln[0] for ln in workflow.get("links") or [] if ln[1] == node_id or ln[3] == node_id]
    op = _new_op("delete_node", actor, base_version, node_id=node_id, removed_links=removed)
    return apply_op(workflow, op, graph), op


# ---------------------------------------------------------------------------
# apply — the deterministic, idempotent replay used by every consumer
# ---------------------------------------------------------------------------


def apply_op(workflow: dict, op: dict, graph) -> dict:
    """Replay one op onto ``workflow`` in place and return it. Idempotent: an
    op whose ``op_id`` was already applied is a no-op."""
    applied = workflow.setdefault("_applied_ops", [])
    if op["op_id"] in applied:
        return workflow
    kind = op["op"]
    if kind == "add_node":
        _apply_add_node(workflow, op)
    elif kind == "set_widget":
        _apply_set_widget(workflow, op, graph)
    elif kind == "connect":
        _apply_connect(workflow, op)
    elif kind == "delete_node":
        _apply_delete_node(workflow, op)
    else:
        raise ValueError(f"unknown op {kind!r}")
    applied.append(op["op_id"])
    return workflow


def _apply_add_node(workflow: dict, op: dict) -> None:
    nodes = workflow.setdefault("nodes", [])
    if any(n.get("id") == op["node_id"] for n in nodes):
        return
    nodes.append(copy.deepcopy(op["node"]))
    workflow["last_node_id"] = max(workflow.get("last_node_id") or 0, op["node_id"])


def _apply_set_widget(workflow: dict, op: dict, graph) -> None:
    node = _require(workflow, op["node_id"])
    idx = _widget_index(graph, node.get("type", ""), op["widget"])
    widgets = node.setdefault("widgets_values", [])
    if idx >= len(widgets):
        widgets.extend([None] * (idx + 1 - len(widgets)))
    widgets[idx] = op["value"]


def _apply_connect(workflow: dict, op: dict) -> None:
    dst = _require(workflow, op["to_node"])
    grow = op.get("grow")
    if grow is not None:
        # Autogrow: append the concrete slot (idempotent by name) and wire it.
        ins = dst.setdefault("inputs", [])
        to_idx = next((k for k, i in enumerate(ins) if i.get("name") == grow["name"]), None)
        if to_idx is None:
            entry = {"name": grow["name"], "type": grow["type"], "link": None}
            if grow.get("widget"):
                # Mark as a converted widget (ComfyUI's widget→input); value stays
                # in widgets_values for positional alignment, converter uses the link.
                entry["widget"] = {"name": grow["widget"]}
            ins.append(entry)
            to_idx = len(ins) - 1
    else:
        to_idx = op["to_slot"]
        # A concrete input holds at most one link. Replacing it must fully retire
        # the old link (drop the tuple + scrub the old source's out-links).
        prev = dst["inputs"][to_idx].get("link")
        if prev is not None and prev != op["link_id"]:
            _remove_link(workflow, prev)
    link = [op["link_id"], op["from_node"], op["from_slot"], op["to_node"], to_idx, op["link_type"]]
    links = workflow.setdefault("links", [])
    if not any(ln[0] == op["link_id"] for ln in links):
        links.append(link)
    dst["inputs"][to_idx]["link"] = op["link_id"]
    src = _require(workflow, op["from_node"])
    out_links = src["outputs"][op["from_slot"]].setdefault("links", [])
    if op["link_id"] not in out_links:
        out_links.append(op["link_id"])


def _remove_link(workflow: dict, link_id: Any) -> None:
    """Drop a link tuple and scrub every input/output reference to it."""
    workflow["links"] = [ln for ln in workflow.get("links") or [] if ln[0] != link_id]
    for n in workflow.get("nodes") or []:
        for inp in n.get("inputs") or []:
            if inp.get("link") == link_id:
                inp["link"] = None
        for out in n.get("outputs") or []:
            if link_id in (out.get("links") or []):
                out["links"] = [lid for lid in out["links"] if lid != link_id]


def _apply_delete_node(workflow: dict, op: dict) -> None:
    node_id = op["node_id"]
    workflow["nodes"] = [n for n in workflow.get("nodes") or [] if n.get("id") != node_id]
    removed = set(op.get("removed_links") or [])
    kept = [
        ln
        for ln in workflow.get("links") or []
        if ln[0] not in removed and ln[1] != node_id and ln[3] != node_id
    ]
    workflow["links"] = kept
    kept_ids = {ln[0] for ln in kept}
    # Scrub dangling references so no input/output points at a gone link.
    for n in workflow.get("nodes") or []:
        for inp in n.get("inputs") or []:
            if inp.get("link") is not None and inp["link"] not in kept_ids:
                inp["link"] = None
        for out in n.get("outputs") or []:
            out["links"] = [lid for lid in (out.get("links") or []) if lid in kept_ids]


# ---------------------------------------------------------------------------
# conflict detection + canonicalization (for ask-to-merge / convergence checks)
# ---------------------------------------------------------------------------


def _write_target(op: dict) -> tuple:
    kind = op["op"]
    if kind == "set_widget":
        return ("widget", op["node_id"], op["widget"])
    if kind in ("add_node", "delete_node"):
        return ("node", op["node_id"])
    if kind == "connect":
        return ("input", op["to_node"], op["to_slot"])
    return (kind,)


def detect_conflict(a: dict, b: dict) -> bool:
    """True iff two ops write the same target incompatibly — the signal V0's
    ask-to-merge raises instead of silently clobbering."""
    if _write_target(a) != _write_target(b):
        return False
    if a["op"] == "set_widget" and b["op"] == "set_widget":
        return a.get("value") != b.get("value")
    return True


def canonical(workflow: dict) -> dict:
    """A comparison-stable view: strip apply bookkeeping, order nodes/links by
    id. Two graphs that converged are ``canonical``-equal regardless of the
    order ops were applied in."""
    w = copy.deepcopy(workflow)
    w.pop("_applied_ops", None)
    if isinstance(w.get("nodes"), list):
        w["nodes"] = sorted(w["nodes"], key=lambda n: n.get("id"))
    if isinstance(w.get("links"), list):
        w["links"] = sorted(w["links"], key=lambda ln: ln[0])
    return w


def strip_internal(workflow: dict) -> dict:
    """Remove apply-only bookkeeping before serializing to disk."""
    workflow.pop("_applied_ops", None)
    return workflow


# ---------------------------------------------------------------------------
# schema helpers
# ---------------------------------------------------------------------------


def _build_node(node_id: int, class_type: str, m, graph, pos: list | None) -> dict:
    inputs = [{"name": p.name, "type": p.type, "link": None} for p in m.inputs if p.is_link]
    outputs = [{"name": p.name, "type": p.type, "links": []} for p in m.outputs]
    # Widget values in positional order, including dynamic-combo selectors and
    # their sub-widgets — sourced from the engine so add-node matches the converter.
    defaults = graph.widget_defaults(class_type)
    widgets = [defaults.get(name) for name in graph.widget_order(class_type)]
    return {
        "id": node_id,
        "type": class_type,
        "pos": list(pos) if pos else [0, 0],
        "size": [210, 100],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": outputs,
        "properties": {},
        "widgets_values": widgets,
    }


def _widget_index(graph, class_type: str, widget: str) -> int:
    order = graph.widget_order(class_type)
    if widget not in order:
        avail = [w for w in order if w != "control_after_generate"]
        raise ValueError(
            f"widget {widget!r} not found on {class_type}; "
            f"available: {', '.join(avail) if avail else '(none — all inputs are links)'}"
        )
    return order.index(widget)


def _validate_widget(graph, class_type: str, widget: str, value: Any) -> list[dict]:
    """Shape-validate a widget value (hard error) and collect catalog warnings
    (soft — e.g. unknown COMBO option, out-of-range number)."""
    m = graph.node(class_type)
    if m is None:
        return []
    port = next((p for p in m.inputs if p.name == widget), None)
    if port is None:
        return []
    err = port.validate_shape(value)
    if err:
        raise ValueError(err)
    return port.validate_catalog(value)


def _resolve_output_slot(node: dict, graph, slot: Any) -> tuple[int, str]:
    outs = node.get("outputs") or []
    if isinstance(slot, int) or (isinstance(slot, str) and slot.lstrip("-").isdigit()):
        i = int(slot)
        if not (0 <= i < len(outs)):
            raise ValueError(f"output slot {i} out of range for node {node.get('id')}")
        return i, outs[i].get("type", "*")
    for i, o in enumerate(outs):
        if o.get("name") == slot:
            return i, o.get("type", "*")
    names = [o.get("name") for o in outs]
    raise ValueError(f"output {slot!r} not found on node {node.get('id')}; outputs: {names}")


def _resolve_input_slot(node: dict, graph, slot: Any) -> int:
    ins = node.get("inputs") or []
    if isinstance(slot, int) or (isinstance(slot, str) and slot.lstrip("-").isdigit()):
        i = int(slot)
        if not (0 <= i < len(ins)):
            raise ValueError(f"input slot {i} out of range for node {node.get('id')}")
        return i
    for i, inp in enumerate(ins):
        if inp.get("name") == slot:
            return i
    names = [inp.get("name") for inp in ins]
    raise ValueError(f"input {slot!r} not found on node {node.get('id')}; inputs: {names}")


def _resolve_input_target(node: dict, graph, slot: Any, elem_type: str | None) -> tuple[int | None, dict | None]:
    """Resolve a connect target. Returns ``(index, None)`` for a concrete input,
    or ``(None, grow)`` where ``grow`` is the input slot to append (autogrow slot,
    or a widget converted to an input).

    - **Autogrow** (``COMFY_AUTOGROW_V3``, e.g. ``BatchImagesNode.images``) declares
      one base input but the server wants one slot key per connection
      (``images.image0``, …). Addressing the base or a dotted ``images.imageN`` key
      grows a concrete slot minted with the source type.
    - **Widget → input** (e.g. ``CreateVideo.fps``): a widget-backed input isn't a
      link slot, so it's converted — a linked input carrying a ``widget`` marker is
      appended. Its value stays in ``widgets_values`` (positional alignment holds);
      the API converter reads the link and skips the widget by name.
    """
    ins = node.get("inputs") or []
    # Concrete slot (index or exact name) that is NOT an autogrow base.
    try:
        idx = _resolve_input_slot(node, None, slot)
        if str(ins[idx].get("type", "")).startswith("COMFY_AUTOGROW"):
            base = ins[idx].get("name")
            return None, _plan_autogrow(ins, base, elem_type)
        return idx, None
    except ValueError:
        pass
    # Dotted autogrow key (images.image0) or a base that has no concrete slot yet.
    if isinstance(slot, str):
        base = slot.split(".", 1)[0]
        ag = next((i for i in ins if i.get("name") == base and str(i.get("type", "")).startswith("COMFY_AUTOGROW")), None)
        if ag is not None:
            requested = slot if "." in slot else None
            return None, _plan_autogrow(ins, base, elem_type, requested=requested)
    # Widget-backed input: convert the widget to a linked input.
    if graph is not None and isinstance(slot, str) and slot in graph.widget_order(node.get("type", "")):
        return None, {"name": slot, "type": elem_type or "*", "widget": slot}
    names = [i.get("name") for i in ins]
    raise ValueError(f"input {slot!r} not found on node {node.get('id')}; inputs: {names}")


def _plan_autogrow(ins: list, base: str, elem_type: str | None, requested: str | None = None) -> dict:
    existing = [i for i in ins if str(i.get("name", "")).startswith(base + ".")]
    if requested and not any(i.get("name") == requested for i in ins):
        name = requested
    else:
        elem = base[:-1] if base.endswith("s") else base
        name = f"{base}.{elem}{len(existing)}"
    return {"name": name, "type": elem_type or "*"}
