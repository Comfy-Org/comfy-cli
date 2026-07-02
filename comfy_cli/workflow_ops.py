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
import re
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
# recipes — a parameterized op-batch. A recipe is `{params?, ops:[...]}`; a bare
# list is a param-less batch. `${name}` placeholders in op values are filled from
# `--param`. Validation is strict: a required param with no value, an unknown
# param, or a `${name}` the recipe didn't declare all fail — never a silent blank.
# ---------------------------------------------------------------------------

_PARAM_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class RecipeError(ValueError):
    """A recipe or its parameters are malformed."""


def parse_recipe(doc: Any) -> tuple[list, dict]:
    """Split a recipe document into (ops, params_decl). Accepts a bare op list."""
    if isinstance(doc, list):
        return doc, {}
    if isinstance(doc, dict) and isinstance(doc.get("ops"), list):
        return doc["ops"], (doc.get("params") or {})
    raise RecipeError("recipe must be a JSON array of ops, or an object with an `ops` array")


def resolve_params(params_decl: dict, provided: dict[str, str]) -> dict[str, Any]:
    """Type-coerce provided values against the declared params. Errors on an
    unknown param, or a declared param with neither a value nor a default."""
    unknown = sorted(set(provided) - set(params_decl))
    if unknown:
        raise RecipeError(f"unknown --param {unknown}; recipe declares {sorted(params_decl)}")
    out: dict[str, Any] = {}
    for name, decl in params_decl.items():
        decl = decl if isinstance(decl, dict) else {}
        if name in provided:
            out[name] = _coerce_param(provided[name], decl.get("type", "string"), name)
        elif "default" in decl:
            out[name] = decl["default"]
        else:
            raise RecipeError(f"missing required --param {name!r}")
    return out


def _coerce_param(raw: str, type_name: str, name: str) -> Any:
    if type_name == "int":
        try:
            return int(raw)
        except ValueError as e:
            raise RecipeError(f"--param {name}: expected int, got {raw!r}") from e
    if type_name in ("float", "number"):
        try:
            return float(raw)
        except ValueError as e:
            raise RecipeError(f"--param {name}: expected number, got {raw!r}") from e
    if type_name in ("bool", "boolean"):
        low = raw.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
        raise RecipeError(f"--param {name}: expected bool, got {raw!r}")
    return raw  # string (the default)


def substitute_params(ops: list, params: dict[str, Any]) -> list:
    """Replace `${name}` in op values. A value that is exactly `${name}` takes the
    param's real (typed) value; embedded refs interpolate as text. An undeclared
    `${name}` is an error, not a blank."""

    def sub(value: Any) -> Any:
        if isinstance(value, str):
            whole = _PARAM_REF.fullmatch(value)
            if whole:
                return _param(whole.group(1), params)
            return _PARAM_REF.sub(lambda m: str(_param(m.group(1), params)), value)
        if isinstance(value, list):
            return [sub(v) for v in value]
        if isinstance(value, dict):
            return {k: sub(v) for k, v in value.items()}
        return value

    return [sub(op) for op in ops]


def _param(name: str, params: dict[str, Any]) -> Any:
    if name not in params:
        raise RecipeError(f"recipe references undeclared param ${{{name}}}")
    return params[name]


def capture_recipe(workflow: dict, graph, name: str = "captured", lift: dict | None = None) -> dict:
    """Project a UI-format graph into a recipe — the op-batch that rebuilds it
    (add_node + non-default set_widget + connect). The inverse of `apply`:
    `apply(empty, capture(wf))` reproduces `wf`. Top-level nodes only.

    `lift` maps `(node_id, widget_name) -> param_name`: those widgets become
    `${param_name}` holes (with a `params` header entry defaulting to the current
    value) even if the value equals the node default — so the fields you want to
    vary are actually parameterizable. No auto-parameterization otherwise."""
    if (workflow.get("definitions") or {}).get("subgraphs"):
        raise RecipeError("capture does not support subgraphs yet — edit/flatten top-level nodes first")
    lift = lift or {}
    nodes = [n for n in (workflow.get("nodes") or []) if isinstance(n, dict) and "id" in n]
    by_id = {n["id"]: n for n in nodes}

    # Validate lift targets up front — no silently-ignored typos.
    for (node_id, widget), _pname in lift.items():
        node = by_id.get(node_id)
        if node is None:
            raise RecipeError(f"--param target node {node_id!r} not in workflow")
        if widget not in graph.widget_order(node.get("type", "")):
            raise RecipeError(f"--param target {node_id}.{widget!r}: not a widget on {node.get('type')}")

    alias_by_id: dict[Any, str] = {}
    counts: dict[str, int] = {}
    for n in nodes:
        slug = re.sub(r"[^a-z0-9]+", "_", str(n.get("type", "node")).lower()).strip("_") or "node"
        counts[slug] = counts.get(slug, 0) + 1
        alias_by_id[n["id"]] = slug if counts[slug] == 1 else f"{slug}_{counts[slug]}"

    ops: list[dict] = []
    params_header: dict[str, Any] = {}
    for n in nodes:
        alias = alias_by_id[n["id"]]
        class_type = n.get("type")
        add: dict[str, Any] = {"op": "add_node", "class_type": class_type, "as": alias}
        if n.get("pos"):
            add["at"] = n["pos"]
        ops.append(add)
        order = graph.widget_order(class_type)
        defaults = graph.widget_defaults(class_type)
        widgets = n.get("widgets_values") or []
        for i, wname in enumerate(order):
            if i >= len(widgets):
                break
            pname = lift.get((n["id"], wname))
            if pname is not None:
                # Explicitly lifted → a ${param} hole, current value as its default.
                ops.append({"op": "set_widget", "node": alias, "widget": wname, "value": f"${{{pname}}}"})
                params_header[pname] = {"type": _widget_param_type(graph, class_type, wname), "default": widgets[i]}
            elif widgets[i] != defaults.get(wname):
                # Only widgets that differ from the fresh-node default — add_node fills the rest.
                ops.append({"op": "set_widget", "node": alias, "widget": wname, "value": widgets[i]})

    node_by_id = {n["id"]: n for n in nodes}
    for ln in workflow.get("links") or []:
        if not (isinstance(ln, list) and len(ln) >= 5):
            continue
        _lid, from_id, from_slot, to_id, to_slot = ln[0], ln[1], ln[2], ln[3], ln[4]
        if from_id not in alias_by_id or to_id not in alias_by_id:
            continue
        out_name = _slot_name(node_by_id[from_id].get("outputs"), from_slot)
        in_name = _slot_name(node_by_id[to_id].get("inputs"), to_slot)
        ops.append(
            {"op": "connect", "from": f"{alias_by_id[from_id]}.{out_name}", "to": f"{alias_by_id[to_id]}.{in_name}"}
        )

    return {"recipe": name, "params": params_header, "ops": ops}


def _widget_param_type(graph, class_type: str, widget: str) -> str:
    """Recipe param type for a widget, from its schema port type."""
    m = graph.node(class_type)
    port = next((p for p in (m.inputs if m else []) if p.name == widget), None)
    t = (port.type if port else "").upper()
    if t == "INT":
        return "int"
    if t in ("FLOAT", "NUMBER"):
        return "float"
    if t == "BOOLEAN":
        return "bool"
    return "string"


def _slot_name(slots: Any, idx: Any) -> Any:
    """A link's slot addressed by name where the node declares one, else by index
    (both are valid connect targets)."""
    if isinstance(slots, list) and isinstance(idx, int) and 0 <= idx < len(slots):
        nm = slots[idx].get("name") if isinstance(slots[idx], dict) else None
        if nm:
            return nm
    return idx


# ---------------------------------------------------------------------------
# apply_specs — run a batch of edit specs (add_node/connect/set_widget/delete_node)
# with `as` aliases so later specs reference just-minted nodes. Shared by the
# `apply` and `foreach` commands. Raises on a malformed spec so the caller can keep
# the batch atomic (write nothing on failure).
# ---------------------------------------------------------------------------


def resolve_ref(ref: Any, aliases: dict[str, Any]) -> Any:
    """Map an alias to its minted id; pass ints/unknown strings through."""
    if isinstance(ref, str):
        if ref in aliases:
            return aliases[ref]
        if ref.lstrip("-").isdigit():
            return int(ref)
    return ref


def _split_ref_slot(spec_val: str, aliases: dict[str, Any]) -> tuple[Any, Any]:
    """Split `<node_or_alias>.<slot>` and resolve the node part."""
    node_part, _, slot = str(spec_val).partition(".")
    return resolve_ref(node_part, aliases), slot


def apply_specs(workflow: dict, graph, specs: list, *, actor: str = "cli", base_version: int = 0) -> tuple[dict, list, dict]:
    """Apply edit specs to ``workflow`` in order. Returns (workflow, ops, aliases)."""
    aliases: dict[str, Any] = {}
    ops: list[dict] = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, dict) or "op" not in spec:
            raise ValueError(f"spec #{i} must be an object with an 'op' field")
        kind = spec["op"]
        if kind == "add_node":
            workflow, op = add_node(workflow, graph, spec["class_type"], pos=spec.get("at"), actor=actor, base_version=base_version)
            if spec.get("as"):
                aliases[spec["as"]] = op["node_id"]
        elif kind == "connect":
            fn, fs = _split_ref_slot(spec["from"], aliases)
            tn, ts = _split_ref_slot(spec["to"], aliases)
            workflow, op = connect(workflow, graph, fn, fs, tn, ts, actor=actor, base_version=base_version)
        elif kind == "set_widget":
            workflow, op = set_widget(
                workflow, graph, resolve_ref(spec["node"], aliases), spec["widget"], spec["value"],
                actor=actor, base_version=base_version,
            )
        elif kind == "delete_node":
            workflow, op = delete_node(workflow, graph, resolve_ref(spec["node"], aliases), actor=actor, base_version=base_version)
        else:
            raise ValueError(f"spec #{i}: unknown op {kind!r}")
        ops.append(op)
    return workflow, ops, aliases


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
