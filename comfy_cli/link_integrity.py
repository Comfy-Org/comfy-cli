"""Broken link rows in a canvas (UI-format) workflow, as validate findings.

The UI→API lowering resolves every input through the node's own
``inputs[].link`` and never reads a link row's slot indexes, so a row that
points at an input slot the node does not have, or at an output slot its
source does not have, vanished silently: validate said "valid" while the value
the row was meant to carry reached nothing. Production trace f8d27ae4: three
interior links of a SAM3 subgraph targeted input slot 6 on nodes with inputs
0-5, validate reported 0 errors, and the agent filled the empty prompts by hand
instead of re-wiring them.

One finding per broken row, addressed the way the edit surface addresses
nodes (``70/2011`` inside a subgraph), each carrying the ``connect`` that
repairs it.
"""

from __future__ import annotations

from typing import Any

_PROXY_IN = "-10"
_PROXY_OUT = "-20"


def _is_slot(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _rows(links: Any) -> list[list]:
    """Link rows as ``[id, src, src_slot, dst, dst_slot, type]`` whichever
    shape they are stored in (top-level arrays, definition objects)."""
    out: list[list] = []
    for link in links or []:
        if isinstance(link, list) and len(link) >= 5:
            out.append(list(link) + [None] * (6 - len(link)))
        elif isinstance(link, dict):
            out.append(
                [
                    link.get("id"),
                    link.get("origin_id"),
                    link.get("origin_slot"),
                    link.get("target_id"),
                    link.get("target_slot"),
                    link.get("type"),
                ]
            )
    return out


def _scope_findings(nodes: list, links: Any, prefix: str | None) -> tuple[list[dict], list[dict]]:
    from comfy_cli.workflow_ops import _types_compatible

    def addr(nid: Any) -> str:
        return f"{prefix}/{nid}" if prefix else str(nid)

    errors: list[dict] = []
    warnings: list[dict] = []
    by_id = {str(n.get("id")): n for n in nodes if isinstance(n, dict)}
    rows = _rows(links)
    rows_by_id = {str(r[0]): r for r in rows}

    def holder(node: dict, link_id: Any) -> dict | None:
        for inp in node.get("inputs") or []:
            if isinstance(inp, dict) and inp.get("link") is not None and str(inp["link"]) == str(link_id):
                return inp
        return None

    def fed_from(node: dict, src: Any, slot: Any, except_id: Any) -> str | None:
        for inp in node.get("inputs") or []:
            if not isinstance(inp, dict) or inp.get("link") is None or str(inp["link"]) == str(except_id):
                continue
            row = rows_by_id.get(str(inp["link"]))
            if row is not None and str(row[1]) == str(src) and row[2] == slot:
                return str(inp.get("name") or "")
        return None

    for link_id, src_id, src_slot, tgt_id, tgt_slot, link_type in rows:
        if str(tgt_id) == _PROXY_OUT:
            continue
        tgt = by_id.get(str(tgt_id))
        if tgt is None:
            continue  # feeds nothing at all; the lowering never sees it
        src_is_proxy = str(src_id) == _PROXY_IN
        src = None if src_is_proxy else by_id.get(str(src_id))
        held = holder(tgt, link_id)
        field = str(held.get("name") or "") if held else None
        if not src_is_proxy and src is None:
            errors.append(
                {
                    "node_id": addr(tgt_id),
                    "field": field,
                    "code": "link_source_missing",
                    "message": f"link {link_id} into node {addr(tgt_id)} comes from node {addr(src_id)}, which does "
                    "not exist — the input receives nothing",
                    "hint": f"wire the input from a real node: `connect <node>.<output> {addr(tgt_id)}.{field or '<input>'}`",
                }
            )
            continue
        if not _is_slot(tgt_slot) or not (src_is_proxy or _is_slot(src_slot)):
            errors.append(
                {
                    "node_id": addr(tgt_id),
                    "field": field,
                    "code": "link_slot_out_of_range",
                    "message": f"link {link_id} into node {addr(tgt_id)} has a non-integer slot — the input "
                    "receives nothing",
                    "hint": f"re-wire it: `connect {addr(src_id)}.<output> {addr(tgt_id)}.{field or '<input>'}`",
                }
            )
            continue
        outputs = (src or {}).get("outputs")
        if not src_is_proxy and isinstance(outputs, list) and _is_slot(src_slot) and not 0 <= src_slot < len(outputs):
            names = ", ".join(
                f"{i}:{o.get('name')}" for i, o in enumerate(outputs) if isinstance(o, dict) and o.get("name")
            )
            errors.append(
                {
                    "node_id": addr(tgt_id),
                    "field": field,
                    "code": "link_slot_out_of_range",
                    "message": f"link {link_id} into node {addr(tgt_id)} reads output slot {src_slot} of node "
                    f"{addr(src_id)}, which has {len(outputs)} output(s) — the input receives nothing",
                    "hint": f"re-wire it from an output node {addr(src_id)} has ({names or 'none'}): "
                    f"`connect {addr(src_id)}.<output> {addr(tgt_id)}.{field or '<input>'}`",
                }
            )
            continue
        inputs = [i for i in tgt.get("inputs") or [] if isinstance(i, dict)]
        if 0 <= tgt_slot < len(tgt.get("inputs") or []) or held is not None:
            continue
        # The row aims past the target's inputs and no input holds it: the
        # value it was drawn to carry reaches nothing.
        if src_is_proxy:
            source_ref = None
            source_type = link_type
        else:
            source_ref = f"{addr(src_id)}.{src_slot}"
            out = outputs[src_slot] if isinstance(outputs, list) and _is_slot(src_slot) else {}
            source_type = (out or {}).get("type") or link_type
            already = fed_from(tgt, src_id, src_slot, link_id)
            if already is not None:
                continue  # a leftover row: that value already reaches input `already`
        candidates = [
            str(i.get("name"))
            for i in inputs
            if i.get("link") is None and i.get("name") and _types_compatible(source_type, i.get("type"))
        ]
        where = (
            f"link {link_id} from {'the subgraph input' if src_is_proxy else f'node {addr(src_id)} output {src_slot}'} "
            f"targets input slot {tgt_slot} of node {addr(tgt_id)}, which has {len(tgt.get('inputs') or [])} inputs — "
            "the value reaches nothing"
        )
        if source_ref and len(candidates) == 1:
            fix = f"`connect {source_ref} {addr(tgt_id)}.{candidates[0]}`"
        elif source_ref and candidates:
            fix = f"`connect {source_ref} {addr(tgt_id)}.<one of: {', '.join(candidates)}>`"
        else:
            fix = None
        finding = {
            "node_id": addr(tgt_id),
            "field": candidates[0] if len(candidates) == 1 else None,
            "code": "link_slot_out_of_range",
            "message": where,
            "hint": (
                f"re-wire it with {fix} — don't retype the value it was meant to carry"
                if fix
                else "re-wire it to the input it was meant for"
            ),
        }
        # Only an error when an input that could take the value sits empty:
        # that is wiring the graph is missing. Otherwise the row is inert.
        (errors if candidates else warnings).append(finding)
    return errors, warnings


def broken_link_findings(workflow: dict) -> tuple[list[dict], list[dict]]:
    """``(errors, warnings)`` for every broken link row in a canvas workflow,
    top level and inside each subgraph definition (addressed through its
    first instance, as ``slots``/``set-widget`` address interior nodes)."""
    if not isinstance(workflow, dict):
        return [], []
    nodes = [n for n in workflow.get("nodes") or [] if isinstance(n, dict)]
    errors, warnings = _scope_findings(nodes, workflow.get("links"), None)

    defs = (workflow.get("definitions") or {}) if isinstance(workflow.get("definitions"), dict) else {}
    subgraphs = [sg for sg in defs.get("subgraphs") or [] if isinstance(sg, dict) and sg.get("id")]
    by_def = {str(sg["id"]): sg for sg in subgraphs}
    prefix: dict[str, str] = {}
    queue: list[tuple[list, str | None]] = [(nodes, None)]
    while queue:
        scope_nodes, scope_prefix = queue.pop(0)
        for n in scope_nodes:
            def_id = str(n.get("type"))
            if def_id in by_def and def_id not in prefix:
                prefix[def_id] = f"{scope_prefix}/{n.get('id')}" if scope_prefix else str(n.get("id"))
                inner = [x for x in by_def[def_id].get("nodes") or [] if isinstance(x, dict)]
                queue.append((inner, prefix[def_id]))
    for def_id, pfx in prefix.items():
        sg = by_def[def_id]
        inner = [x for x in sg.get("nodes") or [] if isinstance(x, dict)]
        e, w = _scope_findings(inner, sg.get("links"), pfx)
        errors.extend(e)
        warnings.extend(w)
    return errors, warnings
