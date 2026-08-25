"""Render a save-format workflow as Python-like source. One direction; nothing parses it back.

Renders the ComfyUI UI-save-format ``workflow`` (``nodes[]`` + ``links[]``) as
one Python-like statement per node, in topological order, with `#`-comments
carrying id/title/mode metadata. Purely a read/display aid — the output is not
meant to be executed or parsed back into a workflow.

``Reroute``/``GetNode``/``SetNode``/``PrimitiveNode`` are treated as skipped
("ui-only") here; a later task adds resolvers that splice through them instead
of leaving a dangling ``None`` — the walk below is structured so that only the
edge-resolution step needs to change for that.

See the design: decisions D1-D13 (Obsidian, "workflow print (design, 2026-08-25)").
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comfy_cli.cql.engine import _expand_widget_entries, _widgets_as_positional

if TYPE_CHECKING:
    from comfy_cli.cql.engine import Graph

_MODE_LABELS = {2: "mute", 4: "bypass"}
# Same set as workflow_to_api._UI_ONLY_NODE_TYPES. Notes are handled here (as
# comments); the rest are skipped with reason "ui-only" until a later task
# adds splicing resolvers for them.
_UI_ONLY = frozenset({"Note", "MarkdownNote", "PrimitiveNode", "GetNode", "SetNode", "Reroute"})
_NOTE_TYPES = frozenset({"Note", "MarkdownNote"})
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class PrintResult:
    source: str  # full text, lines joined with "\n", trailing newline
    bindings: dict[str, str] = field(default_factory=dict)  # binding name -> node id as string ("3", "10/7")
    node_count: int = 0  # nodes that got a line (top level + subgraph interiors)
    skipped: list[dict] = field(default_factory=list)  # {"id": str, "type": str, "reason": str}
    warnings: list[str] = field(default_factory=list)


class PrintUnsupported(Exception):
    """The workflow can't be printed as-is (structural problem, not a rendering choice)."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def class_expr(class_type: str) -> str:
    """The call-target expression for a class type: a bare name when it's a valid,
    non-keyword Python identifier, else a ``Node[...]`` subscript."""
    return (
        class_type
        if _IDENT.match(class_type) and not keyword.iskeyword(class_type)
        else f"Node[{json.dumps(class_type)}]"
    )


def binding_name(class_type: str, used: dict[str, int]) -> str:
    """A snake_case Python identifier for ``class_type``, deduped against ``used``
    (mutated in place) in call order: repeats get a ``_2``, ``_3``, ... suffix.

    CamelCase -> Camel_Case; an acronym of 2+ capitals splits before a following word
    (CLIPText -> CLIP_Text, VAEDecode -> VAE_Decode) but a single capital does not (KSampler -> ksampler).
    """
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z][A-Z])(?=[A-Z][a-z])", "_", class_type)
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()
    s = re.sub(r"_+", "_", s) or "node"
    if s[0].isdigit():
        s = "n" + s
    if keyword.iskeyword(s):
        s += "_"
    used[s] = used.get(s, 0) + 1
    return s if used[s] == 1 else f"{s}_{used[s]}"


def py_literal(value: Any) -> str:
    """A Python literal rendering of ``value``. Strings via ``json.dumps`` (which
    is also valid Python string-literal syntax); lists/dicts via ``json.dumps``
    too — ``True``/``False``/``None`` nested inside those stay JSON-cased
    (``true``/``false``/``null``); that's acceptable and documented here."""
    if value is None:
        return "None"
    if value is True:
        return "True"
    if value is False:
        return "False"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(value, ensure_ascii=False)


def _sort_key(node_id: str) -> tuple[int, Any]:
    """Ascending-numeric sort key; non-numeric ids sort after all numeric ones."""
    try:
        return (0, int(node_id))
    except (TypeError, ValueError):
        return (1, node_id)


def _validate(nodes: list[dict], links: list[list]) -> list[str]:
    """Structural checks that must hold before any ordering/printing work starts.
    Collects every problem found (rather than stopping at the first) so the
    caller's ``PrintUnsupported`` can report them all at once."""
    reasons: list[str] = []
    for n in nodes:
        t = n.get("type")
        extra = n.get("extra")
        is_legacy_group = (isinstance(t, str) and (t.startswith("workflow>") or t.startswith("workflow/"))) or (
            isinstance(extra, dict) and extra.get("groupNodes")
        )
        if is_legacy_group:
            reasons.append(f"node {n.get('id')} is a legacy group node ({t})")

    nodes_by_id = {str(n.get("id")): n for n in nodes}
    for link in links:
        if not isinstance(link, list) or len(link) < 5:
            reasons.append(f"link malformed: {link!r}")
            continue
        link_id, src_id, src_slot, tgt_id, tgt_slot = link[0], link[1], link[2], link[3], link[4]
        src_node = nodes_by_id.get(str(src_id))
        if src_node is None:
            reasons.append(f"link {link_id} references missing node {src_id}")
            continue
        tgt_node = nodes_by_id.get(str(tgt_id))
        if tgt_node is None:
            reasons.append(f"link {link_id} references missing node {tgt_id}")
            continue
        outputs = src_node.get("outputs")
        if isinstance(outputs, list) and not (0 <= src_slot < len(outputs)):
            reasons.append(f"link {link_id} references out-of-range output slot {src_slot} on node {src_id}")
        inputs = tgt_node.get("inputs")
        if isinstance(inputs, list) and not (0 <= tgt_slot < len(inputs)):
            reasons.append(f"link {link_id} references out-of-range input slot {tgt_slot} on node {tgt_id}")
    return reasons


def _toposort(printable: list[dict], link_map: dict[str, tuple]) -> list[dict]:
    """Kahn's algorithm over ``printable`` nodes, using only links whose source
    AND target are both printable. Ties break by ascending numeric id."""
    ids = [str(n.get("id")) for n in printable]
    printable_ids = set(ids)
    node_by_id = {str(n.get("id")): n for n in printable}

    deps: dict[str, set[str]] = {i: set() for i in ids}
    dependents: dict[str, list[str]] = {i: [] for i in ids}
    for src_id, _src_slot, tgt_id, _tgt_slot in link_map.values():
        s, t = str(src_id), str(tgt_id)
        if s in printable_ids and t in printable_ids and s != t:
            if s not in deps[t]:
                deps[t].add(s)
                dependents[s].append(t)

    indegree = {i: len(deps[i]) for i in ids}
    ready = [i for i in ids if indegree[i] == 0]
    ready.sort(key=_sort_key)
    order: list[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for t in dependents[cur]:
            indegree[t] -= 1
            if indegree[t] == 0:
                ready.append(t)
        ready.sort(key=_sort_key)

    if len(order) != len(ids):
        remaining = sorted(set(ids) - set(order), key=_sort_key)
        raise PrintUnsupported([f"link cycle among nodes {', '.join(remaining)}"])
    return [node_by_id[i] for i in order]


def _edge_ref(binding: str, src_node: dict, src_type: str, slot: int, graph: Graph | None) -> str:
    """How to reference one output slot of an already-bound source node."""
    outputs = src_node.get("outputs") or []
    if len(outputs) == 1:
        return binding
    out_name = None
    if 0 <= slot < len(outputs):
        out_name = outputs[slot].get("name")
    if not out_name and graph is not None:
        m = graph.node(src_type)
        if m is not None and 0 <= slot < len(m.outputs):
            out_name = m.outputs[slot].name
    if out_name and _IDENT.match(out_name) and not keyword.iskeyword(out_name):
        return f"{binding}.{out_name}"
    return f"{binding}.out[{slot}]"


def _note_comment(node: dict) -> str:
    widgets = node.get("widgets_values")
    text = str(widgets[0]) if isinstance(widgets, list) and widgets else ""
    first_line = text.split("\n")[0] if text else ""
    non_empty = [ln for ln in text.split("\n") if ln.strip()]
    extra_lines = max(len(non_empty) - 1, 0)
    title = node.get("title") or node.get("type")
    comment = f"# note {node.get('id')} {json.dumps(title, ensure_ascii=False)}: {first_line}"
    if extra_lines > 0:
        comment += f" (+{extra_lines} lines)"
    return comment


def _build_args(
    node: dict,
    m: Any,
    graph: Graph | None,
    bindings: dict[str, str],
    nodes_by_id: dict[str, dict],
    printable_ids: set[str],
    link_map: dict[str, tuple],
    warnings: list[str],
) -> tuple[list[str], bool]:
    """Node call arguments: link inputs (in the node's own input order) first,
    then widgets. Returns ``(args, unknown)`` where ``unknown`` is True when the
    class isn't in the catalog (or there is no catalog), which also determines
    whether widgets print positionally as ``widgets=[...]``."""
    args: list[str] = []
    for inp in node.get("inputs") or []:
        name = inp.get("name")
        if "widget" in inp:
            continue  # widget-backed input: printed in the widget pass instead
        link_id = inp.get("link")
        if link_id is None:
            args.append(f"{name}=None")
            continue
        link = link_map.get(str(link_id))
        if link is None:
            args.append(f"{name}=None")
            continue
        src_id, src_slot, _tgt_id, _tgt_slot = link
        src_node = nodes_by_id.get(str(src_id))
        if src_node is None or str(src_id) not in printable_ids:
            # ui-only source (Reroute/GetNode/SetNode/PrimitiveNode) — no
            # resolver yet, prints as unresolved. See module docstring.
            args.append(f"{name}=None")
            continue
        binding = bindings[str(src_id)]
        ref = _edge_ref(binding, src_node, src_node.get("type", ""), src_slot, graph)
        args.append(f"{name}={ref}")

    class_type = node.get("type", "")
    widgets_values = node.get("widgets_values")
    unknown = m is None
    if unknown:
        positional = widgets_values if isinstance(widgets_values, list) else []
        if positional:
            args.append(f"widgets={py_literal(positional)}")
        warnings.append(f"node {node.get('id')}: class {class_type!r} not in catalog; widgets printed positionally")
    else:
        positional = _widgets_as_positional(widgets_values, graph, class_type)
        entries = _expand_widget_entries(m, positional)
        for idx, entry in enumerate(entries):
            value = positional[idx] if idx < len(positional) else None
            arg_name = "control_after_generate" if entry.port is None else entry.name
            args.append(f"{arg_name}={py_literal(value)}")
    return args, unknown


def render_py(workflow: dict, graph: Graph | None) -> PrintResult:
    nodes = workflow.get("nodes") or []
    links = workflow.get("links") or []

    reasons = _validate(nodes, links)
    if reasons:
        raise PrintUnsupported(reasons)

    link_map: dict[str, tuple] = {}
    for link in links:
        link_id, src_id, src_slot, tgt_id, tgt_slot = link[0], link[1], link[2], link[3], link[4]
        link_map[str(link_id)] = (src_id, src_slot, tgt_id, tgt_slot)

    printable = [n for n in nodes if n.get("type") not in _UI_ONLY]
    notes = [n for n in nodes if n.get("type") in _NOTE_TYPES]
    ui_only_skipped = [n for n in nodes if n.get("type") in _UI_ONLY and n.get("type") not in _NOTE_TYPES]

    order = _toposort(printable, link_map)

    nodes_by_id = {str(n.get("id")): n for n in nodes}
    printable_ids = {str(n.get("id")) for n in printable}

    # binding_name is called once per node, in topological order, so the
    # dedupe suffixes (_2, _3, ...) match print order.
    used: dict[str, int] = {}
    bindings: dict[str, str] = {}
    binding_by_id: dict[str, str] = {}
    for n in order:
        nid = str(n.get("id"))
        name = binding_name(n.get("type", ""), used)
        bindings[name] = nid
        binding_by_id[nid] = name

    warnings: list[str] = []
    lines: list[str] = []
    for n in order:
        nid = str(n.get("id"))
        t = n.get("type", "")
        m = graph.node(t) if graph is not None else None
        args, unknown = _build_args(n, m, graph, binding_by_id, nodes_by_id, printable_ids, link_map, warnings)
        name = binding_by_id[nid]
        line = f"{name} = {class_expr(t)}({', '.join(args)})  # {n.get('id')}"
        title = n.get("title")
        if title and title != t:
            line += f" {json.dumps(title, ensure_ascii=False)}"
        mode = n.get("mode")
        if mode in _MODE_LABELS:
            line += f" mode={_MODE_LABELS[mode]}"
        if unknown:
            line += " class not in catalog"
        lines.append(line)

    skipped: list[dict] = []
    for n in sorted(notes, key=lambda n: _sort_key(str(n.get("id")))):
        lines.append(_note_comment(n))
        skipped.append({"id": str(n.get("id")), "type": n.get("type"), "reason": "note"})
    for n in ui_only_skipped:
        skipped.append({"id": str(n.get("id")), "type": n.get("type"), "reason": "ui-only"})

    source = "\n".join(lines) + "\n"
    return PrintResult(source=source, bindings=bindings, node_count=len(printable), skipped=skipped, warnings=warnings)
