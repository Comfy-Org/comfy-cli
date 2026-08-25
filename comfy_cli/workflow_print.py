"""Render a save-format workflow as Python-like source. One direction; nothing parses it back.

Renders the ComfyUI UI-save-format ``workflow`` (``nodes[]`` + ``links[]``) as
one Python-like statement per node, in topological order, with `#`-comments
carrying id/title/mode metadata. Purely a read/display aid — the output is not
meant to be executed or parsed back into a workflow.

``Reroute``/``GetNode``/``SetNode``/``PrimitiveNode`` are UI-only splicing
nodes: they never get their own line. Instead the edge that would otherwise
dangle through them is resolved to whatever real value they carry (a Reroute
chain, a SetNode/GetNode pair, or a PrimitiveNode feeding a widget-backed
input), with an inline annotation or warning when that resolution can't find
a real value. Subgraph instances (a node whose ``type`` is the UUID of an
entry under ``workflow["definitions"]["subgraphs"]``) get a ``Subgraph[...]``
call line and their definition is rendered once, after the top-level lines,
as an indented block addressed as ``<first instance id>/<inner id>``.

See the design: decisions D1-D13 (Obsidian, "workflow print (design, 2026-08-25)").
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comfy_cli.cql.engine import _expand_widget_entries, _widgets_as_positional
from comfy_cli.workflow_to_api import is_subgraph_uuid

if TYPE_CHECKING:
    from comfy_cli.cql.engine import Graph

_MODE_LABELS = {2: "mute", 4: "bypass"}
# Same set as workflow_to_api._UI_ONLY_NODE_TYPES. Notes are handled here (as
# comments); Reroute/GetNode/SetNode/PrimitiveNode are spliced through (see
# _resolve_source) rather than skipped as dangling.
_UI_ONLY = frozenset({"Note", "MarkdownNote", "PrimitiveNode", "GetNode", "SetNode", "Reroute"})
_NOTE_TYPES = frozenset({"Note", "MarkdownNote"})
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Sentinel ids litegraph uses for a subgraph definition's synthetic input/output
# proxy nodes — matches workflow_to_api._SUBGRAPH_INPUT_NODE_ID / _OUTPUT_NODE_ID.
_PROXY_IN = "-10"
_PROXY_OUT = "-20"
_MAX_SUBGRAPH_DEPTH = 10
_MAX_SPLICE_HOPS = 100


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
    caller's ``PrintUnsupported`` can report them all at once.

    A subgraph instance whose definition is missing is deliberately NOT a
    refusal here (at any depth): it's printed opaquely instead — see
    ``_render_missing_subgraph_line`` and D11 in the task brief (amended).
    """
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


# ---------------------------------------------------------------------------
# D9: splicing through Reroute / SetNode+GetNode / PrimitiveNode
# ---------------------------------------------------------------------------


def _collect_reroute_sources(nodes: list[dict], link_map: dict[str, tuple]) -> dict[str, tuple]:
    """``{reroute_id: (src_id, src_slot)}`` from each Reroute's single input link."""
    out: dict[str, tuple] = {}
    for n in nodes:
        if n.get("type") != "Reroute":
            continue
        inputs = n.get("inputs")
        if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], dict):
            continue
        link_id = inputs[0].get("link")
        if link_id is None:
            continue
        link = link_map.get(str(link_id))
        if link is None:
            continue
        src_id, src_slot, _tgt_id, _tgt_slot = link
        out[str(n.get("id"))] = (src_id, src_slot)
    return out


def _collect_get_set(nodes: list[dict], link_map: dict[str, tuple]) -> tuple[dict[str, tuple], dict[str, str]]:
    """``set_sources[var] = (src_id, src_slot)`` from each SetNode's first linked
    input; ``get_vars[get_id] = var`` for each GetNode."""
    set_sources: dict[str, tuple] = {}
    get_vars: dict[str, str] = {}
    for n in nodes:
        t = n.get("type")
        if t not in ("SetNode", "GetNode"):
            continue
        widgets = n.get("widgets_values")
        var = widgets[0] if isinstance(widgets, list) and widgets else None
        if not isinstance(var, str) or not var:
            continue
        if t == "SetNode":
            for inp in n.get("inputs") or []:
                if not isinstance(inp, dict):
                    continue
                link_id = inp.get("link")
                if link_id is None:
                    continue
                link = link_map.get(str(link_id))
                if link is None:
                    continue
                src_id, src_slot, _tgt_id, _tgt_slot = link
                set_sources[var] = (src_id, src_slot)
                break
        else:
            get_vars[str(n.get("id"))] = var
    return set_sources, get_vars


def _resolve_source(
    src_id: Any,
    src_slot: Any,
    nodes_by_id: dict[str, dict],
    reroute_sources: dict[str, tuple],
    set_sources: dict[str, tuple],
    get_vars: dict[str, str],
    proxy_in_id: str | None,
) -> tuple:
    """Chase an edge's source through Reroute / GetNode+SetNode splices (and,
    inside a subgraph definition, the ``-10`` input-proxy node) to whatever it
    ultimately resolves to. Returns one of:

    - ``("ok", id, slot)`` — a real, printable source.
    - ``("in_proxy", slot)`` — resolves to the definition's own input proxy.
    - ``("dead_reroute", reroute_id)`` — a Reroute with no upstream link.
    - ``("missing_set", get_id, var)`` — a GetNode whose variable no SetNode publishes.
    - ``("primitive_no_widget", primitive_id)`` — resolves directly to a PrimitiveNode.
    """
    seen: set[str] = set()
    for _ in range(_MAX_SPLICE_HOPS):
        key = str(src_id)
        if proxy_in_id is not None and key == proxy_in_id:
            return ("in_proxy", src_slot)
        node = nodes_by_id.get(key)
        t = node.get("type") if node else None
        if t == "Reroute":
            if key in seen:
                return ("ok", src_id, src_slot)
            seen.add(key)
            if key in reroute_sources:
                src_id, src_slot = reroute_sources[key]
                continue
            return ("dead_reroute", key)
        if t == "GetNode":
            if key in seen:
                return ("ok", src_id, src_slot)
            seen.add(key)
            var = get_vars.get(key)
            if var in set_sources:
                src_id, src_slot = set_sources[var]
                continue
            return ("missing_set", key, var)
        if t == "PrimitiveNode":
            return ("primitive_no_widget", key)
        return ("ok", src_id, src_slot)
    return ("ok", src_id, src_slot)


@dataclass
class _RenderCtx:
    """Everything the per-node arg builders need to resolve one edge, for
    either the top-level workflow (``proxy_in_id`` is ``None``) or one
    subgraph definition's interior (``proxy_in_id`` is ``"-10"``)."""

    graph: Any
    link_map: dict[str, tuple]
    nodes_by_id: dict[str, dict]
    printable_ids: set[str]
    binding_by_id: dict[str, str]
    reroute_sources: dict[str, tuple]
    set_sources: dict[str, tuple]
    get_vars: dict[str, str]
    proxy_in_id: str | None = None
    proxy_in_names: dict[int, str] = field(default_factory=dict)

    def proxy_in_name(self, slot: Any) -> str:
        name = self.proxy_in_names.get(slot)
        return name if name else f"slot{slot}"


def _resolve_edge_text(src_id: Any, src_slot: Any, name: str, tgt_id: str, ctx: _RenderCtx) -> tuple:
    """Resolve one edge to ``(ref_text, annotation, warning)``. ``ref_text`` is
    ``None`` when the edge can't be resolved to a value (prints as ``None``)."""
    outcome = _resolve_source(
        src_id, src_slot, ctx.nodes_by_id, ctx.reroute_sources, ctx.set_sources, ctx.get_vars, ctx.proxy_in_id
    )
    kind = outcome[0]
    if kind == "ok":
        _, rid, rslot = outcome
        rid_s = str(rid)
        if rid_s not in ctx.printable_ids:
            return None, None, None
        src_node = ctx.nodes_by_id.get(rid_s)
        binding = ctx.binding_by_id.get(rid_s)
        if src_node is None or binding is None:
            return None, None, None
        return _edge_ref(binding, src_node, src_node.get("type", ""), rslot, ctx.graph), None, None
    if kind == "in_proxy":
        return f"IN.{ctx.proxy_in_name(outcome[1])}", None, None
    if kind == "dead_reroute":
        return None, f" {name} unlinked via reroute {outcome[1]}", None
    if kind == "missing_set":
        get_id, var = outcome[1], outcome[2]
        return (
            None,
            None,
            f"node {tgt_id}: input '{name}' reads GetNode {get_id} variable '{var}' that no SetNode publishes",
        )
    if kind == "primitive_no_widget":
        return (
            None,
            None,
            f"node {tgt_id}: input {name!r} fed by PrimitiveNode {outcome[1]} without a widget; printed as None",
        )
    return None, None, None


def _build_args(
    node: dict, m: Any, ctx: _RenderCtx, warnings: list[str]
) -> tuple[list[str], bool, list[str], list[tuple]]:
    """Node call arguments: link inputs (in the node's own input order) first,
    then widgets. Returns ``(args, unknown, annotations, prim_hits)`` where
    ``unknown`` is True when the class isn't in the catalog (or there is no
    catalog), ``annotations`` are suffix strings to append to the line comment,
    and ``prim_hits`` are ``(primitive_id, skipped_reason)`` pairs for
    PrimitiveNode-into-widget splices found here."""
    nid = str(node.get("id"))
    args: list[str] = []
    annotations: list[str] = []
    prim_hits: list[tuple] = []

    # Pre-pass: widget-backed inputs. A link into one never overrides the
    # node's own widget value (D9) EXCEPT when it resolves to the subgraph
    # input proxy (D10) — that value genuinely comes from outside, there is no
    # static default to fall back to.
    widget_overrides: dict[str, str] = {}
    for inp in node.get("inputs") or []:
        if "widget" not in inp:
            continue
        name = inp.get("name")
        link_id = inp.get("link")
        if link_id is None:
            continue
        link = ctx.link_map.get(str(link_id))
        if link is None:
            continue
        src_id, src_slot, _tgt_id, _tgt_slot = link
        outcome = _resolve_source(
            src_id, src_slot, ctx.nodes_by_id, ctx.reroute_sources, ctx.set_sources, ctx.get_vars, ctx.proxy_in_id
        )
        if outcome[0] == "in_proxy":
            widget_overrides[name] = f"IN.{ctx.proxy_in_name(outcome[1])}"
        elif outcome[0] == "primitive_no_widget":
            prim_id = outcome[1]
            annotations.append(f" {name} from primitive {prim_id}")
            prim_hits.append((prim_id, f"inlined into {nid}.{name}"))

    for inp in node.get("inputs") or []:
        name = inp.get("name")
        if "widget" in inp:
            continue  # widget-backed input: printed in the widget pass instead
        link_id = inp.get("link")
        if link_id is None:
            args.append(f"{name}=None")
            continue
        link = ctx.link_map.get(str(link_id))
        if link is None:
            args.append(f"{name}=None")
            continue
        src_id, src_slot, _tgt_id, _tgt_slot = link
        ref, ann, warn = _resolve_edge_text(src_id, src_slot, name, nid, ctx)
        if warn:
            warnings.append(warn)
        if ann:
            annotations.append(ann)
        args.append(f"{name}={ref if ref is not None else 'None'}")

    class_type = node.get("type", "")
    widgets_values = node.get("widgets_values")
    unknown = m is None
    if unknown:
        positional = widgets_values if isinstance(widgets_values, list) else []
        if positional:
            args.append(f"widgets={py_literal(positional)}")
        warnings.append(f"node {node.get('id')}: class {class_type!r} not in catalog; widgets printed positionally")
    else:
        positional = _widgets_as_positional(widgets_values, ctx.graph, class_type)
        entries = _expand_widget_entries(m, positional)
        for idx, entry in enumerate(entries):
            value = positional[idx] if idx < len(positional) else None
            arg_name = "control_after_generate" if entry.port is None else entry.name
            if arg_name in widget_overrides:
                args.append(f"{arg_name}={widget_overrides[arg_name]}")
            else:
                args.append(f"{arg_name}={py_literal(value)}")
    return args, unknown, annotations, prim_hits


# ---------------------------------------------------------------------------
# D10: subgraph instances and definition blocks
# ---------------------------------------------------------------------------


def _render_subgraph_instance_line(
    node: dict, type_uuid: str, binding: str, ctx: _RenderCtx, warnings: list[str]
) -> tuple:
    """A subgraph instance's own line: exposed link inputs by name (or, when
    the name isn't a Python identifier, collected into a trailing ``**{}``);
    widget-backed exposed inputs print positionally from ``widgets_values``
    (subgraphs are never in the catalog) unless their link resolves to the
    enclosing definition's own input proxy, in which case that ref is used.
    Returns ``(line, prim_hits)``."""
    nid = str(node.get("id"))
    title = node.get("title") or None
    bracket = json.dumps(title, ensure_ascii=False) if title else json.dumps(nid)
    prim_hits: list[tuple] = []
    args: list[str] = []
    extra: dict[str, str] = {}

    inputs = [inp for inp in node.get("inputs") or [] if isinstance(inp, dict)]
    link_inputs = [inp for inp in inputs if "widget" not in inp]
    widget_inputs = [inp for inp in inputs if "widget" in inp]

    def _place(iname: str, text: str) -> None:
        if _IDENT.match(iname) and not keyword.iskeyword(iname):
            args.append(f"{iname}={text}")
        else:
            extra[iname] = text

    for inp in link_inputs:
        iname = inp.get("name") or ""
        ref = None
        link_id = inp.get("link")
        if link_id is not None:
            link = ctx.link_map.get(str(link_id))
            if link is not None:
                src_id, src_slot, _tgt_id, _tgt_slot = link
                ref, _ann, warn = _resolve_edge_text(src_id, src_slot, iname, nid, ctx)
                if warn:
                    warnings.append(warn)
        _place(iname, ref if ref is not None else "None")

    widgets_values = node.get("widgets_values")
    positional = widgets_values if isinstance(widgets_values, list) else []
    for k, inp in enumerate(widget_inputs):
        iname = inp.get("name") or ""
        override = None
        link_id = inp.get("link")
        if link_id is not None:
            link = ctx.link_map.get(str(link_id))
            if link is not None:
                src_id, src_slot, _tgt_id, _tgt_slot = link
                outcome = _resolve_source(
                    src_id,
                    src_slot,
                    ctx.nodes_by_id,
                    ctx.reroute_sources,
                    ctx.set_sources,
                    ctx.get_vars,
                    ctx.proxy_in_id,
                )
                if outcome[0] == "in_proxy":
                    override = f"IN.{ctx.proxy_in_name(outcome[1])}"
                elif outcome[0] == "primitive_no_widget":
                    prim_hits.append((outcome[1], f"inlined into {nid}.{iname}"))
        if override is not None:
            text = override
        else:
            value = positional[k] if k < len(positional) else None
            text = py_literal(value)
        _place(iname, text)

    if extra:
        inner = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {v}" for k, v in extra.items())
        args.append(f"**{{{inner}}}")

    line = f"{binding} = Subgraph[{bracket}]({', '.join(args)})  # {nid} subgraph {type_uuid}"
    return line, prim_hits


def _render_missing_subgraph_line(
    node: dict, type_uuid: str, binding: str, ctx: _RenderCtx, addr: str, warnings: list[str]
) -> str:
    """D11 (amended): a subgraph instance whose definition is missing is never
    a refusal, at any depth — it's printed opaquely instead. Wired link inputs
    resolve by the instance's own input name (non-identifier names via
    ``**{}``, same as a resolved instance); widget-backed instance inputs
    print positionally as a single ``widgets=[...]`` (there's no definition to
    name them against). ``addr`` is the node's address for the warning text —
    the bare id at top level, ``<instance>/<inner>`` inside a definition."""
    nid = str(node.get("id"))
    args: list[str] = []
    extra: dict[str, str] = {}

    inputs = [inp for inp in node.get("inputs") or [] if isinstance(inp, dict)]
    link_inputs = [inp for inp in inputs if "widget" not in inp]

    for inp in link_inputs:
        iname = inp.get("name") or ""
        ref = None
        link_id = inp.get("link")
        if link_id is not None:
            link = ctx.link_map.get(str(link_id))
            if link is not None:
                src_id, src_slot, _tgt_id, _tgt_slot = link
                ref, _ann, warn = _resolve_edge_text(src_id, src_slot, iname, nid, ctx)
                if warn:
                    warnings.append(warn)
        text = ref if ref is not None else "None"
        if _IDENT.match(iname) and not keyword.iskeyword(iname):
            args.append(f"{iname}={text}")
        else:
            extra[iname] = text

    if extra:
        inner = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {v}" for k, v in extra.items())
        args.append(f"**{{{inner}}}")

    widgets_values = node.get("widgets_values")
    if isinstance(widgets_values, list) and widgets_values:
        args.append(f"widgets={py_literal(widgets_values)}")

    warnings.append(f"node {addr}: subgraph definition {type_uuid} missing; printed opaquely")
    return f"{binding} = Subgraph[{json.dumps(type_uuid)}]({', '.join(args)})  # {nid} subgraph {type_uuid} definition missing"


def _definition_header(def_id: str, sg_def: dict, state: _State) -> str:
    name = sg_def.get("name") or def_id
    instances = sorted(state.instances_by_def.get(def_id, []), key=_sort_key)
    first = state.first_instance_by_def.get(def_id, instances[0] if instances else def_id)
    return (
        f"# subgraph {def_id} {json.dumps(name, ensure_ascii=False)} — "
        f"instances: {', '.join(instances)} (address inner nodes as {first}/<id>)"
    )


def _def_links(sg_def: dict) -> dict[str, tuple]:
    """Normalise a definition's dict-shaped links into the array-tuple form
    used everywhere else: ``{str(link_id): (origin_id, origin_slot, target_id, target_slot)}``."""
    out: dict[str, tuple] = {}
    for link in sg_def.get("links") or []:
        if not isinstance(link, dict):
            continue
        lid = link.get("id")
        if lid is None:
            continue
        out[str(lid)] = (link.get("origin_id"), link.get("origin_slot"), link.get("target_id"), link.get("target_slot"))
    return out


class _State:
    """Cross-definition bookkeeping accumulated while rendering: the queue of
    definitions still to render (in first-use order, growing as nested
    instances are discovered), and the shared warnings/skipped/bindings lists
    that end up on the returned ``PrintResult``."""

    def __init__(
        self, defs_by_id: dict[str, dict], warnings: list[str], skipped: list[dict], bindings: dict[str, str]
    ) -> None:
        self.defs_by_id = defs_by_id
        self.warnings = warnings
        self.skipped = skipped
        self.bindings = bindings
        self.def_order: list[str] = []
        self.def_seen: set[str] = set()
        self.def_depth: dict[str, int] = {}
        self.instances_by_def: dict[str, list[str]] = {}
        self.first_instance_by_def: dict[str, str] = {}
        self.primitive_reason: dict[str, str] = {}

    def register_instance(self, def_id: str, instance_id: str, depth: int) -> None:
        self.instances_by_def.setdefault(def_id, []).append(instance_id)
        if def_id not in self.def_seen:
            self.def_seen.add(def_id)
            self.first_instance_by_def[def_id] = instance_id
            self.def_depth[def_id] = depth
            self.def_order.append(def_id)


def _render_node_line(n: dict, name: str, ctx: _RenderCtx, graph: Graph | None, state: _State) -> str:
    """One printed line for a node that is not itself a (resolvable) subgraph
    instance: a regular class call, or the ``Node[...]`` fallback for a
    nested subgraph instance whose definition is missing."""
    nid = str(n.get("id"))
    t = n.get("type", "")
    m = graph.node(t) if graph is not None else None
    args, unknown, annotations, prim_hits = _build_args(n, m, ctx, state.warnings)
    for prim_id, reason in prim_hits:
        state.primitive_reason.setdefault(prim_id, reason)
    line = f"{name} = {class_expr(t)}({', '.join(args)})  # {nid}"
    title = n.get("title")
    if title and title != t:
        line += f" {json.dumps(title, ensure_ascii=False)}"
    mode = n.get("mode")
    if mode in _MODE_LABELS:
        line += f" mode={_MODE_LABELS[mode]}"
    if unknown:
        line += " class not in catalog"
    for ann in annotations:
        line += ann
    return line


def _render_nodes(
    order: list[dict],
    ctx: _RenderCtx,
    graph: Graph | None,
    defs_by_id: dict[str, dict],
    binding_by_id: dict[str, str],
    state: _State,
    depth: int,
    addr_prefix: str | None = None,
) -> list[str]:
    """Render one printed line per node in ``order``: a resolved subgraph
    instance gets the ``Subgraph[...]`` format, an instance whose definition
    is missing gets the opaque D11 fallback (never a refusal), everything else
    is a regular class call. Shared between the top-level workflow and a
    definition's interior (``addr_prefix`` is that definition's first
    instance id, used to address a missing-definition warning as
    ``<instance>/<inner>``)."""
    lines: list[str] = []
    for n in order:
        nid = str(n.get("id"))
        t = n.get("type", "")
        name = binding_by_id[nid]
        if is_subgraph_uuid(t):
            sg_def = defs_by_id.get(t)
            if sg_def is not None:
                line, prim_hits = _render_subgraph_instance_line(n, t, name, ctx, state.warnings)
                for prim_id, reason in prim_hits:
                    state.primitive_reason.setdefault(prim_id, reason)
                state.register_instance(t, nid, depth)
            else:
                addr = f"{addr_prefix}/{nid}" if addr_prefix else nid
                line = _render_missing_subgraph_line(n, t, name, ctx, addr, state.warnings)
        else:
            line = _render_node_line(n, name, ctx, graph, state)
        lines.append(line)
    return lines


def _build_bindings(order: list[dict]) -> dict[str, str]:
    """Local binding names for one node list, in print order: a subgraph
    instance (resolved or not — a missing definition doesn't change what it
    fundamentally is) binds off its own title, falling back to ``"subgraph"``;
    everything else binds off its class type, as usual."""
    used: dict[str, int] = {}
    binding_by_id: dict[str, str] = {}
    for n in order:
        nid = str(n.get("id"))
        t = n.get("type", "")
        if is_subgraph_uuid(t):
            title = n.get("title") or None
            name = binding_name(title or "subgraph", used)
        else:
            name = binding_name(t, used)
        binding_by_id[nid] = name
    return binding_by_id


def _render_definition_block(
    def_id: str, sg_def: dict, graph: Graph | None, state: _State, depth: int
) -> tuple[list[str], int]:
    """Render one subgraph definition's interior: its own nodes (recursing for
    nested subgraph instances via the same ``_render_nodes``), its notes, its
    UI-only-splice skips, and ``OUT.<name> = <ref>`` lines for each of its
    declared outputs. Returns ``(lines, printable_node_count)``."""
    if depth > _MAX_SUBGRAPH_DEPTH:
        raise PrintUnsupported(["subgraph nesting deeper than 10"])

    interior_nodes = [n for n in sg_def.get("nodes") or [] if isinstance(n, dict)]
    all_links = _def_links(sg_def)

    validate_links: list[list] = []
    for lid, (oid, oslot, tid, tslot) in all_links.items():
        if str(oid) == _PROXY_IN or str(tid) == _PROXY_OUT:
            continue
        validate_links.append([lid, oid, oslot, tid, tslot])

    reasons = _validate(interior_nodes, validate_links)
    if reasons:
        raise PrintUnsupported(reasons)

    printable = [n for n in interior_nodes if n.get("type") not in _UI_ONLY]
    notes = [n for n in interior_nodes if n.get("type") in _NOTE_TYPES]
    ui_only_skipped = [n for n in interior_nodes if n.get("type") in _UI_ONLY and n.get("type") not in _NOTE_TYPES]

    order = _toposort(printable, all_links)

    nodes_by_id = {str(n.get("id")): n for n in interior_nodes}
    printable_ids = {str(n.get("id")) for n in printable}

    reroute_sources = _collect_reroute_sources(interior_nodes, all_links)
    set_sources, get_vars = _collect_get_set(interior_nodes, all_links)

    proxy_in_names = {i: inp.get("name") for i, inp in enumerate(sg_def.get("inputs") or []) if isinstance(inp, dict)}
    proxy_out_names = {i: o.get("name") for i, o in enumerate(sg_def.get("outputs") or []) if isinstance(o, dict)}

    binding_by_id = _build_bindings(order)
    first_instance = state.first_instance_by_def.get(def_id, def_id)

    ctx = _RenderCtx(
        graph=graph,
        link_map=all_links,
        nodes_by_id=nodes_by_id,
        printable_ids=printable_ids,
        binding_by_id=binding_by_id,
        reroute_sources=reroute_sources,
        set_sources=set_sources,
        get_vars=get_vars,
        proxy_in_id=_PROXY_IN,
        proxy_in_names=proxy_in_names,
    )

    lines = _render_nodes(
        order, ctx, graph, state.defs_by_id, binding_by_id, state, depth + 1, addr_prefix=first_instance
    )

    for n in sorted(notes, key=lambda n: _sort_key(str(n.get("id")))):
        lines.append(_note_comment(n))
        state.skipped.append({"id": str(n.get("id")), "type": n.get("type"), "reason": "note"})
    for n in ui_only_skipped:
        t = n.get("type")
        nid = str(n.get("id"))
        reason = state.primitive_reason.get(nid, "spliced") if t == "PrimitiveNode" else "spliced"
        state.skipped.append({"id": nid, "type": t, "reason": reason})

    out_sources: dict[Any, tuple] = {}
    for oid, oslot, tid, tslot in all_links.values():
        if str(tid) != _PROXY_OUT:
            continue
        if tslot not in out_sources:
            out_sources[tslot] = (oid, oslot)
    for slot in sorted(out_sources, key=_sort_key):
        oid, oslot = out_sources[slot]
        out_name = proxy_out_names.get(slot) or f"out{slot}"
        ref, _ann, warn = _resolve_edge_text(oid, oslot, out_name, f"{def_id} OUT", ctx)
        if warn:
            state.warnings.append(warn)
        lines.append(f"OUT.{out_name} = {ref if ref is not None else 'None'}")

    for nid, name in binding_by_id.items():
        state.bindings[f"{first_instance}/{name}"] = f"{first_instance}/{nid}"

    return lines, len(printable)


def render_py(workflow: dict, graph: Graph | None) -> PrintResult:
    nodes = workflow.get("nodes") or []
    links = workflow.get("links") or []
    defs_by_id = {
        sg.get("id"): sg
        for sg in (workflow.get("definitions") or {}).get("subgraphs") or []
        if isinstance(sg, dict) and sg.get("id")
    }

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

    reroute_sources = _collect_reroute_sources(nodes, link_map)
    set_sources, get_vars = _collect_get_set(nodes, link_map)

    # binding_name is called once per node, in topological order, so the
    # dedupe suffixes (_2, _3, ...) match print order.
    binding_by_id = _build_bindings(order)
    bindings: dict[str, str] = {name: nid for nid, name in binding_by_id.items()}

    ctx = _RenderCtx(
        graph=graph,
        link_map=link_map,
        nodes_by_id=nodes_by_id,
        printable_ids=printable_ids,
        binding_by_id=binding_by_id,
        reroute_sources=reroute_sources,
        set_sources=set_sources,
        get_vars=get_vars,
    )

    warnings: list[str] = []
    skipped: list[dict] = []
    state = _State(defs_by_id, warnings, skipped, bindings)

    lines = _render_nodes(order, ctx, graph, defs_by_id, binding_by_id, state, depth=1)

    for n in sorted(notes, key=lambda n: _sort_key(str(n.get("id")))):
        lines.append(_note_comment(n))
        skipped.append({"id": str(n.get("id")), "type": n.get("type"), "reason": "note"})
    for n in ui_only_skipped:
        t = n.get("type")
        nid = str(n.get("id"))
        reason = state.primitive_reason.get(nid, "spliced") if t == "PrimitiveNode" else "spliced"
        skipped.append({"id": nid, "type": t, "reason": reason})

    node_count = len(printable)
    for def_id in state.def_order:
        sg_def = defs_by_id[def_id]
        lines.append(_definition_header(def_id, sg_def, state))
        block_lines, block_count = _render_definition_block(def_id, sg_def, graph, state, state.def_depth[def_id])
        lines.extend("    " + bl for bl in block_lines)
        node_count += block_count

    source = "\n".join(lines) + "\n"
    return PrintResult(source=source, bindings=bindings, node_count=node_count, skipped=skipped, warnings=warnings)
