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
as an indented block addressed as ``<first instance address>/<inner id>`` —
fully qualified through every nesting level, matching
``workflow_ops._navigate_subgraph_path``.

Two rules keep the printed names the ones the editors take:

* a regular node's widgets print by their value-aware names
  (``Graph.widget_order_for_node``: ``model``, ``model.resolution``, …); an
  auto-grow group (``COMFY_AUTOGROW_V3``) owns no widget slot — its grown
  slots print as links, one open slot stays visible, and the line comment
  names the group with its element type;
* a subgraph instance's promoted widgets (``cql.promoted``) print their
  EFFECTIVE value — the host's, else the interior default — under the
  instance address (``57.width``); the interior line keeps ``IN.width`` and
  the definition header says where that value lives.

See the design: decisions D1-D13 (Obsidian, "workflow print (design, 2026-08-25)").
"""

from __future__ import annotations

import json
import keyword
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from comfy_cli.cql import promoted as _promoted
from comfy_cli.cql.engine import _UNRESOLVED, _expand_widget_entries, _resolve_proxy_value, _widgets_as_positional
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


def _member_ref(base: str, name: str) -> str:
    """``base.name`` when ``name`` is a plain Python identifier, else
    ``base["name"]`` — so a subgraph proxy input/output whose declared name has
    a space (or any other non-identifier character) still prints as something
    that parses."""
    if _IDENT.match(name) and not keyword.iskeyword(name):
        return f"{base}.{name}"
    return f"{base}[{json.dumps(name, ensure_ascii=False)}]"


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


def _is_slot_index(value: Any) -> bool:
    """A link slot must be a real integer index (``bool`` is an int in Python,
    and is not one)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _validate(nodes: list[dict], links: list[list]) -> list[str]:
    """Structural checks that must hold before any ordering/printing work starts.
    Collects every problem found (rather than stopping at the first) so the
    caller's ``PrintUnsupported`` can report them all at once.

    A subgraph instance whose definition is missing is deliberately NOT a
    refusal here (at any depth): it's printed opaquely instead — see
    ``_render_missing_subgraph_line`` and D11 in the task brief (amended).
    """
    reasons: list[str] = []
    seen_ids: set[str] = set()
    reported_dupes: set[str] = set()
    for n in nodes:
        t = n.get("type")
        extra = n.get("extra")
        is_legacy_group = (isinstance(t, str) and (t.startswith("workflow>") or t.startswith("workflow/"))) or (
            isinstance(extra, dict) and extra.get("groupNodes")
        )
        if is_legacy_group:
            reasons.append(f"node {n.get('id')} is a legacy group node ({t})")
        # Ids key every map here (nodes_by_id, bindings, skipped addresses), so
        # a repeat silently drops a node and mis-resolves its edges. Report each
        # duplicated id once, in first-seen order, rather than printing a lie.
        nid = str(n.get("id"))
        if nid in seen_ids:
            if nid not in reported_dupes:
                reported_dupes.add(nid)
                reasons.append(f"duplicate node id {nid}")
        else:
            seen_ids.add(nid)

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
        # Slots index into outputs/inputs below and into widgets_values later;
        # a None or "0" would raise a TypeError deep in the render instead of
        # being reported here with every other structural problem.
        if not _is_slot_index(src_slot) or not _is_slot_index(tgt_slot):
            reasons.append(f"link {link_id} has a non-integer slot")
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
    AND target are both printable. Ties break by ascending numeric id.

    ``link_map`` must already have each link's source resolved past any
    Reroute/GetNode-SetNode splice (see ``_splice_link_map``) — a raw,
    unresolved ``A -> Reroute -> B`` link contributes no ordering constraint
    here (Reroute is never printable), which would silently let ``B`` print
    before ``A`` whenever ``B``'s id sorts lower.
    """
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


def _edge_ref(binding: str, src_node: dict, src_type: str, slot: int, ctx: _RenderCtx) -> tuple[str, str | None]:
    """How to reference one output slot of an already-bound source node.

    Returns ``(ref, warning)``. ``warning`` is non-None only when litegraph
    serialized something other than a dict into ``outputs[slot]`` — the ref
    still falls back to the ``.out[<slot>]`` index form rather than raising.
    """
    outputs = src_node.get("outputs") or []
    if len(outputs) == 1:
        return binding, None
    out_name = None
    warning = None
    if 0 <= slot < len(outputs):
        entry = outputs[slot]
        if isinstance(entry, dict):
            out_name = entry.get("name")
        else:
            warning = (
                f"node {ctx.qualify(src_node.get('id'))}: malformed outputs entry at slot {slot}; referenced by index"
            )
    if not out_name and ctx.graph is not None:
        m = ctx.graph.node(src_type)
        if m is not None and 0 <= slot < len(m.outputs):
            out_name = m.outputs[slot].name
    if out_name and isinstance(out_name, str) and _IDENT.match(out_name) and not keyword.iskeyword(out_name):
        return f"{binding}.{out_name}", warning
    return f"{binding}.out[{slot}]", warning


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
    """``{reroute_id: (src_id, src_slot)}`` from each Reroute's single input link.

    Cross-reference: workflow_to_api.py's ``_collect_reroute_sources`` (~line
    559) does the same collection for API-format conversion — the two must
    agree on what counts as a Reroute's source.
    """
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
    input; ``get_vars[get_id] = var`` for each GetNode.

    Cross-reference: workflow_to_api.py's ``_collect_get_set_mappings`` (~line
    578) does the same collection for API-format conversion — the two must
    agree on what a SetNode publishes and what a GetNode reads.
    """
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
    - ``("splice_cycle", node_id)`` — a Reroute/GetNode chain that loops back on
      itself, so it never bottoms out at a real value.

    Cross-reference: workflow_to_api.py's ``_Tracers.trace_reroute`` /
    ``trace_get_set`` (~line 659) walk the same chains for API-format
    conversion — the two must agree on how far a chain is followed and where
    it bottoms out.
    """
    seen: set[str] = set()
    while True:
        key = str(src_id)
        if proxy_in_id is not None and key == proxy_in_id:
            return ("in_proxy", src_slot)
        node = nodes_by_id.get(key)
        t = node.get("type") if node else None
        if t == "Reroute":
            if key in seen:
                return ("splice_cycle", key)
            seen.add(key)
            if key in reroute_sources:
                src_id, src_slot = reroute_sources[key]
                continue
            return ("dead_reroute", key)
        if t == "GetNode":
            if key in seen:
                return ("splice_cycle", key)
            seen.add(key)
            var = get_vars.get(key)
            if var in set_sources:
                src_id, src_slot = set_sources[var]
                continue
            return ("missing_set", key, var)
        if t == "PrimitiveNode":
            return ("primitive_no_widget", key)
        return ("ok", src_id, src_slot)


def _splice_link_map(
    link_map: dict[str, tuple],
    nodes_by_id: dict[str, dict],
    reroute_sources: dict[str, tuple],
    set_sources: dict[str, tuple],
    get_vars: dict[str, str],
    proxy_in_id: str | None,
) -> dict[str, tuple]:
    """The dependency graph ``_toposort`` should use: each link's source
    resolved past any Reroute/GetNode-SetNode splice (and the ``-10`` proxy,
    which never contributes a dependency). A link whose source doesn't
    resolve to a real node (dead-end Reroute, unpublished GetNode variable,
    PrimitiveNode) is dropped — there's no real upstream to order against."""
    resolved: dict[str, tuple] = {}
    for lid, (src_id, src_slot, tgt_id, tgt_slot) in link_map.items():
        outcome = _resolve_source(src_id, src_slot, nodes_by_id, reroute_sources, set_sources, get_vars, proxy_in_id)
        if outcome[0] != "ok":
            continue
        resolved[lid] = (outcome[1], outcome[2], tgt_id, tgt_slot)
    return resolved


@dataclass
class _RenderCtx:
    """Everything the per-node arg builders need to resolve one edge, for
    either the top-level workflow (``proxy_in_id``/``addr_prefix`` are
    ``None``) or one subgraph definition's interior (``proxy_in_id`` is
    ``"-10"``, ``addr_prefix`` is this block's own fully-qualified address —
    e.g. ``"10"`` or ``"10/3"`` for a definition nested inside another)."""

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
    addr_prefix: str | None = None
    # The definition whose interior this context renders (``None`` at top
    # level). A nested subgraph instance registers against it so its own
    # address can later be expanded through every instance of THIS definition.
    owner_def: str | None = None

    def proxy_in_name(self, slot: Any) -> str:
        name = self.proxy_in_names.get(slot)
        return name if name else f"slot{slot}"

    def in_ref(self, slot: Any) -> str:
        """``IN.<name>`` / ``IN["<name>"]`` for one input-proxy slot."""
        return _member_ref("IN", self.proxy_in_name(slot))

    def qualify(self, nid: Any) -> str:
        """This block's fully-qualified address for a bare local node id —
        identity at the top level, ``"<addr_prefix>/<nid>"`` inside a
        definition (chaining through every nesting level, matching
        ``workflow_ops._navigate_subgraph_path``). Used for bindings, skipped
        ids, and warning text — NOT for a line's own trailing ``# <id>``
        comment or its annotations, which stay bare inner ids by design."""
        nid = str(nid)
        return f"{self.addr_prefix}/{nid}" if self.addr_prefix else nid


def _resolve_edge_text(src_id: Any, src_slot: Any, name: str, tgt_id: str, ctx: _RenderCtx) -> tuple:
    """Resolve one edge to ``(ref_text, annotation, warning)``. ``ref_text`` is
    ``None`` when the edge can't be resolved to a value (prints as ``None``).
    ``tgt_id`` is the bare id of the node whose input this is; warning text
    (which ends up in ``PrintResult.warnings``) is qualified via
    ``ctx.qualify`` — the ``annotation`` (which becomes part of the printed
    line itself) is not, matching the "bare inner ids in comments" rule."""
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
        ref, edge_warning = _edge_ref(binding, src_node, src_node.get("type", ""), rslot, ctx)
        # R12 (amending D8): the link still prints as wired — the marker only
        # tells the reader the value it carries comes from a bypassed node, so
        # at runtime it is really whatever that node passes through.
        ann = f" {name} via bypassed {rid_s}" if src_node.get("mode") == 4 else None
        return ref, ann, edge_warning
    if kind == "in_proxy":
        return ctx.in_ref(outcome[1]), None, None
    if kind == "dead_reroute":
        return None, f" {name} unlinked via reroute {outcome[1]}", None
    if kind == "splice_cycle":
        return (
            None,
            None,
            f"node {ctx.qualify(tgt_id)}: input '{name}' unresolved through a reroute/getnode cycle",
        )
    if kind == "missing_set":
        get_id, var = outcome[1], outcome[2]
        return (
            None,
            None,
            f"node {ctx.qualify(tgt_id)}: input '{name}' reads GetNode {ctx.qualify(get_id)} variable '{var}' "
            "that no SetNode publishes",
        )
    if kind == "primitive_no_widget":
        return (
            None,
            None,
            f"node {ctx.qualify(tgt_id)}: input {name!r} fed by PrimitiveNode {ctx.qualify(outcome[1])} "
            "without a widget; printed as None",
        )
    return None, None, None


def _place_arg(name: str, text: str, args: list[str], extra: dict[str, str]) -> None:
    """Append ``name=text`` to ``args`` when ``name`` is a valid Python
    identifier, else collect it into ``extra`` for a trailing ``**{}``."""
    if _IDENT.match(name) and not keyword.iskeyword(name):
        args.append(f"{name}={text}")
    else:
        extra[name] = text


def _autogrow_annotation(group: Any, inputs: list[dict]) -> str:
    """The line-comment marker for one auto-grow group: its name, the socket
    type every grown slot carries, and the schema's capacity — so a reader
    knows the group exists (and what to wire into it) even before anything is
    connected. The element type comes from the schema template; a catalog
    that omits it falls back to the type litegraph stamped on a grown slot."""
    elem = group.autogrow_element_type
    if not elem:
        prefix = f"{group.name}."
        elem = next(
            (
                str(inp["type"])
                for inp in inputs
                if str(inp.get("name") or "").startswith(prefix) and isinstance(inp.get("type"), str) and inp["type"]
            ),
            "any",
        )
    _lo, hi = group.autogrow_limits
    return f" {group.name} grows {elem}" + (f" (max {hi})" if hi is not None else "")


def _ensure_open_autogrow_slot(group: Any, inputs: list[dict], extra: dict[str, str]) -> None:
    """Show ONE open slot for ``group`` when none of its serialized slots is
    free and the schema allows another — the frontend keeps exactly that free
    trailing slot on a loaded node, so a UI-built and an agent-built node
    (whose ``inputs[]`` only carry what ``connect`` grew) print alike, and the
    printed name is the one ``connect`` accepts next. Inserted right after the
    group's last existing slot so the group stays contiguous. Never touches the
    workflow itself."""
    from comfy_cli.workflow_ops import _autogrow_elem_name, _first_free_autogrow_index

    prefix = f"{group.name}."
    members = [str(inp.get("name") or "") for inp in inputs if str(inp.get("name") or "").startswith(prefix)]
    if any(extra.get(name) == "None" for name in members):
        return
    _lo, hi = group.autogrow_limits
    if hi is not None and len(members) >= hi:
        return
    template = group.autogrow_template
    n = _first_free_autogrow_index(set(members), group.name, template)
    slot = f"{group.name}.{_autogrow_elem_name(group.name, n, template)}"
    items = list(extra.items())
    last = max((i for i, (k, _v) in enumerate(items) if k in members), default=None)
    if last is None:
        extra[slot] = "None"
        return
    items.insert(last + 1, (slot, "None"))
    extra.clear()
    extra.update(items)


def _build_args(
    node: dict, m: Any, ctx: _RenderCtx, warnings: list[str]
) -> tuple[list[str], bool, list[str], list[tuple]]:
    """Node call arguments: link inputs (in the node's own input order,
    non-identifier names collected into a trailing ``**{}``) first, then
    widgets. Returns ``(args, unknown, annotations, prim_hits)`` where
    ``unknown`` is True when the class isn't in the catalog (or there is no
    catalog), ``annotations`` are suffix strings to append to the line
    comment, and ``prim_hits`` are ``(qualified_primitive_id, qualified_skip_reason)``
    pairs for PrimitiveNode-into-widget splices found here.

    Widgets print by their VALUE-AWARE names (``Graph.widget_order_for_node``):
    a dynamic combo's selector followed by the selected option's sub-widgets
    (``model.resolution``…), read at the frontend's positions. A link-only
    sub-input (a ``COMFY_AUTOGROW_V3`` group) owns no widget slot, so it never
    prints as a value: its grown slots print as links, one open slot is kept
    visible (``_ensure_open_autogrow_slot``), and the group itself is named in
    the line comment with its element type (``_autogrow_annotation``).
    """
    nid = str(node.get("id"))
    args: list[str] = []
    extra: dict[str, str] = {}
    annotations: list[str] = []
    prim_hits: list[tuple] = []

    # Normalised once so both passes below can assume a dict: litegraph has
    # been seen to serialize junk into ``inputs``, and a bare ``"widget" in inp``
    # or ``inp.get(...)`` on a str/int would traceback out of the whole render.
    inputs: list[dict] = []
    for inp in node.get("inputs") or []:
        if not isinstance(inp, dict):
            warnings.append(f"node {ctx.qualify(nid)}: ignoring malformed input entry {inp!r}")
            continue
        inputs.append(inp)

    class_type = node.get("type", "")
    widgets_values = node.get("widgets_values")
    unknown = m is None
    if unknown:
        positional = widgets_values if isinstance(widgets_values, list) else []
        groups: list = []
    else:
        positional = _widgets_as_positional(widgets_values, ctx.graph, class_type)
        # The auto-grow groups this node exposes for its CURRENT selection —
        # a group nested under a dynamic combo disappears when the selector
        # moves to an option without it.
        groups = ctx.graph.autogrow_groups(class_type, positional)
    group_names = {g.name for g in groups}

    # Pre-pass: widget-backed inputs. A LIVE link to a real, printable node is
    # the truth — it's printed as a ref in the link pass below and skipped in
    # the widget-values pass. A link that resolves to the subgraph input
    # proxy is similarly a real (if opaque) upstream value. A PrimitiveNode
    # source is the one case that keeps the node's own static widget value
    # (that's what ComfyUI itself shows), annotated instead of substituted. A
    # dead-end (unlinked Reroute, unpublished SetNode variable) falls through
    # to the static value too — there's nothing better to show.
    live_link_refs: dict[str, str] = {}
    widget_overrides: dict[str, str] = {}
    for inp in inputs:
        if "widget" not in inp:
            continue
        name = inp.get("name") or ""
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
        if outcome[0] == "ok":
            rid_s = str(outcome[1])
            if rid_s in ctx.printable_ids:
                src_node = ctx.nodes_by_id.get(rid_s)
                binding = ctx.binding_by_id.get(rid_s)
                if src_node is not None and binding is not None:
                    edge_ref, edge_warning = _edge_ref(binding, src_node, src_node.get("type", ""), outcome[2], ctx)
                    live_link_refs[name] = edge_ref
                    if edge_warning:
                        warnings.append(edge_warning)
                    if src_node.get("mode") == 4:  # R12, same marker as the link pass
                        annotations.append(f" {name} via bypassed {rid_s}")
        elif outcome[0] == "in_proxy":
            widget_overrides[name] = ctx.in_ref(outcome[1])
        elif outcome[0] == "primitive_no_widget":
            prim_id = outcome[1]
            annotations.append(f" {name} from primitive {prim_id}")
            prim_hits.append((ctx.qualify(prim_id), f"inlined into {ctx.qualify(nid)}.{name}"))

    for inp in inputs:
        name = inp.get("name") or ""
        if "widget" in inp:
            if name in live_link_refs:
                _place_arg(name, live_link_refs[name], args, extra)
            continue  # otherwise: printed in the widget pass instead
        link_id = inp.get("link")
        # An auto-grow group's BASE entry (``images`` on an agent-built
        # BatchImagesNode) is the group itself, not a wireable slot: its slots
        # are the grown ``images.image0`` entries, handled with the group below.
        if link_id is None and name in group_names:
            continue
        if link_id is None:
            _place_arg(name, "None", args, extra)
            continue
        link = ctx.link_map.get(str(link_id))
        if link is None:
            _place_arg(name, "None", args, extra)
            continue
        src_id, src_slot, _tgt_id, _tgt_slot = link
        ref, ann, warn = _resolve_edge_text(src_id, src_slot, name, nid, ctx)
        if warn:
            warnings.append(warn)
        if ann:
            annotations.append(ann)
        _place_arg(name, ref if ref is not None else "None", args, extra)

    for group in groups:
        _ensure_open_autogrow_slot(group, inputs, extra)
        annotations.append(_autogrow_annotation(group, inputs))

    if unknown:
        if positional:
            args.append(f"widgets={py_literal(positional)}")
        warnings.append(f"node {ctx.qualify(nid)}: class {class_type!r} not in catalog; widgets printed positionally")
    else:
        entries = _expand_widget_entries(m, positional)
        for idx, entry in enumerate(entries):
            if entry.frontend_injected:
                # ``upload`` / ``audioUI`` / PREVIEW_3D ``image``: a frontend
                # button or DOM slot with no schema port and no editable
                # value (``slots`` omits it, every write surface refuses it).
                # It owns a positional slot, so it is walked — never printed.
                continue
            arg_name = entry.name
            if arg_name in live_link_refs:
                continue  # already printed in the link pass above
            if arg_name in widget_overrides:
                _place_arg(arg_name, widget_overrides[arg_name], args, extra)
                continue
            # R11 (amending D4): a dynamic combo's sub-widget the frontend never
            # serialized a value for prints the catalog's own default, not a
            # bare None that reads as "explicitly unset".
            if idx < len(positional):
                value = positional[idx]
            elif entry.port is not None and entry.port.options.default is not None:
                value = entry.port.options.default
            else:
                value = None
            _place_arg(arg_name, py_literal(value), args, extra)

    # Flushed once, AFTER the widget pass: a dynamic combo's sub-widget name is
    # dotted ("model.size_preset"), which is no more a Python identifier than a
    # dotted link-input name — both belong in the same trailing **{...}.
    if extra:
        inner = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {v}" for k, v in extra.items())
        args.append(f"**{{{inner}}}")
    return args, unknown, annotations, prim_hits


# ---------------------------------------------------------------------------
# D10: subgraph instances and definition blocks
# ---------------------------------------------------------------------------


def _subgraph_instance_name(node: dict, sg_def: dict | None, type_uuid: str) -> str:
    """The identity string for a subgraph instance — used for both its
    ``Subgraph[...]`` bracket and its binding name: the instance's own title,
    else the definition's own name, else the definition's uuid."""
    title = node.get("title") or None
    if title:
        return title
    if sg_def is not None:
        def_name = sg_def.get("name") or None
        if def_name:
            return def_name
    return type_uuid


def _promoted_widget_link_text(
    iname: str,
    link_id: Any,
    nid: str,
    ctx: _RenderCtx,
    annotations: list[str],
    prim_hits: list[tuple],
    warnings: list[str],
) -> str | None:
    """What an outside link into a promoted WIDGET input prints as — the same
    rules a regular node's widget-backed input follows (``_build_args``'s
    pre-pass): a live link to a printable node is the value (its ref), the
    enclosing definition's input proxy is ``IN.<name>``, a PrimitiveNode keeps
    the stored value with a ``from primitive`` marker, and a dead-end Reroute
    keeps it with an ``unlinked`` marker. ``None`` means "print the value"."""
    link = ctx.link_map.get(str(link_id))
    if link is None:
        return None
    src_id, src_slot, _tgt_id, _tgt_slot = link
    outcome = _resolve_source(
        src_id, src_slot, ctx.nodes_by_id, ctx.reroute_sources, ctx.set_sources, ctx.get_vars, ctx.proxy_in_id
    )
    if outcome[0] == "ok":
        rid_s = str(outcome[1])
        if rid_s not in ctx.printable_ids:
            return None
        src_node = ctx.nodes_by_id.get(rid_s)
        binding = ctx.binding_by_id.get(rid_s)
        if src_node is None or binding is None:
            return None
        ref, edge_warning = _edge_ref(binding, src_node, src_node.get("type", ""), outcome[2], ctx)
        if edge_warning:
            warnings.append(edge_warning)
        if src_node.get("mode") == 4:
            annotations.append(f" {iname} via bypassed {rid_s}")
        return ref
    if outcome[0] == "in_proxy":
        return ctx.in_ref(outcome[1])
    if outcome[0] == "primitive_no_widget":
        annotations.append(f" {iname} from primitive {outcome[1]}")
        prim_hits.append((ctx.qualify(outcome[1]), f"inlined into {ctx.qualify(nid)}.{iname}"))
    elif outcome[0] == "dead_reroute":
        annotations.append(f" {iname} unlinked via reroute {outcome[1]}")
    return None


def _render_subgraph_instance_line(
    node: dict, type_uuid: str, sg_def: dict, binding: str, ctx: _RenderCtx, state: _State
) -> tuple:
    """A subgraph instance's own line, argument per declared subgraph input in
    declaration order (non-identifier names collected into a trailing
    ``**{}``):

    * a promoted WIDGET input (``cql.promoted``: a declared input a boundary
      link feeds into an interior widget) prints its EFFECTIVE value — the
      host instance's own value when materialized, else the interior default
      — under the name ``set-widget <instance>.<name>`` takes. The host's
      positional ``widgets_values`` are read the way the frontend reads them
      (one slot per widget-backed input, in declaration order), never by the
      instance's serialized ``inputs[]``, which only lists what the UI showed.
      An outside link into it prints as that link instead (the link is what
      runs), with the same PrimitiveNode/Reroute markers a regular node gets;
    * a socket input prints its link (or ``None``).

    A declared input no boundary link backs falls back to the legacy
    ``proxyWidgets`` route, exactly as ``slots`` does. Any instance
    ``inputs[]`` entry the definition does not declare prints last, by link.
    Gets the same ``mode=`` suffix and D9 annotations as a regular node.
    Returns ``(line, prim_hits)``."""
    nid = str(node.get("id"))
    name_str = _subgraph_instance_name(node, sg_def, type_uuid)
    bracket = json.dumps(name_str, ensure_ascii=False)
    prim_hits: list[tuple] = []
    annotations: list[str] = []
    args: list[str] = []
    extra: dict[str, str] = {}
    warnings = state.warnings

    inputs = [inp for inp in node.get("inputs") or [] if isinstance(inp, dict)]
    by_name: dict[str, dict] = {}
    for inp in inputs:
        iname = inp.get("name")
        if isinstance(iname, str) and iname not in by_name:
            by_name[iname] = inp
    rendered: set[str] = set()

    def socket_text(iname: str, link_id: Any) -> str:
        link = ctx.link_map.get(str(link_id)) if link_id is not None else None
        if link is None:
            return "None"
        src_id, src_slot, _tgt_id, _tgt_slot = link
        ref, ann, warn = _resolve_edge_text(src_id, src_slot, iname, nid, ctx)
        if warn:
            warnings.append(warn)
        if ann:
            annotations.append(ann)
        return ref if ref is not None else "None"

    for pi in _promoted.promoted_inputs(sg_def, state.promoted_defs):
        rendered.add(pi.name)
        entry = by_name.get(pi.name)
        link_id = entry.get("link") if entry is not None else None
        text = None
        if pi.is_widget:
            if link_id is not None:
                text = _promoted_widget_link_text(pi.name, link_id, nid, ctx, annotations, prim_hits, warnings)
            if text is None:
                # ``pi`` and the definition index are this loop's own; the
                # name-lookup entry point would re-derive both per widget.
                value = _promoted.effective_value_for(state.workflow, node, sg_def, pi, ctx.graph, state.promoted_defs)
                text = py_literal(None if value is _promoted.UNSET else value)
        elif pi.source_node is None and link_id is None and ctx.graph is not None:
            # Declared but backed by no boundary link: a legacy template still
            # routes it through ``proxyWidgets`` to an interior widget — the
            # same fallback ``slots`` applies (``_declared_subgraph_slots``).
            value = _resolve_proxy_value(node, sg_def, pi.name, ctx.graph)
            if value is not _UNRESOLVED:
                text = py_literal(value)
        if text is None:
            text = socket_text(pi.name, link_id)
        _place_arg(pi.name, text, args, extra)

    for inp in inputs:
        iname = inp.get("name") or ""
        if iname in rendered:
            continue
        rendered.add(iname)
        link_id = inp.get("link")
        text = None
        if "widget" in inp and link_id is not None:
            text = _promoted_widget_link_text(iname, link_id, nid, ctx, annotations, prim_hits, warnings)
            if text is None:
                text = "None"
        else:
            text = socket_text(iname, link_id)
        _place_arg(iname, text, args, extra)

    if extra:
        inner = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {v}" for k, v in extra.items())
        args.append(f"**{{{inner}}}")

    line = f"{binding} = Subgraph[{bracket}]({', '.join(args)})  # {nid} subgraph {type_uuid}"
    mode = node.get("mode")
    if mode in _MODE_LABELS:
        line += f" mode={_MODE_LABELS[mode]}"
    for ann in annotations:
        line += ann
    return line, prim_hits


def _render_missing_subgraph_line(
    node: dict, type_uuid: str, binding: str, ctx: _RenderCtx, warnings: list[str]
) -> str:
    """D11 (amended): a subgraph instance whose definition is missing is never
    a refusal, at any depth — it's printed opaquely instead. Wired link inputs
    resolve by the instance's own input name (non-identifier names via
    ``**{}``, same as a resolved instance); widget-backed instance inputs
    print positionally as a single ``widgets=[...]`` (there's no definition to
    name them against). Gets the same ``mode=bypass``/``mode=mute`` suffix as a
    resolved instance. Any D9 annotation is appended, same as a resolved
    instance. The warning uses this node's fully-qualified address."""
    nid = str(node.get("id"))
    args: list[str] = []
    extra: dict[str, str] = {}
    annotations: list[str] = []

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
                ref, ann, warn = _resolve_edge_text(src_id, src_slot, iname, nid, ctx)
                if warn:
                    warnings.append(warn)
                if ann:
                    annotations.append(ann)
        _place_arg(iname, ref if ref is not None else "None", args, extra)

    if extra:
        inner = ", ".join(f"{json.dumps(k, ensure_ascii=False)}: {v}" for k, v in extra.items())
        args.append(f"**{{{inner}}}")

    widgets_values = node.get("widgets_values")
    if isinstance(widgets_values, list) and widgets_values:
        args.append(f"widgets={py_literal(widgets_values)}")

    warnings.append(f"node {ctx.qualify(nid)}: subgraph definition {type_uuid} missing; printed opaquely")
    call = f"{binding} = Subgraph[{json.dumps(type_uuid)}]({', '.join(args)})"
    line = f"{call}  # {nid} subgraph {type_uuid} definition missing"
    mode = node.get("mode")
    if mode in _MODE_LABELS:
        line += f" mode={_MODE_LABELS[mode]}"
    for ann in annotations:
        line += ann
    return line


def _definition_header(def_id: str, sg_def: dict, state: _State) -> str:
    """Called only after every block has been rendered (so nested instances
    discovered while rendering other blocks are already registered) — lists
    every fully-qualified instance address, not just the first, and spells out
    the form of BOTH the inner-node addresses and the ``bindings`` keys (they
    differ: one ends in the inner id, the other in the binding name)."""
    name = sg_def.get("name") or def_id
    instances = sorted(set(state.instance_addresses(def_id)), key=_sort_key)
    first = state.first_instance_by_def.get(def_id, instances[0] if instances else def_id)
    return (
        f"# subgraph {def_id} {json.dumps(name, ensure_ascii=False)} — "
        f"instances: {', '.join(instances)} "
        f"(address inner nodes as {first}/<id>; bindings keyed {first}/<name>)"
    )


def _promoted_header_line(def_id: str, sg_def: dict, state: _State) -> str | None:
    """The line under a definition's header that names its promoted WIDGETS —
    the ``IN.<name>`` references the interior lines below carry. Their values
    live on the instance (``cql.promoted``), so it spells out the address to
    edit (``57.<name>``) and the one NOT to (``57/<id>.<name>``, the interior
    widget the host value overrides). ``None`` when nothing is promoted as a
    widget (socket-only inputs are links, not values)."""
    pis = [p for p in _promoted.promoted_inputs(sg_def, state.promoted_defs) if p.is_widget]
    if not pis:
        return None
    first = state.first_instance_by_def.get(def_id, def_id)
    names = ", ".join(_member_ref("IN", p.name) for p in pis)
    return (
        f"#   promoted widgets: {names} — each is the instance's own value: "
        f"edit {first}.<name>, never {first}/<id>.<name>"
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
        self,
        defs_by_id: dict[str, dict],
        warnings: list[str],
        skipped: list[dict],
        bindings: dict[str, str],
        workflow: dict | None = None,
    ) -> None:
        self.defs_by_id = defs_by_id
        # The whole workflow and cql.promoted's own definition index: a
        # promoted widget's effective value is resolved from the HOST instance
        # against its definition (``promoted.effective_value_for``), at any depth.
        self.workflow: dict = workflow if workflow is not None else {}
        self.promoted_defs: dict[str, dict] = _promoted.defs_by_id(self.workflow)
        self.warnings = warnings
        self.skipped = skipped
        self.bindings = bindings
        self.def_order: list[str] = []
        self.def_seen: set[str] = set()
        self.def_depth: dict[str, int] = {}
        # def_id -> [(owning definition id or None at top level, bare local id)].
        # Stored RELATIVE, not pre-qualified: a definition's interior is rendered
        # exactly once (under its first instance's address), so an instance
        # discovered inside a definition that is itself instantiated N times
        # stands for N real addresses. instance_addresses() expands that.
        self.instance_sites: dict[str, list[tuple[str | None, str]]] = {}
        self.first_instance_by_def: dict[str, str] = {}
        self.primitive_reason: dict[str, str] = {}

    def register_instance(
        self, def_id: str, owner_def: str | None, local_id: Any, instance_addr: str, depth: int
    ) -> None:
        """``owner_def``/``local_id`` locate the instance relative to whatever
        block it was found in; ``instance_addr`` is the fully-qualified address
        that block itself rendered under (``"10"`` at top level, ``"10/3"``
        nested one level) and is what interior addressing keys off."""
        self.instance_sites.setdefault(def_id, []).append((owner_def, str(local_id)))
        if def_id not in self.def_seen:
            self.def_seen.add(def_id)
            self.first_instance_by_def[def_id] = instance_addr
            self.def_depth[def_id] = depth
            self.def_order.append(def_id)

    def instance_addresses(self, def_id: str, _seen: frozenset = frozenset()) -> list[str]:
        """Every fully-qualified address ``def_id`` is instantiated at, expanding
        each ancestor definition's own instance list — Outer instantiated at 100
        and 200, each holding Inner at inner id 5, gives ``100/5`` AND ``200/5``.
        ``_seen`` guards a malformed self-referential definition chain."""
        if def_id in _seen:
            return []
        seen = _seen | {def_id}
        out: list[str] = []
        for owner_def, local_id in self.instance_sites.get(def_id, []):
            if owner_def is None:
                out.append(local_id)
                continue
            out.extend(f"{parent}/{local_id}" for parent in self.instance_addresses(owner_def, seen))
        return out


def _render_node_line(n: dict, name: str, ctx: _RenderCtx, graph: Graph | None, state: _State) -> str:
    """One printed line for a regular (non-subgraph-instance) node."""
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
) -> list[str]:
    """Render one printed line per node in ``order``: a resolved subgraph
    instance gets the ``Subgraph[...]`` format, an instance whose definition
    is missing gets the opaque D11 fallback (never a refusal), everything else
    is a regular class call. Shared between the top-level workflow and a
    definition's interior — ``ctx.addr_prefix`` (this block's own
    fully-qualified address) is what makes a nested instance's address and a
    nested skip/warning fully qualified through every level."""
    lines: list[str] = []
    for n in order:
        nid = str(n.get("id"))
        t = n.get("type", "")
        name = binding_by_id[nid]
        if is_subgraph_uuid(t):
            sg_def = defs_by_id.get(t)
            addr = ctx.qualify(nid)
            if sg_def is not None:
                line, prim_hits = _render_subgraph_instance_line(n, t, sg_def, name, ctx, state)
                for prim_id, reason in prim_hits:
                    state.primitive_reason.setdefault(prim_id, reason)
                state.register_instance(t, ctx.owner_def, nid, addr, depth)
            else:
                line = _render_missing_subgraph_line(n, t, name, ctx, state.warnings)
        else:
            line = _render_node_line(n, name, ctx, graph, state)
        lines.append(line)
    return lines


def _build_bindings(order: list[dict], defs_by_id: dict[str, dict]) -> dict[str, str]:
    """Local binding names for one node list, in print order: a subgraph
    instance (resolved or not) binds off the same identity string as its
    bracket — its own title, else its definition's name, else its
    definition's uuid; everything else binds off its class type, as usual."""
    used: dict[str, int] = {}
    binding_by_id: dict[str, str] = {}
    for n in order:
        nid = str(n.get("id"))
        t = n.get("type", "")
        if is_subgraph_uuid(t):
            name_str = _subgraph_instance_name(n, defs_by_id.get(t), t)
            name = binding_name(name_str, used)
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
    declared outputs. Returns ``(lines, printable_node_count)``.

    No depth cap: ``state.def_seen`` (see ``_State.register_instance``) already
    dedupes a definition to its first occurrence in ``state.def_order``, so a
    self-referential (or mutually-referential) subgraph chain can't grow that
    queue without bound — depth only grows with genuinely distinct nesting
    levels, and capping it would misfire on legitimate deep nesting instead of
    catching anything a cycle could cause."""
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

    nodes_by_id = {str(n.get("id")): n for n in interior_nodes}
    printable_ids = {str(n.get("id")) for n in printable}

    # Collected BEFORE toposort so a splice contributes a resolved dependency
    # edge instead of silently dropping the ordering constraint (item 1).
    reroute_sources = _collect_reroute_sources(interior_nodes, all_links)
    set_sources, get_vars = _collect_get_set(interior_nodes, all_links)
    toposort_links = _splice_link_map(all_links, nodes_by_id, reroute_sources, set_sources, get_vars, _PROXY_IN)
    order = _toposort(printable, toposort_links)

    proxy_in_names = {i: inp.get("name") for i, inp in enumerate(sg_def.get("inputs") or []) if isinstance(inp, dict)}
    proxy_out_names = {i: o.get("name") for i, o in enumerate(sg_def.get("outputs") or []) if isinstance(o, dict)}

    binding_by_id = _build_bindings(order, state.defs_by_id)
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
        addr_prefix=first_instance,
        owner_def=def_id,
    )

    lines = _render_nodes(order, ctx, graph, state.defs_by_id, binding_by_id, state, depth + 1)

    for n in sorted(notes, key=lambda n: _sort_key(str(n.get("id")))):
        lines.append(_note_comment(n))
        state.skipped.append({"id": ctx.qualify(n.get("id")), "type": n.get("type"), "reason": "note"})
    for n in ui_only_skipped:
        t = n.get("type")
        addr = ctx.qualify(n.get("id"))
        reason = state.primitive_reason.get(addr, "spliced") if t == "PrimitiveNode" else "spliced"
        state.skipped.append({"id": addr, "type": t, "reason": reason})

    out_sources: dict[Any, tuple] = {}
    for oid, oslot, tid, tslot in all_links.values():
        if str(tid) != _PROXY_OUT:
            continue
        if tslot not in out_sources:
            out_sources[tslot] = (oid, oslot)
    for slot in sorted(out_sources, key=_sort_key):
        oid, oslot = out_sources[slot]
        out_name = proxy_out_names.get(slot) or f"out{slot}"
        # "OUT", not a pre-qualified label: _resolve_edge_text runs the target
        # through ctx.qualify(), which prefixes this block's own address — so
        # anything already carrying it comes back doubled ("11/11 OUT").
        ref, _ann, warn = _resolve_edge_text(oid, oslot, out_name, "OUT", ctx)
        if warn:
            state.warnings.append(warn)
        lines.append(f"{_member_ref('OUT', out_name)} = {ref if ref is not None else 'None'}")

    for nid, name in binding_by_id.items():
        state.bindings[f"{first_instance}/{name}"] = f"{first_instance}/{nid}"

    return lines, len(printable)


def render_py(workflow: dict, graph: Graph | None) -> PrintResult:
    warnings: list[str] = []

    raw_nodes = workflow.get("nodes") or []
    nodes = [n for n in raw_nodes if isinstance(n, dict)]
    if len(nodes) != len(raw_nodes):
        warnings.append(f"workflow: ignoring {len(raw_nodes) - len(nodes)} non-object node entries")

    links = workflow.get("links") or []

    definitions = workflow.get("definitions")
    promoted_workflow = workflow
    if definitions is not None and not isinstance(definitions, dict):
        warnings.append("workflow: ignoring non-object definitions block")
        definitions = None
        # cql.promoted indexes the same block; hand it the sanitized view.
        promoted_workflow = {**workflow, "definitions": None}
    subgraphs = definitions.get("subgraphs") if definitions else None
    defs_by_id = {sg.get("id"): sg for sg in (subgraphs or []) if isinstance(sg, dict) and sg.get("id")}

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

    nodes_by_id = {str(n.get("id")): n for n in nodes}
    printable_ids = {str(n.get("id")) for n in printable}

    # Collected BEFORE toposort — see _render_definition_block and item 1.
    reroute_sources = _collect_reroute_sources(nodes, link_map)
    set_sources, get_vars = _collect_get_set(nodes, link_map)
    toposort_links = _splice_link_map(link_map, nodes_by_id, reroute_sources, set_sources, get_vars, None)
    order = _toposort(printable, toposort_links)

    # binding_name is called once per node, in topological order, so the
    # dedupe suffixes (_2, _3, ...) match print order.
    binding_by_id = _build_bindings(order, defs_by_id)
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

    skipped: list[dict] = []
    state = _State(defs_by_id, warnings, skipped, bindings, promoted_workflow)

    lines = _render_nodes(order, ctx, graph, defs_by_id, binding_by_id, state, depth=1)

    for n in sorted(notes, key=lambda n: _sort_key(str(n.get("id")))):
        lines.append(_note_comment(n))
        skipped.append({"id": ctx.qualify(n.get("id")), "type": n.get("type"), "reason": "note"})
    for n in ui_only_skipped:
        t = n.get("type")
        addr = ctx.qualify(n.get("id"))
        reason = state.primitive_reason.get(addr, "spliced") if t == "PrimitiveNode" else "spliced"
        skipped.append({"id": addr, "type": t, "reason": reason})

    # Render every definition block first (this may discover further nested
    # definitions, appended to state.def_order and picked up by this same
    # loop) — only THEN emit headers, so each header's instance list is
    # complete rather than reporting only what was known when it was reached.
    node_count = len(printable)
    block_lines_by_def: dict[str, list[str]] = {}
    for def_id in state.def_order:
        sg_def = defs_by_id[def_id]
        block_lines, block_count = _render_definition_block(def_id, sg_def, graph, state, state.def_depth[def_id])
        block_lines_by_def[def_id] = block_lines
        node_count += block_count
    for def_id in state.def_order:
        lines.append(_definition_header(def_id, defs_by_id[def_id], state))
        promoted_line = _promoted_header_line(def_id, defs_by_id[def_id], state)
        if promoted_line is not None:
            lines.append(promoted_line)
        lines.extend("    " + bl for bl in block_lines_by_def[def_id])

    source = "\n".join(lines) + "\n"
    return PrintResult(source=source, bindings=bindings, node_count=node_count, skipped=skipped, warnings=warnings)
