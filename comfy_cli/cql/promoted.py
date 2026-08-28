"""Promoted subgraph widgets — the host-owned value model.

The frontend (``ComfyUI_frontend`` ADR 0009 *"Subgraph promoted widgets use
linked inputs"*, ``SubgraphNode.ts``) represents a promoted widget as a
**linked subgraph input**:

* the subgraph definition declares the input (``definitions.subgraphs[].inputs``);
* a boundary link (origin ``-10``, the subgraph input node) feeds it into an
  interior node's widget-backed input (an ``inputs[]`` entry carrying a
  ``widget`` marker), or into a nested subgraph instance's own promoted input;
* the HOST instance owns the value: ``widgets_values[i]`` on the instance is
  consumed positionally by the i-th subgraph input that resolves to a widget
  (``_applyPromotedWidgetValues`` on load, ``serializeFromStoreState`` on
  save). Socket-only inputs (``VIDEO``, ``MODEL``, an unlinked declaration)
  own no slot.

The interior widget is only a schema/default provider — *"the host/exterior
value wins over the interior/source value during repair, persistence, and
prompt serialization"*. ``properties.proxyWidgets`` is legacy load-time input
the frontend consumes on migration; it is honored here only as a fallback for
a name the definition does not declare as an input.

Everything in this module is a pure function over workflow JSON. It never
mutates a subgraph definition: a promoted value is edited on the instance,
so sibling instances of a shared definition stay independent by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

#: LiteGraph's id for the virtual subgraph *input* node inside a definition.
SUBGRAPH_INPUT_NODE_ID = -10

#: Recursion cap for nested promotion chains (an inner instance promoting a
#: deeper instance's input). Mirrors the frontend's own bounded traversal.
_MAX_NESTED_PROMOTION_DEPTH = 32

#: Sentinel: "no value is materialized here" (distinct from a stored ``None``).
UNSET: Any = object()


@dataclass(frozen=True)
class PromotedInput:
    """One declared subgraph input, as the host instance sees it.

    ``value_index`` is the position in the host's ``widgets_values`` (``None``
    for a socket-only input). ``source_node``/``source_input`` name the
    interior input the first resolving boundary link targets;
    ``source_widget`` is that input's widget name (``None`` when the source is
    a nested instance's promoted input — ``nested`` is then ``True``).
    """

    name: str
    type: str
    index: int
    value_index: int | None
    source_node: str | None = None
    source_input: str | None = None
    source_widget: str | None = None
    nested: bool = False

    @property
    def is_widget(self) -> bool:
        return self.value_index is not None


def defs_by_id(workflow: dict) -> dict[str, dict]:
    """Subgraph definitions keyed by id (with the engine's unique-name fallback)."""
    from comfy_cli.cql.engine import _subgraph_defs_by_id

    return _subgraph_defs_by_id(workflow)


def promoted_inputs(sg: dict, defs: dict[str, dict], depth: int = 0) -> list[PromotedInput]:
    """Every declared input of definition ``sg`` in declaration order, with the
    host value slot each widget-backed one owns — the frontend's own rule
    (``SubgraphNode._resolveInputWidget``): walk the input's ``linkIds`` in
    order and take the first boundary link whose interior target is a
    widget-backed input, or a nested instance's promoted widget input."""
    inner = {str(n.get("id")): n for n in sg.get("nodes") or [] if isinstance(n, dict)}
    # Only hashable ids can be looked up; a malformed (list/dict) id is skipped
    # rather than crashing conversion of the whole workflow.
    links = {
        link.get("id"): link
        for link in sg.get("links") or []
        if isinstance(link, dict) and isinstance(link.get("id"), (int, str))
    }
    out: list[PromotedInput] = []
    value_index = 0
    for idx, inp in enumerate(sg.get("inputs") or []):
        if not isinstance(inp, dict):
            continue
        name = str(inp.get("name") or "")
        type_id = inp.get("type")
        # A missing or non-string declared type is UNKNOWN (""), never the
        # repr of whatever was there — callers treat "" as "accept the source".
        type_str = type_id if isinstance(type_id, str) else ""
        source: tuple[str, str, str | None, bool] | None = None
        for link_id in inp.get("linkIds") or []:
            if not isinstance(link_id, (int, str)):
                continue
            link = links.get(link_id)
            if not isinstance(link, dict):
                continue
            target = inner.get(str(link.get("target_id")))
            if target is None:
                continue
            target_inputs = target.get("inputs") or []
            slot = link.get("target_slot")
            entry = target_inputs[slot] if isinstance(slot, int) and 0 <= slot < len(target_inputs) else None
            if not isinstance(entry, dict):
                continue
            inner_def = defs.get(str(target.get("type", "")))
            if inner_def is not None:
                # The target is itself a subgraph instance: its input entry
                # carries a widget marker when promoted, but the concrete
                # widget lives deeper — resolve through its own promotion.
                if depth < _MAX_NESTED_PROMOTION_DEPTH:
                    inner_by_name = {p.name: p for p in promoted_inputs(inner_def, defs, depth + 1)}
                    inner_pi = inner_by_name.get(str(entry.get("name")))
                    if inner_pi is not None and inner_pi.is_widget:
                        source = (str(target.get("id")), str(entry.get("name")), None, True)
                        break
                continue
            marker = entry.get("widget")
            if marker:
                widget_name = marker.get("name") if isinstance(marker, dict) else None
                source = (str(target.get("id")), str(entry.get("name")), str(widget_name or entry.get("name")), False)
                break
        if source is None:
            out.append(PromotedInput(name=name, type=type_str, index=idx, value_index=None))
            continue
        node_id, input_name, widget_name, nested = source
        out.append(
            PromotedInput(
                name=name,
                type=type_str,
                index=idx,
                value_index=value_index,
                source_node=node_id,
                source_input=input_name,
                source_widget=widget_name,
                nested=nested,
            )
        )
        value_index += 1
    return out


def find_promoted(sg: dict, defs: dict[str, dict], name: str) -> PromotedInput | None:
    return next((p for p in promoted_inputs(sg, defs) if p.name == name), None)


def instance_input(instance: dict, name: str) -> dict | None:
    """The instance's serialized ``inputs[]`` entry for subgraph input ``name``."""
    for entry in instance.get("inputs") or []:
        if isinstance(entry, dict) and entry.get("name") == name:
            return entry
    return None


def external_link(instance: dict, name: str) -> Any:
    """The link id feeding the instance's input ``name`` from outside, else None."""
    entry = instance_input(instance, name)
    return None if entry is None else entry.get("link")


def _quarantine_entries(instance: dict) -> list[dict]:
    raw = (instance.get("properties") or {}).get("proxyWidgetErrorQuarantine")
    return [e for e in raw if isinstance(e, dict)] if isinstance(raw, list) else []


def host_value(instance: dict, pi: PromotedInput) -> Any:
    """The value the host instance materialized for ``pi``, or :data:`UNSET`.

    Order is the frontend's (``_applyPromotedWidgetValues``): a quarantined
    legacy entry's ``hostValue`` for this name first, then the positional
    ``widgets_values`` slot.
    """
    if not pi.is_widget:
        return UNSET
    for entry in reversed(_quarantine_entries(instance)):
        original = entry.get("originalEntry")
        if (
            isinstance(original, list)
            and len(original) >= 2
            and str(original[0]) == "-1"
            and original[1] == pi.name
            and "hostValue" in entry
        ):
            return entry["hostValue"]
    values = instance.get("widgets_values")
    if isinstance(values, list) and pi.value_index < len(values):
        return values[pi.value_index]
    return UNSET


def _interior_widget_value(node: dict, widget: str, graph) -> Any:
    """Read ``widget`` off an interior node's positional ``widgets_values``."""
    from comfy_cli.cql.engine import _widgets_as_positional

    node_type = str(node.get("type", ""))
    values = _widgets_as_positional(node.get("widgets_values"), graph, node_type)
    order = graph.widget_order_for_node(node_type, values) if graph is not None else []
    if widget in order:
        idx = order.index(widget)
        return values[idx] if idx < len(values) else None
    return UNSET


def source_value(workflow: dict, sg: dict, pi: PromotedInput, graph, defs: dict[str, dict] | None = None) -> Any:
    """The interior value ``pi`` falls back to when the host has none: the
    interior widget, or — through a nested instance — that instance's own
    effective value. :data:`UNSET` when the source cannot be read."""
    if pi.source_node is None:
        return UNSET
    defs = defs if defs is not None else defs_by_id(workflow)
    inner = next((n for n in sg.get("nodes") or [] if str(n.get("id")) == pi.source_node), None)
    if inner is None:
        return UNSET
    if pi.nested:
        # The frontend seeds a host widget promoted THROUGH an inner instance
        # from the deepest concrete widget (``_resolveNestedPromotedSource`` →
        # ``resolveConcretePromotedWidget``), not from the inner host's value.
        inner_def = defs.get(str(inner.get("type", "")))
        if inner_def is None or pi.source_input is None:
            return UNSET
        inner_pi = find_promoted(inner_def, defs, pi.source_input)
        if inner_pi is None:
            return UNSET
        return source_value(workflow, inner_def, inner_pi, graph, defs)
    return _interior_widget_value(inner, str(pi.source_widget), graph)


def _effective(workflow: dict, instance: dict, sg: dict, name: str, graph, defs: dict[str, dict]) -> Any:
    pi = find_promoted(sg, defs, name)
    if pi is None:
        raise ValueError(f"{name!r} is not a promoted input of subgraph {sg.get('id')}")
    if not pi.is_widget:
        raise ValueError(f"promoted input {name!r} on subgraph node {instance.get('id')} is a link input, not a widget")
    value = host_value(instance, pi)
    if value is not UNSET:
        return value
    return source_value(workflow, sg, pi, graph, defs)


def effective_value(workflow: dict, instance: dict, name: str, graph) -> Any:
    """The value the frontend runs for promoted input ``name`` on ``instance``
    when nothing outside feeds it: the host value if materialized, else the
    interior source value (:data:`UNSET` if even that cannot be read).

    Raises ``ValueError`` for a name the definition does not declare, or one
    that is a socket-only input.
    """
    defs = defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    if sg is None:
        raise ValueError(f"node {instance.get('id')} is not a subgraph instance")
    return _effective(workflow, instance, sg, name, graph, defs)


def set_host_value(workflow: dict, instance: dict, name: str, value: Any, graph) -> Any:
    """Write ``value`` as the host-owned value of promoted input ``name``.

    Materializes the host's ``widgets_values`` to one entry per widget-backed
    input (in declaration order, seeded from each input's current effective
    value) so the positional array the frontend restores stays aligned, then
    sets the one slot. A quarantined legacy ``hostValue`` for the same name is
    rewritten too, because the frontend reads it first. The interior
    definition is never touched. Returns the previous effective value.
    """
    defs = defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    if sg is None:
        raise ValueError(f"node {instance.get('id')} is not a subgraph instance")
    pis = promoted_inputs(sg, defs)
    pi = next((p for p in pis if p.name == name), None)
    if pi is None:
        raise ValueError(f"{name!r} is not a promoted input of subgraph node {instance.get('id')}")
    if not pi.is_widget:
        raise ValueError(f"promoted input {name!r} on subgraph node {instance.get('id')} is a link input, not a widget")
    old = _effective(workflow, instance, sg, name, graph, defs)
    widget_backed = [p for p in pis if p.is_widget]
    current = instance.get("widgets_values")
    values = list(current) if isinstance(current, list) else []
    for other in widget_backed:
        if other.value_index >= len(values):
            seeded = host_value(instance, other)
            if seeded is UNSET:
                seeded = source_value(workflow, sg, other, graph, defs)
            values.append(None if seeded is UNSET else seeded)
    values[pi.value_index] = value
    instance["widgets_values"] = values
    for entry in _quarantine_entries(instance):
        original = entry.get("originalEntry")
        if isinstance(original, list) and len(original) >= 2 and str(original[0]) == "-1" and original[1] == name:
            if "hostValue" in entry:
                entry["hostValue"] = value
    return None if old is UNSET else old


def deepest_source(sg: dict, pi: PromotedInput, defs: dict[str, dict]) -> tuple[list[str], str] | None:
    """``(interior node path, widget name)`` of the concrete widget behind
    ``pi`` — through any chain of nested instances. ``None`` when unresolvable."""
    path: list[str] = []
    current_sg, current = sg, pi
    for _ in range(_MAX_NESTED_PROMOTION_DEPTH):
        if current.source_node is None:
            return None
        path.append(current.source_node)
        if not current.nested:
            return path, str(current.source_widget)
        inner = next((n for n in current_sg.get("nodes") or [] if str(n.get("id")) == current.source_node), None)
        inner_def = defs.get(str(inner.get("type", ""))) if inner is not None else None
        if inner_def is None or current.source_input is None:
            return None
        nxt = find_promoted(inner_def, defs, current.source_input)
        if nxt is None:
            return None
        current_sg, current = inner_def, nxt
    return None


def host_widgets_values(instance: dict) -> list[Any]:
    values = instance.get("widgets_values")
    return list(values) if isinstance(values, list) else []


# --------------------------------------------------------------------------- #
# Where a widget write lands
# --------------------------------------------------------------------------- #

#: Frontend virtual nodes that pass a value through (``Reroute``) or hold one
#: without a server schema (``PrimitiveNode``, ``widgets_values[0]``).
REROUTE_TYPE = "Reroute"
LEGACY_PRIMITIVE_TYPE = "PrimitiveNode"
_MAX_REROUTE_HOPS = 64


@dataclass
class WriteTarget:
    """The place a ``set-widget``/``set-slot`` address resolves to.

    ``kind`` is ``top`` (an ordinary node's widget — possibly the SOURCE node a
    promoted input is wired from), ``host`` (a subgraph instance's own value
    for a promoted input; ``segments`` is the instance path, one segment when
    top-level), ``interior`` (an unpromoted interior widget; ``segments`` is
    the interior node path), or ``legacy_primitive`` (a schema-less frontend
    ``PrimitiveNode``). ``redirected_from`` is the address the caller gave when
    it was not the effective one.
    """

    kind: str
    node: dict | None = None
    widget: str | None = None
    segments: list[str] | None = None
    redirected_from: str | None = None


def _find_top(workflow: dict, node_id: Any) -> dict | None:
    for n in workflow.get("nodes") or []:
        if isinstance(n, dict) and (n.get("id") == node_id or str(n.get("id")) == str(node_id)):
            return n
    return None


def _navigate(workflow: dict, segments: list[str], defs: dict[str, dict]) -> dict:
    node = _find_top(workflow, segments[0])
    if node is None:
        raise ValueError(f"node {segments[0]} not found in workflow")
    for seg in segments[1:]:
        sg = defs.get(str(node.get("type", "")))
        if sg is None:
            raise ValueError(f"node {node.get('id')} is not a subgraph; cannot descend to {seg!r}")
        node = next((n for n in sg.get("nodes") or [] if isinstance(n, dict) and str(n.get("id")) == str(seg)), None)
        if node is None:
            raise ValueError(f"interior node {seg} not found in subgraph {sg.get('id')}")
    return node


def legacy_proxy_interior(instance: dict, widget: str) -> str | None:
    """The interior node id a legacy ``proxyWidgets`` entry routes ``widget`` to."""
    for entry in (instance.get("properties") or {}).get("proxyWidgets") or []:
        if isinstance(entry, list) and len(entry) >= 2 and str(entry[1]) == widget and str(entry[0]) != "-1":
            return str(entry[0])
    return None


def resolve_write(
    workflow: dict, graph, segments: list[str], widget: str, *, redirected_from: str | None = None
) -> WriteTarget:
    """Resolve a widget address to the place whose value the graph runs.

    * ``<instance>.<promoted>`` — the host instance's own value (ADR 0009)…
      unless an outside link feeds that input, in which case the write follows
      the link to its source (a primitive's single widget, through reroutes);
      a source that is not a primitive is refused by name, because no widget
      write could take effect.
    * ``<instance>/<inner>.<widget>`` whose interior input is fed by the
      subgraph input node (``-10``) — the same host value, so it redirects
      there (``redirected_from`` records the given address). An interior
      widget fed by another interior node is refused the same way.
    * any other interior widget — written in the definition, as before.
    * a promoted name the definition does not declare as an input but the
      legacy ``proxyWidgets`` still route — the interior, as before.
    """
    defs = defs_by_id(workflow)
    if len(segments) > 1:
        target = _navigate(workflow, segments, defs)
        parent = _navigate(workflow, segments[:-1], defs)
        parent_def = defs.get(str(parent.get("type", "")))
        target_def = defs.get(str(target.get("type", "")))
        given = f"{'/'.join(segments)}.{widget}"
        entry = next(
            (
                i
                for i in target.get("inputs") or []
                if isinstance(i, dict)
                and (
                    (isinstance(i.get("widget"), dict) and i["widget"].get("name") == widget)
                    or (i.get("name") == widget and (i.get("widget") or target_def is not None))
                )
            ),
            None,
        )
        link_id = entry.get("link") if entry is not None else None
        if link_id is not None and parent_def is not None:
            link = next(
                (x for x in parent_def.get("links") or [] if isinstance(x, dict) and x.get("id") == link_id), None
            )
            if link is not None and link.get("origin_id") == SUBGRAPH_INPUT_NODE_ID:
                sg_inputs = parent_def.get("inputs") or []
                origin_slot = link.get("origin_slot")
                sg_input = (
                    sg_inputs[origin_slot]
                    if isinstance(origin_slot, int) and 0 <= origin_slot < len(sg_inputs)
                    else None
                )
                if isinstance(sg_input, dict) and sg_input.get("name"):
                    return resolve_write(
                        workflow, graph, segments[:-1], str(sg_input["name"]), redirected_from=redirected_from or given
                    )
            if link is not None:
                origin = link.get("origin_id")
                origin_node = next((n for n in parent_def.get("nodes") or [] if str(n.get("id")) == str(origin)), None)
                raise ValueError(
                    f"{given} is fed by interior node {'/'.join(segments[:-1])}/{origin} "
                    f"({origin_node.get('type') if origin_node else '?'}) — that link supplies the value, so a widget "
                    f"write there would be ignored; edit that node's own widgets instead (see `comfy workflow slots`)"
                )
        if target_def is not None:
            pi = find_promoted(target_def, defs, widget)
            if pi is not None and pi.is_widget:
                return WriteTarget(
                    "host", node=target, widget=widget, segments=segments, redirected_from=redirected_from
                )
        return WriteTarget("interior", widget=widget, segments=segments, redirected_from=redirected_from)

    node_str = segments[0]
    node = _find_top(workflow, node_str)
    if node is None:
        raise ValueError(f"node {node_str} not found in workflow")
    sg = defs.get(str(node.get("type", "")))
    if sg is None:
        return WriteTarget("top", node=node, widget=widget, redirected_from=redirected_from)
    given = f"{node_str}.{widget}"
    pi = find_promoted(sg, defs, widget)
    if pi is not None:
        if not pi.is_widget:
            legacy = legacy_proxy_interior(node, widget)
            if legacy is not None:
                return WriteTarget(
                    "interior", widget=widget, segments=[node_str, legacy], redirected_from=redirected_from
                )
            raise ValueError(
                f"promoted input {widget!r} on subgraph node {node_str} is a link input ({pi.type}), not a widget — "
                f"wire it with `comfy workflow connect <file> <node>.<output> {node_str}.{widget}`"
            )
        link_id = external_link(node, widget)
        if link_id is not None and _link_exists(workflow, link_id):
            return trace_upstream_write(workflow, graph, link_id, given, redirected_from=redirected_from or given)
        # No link, or a dangling link id the frontend drops on load: the host
        # value is what runs.
        return WriteTarget("host", node=node, widget=widget, segments=[node_str], redirected_from=redirected_from)
    legacy = legacy_proxy_interior(node, widget)
    if legacy is not None:
        return WriteTarget("interior", widget=widget, segments=[node_str, legacy], redirected_from=redirected_from)
    names = [p.name for p in promoted_inputs(sg, defs) if p.is_widget]
    raise ValueError(
        f"promoted input {widget!r} not found on subgraph node {node_str}; "
        f"promoted widgets: {', '.join(names) if names else '(none)'} "
        f"(or address an interior widget directly, e.g. {node_str}/<innerId>.<input>)"
    )


def _link_exists(workflow: dict, link_id: Any) -> bool:
    return any(isinstance(x, (list, tuple)) and x and x[0] == link_id for x in workflow.get("links") or [])


def trace_upstream_write(
    workflow: dict, graph, link_id: Any, given: str, *, redirected_from: str | None
) -> WriteTarget:
    """Follow the link feeding a promoted input to the widget that is its source
    of truth: a primitive's single value widget (``PrimitiveInt.value``, a
    legacy ``PrimitiveNode``), through any ``Reroute`` chain. Anything else
    computes the value (``ResolutionSelector``, ``GetImageSize``, another
    subgraph's output), so a widget write could not take effect — refused
    with the driver named so the caller edits it or rewires."""
    from comfy_cli.cql.engine import FRONTEND_MARKER_SLOTS, _widgets_as_positional

    defs = defs_by_id(workflow)
    links = workflow.get("links") or []
    for _ in range(_MAX_REROUTE_HOPS):
        link = next((x for x in links if isinstance(x, (list, tuple)) and len(x) >= 3 and x[0] == link_id), None)
        if link is None:
            raise ValueError(f"{given} is fed by link {link_id}, which does not exist in the workflow")
        src_id, src_slot = link[1], link[2]
        src = _find_top(workflow, src_id)
        if src is None:
            raise ValueError(f"{given} is fed by node {src_id}, which does not exist in the workflow")
        src_type = str(src.get("type", ""))
        if src_type == REROUTE_TYPE:
            inputs = src.get("inputs") or []
            link_id = inputs[0].get("link") if inputs and isinstance(inputs[0], dict) else None
            if link_id is None:
                raise ValueError(f"{given} is fed by Reroute {src_id}, which has no input — connect a primitive to it")
            continue
        if src_type == LEGACY_PRIMITIVE_TYPE:
            outputs = src.get("outputs") or []
            marker = outputs[0].get("widget") if outputs and isinstance(outputs[0], dict) else None
            name = marker.get("name") if isinstance(marker, dict) and marker.get("name") else "value"
            return WriteTarget("legacy_primitive", node=src, widget=str(name), redirected_from=redirected_from)
        m = graph.node(src_type) if graph is not None and src_type not in defs else None
        if m is not None:
            widgets = _widgets_as_positional(src.get("widgets_values"), graph, src_type)
            order = [n for n in graph.widget_order_for_node(src_type, widgets) if n not in FRONTEND_MARKER_SLOTS]
            if len(order) == 1:
                return WriteTarget("top", node=src, widget=order[0], redirected_from=redirected_from)
        raise ValueError(
            f"{given} is driven by node {src_id} ({src_type}) output {src_slot} — that link supplies the value, "
            f"so writing the widget would be ignored. Edit node {src_id}'s own widgets (see `comfy workflow slots`), "
            f"or connect a primitive (e.g. PrimitiveInt) to {given} and set its value."
        )
    raise ValueError(f"{given}: reroute chain too deep")
