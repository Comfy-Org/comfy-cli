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
the frontend consumes on migration. A legacy entry the definition does not
back with a linked input is repaired the way the frontend repairs it on load
(:func:`flush_proxy_migration`, the "forward migration" section below) before
a write lands on it; an entry the migration cannot repair is quarantined and
its interior widget stays the live one.

Everything in this module is a pure function over workflow JSON. A promoted
value is edited on the instance — the definition is touched only by the
forward migration, which the op layer runs on an isolated (forked) copy when
the definition is shared, so sibling instances stay independent.
"""

from __future__ import annotations

import hashlib as _hashlib
import re as _re
import uuid as _uuid
from dataclasses import dataclass, field
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
    """The RAW link id serialized on the instance's input ``name``, else None.
    Prefer :func:`live_external_link`: a serialized id whose link no longer
    exists is one the frontend drops on load, i.e. the input is unlinked."""
    entry = instance_input(instance, name)
    return None if entry is None else entry.get("link")


def link_exists(scope: dict, link_id: Any) -> bool:
    """Whether ``link_id`` is a live link in ``scope`` — the workflow (top-level
    links are ``[id, src, slot, dst, slot, type]`` arrays) or a subgraph
    definition (interior links are ``{"id": …}`` dicts)."""
    for link in scope.get("links") or []:
        if isinstance(link, dict):
            if link.get("id") == link_id:
                return True
        elif isinstance(link, (list, tuple)) and link and link[0] == link_id:
            return True
    return False


def live_external_link(scope: dict, instance: dict, name: str) -> Any:
    """The link id feeding the instance's input ``name`` from outside, if that
    link exists in ``scope`` (the workflow for a top-level instance, the
    containing definition for a nested one); else None. A dangling id is
    treated as unlinked everywhere — ``set-widget``, ``slots`` and the
    converter must agree on what runs."""
    link_id = external_link(instance, name)
    if link_id is None or not link_exists(scope, link_id):
        return None
    return link_id


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


def effective_value_for(
    workflow: dict, instance: dict, sg: dict, pi: PromotedInput, graph, defs: dict[str, dict]
) -> Any:
    """:func:`effective_value` for a caller that already holds the resolved
    ``PromotedInput`` and the definition index — the host value when
    materialized, else the interior source value (:data:`UNSET` when even that
    cannot be read). No name lookup, no re-walk of ``promoted_inputs``: a
    renderer iterating every promoted input of every instance calls this once
    per widget without re-deriving what its own loop produced."""
    value = host_value(instance, pi)
    if value is not UNSET:
        return value
    return source_value(workflow, sg, pi, graph, defs)


def _effective(workflow: dict, instance: dict, sg: dict, name: str, graph, defs: dict[str, dict]) -> Any:
    pi = find_promoted(sg, defs, name)
    if pi is None:
        raise ValueError(f"{name!r} is not a promoted input of subgraph {sg.get('id')}")
    if not pi.is_widget:
        raise ValueError(f"promoted input {name!r} on subgraph node {instance.get('id')} is a link input, not a widget")
    return effective_value_for(workflow, instance, sg, pi, graph, defs)


def effective_value(workflow: dict, instance: dict, name: str, graph) -> Any:
    """The value the frontend runs for promoted input ``name`` on ``instance``
    when nothing outside feeds it: the host value if materialized, else the
    interior source value (:data:`UNSET` if even that cannot be read).

    A legacy ``proxyWidgets`` promotion the definition does not back yet
    reads as the frontend will show it after its own migration: the legacy
    positional host value, else the interior source (the address is the
    input name the repair mints — see :func:`plan_proxy_migration`).

    Raises ``ValueError`` for a name the definition does not declare, or one
    that is a socket-only input.
    """
    defs = defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    if sg is None:
        raise ValueError(f"node {instance.get('id')} is not a subgraph instance")
    pi = find_promoted(sg, defs, name)
    if pi is None or not pi.is_widget:
        pending = next(
            (e for e in plan_proxy_migration(workflow, instance, graph, defs) if e.repairable and e.name == name), None
        )
        if pending is not None:
            return entry_effective_value(workflow, sg, pending, graph, defs)
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
    #: A ``host`` target whose promotion is still a legacy ``proxyWidgets``
    #: entry: the write must first run :func:`flush_proxy_migration` on the
    #: instance. This is the entry it repairs; ``widget`` is the input name
    #: the repair mints.
    repair: Any = None


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
    * a name only the legacy ``proxyWidgets`` route — a promotion the
      frontend's load-time migration turns into a linked input. The host,
      after the same repair (``repair`` names the entry; see
      :func:`flush_proxy_migration`). An interior address the repair would
      own (the entry's source widget, a primitive's fan-out target) redirects
      there too. An entry the migration cannot repair (quarantined — e.g.
      ``control_after_generate``, which has no backing input slot) keeps the
      interior widget as the live one, so the write lands there as before.
    """
    defs = defs_by_id(workflow)
    if len(segments) > 1:
        target = _navigate(workflow, segments, defs)
        parent = _navigate(workflow, segments[:-1], defs)
        parent_def = defs.get(str(parent.get("type", "")))
        target_def = defs.get(str(target.get("type", "")))
        given = f"{'/'.join(segments)}.{widget}"
        if parent_def is not None:
            # Checked BEFORE the interior-feeder refusal below on purpose: a
            # widget a legacy entry owns is repaired by the frontend on load,
            # and that repair replaces any interior link feeding its slot
            # (see ``plan_proxy_migration``), so the host value — not the
            # feeder — is what will run. The same slot with no legacy entry
            # keeps its feeder and is refused.
            pending = planned_repair_for_interior(workflow, parent, graph, str(target.get("id")), widget, defs)
            if pending is not None:
                return WriteTarget(
                    "host",
                    node=parent,
                    widget=str(pending.name),
                    segments=segments[:-1],
                    redirected_from=redirected_from or given,
                    repair=pending,
                )
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
    if pi is None or not pi.is_widget:
        pending = planned_repair(workflow, node, graph, widget, defs)
        if pending is not None:
            return WriteTarget(
                "host",
                node=node,
                widget=str(pending.name),
                segments=[node_str],
                redirected_from=redirected_from or (given if pending.name != widget else None),
                repair=pending,
            )
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
        link_id = live_external_link(workflow, node, widget)
        if link_id is not None:
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


# --------------------------------------------------------------------------- #
# Legacy ``properties.proxyWidgets`` → linked inputs: the forward migration
# --------------------------------------------------------------------------- #
#
# A port of the frontend's ``flushProxyWidgetMigration``
# (``src/core/graph/subgraph/migration/proxyWidgetMigration.ts``; ADR 0009
# "Forward migration", "Primitive-node repair", "Proxy widget error
# quarantine", "Serialization"). The frontend runs it on every subgraph host
# when a workflow loads; the CLI runs it on the one instance whose legacy
# promotion a write is about to edit, so that write can land on the HOST
# value like every other promoted widget instead of on the interior node.
#
# Per entry ``[sourceNodeId, sourceWidgetName(, disambiguator)]``:
#
# * already represented by a linked subgraph input → consumed (``alreadyLinked``);
# * a ``$$``-prefixed name or a preview pseudo-widget → a display-only preview
#   exposure: no input, no value. The frontend moves it into its
#   preview-exposure store; the CLI leaves the entry in ``proxyWidgets`` so the
#   frontend's own load-time migration (which also auto-exposes preview nodes
#   when ``previewExposures`` is unset) produces exactly the state it would
#   have produced from the untouched file;
# * a ``PrimitiveNode`` source → ONE subgraph input for its whole fan-out, every
#   former target reconnected to it, the primitive left in place but inert
#   (``primitiveBypass``; all-or-quarantine);
# * any other value widget → a subgraph input named after the widget
#   (``nextUniqueName`` on collision), a boundary link from the subgraph input
#   node (``-10``) into the widget's backing input slot, and the host value
#   (``createSubgraphInput``);
# * anything unrepairable → ``properties.proxyWidgetErrorQuarantine`` with
#   ``{originalEntry, reason, hostValue?, attemptedAtVersion: 1}``.
#
# Host values come from the instance's pre-migration ``widgets_values``
# POSITIONALLY BY ``proxyWidgets`` ORDER (``pickHostValue``) — a hole (no
# such index) means "keep the interior default". Consumed and quarantined
# entries are removed from ``proxyWidgets``; canonical saves never re-emit
# them.
#
# Determinism: every id the repair mints (the subgraph input's uuid, each
# boundary link id) is DERIVED from ``(instance path, source node, widget)``
# — never random, never a counter — so two replicas replaying the same op, or
# two replicas independently repairing the same entry, produce byte-identical
# documents (the convergence requirement of :mod:`comfy_cli.workflow_ops`).

QUARANTINE_PROPERTY = "proxyWidgetErrorQuarantine"
PROXY_BYPASS_MARKER_PROPERTY = "proxyBypassedToSubgraphInput"
QUARANTINE_VERSION = 1
CANVAS_IMAGE_PREVIEW_WIDGET = "$$canvas-image-preview"

#: ``isPreviewPseudoWidget``: a ``$$`` name, or a ``serialize:false`` widget of
#: type ``preview``/``video``/``audioUI``. The catalog carries no frontend-only
#: widgets, so the CLI projects the type rule onto the one such widget the
#: engine already models by name (``FRONTEND_MARKER_SLOTS``): ``audioUI``.
_PREVIEW_PSEUDO_WIDGET_NAMES = frozenset({"audioUI"})

#: The legacy ``PrimitiveNode`` has no server schema: its value widget is named
#: ``value`` and, when the target widget carries one, a linked
#: ``control_after_generate`` follows it in ``widgets_values``.
_PRIMITIVE_WIDGET_ORDER = ("value", "control_after_generate")

#: ``LEGACY_PROXY_WIDGET_PREFIX_PATTERN``: an older save could prefix a nested
#: widget name with its node id (``"12:seed"``).
_LEGACY_PREFIX_RE = _re.compile(r"^\s*(\d+)\s*:\s*(.+)$")

#: Minted link ids share the op model's leaderless range (see
#: ``workflow_ops.mint_id``): always above any frontend counter id, always
#: inside JS ``Number.MAX_SAFE_INTEGER``.
_LINK_ID_FLOOR = 1 << 40
_LINK_ID_BITS = 52

PLAN_ALREADY_LINKED = "alreadyLinked"
PLAN_CREATE = "createSubgraphInput"
PLAN_PRIMITIVE = "primitiveBypass"
PLAN_PREVIEW = "previewExposure"
PLAN_QUARANTINE = "quarantine"


def next_unique_name(name: str, existing: Any = ()) -> str:
    """``nextUniqueName``: ``seed`` → ``seed_1`` → ``seed_2`` … while taken."""
    existing = list(existing)
    base, i = name, 1
    while name in existing:
        name = f"{base}_{i}"
        i += 1
    return name


@dataclass
class LegacyEntry:
    """One ``proxyWidgets`` tuple with the migration's decision for it.

    ``name`` is the subgraph input the entry resolves to: the linked input for
    ``alreadyLinked``, the (collision-free) input the flush will mint for
    ``createSubgraphInput``/``primitiveBypass``. ``host_value`` is the legacy
    positional host value (:data:`UNSET` for a hole). ``targets`` are a
    primitive's ``(target node id, slot)`` fan-out in reconnect order.
    """

    original: list
    index: int
    source_node: str
    widget: str
    disambiguator: str | None
    plan: str
    name: str | None = None
    reason: str | None = None
    host_value: Any = UNSET
    type: str | None = None
    targets: list[tuple[str, int]] = field(default_factory=list)
    source_input: str | None = None  # a nested instance source: the input it projects
    slot_index: int | None = None  # createSubgraphInput: the backing input slot (None → synthesize)
    label: str | None = None

    @property
    def repairable(self) -> bool:
        return self.plan in (PLAN_CREATE, PLAN_PRIMITIVE)

    @property
    def key(self) -> str:
        return f"{self.source_node}.{self.widget}"

    def as_promoted_input(self, sg: dict) -> PromotedInput:
        """A schema-only view of the input the repair mints, so the same
        validation/source-port lookup a linked promotion gets applies."""
        if self.plan == PLAN_PRIMITIVE:
            tid, slot = self.targets[0]
            target = next((n for n in sg.get("nodes") or [] if str(n.get("id")) == tid), None)
            entry = (target.get("inputs") or [])[slot] if target is not None else None
            marker = entry.get("widget") if isinstance(entry, dict) else None
            widget = marker.get("name") if isinstance(marker, dict) and marker.get("name") else entry.get("name")
            return PromotedInput(
                name=str(self.name),
                type=str(self.type),
                index=-1,
                value_index=-1,
                source_node=tid,
                source_input=str(entry.get("name")) if isinstance(entry, dict) else None,
                source_widget=str(widget),
            )
        nested = self.source_input is not None
        return PromotedInput(
            name=str(self.name),
            type=str(self.type),
            index=-1,
            value_index=-1,
            source_node=self.source_node,
            source_input=self.source_input if nested else self.widget,
            source_widget=None if nested else self.widget,
            nested=nested,
        )


@dataclass
class FlushReport:
    consumed: list[list] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    quarantined: list[dict] = field(default_factory=list)
    remaining: list[list] = field(default_factory=list)


def _proxy_tuples(instance: dict) -> list[list]:
    """``parseProxyWidgets``: the whole property must parse (an array of
    2/3-string tuples) or none of it migrates."""
    raw = (instance.get("properties") or {}).get("proxyWidgets")
    if not isinstance(raw, list):
        return []
    out: list[list] = []
    for entry in raw:
        if not isinstance(entry, list) or len(entry) not in (2, 3) or not all(isinstance(x, str) for x in entry):
            return []
        out.append(list(entry))
    return out


def _inner_node(sg: dict, node_id: Any) -> dict | None:
    return next((n for n in sg.get("nodes") or [] if isinstance(n, dict) and str(n.get("id")) == str(node_id)), None)


def _node_widget_names(node: dict, graph) -> list[str]:
    """The widgets the frontend would find on ``node`` (``node.widgets``)."""
    if str(node.get("type", "")) == LEGACY_PRIMITIVE_TYPE:
        return list(_PRIMITIVE_WIDGET_ORDER)
    if graph is None:
        return []
    from comfy_cli.cql.engine import _widgets_as_positional

    node_type = str(node.get("type", ""))
    values = _widgets_as_positional(node.get("widgets_values"), graph, node_type)
    return list(graph.widget_order_for_node(node_type, values))


def _node_widget_value(node: dict, widget: str, graph) -> Any:
    order = _node_widget_names(node, graph)
    if widget not in order:
        return UNSET
    values = node.get("widgets_values")
    if str(node.get("type", "")) != LEGACY_PRIMITIVE_TYPE and graph is not None:
        from comfy_cli.cql.engine import _widgets_as_positional

        values = _widgets_as_positional(values, graph, str(node.get("type", "")))
    if not isinstance(values, list):
        return UNSET
    idx = order.index(widget)
    return values[idx] if idx < len(values) else UNSET


def _is_preview_pseudo_widget(widget: str) -> bool:
    return widget.startswith("$$") or widget in _PREVIEW_PSEUDO_WIDGET_NAMES


def _promotion_source(sg: dict, inp: dict, defs: dict[str, dict]) -> tuple[str, str] | None:
    """``resolvePromotionSource``: the ``(interior node, widget)`` a subgraph
    input projects — the first of its links that lands on a nested instance's
    input or on a widget-backed input slot."""
    links = {x.get("id"): x for x in sg.get("links") or [] if isinstance(x, dict)}
    for link_id in inp.get("linkIds") or []:
        link = links.get(link_id) if isinstance(link_id, (int, str)) else None
        if link is None:
            continue
        target = _inner_node(sg, link.get("target_id"))
        if target is None:
            continue
        entries = target.get("inputs") or []
        slot = link.get("target_slot")
        entry = entries[slot] if isinstance(slot, int) and 0 <= slot < len(entries) else None
        if not isinstance(entry, dict):
            continue
        if str(target.get("type", "")) in defs:
            return str(target.get("id")), str(entry.get("name"))
        marker = entry.get("widget")
        if isinstance(marker, dict) and marker.get("name"):
            return str(target.get("id")), str(marker["name"])
    return None


def _find_host_input_for_promotion(sg: dict, defs: dict[str, dict], node_id: str, widget: str) -> str | None:
    for inp in sg.get("inputs") or []:
        if isinstance(inp, dict) and _promotion_source(sg, inp, defs) == (node_id, widget):
            return str(inp.get("name"))
    return None


def _subgraph_input_target(inner_def: dict, defs: dict[str, dict], input_name: str) -> tuple[str, str] | None:
    """``resolveSubgraphInputTarget``: what a nested instance's input projects."""
    inp = next((i for i in inner_def.get("inputs") or [] if isinstance(i, dict) and i.get("name") == input_name), None)
    return _promotion_source(inner_def, inp, defs) if inp is not None else None


def _can_resolve_legacy_proxy(sg: dict, defs: dict[str, dict], node_id: str, widget: str, graph) -> bool:
    """``resolveConcretePromotedWidget(...).status === 'resolved'``: walk
    through nested instances to a concrete interior widget."""
    current_sg, current_id, current_widget = sg, node_id, widget
    for _ in range(_MAX_NESTED_PROMOTION_DEPTH):
        node = _inner_node(current_sg, current_id)
        if node is None:
            return False
        inner_def = defs.get(str(node.get("type", "")))
        if inner_def is not None:
            target = _subgraph_input_target(inner_def, defs, current_widget)
            if target is None:
                return False
            current_sg, (current_id, current_widget) = inner_def, target
            continue
        return current_widget in _node_widget_names(node, graph)
    return False


def _normalize_entry(
    sg: dict, defs: dict[str, dict], node_id: str, widget: str, disambiguator: str | None, graph
) -> tuple[str, str, str | None]:
    """``normalizeLegacyProxyWidgetEntry``: strip ``<id>:`` prefixes off a
    widget name that does not resolve as written, surfacing the deepest
    prefix as the disambiguator."""
    if _can_resolve_legacy_proxy(sg, defs, node_id, widget, graph):
        return node_id, widget, disambiguator
    remaining, deepest = widget, None
    while True:
        m = _LEGACY_PREFIX_RE.match(remaining)
        if m is None:
            break
        deepest, remaining = m.group(1), m.group(2)
    return node_id, remaining, deepest or disambiguator


def _resolve_source_widget(
    sg: dict, source: dict, widget: str, disambiguator: str | None, defs: dict[str, dict], graph
) -> tuple[str, str | None] | None:
    """``resolveSourceWidget``: ``(widget name, projecting input name)``.

    On a nested instance the widget is one of its widget-backed promoted
    inputs (matched by what the input projects, or by the input's own name);
    on an ordinary node it is one of the node's widgets — plus the virtual
    canvas preview every image node can expose (``getPromotableWidgets``).
    """
    inner_def = defs.get(str(source.get("type", "")))
    if inner_def is not None:
        by_name = {p.name: p for p in promoted_inputs(inner_def, defs)}
        for inp in inner_def.get("inputs") or []:
            if not isinstance(inp, dict):
                continue
            target = _subgraph_input_target(inner_def, defs, str(inp.get("name")))
            if disambiguator is not None:
                hit = target is not None and target[1] == widget and target[0] == disambiguator
            else:
                hit = inp.get("name") == widget or (target is not None and target[1] == widget)
            if not hit:
                continue
            pi = by_name.get(str(inp.get("name")))
            return (widget, str(inp.get("name"))) if pi is not None and pi.is_widget else None
        return None
    if widget in _node_widget_names(source, graph) or widget == CANVAS_IMAGE_PREVIEW_WIDGET:
        return widget, None
    return None


def _slot_for_widget(source: dict, widget: str, projecting_input: str | None, defs: dict[str, dict], graph):
    """``getSlotFromWidget``: ``(index, entry, type)`` of the input slot backing
    ``widget``. The serialized ``inputs[]`` may omit an unlinked widget slot
    the runtime node still has (older saves); then ``index`` is ``None`` and
    the flush appends one with ``type``. ``None`` when no slot backs it — a
    frontend-only widget such as ``control_after_generate``."""
    entries = source.get("inputs") or []
    inner_def = defs.get(str(source.get("type", "")))
    if inner_def is not None:
        name = projecting_input or widget
        for idx, entry in enumerate(entries):
            if isinstance(entry, dict) and entry.get("name") == name:
                return idx, entry, str(entry.get("type") or "*")
        pi = find_promoted(inner_def, defs, name)
        return (None, None, pi.type) if pi is not None else None
    for idx, entry in enumerate(entries):
        marker = entry.get("widget") if isinstance(entry, dict) else None
        if isinstance(marker, dict) and marker.get("name") == widget:
            return idx, entry, str(entry.get("type") or "*")
    # An older save can serialize the widget's slot by ``name`` alone (no
    # marker): that IS the backing slot — the flush restores its marker —
    # never a reason to append a duplicate beside it.
    for idx, entry in enumerate(entries):
        if isinstance(entry, dict) and entry.get("widget") is None and entry.get("name") == widget:
            return idx, entry, str(entry.get("type") or "*")
    m = graph.node(str(source.get("type", ""))) if graph is not None else None
    port = next((p for p in m.inputs if p.name == widget), None) if m is not None else None
    if port is None or port.is_link:
        return None
    return None, None, str(port.type)


def _primitive_targets(sg: dict, primitive: dict) -> list[tuple[str, int]]:
    """Every existing link out of the primitive's output 0, in link order."""
    links = {x.get("id"): x for x in sg.get("links") or [] if isinstance(x, dict)}
    outputs = primitive.get("outputs") or []
    listed = outputs[0].get("links") if outputs and isinstance(outputs[0], dict) else None
    ordered = [links[i] for i in listed if i in links] if isinstance(listed, list) else []
    if not ordered:
        ordered = [
            x
            for x in links.values()
            if str(x.get("origin_id")) == str(primitive.get("id")) and x.get("origin_slot") == 0
        ]
    return [(str(x.get("target_id")), x.get("target_slot")) for x in ordered if isinstance(x.get("target_slot"), int)]


def _types_compatible(a: str, b: str) -> bool:
    return a == b or a == "*" or b == "*"


def plan_proxy_migration(workflow: dict, instance: dict, graph, defs: dict[str, dict] | None = None) -> list:
    """The migration's decision for every ``proxyWidgets`` entry of
    ``instance`` — a dry run of :func:`flush_proxy_migration`, in entry order,
    with the input names the flush will mint (creates first, then primitive
    cohorts, each ``nextUniqueName``-collision-free against what precedes it).
    Empty when the instance has no (valid) legacy entries."""
    defs = defs if defs is not None else defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    tuples = _proxy_tuples(instance)
    if sg is None or not tuples:
        return []
    host_values = instance.get("widgets_values")
    host_values = host_values if isinstance(host_values, list) else None

    normalized = [_normalize_entry(sg, defs, t[0], t[1], t[2] if len(t) > 2 else None, graph) for t in tuples]
    cohort = [(n[0], n[1]) for n in normalized]
    entries: list[LegacyEntry] = []
    for index, (original, (node_id, widget, disambiguator)) in enumerate(zip(tuples, normalized)):
        host_value = host_values[index] if host_values is not None and index < len(host_values) else UNSET
        e = LegacyEntry(original, index, node_id, widget, disambiguator, PLAN_QUARANTINE, host_value=host_value)
        entries.append(e)
        linked = _find_host_input_for_promotion(sg, defs, node_id, widget)
        if linked is not None:
            matches = [p for p in promoted_inputs(sg, defs) if p.name == linked]
            if len(matches) > 1:
                e.reason = "ambiguousSubgraphInput"
            elif not matches or not matches[0].is_widget:
                e.reason = "missingSubgraphInput"
            else:
                e.plan, e.name, e.type = PLAN_ALREADY_LINKED, linked, matches[0].type
            continue
        source = _inner_node(sg, node_id)
        if source is None:
            e.reason = "missingSourceNode"
            continue
        if str(source.get("type", "")) == LEGACY_PRIMITIVE_TYPE:
            bypassed = (source.get("properties") or {}).get(PROXY_BYPASS_MARKER_PROPERTY)
            if isinstance(bypassed, str):
                existing = next((p for p in promoted_inputs(sg, defs) if p.name == bypassed), None)
                if existing is not None:
                    if existing.is_widget:
                        e.plan, e.name, e.type = PLAN_ALREADY_LINKED, bypassed, existing.type
                    else:
                        e.reason = "missingSubgraphInput"
                    continue
            targets = _primitive_targets(sg, source)
            # ``cohortDuplicatesPrimitive``: two entries naming this primitive
            # (whatever their widget) still form a cohort to validate.
            if targets or sum(1 for n in cohort if n[0] == node_id) >= 2:
                e.plan, e.targets = PLAN_PRIMITIVE, targets
            else:
                e.reason = "unlinkedSourceWidget"
            continue
        resolved = _resolve_source_widget(sg, source, widget, disambiguator, defs, graph)
        if resolved is None:
            e.reason = "missingSourceWidget"
            continue
        widget_name, projecting = resolved
        if widget.startswith("$$") or _is_preview_pseudo_widget(widget_name):
            e.plan = PLAN_PREVIEW
            continue
        slot = _slot_for_widget(source, widget_name, projecting, defs, graph)
        if slot is None:
            # The widget has no backing input slot (``control_after_generate``).
            # ``missingSubgraphInput`` is the frontend's own reason code for
            # this case — ``repairCreateSubgraphInput``: "source widget has no
            # backing input slot; quarantining" → ``reason: 'missingSubgraphInput'``.
            e.reason = "missingSubgraphInput"
            continue
        idx, entry, slot_type = slot
        # A slot already fed by an interior link is repaired all the same: the
        # frontend calls ``SubgraphInput.connect(slot, node)`` unconditionally
        # and ``connect`` replaces the incumbent link (``replaceLinkTopology``
        # + ``_disconnectNodeInput``), so the boundary link takes the slot over
        # and the interior feeder is dropped — see ``_add_boundary_link``.
        e.plan, e.type, e.slot_index, e.source_input = PLAN_CREATE, slot_type, idx, projecting
        if isinstance(entry, dict) and entry.get("label") is not None:
            e.label = str(entry["label"])

    # Names, in flush order: creates as they come, then one input per primitive
    # cohort (validated all-or-quarantine, exactly like ``repairPrimitive``).
    existing = [str(i.get("name")) for i in sg.get("inputs") or [] if isinstance(i, dict)]
    for e in entries:
        if e.plan == PLAN_CREATE:
            e.name = next_unique_name(e.widget, existing)
            existing.append(e.name)
    cohorts: dict[str, list[LegacyEntry]] = {}
    for e in entries:
        if e.plan == PLAN_PRIMITIVE:
            cohorts.setdefault(e.source_node, []).append(e)
    for primitive_id, members in cohorts.items():
        primitive = _inner_node(sg, primitive_id)
        failure = _validate_primitive_cohort(sg, primitive, members)
        if failure is not None:
            for e in members:
                e.plan, e.reason, e.targets = PLAN_QUARANTINE, "primitiveBypassFailed", []
            continue
        outputs = primitive.get("outputs") or []
        output_type = str(outputs[0].get("type") or "*")
        title = primitive.get("title")
        base = title if isinstance(title, str) and title and title != LEGACY_PRIMITIVE_TYPE else members[0].widget
        name = next_unique_name(base, existing)
        existing.append(name)
        for e in members:
            e.name, e.type = name, output_type
    return entries


def _validate_primitive_cohort(sg: dict, primitive: dict | None, members: list) -> str | None:
    """``validateCohort`` + the pre-mutation checks of ``repairPrimitive``:
    one primitive, one widget name, every target present and type-compatible."""
    first = members[0]
    if any(m.widget != first.widget for m in members):
        return "cohort validation failed"
    if primitive is None or str(primitive.get("type", "")) != LEGACY_PRIMITIVE_TYPE:
        return "node is not a PrimitiveNode"
    targets = _primitive_targets(sg, primitive)
    if not targets:
        return "no targets to reconnect"
    outputs = primitive.get("outputs") or []
    if not outputs or not isinstance(outputs[0], dict):
        return "primitive has no output"
    output_type = str(outputs[0].get("type") or "*")
    for tid, slot in targets:
        target = _inner_node(sg, tid)
        entries = target.get("inputs") if target is not None else None
        entry = entries[slot] if isinstance(entries, list) and 0 <= slot < len(entries) else None
        if not isinstance(entry, dict):
            return "target slot missing"
        if not _types_compatible(str(entry.get("type") or "*"), output_type):
            return "target slot type incompatible"
    return None


def planned_repair(workflow: dict, instance: dict, graph, widget: str, defs: dict[str, dict] | None = None):
    """The repairable legacy entry a host address ``<instance>.<widget>`` names:
    by the input name the repair will mint, else by the legacy tuple's own
    widget name (an alias the caller reports as a redirect)."""
    plan = [e for e in plan_proxy_migration(workflow, instance, graph, defs) if e.repairable]
    by_name = next((e for e in plan if e.name == widget), None)
    if by_name is not None:
        return by_name
    aliased: list[LegacyEntry] = []
    for e in plan:
        if e.widget == widget and all(a.name != e.name for a in aliased):
            aliased.append(e)
    if len(aliased) > 1:
        # Two entries share the legacy widget name (two primitives both named
        # ``value``): never pick one silently — name the minted inputs.
        node_id = instance.get("id")
        raise ValueError(
            f"{node_id}.{widget} is ambiguous: {len(aliased)} legacy promotions on subgraph node {node_id} share "
            f"the widget name {widget!r}; address one by its input name — "
            f"{', '.join(f'{node_id}.{a.name}' for a in aliased)}"
        )
    return aliased[0] if aliased else None


def planned_repair_for_interior(
    workflow: dict, instance: dict, graph, inner_id: str, widget: str, defs: dict[str, dict] | None = None
):
    """The repairable legacy entry whose value the interior address
    ``<instance>/<inner>.<widget>`` would edit: the entry's own source widget,
    or one of a primitive's fan-out targets."""
    defs = defs if defs is not None else defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    if sg is None:
        return None
    for e in plan_proxy_migration(workflow, instance, graph, defs):
        if not e.repairable:
            continue
        if e.source_node == inner_id and e.widget == widget:
            return e
        for tid, slot in e.targets:
            target = _inner_node(sg, tid)
            entries = target.get("inputs") if target is not None else None
            entry = entries[slot] if isinstance(entries, list) and 0 <= slot < len(entries) else None
            marker = entry.get("widget") if isinstance(entry, dict) else None
            if tid == inner_id and isinstance(marker, dict) and marker.get("name") == widget:
                return e
    return None


def planned_hidden_sources(workflow: dict, instance: dict, graph, defs: dict[str, dict] | None = None) -> set:
    """``(interior node, widget)`` pairs a pending repair will own — the
    interior addresses whose edits the host surface overrides."""
    defs = defs if defs is not None else defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    hidden: set[tuple[str, str]] = set()
    if sg is None:
        return hidden
    for e in plan_proxy_migration(workflow, instance, graph, defs):
        if not e.repairable:
            continue
        hidden.add((e.source_node, e.widget))
        for tid, slot in e.targets:
            target = _inner_node(sg, tid)
            entries = target.get("inputs") if target is not None else None
            entry = entries[slot] if isinstance(entries, list) and 0 <= slot < len(entries) else None
            marker = entry.get("widget") if isinstance(entry, dict) else None
            if isinstance(marker, dict) and marker.get("name"):
                hidden.add((tid, str(marker["name"])))
    return hidden


def entry_effective_value(workflow: dict, sg: dict, entry: LegacyEntry, graph, defs: dict[str, dict]) -> Any:
    """What the frontend shows for the entry after its own migration: the
    legacy positional host value when present, else the source widget's
    current value (a primitive's own value for a bypass)."""
    if entry.host_value is not UNSET:
        return entry.host_value
    if entry.plan == PLAN_PRIMITIVE:
        primitive = _inner_node(sg, entry.source_node)
        return _node_widget_value(primitive, entry.widget, graph) if primitive is not None else UNSET
    return source_value(workflow, sg, entry.as_promoted_input(sg), graph, defs)


def repair_ids(instance_path: list, source_node: str, widget: str, n_links: int) -> dict:
    """The ids a repair of ``(source_node, widget)`` on the instance at
    ``instance_path`` mints: the subgraph input's uuid and one link id per
    boundary link. Derived with SHA-256 from the identifiers alone (like
    ``engine._deterministic_fork_id``), so they are stable across processes
    and replicas."""
    seed = "\x00".join(["proxy-repair", *[str(s) for s in instance_path], str(source_node), str(widget)])
    input_id = str(_uuid.UUID(bytes=_hashlib.sha256(seed.encode()).digest()[:16], version=4))
    links = []
    for k in range(n_links):
        digest = _hashlib.sha256(f"{seed}\x00link{k}".encode()).digest()
        links.append(_LINK_ID_FLOOR | (int.from_bytes(digest[:8], "big") & ((1 << _LINK_ID_BITS) - 1)))
    return {"input": input_id, "links": links}


def plan_repair_ids(instance_path: list, plan: list) -> dict[str, dict]:
    """Every repairable entry's ids, keyed ``<source node>.<widget>``."""
    out: dict[str, dict] = {}
    for e in plan:
        if e.repairable and e.key not in out:
            out[e.key] = repair_ids(instance_path, e.source_node, e.widget, max(1, len(e.targets)))
    return out


def _remove_link(sg: dict, link_id: Any) -> None:
    link = next((x for x in sg.get("links") or [] if isinstance(x, dict) and x.get("id") == link_id), None)
    if link is None:
        return
    sg["links"] = [x for x in sg.get("links") or [] if x is not link]
    origin = link.get("origin_id")
    if origin == SUBGRAPH_INPUT_NODE_ID:
        for inp in sg.get("inputs") or []:
            if isinstance(inp, dict) and isinstance(inp.get("linkIds"), list) and link_id in inp["linkIds"]:
                inp["linkIds"] = [x for x in inp["linkIds"] if x != link_id]
    else:
        origin_node = _inner_node(sg, origin)
        outputs = origin_node.get("outputs") if origin_node is not None else None
        slot = link.get("origin_slot")
        if isinstance(outputs, list) and isinstance(slot, int) and 0 <= slot < len(outputs):
            links = outputs[slot].get("links")
            if isinstance(links, list):
                outputs[slot]["links"] = [x for x in links if x != link_id]
    target = _inner_node(sg, link.get("target_id"))
    entries = target.get("inputs") if target is not None else None
    slot = link.get("target_slot")
    if isinstance(entries, list) and isinstance(slot, int) and 0 <= slot < len(entries):
        if isinstance(entries[slot], dict) and entries[slot].get("link") == link_id:
            entries[slot]["link"] = None


def _add_boundary_link(sg: dict, new_input: dict, link_id: Any, target: dict, slot: int, slot_type: str) -> None:
    """``SubgraphInput.connect``: replace whatever fed the slot with a link
    from the subgraph input node, as the frontend serializes one."""
    entries = target["inputs"]
    existing = entries[slot].get("link")
    if existing is not None:
        _remove_link(sg, existing)
    origin_slot = next(i for i, inp in enumerate(sg["inputs"]) if inp is new_input)
    sg.setdefault("links", []).append(
        {
            "id": link_id,
            "origin_id": SUBGRAPH_INPUT_NODE_ID,
            "origin_slot": origin_slot,
            "target_id": target.get("id"),
            "target_slot": slot,
            "type": slot_type,
        }
    )
    entries[slot]["link"] = link_id
    new_input.setdefault("linkIds", []).append(link_id)
    state = sg.get("state")
    if isinstance(state, dict):
        last = state.get("lastLinkId")
        state["lastLinkId"] = max(last if isinstance(last, int) else 0, link_id)


def _add_subgraph_input(sg: dict, input_id: str, name: str, slot_type: str, label: str | None) -> dict:
    """``Subgraph.addInput`` as the frontend serializes a ``SubgraphInput``
    (``id``, ``name``, ``type``, ``linkIds``; ``label`` when the source slot
    carried one). ``pos`` is layout the input node assigns on load."""
    new_input: dict = {"id": input_id, "name": name, "type": slot_type, "linkIds": []}
    if label is not None:
        new_input["label"] = label
    sg.setdefault("inputs", []).append(new_input)
    return new_input


def _add_host_input(instance: dict, name: str, slot_type: str, label: str | None) -> None:
    """The host's own ``inputs[]`` entry for a new promoted widget input —
    the shape a post-migration save carries (``audio_minimax_music_3.json``
    node 37). Left alone when the instance already serializes one."""
    if instance_input(instance, name) is not None:
        return
    entry: dict = {"name": name, "type": slot_type, "widget": {"name": name}, "link": None}
    if label is not None:
        entry = {"label": label, **entry}
    instance.setdefault("inputs", []).append(entry)


def _quarantine_entry(entry: LegacyEntry, reason: str) -> dict:
    out = {"originalEntry": list(entry.original), "reason": reason, "attemptedAtVersion": QUARANTINE_VERSION}
    if entry.host_value is not UNSET:
        out["hostValue"] = entry.host_value
    return out


def flush_proxy_migration(
    workflow: dict, instance: dict, graph, ids: dict[str, dict] | None = None, instance_path: list | None = None
) -> FlushReport:
    """Run the forward migration on ``instance`` IN PLACE (its definition, its
    interior nodes, its own ``inputs``/``widgets_values``/``properties``).

    ``ids`` are the minted ids per repairable entry (:func:`plan_repair_ids`);
    derived from ``instance_path`` when not given. Idempotent: an instance
    without legacy value entries is left untouched. The definition is
    mutated — callers isolate a shared definition first (the op layer does,
    via ``engine._isolate_shared_subgraph``).
    """
    defs = defs_by_id(workflow)
    sg = defs.get(str(instance.get("type", "")))
    report = FlushReport()
    if sg is None or not _proxy_tuples(instance):
        return report
    if graph is None:
        # Without the catalog no interior widget can be resolved, and the
        # honest answer to that is not "quarantine everything".
        raise ValueError(
            f"repairing the legacy proxyWidgets of subgraph node {instance.get('id')} needs the node catalog"
        )
    plan = plan_proxy_migration(workflow, instance, graph, defs)
    if not plan:
        return report
    instance_path = instance_path if instance_path is not None else [str(instance.get("id"))]
    ids = dict(ids or {})
    for e in plan:
        if e.repairable and e.key not in ids:
            ids[e.key] = repair_ids(instance_path, e.source_node, e.widget, max(1, len(e.targets)))

    # Values the host already carries, read the way ``configure`` applies them
    # (positionally over the linked inputs) — before the structure changes.
    pre = {p.name: host_value(instance, p) for p in promoted_inputs(sg, defs) if p.is_widget}
    applied: dict[str, Any] = {}
    quarantine: list[dict] = []
    remaining: list[list] = []
    cohorts: dict[str, list[LegacyEntry]] = {}
    for e in plan:
        if e.plan == PLAN_PREVIEW:
            remaining.append(list(e.original))
            continue
        if e.plan == PLAN_QUARANTINE:
            if e.reason != "primitiveBypassFailed":
                quarantine.append(_quarantine_entry(e, str(e.reason)))
            else:
                cohorts.setdefault(e.source_node, []).append(e)
            continue
        report.consumed.append(list(e.original))
        if e.plan == PLAN_ALREADY_LINKED:
            if e.host_value is not UNSET:
                applied[str(e.name)] = e.host_value
            continue
        if e.plan == PLAN_PRIMITIVE:
            cohorts.setdefault(e.source_node, []).append(e)
            continue
        # createSubgraphInput
        source = _inner_node(sg, e.source_node)
        slot_index = e.slot_index
        if slot_index is not None and not isinstance(source["inputs"][slot_index].get("widget"), dict):
            # A slot serialized by name alone: give it back the marker a
            # converted widget carries, in the frontend's key order.
            found = source["inputs"][slot_index]
            marked = {k: v for k, v in found.items() if k != "link"}
            marked["widget"] = {"name": e.source_input or e.widget}
            marked["link"] = found.get("link")
            source["inputs"][slot_index] = marked
        if slot_index is None:
            if e.source_input is not None:
                # A nested instance backs the widget with its OWN host input
                # for the projecting subgraph input (``getSlotFromWidget``
                # over ``promotedInputWidget``): the slot carries that input's
                # name, or the outer definition could never resolve the
                # promotion through the inner one.
                entry = {
                    "name": e.source_input,
                    "type": e.type,
                    "widget": {"name": e.source_input},
                    "link": None,
                }
            else:
                entry = {
                    "localized_name": e.widget,
                    "name": e.widget,
                    "type": e.type,
                    "widget": {"name": e.widget},
                    "link": None,
                }
            source.setdefault("inputs", []).append(entry)
            slot_index = len(source["inputs"]) - 1
        new_input = _add_subgraph_input(sg, ids[e.key]["input"], str(e.name), str(e.type), e.label)
        _add_boundary_link(sg, new_input, ids[e.key]["links"][0], source, slot_index, str(e.type))
        _add_host_input(instance, str(e.name), str(e.type), e.label)
        report.created.append(str(e.name))
        if e.host_value is not UNSET:
            applied[str(e.name)] = e.host_value

    for primitive_id, members in cohorts.items():
        if members[0].plan == PLAN_QUARANTINE:
            quarantine.extend(_quarantine_entry(m, "primitiveBypassFailed") for m in members)
            continue
        first = members[0]
        primitive = _inner_node(sg, primitive_id)
        # ``repairPrimitive`` re-validates against the graph as it is NOW: a
        # ``createSubgraphInput`` above may have taken over one of the
        # primitive's targets, so the fan-out is collected again.
        if _validate_primitive_cohort(sg, primitive, members) is not None:
            quarantine.extend(_quarantine_entry(m, "primitiveBypassFailed") for m in members)
            continue
        targets = _primitive_targets(sg, primitive)
        for link in [
            x
            for x in sg.get("links") or []
            if isinstance(x, dict) and str(x.get("origin_id")) == primitive_id and x.get("origin_slot") == 0
        ]:
            _remove_link(sg, link.get("id"))
        link_ids = ids[first.key]["links"]
        if len(link_ids) < len(targets):
            link_ids = repair_ids(instance_path, first.source_node, first.widget, len(targets))["links"]
        new_input = _add_subgraph_input(sg, ids[first.key]["input"], str(first.name), str(first.type), None)
        for k, (tid, slot) in enumerate(targets):
            target = _inner_node(sg, tid)
            _add_boundary_link(sg, new_input, link_ids[k], target, slot, str(target["inputs"][slot].get("type") or "*"))
        _add_host_input(instance, str(first.name), str(first.type), None)
        primitive.setdefault("properties", {})[PROXY_BYPASS_MARKER_PROPERTY] = str(first.name)
        report.created.append(str(first.name))
        unique: list[LegacyEntry] = []
        for m in members:
            if not any(
                (u.source_node, u.widget, u.disambiguator) == (m.source_node, m.widget, m.disambiguator) for u in unique
            ):
                unique.append(m)
        valued = next((u for u in unique if u.host_value is not UNSET), None)
        if valued is not None:
            applied[str(first.name)] = valued.host_value
        else:
            seed = _node_widget_value(primitive, first.widget, graph)
            if seed is not UNSET:
                applied[str(first.name)] = seed

    if not report.consumed and not quarantine:
        return report  # previews only: nothing to consume, nothing to write

    # ``appendQuarantine``: a ``-1`` entry's host value lands on the input of
    # that name; entries are deduplicated by their original tuple.
    props = instance.setdefault("properties", {})
    existing_q = [q for q in props.get(QUARANTINE_PROPERTY) or [] if isinstance(q, dict)]
    for q in quarantine:
        if q["originalEntry"][0] == "-1" and "hostValue" in q:
            applied[str(q["originalEntry"][1])] = q["hostValue"]
        if not any(x.get("originalEntry") == q["originalEntry"] for x in existing_q):
            existing_q.append(q)
            report.quarantined.append(q)
    if existing_q:
        props[QUARANTINE_PROPERTY] = existing_q
    if remaining:
        props["proxyWidgets"] = remaining
    else:
        props.pop("proxyWidgets", None)
    report.remaining = remaining

    # The host array, positional over the (now repaired) widget-backed inputs:
    # migrated values first, then what the host already carried, then the
    # interior default — the serialization the frontend writes after its flush.
    values: list[Any] = []
    for p in promoted_inputs(sg, defs):
        if not p.is_widget:
            continue
        value = applied.get(p.name, pre.get(p.name, UNSET))
        if value is UNSET:
            value = source_value(workflow, sg, p, graph, defs)
        values.append(None if value is UNSET else value)
    instance["widgets_values"] = values
    return report


def boundary_widget_targets(sg: dict, pi: PromotedInput, defs: dict[str, dict]) -> list[tuple[list[str], str]]:
    """Every concrete ``(interior node path, widget)`` a promoted input feeds —
    all of its boundary links, through nested instances. A repaired primitive
    fan-out has several; :func:`deepest_source` reports only the first."""
    inputs = sg.get("inputs") or []
    inp = inputs[pi.index] if 0 <= pi.index < len(inputs) and isinstance(inputs[pi.index], dict) else None
    if inp is None:
        single = deepest_source(sg, pi, defs)
        return [single] if single is not None else []
    return _boundary_targets(sg, inp, defs, 0)


def _boundary_targets(sg: dict, inp: dict, defs: dict[str, dict], depth: int) -> list[tuple[list[str], str]]:
    if depth > _MAX_NESTED_PROMOTION_DEPTH:
        return []
    links = {x.get("id"): x for x in sg.get("links") or [] if isinstance(x, dict)}
    out: list[tuple[list[str], str]] = []
    for link_id in inp.get("linkIds") or []:
        link = links.get(link_id) if isinstance(link_id, (int, str)) else None
        target = _inner_node(sg, link.get("target_id")) if link is not None else None
        if target is None:
            continue
        entries = target.get("inputs") or []
        slot = link.get("target_slot")
        entry = entries[slot] if isinstance(slot, int) and 0 <= slot < len(entries) else None
        if not isinstance(entry, dict):
            continue
        tid = str(target.get("id"))
        inner_def = defs.get(str(target.get("type", "")))
        if inner_def is not None:
            inner_inp = next(
                (
                    i
                    for i in inner_def.get("inputs") or []
                    if isinstance(i, dict) and i.get("name") == entry.get("name")
                ),
                None,
            )
            if inner_inp is not None:
                out.extend(([tid, *path], w) for path, w in _boundary_targets(inner_def, inner_inp, defs, depth + 1))
            continue
        marker = entry.get("widget")
        if marker:
            widget = marker.get("name") if isinstance(marker, dict) else None
            out.append(([tid], str(widget or entry.get("name"))))
    return out
