"""Convert ComfyUI UI-format workflows to API ("prompt") format.

The UI format is what the ComfyUI frontend saves by default — a litegraph dump
with `nodes` and `links` arrays. The API format is the flat
``{node_id: {class_type, inputs, _meta}}`` shape that the server's ``/prompt``
endpoint accepts.

The conversion needs schema information about each node type (which inputs are
widgets vs connections, what their order is, defaults, combo options, etc.).
That information is available from the running server's ``/object_info``
endpoint — the same data the frontend uses to render the graph editor.

This module is a Python port of Seth A. Robinson's
``comfyui-workflow-to-api-converter-endpoint`` (Unlicense), restructured to
take a fetched ``object_info`` dict instead of importing ComfyUI's in-process
``nodes`` module.
"""

from __future__ import annotations

import copy
import logging
import random
import re
from typing import Any

logger = logging.getLogger(__name__)

# C-style comments stripped from dynamic-prompt strings before group parsing.
_DYNAMIC_PROMPT_COMMENT_RE = re.compile(r"/\*[\s\S]*?\*/|//.*")
_DYNAMIC_PROMPT_UNESCAPE_RE = re.compile(r"\\([{}|])")


# Mode values from litegraph: see frontend's LGraphEventMode enum.
_MODE_MUTED = 2  # excluded from execution; outputs not produced
_MODE_BYPASS = 4  # node skipped; inputs passed through to outputs

# Node types that exist only in the UI graph and never appear in API output.
# Aligns with cloud-mcp-server's VIRTUAL_NODE_TYPES and the frontend's
# isVirtualNode set — every type the frontend's graphToPrompt() skips.
_UI_ONLY_NODE_TYPES = frozenset({"Note", "MarkdownNote", "PrimitiveNode", "GetNode", "SetNode", "Reroute"})

# Sentinel IDs litegraph uses inside a subgraph definition for the synthetic
# input and output proxy nodes (the boxes the user wires through).
_SUBGRAPH_INPUT_NODE_ID = -10
_SUBGRAPH_OUTPUT_NODE_ID = -20

# Cap on recursive subgraph / passthrough resolution to defend against cycles
# in malformed inputs.
_MAX_RESOLUTION_DEPTH = 100
_MAX_SUBGRAPH_ITERATIONS = 10

# Strings that ComfyUI appends after seed-like INT widgets to control how the
# value changes between runs. They're not real inputs and must be stripped from
# the widget-value list before mapping to input names.
_CONTROL_AFTER_GENERATE_VALUES = frozenset({"fixed", "increment", "decrement", "randomize"})

# A dynamic combo may nest another dynamic combo among its sub-inputs. Real
# schemas nest at most a couple of levels; a deeper chain (e.g. a pathological
# third-party ``/object_info`` entry) would otherwise recurse the widget walk
# without bound. Cap it so a malformed schema degrades to a warning instead of a
# ``RecursionError`` that aborts the whole conversion.
_MAX_DYNAMIC_COMBO_DEPTH = 16


class WorkflowConversionError(Exception):
    """Raised when a workflow can't be converted to API format."""


def is_api_format(workflow: Any) -> bool:
    """Return True if ``workflow`` already looks like an API-format prompt."""
    if not isinstance(workflow, dict):
        return False
    if "nodes" in workflow and "links" in workflow:
        return False
    for key, value in workflow.items():
        if key in ("prompt", "extra_data", "client_id"):
            continue
        if isinstance(value, dict) and "class_type" in value:
            return True
    return False


def is_subgraph_uuid(node_type: Any) -> bool:
    """A subgraph instance's node ``type`` field is the UUID of a subgraph def."""
    if not isinstance(node_type, str) or len(node_type) != 36:
        return False
    parts = node_type.split("-")
    if len(parts) != 5:
        return False
    return tuple(len(p) for p in parts) == (8, 4, 4, 4, 12)


def convert_ui_to_api(workflow: dict, object_info: dict) -> dict:
    """Convert a UI-format workflow to API format.

    Args:
        workflow: UI workflow with ``nodes`` and ``links`` keys.
        object_info: ``/object_info`` response: ``{node_type: schema}``.

    Returns:
        API-format dict: ``{node_id_str: {class_type, inputs, _meta}}``.
    """
    if is_api_format(workflow):
        return workflow
    if not isinstance(workflow, dict):
        raise WorkflowConversionError("Workflow must be a JSON object")
    if not isinstance(workflow.get("nodes"), list) or not isinstance(workflow.get("links"), list):
        raise WorkflowConversionError("Workflow is missing 'nodes' or 'links' list")
    if not isinstance(object_info, dict):
        raise WorkflowConversionError("object_info must be a JSON object")

    workflow = copy.deepcopy(workflow)
    # Discard any non-dict entries up front so the rest of the pipeline doesn't
    # have to defend against malformed nodes inside the list.
    nodes = [n for n in workflow["nodes"] if isinstance(n, dict)]
    links = list(workflow["links"])

    subgraph_defs = _collect_subgraph_defs(workflow)
    nodes, links, subgraph_ctx = _expand_subgraphs(nodes, links, subgraph_defs)

    links = _rewrite_links_for_subgraphs(links, subgraph_ctx, nodes)
    link_map = _build_link_map(links)

    node_by_id = {str(n.get("id")): n for n in nodes}
    primitive_values = _collect_primitive_values(nodes)
    bypassed = _collect_bypassed(nodes)
    nodes_to_exclude = _collect_excluded(nodes)
    reroute_sources = _collect_reroute_sources(nodes, link_map)
    set_sources, get_vars = _collect_get_set_mappings(nodes, link_map)

    tracers = _Tracers(
        link_map=link_map,
        nodes=nodes,
        node_by_id=node_by_id,
        bypassed=bypassed,
        reroute_sources=reroute_sources,
        set_sources=set_sources,
        get_vars=get_vars,
        subgraph_ctx=subgraph_ctx,
    )

    if _has_group_nodes(workflow):
        logger.warning(
            "Workflow uses legacy 'group nodes' (extra.groupNodes); these aren't "
            "expanded by this converter. Recreate them as subgraphs in the frontend."
        )

    api_prompt: dict[str, dict] = {}
    for node in nodes:
        node_id_str = str(node.get("id"))
        node_type = node.get("type")
        if not node_type:
            continue
        node_mode = node.get("mode", 0)
        if node_mode in (_MODE_MUTED, _MODE_BYPASS):
            continue
        if node_type in _UI_ONLY_NODE_TYPES:
            continue
        if node_id_str in nodes_to_exclude:
            continue

        try:
            api_prompt[node_id_str] = _build_api_node(
                node=node,
                node_type=node_type,
                object_info=object_info,
                tracers=tracers,
                primitive_values=primitive_values,
                bypassed=bypassed,
                nodes_to_exclude=nodes_to_exclude,
            )
        except Exception:
            # An individual malformed node should not torpedo the whole prompt.
            # The executor will fail loudly on missing nodes if this matters.
            logger.exception("Failed to convert node id=%s type=%s; skipping", node_id_str, node_type)

    _strip_orphan_link_inputs(api_prompt)
    return api_prompt


def _has_group_nodes(workflow: dict) -> bool:
    """Legacy 'group nodes' (workflow> types) live under extra.groupNodes."""
    extra = workflow.get("extra")
    if isinstance(extra, dict) and isinstance(extra.get("groupNodes"), dict) and extra["groupNodes"]:
        return True
    for node in workflow.get("nodes") or []:
        if not isinstance(node, dict):
            continue
        t = node.get("type")
        if isinstance(t, str) and (t.startswith("workflow>") or t.startswith("workflow/")):
            return True
    return False


def _strip_orphan_link_inputs(api_prompt: dict[str, dict]) -> None:
    """Drop any link inputs that reference a node we didn't emit.

    Defensive mirror of the frontend's final cleanup pass. We already skip
    most orphans during emission, but a stray reference can survive if the
    upstream tracing terminated on a node that later got pruned.
    """
    for node in api_prompt.values():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for name in list(inputs):
            value = inputs[name]
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str) and value[0] not in api_prompt:
                del inputs[name]


# ---------------------------------------------------------------------------
# Subgraph handling
# ---------------------------------------------------------------------------


class _SubgraphCtx:
    """Bookkeeping built during subgraph expansion, used later to rewrite links."""

    def __init__(self) -> None:
        # subgraph_node_id_str -> {subgraph_input_idx: [(internal_node_id, internal_slot), ...]}
        self.input_targets: dict[str, dict[int, list[tuple[Any, int]]]] = {}
        # subgraph_node_id_str -> {(internal_node_id, internal_slot): output_slot_idx}
        self.output_sources: dict[str, dict[tuple[Any, int], int]] = {}
        # subgraph_node_id_str -> {outer_slot: subgraph_input_idx} (when names differ in order)
        self.outer_to_input_idx: dict[str, dict[int, int]] = {}


def _collect_subgraph_defs(workflow: dict) -> dict[str, dict]:
    definitions = workflow.get("definitions")
    if not isinstance(definitions, dict):
        return {}
    subgraphs = definitions.get("subgraphs")
    if not isinstance(subgraphs, list):
        return {}
    defs: dict[str, dict] = {}
    for sg in subgraphs:
        if not isinstance(sg, dict):
            continue
        sg_id = sg.get("id")
        # sg_id has to be a string both because we use it as a dict key and
        # because is_subgraph_uuid (used to match instances) only accepts str.
        if isinstance(sg_id, str) and sg_id:
            defs[sg_id] = sg
    return defs


def _expand_subgraphs(
    nodes: list[dict], links: list, subgraph_defs: dict[str, dict]
) -> tuple[list[dict], list, _SubgraphCtx]:
    """Recursively expand subgraph instances into their constituent nodes."""
    ctx = _SubgraphCtx()
    if not subgraph_defs:
        return nodes, links, ctx

    for _iteration in range(_MAX_SUBGRAPH_ITERATIONS):
        expanded: list[dict] = []
        found_any = False
        for node in nodes:
            node_type = node.get("type")
            if is_subgraph_uuid(node_type) and node_type in subgraph_defs:
                # Frontend semantics (executionUtil.ts): if the subgraph
                # instance node itself is muted (mode 2) or bypassed (mode 4),
                # do NOT pull its inner nodes into the prompt. The instance
                # stays in the node list where the normal mode-check excludes
                # it from emission; for bypass, downstream consumers route
                # through ``trace_bypassed`` on the instance's external
                # inputs, the same way a bypassed regular node is handled.
                if node.get("mode") in (_MODE_MUTED, _MODE_BYPASS):
                    expanded.append(node)
                    continue
                found_any = True
                sg_nodes, sg_links, input_map, output_map = _expand_one_subgraph(node, subgraph_defs[node_type], links)
                expanded.extend(sg_nodes)
                links.extend(sg_links)
                ctx.input_targets[str(node.get("id"))] = input_map
                ctx.output_sources[str(node.get("id"))] = output_map
                ctx.outer_to_input_idx[str(node.get("id"))] = _outer_slot_to_input_idx(node, subgraph_defs[node_type])
            else:
                expanded.append(node)
        nodes = expanded
        if not found_any:
            return nodes, links, ctx

    logger.warning("Subgraph expansion hit iteration cap — possible cyclic reference")
    return nodes, links, ctx


def _outer_slot_to_input_idx(outer_node: dict, sg_def: dict) -> dict[int, int]:
    """Map the outer node's input slots to subgraph-definition input indices."""
    sg_input_names: dict[Any, int] = {}
    for idx, inp in enumerate(sg_def.get("inputs") or []):
        if isinstance(inp, dict):
            sg_input_names[inp.get("name")] = idx
    mapping: dict[int, int] = {}
    for outer_idx, outer_input in enumerate(outer_node.get("inputs") or []):
        if not isinstance(outer_input, dict):
            continue
        name = outer_input.get("name")
        if name in sg_input_names:
            mapping[outer_idx] = sg_input_names[name]
    return mapping


def _expand_one_subgraph(
    outer_node: dict, sg_def: dict, existing_links: list
) -> tuple[list[dict], list, dict[int, list[tuple[Any, int]]], dict[tuple[Any, int], int]]:
    outer_id = outer_node.get("id")
    internal_nodes = [n for n in (sg_def.get("nodes") or []) if isinstance(n, dict)]
    internal_links = sg_def.get("links") or []

    # Subgraph internal link IDs may collide with the outer workflow's IDs.
    # Allocate fresh IDs starting above the current maximum.
    max_link_id = 0
    for link in existing_links:
        if isinstance(link, (list, tuple)) and link:
            lid = link[0]
            if isinstance(lid, int) and lid > max_link_id:
                max_link_id = lid
    next_id = max_link_id + 1

    link_id_remap: dict[int, int] = {}
    internal_link_map: dict[int, dict] = {}
    for link in internal_links:
        if not isinstance(link, dict):
            continue
        old_id = link.get("id")
        # Only int IDs are usable here: link_id_remap[old_id] / internal_link_map[old_id]
        # need a hashable key, and the wider pipeline later does ``link_id in
        # link_id_remap`` lookups keyed by int link IDs from the outer workflow.
        # Skip the entry entirely on a missing/unhashable/wrong-typed id so a
        # bad apple can't crash the whole subgraph expansion (which runs
        # before the per-node try/except wrapper).
        if not isinstance(old_id, int):
            continue
        link_id_remap[old_id] = next_id
        next_id += 1
        internal_link_map[old_id] = link

    input_targets: dict[int, list[tuple[Any, int]]] = {}
    for idx, in_def in enumerate(sg_def.get("inputs") or []):
        if not isinstance(in_def, dict):
            continue
        targets = []
        for lid in in_def.get("linkIds") or []:
            if not isinstance(lid, int):
                continue
            link = internal_link_map.get(lid)
            if isinstance(link, dict):
                targets.append((link.get("target_id"), link.get("target_slot")))
        if targets:
            input_targets[idx] = targets

    output_sources: dict[tuple[Any, int], int] = {}
    for idx, out_def in enumerate(sg_def.get("outputs") or []):
        if not isinstance(out_def, dict):
            continue
        for lid in out_def.get("linkIds") or []:
            if not isinstance(lid, int):
                continue
            link = internal_link_map.get(lid)
            if isinstance(link, dict):
                output_sources[(link.get("origin_id"), link.get("origin_slot"))] = idx

    expanded_nodes: list[dict] = []
    for inner in internal_nodes:
        expanded = inner.copy()
        expanded["id"] = f"{outer_id}:{inner.get('id')}"
        expanded["inputs"] = [
            _rewrite_internal_input(inp, internal_link_map, link_id_remap) for inp in inner.get("inputs", []) or []
        ]
        expanded_nodes.append(expanded)

    expanded_links: list = []
    for link in internal_links:
        if not isinstance(link, dict):
            continue
        origin_id = link.get("origin_id")
        target_id = link.get("target_id")
        if origin_id in (_SUBGRAPH_INPUT_NODE_ID, _SUBGRAPH_OUTPUT_NODE_ID):
            continue
        if target_id in (_SUBGRAPH_INPUT_NODE_ID, _SUBGRAPH_OUTPUT_NODE_ID):
            continue
        old_id = link.get("id")
        if not isinstance(old_id, int):
            continue
        new_id = link_id_remap.get(old_id, old_id)
        expanded_links.append(
            [
                new_id,
                f"{outer_id}:{origin_id}",
                link.get("origin_slot"),
                f"{outer_id}:{target_id}",
                link.get("target_slot"),
                link.get("type"),
            ]
        )

    return expanded_nodes, expanded_links, input_targets, output_sources


def _rewrite_internal_input(
    input_info: dict, internal_link_map: dict[int, dict], link_id_remap: dict[int, int]
) -> dict:
    input_copy = input_info.copy()
    link_id = input_info.get("link")
    if not isinstance(link_id, int):
        # Both internal_link_map and link_id_remap are keyed by int IDs; an
        # unhashable (list/dict) link_id would otherwise crash the lookup
        # and abort the whole subgraph expansion.
        return input_copy
    link = internal_link_map.get(link_id)
    if not isinstance(link, dict):
        return input_copy
    if link.get("origin_id") == _SUBGRAPH_INPUT_NODE_ID:
        # Will be reattached to an external link by _rewrite_links_for_subgraphs.
        input_copy["link"] = None
    elif link_id in link_id_remap:
        input_copy["link"] = link_id_remap[link_id]
    return input_copy


def _rewrite_links_for_subgraphs(links: list, ctx: _SubgraphCtx, nodes: list[dict]) -> list:
    """Resolve links that cross subgraph boundaries to their internal endpoints."""
    if not ctx.output_sources and not ctx.input_targets:
        return links

    node_input_updates: dict[str, dict[int, int]] = {}
    updated: list = []
    for link in links:
        if not isinstance(link, (list, tuple)) or len(link) < 6:
            updated.append(link)
            continue
        link_id, src_id, src_slot, tgt_id, tgt_slot, link_type = link[:6]

        src_id_str = str(src_id)
        src_id_out, src_slot_out = _resolve_subgraph_output(src_id_str, src_slot, ctx)

        tgt_id_str = str(tgt_id)
        all_targets = _resolve_subgraph_input_all(tgt_id_str, tgt_slot, ctx)
        # Track input-slot rewrites for ALL targets (one outer input may fan out).
        for resolved_tgt_id, resolved_tgt_slot in all_targets:
            if resolved_tgt_id != tgt_id_str:
                node_input_updates.setdefault(resolved_tgt_id, {})[resolved_tgt_slot] = link_id

        first_tgt_id, first_tgt_slot = all_targets[0]
        updated.append([link_id, src_id_out, src_slot_out, first_tgt_id, first_tgt_slot, link_type])

    # Apply input updates to the expanded internal nodes.
    for node in nodes:
        node_id_str = str(node.get("id"))
        if node_id_str not in node_input_updates:
            continue
        slot_to_link = node_input_updates[node_id_str]
        for slot_idx, input_info in enumerate(node.get("inputs", []) or []):
            if slot_idx in slot_to_link:
                input_info["link"] = slot_to_link[slot_idx]

    return updated


def _resolve_subgraph_output(node_id_str: str, slot: Any, ctx: _SubgraphCtx, depth: int = 0) -> tuple[Any, Any]:
    if depth > _MAX_RESOLUTION_DEPTH:
        return node_id_str, slot
    mapping = ctx.output_sources.get(node_id_str)
    if not mapping:
        return node_id_str, slot
    for (internal_node, internal_slot), out_slot in mapping.items():
        if out_slot == slot:
            new_id = f"{node_id_str}:{internal_node}"
            return _resolve_subgraph_output(new_id, internal_slot, ctx, depth + 1)
    return node_id_str, slot


def _resolve_subgraph_input_all(
    node_id_str: str, slot: Any, ctx: _SubgraphCtx, depth: int = 0
) -> list[tuple[Any, Any]]:
    if depth > _MAX_RESOLUTION_DEPTH:
        return [(node_id_str, slot)]
    mapping = ctx.input_targets.get(node_id_str)
    if not mapping:
        return [(node_id_str, slot)]

    sg_input_idx = slot
    outer_map = ctx.outer_to_input_idx.get(node_id_str)
    if outer_map and slot in outer_map:
        sg_input_idx = outer_map[slot]

    targets = mapping.get(sg_input_idx)
    if not targets:
        return [(node_id_str, slot)]

    out: list[tuple[Any, Any]] = []
    for internal_node, internal_slot in targets:
        new_id = f"{node_id_str}:{internal_node}"
        out.extend(_resolve_subgraph_input_all(new_id, internal_slot, ctx, depth + 1))
    return out or [(node_id_str, slot)]


# ---------------------------------------------------------------------------
# Link map + tracing helpers
# ---------------------------------------------------------------------------


def _is_valid_connection(type_a: Any, type_b: Any) -> bool:
    """Mirror of LiteGraph.isValidConnection from the frontend.

    ``*`` and ``""`` wildcards match anything; comma-separated alternatives are
    expanded; otherwise we case-insensitively compare type names.
    """
    if type_a in (0, "", "*"):
        type_a = 0
    if type_b in (0, "", "*"):
        type_b = 0
    if not type_a or not type_b or type_a == type_b:
        return True
    type_a_s = str(type_a).lower()
    type_b_s = str(type_b).lower()
    if "," not in type_a_s and "," not in type_b_s:
        return type_a_s == type_b_s
    for a in type_a_s.split(","):
        for b in type_b_s.split(","):
            if _is_valid_connection(a.strip(), b.strip()):
                return True
    return False


def _build_link_map(links: list) -> dict[int, dict]:
    link_map: dict[int, dict] = {}
    for link in links:
        if not isinstance(link, (list, tuple)) or len(link) < 6:
            continue
        link_id, src_id, src_slot, tgt_id, tgt_slot, link_type = link[:6]
        link_map[link_id] = {
            "source_id": src_id,
            "source_slot": src_slot,
            "target_id": tgt_id,
            "target_slot": tgt_slot,
            "type": link_type,
        }
    return link_map


def _collect_primitive_values(nodes: list[dict]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for node in nodes:
        if node.get("type") != "PrimitiveNode":
            continue
        widgets = node.get("widgets_values")
        if isinstance(widgets, list) and widgets:
            out[str(node.get("id"))] = widgets[0]
    return out


def _collect_bypassed(nodes: list[dict]) -> set[str]:
    return {str(n.get("id")) for n in nodes if n.get("mode") == _MODE_BYPASS}


def _collect_reroute_sources(nodes: list[dict], link_map: dict[int, dict]) -> dict[str, tuple[Any, Any]]:
    out: dict[str, tuple[Any, Any]] = {}
    for node in nodes:
        if node.get("type") != "Reroute":
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, list) or not inputs or not isinstance(inputs[0], dict):
            continue
        link_id = inputs[0].get("link")
        # ``link_id in link_map`` raises TypeError on unhashable values
        # (e.g. ``link: []`` in a malformed saved file). _collect_reroute_sources
        # runs before the per-node try/except wrapper, so a single bad Reroute
        # would otherwise abort the entire conversion.
        if not isinstance(link_id, int) or link_id not in link_map:
            continue
        ld = link_map[link_id]
        out[str(node.get("id"))] = (ld["source_id"], ld["source_slot"])
    return out


def _collect_get_set_mappings(
    nodes: list[dict], link_map: dict[int, dict]
) -> tuple[dict[str, tuple[Any, Any]], dict[str, str]]:
    """SetNode publishes a value under a name; GetNode reads it back."""
    set_sources: dict[str, tuple[Any, Any]] = {}
    get_vars: dict[str, str] = {}
    for node in nodes:
        node_type = node.get("type")
        widgets = node.get("widgets_values")
        if not isinstance(widgets, list) or not widgets:
            continue
        var_name = widgets[0]
        # var_name becomes a dict key (set_sources[var_name]) and is later
        # checked with ``var_name in set_sources`` inside the tracer. Both
        # require it to be a non-empty string; reject anything else early.
        if not isinstance(var_name, str) or not var_name:
            continue
        if node_type == "SetNode":
            for inp in node.get("inputs") or []:
                if not isinstance(inp, dict):
                    continue
                lid = inp.get("link")
                # See _collect_reroute_sources: unhashable lid would crash
                # the global pre-pass before any per-node guard kicks in.
                if not isinstance(lid, int) or lid not in link_map:
                    continue
                ld = link_map[lid]
                set_sources[var_name] = (ld["source_id"], ld["source_slot"])
                break
        elif node_type == "GetNode":
            get_vars[str(node.get("id"))] = var_name
    return set_sources, get_vars


def _collect_excluded(nodes: list[dict]) -> set[str]:
    """Identify nodes that should never appear in the API output.

    Only ``LoadImageOutput`` is excluded here — it's a UI-only file picker
    for browsing the output folder, with no Python class behind it. All
    other UI-only types are filtered by name via ``_UI_ONLY_NODE_TYPES``.

    Matches the frontend's policy (``executionUtil.ts:graphToPrompt``) and
    cloud-mcp-server's ``shouldIncludeInOutput`` of emitting every
    non-virtual, non-muted, non-bypassed node regardless of whether its
    outputs are wired. The executor only runs nodes reachable from sinks
    (SaveImage, etc.), so unwired nodes are harmless in the prompt.

    We previously applied a "dead-branch" heuristic that dropped any node
    with no downstream consumer; that excluded legitimate sources like an
    unwired ``LoadAudio`` and caused 20+ cloud-mcp oracle fixtures to lose
    nodes that the live frontend emits.
    """
    return {str(n.get("id")) for n in nodes if n.get("type") == "LoadImageOutput"}


class _Tracers:
    """Bundle of upstream-resolution helpers used while emitting each API node."""

    def __init__(
        self,
        *,
        link_map: dict[int, dict],
        nodes: list[dict],
        node_by_id: dict[str, dict],
        bypassed: set[str],
        reroute_sources: dict[str, tuple[Any, Any]],
        set_sources: dict[str, tuple[Any, Any]],
        get_vars: dict[str, str],
        subgraph_ctx: _SubgraphCtx,
    ) -> None:
        self.link_map = link_map
        self.nodes = nodes
        self.node_by_id = node_by_id
        self.bypassed = bypassed
        self.reroute_sources = reroute_sources
        self.set_sources = set_sources
        self.get_vars = get_vars
        self.subgraph_ctx = subgraph_ctx

    def trace_reroute(self, src_id: Any, src_slot: Any) -> tuple[Any, Any]:
        # Iterative to avoid Python's recursion limit on long chains. The
        # body matches a tail-recursive version exactly; the seen-set guards
        # against cyclic ``Reroute -> Reroute -> ...`` loops.
        seen: set[str] = set()
        while True:
            key = str(src_id)
            if key in seen or key not in self.reroute_sources:
                return src_id, src_slot
            seen.add(key)
            src_id, src_slot = self.reroute_sources[key]

    def trace_get_set(self, src_id: Any, src_slot: Any) -> tuple[Any, Any]:
        # Same iterative shape as trace_reroute. Hops through one
        # GetNode -> SetNode pair per step; the seen-set guards against
        # cycles via repeated variable names.
        seen: set[str] = set()
        while True:
            key = str(src_id)
            if key in seen or key not in self.get_vars:
                return src_id, src_slot
            seen.add(key)
            var_name = self.get_vars[key]
            if var_name not in self.set_sources:
                return src_id, src_slot
            src_id, src_slot = self.set_sources[var_name]

    def trace_bypassed(self, src_id: Any, src_slot: Any) -> tuple[Any, Any]:
        # Iterative. Each loop iteration corresponds to walking through one
        # bypassed node; inner calls to trace_get_set / trace_reroute already
        # iterate over their respective chains (no recursion).
        seen: set[Any] = set()
        while True:
            if src_id in seen:
                return src_id, src_slot
            seen.add(src_id)
            if str(src_id) not in self.bypassed:
                return src_id, src_slot

            node = self.node_by_id.get(str(src_id))
            if not node:
                return src_id, src_slot

            outputs = node.get("outputs") or []
            # Guard the slot index — malformed workflows can have non-numeric slots.
            try:
                slot_idx = int(src_slot) if src_slot is not None else 0
            except (TypeError, ValueError):
                slot_idx = 0
            output_type = (
                outputs[slot_idx].get("type")
                if 0 <= slot_idx < len(outputs) and isinstance(outputs[slot_idx], dict)
                else None
            )

            # Pick the input we'll forward the output through. We mix the frontend's
            # strict matcher (ExecutableNodeDTO._getBypassSlotIndex) with the
            # reference converter's permissive fallback, in order of preference:
            #   1. Same-slot input if its type connects to the output type
            #   2. First input whose type matches the output type exactly
            #   3. First input whose type is connection-compatible (handles ``*``
            #      and ``,``-separated alternatives via LiteGraph.isValidConnection)
            #   4. First linked input regardless of type — preserves user intent
            #      when types disagree, matching SethRobinson's reference. The
            #      executor will surface a type mismatch loudly if it matters.
            inputs = node.get("inputs") or []
            chosen_link: int | None = None
            exact_link: int | None = None
            compat_link: int | None = None
            fallback_link: int | None = None

            same_slot_inp = (
                inputs[slot_idx] if 0 <= slot_idx < len(inputs) and isinstance(inputs[slot_idx], dict) else None
            )
            if same_slot_inp:
                lid = same_slot_inp.get("link")
                if (
                    lid is not None
                    and lid in self.link_map
                    and _is_valid_connection(same_slot_inp.get("type"), output_type)
                ):
                    chosen_link = lid

            if chosen_link is None:
                for inp in inputs:
                    if not isinstance(inp, dict):
                        continue
                    lid = inp.get("link")
                    if lid is None or lid not in self.link_map:
                        continue
                    inp_type = inp.get("type")
                    if fallback_link is None:
                        fallback_link = lid
                    if output_type and inp_type == output_type and exact_link is None:
                        exact_link = lid
                    if compat_link is None and _is_valid_connection(inp_type, output_type):
                        compat_link = lid
                chosen_link = exact_link if exact_link is not None else compat_link
                if chosen_link is None:
                    chosen_link = fallback_link

            if chosen_link is None:
                return src_id, src_slot

            ld = self.link_map[chosen_link]
            upstream_id, upstream_slot = ld["source_id"], ld["source_slot"]
            upstream_id, upstream_slot = self.trace_get_set(upstream_id, upstream_slot)
            upstream_id, upstream_slot = self.trace_reroute(upstream_id, upstream_slot)
            src_id, src_slot = upstream_id, upstream_slot
            # Loop continues with the new src_id/src_slot.


# ---------------------------------------------------------------------------
# Per-node emission
# ---------------------------------------------------------------------------


def _wrap_widget_value(value: Any) -> Any:
    """Wrap list widget values to disambiguate them from [node_id, slot] links.

    ComfyUI's executor strips the wrapper before passing to the node. See
    execution.py: ``if "__value__" in val: val = val["__value__"]``.
    """
    if isinstance(value, list):
        return {"__value__": value}
    return value


def process_dynamic_prompt(value: str) -> str:
    """Resolve the ``{a|b|c}`` syntax used in CLIPTextEncode text widgets.

    Port of the frontend's ``processDynamicPrompt`` (``formatUtil.ts``):

    * Strips ``/* ... */`` and ``// ...`` comments first.
    * Picks one alternative at random from each top-level ``{a|b|...}``
      group. Nested groups are recursed into after a choice is made.
    * ``\\{``, ``\\}``, ``\\|`` escape their literal characters.

    Non-deterministic by design — the backend doesn't process the syntax,
    so a workflow saved with ``{red|blue} hat`` would otherwise tokenize
    the braces literally and produce a junk image.
    """
    return _resolve_dynamic_prompt(_DYNAMIC_PROMPT_COMMENT_RE.sub("", value))


def _resolve_dynamic_prompt(value: str) -> str:
    out: list[str] = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        i += 1
        if ch == "\\" and i < n:
            # Preserve the escape marker so the unescape pass at the end can
            # restore the literal character without it being consumed earlier.
            out.append("\\" + value[i])
            i += 1
        elif ch == "{":
            chosen, i = _parse_dynamic_prompt_block(value, i)
            out.append(_resolve_dynamic_prompt(chosen))
        else:
            out.append(ch)
    return _DYNAMIC_PROMPT_UNESCAPE_RE.sub(r"\1", "".join(out))


def _parse_dynamic_prompt_block(value: str, i: int) -> tuple[str, int]:
    """Parse a ``{a|b|...}`` group starting at index ``i`` (just past the ``{``).

    Returns ``(chosen_option, new_i)``. ``new_i`` points past the closing
    ``}`` (or past end-of-string if the group is unterminated — the frontend
    silently degrades on malformed input and we match that).
    """
    options: list[str] = []
    choice: list[str] = []
    depth = 0
    n = len(value)
    while i < n:
        ch = value[i]
        i += 1
        if ch == "\\" and i < n:
            choice.append("\\" + value[i])
            i += 1
            continue
        if ch == "{":
            depth += 1
            choice.append(ch)
        elif ch == "}":
            if depth == 0:
                break
            depth -= 1
            choice.append(ch)
        elif ch == "|" and depth == 0:
            options.append("".join(choice))
            choice = []
        else:
            choice.append(ch)
    options.append("".join(choice))
    return random.choice(options), i


def _dynamic_prompt_input_names(node_type: str | None, node: dict | None, object_info: dict) -> set[str]:
    """Names of inputs whose schema declares ``dynamicPrompts: True``."""
    if not node_type or not node:
        return set()
    schema = _schema_for(node_type, node, object_info)
    if not schema:
        return set()
    input_def = _schema_input_def(schema)
    out: set[str] = set()
    for section in ("required", "optional"):
        section_def = input_def.get(section) or {}
        if not isinstance(section_def, dict):
            continue
        for input_name, input_spec in section_def.items():
            if not isinstance(input_spec, (list, tuple)) or len(input_spec) < 2:
                continue
            options = input_spec[1] if isinstance(input_spec[1], dict) else {}
            if options.get("dynamicPrompts"):
                out.add(input_name)
    return out


def _build_api_node(
    *,
    node: dict,
    node_type: str,
    object_info: dict,
    tracers: _Tracers,
    primitive_values: dict[str, Any],
    bypassed: set[str],
    nodes_to_exclude: set[str],
) -> dict:
    api_node: dict = {"inputs": {}, "class_type": node_type}
    # Resolve the schema once via _schema_for so every consumer
    # (_meta.title, defaults, combo normalization) sees the same thing
    # as the widget-mapping path, even on nodes that carry a ``Node name
    # for S&R`` property pointing at a different class.
    schema = _schema_for(node_type, node, object_info) or {}

    if "title" in node:
        api_node["_meta"] = {"title": node["title"]}
    else:
        api_node["_meta"] = {"title": schema.get("display_name") or node_type}

    link_inputs: dict[str, list] = {}
    primitive_inputs: dict[str, Any] = {}
    for inp in node.get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        input_name = inp.get("name")
        link_id = inp.get("link")
        if not input_name or not isinstance(link_id, int) or link_id not in tracers.link_map:
            continue
        ld = tracers.link_map[link_id]
        actual_id, actual_slot = ld["source_id"], ld["source_slot"]

        actual_id, actual_slot = tracers.trace_get_set(actual_id, actual_slot)
        actual_id, actual_slot = tracers.trace_reroute(actual_id, actual_slot)
        if str(actual_id) in bypassed:
            actual_id, actual_slot = tracers.trace_bypassed(actual_id, actual_slot)
            if str(actual_id) in bypassed:
                # Couldn't find a non-bypassed source — let widget default cover it.
                continue
        # Bypassed source may itself have referenced a GetNode or Reroute.
        actual_id, actual_slot = tracers.trace_get_set(actual_id, actual_slot)
        actual_id, actual_slot = tracers.trace_reroute(actual_id, actual_slot)
        # If we crossed a subgraph boundary while tracing, finalize to internal node.
        actual_id, actual_slot = _resolve_subgraph_output(str(actual_id), actual_slot, tracers.subgraph_ctx)

        actual_id_str = str(actual_id)
        if actual_id_str in primitive_values:
            primitive_inputs[input_name] = _wrap_widget_value(primitive_values[actual_id_str])
        elif actual_id_str in nodes_to_exclude:
            continue
        elif actual_id_str in bypassed:
            continue
        else:
            link_inputs[input_name] = [actual_id_str, actual_slot]

    widget_inputs = _collect_widget_inputs(node, node_type, object_info, link_inputs)
    default_inputs = _collect_default_inputs(schema, widget_inputs, primitive_inputs, link_inputs)

    ordered = _get_ordered_input_names(node_type, node, object_info)
    if ordered:
        # First widget-like values in the declared order, then link inputs.
        # This matches what ComfyUI's "Save (API)" produces.
        for name in ordered:
            if name in widget_inputs:
                api_node["inputs"][name] = widget_inputs[name]
            elif name in primitive_inputs:
                api_node["inputs"][name] = primitive_inputs[name]
            elif name in default_inputs:
                api_node["inputs"][name] = default_inputs[name]
        for name in ordered:
            if name in link_inputs and name not in api_node["inputs"]:
                api_node["inputs"][name] = link_inputs[name]

    # Anything we didn't know an order for is still emitted (preserves data).
    for source in (widget_inputs, primitive_inputs, default_inputs, link_inputs):
        for key, value in source.items():
            if key not in api_node["inputs"]:
                api_node["inputs"][key] = value

    _normalize_combo_values(schema, api_node["inputs"])
    return api_node


# ---------------------------------------------------------------------------
# Widget / input order helpers (driven by /object_info)
# ---------------------------------------------------------------------------


def _schema_for(node_type: str, node: dict, object_info: dict) -> dict | None:
    # Some nodes (litegraph subgraphs) store the real class name under properties.
    properties = node.get("properties") or {}
    alt_name = properties.get("Node name for S&R")
    if isinstance(alt_name, str) and alt_name in object_info:
        return object_info[alt_name]
    return object_info.get(node_type) if isinstance(node_type, str) else None


def _schema_input_def(schema: Any) -> dict:
    """Return the schema's ``input`` block as a dict, or ``{}`` if absent/malformed.

    Every helper that walks INPUT_TYPES sections needs this guard: the raw
    ``schema.get("input") or {}`` pattern returns the value as-is when it's
    truthy, so a malformed schema with ``"input": [...]`` would later crash
    on ``.get(section)``. In practice ``/object_info`` never sends a non-dict
    here, but the rest of the converter follows the same defensive contract.
    """
    if not isinstance(schema, dict):
        return {}
    input_def = schema.get("input")
    return input_def if isinstance(input_def, dict) else {}


def _get_ordered_input_names(node_type: str, node: dict, object_info: dict) -> list[str]:
    schema = _schema_for(node_type, node, object_info)
    if not schema:
        return []
    input_order = schema.get("input_order")
    if not isinstance(input_order, dict):
        input_order = {}
    out: list[str] = []
    for section in ("required", "optional"):
        section_order = input_order.get(section)
        if isinstance(section_order, list):
            out.extend(section_order)
    if out:
        return out
    # Fall back to whatever order is in the input dict itself.
    input_def = _schema_input_def(schema)
    for section in ("required", "optional"):
        section_def = input_def.get(section) or {}
        if isinstance(section_def, dict):
            out.extend(section_def.keys())
    return out


def _is_widget_input(input_spec: Any) -> tuple[bool, bool]:
    """Return (is_widget, is_dynamic_combo) for an INPUT_TYPES spec."""
    if not isinstance(input_spec, (list, tuple)) or not input_spec:
        return False, False
    # ``forceInput: True`` (legacy alias: ``defaultInput``) explicitly demotes
    # a widget-type input to a connection-only slot; the frontend doesn't
    # render a widget for it and the saved workflow has no value for it in
    # widgets_values. Treating it as a widget here would consume a value-slot
    # that doesn't exist and shift every later widget out of position.
    options = input_spec[1] if len(input_spec) >= 2 and isinstance(input_spec[1], dict) else {}
    if options.get("forceInput") or options.get("defaultInput"):
        return False, False
    input_type = input_spec[0]
    if isinstance(input_type, (list, tuple)):
        return True, False  # combo of choices
    if isinstance(input_type, str):
        # ``*`` and ``""`` are wildcard *connection* types — the frontend
        # never renders a widget for them. They slipped through the
        # lowercase fallback below because they have no cased characters
        # (``"*".isupper()`` returns ``False``), so we have to filter them
        # out explicitly. PreviewAny.source: ["*", {}] is the canonical
        # case this used to mis-handle.
        if input_type in ("", "*"):
            return False, False
        if input_type in {"INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"}:
            return True, False
        if input_type.startswith("COMFY_") and "COMBO" in input_type:
            return True, True
        if not input_type.isupper():
            return True, False  # custom (lowercase) widget types
    return False, False


def _dynamic_combo_selected_subs(input_name: str, input_spec: Any, selected: Any) -> list[tuple[str, Any]]:
    """``(dotted_name, spec)`` pairs for the selected option's sub-inputs.

    The dynamic combo's ``widgets_values`` selector value picks one option; that
    option's ``inputs`` mirror an INPUT_TYPES dict. We return every sub-input's
    *spec* (dotted with the parent name, e.g. ``model.size_preset``) and leave
    the widget-vs-connection decision to the caller's ``_is_widget_input`` — the
    same test applied to top-level inputs — so a connection-only sub-input (e.g.
    a ``COMFY_AUTOGROW_V3`` image list) consumes no ``widgets_values`` slot.
    Returns ``[]`` for an unknown selection or a malformed option block (the
    latter would otherwise crash the per-node wrapper and drop the whole node).
    """
    if not isinstance(input_spec, (list, tuple)) or len(input_spec) < 2:
        return []
    options_meta = input_spec[1] if isinstance(input_spec[1], dict) else {}
    options = options_meta.get("options") or []
    for option in options:
        if not isinstance(option, dict) or option.get("key") != selected:
            continue
        sub_def = option.get("inputs")
        if not isinstance(sub_def, dict):
            return []
        subs: list[tuple[str, Any]] = []
        for section in ("required", "optional"):
            section_def = sub_def.get(section) or {}
            if isinstance(section_def, dict):
                for sub_name, sub_spec in section_def.items():
                    subs.append((f"{input_name}.{sub_name}", sub_spec))
        return subs
    return []


def _dynamic_combo_option_keys(input_spec: Any) -> list[Any]:
    """The ``key`` of every option a dynamic combo declares (order preserved)."""
    if not isinstance(input_spec, (list, tuple)) or len(input_spec) < 2:
        return []
    options_meta = input_spec[1] if isinstance(input_spec[1], dict) else {}
    options = options_meta.get("options") or []
    return [option.get("key") for option in options if isinstance(option, dict)]


def _schema_widget_pairs(schema: Any, widget_values: list[Any]) -> list[tuple[str, Any]]:
    """Pair a schema's widget inputs with their ``widgets_values`` slots.

    One ordered walk that unifies name-order and control-marker filtering so the
    two can never disagree. For each schema input in order:

    * skip non-widget inputs (connections, ``forceInput`` demotions, wildcards)
      — they own no slot;
    * consume one slot for a widget input, emitting ``(name, value)``;
    * for a V3 dynamic combo (``COMFY_*COMBO*``) the consumed value is the
      selector — recurse into the selected option's *widget* sub-inputs, each
      consuming a following slot (connection-only subs are skipped, nested
      dynamic combos recurse), so ``model.size_preset``/``model.width`` land in
      order and ``model.images`` never steals a slot;
    * drop a trailing ``control_after_generate`` marker string when the
      just-consumed input is control-flagged (explicit flag or an implicit INT
      ``seed``/``noise_seed``) — sub-inputs are handled identically via recursion.

    Returns ``[]`` when the schema declares no widget inputs, so the caller can
    fall back to node-input inspection exactly as before.
    """
    input_def = _schema_input_def(schema)
    pairs: list[tuple[str, Any]] = []
    vidx = 0

    def consume(name: str, spec: Any, depth: int = 0) -> None:
        nonlocal vidx
        is_widget, is_dynamic = _is_widget_input(spec)
        if not is_widget or vidx >= len(widget_values):
            return
        value = widget_values[vidx]
        pairs.append((name, value))
        vidx += 1
        if is_dynamic:
            subs = _dynamic_combo_selected_subs(name, spec, value)
            if not subs and value not in _dynamic_combo_option_keys(spec):
                # The saved selector no longer names any option in the current
                # schema (model renamed/removed server-side, or object_info /
                # workflow version skew). Its sub-input value slots go
                # unconsumed, so every following widget reads a shifted slot.
                # We can't recover the alignment, but warn so the corruption
                # isn't silent.
                logger.warning(
                    "Dynamic-combo input %r selector %r matched no option in the current "
                    "schema; following widget values may be misaligned",
                    name,
                    value,
                )
            elif depth >= _MAX_DYNAMIC_COMBO_DEPTH:
                logger.warning(
                    "Dynamic-combo nesting for input %r exceeded depth %d; stopping sub-input expansion",
                    name,
                    _MAX_DYNAMIC_COMBO_DEPTH,
                )
            else:
                for sub_name, sub_spec in subs:
                    consume(sub_name, sub_spec, depth + 1)
        elif vidx < len(widget_values) and _has_control_after_generate_companion(name, spec, widget_values[vidx]):
            vidx += 1

    for section in ("required", "optional"):
        section_def = input_def.get(section) or {}
        if not isinstance(section_def, dict):
            continue
        for input_name, input_spec in section_def.items():
            consume(input_name, input_spec)
    return pairs


def _fallback_widget_names(node: dict, widget_values: list[Any]) -> list[str | None]:
    properties = node.get("properties") or {}
    ue_properties = properties.get("ue_properties") or {}
    ue_connectable = ue_properties.get("widget_ue_connectable")
    if isinstance(ue_connectable, dict) and ue_connectable:
        names = list(ue_connectable.keys())
        if len(names) >= len(widget_values):
            return list(names[: len(widget_values)])

    all_inputs: list[str] = []
    connected: set[str] = set()
    widget_flagged: list[str] = []
    for inp in node.get("inputs") or []:
        if not isinstance(inp, dict):
            continue
        name = inp.get("name")
        if not name:
            continue
        all_inputs.append(name)
        if inp.get("link") is not None:
            connected.add(name)
        if inp.get("widget"):
            widget_flagged.append(name)

    if widget_flagged:
        if len(widget_values) > len(widget_flagged):
            extras = [n for n in all_inputs if n not in connected and n not in widget_flagged]
            return widget_flagged + extras[: len(widget_values) - len(widget_flagged)]
        return list(widget_flagged)

    unconnected = [n for n in all_inputs if n not in connected]
    if len(unconnected) >= len(widget_values):
        return unconnected[: len(widget_values)]
    return []


def _filter_control_values(
    widget_values: list[Any],
    node_type: str | None = None,
    node: dict | None = None,
    object_info: dict | None = None,
) -> list[Any]:
    """Drop the control_after_generate strings that follow seed-like INT widgets.

    Schema-aware when a schema is available: only a string immediately
    following an input that declares ``control_after_generate: True`` is
    treated as a control marker. This avoids false positives on legitimate
    STRING/COMBO widget values that happen to equal one of the control
    keywords (e.g. a combo option literally named ``"fixed"``).

    Falls back to a positional string-match heuristic when the schema is
    unavailable — matches SethRobinson's behavior for unknown node types.
    """

    def is_control(v: Any) -> bool:
        return isinstance(v, str) and v in _CONTROL_AFTER_GENERATE_VALUES

    schema = _schema_for(node_type, node, object_info) if node_type and node and object_info else None
    if not schema:
        out: list[Any] = []
        i = 0
        while i < len(widget_values):
            value = widget_values[i]
            if is_control(value):
                i += 1
                continue
            if i + 1 < len(widget_values) and is_control(widget_values[i + 1]):
                out.append(value)
                i += 2
                continue
            out.append(value)
            i += 1
        return out

    out = []
    vidx = 0
    input_def = _schema_input_def(schema)
    for section in ("required", "optional"):
        section_def = input_def.get(section) or {}
        if not isinstance(section_def, dict):
            continue
        for input_name, input_spec in section_def.items():
            if vidx >= len(widget_values):
                break
            is_widget, _is_dynamic = _is_widget_input(input_spec)
            if not is_widget:
                continue
            out.append(widget_values[vidx])
            vidx += 1
            if vidx < len(widget_values) and _has_control_after_generate_companion(
                input_name, input_spec, widget_values[vidx]
            ):
                vidx += 1
    while vidx < len(widget_values):
        out.append(widget_values[vidx])
        vidx += 1
    return out


def _has_control_after_generate_companion(input_name: str, input_spec: Any, next_value: Any) -> bool:
    """True if ``next_value`` should be consumed as a control_after_generate marker.

    Two ways the frontend adds the companion widget:

    * Explicit: the input spec sets ``control_after_generate: True``.
    * Implicit: the input is named ``seed`` or ``noise_seed`` and is INT-typed.
      The frontend's ``useIntWidget`` composable adds the companion in that case
      regardless of the schema flag.

    For the implicit path we peek at the next value: older workflows saved
    before the companion existed don't have the marker string, so we must
    verify the slot really is a control keyword before consuming it.
    """
    options = input_spec[1] if len(input_spec) >= 2 and isinstance(input_spec[1], dict) else {}
    if options.get("control_after_generate"):
        return isinstance(next_value, str) and next_value in _CONTROL_AFTER_GENERATE_VALUES
    input_type = input_spec[0] if input_spec else None
    # ``input_name`` may be dotted for a dynamic-combo sub-input (e.g.
    # ``model.seed``); the frontend's implicit companion keys off the leaf
    # widget name, so match on the final segment.
    leaf_name = input_name.rsplit(".", 1)[-1]
    if input_type == "INT" and leaf_name in ("seed", "noise_seed"):
        return isinstance(next_value, str) and next_value in _CONTROL_AFTER_GENERATE_VALUES
    return False


def _collect_widget_inputs(
    node: dict, node_type: str, object_info: dict, link_inputs: dict[str, list]
) -> dict[str, Any]:
    widget_values = node.get("widgets_values")
    if widget_values is None:
        return {}
    dynamic_prompt_names = _dynamic_prompt_input_names(node_type, node, object_info)

    def emit(name: str, value: Any) -> Any:
        if name in dynamic_prompt_names and isinstance(value, str):
            value = process_dynamic_prompt(value)
        return _wrap_widget_value(value)

    out: dict[str, Any] = {}
    if isinstance(widget_values, dict):
        # Already self-describing; drop UI-only keys and respect link overrides.
        for key, value in widget_values.items():
            if key in ("videopreview", "preview"):
                continue
            if key in link_inputs:
                continue
            out[key] = emit(key, value)
        return out
    if not isinstance(widget_values, list):
        return {}

    if any(isinstance(v, dict) for v in widget_values):
        _absorb_dict_widget_values(widget_values, out, link_inputs)
        return out

    # When a schema is available, a single walk pairs names with values while
    # expanding V3 dynamic combos and dropping control markers together — the
    # name order and the marker filtering can never disagree (which is what let
    # a dynamic combo before a seed steal the seed's slot, e.g. Seedream's
    # ``model`` before ``seed``). It also skips connection-only sub-inputs
    # (e.g. ``COMFY_AUTOGROW_V3`` images) so they never consume a value slot.
    schema = _schema_for(node_type, node, object_info)
    pairs = _schema_widget_pairs(schema, widget_values) if schema else []
    if pairs:
        for name, value in pairs:
            if not name or name in link_inputs:
                continue
            out[name] = emit(name, value)
        return out

    # No schema, or the schema declares no widget inputs: fall back to the
    # positional control-marker heuristic + node-input name inspection, which
    # matches the reference behavior for unknown node types.
    filtered = _filter_control_values(widget_values, node_type, node, object_info)
    names = _fallback_widget_names(node, filtered)
    if not names:
        if filtered:
            logger.warning(
                "Could not map widget_values for unknown node type %r (node %s)",
                node_type,
                node.get("id"),
            )
        return out
    for i, value in enumerate(filtered):
        if i >= len(names):
            break
        name = names[i]
        if not name or name in link_inputs:
            continue
        out[name] = emit(name, value)
    return out


def _absorb_dict_widget_values(widget_values: list[Any], out: dict[str, Any], link_inputs: dict[str, list]) -> None:
    lora_counter = 0
    for value in widget_values:
        if isinstance(value, dict):
            if not value:
                continue
            if "type" in value:
                name = value.get("type")
                if name and name not in link_inputs:
                    out[name] = value
            elif "lora" in value:
                lora_counter += 1
                name = f"lora_{lora_counter}"
                if name in link_inputs:
                    continue
                clean = {k: v for k, v in value.items() if k != "strengthTwo" or v is not None}
                out[name] = clean
        elif isinstance(value, str) and value == "":
            # Frontend's "Add Lora" button serializes as an empty string trailer.
            out.setdefault("➕ Add Lora", value)


def _collect_default_inputs(
    schema: dict | None,
    widget_inputs: dict[str, Any],
    primitive_inputs: dict[str, Any],
    link_inputs: dict[str, list],
) -> dict[str, Any]:
    if not schema:
        return {}
    input_def = _schema_input_def(schema)
    defaults: dict[str, Any] = {}
    for section in ("required", "optional"):
        section_def = input_def.get(section) or {}
        if not isinstance(section_def, dict):
            continue
        for input_name, input_spec in section_def.items():
            if input_name in widget_inputs or input_name in primitive_inputs or input_name in link_inputs:
                continue
            default = _extract_default(input_spec)
            if default is not _MISSING:
                defaults[input_name] = _wrap_widget_value(default)
    return defaults


_MISSING = object()


def _extract_default(input_spec: Any) -> Any:
    if not isinstance(input_spec, (list, tuple)) or not input_spec:
        return _MISSING
    input_type = input_spec[0]
    options = input_spec[1] if len(input_spec) >= 2 and isinstance(input_spec[1], dict) else {}
    if "default" in options:
        return options["default"]
    if isinstance(input_type, list) and input_type:
        return input_type[0]
    if input_type == "COMBO":
        opts = options.get("options")
        if isinstance(opts, list) and opts:
            return opts[0]
    return _MISSING


def _normalize_combo_values(schema: dict | None, inputs: dict[str, Any]) -> None:
    if not schema:
        return
    input_def = _schema_input_def(schema)
    for section in ("required", "optional"):
        section_def = input_def.get(section) or {}
        if not isinstance(section_def, dict):
            continue
        for input_name, input_spec in section_def.items():
            if input_name not in inputs:
                continue
            value = inputs[input_name]
            if not isinstance(value, str):
                continue
            if not isinstance(input_spec, (list, tuple)) or not input_spec:
                continue
            allowed = input_spec[0]
            if not isinstance(allowed, (list, tuple)):
                continue
            if value in allowed:
                continue
            lower_value = value.lower()
            for option in allowed:
                if isinstance(option, str) and option.lower() == lower_value:
                    inputs[input_name] = option
                    break
