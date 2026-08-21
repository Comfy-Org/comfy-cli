"""Deterministic canvas placement for CLI-minted nodes.

Positions are decided at op-mint time (add_node / apply_specs) and frozen into
the emitted ops, so replay stays convergent. Everything here is a pure function
of its inputs — no randomness, no clock, no I/O. Existing nodes are NEVER moved:
layout only chooses positions for nodes being minted in the current call.
"""

from __future__ import annotations

COL_GAP = 80.0
ROW_GAP = 40.0
NODE_W = 240.0
HEADER_H = 30.0
SLOT_H = 20.0
WIDGET_H = 24.0
PAD_H = 12.0
MIN_H = 60.0
ORIGIN = (40.0, 60.0)
DEFAULT_SIZE = (210.0, 100.0)
_MARGIN = 10.0
_GUARD = 1000  # bounded collision-shift loop


def estimate_size(n_link_inputs: int, n_outputs: int, n_widgets: int) -> list[float]:
    h = HEADER_H + SLOT_H * max(n_link_inputs, n_outputs) + WIDGET_H * n_widgets + PAD_H
    return [NODE_W, max(h, MIN_H)]


def _rect(node: dict) -> tuple[float, float, float, float]:
    pos = node.get("pos") or [0.0, 0.0]
    size = node.get("size") or list(DEFAULT_SIZE)
    try:
        return (float(pos[0]), float(pos[1]), float(size[0]), float(size[1]))
    except (TypeError, ValueError, IndexError):
        return (0.0, 0.0, *DEFAULT_SIZE)


def _overlaps(a: tuple, b: tuple, margin: float = _MARGIN) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw + margin <= bx or bx + bw + margin <= ax or ay + ah + margin <= by or by + bh + margin <= ay)


def _bbox(nodes: list) -> tuple[float, float, float, float] | None:
    rects = [_rect(n) for n in nodes if isinstance(n, dict)]
    if not rects:
        return None
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[0] + r[2] for r in rects),
        max(r[1] + r[3] for r in rects),
    )


def cascade_pos(workflow: dict, size: list[float]) -> list[float]:
    """Default position for a single minted node: right of the graph's bounding
    box, top-aligned, sliding down past any collision."""
    nodes = [n for n in workflow.get("nodes") or [] if isinstance(n, dict)]
    box = _bbox(nodes)
    if box is None:
        return list(ORIGIN)
    x, y = box[2] + COL_GAP, box[1]
    for _ in range(_GUARD):
        if not any(_overlaps((x, y, size[0], size[1]), _rect(n)) for n in nodes):
            break
        y += ROW_GAP
    return [x, y]


def assign_positions(workflow: dict, graph, specs: list) -> list:
    """Fill `at` on every add_node spec that lacks one, using the batch's own
    connects for dataflow layering. Returns spec copies; non-add specs and
    explicit `at` values pass through untouched. Pure: same inputs → same output."""
    out = [dict(s) if isinstance(s, dict) else s for s in specs]
    adds: dict[str, dict] = {}
    order: list[str] = []
    for i, spec in enumerate(out):
        if not (isinstance(spec, dict) and spec.get("op") == "add_node"):
            continue
        m = graph.node(spec.get("class_type") or "")
        if m is not None:
            size = estimate_size(
                len([p for p in m.inputs if p.is_link]),
                len(m.outputs),
                len(graph.widget_order(spec["class_type"])),
            )
        else:
            size = list(DEFAULT_SIZE)  # unknown type: apply_specs will error later
        key = spec.get("as") or f"__new{i}"
        adds[key] = {"i": i, "size": size, "depth": 0, "pinned": spec.get("at")}
        order.append(key)
    if not adds:
        return out

    existing = {n.get("id"): n for n in workflow.get("nodes") or [] if isinstance(n, dict)}
    edges: list[tuple[str, str]] = []
    src_anchors: list[dict] = []  # existing nodes that feed a new node (old -> new)
    dst_anchors: list[dict] = []  # existing nodes fed by a new node (new -> old)

    def endpoint(ref):
        node_part = str(ref).partition(".")[0].strip()
        # `$alias` is sugar for `alias` (see workflow_ops.resolve_ref); `${...}`
        # is a recipe-param hole that apply_specs rejects — not an alias.
        if node_part.startswith("$") and not node_part.startswith("${"):
            node_part = node_part[1:]
        if node_part in adds:
            return ("new", node_part)
        nid = int(node_part) if node_part.lstrip("-").isdigit() else node_part
        if nid in existing:
            return ("old", nid)
        return (None, None)

    for spec in out:
        if not (isinstance(spec, dict) and spec.get("op") == "connect"):
            continue
        skind, s = endpoint(spec.get("from", ""))
        tkind, t = endpoint(spec.get("to", ""))
        if skind == "old" and tkind == "new":
            src_anchors.append(existing[s])
            adds[t]["depth"] = max(adds[t]["depth"], 1)
        elif skind == "new" and tkind == "old":
            dst_anchors.append(existing[t])
        elif skind == "new" and tkind == "new":
            edges.append((s, t))

    # Longest-path layering over new→new edges via relaxation to a fixpoint,
    # bounded by the worst-case chain length. A valid batch has no cycles
    # among new nodes, so this always converges within the bound regardless
    # of the order connects appear in the spec list.
    passes = max(1, len(adds) - 1)
    for _ in range(passes):
        changed = False
        for s, t in edges:
            cand = adds[s]["depth"] + 1
            if cand > adds[t]["depth"]:
                adds[t]["depth"] = cand
                changed = True
        if not changed:
            break

    movable = [k for k in order if adds[k]["pinned"] is None]

    if src_anchors:
        # New nodes fed by existing ones: place right of the feeders, as before.
        arects = [_rect(a) for a in src_anchors]
        base_x = max(r[0] + r[2] for r in arects) + COL_GAP
        base_y = min(r[1] for r in arects)
    elif dst_anchors:
        # New nodes that feed INTO existing ones: place the whole new block to
        # the left so the edge still reads left-to-right, not backwards.
        drects = [_rect(a) for a in dst_anchors]
        max_depth = max((adds[k]["depth"] for k in movable), default=0)
        base_x = min(r[0] for r in drects) - (max_depth + 1) * (NODE_W + COL_GAP)
        base_y = min(r[1] for r in drects)
    else:
        box = _bbox(list(existing.values()))
        base_x, base_y = (box[2] + COL_GAP, box[1]) if box else ORIGIN

    col_y: dict[int, float] = {}
    for k in movable:
        a = adds[k]
        x = base_x + a["depth"] * (NODE_W + COL_GAP)
        y = col_y.get(a["depth"], base_y)
        col_y[a["depth"]] = y + a["size"][1] + ROW_GAP
        a["pos"] = [x, y]

    def collides() -> bool:
        return any(_overlaps((*adds[k]["pos"], *adds[k]["size"]), _rect(n)) for k in movable for n in existing.values())

    for _ in range(_GUARD):
        if not movable or not collides():
            break
        for k in movable:  # shift the whole new block, never existing nodes
            adds[k]["pos"][1] += ROW_GAP

    for k in movable:
        out[adds[k]["i"]]["at"] = adds[k]["pos"]
    return out
