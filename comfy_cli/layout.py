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
# Matches LiteGraph's NODE_WIDGET_HEIGHT (LiteGraphGlobal.ts). Was 24, which
# over-measured every widget row and compounded on widget-heavy nodes.
WIDGET_H = 20.0
PAD_H = 12.0
MIN_H = 60.0
ORIGIN = (40.0, 60.0)
DEFAULT_SIZE = (210.0, 100.0)
_MARGIN = 10.0
_GUARD = 1000  # bounded collision-shift loop
_ORDER_SWEEPS = 4  # barycentre passes for crossing reduction; converges well before this

# LiteGraph's NODE_TITLE_HEIGHT (LiteGraphGlobal.ts). A node's `pos` is the top-left
# of its BODY; the title bar is drawn ABOVE it, at `pos[1] - NODE_TITLE_HEIGHT`
# (LGraphCanvas.ts:6624). Every rectangle this module reasons about must therefore
# include that band, or the collision check is blind to the top 30px of every node --
# which, against a _MARGIN of 10, let nodes the model called clear sit 20px inside each
# other. The frontend's own arrange is title-aware (useArrangeNodes.ts:81); this is the
# same correction on the CLI side.
TITLE_H = 30.0

# The citations above and below name upstream FILES, not line numbers: the line numbers
# these comments originally carried were already stale by the time they landed (they said
# LiteGraphGlobal.ts:61/64/65/71; the declarations are nowhere near). Every constant in
# this module is checked against upstream BY NAME by scripts/check_litegraph_parity.py.


# --- width model, ported from LiteGraph's own computeSize (LGraphNode.ts:2020-2072) ---
#
# Width was a flat NODE_W for every node, while LiteGraph derives it from label text.
# COL_GAP absorbs 80px of error and then fails: a node rendering 360px wide overlaps its
# neighbour by 40px, 420px by 100px. That is the n8n#38093 failure mode (their SDK sized a
# node 96x96 while the editor drew a 320x128 card).
#
# The port is faithful rather than invented: when no canvas is available LiteGraph itself
# falls back to `font_size * text.length * 0.6` (its compute_text_size), which is exactly
# what a CLI can compute. It is NOT pixel-identical to a browser with real font metrics —
# proportional fonts vary per glyph — but it tracks content instead of ignoring it.
NODE_TEXT_SIZE = 14.0  # LiteGraph NODE_TEXT_SIZE (LiteGraphGlobal.ts)
LG_NODE_WIDTH = 140.0  # LiteGraph NODE_WIDTH (LiteGraphGlobal.ts)
_CHAR_W = 0.6  # LiteGraph's no-canvas glyph-width fallback
# Horizontal room a widget row needs beyond its label, from BaseWidget: a minimum value
# width of 42, plus a margin of 15, an arrow margin of 6 and an arrow width of 10 on each
# side (BaseWidget.ts).
_WIDGET_MIN_VALUE_WIDTH = 42.0
_WIDGET_MARGIN = 15.0
_WIDGET_ARROW_MARGIN = 6.0
_WIDGET_ARROW_WIDTH = 10.0
_WIDGET_PADDING = _WIDGET_MIN_VALUE_WIDTH + 2.0 * (_WIDGET_MARGIN + _WIDGET_ARROW_MARGIN + _WIDGET_ARROW_WIDTH)


# LiteGraph does not stack widget rows flush. `LGraphNode.computeSize` accumulates
# each widget's own height plus a 4px inter-row gap, looping over the node's widgets, and
# then adds a single 8px pad for the whole block after the loop finishes.
#
# The flat PAD_H this module used instead happens to land within 4px at exactly ONE
# widget and drifts further with every additional row: at three widgets it under-measures
# by 8px, at six by 20px. Under-measurement is the direction that produces the overlap
# users report, because the placer leaves a gap it believes is clear.
#
# Found by the browser harness rather than by reading: ComfyUI_frontend
# `browser_tests/tests/agent/agentLayoutQuality.spec.ts` measured two agent-added
# CLIPTextEncode nodes overlapping by 6 graph px on a recording whose positions this
# module chose. See that spec for the full measurement.
_WIDGET_ROW_GAP = 4.0
_WIDGET_BLOCK_PAD = 8.0


# A multiline text box is not a widget ROW, it is a text AREA, and the difference is
# most of a node. Measured against the real renderer (see the table in _widgets_height):
# an ordinary widget occupies 24px, a multiline one 166px -- roughly seven rows. Charging
# every widget the ordinary rate under-measures a CLIPTextEncode by 142px, which is the
# dominant term in the overlap the browser harness reproduces.
MULTILINE_WIDGET_H = 166.0

# Image-upload widgets attach a DOM image host below their ordinary combo/button
# rows. On its first populated render the frontend deliberately grows that host
# to at least 190px (`createImageHost` in `src/scripts/ui/imagePreview.ts`). It is
# not described as another object_info widget, so the catalog-only model used to
# omit the whole block and stack rows of populated LoadImage nodes roughly 190px
# into one another.
IMAGE_PREVIEW_MIN_H = 190.0
_IMAGE_PREVIEW_UPLOAD_FLAGS = frozenset({"image_upload", "animated_image_upload"})


def count_multiline(node_meta, widget_names) -> int:
    """How many of `widget_names` are multiline, per the catalog.

    Both callers that size a node must derive this the SAME way, and they did not. The
    planner passed a multiline count; `workflow_ops.add_node` computed its own size without
    one, and that is the size PERSISTED onto the node -- the size every later collision
    check reads. So the planner was right and the saved state was wrong, and a second call
    would place a node on top of one it had itself under-measured.

    Matching is by name because widget order is the render order, not the declaration
    order, and `options` is guarded because a port may carry none at all.
    """
    names = set(widget_names)
    return sum(
        1
        for p in getattr(node_meta, "inputs", [])
        if p.name in names and getattr(getattr(p, "options", None), "multiline", False)
    )


def count_image_previews(node_meta, widget_names) -> int:
    """Count frontend image hosts implied by upload-backed widget inputs."""
    names = set(widget_names)
    return sum(
        1
        for p in getattr(node_meta, "inputs", [])
        if p.name in names
        and bool(set(getattr(getattr(p, "options", None), "upload_flags", ())) & _IMAGE_PREVIEW_UPLOAD_FLAGS)
    )


def _widgets_height(n_widgets: int, n_multiline: int = 0) -> float:
    """Vertical space n widget rows occupy, using LiteGraph's own accumulation.

    Zero widgets take zero space -- the block padding is inside the `if (widgets?.length)`
    guard upstream, so a node with no widgets must not pay for it.

    The constants are not read off the source, they are SOLVED from rendered geometry.
    Measuring twelve core classes in a real browser and fitting
    `H = 6 + 20*max(link_inputs, outputs) + (widgets + 8)` gives an exact match on all
    twelve, with an ordinary widget at 24 and a multiline one at 166:

        class                   measured   predicted
        CLIPTextEncode               200         200   <- 1 multiline
        KSampler                     262         262   <- 7 ordinary, 4 links
        EmptyLatentImage             106         106
        CheckpointLoaderSimple        98          98
        SaveImage                     58          58
        LoadImage                    102         102
        VAEDecode                     46          46   <- 0 widgets
        PreviewImage                  26          26   <- 0 widgets
        ConditioningCombine           46          46   <- 0 widgets
        LatentUpscale                130         130
        CLIPSetLastLayer              58          58
        ImageScale                   130         130

    The 24 and the 8 were already here from the per-row-gap fix and this measurement
    confirms both independently. Note the fit needs TRUE link inputs: KSampler has seven
    widgets but only six widget-backed inputs, because `control_after_generate` is a
    widget with no input at all, so deriving links as `len(inputs) - len(widgets)` gets
    it wrong by a row.
    """
    if n_widgets <= 0:
        return 0.0
    # Clamp rather than trust. A node cannot have more multiline widgets than widgets, and
    # charging five multiline areas to a one-widget node is not a case worth codifying --
    # it only arises from a bad catalog or a caller bug, and silently over-measuring by
    # 700px hides both.
    n_multiline = min(max(n_multiline, 0), n_widgets)
    ordinary = n_widgets - n_multiline
    return ordinary * (WIDGET_H + _WIDGET_ROW_GAP) + n_multiline * MULTILINE_WIDGET_H + _WIDGET_BLOCK_PAD


def _text_w(text: str | None) -> float:
    return NODE_TEXT_SIZE * len(text or "") * _CHAR_W


# Minimum rendered width, measured rather than derived. LiteGraph's own formula is
# `NODE_WIDTH * (1.5 if widgets else 1.0)` = 210, which no widget-bearing node actually
# renders at: across 27 classes measured in a real browser, every node carrying a widget
# renders at 270 or wider, and every node carrying a MULTILINE widget renders at 400 or
# wider. Fourteen of fourteen non-multiline widget nodes land at exactly 270 when their
# content is narrower, and seven of seven multiline nodes at exactly 400 -- the shape of a
# floor, not a fixed width (KSamplerAdvanced 312, CheckpointLoader 396.9, ControlNetApply
# 317.9 all exceed it on content).
#
# Direction matters more than precision here. Under-estimating width is what produces the
# overlap users report: the placer puts the next column a node-width away and the real node
# reaches past it. Over-estimating only wastes canvas, which nobody has ever filed.
#
# Known over-estimate: `Note` and `MarkdownNote` carry a multiline widget but render at 140
# because they are annotation nodes with no slots. They get floored to 400 here and will be
# over-spaced. Accepted deliberately -- see the direction argument above.
WIDGET_MIN_WIDTH = 270.0
MULTILINE_MIN_WIDTH = 400.0


def _min_width(has_widgets: bool, n_multiline: int = 0) -> float:
    if n_multiline > 0:
        return MULTILINE_MIN_WIDTH
    if has_widgets:
        return WIDGET_MIN_WIDTH
    return LG_NODE_WIDTH


def estimate_width(
    title: str | None = None,
    input_labels: tuple[str, ...] = (),
    output_labels: tuple[str, ...] = (),
    widget_labels: tuple[str, ...] = (),
    n_multiline: int = 0,
) -> float:
    """LiteGraph's computeSize width, using its own no-canvas text metric."""
    title_width = TITLE_H + _text_w(title) + TITLE_H * 0.33
    input_width = max((_text_w(t) for t in input_labels), default=0.0)
    output_width = max((_text_w(t) for t in output_labels), default=0.0)
    widget_width = max((_text_w(t) for t in widget_labels), default=0.0)
    if widget_width:
        widget_width += _WIDGET_PADDING
    min_width = _min_width(bool(widget_labels), n_multiline)
    centre_padding = 5.0 if (input_width and output_width) else 0.0
    slots_width = input_width + output_width + 2.0 * SLOT_H + centre_padding
    return max(slots_width, widget_width, title_width, min_width)


def estimate_size(
    n_link_inputs: int,
    n_outputs: int,
    n_widgets: int,
    *,
    n_multiline: int = 0,
    n_image_previews: int = 0,
    title: str | None = None,
    input_labels: tuple[str, ...] = (),
    output_labels: tuple[str, ...] = (),
    widget_labels: tuple[str, ...] = (),
) -> list[float]:
    """Estimated BODY size, excluding the title bar (see TITLE_H).

    Pass the label strings to get a content-derived width (see `estimate_width`). Without
    them the width falls back to the flat NODE_W, which is what every caller used before
    and is kept so this stays additive — but it is the weaker estimate, and callers that
    have the catalog metadata should pass it.

    Deliberately conservative on height: HEADER_H is retained even though the title is
    now modelled separately, so the estimate runs ~30px tall. Over-spacing is invisible;
    under-spacing is the overlap users report.
    """
    h = (
        HEADER_H
        + SLOT_H * max(n_link_inputs, n_outputs)
        + _widgets_height(n_widgets, n_multiline)
        + max(n_image_previews, 0) * IMAGE_PREVIEW_MIN_H
        + PAD_H
    )
    if title is None and not (input_labels or output_labels or widget_labels):
        w = NODE_W
    else:
        w = estimate_width(title, input_labels, output_labels, widget_labels, n_multiline)
    return [w, max(h, MIN_H)]


def _pair(value) -> tuple[float, float] | None:
    """Read a geometry pair, tolerating both shapes litegraph serialises.

    `schemas/workflow.json` documents `pos`/`size` as "passed through verbatim --
    litegraph has serialized this as both [x, y] and {"0": x, "1": y}". Integer indexing
    a mapping raises KeyError, so the object form has to be read by string key rather
    than merely caught: falling back to a default here would discard geometry that is
    present and usable, and silently mis-place a real node.
    """
    if isinstance(value, dict):
        try:
            return float(value["0"]), float(value["1"])
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None


def occupied(pos, size) -> tuple[float, float, float, float]:
    """The rectangle a node actually covers on the canvas, title bar included.

    Kept public so callers building a candidate rect for a not-yet-placed node use the
    same convention as `_rect` does for placed ones. Mixing the two spaces silently
    reintroduces the title-band blindness this function exists to remove.
    """
    xy, wh = _pair(pos), _pair(size)
    if xy is None or wh is None:
        # Unreadable geometry: fall back to the default BODY, then apply the same title
        # band the normal path does. Returning the bare DEFAULT_SIZE here would
        # under-report the node's lower 30px and let a collision check accept an overlap.
        return (0.0, -TITLE_H, DEFAULT_SIZE[0], DEFAULT_SIZE[1] + TITLE_H)
    return (xy[0], xy[1] - TITLE_H, wh[0], wh[1] + TITLE_H)


def _rect(node: dict) -> tuple[float, float, float, float]:
    """Occupied rect of a node already in the workflow (title bar included)."""
    pos = node.get("pos") or [0.0, 0.0]
    size = node.get("size") or list(DEFAULT_SIZE)
    return occupied(pos, size)


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
    box, top-aligned, sliding down past any collision.

    Returns a POS (body top-left). `_bbox` is in occupied space, which sits TITLE_H
    above the pos of its topmost node, so the vertical alignment converts back.
    """
    nodes = [n for n in workflow.get("nodes") or [] if isinstance(n, dict)]
    box = _bbox(nodes)
    if box is None:
        return list(ORIGIN)
    # box[1] is the top of the highest TITLE BAR; + TITLE_H puts this node's body top
    # level with the others' rather than one title-height too high.
    x, y = box[2] + COL_GAP, box[1] + TITLE_H
    for _ in range(_GUARD):
        if not any(_overlaps(occupied((x, y), size), _rect(n)) for n in nodes):
            break
        y += ROW_GAP
    return [x, y]


def rebase_template(existing_nodes: list, template: dict) -> None:
    """Translate a whole inserted template so it lands beside the target
    graph's existing nodes, instead of at whatever absolute coordinates the
    template happened to be authored/exported with.

    Mutates `template` in place. Every node (and group) that carries a `pos`
    (or `bounding`) is shifted by the SAME delta, so the template's own
    internal relative layout is preserved exactly -- this is a uniform
    translation of the whole block, not a per-node reshuffle, the same
    convention `cascade_pos` uses for a single new node.

    A no-op when there is nothing to be beside (`existing_nodes` is empty) or
    nothing positioned to move (no template node carries a `pos`); an
    unpositioned node is left exactly as it was, matching the empty-graph
    case in `cascade_pos`.
    """
    existing_box = _bbox([n for n in existing_nodes if isinstance(n, dict)])
    if existing_box is None:
        return

    positioned = [n for n in template.get("nodes") or [] if isinstance(n, dict) and _pair(n.get("pos")) is not None]
    if not positioned:
        return

    template_box = _bbox(positioned)
    # Mirrors cascade_pos: place the block to the right of the existing
    # bounding box, top-aligned with it. Both boxes are in the same
    # occupied-space (title bar included), and that band cancels out of a
    # delta, so it applies directly to `pos` values below.
    dx = (existing_box[2] + COL_GAP) - template_box[0]
    dy = existing_box[1] - template_box[1]

    for node in positioned:
        x, y = _pair(node["pos"])
        node["pos"] = [x + dx, y + dy]

    for group in template.get("groups") or []:
        if not isinstance(group, dict):
            continue
        bounding = group.get("bounding")
        pair = _pair(bounding[:2]) if isinstance(bounding, (list, tuple)) and len(bounding) >= 2 else None
        if pair is None:
            continue
        gx, gy = pair
        group["bounding"] = [gx + dx, gy + dy, *bounding[2:]]


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
            widget_names = tuple(graph.widget_order(spec["class_type"]))
            # object_info already marks multiline inputs; the catalog parses it into
            # PortOptions.multiline. Match by name because widget_order is the render
            # order, which is not the order inputs are declared in.
            # Guard `options` itself, not just the attribute on it. A port that carries
            # no options at all is not hypothetical -- every test double here is one, and
            # so is any catalog entry parsed from an object_info that omitted the block.
            # `getattr(p.options, ...)` raises AttributeError on those before the default
            # can apply.
            _multiline = {p.name for p in m.inputs if getattr(getattr(p, "options", None), "multiline", False)}
            size = estimate_size(
                len([p for p in m.inputs if p.is_link]),
                len(m.outputs),
                len(widget_names),
                n_multiline=count_multiline(m, widget_names),
                n_image_previews=count_image_previews(m, widget_names),
                # LiteGraph renders `display_name || name`, and width is derived from the
                # title, so sizing from class_type under-estimates whenever they differ.
                title=(getattr(m, "display_name", "") or spec["class_type"]),
                input_labels=tuple(p.name for p in m.inputs if p.is_link),
                output_labels=tuple(p.name for p in m.outputs),
                widget_labels=widget_names,
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

    # Column stride must follow the widest node in each depth, not a constant.
    # Widths are content-derived now (see estimate_width), so a fixed NODE_W + COL_GAP
    # stride of 320 lets a 360px depth-0 node reach 40px into depth 1 — an overlap
    # `collides()` cannot catch, because it only compares new nodes against EXISTING
    # workflow nodes, never against each other.
    depth_w: dict[int, float] = {}
    for k in movable:
        d = adds[k]["depth"]
        depth_w[d] = max(depth_w.get(d, 0.0), adds[k]["size"][0])
    max_depth = max(depth_w, default=0)
    # x offset of each depth from the block's left edge.
    depth_dx: dict[int, float] = {}
    _run = 0.0
    for d in range(max_depth + 1):
        depth_dx[d] = _run
        _run += depth_w.get(d, NODE_W) + COL_GAP
    block_width = max(_run - COL_GAP, 0.0)  # trailing gap is not part of the block

    if src_anchors:
        # New nodes fed by existing ones: place right of the feeders, as before.
        arects = [_rect(a) for a in src_anchors]
        base_x = max(r[0] + r[2] for r in arects) + COL_GAP
        base_y = min(r[1] for r in arects) + TITLE_H  # occupied-space top -> pos
    elif dst_anchors:
        # New nodes that feed INTO existing ones: place the whole new block to
        # the left so the edge still reads left-to-right, not backwards.
        drects = [_rect(a) for a in dst_anchors]
        # Offset by the block's REAL width so its rightmost column clears the
        # destination; the old fixed stride let a wide deepest node overlap it.
        base_x = min(r[0] for r in drects) - block_width - COL_GAP
        base_y = min(r[1] for r in drects) + TITLE_H  # occupied-space top -> pos
    else:
        box = _bbox(list(existing.values()))
        base_x, base_y = (box[2] + COL_GAP, box[1] + TITLE_H) if box else ORIGIN

    # --- Sugiyama step 2: crossing reduction -----------------------------------
    # Layer assignment alone (step 1, above) decides WHICH column a node is in but not
    # its order WITHIN the column, so nodes used to stack in the order they happened to
    # appear in the ops array. Three inputs feeding one sampler crossed their wires for
    # no reason other than authoring order. Order each column by the barycentre of its
    # neighbours, the standard heuristic, with the insertion rank as a deterministic
    # tiebreak so the result stays replay-convergent.
    preds: dict[str, list[str]] = {k: [] for k in movable}
    succs: dict[str, list[str]] = {k: [] for k in movable}
    for s, t in edges:
        if s in preds and t in preds:
            succs[s].append(t)
            preds[t].append(s)

    rank = {k: i for i, k in enumerate(order)}
    cols: dict[int, list[str]] = {}
    for k in movable:
        cols.setdefault(adds[k]["depth"], []).append(k)
    slot = {k: i for d in cols for i, k in enumerate(cols[d])}

    def _bary(k: str, side: dict[str, list[str]]) -> float:
        ns = [slot[n] for n in side[k] if n in slot]
        return sum(ns) / len(ns) if ns else float(slot[k])

    for _ in range(_ORDER_SWEEPS):
        moved = False
        for forward in (True, False):
            side = preds if forward else succs
            for d in sorted(cols, reverse=not forward):
                reordered = sorted(cols[d], key=lambda k: (_bary(k, side), rank[k]))
                if reordered != cols[d]:
                    cols[d] = reordered
                    moved = True
                for i, k in enumerate(cols[d]):
                    slot[k] = i
        if not moved:
            break

    # --- Sugiyama step 3: coordinate assignment --------------------------------
    # y used to stack from base_y regardless of what a node connects to, so a one-node
    # column sat flush against the top of a five-node column instead of level with the
    # node it feeds. Centre each node on the mean centre of its predecessors, then push
    # down only as far as the column order requires. Depths are processed in increasing
    # order and edges always run to a strictly greater depth, so predecessors are placed.
    for d in sorted(cols):
        cursor = base_y
        for k in cols[d]:
            h = adds[k]["size"][1]
            want = cursor
            centres = [adds[p]["pos"][1] + adds[p]["size"][1] / 2.0 for p in preds[k] if "pos" in adds[p]]
            if centres:
                want = sum(centres) / len(centres) - h / 2.0
            y = max(want, cursor)
            adds[k]["pos"] = [base_x + depth_dx[d], y]
            # Clear this node's body bottom, the gap, AND the next node's title bar.
            cursor = y + h + ROW_GAP + TITLE_H

    # --- collision resolution ---------------------------------------------------
    # Obstacles are every EXISTING node plus any new node pinned to an explicit `at`:
    # a pinned sibling is as real on the canvas as a pre-existing one, and nothing
    # used to check against it. New-vs-new among MOVABLE nodes needs no check: columns
    # are spaced by the widest node at each depth and `cursor` is monotone within a
    # column, so the construction above cannot produce an internal overlap.
    obstacles = [_rect(n) for n in existing.values()]
    obstacles += [occupied(adds[k]["pinned"], adds[k]["size"]) for k in order if adds[k]["pinned"] is not None]

    def _overlap_depth() -> float:
        """How far down the block must move to clear every obstacle, 0 if clear."""
        worst = 0.0
        for k in movable:
            r = occupied(adds[k]["pos"], adds[k]["size"])
            for o in obstacles:
                if _overlaps(r, o):
                    worst = max(worst, (o[1] + o[3] + _MARGIN) - r[1])
        return worst

    # Jump straight past the blocking obstacle instead of stepping ROW_GAP at a time.
    # The old loop could march 1000 * ROW_GAP = 40,000px in 40px increments, which is
    # both slow and lands the block far below where it needed to be.
    for _ in range(_GUARD):
        if not movable:
            break
        delta = _overlap_depth()
        if delta <= 0:
            break
        for k in movable:
            adds[k]["pos"][1] += delta

    for k in movable:
        out[adds[k]["i"]]["at"] = adds[k]["pos"]
    return out
