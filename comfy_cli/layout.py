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
