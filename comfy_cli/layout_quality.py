"""Scoring a canvas layout without a human looking at it.

Every prior round of the "is the agent's layout any good" discussion ended in
screenshot opinions, which do not compare across runs and cannot regress a build.
These are the numbers that replace them.

A pure function of a workflow's geometry and links: no rendering, no canvas, no I/O.
That is what makes it usable in three places at once -- as test assertions, as a
regression baseline in CI, and as a telemetry signal emitted at op-mint time so a
layout regression surfaces before a user files it. The overlap this module was built
to detect took two months to reach the team through a user-reported ticket.

Every metric is "lower is better" and 0.0 is the ideal, so they compose into a
single score and a threshold reads the same way for all of them.

What these metrics do NOT capture: whether the layout is *meaningful*. A scorer built
around overlap and crossings will rank a mechanically tidy arrangement above a
semantically grouped one, so a falling score is evidence and not proof. Keep a
human-judged fixture alongside any gate built on this.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

# LiteGraph draws a node's title bar ABOVE its `pos`, so the rectangle a user sees is
# taller than `size`. A scorer that ignores the band reports zero overlap for nodes
# that visibly overlap by up to 30px -- the same blind spot that made the placer
# itself wrong before the title band was modelled.
TITLE_H = 30.0

# Horizontal slack within which two nodes count as the same column.
#
# Chosen against the placer's own geometry rather than picked: adjacent columns are at
# least a node width plus COL_GAP apart (140 + 80 = 220 at the narrowest), so 40 cannot
# merge two real columns, while absorbing the drag, snap and float-round-trip jitter
# that makes exact equality useless on any workflow a user has touched.
COLUMN_TOLERANCE = 40.0


@dataclass(frozen=True)
class LayoutScore:
    """Lower is better on every field; 0.0 is ideal."""

    overlap_area: float
    """Total pairwise overlapping area in px^2. Non-zero means nodes visibly collide."""

    crossings: int
    """Edge pairs that cross between adjacent columns. The Sugiyama step-2 objective."""

    align_deviation: float
    """Mean px between a node's vertical centre and the mean centre of its inputs.

    0 means every edge is horizontal. The Sugiyama step-3 objective, and the metric
    that moves when coordinate assignment works.
    """

    backward_edges: int
    """Links whose target is left of (or level with) their source.

    Dataflow should read left to right. A backward edge is the most legible kind of
    layout failure and the easiest to assert on.
    """

    def is_clean(self) -> bool:
        """No hard failures. Alignment is a quality gradient, not a failure, so it is
        deliberately excluded -- a perfectly readable graph can have a non-zero value."""
        return self.overlap_area == 0.0 and self.crossings == 0 and self.backward_edges == 0


def _rect(node: dict) -> tuple[float, float, float, float] | None:
    """(x, y, w, h) including the title band, or None if the node lacks geometry."""
    if not isinstance(node, dict):
        return None
    pos, size = node.get("pos"), node.get("size")
    if not pos or not size:
        return None
    position = _pair(pos)
    dimensions = _pair(size)
    if position is None or dimensions is None:
        return None
    x, y = position
    w, h = dimensions
    return (x, y - TITLE_H, w, h + TITLE_H)


def _pair(value) -> tuple[float, float] | None:
    """Read a finite `[x, y]` or `{"0": x, "1": y}`, else return None.

    Both shapes occur in real workflow JSON: the mapping form comes out of some
    serialisation paths. Malformed and non-finite values are absent geometry for
    scoring purposes, rather than errors or inputs to nonsensical metrics.
    """
    try:
        if isinstance(value, dict):
            pair = float(value["0"]), float(value["1"])
        elif isinstance(value, (list, tuple)):
            pair = float(value[0]), float(value[1])
        else:
            return None
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    return pair if all(math.isfinite(item) for item in pair) else None


def _centre_y(rect: tuple[float, float, float, float]) -> float:
    """Return the node body's vertical centre from its title-inclusive rectangle."""
    _, y, _, h = rect
    return y + TITLE_H + (h - TITLE_H) / 2


def _columns(rects: dict) -> dict:
    """Group nodes into columns, tolerating small horizontal drift.

    Single-pass clustering over sorted x, starting a new column whenever the gap to
    that column's left bound exceeds COLUMN_TOLERANCE. Deliberately not `round(x / tol)`:
    fixed buckets put x=19 and x=21 in different columns while claiming a 40px
    tolerance, so the tolerance would be real only for pairs that happen to miss a
    bucket boundary.
    """
    order = sorted(rects, key=lambda k: rects[k][0])
    column: dict = {}
    index = 0
    left = rects[order[0]][0] if order else 0.0
    for at, key in enumerate(order):
        if at and rects[key][0] - left > COLUMN_TOLERANCE:
            index += 1
            left = rects[key][0]
        column[key] = index
    return column


def score(nodes: dict, edges: list[tuple]) -> LayoutScore:
    """Score a layout.

    `nodes` maps node id to a dict with `pos` and `size`. `edges` is a list of
    (source_id, target_id). Nodes without geometry, and edges naming an unknown
    node, are skipped rather than raising: a partial graph should produce a partial
    score, not an exception in a telemetry path.
    """
    rects = {k: r for k, v in nodes.items() if (r := _rect(v)) is not None}

    overlap = 0.0
    for a, b in itertools.combinations(rects, 2):
        ax, ay, aw, ah = rects[a]
        bx, by, bw, bh = rects[b]
        dx = min(ax + aw, bx + bw) - max(ax, bx)
        dy = min(ay + ah, by + bh) - max(ay, by)
        if dx > 0 and dy > 0:
            overlap += dx * dy

    live = [(a, b) for a, b in edges if a in rects and b in rects]

    backward = sum(1 for a, b in live if rects[b][0] <= rects[a][0])

    # Crossings are counted only between edges sharing both columns. Counting every
    # geometric intersection instead would make the number depend on column spacing,
    # which is not what crossing reduction optimises and would drift whenever COL_GAP
    # changed.
    #
    # Columns are matched within COLUMN_TOLERANCE rather than by equality. Exact
    # equality holds for freshly placed nodes and for nothing else: one drag, one
    # snap, one float round-trip through JSON and every node is in its own column, so
    # every pair is skipped and the metric reports a confident zero. For a telemetry
    # signal read off user-edited workflows that is the whole failure mode -- a broken
    # measurement and a perfect score are the same number.
    column = _columns(rects)

    crossings = 0
    for (a1, b1), (a2, b2) in itertools.combinations(live, 2):
        if column[a1] != column[a2] or column[b1] != column[b2]:
            continue
        if (_centre_y(rects[a1]) - _centre_y(rects[a2])) * (_centre_y(rects[b1]) - _centre_y(rects[b2])) < 0:
            crossings += 1

    preds: dict = {}
    for a, b in live:
        preds.setdefault(b, []).append(a)
    devs = [abs(_centre_y(rects[k]) - sum(_centre_y(rects[p]) for p in ps) / len(ps)) for k, ps in preds.items() if ps]

    return LayoutScore(
        overlap_area=overlap,
        crossings=crossings,
        align_deviation=(sum(devs) / len(devs) if devs else 0.0),
        backward_edges=backward,
    )


def score_workflow(workflow: dict) -> LayoutScore:
    """Score a workflow in the frontend's serialised format.

    Convenience for the telemetry and CI paths, which hold a whole workflow rather
    than the placer's internal dicts. Understands both the `links` array and
    per-node `inputs[].link` back-references, because both appear in the wild.
    """
    nodes = {str(n["id"]): n for n in workflow.get("nodes", []) if "id" in n}

    edges: list[tuple] = []
    for link in workflow.get("links") or []:
        # A serialised link is an array of link id, origin node, origin slot, target
        # node, target slot and type; positions 1 and 3 are the two node ids.
        if isinstance(link, (list, tuple)) and len(link) >= 4:
            edges.append((str(link[1]), str(link[3])))

    if not edges:
        by_link = {}
        for n in workflow.get("nodes", []):
            for out in n.get("outputs") or []:
                for link_id in out.get("links") or []:
                    by_link[link_id] = str(n["id"])
        for n in workflow.get("nodes", []):
            for inp in n.get("inputs") or []:
                src = by_link.get(inp.get("link"))
                if src is not None:
                    edges.append((src, str(n["id"])))

    return score(nodes, edges)
