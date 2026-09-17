"""Scoring a canvas layout without a human looking at it.

Every prior round of the "is the agent's layout any good" discussion ended in
screenshot opinions, which do not compare across runs and cannot regress a build.
These are the numbers that replace them.

A pure function of a workflow's geometry and links: no rendering, no canvas, no I/O.
That is what makes it usable in three places at once -- as test assertions, as a
regression baseline in CI, and as a telemetry signal emitted at op-mint time so a
layout regression surfaces before a user files it. The overlap Jo reported in
September 2026 took two months to reach us through FE-1653.

Every metric is "lower is better" and 0.0 is the ideal, so they compose into a
single score and a threshold reads the same way for all of them.

What these metrics do NOT capture: whether the layout is *meaningful*. A scorer built
around overlap and crossings will rank a mechanically tidy arrangement above a
semantically grouped one, so a falling score is evidence and not proof. Keep a
human-judged fixture alongside any gate built on this.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

# LiteGraph draws a node's title bar ABOVE its `pos`, so the rectangle a user sees is
# taller than `size`. A scorer that ignores the band reports zero overlap for nodes
# that visibly overlap by up to 30px -- the same blind spot that made the placer
# itself wrong before the title band was modelled.
TITLE_H = 30.0


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
    pos, size = node.get("pos"), node.get("size")
    if not pos or not size:
        return None
    x, y = _pair(pos)
    w, h = _pair(size)
    return (x, y - TITLE_H, w, h + TITLE_H)


def _pair(value) -> tuple[float, float]:
    """Read `[x, y]` or `{"0": x, "1": y}`.

    Both shapes occur in real workflow JSON: the mapping form comes out of some
    serialisation paths, and assuming the list form raises KeyError on files that
    are otherwise perfectly valid.
    """
    if isinstance(value, dict):
        return float(value["0"]), float(value["1"])
    return float(value[0]), float(value[1])


def _centre_y(node: dict) -> float:
    _, y = _pair(node["pos"])
    _, h = _pair(node["size"])
    return y + h / 2


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
    crossings = 0
    for (a1, b1), (a2, b2) in itertools.combinations(live, 2):
        if rects[a1][0] != rects[a2][0] or rects[b1][0] != rects[b2][0]:
            continue
        if (_centre_y(nodes[a1]) - _centre_y(nodes[a2])) * (_centre_y(nodes[b1]) - _centre_y(nodes[b2])) < 0:
            crossings += 1

    preds: dict = {}
    for a, b in live:
        preds.setdefault(b, []).append(a)
    devs = [abs(_centre_y(nodes[k]) - sum(_centre_y(nodes[p]) for p in ps) / len(ps)) for k, ps in preds.items() if ps]

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
