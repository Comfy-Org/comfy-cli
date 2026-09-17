"""Tests for the layout quality scorer.

The scorer is what lets "is the layout better" be answered with a number instead of a
screenshot, so its own correctness has to be established first: a metric that reports
0 on a broken layout would launder every future regression as an improvement.
"""

from __future__ import annotations

import pytest

from comfy_cli.layout_quality import TITLE_H, score, score_workflow


def _n(x, y, w=200.0, h=100.0):
    return {"pos": [x, y], "size": [w, h]}


class TestOverlap:
    def test_separated_nodes_score_zero(self):
        nodes = {"a": _n(0, 0), "b": _n(400, 0)}
        assert score(nodes, []).overlap_area == 0.0

    def test_overlapping_nodes_report_the_area(self):
        # Bodies intersect 50x50, but the rectangles include the 30px title band, so
        # `a` spans y -30..100 and `b` spans y 20..150: 50 wide by 80 tall.
        nodes = {"a": _n(0, 0, 100, 100), "b": _n(50, 50, 100, 100)}
        assert score(nodes, []).overlap_area == pytest.approx(50 * 80)

    def test_title_band_counts_as_occupied(self):
        """The blind spot this scorer exists to not repeat.

        `b` sits 10px below `a`'s body, so a scorer comparing only `pos`+`size` calls
        this clean. LiteGraph draws b's 30px title bar upward into that gap, so the
        user sees a 20px overlap. Getting this wrong is what made the placer itself
        wrong before the title band was modelled.
        """
        nodes = {"a": _n(0, 0, 100, 100), "b": _n(0, 110, 100, 100)}
        s = score(nodes, [])
        assert s.overlap_area == pytest.approx(100 * 20)

    def test_touching_edges_do_not_count(self):
        """Adjacency is not overlap; a strict inequality keeps the metric from crying wolf."""
        nodes = {"a": _n(0, 0, 100, 100), "b": _n(100, 0, 100, 100)}
        assert score(nodes, []).overlap_area == 0.0

    def test_nodes_without_geometry_are_skipped(self):
        nodes = {"a": _n(0, 0), "b": {"pos": [0, 0]}, "c": {}}
        assert score(nodes, []).overlap_area == 0.0


class TestCrossings:
    def test_parallel_edges_do_not_cross(self):
        nodes = {"a": _n(0, 0), "b": _n(0, 200), "c": _n(400, 0), "d": _n(400, 200)}
        assert score(nodes, [("a", "c"), ("b", "d")]).crossings == 0

    def test_swapped_targets_cross(self):
        nodes = {"a": _n(0, 0), "b": _n(0, 200), "c": _n(400, 0), "d": _n(400, 200)}
        assert score(nodes, [("a", "d"), ("b", "c")]).crossings == 1

    def test_a_dragged_node_still_counts_as_its_column(self):
        """The bug that made this metric useless as telemetry.

        Exact x equality holds for freshly placed nodes and for nothing else. One
        drag, one snap, one float round-trip through JSON, and every node sits in its
        own column, so every pair is skipped and the score reads a confident 0. For a
        signal read off user-edited workflows a broken measurement and a perfect score
        would be the same number.
        """
        nodes = {"a": _n(0, 0), "b": _n(1, 300), "c": _n(400, 0), "d": _n(400, 300)}
        assert score(nodes, [("a", "d"), ("b", "c")]).crossings == 1

    def test_tolerance_is_not_a_fixed_bucket(self):
        """`round(x / 40)` would split 19 and 21 into different columns.

        A fixed bucket advertises a 40px tolerance but only delivers it to pairs that
        happen not to straddle a boundary, which is worse than no tolerance because it
        fails intermittently and looks deliberate.
        """
        straddling = {"a": _n(19, 0), "b": _n(21, 300), "c": _n(400, 0), "d": _n(400, 300)}
        assert score(straddling, [("a", "d"), ("b", "c")]).crossings == 1

        near_edge = {"a": _n(0, 0), "b": _n(39, 300), "c": _n(400, 0), "d": _n(400, 300)}
        assert score(near_edge, [("a", "d"), ("b", "c")]).crossings == 1

    def test_tolerance_cannot_merge_two_real_columns(self):
        """40px is under the narrowest real column stride (NODE_WIDTH 140 + COL_GAP 80)."""
        nodes = {"a": _n(0, 0), "b": _n(0, 300), "c": _n(400, 0), "d": _n(800, 300)}
        assert score(nodes, [("a", "d"), ("b", "c")]).crossings == 0

    def test_edges_in_different_columns_are_not_compared(self):
        """Only edges sharing both columns are counted.

        Counting every geometric intersection would make the number depend on COL_GAP,
        so a spacing change would look like a layout regression.
        """
        nodes = {"a": _n(0, 0), "b": _n(400, 200), "c": _n(800, 0), "d": _n(1200, 200)}
        assert score(nodes, [("a", "b"), ("c", "d")]).crossings == 0


class TestAlignment:
    def test_horizontal_edge_has_zero_deviation(self):
        nodes = {"a": _n(0, 0), "b": _n(400, 0)}
        assert score(nodes, [("a", "b")]).align_deviation == 0.0

    def test_offset_target_reports_the_offset(self):
        nodes = {"a": _n(0, 0), "b": _n(400, 60)}
        assert score(nodes, [("a", "b")]).align_deviation == pytest.approx(60.0)

    def test_a_node_is_measured_against_the_mean_of_its_inputs(self):
        """Two inputs 200px apart: the ideal position is the midpoint between them."""
        nodes = {"a": _n(0, 0), "b": _n(0, 200), "c": _n(400, 100)}
        assert score(nodes, [("a", "c"), ("b", "c")]).align_deviation == 0.0

    def test_no_edges_gives_zero_rather_than_an_error(self):
        assert score({"a": _n(0, 0)}, []).align_deviation == 0.0


class TestBackwardEdges:
    def test_left_to_right_is_clean(self):
        nodes = {"a": _n(0, 0), "b": _n(400, 0)}
        assert score(nodes, [("a", "b")]).backward_edges == 0

    def test_right_to_left_is_counted(self):
        nodes = {"a": _n(400, 0), "b": _n(0, 0)}
        assert score(nodes, [("a", "b")]).backward_edges == 1

    def test_same_column_is_counted(self):
        """A link inside one column has no direction on screen and reads as a defect."""
        nodes = {"a": _n(0, 0), "b": _n(0, 200)}
        assert score(nodes, [("a", "b")]).backward_edges == 1


class TestEdgeRobustness:
    def test_edges_naming_unknown_nodes_are_skipped(self):
        """Telemetry must not raise on a partial graph."""
        s = score({"a": _n(0, 0)}, [("a", "ghost"), ("ghost", "a")])
        assert s.crossings == 0 and s.backward_edges == 0

    def test_mapping_shaped_geometry_is_read(self):
        """Real workflow JSON contains both `[x, y]` and `{"0": x, "1": y}`."""
        nodes = {
            "a": {"pos": {"0": 0.0, "1": 0.0}, "size": {"0": 100.0, "1": 100.0}},
            "b": {"pos": {"0": 50.0, "1": 50.0}, "size": {"0": 100.0, "1": 100.0}},
        }
        assert score(nodes, []).overlap_area == pytest.approx(50 * 80)


class TestIsClean:
    def test_a_tidy_graph_is_clean(self):
        nodes = {"a": _n(0, 0), "b": _n(400, 0)}
        assert score(nodes, [("a", "b")]).is_clean()

    def test_alignment_alone_does_not_make_it_dirty(self):
        """Alignment is a gradient, not a failure: a readable graph can be misaligned."""
        s = score({"a": _n(0, 0), "b": _n(400, 300)}, [("a", "b")])
        assert s.align_deviation > 0 and s.is_clean()

    def test_overlap_makes_it_dirty(self):
        nodes = {"a": _n(0, 0, 100, 100), "b": _n(50, 50, 100, 100)}
        assert not score(nodes, []).is_clean()


class TestScoreWorkflow:
    def test_reads_the_links_array(self):
        wf = {
            "nodes": [
                {"id": 1, "pos": [0, 0], "size": [200, 100]},
                {"id": 2, "pos": [400, 0], "size": [200, 100]},
            ],
            "links": [[10, 1, 0, 2, 0, "IMAGE"]],
        }
        assert score_workflow(wf).is_clean()
        assert score_workflow(wf).align_deviation == 0.0

    def test_falls_back_to_slot_link_ids(self):
        """Some serialisations omit the top-level `links` array."""
        wf = {
            "nodes": [
                {"id": 1, "pos": [0, 0], "size": [200, 100], "outputs": [{"links": [10]}]},
                {"id": 2, "pos": [400, 200], "size": [200, 100], "inputs": [{"link": 10}]},
            ]
        }
        assert score_workflow(wf).align_deviation == pytest.approx(200.0)

    def test_empty_workflow_scores_zero(self):
        s = score_workflow({})
        assert s.is_clean() and s.overlap_area == 0.0

    def test_detects_a_backward_link(self):
        wf = {
            "nodes": [
                {"id": 1, "pos": [400, 0], "size": [200, 100]},
                {"id": 2, "pos": [0, 0], "size": [200, 100]},
            ],
            "links": [[10, 1, 0, 2, 0, "IMAGE"]],
        }
        assert score_workflow(wf).backward_edges == 1


def test_title_constant_matches_the_placer():
    """The scorer and the placer must agree about what a node occupies.

    If they drift, the scorer starts grading the placer against a different geometry
    than the placer optimises for, and both can be self-consistently wrong.
    """
    from comfy_cli import layout

    assert TITLE_H == layout.TITLE_H


# --- the scorer applied to the placer's own output ---------------------------------
#
# The tests above prove the metric is correct. These prove it is USEFUL: they score
# what assign_positions actually produces, so a future change that degrades layout
# fails here instead of being noticed on a canvas weeks later. This is the regression
# baseline the QA plan calls for, at the cheapest tier that does not need a browser.


class _Port:
    def __init__(self, n):
        self.name, self.is_link = n, True


class _Meta:
    display_name = "N"
    inputs = [_Port("in")]
    outputs = [_Port("out")]


class _Graph:
    def node(self, _ct):
        return _Meta()

    def widget_order(self, _ct):
        return []


def _place(specs):
    from comfy_cli import layout

    out = layout.assign_positions({"nodes": []}, _Graph(), [dict(s) for s in specs])
    size = layout.estimate_size(1, 1, 0, title="N", input_labels=("in",), output_labels=("out",))
    nodes = {s["as"]: {"pos": s["at"], "size": size} for s in out if s.get("op") == "add_node"}
    edges = [(s["from"].split(".")[0], s["to"].split(".")[0]) for s in specs if s.get("op") == "connect"]
    return nodes, edges


def _chain(names):
    specs = [{"op": "add_node", "class_type": "N", "as": n} for n in names]
    specs += [{"op": "connect", "from": f"{a}.out", "to": f"{b}.in"} for a, b in zip(names, names[1:])]
    return specs


class TestPlacerBaseline:
    """Baselines over the placer's real output.

    Verified against the pre-#883 parent: only `test_interleaved_chains_score_perfectly`
    goes red there (3 crossings, 176px mean deviation). The other four pass on the
    parent too, so they are FORWARD guards -- they lock in behaviour that is already
    correct -- not evidence that #883 fixed them. Stating that here because a green
    block of five reads as five fixes, and only one of them is.

    In particular `test_a_wide_batch_never_overlaps` does not exercise the new-vs-new
    collision fix: with no edges the placer assigns each node from a descending cursor,
    so the old code never reached the branch either. The case that does exercise it is
    `test_pinned_siblings_are_obstacles_for_movable_nodes` in test_layout.py.
    """

    def test_a_linear_chain_scores_perfectly(self):
        s = score(*_place(_chain(["load", "encode", "sample", "decode", "save"])))
        assert s.is_clean()
        assert s.align_deviation == 0.0

    def test_interleaved_chains_score_perfectly(self):
        """Authored in the order that crosses every wire; step 2 must undo it."""
        names = ("a1", "b1", "c1", "a2", "b2", "c2")
        specs = [{"op": "add_node", "class_type": "N", "as": n} for n in names]
        specs += [
            {"op": "connect", "from": "a1.out", "to": "c2.in"},
            {"op": "connect", "from": "b1.out", "to": "b2.in"},
            {"op": "connect", "from": "c1.out", "to": "a2.in"},
        ]
        s = score(*_place(specs))
        assert s.crossings == 0
        assert s.align_deviation == 0.0

    def test_a_diamond_scores_clean(self):
        """One source fanning into two branches that rejoin -- the SDXL refiner shape."""
        specs = [{"op": "add_node", "class_type": "N", "as": n} for n in ("src", "l", "r", "join")]
        specs += [
            {"op": "connect", "from": "src.out", "to": "l.in"},
            {"op": "connect", "from": "src.out", "to": "r.in"},
            {"op": "connect", "from": "l.out", "to": "join.in"},
            {"op": "connect", "from": "r.out", "to": "join.in"},
        ]
        s = score(*_place(specs))
        assert s.is_clean(), f"diamond layout is not clean: {s}"

    def test_a_wide_batch_never_overlaps(self):
        """Twenty unconnected nodes in one batch.

        With no edges there is no dataflow to lay out, so every node lands in the same
        column and the only thing keeping them apart is collision resolution against
        the other NEW nodes -- the check that did not exist before #883.
        """
        specs = [{"op": "add_node", "class_type": "N", "as": f"n{i}"} for i in range(20)]
        s = score(*_place(specs))
        assert s.overlap_area == 0.0, f"{s.overlap_area}px^2 of overlap across a 20-node batch"

    def test_scoring_is_deterministic(self):
        """Replay convergence depends on placement being a pure function of its inputs.

        If the same specs scored differently across runs, the metric could not gate
        anything -- and, more importantly, two replicas replaying one op log would
        disagree about where the nodes are.
        """
        specs = _chain(["a", "b", "c", "d"])
        assert score(*_place(specs)) == score(*_place(specs))
