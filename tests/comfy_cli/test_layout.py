import pytest

from comfy_cli import layout
from comfy_cli import layout_quality as quality


def _node(nid, pos, size=(210, 100)):
    return {"id": nid, "type": "X", "pos": list(pos), "size": list(size)}


def test_estimate_size_grows_with_slots_and_widgets():
    small = layout.estimate_size(1, 1, 0)
    big = layout.estimate_size(4, 2, 5)
    assert big[1] > small[1]
    assert small[1] >= 60  # never below a renderable minimum
    assert small[0] == big[0] == layout.NODE_W


def test_cascade_pos_empty_graph_is_origin():
    wf = {"nodes": [], "links": []}
    assert layout.cascade_pos(wf, [210, 100]) == list(layout.ORIGIN)


def test_cascade_pos_places_right_of_bbox_without_overlap():
    wf = {"nodes": [_node(1, (0, 0)), _node(2, (300, 200))], "links": []}
    x, y = layout.cascade_pos(wf, [210, 100])
    assert x >= 300 + 210  # strictly right of the rightmost node's right edge
    new_rect = (x, y, 210, 100)
    for n in wf["nodes"]:
        assert not layout._overlaps(new_rect, layout._rect(n))


def test_cascade_pos_is_deterministic():
    wf = {"nodes": [_node(1, (50, 80))], "links": []}
    assert layout.cascade_pos(wf, [240, 120]) == layout.cascade_pos(wf, [240, 120])


class _FakeMeta:
    def __init__(self, n_in, n_out):
        self.inputs = [type("P", (), {"is_link": True, "name": f"i{k}", "type": "X"})() for k in range(n_in)]
        self.outputs = [type("P", (), {"name": f"o{k}", "type": "X"})() for k in range(n_out)]


class _FakeGraph:
    def node(self, class_type):
        return _FakeMeta(2, 1)

    def widget_order(self, class_type):
        return ["a", "b"]


def test_assign_positions_layers_by_dataflow():
    wf = {"nodes": [], "links": []}
    specs = [
        {"op": "add_node", "class_type": "Loader", "as": "l"},
        {"op": "add_node", "class_type": "Sampler", "as": "s"},
        {"op": "add_node", "class_type": "Save", "as": "v"},
        {"op": "connect", "from": "l.0", "to": "s.model"},
        {"op": "connect", "from": "s.0", "to": "v.images"},
    ]
    out = layout.assign_positions(wf, _FakeGraph(), specs)
    xl, xs, xv = (out[i]["at"][0] for i in range(3))
    assert xl < xs < xv  # left-to-right by dataflow depth


def test_assign_positions_anchors_right_of_existing_source():
    wf = {"nodes": [_node(7, (100, 100), (200, 120))], "links": []}
    specs = [
        {"op": "add_node", "class_type": "Upscale", "as": "u"},
        {"op": "connect", "from": "7.0", "to": "u.image"},
    ]
    out = layout.assign_positions(wf, _FakeGraph(), specs)
    assert out[0]["at"][0] >= 100 + 200  # right of the anchor's right edge
    # and it must not overlap the anchor
    assert not layout._overlaps((*out[0]["at"], layout.NODE_W, 100), layout._rect(wf["nodes"][0]))


def test_assign_positions_respects_explicit_at_and_is_deterministic():
    wf = {"nodes": [], "links": []}
    specs = [
        {"op": "add_node", "class_type": "A", "as": "a", "at": [999, 999]},
        {"op": "add_node", "class_type": "B", "as": "b"},
    ]
    out1 = layout.assign_positions(wf, _FakeGraph(), specs)
    out2 = layout.assign_positions(wf, _FakeGraph(), specs)
    assert out1[0]["at"] == [999, 999]
    assert out1 == out2


def test_assign_positions_new_source_into_existing_target_goes_left():
    wf = {"nodes": [_node(7, (500, 100), (200, 120))], "links": []}
    specs = [
        {"op": "add_node", "class_type": "Upscale", "as": "u"},
        {"op": "connect", "from": "u.0", "to": "7.image"},
    ]
    out = layout.assign_positions(wf, _FakeGraph(), specs)
    assert out[0]["at"][0] + layout.NODE_W <= 500  # fully left of the anchor
    assert not layout._overlaps((*out[0]["at"], layout.NODE_W, 100), layout._rect(wf["nodes"][0]))


def test_assign_positions_reverse_order_connects_full_depth():
    wf = {"nodes": [], "links": []}
    specs = [
        {"op": "add_node", "class_type": "A", "as": "a"},
        {"op": "add_node", "class_type": "B", "as": "b"},
        {"op": "add_node", "class_type": "C", "as": "c"},
        {"op": "add_node", "class_type": "D", "as": "d"},
        {"op": "connect", "from": "c.0", "to": "d.in"},
        {"op": "connect", "from": "b.0", "to": "c.in"},
        {"op": "connect", "from": "a.0", "to": "b.in"},
    ]
    out = layout.assign_positions(wf, _FakeGraph(), specs)
    xa, xb, xc, xd = (out[i]["at"][0] for i in range(4))
    assert xa < xb < xc < xd


# --- title-band regression ------------------------------------------------
# A node's `pos` is its BODY top-left; LiteGraph draws the title bar ABOVE it at
# `pos[1] - NODE_TITLE_HEIGHT`. layout.py used to treat `pos` as the top of the whole
# box, so its collision check was blind to the top 30px of every node and, against a
# 10px margin, let nodes it called clear sit 20px inside each other on screen.


def test_occupied_includes_the_title_band_above_pos():
    x, y, w, h = layout.occupied((100.0, 200.0), (240.0, 120.0))
    assert (x, w) == (100.0, 240.0)
    assert y == 200.0 - layout.TITLE_H, "occupied rect must start at the title bar, not the body"
    assert h == 120.0 + layout.TITLE_H


def test_widget_height_matches_litegraph():
    """Each extra widget row costs its height PLUS LiteGraph's 4px inter-row gap.

    This test previously asserted a delta of exactly 20 and so encoded the bug it was
    named after: `LGraphNode.computeSize` accumulates `widget_height + 4` per row, and
    modelling only the 20 under-measures every node with more than one widget. The
    browser harness caught the consequence -- two agent-added CLIPTextEncode nodes
    rendered 20 graph px taller than this module predicted and overlapped by 6px.
    """
    assert layout.WIDGET_H == 20.0  # LiteGraph NODE_WIDGET_HEIGHT; this was 24
    one, two = layout.estimate_size(1, 1, 1), layout.estimate_size(1, 1, 2)
    assert two[1] - one[1] == layout.WIDGET_H + layout._WIDGET_ROW_GAP == 24.0


def test_widgetless_node_pays_no_widget_block_padding():
    """Upstream's block padding lives inside `if (widgets?.length)`.

    A node with no widgets must not be charged the 8px, or every Reroute and every
    pure-routing node is modelled 8px taller than it draws -- harmless for overlap,
    but it would make the estimate wrong in the direction that wastes canvas.
    """
    none_, one = layout.estimate_size(1, 1, 0), layout.estimate_size(1, 1, 1)
    assert one[1] - none_[1] == 32.0


def test_widget_block_grows_the_way_litegraph_accumulates():
    """Three widgets cost 3*(20+4)+8, not 3*20 -- a 20px difference at three rows."""
    base = layout.estimate_size(1, 1, 0)[1]
    three = layout.estimate_size(1, 1, 3)[1]
    assert three - base == 3 * (20.0 + 4.0) + 8.0
    assert three - base > 3 * 20.0, "the old flat model under-measured a three-widget node"


def test_cascade_leaves_room_for_the_next_node_title():
    """Regression: stacking must clear the title bar, not just the bodies.

    Pre-fix, a node forced to slide down could land with its title bar overlapping the
    body of the node above it, because neither rectangle modelled the band.
    """
    wf = {"nodes": [{"id": 1, "pos": [0.0, 0.0], "size": [240.0, 100.0]}]}
    # Same column: force a vertical slide by asking for a spot the cascade must move.
    size = [240.0, 100.0]
    pos = layout.cascade_pos(wf, size)
    placed = layout.occupied(pos, size)
    existing = layout._rect(wf["nodes"][0])
    assert not layout._overlaps(placed, existing)
    # And the occupied rects genuinely do not intersect, margin aside.
    px, py, pw, ph = placed
    ex, ey, ew, eh = existing
    assert px >= ex + ew or ex >= px + pw or py >= ey + eh or ey >= py + ph


def test_stacked_column_bodies_clear_by_at_least_the_title_band():
    """Two new nodes in one column must not have the lower one's title in the upper's body."""

    class _Port:
        is_link = True
        name = "samples"

    class _Meta:
        inputs = [_Port()]
        outputs = [_Port()]

    class _Graph:
        def node(self, _ct):
            return _Meta()

        def widget_order(self, _ct):
            return []

    specs = [
        {"op": "add_node", "class_type": "A", "as": "a"},
        {"op": "add_node", "class_type": "B", "as": "b"},
    ]
    out = layout.assign_positions({"nodes": []}, _Graph(), specs)
    ats = [s["at"] for s in out]
    assert len(ats) == 2
    if ats[0][0] == ats[1][0]:  # same column -> stacked
        upper, lower = sorted(ats, key=lambda p: p[1])
        size = layout.estimate_size(1, 1, 0)
        upper_bottom = upper[1] + size[1]
        lower_title_top = lower[1] - layout.TITLE_H
        # Require the FULL row gap, not merely non-overlap: the pre-fix stride (which
        # omitted TITLE_H) still left the lower title 10px below the upper body, so a
        # bare non-overlap assertion passes on the broken code and protects nothing.
        assert lower_title_top - upper_bottom >= layout.ROW_GAP, (
            f"stacked column must clear ROW_GAP between bodies and the next title; got {lower_title_top - upper_bottom}"
        )


# --- content-derived width ------------------------------------------------
# Width used to be a flat NODE_W=240 for every node while LiteGraph derives it from label
# text. COL_GAP absorbs 80px of error and then fails, which is the n8n#38093 failure mode.


def test_width_tracks_content_instead_of_being_constant():
    narrow = layout.estimate_width("Reroute", ("in",), ("out",), ())
    wide = layout.estimate_width(
        "CheckpointLoaderSimple",
        (),
        ("MODEL", "CLIP", "VAE"),
        ("ckpt_name",),
    )
    assert wide > narrow, "a long-titled, widget-bearing node must estimate wider than a Reroute"


def test_width_respects_litegraph_minimums():
    # No widgets -> NODE_WIDTH floor; widgets -> NODE_WIDTH * 1.5.
    assert layout.estimate_width("x", (), (), ()) >= layout.LG_NODE_WIDTH
    assert layout.estimate_width("x", (), (), ("w",)) >= layout.LG_NODE_WIDTH * 1.5


def test_long_title_widens_the_node():
    short = layout.estimate_width("A", (), (), ())
    long = layout.estimate_width("A" * 60, (), (), ())
    assert long > short + 200, "title text must drive width like LiteGraph's title_width does"


def test_estimate_size_without_labels_keeps_the_old_constant_width():
    # Additive: existing callers that pass only counts are unchanged.
    assert layout.estimate_size(1, 1, 0)[0] == layout.NODE_W


def test_wide_node_does_not_get_a_neighbour_placed_inside_it():
    """Regression: the pre-fix flat 240 put the next node inside a wide one.

    A node whose real width is 360 covers x=[0,360]; the old model thought 240, so the
    cascade placed the next at 240+80=320 — 40px inside it.
    """
    wide = layout.estimate_width("CheckpointLoaderSimpleWithNoiseSelect", (), ("MODEL", "CLIP", "VAE"), ("ckpt_name",))
    wf = {"nodes": [{"id": 1, "pos": [0.0, 0.0], "size": [wide, 100.0]}]}
    nxt = layout.cascade_pos(wf, [240.0, 100.0])
    assert nxt[0] >= wide + layout.COL_GAP, "next node must clear the wide node's real extent"


# --- review findings on PR #882 -------------------------------------------------------


def test_mapping_shaped_geometry_is_read_not_discarded():
    """litegraph serialises pos/size as both [x, y] and {"0": x, "1": y}.

    schemas/workflow.json documents both shapes. Integer-indexing the object form raises
    KeyError; falling back to a default would silently mis-place a node whose real
    geometry was right there.
    """
    node = {"pos": {"0": 100.0, "1": 200.0}, "size": {"0": 240.0, "1": 120.0}}
    x, y, w, h = layout._rect(node)
    assert (x, w) == (100.0, 240.0)
    assert y == 200.0 - layout.TITLE_H
    assert h == 120.0 + layout.TITLE_H


def test_unreadable_geometry_fallback_still_includes_the_title_band():
    x, y, w, h = layout.occupied(None, None)
    assert y == -layout.TITLE_H
    assert h == layout.DEFAULT_SIZE[1] + layout.TITLE_H, "fallback must not under-report the body"


def test_batch_columns_use_the_widest_node_not_a_fixed_stride():
    """A wide depth-0 node must not reach into depth 1.

    collides() only compares new nodes against EXISTING workflow nodes, never against
    each other, so a fixed NODE_W + COL_GAP stride hides this overlap entirely.
    """

    class _P:
        is_link = True
        name = "a_very_long_input_slot_name_to_force_width"

    class _Wide:
        display_name = "A Node With A Deliberately Very Long Display Name"
        inputs = [_P()]
        outputs = [_P()]

    class _Graph:
        def node(self, _ct):
            return _Wide()

        def widget_order(self, _ct):
            return ["a_long_widget_name"]

    specs = [
        {"op": "add_node", "class_type": "Wide", "as": "a"},
        {"op": "add_node", "class_type": "Wide", "as": "b"},
        {"op": "connect", "from": "a.out", "to": "b.in"},
    ]
    out = layout.assign_positions({"nodes": []}, _Graph(), specs)
    at = {s["as"]: s["at"] for s in out if s.get("op") == "add_node"}
    width = layout.estimate_size(
        1,
        1,
        1,
        title="A Node With A Deliberately Very Long Display Name",
        input_labels=("a_very_long_input_slot_name_to_force_width",),
        output_labels=("a_very_long_input_slot_name_to_force_width",),
        widget_labels=("a_long_widget_name",),
    )[0]
    assert width > layout.NODE_W + layout.COL_GAP, "fixture must be wide enough to expose a fixed stride"
    assert at["b"][0] - at["a"][0] >= width + layout.COL_GAP, "depth-1 column must clear the widest depth-0 node"


# --- batch layout quality (crossing reduction + coordinate assignment) ----------------
# `assign_positions` implemented only Sugiyama's step 1 (layer assignment). Nodes stacked
# within a column in ops-array order, and y ignored what a node connected to. These
# helpers score a placement so the improvement is measured rather than asserted by eye.


def _score(nodes, edges):
    """(crossings, mean input-alignment deviation).

    Delegates to comfy_cli.layout_quality rather than keeping a second copy. The two
    used to be separate implementations of the same metric, which is the shape of bug
    where the tests and the telemetry quietly disagree about whether a layout improved.
    """
    s = quality.score(nodes, edges)
    return s.crossings, s.align_deviation


class _QPort:
    def __init__(self, n):
        self.name, self.is_link = n, True


class _QMeta:
    display_name = "N"
    inputs = [_QPort("in")]
    outputs = [_QPort("out")]


class _QGraph:
    def node(self, _ct):
        return _QMeta()

    def widget_order(self, _ct):
        return []


def _place(specs):
    out = layout.assign_positions({"nodes": []}, _QGraph(), [dict(s) for s in specs])
    size = layout.estimate_size(1, 1, 0, title="N", input_labels=("in",), output_labels=("out",))
    return {s["as"]: {"pos": s["at"], "size": size} for s in out if s.get("op") == "add_node"}


def test_crossing_reduction_orders_columns_by_barycentre():
    """Three parallel chains authored in an order that crosses every wire.

    Pre-fix this produced 3 crossings, because a column stacked in ops-array order.
    """
    specs = [{"op": "add_node", "class_type": "N", "as": n} for n in ("a1", "b1", "c1", "a2", "b2", "c2")]
    specs += [
        {"op": "connect", "from": "a1.out", "to": "c2.in"},
        {"op": "connect", "from": "b1.out", "to": "b2.in"},
        {"op": "connect", "from": "c1.out", "to": "a2.in"},
    ]
    edges = [(s["from"].split(".")[0], s["to"].split(".")[0]) for s in specs if s["op"] == "connect"]
    crossings, _ = _score(_place(specs), edges)
    assert crossings == 0, f"barycentre ordering should remove all crossings, got {crossings}"


def test_coordinate_assignment_centres_a_node_on_its_inputs():
    """A node should sit level with what feeds it, not flush with the column top."""
    specs = [{"op": "add_node", "class_type": "N", "as": n} for n in ("a1", "b1", "c1", "a2", "b2", "c2")]
    specs += [
        {"op": "connect", "from": "a1.out", "to": "c2.in"},
        {"op": "connect", "from": "b1.out", "to": "b2.in"},
        {"op": "connect", "from": "c1.out", "to": "a2.in"},
    ]
    edges = [(s["from"].split(".")[0], s["to"].split(".")[0]) for s in specs if s["op"] == "connect"]
    _, align_dev = _score(_place(specs), edges)
    assert align_dev == 0.0, f"each target should be centred on its source; mean deviation {align_dev}px"


def test_pinned_siblings_are_obstacles_for_movable_nodes():
    """A new node pinned to an explicit `at` is as real as an existing one.

    collides() only ever compared movable nodes against EXISTING workflow nodes, so a
    movable node could be placed straight on top of a pinned sibling.
    """
    size = layout.estimate_size(1, 1, 0, title="N", input_labels=("in",), output_labels=("out",))
    specs = [
        {"op": "add_node", "class_type": "N", "as": "pinned", "at": [40.0, 60.0]},
        {"op": "add_node", "class_type": "N", "as": "free"},
    ]
    out = layout.assign_positions({"nodes": []}, _QGraph(), specs)
    at = {s["as"]: s["at"] for s in out}
    assert not layout._overlaps(layout.occupied(at["free"], size), layout.occupied(at["pinned"], size)), (
        "movable node was placed on top of a pinned sibling"
    )


# NOTE: the direct-jump collision change (replacing a ROW_GAP-at-a-time march that could
# run _GUARD * ROW_GAP = 40,000px) ships WITHOUT a red-green regression test. Every fixture
# tried either did not reach the collision branch or was cleared by the old march too, so
# the intended proof -- old code exhausts its budget and leaves an overlap -- was not
# demonstrated. It is a robustness and efficiency change, not a verified bug fix; treat it
# as unproven until someone builds a case that actually exhausts the guard.


# --- multiline widget height, solved from rendered geometry ---------------------------
#
# These numbers are not read off the frontend source, they are fitted to what a real
# browser drew for twelve core node classes. The fixture below is that measurement,
# recorded so the fit can be re-checked without a browser.

# (class, rendered height, link inputs, outputs, ordinary widgets, multiline widgets)
_RENDERED = [
    ("CLIPTextEncode", 200, 1, 1, 0, 1),
    ("KSampler", 262, 4, 1, 7, 0),
    ("EmptyLatentImage", 106, 0, 1, 3, 0),
    ("CheckpointLoaderSimple", 98, 0, 3, 1, 0),
    ("SaveImage", 58, 1, 1, 1, 0),
    ("LoadImage", 102, 0, 2, 2, 0),
    ("VAEDecode", 46, 2, 1, 0, 0),
    ("PreviewImage", 26, 1, 1, 0, 0),
    ("ConditioningCombine", 46, 2, 1, 0, 0),
    ("LatentUpscale", 130, 1, 1, 4, 0),
    ("CLIPSetLastLayer", 58, 1, 1, 1, 0),
    ("ImageScale", 130, 1, 1, 4, 0),
]

# The renderer's own base: 6px plus one 20px slot row per max(link_inputs, outputs).
# estimate_size deliberately runs taller (HEADER_H + PAD_H = 42 instead of 6) because
# over-spacing is invisible and under-spacing is the overlap users report. So the
# assertions below check the WIDGET term, which is the part that was wrong.
_RENDER_BASE = 6.0


@pytest.mark.parametrize("name,height,links,outputs,ordinary,multiline", _RENDERED)
def test_widget_block_matches_rendered_geometry(name, height, links, outputs, ordinary, multiline):
    """The widget term reproduces what the browser drew, for every measured class."""
    rendered_widget_block = height - _RENDER_BASE - layout.SLOT_H * max(links, outputs)
    assert layout._widgets_height(ordinary + multiline, multiline) == rendered_widget_block, name


def test_multiline_widget_is_not_charged_as_an_ordinary_row():
    """The bug this fixes.

    A multiline text box is a text AREA, not a widget ROW. Charging it the ordinary
    24px under-measures a CLIPTextEncode by 142px, which is the dominant term in the
    overlap the browser harness reproduces on the batched recording.
    """
    ordinary = layout.estimate_size(1, 1, 1)[1]
    multiline = layout.estimate_size(1, 1, 1, n_multiline=1)[1]
    assert multiline - ordinary == layout.MULTILINE_WIDGET_H - (layout.WIDGET_H + layout._WIDGET_ROW_GAP)
    assert multiline - ordinary == 142.0


def test_multiline_count_is_clamped_to_the_widget_count():
    """A node cannot have more multiline widgets than widgets.

    This previously asserted that `_widgets_height(1, 5)` charged FIVE multiline areas to a
    one-widget node, codifying a 700px over-measure as intended behaviour. It only arises
    from a bad catalog or a caller bug, and silently over-measuring hides both. Clamped.
    """
    assert layout._widgets_height(1, 5) == layout._widgets_height(1, 1)
    assert layout._widgets_height(3, -2) == layout._widgets_height(3, 0)


def test_count_multiline_matches_by_name_not_position():
    """Widget order is the render order, not the declaration order."""

    class _P:
        def __init__(self, name, multiline=False):
            self.name = name
            self.options = type("O", (), {"multiline": multiline})()

    class _M:
        inputs = [_P("clip"), _P("text", True), _P("seed")]

    assert layout.count_multiline(_M(), ("text", "seed")) == 1
    assert layout.count_multiline(_M(), ("seed",)) == 0


def test_count_multiline_tolerates_a_port_without_options():
    """Every test double here is such a port, and so is a catalog entry whose object_info
    omitted the options block."""

    class _Bare:
        def __init__(self, name):
            self.name = name

    class _M:
        inputs = [_Bare("text")]

    assert layout.count_multiline(_M(), ("text",)) == 0


def test_estimate_size_default_is_unchanged_without_multiline():
    """Additive: every existing caller keeps its old result."""
    assert layout.estimate_size(2, 1, 3) == layout.estimate_size(2, 1, 3, n_multiline=0)


# --- minimum rendered width, measured across 27 classes --------------------------------
#
# LiteGraph's own formula is NODE_WIDTH * (1.5 if widgets else 1.0) = 210, which no
# widget-bearing node actually renders at. Measured in a real browser: every node with a
# widget renders at 270 or wider, every node with a MULTILINE widget at 400 or wider.
# Fourteen of fourteen non-multiline widget classes sit at exactly 270 when their content
# is narrower, and seven of seven multiline classes at exactly 400 -- a floor, not a fixed
# width. Classes that exceed it on content (KSamplerAdvanced 312, ControlNetApply 317.9,
# CheckpointLoader 396.9) confirm the shape.

# (class, rendered width, has widgets, multiline count)
_RENDERED_WIDTHS = [
    ("CLIPTextEncode", 400, True, 1),
    ("CLIPTextEncodeSDXL", 400, True, 2),
    ("CLIPTextEncodeFlux", 400, True, 2),
    ("PrimitiveStringMultiline", 400, True, 1),
    ("KSampler", 270, True, 0),
    ("EmptyLatentImage", 270, True, 0),
    ("CheckpointLoaderSimple", 270, True, 0),
    ("SaveImage", 270, True, 0),
    ("LatentUpscale", 270, True, 0),
    ("PrimitiveString", 270, True, 0),
    ("UNETLoader", 270, True, 0),
    ("KSamplerAdvanced", 312, True, 0),
    ("ControlNetApply", 317.9, True, 0),
    ("CheckpointLoader", 396.9, True, 0),
    ("VAEDecode", 140, False, 0),
    ("VAEEncode", 140, False, 0),
    ("ConditioningCombine", 215, False, 0),
]


@pytest.mark.parametrize("name,width,has_widgets,multiline", _RENDERED_WIDTHS)
def test_min_width_never_exceeds_what_the_class_renders(name, width, has_widgets, multiline):
    """The floor must be a floor: never above the narrowest real node of its kind.

    A floor above the rendered width would over-space every node of that shape, which is
    the harmless direction but still wrong.
    """
    assert layout._min_width(has_widgets, multiline) <= width, name


def test_widget_nodes_floor_at_the_measured_270():
    assert layout._min_width(True, 0) == layout.WIDGET_MIN_WIDTH == 270.0


def test_multiline_nodes_floor_at_the_measured_400():
    assert layout._min_width(True, 1) == layout.MULTILINE_MIN_WIDTH == 400.0


def test_widgetless_nodes_keep_litegraphs_own_floor():
    """Only widget-bearing nodes get the raised floor; a Reroute must stay narrow."""
    assert layout._min_width(False, 0) == layout.LG_NODE_WIDTH == 140.0


def test_under_estimating_width_is_what_causes_overlap():
    """Direction check, and the reason this fix exists.

    Before the floors, a CLIPTextEncode was estimated at 250 against a rendered 400, so the
    placer put the next column 330px away and the real node reached 400 -- a 70px overlap
    that the model believed was a 80px gap. Over-estimating only wastes canvas.
    """
    w = layout.estimate_width("CLIP Text Encode (Prompt)", ("clip",), ("CONDITIONING",), ("text",), 1)
    assert w >= 400.0
