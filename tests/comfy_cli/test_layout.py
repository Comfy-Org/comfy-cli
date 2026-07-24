from comfy_cli import layout


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
