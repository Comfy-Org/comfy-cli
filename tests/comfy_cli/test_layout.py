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
