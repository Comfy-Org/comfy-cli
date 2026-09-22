"""The size PERSISTED onto a node, not the size the planner computed.

`assign_positions` sizes nodes to place them; `workflow_ops.add_node` sizes them again to
write `size` onto the node. Only the second one survives the call, and it is what every
LATER collision check reads. They diverged: the planner took a multiline term and the
persisted path did not, so a batch placed correctly and then the next call put a node on
top of one it had itself under-measured by 142px.

Every existing layout test exercised `assign_positions`, which is exactly why that got
through. These assert the saved artefact instead.
"""

from __future__ import annotations

from comfy_cli import layout
from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph

_OBJECT_INFO = {
    "CLIPTextEncode": {
        "input": {
            "required": {
                "text": ["STRING", {"multiline": True}],
                "clip": ["CLIP", {}],
            }
        },
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
        "name": "CLIPTextEncode",
        "display_name": "CLIP Text Encode (Prompt)",
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            }
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "name": "EmptyLatentImage",
        "display_name": "Empty Latent Image",
    },
    "LoadImage": {
        "input": {
            "required": {
                "image": [["fixture.png"], {"image_upload": True}],
            }
        },
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
        "name": "LoadImage",
        "display_name": "Load Image",
    },
    "SaveImage": {
        "input": {
            "required": {
                "images": ["IMAGE", {}],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        },
        "output": [],
        "output_name": [],
        "name": "SaveImage",
        "display_name": "Save Image",
    },
}


def _graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _saved_sizes(workflow: dict) -> dict:
    return {str(n.get("type")): n.get("size") for n in workflow.get("nodes", [])}


def test_persisted_size_carries_the_multiline_term():
    """The regression. A multiline node must be SAVED at its multiline height."""
    wf = {"nodes": [], "links": []}
    W.apply_specs(
        wf,
        _graph(),
        [{"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"}],
        actor="agent",
        base_version=1,
    )
    saved = _saved_sizes(wf)["CLIPTextEncode"]
    expected = layout.estimate_size(
        1,
        1,
        1,
        n_multiline=1,
        title="CLIP Text Encode (Prompt)",
        input_labels=("clip",),
        output_labels=("CONDITIONING",),
        widget_labels=("text",),
    )
    assert list(saved) == list(expected), "saved size dropped the multiline term"


def test_persisted_size_matches_what_the_planner_used():
    """Planner and persisted path must agree, which is the invariant that broke.

    Asserting equality rather than a literal keeps this honest if the constants move: the
    bug was never a wrong number, it was two callers disagreeing.
    """
    wf = {"nodes": [], "links": []}
    g = _graph()
    W.apply_specs(
        wf,
        g,
        [
            {"op": "add_node", "class_type": "CLIPTextEncode", "as": "pos"},
            {"op": "add_node", "class_type": "CLIPTextEncode", "as": "neg"},
        ],
        actor="agent",
        base_version=1,
    )
    sizes = [n["size"] for n in wf["nodes"] if n.get("type") == "CLIPTextEncode"]
    assert len(sizes) == 2
    meta = g.node("CLIPTextEncode")
    planner = layout.estimate_size(
        len([p for p in meta.inputs if p.is_link]),
        len(meta.outputs),
        len(tuple(g.widget_order_default("CLIPTextEncode"))),
        n_multiline=layout.count_multiline(meta, tuple(g.widget_order_default("CLIPTextEncode"))),
        title=meta.display_name,
        input_labels=tuple(p.name for p in meta.inputs if p.is_link),
        output_labels=tuple(p.name for p in meta.outputs),
        widget_labels=tuple(g.widget_order_default("CLIPTextEncode")),
    )
    for s in sizes:
        assert list(s) == list(planner)


# NOT HERE: the reviewer's two-batches-then-one-more overlap repro. On this branch the
# multiline HEIGHT is fixed but the width is still text-derived (249.9 for CLIPTextEncode),
# so the third node lands clear and the case cannot go red. The 400px width floor that
# makes it overlap is in the next PR, which is where that test belongs and where the
# reviewer reproduced it. Asserting it here would pass for the wrong reason.


def test_non_multiline_nodes_are_unaffected():
    """Additive: a node with no multiline widget saves exactly what it did before."""
    wf = {"nodes": [], "links": []}
    W.apply_specs(
        wf,
        _graph(),
        [{"op": "add_node", "class_type": "EmptyLatentImage", "as": "latent"}],
        actor="agent",
        base_version=1,
    )
    saved = _saved_sizes(wf)["EmptyLatentImage"]
    expected = layout.estimate_size(
        0,
        1,
        3,
        n_multiline=0,
        title="Empty Latent Image",
        input_labels=(),
        output_labels=("LATENT",),
        widget_labels=("width", "height", "batch_size"),
    )
    assert list(saved) == list(expected)


def test_a_second_call_does_not_overlap_the_first_batch():
    """The reviewer's repro, and it only reproduces once width is floored too.

    Two multiline nodes in one batch, then one more in a second call. The second call reads
    the SAVED sizes; under-measure those and it places the new node inside the first.

    This belongs on this branch rather than the height PR: with a text-derived width
    (249.9) the third node lands at x=369.9 and clears, so the case passes for the wrong
    reason. At the 400px floor the first node truly spans 40..440 and the overlap is real.

    Measured against the size the node TRULY renders at, not the size that was saved.
    Scoring against the saved size lets the bug hide: an under-measured node "does not
    overlap" precisely because the record of it is too small.
    """
    wf = {"nodes": [], "links": []}
    g = _graph()
    W.apply_specs(
        wf,
        g,
        [
            {"op": "add_node", "class_type": "CLIPTextEncode", "as": "a"},
            {"op": "add_node", "class_type": "CLIPTextEncode", "as": "b"},
        ],
        actor="agent",
        base_version=1,
    )
    W.apply_specs(
        wf,
        g,
        [{"op": "add_node", "class_type": "CLIPTextEncode", "as": "c"}],
        actor="agent",
        base_version=2,
    )

    meta = g.node("CLIPTextEncode")
    widgets = tuple(g.widget_order_default("CLIPTextEncode"))
    true_size = layout.estimate_size(
        len([p for p in meta.inputs if p.is_link]),
        len(meta.outputs),
        len(widgets),
        n_multiline=layout.count_multiline(meta, widgets),
        title=meta.display_name,
        input_labels=tuple(p.name for p in meta.inputs if p.is_link),
        output_labels=tuple(p.name for p in meta.outputs),
        widget_labels=widgets,
    )
    assert true_size[0] >= layout.MULTILINE_MIN_WIDTH, (
        "this test is only meaningful once the multiline width floor applies"
    )

    rects = [layout.occupied(n["pos"], true_size) for n in wf["nodes"]]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not layout._overlaps(rects[i], rects[j]), (
                f"node {i} at {wf['nodes'][i]['pos']} overlaps node {j} at {wf['nodes'][j]['pos']}"
            )


def test_populated_image_loaders_reserve_the_frontend_preview_height():
    """Five loader→save rows stay disjoint after image previews materialize.

    Regression for PM-1607: object_info exposes the image combo and upload
    button, but not the 190px DOM preview host the frontend adds after loading
    the selected image. The old row pitch therefore looked aligned while every
    loader overlapped the one below it once its preview appeared.
    """
    wf = {"nodes": [], "links": []}
    specs = []
    for i in range(5):
        specs.extend(
            [
                {"op": "add_node", "class_type": "LoadImage", "as": f"load{i}"},
                {"op": "add_node", "class_type": "SaveImage", "as": f"save{i}"},
                {"op": "connect", "from": f"$load{i}.IMAGE", "to": f"$save{i}.images"},
            ]
        )

    W.apply_specs(wf, _graph(), specs, actor="agent", base_version=1)

    loaders = [n for n in wf["nodes"] if n.get("type") == "LoadImage"]
    assert len(loaders) == 5
    assert all(n["size"][1] >= layout.IMAGE_PREVIEW_MIN_H for n in loaders)
    rects = [layout.occupied(n["pos"], n["size"]) for n in loaders]
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not layout._overlaps(rects[i], rects[j]), (
                f"loader {i} at {loaders[i]['pos']} overlaps loader {j} at {loaders[j]['pos']}"
            )
