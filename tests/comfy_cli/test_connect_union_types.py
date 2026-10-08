"""A connect must be allowed when the destination input accepts a UNION of types
that contains the source type.

ComfyUI expresses a multi-type input as a comma-separated union
("MESH,FILE_3D_GLB,FILE_3D_GLTF,..."). The type gate compared slot types as
whole strings, so a FILE_3D_GLB output was refused by an input that explicitly
accepts FILE_3D_GLB.

connect/apply_ops could fail with errors of this shape, e.g.

  type mismatch: FILE_3D_GLB output of node 4 cannot connect to
  MESH,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_FBX,FILE_3D_STL,FILE_3D_USDZ,
  ... input 'mesh' of node 7

  type mismatch: FILE_3D output of node 3 cannot connect to
  FILE_3D_GLB,FILE_3D_FBX,FILE_3D_OBJ,FILE_3D_STL,FILE_3D input 'model_3d' of ...

Both name the source type inside the accepted list, so the agent reads the hint,
sees its own type listed, and retries the identical edit.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph

MESH_UNION = (
    "MESH,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_FBX,FILE_3D_STL,FILE_3D_USDZ,FILE_3D_PLY,FILE_3D_SPLAT,FILE_3D"
)
MODEL3D_UNION = "FILE_3D_GLB,FILE_3D_FBX,FILE_3D_OBJ,FILE_3D_STL,FILE_3D"


def _wf(out_type: str, in_type: str) -> dict:
    return {
        "last_node_id": 2,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "Load3D",
                "pos": [0, 0],
                "inputs": [],
                "outputs": [{"name": "OUT", "type": out_type, "links": []}],
            },
            {
                "id": 2,
                "type": "Import3D",
                "pos": [300, 0],
                "inputs": [{"name": "slot", "type": in_type, "link": None}],
                "outputs": [],
            },
        ],
        "links": [],
    }


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info({})


@pytest.mark.parametrize(
    "out_type,in_type",
    [
        ("FILE_3D_GLB", MESH_UNION),  # e.g. Load3D GLB -> mesh
        ("FILE_3D", MODEL3D_UNION),  # e.g. Load3D FILE_3D -> model_3d
        ("MESH", MESH_UNION),  # first member of the union
        ("FILE_3D", MESH_UNION),  # last member of the union
    ],
)
def test_connect_accepts_a_member_of_a_union_input(graph, out_type, in_type):
    wf = _wf(out_type, in_type)
    wf, op = workflow_ops.connect(wf, graph, 1, "OUT", 2, "slot")
    assert op["op"] == "connect"
    assert wf["nodes"][1]["inputs"][0]["link"] is not None, "the link must be wired"


def test_connect_still_rejects_a_type_outside_the_union(graph):
    """The gate must keep its teeth: a genuine mis-wire is still refused."""
    wf = _wf("IMAGE", MESH_UNION)
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, 1, "OUT", 2, "slot")


def test_connect_union_matching_is_not_substring_based(graph):
    """FILE_3D_GLB must not satisfy an input accepting only FILE_3D_GLTF — a
    naive `in` check on the joined string would wrongly allow it."""
    wf = _wf("FILE_3D_GL", "FILE_3D_GLTF,FILE_3D_GLB")
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, 1, "OUT", 2, "slot")


@pytest.mark.parametrize(
    "out_type,in_type",
    [
        ("IMAGE", "COMFY_MATCHTYPE_V3"),  # e.g. LoadImage -> ResizeImageMaskNode.input
        ("MASK", "COMFY_MATCHTYPE_V3"),
        ("COMFY_MATCHTYPE_V3", "IMAGE"),  # and back out: .resized -> PreviewImage.images
        ("COMFY_MATCHTYPE_V3", MESH_UNION),
    ],
)
def test_connect_accepts_a_matchtype_slot_on_either_end(graph, out_type, in_type):
    """`COMFY_MATCHTYPE_V3` is a wildcard that takes the type of whatever is
    wired to it — the validator has always read it that way
    (`_WILDCARD_TYPE_PREFIX`), but the connect gate had its own type test that
    knew only `*`. Both directions were refused, so no node using a match-type
    socket could be wired at all: an end-to-end run against the live agent
    could not build a ResizeImageMaskNode graph by any route.
    """
    wf = _wf(out_type, in_type)
    wf, op = workflow_ops.connect(wf, graph, 1, "OUT", 2, "slot")
    assert op["op"] == "connect"
    assert wf["nodes"][1]["inputs"][0]["link"] is not None, "the link must be wired"
