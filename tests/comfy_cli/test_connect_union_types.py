"""A connect must be allowed when the destination input accepts a UNION of types
that contains the source type.

ComfyUI expresses a multi-type input as a comma-separated union
("MESH,FILE_3D_GLB,FILE_3D_GLTF,..."). The type gate compared slot types as
whole strings, so a FILE_3D_GLB output was refused by an input that explicitly
accepts FILE_3D_GLB.

Measured on prod comfy-agent traces (2026-07-23 → 07-28): ~23 connect/apply_ops
failures of this exact shape, e.g.

  type mismatch: FILE_3D_GLB output of node 4318783979958460 cannot connect to
  MESH,FILE_3D_GLB,FILE_3D_GLTF,FILE_3D_OBJ,FILE_3D_FBX,FILE_3D_STL,FILE_3D_USDZ,
  ... input 'mesh' of node 2451178264782280

  type mismatch: FILE_3D output of node 890530584279986 cannot connect to
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
        ("FILE_3D_GLB", MESH_UNION),  # prod: Load3D GLB -> mesh
        ("FILE_3D", MODEL3D_UNION),  # prod: Load3D FILE_3D -> model_3d
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
