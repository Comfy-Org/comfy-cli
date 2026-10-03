"""`workflow connect` between two nodes INSIDE one subgraph definition.

Prod trace f8d27ae4 (SAM3, 357 nodes): three interior links of subgraph 70
pointed at an input slot their SAM3 nodes no longer had, so the RegexExtract
prompts fed nothing and the text prompts sat empty. The repair is a re-wire
inside the definition (`70/2005.0` -> `70/2011.text_prompt`), but connect
refused every interior address ("a link cannot cross a subgraph boundary"),
so the agent hardcoded the prompt text instead.

A link between two interior nodes of the same definition crosses no
boundary. connect now wires it and emits a `connect` op carrying the
instance `path`, the shape the doc host's applier (comfy-multi-player
applyInteriorConnect) already accepts. Crossing a boundary is still refused.
"""

from __future__ import annotations

import copy

import pytest
from test_workflow_edit import (  # type: ignore[import-not-found]
    _object_info,
    _run,
    _write,
    reset_singleton,  # noqa: F401  (autouse fixture)
)

from comfy_cli import workflow_ops
from comfy_cli.command import workflow_edit
from comfy_cli.cql.engine import Graph

SG = "5a1c362b-0000-4000-8000-000000000070"


def _graph() -> Graph:
    info = copy.deepcopy(_object_info())
    info["StringSource"] = {
        "input": {"required": {"value": ["STRING", {"default": ""}]}},
        "input_order": {"required": ["value"]},
        "output": ["STRING"],
        "output_name": ["STRING"],
        "category": "utils",
        "display_name": "String Source",
        "output_node": False,
        "python_module": "nodes",
    }
    return Graph.from_object_info(info)


@pytest.fixture(autouse=True)
def patched_graph(monkeypatch):
    monkeypatch.setattr(workflow_edit, "_get_graph", lambda *a, **kw: _graph())


def _workflow(instances: int = 1) -> dict:
    nodes = [{"id": 70 + i, "type": SG, "pos": [0, 0], "inputs": [], "outputs": []} for i in range(instances)]
    return {
        "last_node_id": 80,
        "last_link_id": 0,
        "nodes": nodes,
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": SG,
                    "name": "Region prompts",
                    "inputs": [],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 2005,
                            "type": "StringSource",
                            "inputs": [{"name": "value", "type": "STRING", "widget": {"name": "value"}, "link": None}],
                            "outputs": [{"name": "STRING", "type": "STRING", "links": [9001]}],
                            "widgets_values": ["a red car"],
                        },
                        {
                            "id": 2011,
                            "type": "CLIPTextEncode",
                            "inputs": [
                                {"name": "clip", "type": "CLIP", "link": None},
                                {"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None},
                            ],
                            "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
                            "widgets_values": [""],
                        },
                        {
                            "id": 2012,
                            "type": "CheckpointLoaderSimple",
                            "inputs": [],
                            "outputs": [
                                {"name": "MODEL", "type": "MODEL", "links": []},
                                {"name": "CLIP", "type": "CLIP", "links": []},
                                {"name": "VAE", "type": "VAE", "links": []},
                            ],
                            "widgets_values": ["a.safetensors"],
                        },
                    ],
                    # The stale row: aimed at input slot 6, which 2011 does not have.
                    "links": [
                        {
                            "id": 9001,
                            "origin_id": 2005,
                            "origin_slot": 0,
                            "target_id": 2011,
                            "target_slot": 6,
                            "type": "STRING",
                        }
                    ],
                }
            ]
        },
    }


def _sg(wf: dict) -> dict:
    return wf["definitions"]["subgraphs"][0]


def _node(wf: dict, nid: int) -> dict:
    return next(n for n in _sg(wf)["nodes"] if n["id"] == nid)


def test_connect_two_interior_nodes_wires_inside_the_definition(tmp_path, capsys):
    path = _write(tmp_path, _workflow())
    env = _run(["connect", str(path), "70/2005.STRING", "70/2011.text"], capsys)
    assert env["ok"] is True, env
    op = env["data"]["op"]
    assert op["op"] == "connect"
    assert op["path"] == ["70"]
    assert (str(op["from_node"]), op["from_slot"], str(op["to_node"]), op["to_slot"]) == ("2005", 0, "2011", 1)
    assert op["link_type"] == "STRING"
    import json

    wf = json.loads(path.read_text())
    link_id = op["link_id"]
    assert _node(wf, 2011)["inputs"][1]["link"] == link_id
    assert link_id in _node(wf, 2005)["outputs"][0]["links"]
    row = next(lk for lk in _sg(wf)["links"] if lk["id"] == link_id)
    assert (row["origin_id"], row["origin_slot"], row["target_id"], row["target_slot"]) == (2005, 0, 2011, 1)
    # Top level untouched.
    assert wf["links"] == []


def test_interior_connect_into_a_link_input_by_slot_name(tmp_path, capsys):
    path = _write(tmp_path, _workflow())
    env = _run(["connect", str(path), "70/2012.CLIP", "70/2011.clip"], capsys)
    assert env["ok"] is True, env
    assert env["data"]["op"]["to_slot"] == 0


def test_interior_connect_replaces_the_incumbent_link(tmp_path, capsys):
    import json

    path = _write(tmp_path, _workflow())
    first = _run(["connect", str(path), "70/2005.STRING", "70/2011.text", "--base-version", "1"], capsys)
    second = _run(["connect", str(path), "70/2005.STRING", "70/2011.text", "--base-version", "2"], capsys)
    first_id, second_id = first["data"]["op"]["link_id"], second["data"]["op"]["link_id"]
    wf = json.loads(path.read_text())
    ids = [lk["id"] for lk in _sg(wf)["links"]]
    assert second_id in ids and first_id not in ids
    assert _node(wf, 2011)["inputs"][1]["link"] == second_id
    assert first_id not in _node(wf, 2005)["outputs"][0]["links"]


def test_interior_connect_type_mismatch_is_refused(tmp_path, capsys):
    path = _write(tmp_path, _workflow())
    env = _run(["connect", str(path), "70/2012.MODEL", "70/2011.clip"], capsys)
    assert env["ok"] is False
    assert "type mismatch" in env["error"]["message"]


def test_crossing_the_boundary_is_still_refused(tmp_path, capsys):
    wf = _workflow()
    wf["nodes"].append(
        {"id": 9, "type": "EmptyLatentImage", "inputs": [], "outputs": [{"name": "LATENT", "type": "LATENT"}]}
    )
    path = _write(tmp_path, wf)
    env = _run(["connect", str(path), "9.LATENT", "70/2011.text"], capsys)
    assert env["ok"] is False
    assert "inside subgraph 70" in env["error"]["message"]


def test_a_shared_definition_is_refused(tmp_path, capsys):
    path = _write(tmp_path, _workflow(instances=2))
    env = _run(["connect", str(path), "70/2005.STRING", "70/2011.text"], capsys)
    assert env["ok"] is False
    assert "2 instances" in env["error"]["message"]


def test_the_op_replays_to_the_same_document():
    wf = _workflow()
    graph = _graph()
    out, op = workflow_ops.connect(copy.deepcopy(wf), graph, "70/2005", "STRING", "70/2011", "text")
    replayed = workflow_ops.apply_op(copy.deepcopy(wf), op, graph)
    assert replayed["definitions"] == out["definitions"]
    assert workflow_ops._write_target(op) == ("input", ("70",), "2011", 1)


def test_print_before_and_after_the_rewire():
    """Before: print names the stale row and the exact re-wire. After: the
    interior input renders from its source and the row is called a harmless
    leftover, so a caller is not sent round the loop again."""
    from comfy_cli.workflow_print import render_py

    wf = _workflow()
    graph = _graph()
    before = render_py(copy.deepcopy(wf), graph)
    (stale,) = [w for w in before.warnings if "link 9001" in w]
    assert "`connect 70/2005.0 70/2011.<input>`" in stale
    out, _op = workflow_ops.connect(wf, graph, "70/2005", "STRING", "70/2011", "text")
    after = render_py(out, graph)
    (leftover,) = [w for w in after.warnings if "link 9001" in w]
    assert "nothing needs re-wiring" in leftover and "'text'" in leftover
    assert "text=string_source" in after.source


def test_replacing_a_boundary_link_drops_it_from_the_boundary_input():
    wf = _workflow()
    sg = _sg(wf)
    sg["inputs"] = [{"id": "in-1", "name": "prompt", "type": "STRING", "linkIds": [8001, 8002]}]
    sg["links"].append(
        {"id": 8001, "origin_id": -10, "origin_slot": 0, "target_id": 2011, "target_slot": 1, "type": "STRING"}
    )
    _node(wf, 2011)["inputs"][1]["link"] = 8001
    out, op = workflow_ops.connect(wf, _graph(), "70/2005", "STRING", "70/2011", "text")
    assert _sg(out)["inputs"][0]["linkIds"] == [8002]
    assert 8001 not in [lk["id"] for lk in _sg(out)["links"]]


OUTER = "5a1c362b-0000-4000-8000-0000000000aa"


def _nested_workflow(outer_instances: int) -> dict:
    """``outer_instances`` top-level instances of OUTER, whose definition holds
    ONE instance (node 300) of the inner definition SG. Each OUTER instance
    therefore carries its own live SG instance, all sharing SG's one body."""
    wf = _workflow(instances=0)
    wf["nodes"] = [
        {"id": 80 + i, "type": OUTER, "pos": [0, 0], "inputs": [], "outputs": []} for i in range(outer_instances)
    ]
    wf["definitions"]["subgraphs"].append(
        {
            "id": OUTER,
            "name": "Outer",
            "inputs": [],
            "outputs": [],
            "nodes": [{"id": 300, "type": SG, "pos": [0, 0], "inputs": [], "outputs": []}],
            "links": [],
        }
    )
    return wf


def test_a_definition_shared_through_its_ancestor_is_refused(tmp_path, capsys):
    # Two OUTER instances each hold one SG instance: SG has 2 live instances,
    # though its body appears once in OUTER's definition. Wiring inside it
    # would rewire both, so connect must refuse as for a flat shared def.
    path = _write(tmp_path, _nested_workflow(outer_instances=2))
    before = path.read_text()
    env = _run(["connect", str(path), "80/300/2005.STRING", "80/300/2011.text"], capsys)
    assert env["ok"] is False, env
    assert "2 instances" in env["error"]["message"]
    assert path.read_text() == before


def test_a_nested_single_instance_definition_is_wired(tmp_path, capsys):
    path = _write(tmp_path, _nested_workflow(outer_instances=1))
    env = _run(["connect", str(path), "80/300/2005.STRING", "80/300/2011.text"], capsys)
    assert env["ok"] is True, env
    assert env["data"]["op"]["path"] == ["80", "300"]


def test_instance_count_multiplies_through_ancestors():
    assert workflow_ops._definition_instance_count(_nested_workflow(3), SG) == 3
    assert workflow_ops._definition_instance_count(_nested_workflow(3), OUTER) == 3


def test_instance_count_terminates_on_a_self_instantiating_definition():
    wf = _workflow(instances=1)
    _sg(wf)["nodes"].append({"id": 2099, "type": SG, "pos": [0, 0], "inputs": [], "outputs": []})
    # A definition that names itself cannot be expanded; it must not hang and
    # must report it as shared (more than one live instance).
    assert workflow_ops._definition_instance_count(wf, SG) > 1
