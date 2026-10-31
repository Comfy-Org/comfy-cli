"""Agent-built nodes carry a socket entry for every scalar widget input.

Reported: the agent said ``frame_load_cap`` on an agent-built Load Video (VHS)
and an agent-made custom node was a plain widget = 1000, while the user had
wired a PrimitiveInt into both on the canvas. ``_build_node`` wrote only LINK
inputs, so the shared document had no ``frame_load_cap`` slot. The frontend
gives every non-socketless widget input a socket, so a human link drawn into
it named an input the document did not hold and never reached it.

A node the frontend creates serializes those entries
(``{"name", "type", "widget": {"name"}, "link": None}``), so an agent-built
node now does too. A link into such an input is then a CONCRETE connect: one
last-writer-wins register per input, instead of an autogrow-style ``grow``
that mints a fresh slot per link and leaves two slots when two writers race.
"""

from __future__ import annotations

import copy

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_print import render_py
from comfy_cli.workflow_to_api import convert_ui_to_api


def _node(required: dict, outputs: list[str], optional: dict | None = None, names: list[str] | None = None) -> dict:
    spec = {"required": required}
    order = {"required": list(required)}
    if optional:
        spec["optional"] = optional
        order["optional"] = list(optional)
    return {
        "input": spec,
        "input_order": order,
        "output": outputs,
        "output_name": names or outputs,
        "category": "test",
        "display_name": "x",
        "python_module": "custom_nodes.test",
    }


OBJECT_INFO = {
    "PrimitiveInt": _node({"value": ["INT", {"default": 0}]}, ["INT"]),
    "VHS_LoadVideo": _node(
        {
            "video": [["clip.mp4"]],
            "force_rate": ["FLOAT", {"default": 0}],
            "frame_load_cap": ["INT", {"default": 0}],
            "skip_first_frames": ["INT", {"default": 0}],
        },
        ["IMAGE", "INT"],
        optional={"meta_batch": ["VHS_BatchManager"], "vae": ["VAE"]},
        names=["IMAGE", "frame_count"],
    ),
    "CSVCVColumn": _node(
        {
            "csv_path": ["STRING", {"default": ""}],
            "enabled": ["BOOLEAN", {"default": True}],
            "frame_load_cap": ["INT", {"default": 0}],
        },
        ["FLOAT"],
    ),
}

LOAD_VIDEO_LINK_INPUTS = [("meta_batch", "VHS_BatchManager"), ("vae", "VAE")]


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(OBJECT_INFO)


def _add(workflow: dict, graph: Graph, class_type: str) -> dict:
    workflow, op = workflow_ops.add_node(workflow, graph, class_type, pos=[0, 0])
    return next(n for n in workflow["nodes"] if n["id"] == op["node_id"])


def _input(node: dict, name: str) -> list[dict]:
    return [i for i in node["inputs"] if i.get("name") == name]


def _built(graph: Graph) -> tuple[dict, dict, dict]:
    workflow: dict = {"nodes": [], "links": []}
    primitive = _add(workflow, graph, "PrimitiveInt")
    video = _add(workflow, graph, "VHS_LoadVideo")
    return workflow, primitive, video


def test_add_node_writes_a_widget_socket_for_each_scalar_widget_input(graph):
    workflow: dict = {"nodes": [], "links": []}
    video = _add(workflow, graph, "VHS_LoadVideo")
    csv = _add(workflow, graph, "CSVCVColumn")

    assert [(i["name"], i["type"]) for i in video["inputs"][:2]] == LOAD_VIDEO_LINK_INPUTS
    for name, slot_type in [("force_rate", "FLOAT"), ("frame_load_cap", "INT"), ("skip_first_frames", "INT")]:
        assert _input(video, name) == [{"name": name, "type": slot_type, "widget": {"name": name}, "link": None}]
    assert _input(video, "video") == []
    for name, slot_type in [("csv_path", "STRING"), ("enabled", "BOOLEAN"), ("frame_load_cap", "INT")]:
        assert _input(csv, name) == [{"name": name, "type": slot_type, "widget": {"name": name}, "link": None}]


def test_connect_into_a_widget_input_is_a_concrete_connect(graph):
    workflow, primitive, video = _built(graph)

    workflow, op = workflow_ops.connect(workflow, graph, primitive["id"], 0, video["id"], "frame_load_cap")

    assert "grow" not in op
    assert video["inputs"][op["to_slot"]]["name"] == "frame_load_cap"
    assert _input(video, "frame_load_cap") == [
        {"name": "frame_load_cap", "type": "INT", "widget": {"name": "frame_load_cap"}, "link": op["link_id"]}
    ]


def test_two_racing_links_into_one_widget_input_converge_on_one_slot(graph):
    base, primitive, video = _built(graph)
    second = _add(base, graph, "PrimitiveInt")
    _, first_op = workflow_ops.connect(copy.deepcopy(base), graph, primitive["id"], 0, video["id"], "frame_load_cap")
    _, second_op = workflow_ops.connect(
        copy.deepcopy(base), graph, second["id"], 0, video["id"], "frame_load_cap", base_version=1
    )

    converged = []
    for order in ([first_op, second_op], [second_op, first_op]):
        replica = copy.deepcopy(base)
        for op in order:
            workflow_ops.apply_op(replica, op, graph)
        node = next(n for n in replica["nodes"] if n["id"] == video["id"])
        converged.append(_input(node, "frame_load_cap"))

    assert converged[0] == converged[1]
    assert [slot["link"] for slot in converged[0]] == [second_op["link_id"]]


def test_api_prompt_reads_the_widget_value_until_linked_then_the_link(graph):
    workflow, primitive, video = _built(graph)
    workflow_ops.set_widget(workflow, graph, video["id"], "frame_load_cap", 1000)

    unlinked = convert_ui_to_api(copy.deepcopy(workflow), OBJECT_INFO)
    assert unlinked[str(video["id"])]["inputs"]["frame_load_cap"] == 1000

    workflow_ops.connect(workflow, graph, primitive["id"], 0, video["id"], "frame_load_cap")
    linked = convert_ui_to_api(workflow, OBJECT_INFO)
    assert linked[str(video["id"])]["inputs"]["frame_load_cap"] == [str(primitive["id"]), 0]


def test_the_agent_print_shows_a_human_link_into_frame_load_cap(graph):
    workflow, primitive, video = _built(graph)
    slot = next(k for k, i in enumerate(video["inputs"]) if i["name"] == "frame_load_cap")
    link_id = 501
    video["inputs"][slot]["link"] = link_id
    primitive["outputs"][0]["links"] = [link_id]
    workflow["links"].append([link_id, primitive["id"], 0, video["id"], slot, "INT"])

    source = render_py(workflow, graph).source

    assert "frame_load_cap=primitive_int" in source
    assert "frame_load_cap=0" not in source


def test_a_widget_socket_refuses_a_link_of_another_scalar_type(graph):
    workflow, primitive, video = _built(graph)

    with pytest.raises(ValueError, match=r"type mismatch: INT output .* FLOAT input 'force_rate'"):
        workflow_ops.connect(workflow, graph, primitive["id"], 0, video["id"], "force_rate")
