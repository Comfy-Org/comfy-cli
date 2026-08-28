"""``set-widget`` / ``connect`` on promoted subgraph widgets (the "agent editing subgraph" report, scenarios 1 & 2).

Scenario 1 — *"the agent edited the widget under layer 2, not on the surface;
the value on the surface overwrites it and the user never sees it."* The
frontend serializes a promoted widget's value on the HOST instance
(ADR 0009); ``set-widget 57.width`` used to follow ``proxyWidgets`` into the
interior node, a value the frontend neither runs nor displays. The write must
land on the host — and when an outside node feeds the promoted input
(*"users even have number or text boxes connected to the promoted widget"*),
on that node's widget, because the link is what the graph runs.

Scenario 2 — *"wire something outside to the promoted widget, e.g. Int node to
width — the agent cannot do that yet."* A promoted widget IS a subgraph input,
so ``connect`` materializes it on the instance and wires it.

The rule, in one line: never edit a subgraph's interior for a promoted value —
edit what it is linked to.

Fixtures: verbatim gallery templates (``fixtures/gallery``) plus source-node
shapes copied from the wild (``PrimitiveNode`` from
``templates-multiple_consistent_shots-nb_pro.json``, ``Reroute`` from
``template_contact_sheet-step_3.app.json``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_GALLERY = _FIXTURES / "gallery"
OBJECT_INFO = _FIXTURES / "object_info_subgraph_promoted.json"

Z_IMAGE_HOST_DEFAULTS = [
    "Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up. ",
    1024,
    1024,
    0,
    8,
    "z_image_turbo_bf16.safetensors",
    "qwen_3_4b.safetensors",
    "ae.safetensors",
]


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(json.loads(OBJECT_INFO.read_text()))


def _load(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


def _node(wf: dict, node_id: Any) -> dict:
    return next(n for n in wf["nodes"] if str(n["id"]) == str(node_id))


def _interior(wf: dict, instance_id: Any, inner_id: Any) -> dict:
    inst = _node(wf, instance_id)
    sg = next(d for d in wf["definitions"]["subgraphs"] if d["id"] == inst["type"])
    return next(n for n in sg["nodes"] if str(n["id"]) == str(inner_id))


def _input(node: dict, name: str) -> dict:
    return next(i for i in node.get("inputs") or [] if i["name"] == name)


# --------------------------------------------------------------------------- #
# Scenario 1: the write lands on the host, not the interior
# --------------------------------------------------------------------------- #


def test_promoted_address_writes_the_host_value(graph):
    wf = _load("image_z_image_turbo.json")
    wf, op = workflow_ops.set_widget(wf, graph, 57, "width", 768)
    expected = list(Z_IMAGE_HOST_DEFAULTS)
    expected[1] = 768
    assert _node(wf, 57)["widgets_values"] == expected
    assert _interior(wf, 57, 13)["widgets_values"] == [1024, 1024, 1]
    assert op["node_id"] == 57
    assert op["widget"] == "width"
    assert op["old"] == 1024
    assert "path" not in op
    assert op["promoted"]["value_index"] == 1
    assert op["promoted"]["host_widgets_values"] == expected


def test_interior_address_of_a_promoted_widget_is_redirected_to_the_host(graph):
    wf = _load("image_z_image_turbo.json")
    wf, op = workflow_ops.set_widget(wf, graph, "57/13", "width", 768)
    assert _node(wf, 57)["widgets_values"][1] == 768
    assert _interior(wf, 57, 13)["widgets_values"][0] == 1024
    assert op["node_id"] == 57
    assert op["widget"] == "width"
    assert op["redirected_from"] == "57/13.width"


def test_unpromoted_interior_widget_still_writes_the_interior(graph):
    wf = _load("image_z_image_turbo.json")
    wf, op = workflow_ops.set_widget(wf, graph, "57/3", "cfg", 2.0)
    assert op["path"] == ["57", "3"]
    assert _interior(wf, 57, 3)["widgets_values"][3] == 2.0
    assert _node(wf, 57)["widgets_values"] == []


def test_post_migration_host_write_touches_one_slot(graph):
    wf = _load("audio_minimax_music_3.json")
    before = list(_node(wf, 37)["widgets_values"])
    wf, op = workflow_ops.set_widget(wf, graph, 37, "max_duration", 90)
    after = _node(wf, 37)["widgets_values"]
    assert after[2] == 90 and after[:2] == before[:2] and after[3:] == before[3:]
    assert op["old"] == 60
    assert _interior(wf, 37, 13)["widgets_values"][2] == 222  # interior default untouched


def test_socket_only_promoted_input_is_not_a_widget(graph):
    wf = _load("api_seedance2_5_video_extend.json")
    with pytest.raises(ValueError, match="link input"):
        workflow_ops.set_widget(wf, graph, 39, "clip_to_resize", "x")


def test_host_write_replays_idempotently_and_targets_one_register(graph):
    wf = _load("image_z_image_turbo.json")
    base = copy.deepcopy(wf)
    wf, op_host = workflow_ops.set_widget(wf, graph, 57, "width", 768)
    _, op_interior = workflow_ops.set_widget(copy.deepcopy(base), graph, "57/13", "width", 512)
    assert workflow_ops._write_target(op_host) == workflow_ops._write_target(op_interior)
    replayed = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), op_host, graph), op_host, graph)
    assert _node(replayed, 57)["widgets_values"][1] == 768


# --------------------------------------------------------------------------- #
# Scenario 2: wire an outside node onto the promoted widget
# --------------------------------------------------------------------------- #


def _with_primitive(graph, value: int = 640):
    wf = _load("image_z_image_turbo.json")
    wf, prim = workflow_ops.add_node(wf, graph, "PrimitiveInt")
    wf, _ = workflow_ops.set_widget(wf, graph, prim["node_id"], "value", value)
    return wf, prim["node_id"]


def test_connect_materializes_the_promoted_input_and_wires_it(graph):
    wf, prim = _with_primitive(graph)
    wf, op = workflow_ops.connect(wf, graph, prim, "INT", 57, "width")
    width = _input(_node(wf, 57), "width")
    assert width["type"] == "INT"
    assert width["widget"] == {"name": "width"}
    assert width["link"] == op["link_id"]
    assert [i["name"] for i in _node(wf, 57)["inputs"]] == ["text", "width"]
    link = next(link for link in wf["links"] if link[0] == op["link_id"])
    assert link[1] == prim and link[3] == 57


def test_connect_type_mismatch_into_a_promoted_input_is_refused(graph):
    wf = _load("image_z_image_turbo.json")
    wf, text = workflow_ops.add_node(wf, graph, "PrimitiveStringMultiline")
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, text["node_id"], "STRING", 57, "width")


def test_connect_to_an_already_materialized_promoted_input_reuses_it(graph):
    wf, prim = _with_primitive(graph)
    wf, first = workflow_ops.connect(wf, graph, prim, "INT", 57, "width")
    wf, second = workflow_ops.connect(wf, graph, prim, "INT", 57, "width")
    assert [i["name"] for i in _node(wf, 57)["inputs"]] == ["text", "width"]
    assert _input(_node(wf, 57), "width")["link"] == second["link_id"]
    assert first["link_id"] not in {link[0] for link in wf["links"]}  # replaced, not duplicated


# --------------------------------------------------------------------------- #
# Scenario 1, nuance: follow the link to the source of truth
# --------------------------------------------------------------------------- #


def test_set_widget_follows_the_link_to_the_primitive(graph):
    wf, prim = _with_primitive(graph)
    wf, _ = workflow_ops.connect(wf, graph, prim, "INT", 57, "width")
    # A later write to the same register carries a later base_version (the
    # CLI's --base-version); equal stamps tie-break by op_id under LWW.
    wf, op = workflow_ops.set_widget(wf, graph, 57, "width", 512, base_version=1)
    assert op["node_id"] == prim
    assert op["widget"] == "value"
    assert op["redirected_from"] == "57.width"
    assert _node(wf, prim)["widgets_values"][0] == 512
    assert _node(wf, 57)["widgets_values"] == []  # host untouched: the link is the truth


def test_set_widget_follows_a_reroute_chain(graph):
    wf, prim = _with_primitive(graph)
    # Reroute as the frontend serializes it (template_contact_sheet-step_3.app.json node 394)
    wf["last_node_id"] = max(int(n["id"]) for n in wf["nodes"] if isinstance(n["id"], int)) + 1
    reroute_id = wf["last_node_id"]
    wf["nodes"].append(
        {
            "id": reroute_id,
            "type": "Reroute",
            "pos": [0, 0],
            "size": [75, 26],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [{"name": "", "type": "*", "widget": {"name": "value"}, "link": None}],
            "outputs": [{"name": "", "type": "INT", "links": []}],
            "properties": {"showOutputText": False, "horizontal": False},
        }
    )
    wf, _ = workflow_ops.connect(wf, graph, prim, "INT", reroute_id, 0)
    wf, _ = workflow_ops.connect(wf, graph, reroute_id, 0, 57, "width")
    wf, op = workflow_ops.set_widget(wf, graph, 57, "width", 512, base_version=1)
    assert op["node_id"] == prim
    assert _node(wf, prim)["widgets_values"][0] == 512


def test_set_widget_follows_the_link_to_a_legacy_primitive_node(graph):
    wf = _load("image_z_image_turbo.json")
    wf["last_node_id"] += 1
    legacy = wf["last_node_id"]
    # PrimitiveNode as the frontend serializes it (templates-multiple_consistent_shots-nb_pro.json node 7)
    wf["nodes"].append(
        {
            "id": legacy,
            "type": "PrimitiveNode",
            "pos": [0, 0],
            "size": [210, 82],
            "flags": {},
            "order": 0,
            "mode": 0,
            "inputs": [],
            "outputs": [{"name": "INT", "type": "INT", "widget": {"name": "width"}, "links": []}],
            "properties": {"Run widget replace on values": False},
            "widgets_values": [1024, "fixed"],
        }
    )
    wf, _ = workflow_ops.connect(wf, graph, legacy, "INT", 57, "width")
    wf, op = workflow_ops.set_widget(wf, graph, 57, "width", 512)
    assert op["node_id"] == legacy
    assert _node(wf, legacy)["widgets_values"][0] == 512


def test_set_widget_refuses_a_driver_it_cannot_edit(graph):
    wf = _load("image_z_image_turbo.json")
    wf, sel = workflow_ops.add_node(wf, graph, "ResolutionSelector")
    wf, _ = workflow_ops.connect(wf, graph, sel["node_id"], "width", 57, "width")
    with pytest.raises(ValueError) as e:
        workflow_ops.set_widget(wf, graph, 57, "width", 512)
    msg = str(e.value)
    assert "ResolutionSelector" in msg and str(sel["node_id"]) in msg
    assert _node(wf, 57)["widgets_values"] == []


# --------------------------------------------------------------------------- #
# CLI surface
# --------------------------------------------------------------------------- #


def _run(args: list[str], capsys) -> dict[str, Any]:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(workflow_cmd.app, args, standalone_mode=False)
    out = capsys.readouterr().out
    if not out.strip():
        out = result.stdout or ""
    for line in reversed([ln for ln in out.strip().splitlines() if ln.strip()]):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no JSON envelope (rc={result.exit_code}, exc={result.exception}, out={out[:600]})")


def test_cli_set_widget_then_slots_show_the_new_host_value(tmp_path, capsys):
    path = tmp_path / "z.json"
    path.write_text(json.dumps(_load("image_z_image_turbo.json")))
    env = _run(["set-widget", str(path), "57.width", "768", "--input", str(OBJECT_INFO)], capsys)
    assert env["ok"] is True, env
    assert env["data"]["op"]["node_id"] == 57 and "path" not in env["data"]["op"]
    env = _run(["slots", str(path), "--input", str(OBJECT_INFO)], capsys)
    slots = {s["address"]: s for s in env["data"]["slots"]}
    assert slots["57.width"]["current_value"] == 768
    assert slots["57.height"]["current_value"] == 1024


def test_cli_connect_int_node_to_promoted_width(tmp_path, capsys):
    path = tmp_path / "z.json"
    path.write_text(json.dumps(_load("image_z_image_turbo.json")))
    env = _run(["add-node", str(path), "PrimitiveInt", "--input", str(OBJECT_INFO)], capsys)
    prim = env["data"]["op"]["node_id"]
    env = _run(["connect", str(path), f"{prim}.INT", "57.width", "--input", str(OBJECT_INFO)], capsys)
    assert env["ok"] is True, env
    saved = json.loads(path.read_text())
    assert _input(_node(saved, 57), "width")["link"] == env["data"]["op"]["link_id"]


# --------------------------------------------------------------------------- #
# Review findings on #815
# --------------------------------------------------------------------------- #


def test_nested_host_payload_reflects_the_forked_instance(graph):
    """A nested host inside a definition SHARED by two instances is forked on
    write; the op's ``host_widgets_values`` must be read from the written
    (forked) instance, not the pre-fork dict captured during resolution."""
    wf = _load("image_z_image_turbo.json")
    inner_sg = wf["definitions"]["subgraphs"][0]
    outer_id = "0e0e0e0e-0000-4000-8000-000000000001"
    outer = {
        "id": outer_id,
        "name": "Outer",
        "inputs": [],
        "outputs": [],
        "widgets": [],
        "links": [],
        "nodes": [
            {
                "id": 7,
                "type": inner_sg["id"],
                "pos": [0, 0],
                "size": [1, 1],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [],
            }
        ],
    }
    wf["definitions"]["subgraphs"].append(outer)
    for nid in (900, 901):  # two instances share the outer definition
        wf["nodes"].append(
            {
                "id": nid,
                "type": outer_id,
                "pos": [0, 0],
                "size": [1, 1],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [],
                "properties": {},
                "widgets_values": [],
            }
        )
    wf, op = workflow_ops.set_widget(wf, graph, "900/7", "width", 640)
    forked = _node(wf, 900)["type"]
    assert forked != outer_id  # the shared definition was forked for instance 900
    inner = next(
        n for n in next(d for d in wf["definitions"]["subgraphs"] if d["id"] == forked)["nodes"] if n["id"] == 7
    )
    assert inner["widgets_values"][1] == 640
    assert op["promoted"]["host_widgets_values"] == inner["widgets_values"]
    # the sibling's definition is untouched
    assert next(n for n in outer["nodes"] if n["id"] == 7)["widgets_values"] == []


def test_connect_does_not_type_check_against_an_untyped_declared_input(graph):
    wf, prim = _with_primitive(graph)
    sg = next(d for d in wf["definitions"]["subgraphs"] if d["id"] == _node(wf, 57)["type"])
    width = next(i for i in sg["inputs"] if i["name"] == "width")
    del width["type"]
    wf, op = workflow_ops.connect(wf, graph, prim, "INT", 57, "width")
    assert _input(_node(wf, 57), "width")["link"] == op["link_id"]
    assert op["grow"]["type"] == "INT"  # falls back to the source type
