"""``connect`` / ``add-node`` on nodes whose auto-grow groups live under a dynamic combo.

Reproduces the in-app agent defect: on an agent-built ``GeminiNanoBanana2V2``
the reference input did not exist (``add_node`` wrote ``inputs: []``) and no
address could create it; on a UI-built ``MinimaxHailuo03ReferenceNode`` only
the one free slot the frontend pre-creates could be wired, so a second or
third reference — or any Load Video / Load Audio — failed with
``input ... not found``. The same slot-driven resolver also could not
base-address or grow past the free slot on a UI-built *top-level* group
(``BatchImagesNode``).

The frontend contract these tests hold the CLI to
(``ComfyUI_frontend/src/core/graph/widgets/dynamicWidgets.ts``):

* slot names are ``<group>.<names[ordinal]>`` (or ``<group>.<prefix><ordinal>``),
  with ``<group>`` dotted under its combo (``model.reference_images.image_1``);
* a group holds at most ``names.length`` (or ``template.max``) slots;
* a grown slot carries the group's declared element type;
* the group owns no ``widgets_values`` slot, so wiring never shifts a widget.

Fixture: ``tests/comfy_cli/fixtures/object_info_nested_autogrow.json`` — the
production catalog's entries with tooltips stripped.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli import workflow_ops, workflow_to_api
from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

FIXTURE = Path(__file__).resolve().parents[2] / "comfy_cli" / "fixtures" / "object_info_nested_autogrow.json"


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def graph(object_info) -> Graph:
    return Graph.from_object_info(object_info)


def _empty() -> dict[str, Any]:
    return {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [], "version": 0.4}


def _node(wf: dict, node_id: Any) -> dict:
    return next(n for n in wf["nodes"] if n["id"] == node_id)


def _inputs(wf: dict, node_id: Any) -> list[str]:
    return [i["name"] for i in _node(wf, node_id).get("inputs") or []]


def _link_of(wf: dict, node_id: Any, name: str):
    return next(i for i in _node(wf, node_id)["inputs"] if i["name"] == name)["link"]


def _minimax_ui_workflow() -> dict[str, Any]:
    """The ``api_minimax_h3_r2v`` gallery template's shape: ``image_1`` wired,
    one free trailing slot per group (``shape: 7`` = optional), 8 widget values."""
    return {
        "last_node_id": 4,
        "last_link_id": 4,
        "version": 0.4,
        "groups": [],
        "nodes": [
            {
                "id": 2,
                "type": "LoadImage",
                "pos": [0, 0],
                "size": [300, 300],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": "IMAGE", "type": "IMAGE", "links": [4]},
                    {"name": "MASK", "type": "MASK", "links": []},
                ],
                "properties": {},
                "widgets_values": ["storyboard.png", "image"],
            },
            {
                "id": 4,
                "type": "MinimaxHailuo03ReferenceNode",
                "pos": [400, 0],
                "size": [400, 300],
                "flags": {},
                "order": 1,
                "mode": 0,
                "inputs": [
                    {
                        "label": "image_1",
                        "name": "model.reference_images.image_1",
                        "shape": 7,
                        "type": "IMAGE",
                        "link": 4,
                    },
                    {
                        "label": "image_2",
                        "name": "model.reference_images.image_2",
                        "shape": 7,
                        "type": "IMAGE",
                        "link": None,
                    },
                    {
                        "label": "video_1",
                        "name": "model.reference_videos.video_1",
                        "shape": 7,
                        "type": "VIDEO",
                        "link": None,
                    },
                    {
                        "label": "audio_1",
                        "name": "model.reference_audios.audio_1",
                        "shape": 7,
                        "type": "AUDIO",
                        "link": None,
                    },
                ],
                "outputs": [{"name": "VIDEO", "type": "VIDEO", "links": []}],
                "properties": {},
                "widgets_values": ["MiniMax H3", "storyboard prompt", "768P", "adaptive", 5, 42, "randomize", False],
            },
        ],
        "links": [[4, 2, 0, 4, 0, "IMAGE"]],
    }


def _add_loader(wf: dict, graph: Graph, class_type: str) -> tuple[dict, int]:
    wf, op = workflow_ops.add_node(wf, graph, class_type)
    return wf, op["node_id"]


# --------------------------------------------------------------------------- #
# add-node: no phantom widget slots, no phantom input slots
# --------------------------------------------------------------------------- #


def test_add_node_writes_frontend_width_widgets_and_no_group_inputs(graph):
    wf, _ = workflow_ops.add_node(_empty(), graph, "MinimaxHailuo03ReferenceNode")
    node = wf["nodes"][0]
    assert node["inputs"] == []
    order = graph.widget_order_default("MinimaxHailuo03ReferenceNode")
    assert len(node["widgets_values"]) == len(order) == 8
    by_name = dict(zip(order, node["widgets_values"]))
    assert by_name["model"] == "MiniMax H3"
    assert by_name["seed"] == 42
    assert by_name["watermark"] is False


# --------------------------------------------------------------------------- #
# connect on an agent-built node: grow nested slots by name, by base, in order
# --------------------------------------------------------------------------- #


def test_connect_grows_first_nested_slot_by_its_schema_name(graph):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    widgets_before = list(_node(wf, b)["widgets_values"])
    wf, op = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images.image_1")
    grown = _node(wf, b)["inputs"]
    assert [i["name"] for i in grown] == ["model.images.image_1"]
    assert grown[0]["type"] == "IMAGE"
    assert grown[0]["link"] == op["link_id"]
    assert _node(wf, b)["widgets_values"] == widgets_before


def test_connect_to_the_group_base_appends_the_next_slot(graph):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images")
    assert _inputs(wf, b) == ["model.images.image_1", "model.images.image_2"]


def test_connect_explicit_next_key_ok_and_gap_rejected_with_the_next_free_key(graph):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images.image_1")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images.image_2")
    with pytest.raises(ValueError, match=r"model\.images\.image_3"):
        workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images.image_5")
    assert _inputs(wf, b) == ["model.images.image_1", "model.images.image_2"]


def test_bare_element_name_maps_onto_the_nested_group(graph):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "image_1")
    assert _inputs(wf, b) == ["model.images.image_1"]


def test_group_resolution_follows_the_node_selection(graph):
    """Switching the selector to an option without the group makes the
    group unaddressable — the resolver reads the node, not the first key."""
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.set_widget(wf, graph, b, "model", "Nano Banana 2 Lite")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images")
    assert _inputs(wf, b) == ["model.images.image_1"]


# --------------------------------------------------------------------------- #
# connect on a UI-built node: reuse the free slot, then keep growing
# --------------------------------------------------------------------------- #


def test_ui_built_free_slot_is_reused_before_growing(graph):
    wf = _minimax_ui_workflow()
    wf, op = workflow_ops.connect(wf, graph, 2, "IMAGE", 4, "model.reference_images")
    assert _inputs(wf, 4) == [
        "model.reference_images.image_1",
        "model.reference_images.image_2",
        "model.reference_videos.video_1",
        "model.reference_audios.audio_1",
    ]
    assert _link_of(wf, 4, "model.reference_images.image_2") == op["link_id"]
    wf, _ = workflow_ops.connect(wf, graph, 2, "IMAGE", 4, "model.reference_images")
    assert "model.reference_images.image_3" in _inputs(wf, 4)


def test_ui_built_explicit_third_slot_grows(graph):
    wf = _minimax_ui_workflow()
    wf, _ = workflow_ops.connect(wf, graph, 2, "IMAGE", 4, "model.reference_images.image_2")
    wf, op = workflow_ops.connect(wf, graph, 2, "IMAGE", 4, "model.reference_images.image_3")
    grown = next(i for i in _node(wf, 4)["inputs"] if i["name"] == "model.reference_images.image_3")
    assert grown["type"] == "IMAGE"
    assert grown["link"] == op["link_id"]


def test_load_video_and_load_audio_wire_into_their_reference_groups(graph):
    wf = _minimax_ui_workflow()
    wf, v = _add_loader(wf, graph, "LoadVideo")
    wf, au = _add_loader(wf, graph, "LoadAudio")
    wf, op_v = workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_videos")
    wf, op_a = workflow_ops.connect(wf, graph, au, "AUDIO", 4, "model.reference_audios.audio_1")
    assert _link_of(wf, 4, "model.reference_videos.video_1") == op_v["link_id"]
    assert _link_of(wf, 4, "model.reference_audios.audio_1") == op_a["link_id"]
    wf, _ = workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_videos.video_2")
    assert "model.reference_videos.video_2" in _inputs(wf, 4)


def test_wrong_element_type_into_a_group_is_refused(graph):
    wf = _minimax_ui_workflow()
    wf, v = _add_loader(wf, graph, "LoadVideo")
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_images")
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_images.image_3")


def test_group_max_from_the_schema_is_enforced(graph):
    wf = _minimax_ui_workflow()
    wf, v = _add_loader(wf, graph, "LoadVideo")
    for _ in range(3):
        wf, _ = workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_videos")
    assert [n for n in _inputs(wf, 4) if n.startswith("model.reference_videos.")] == [
        "model.reference_videos.video_1",
        "model.reference_videos.video_2",
        "model.reference_videos.video_3",
    ]
    with pytest.raises(ValueError, match=r"(?i)max(imum)? .*3|3 slots|full"):
        workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_videos")


def test_ui_built_top_level_group_base_and_next_key(graph):
    wf = _empty()
    wf, a = _add_loader(wf, graph, "LoadImage")
    wf["nodes"].append(
        {
            "id": 99,
            "type": "BatchImagesNode",
            "pos": [0, 0],
            "size": [200, 100],
            "flags": {},
            "order": 1,
            "mode": 0,
            "inputs": [
                {"name": "images.image0", "type": "IMAGE", "link": 77},
                {"name": "images.image1", "type": "IMAGE", "link": None, "shape": 7},
            ],
            "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": []}],
            "properties": {},
            "widgets_values": [],
        }
    )
    wf["last_node_id"] = 99
    wf["last_link_id"] = 77
    wf, op = workflow_ops.connect(wf, graph, a, "IMAGE", 99, "images")
    assert _inputs(wf, 99) == ["images.image0", "images.image1"]
    assert _link_of(wf, 99, "images.image1") == op["link_id"]
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", 99, "images.image2")
    assert _inputs(wf, 99) == ["images.image0", "images.image1", "images.image2"]


# --------------------------------------------------------------------------- #
# widgets never shift; the converter carries every reference
# --------------------------------------------------------------------------- #


def test_set_widget_after_the_groups_lands_in_the_frontend_position(graph):
    wf = _minimax_ui_workflow()
    wf, _ = workflow_ops.set_widget(wf, graph, 4, "seed", 7)
    values = _node(wf, 4)["widgets_values"]
    assert values[5] == 7
    assert values[7] is False
    assert len(values) == 8


def test_agent_built_node_converts_with_every_reference_and_unshifted_widgets(graph, object_info):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, c = _add_loader(wf, graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images")
    wf, _ = workflow_ops.connect(wf, graph, c, "IMAGE", b, "model.images")
    wf, _ = workflow_ops.set_widget(wf, graph, b, "seed", 7)
    wf, save = _add_loader(wf, graph, "SaveImage")
    wf, _ = workflow_ops.connect(wf, graph, b, "IMAGE", save, "images")
    api = workflow_to_api.convert_ui_to_api(wf, object_info)
    inputs = api[str(b)]["inputs"]
    assert inputs["model.images.image_1"] == [str(a), 0]
    assert inputs["model.images.image_2"] == [str(c), 0]
    assert inputs["seed"] == 7
    assert inputs["response_modalities"] == "IMAGE"
    assert inputs["temperature"] == 1.0
    assert inputs["top_p"] == 0.95
    assert "model.images" not in inputs
    assert "model.files" not in inputs
    assert graph.validate_workflow(api)["errors"] == []


def test_replaying_a_nested_grow_op_is_idempotent(graph):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, b = _add_loader(wf, graph, "GeminiNanoBanana2V2")
    base = copy.deepcopy(wf)
    wf, op = workflow_ops.connect(wf, graph, a, "IMAGE", b, "model.images")
    replayed = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), op, graph), op, graph)
    assert _inputs(replayed, b) == ["model.images.image_1"]
    assert _inputs(replayed, b) == _inputs(wf, b)


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


def test_cli_connect_load_image_into_nano_banana_reference_group(tmp_path, capsys):
    wf_path = tmp_path / "wf.json"
    wf_path.write_text(json.dumps(_empty()))
    env_a = _run(["add-node", str(wf_path), "LoadImage", "--input", str(FIXTURE)], capsys)
    env_b = _run(["add-node", str(wf_path), "GeminiNanoBanana2V2", "--input", str(FIXTURE)], capsys)
    assert env_a["ok"] and env_b["ok"], (env_a, env_b)
    a, b = env_a["data"]["op"]["node_id"], env_b["data"]["op"]["node_id"]
    env = _run(["connect", str(wf_path), f"{a}.IMAGE", f"{b}.model.images.image_1", "--input", str(FIXTURE)], capsys)
    assert env["ok"] is True, env
    env = _run(["connect", str(wf_path), f"{a}.IMAGE", f"{b}.model.images", "--input", str(FIXTURE)], capsys)
    assert env["ok"] is True, env
    saved = json.loads(wf_path.read_text())
    assert _inputs(saved, b) == ["model.images.image_1", "model.images.image_2"]


def test_grows_on_different_nested_groups_are_different_conflict_targets(graph):
    """``model.reference_images.image_1`` and ``model.reference_videos.video_1``
    belong to two groups; keying the grow target on the FIRST dot made both
    ``model`` and reported a false conflict between them."""
    wf = _minimax_ui_workflow()
    wf, v = _add_loader(wf, graph, "LoadVideo")
    _, op_img = workflow_ops.connect(copy.deepcopy(wf), graph, 2, "IMAGE", 4, "model.reference_images.image_3")
    _, op_vid = workflow_ops.connect(copy.deepcopy(wf), graph, v, "VIDEO", 4, "model.reference_videos.video_2")
    assert workflow_ops._write_target(op_img) == ("input", "4", "grow", "model.reference_images")
    assert workflow_ops._write_target(op_vid) == ("input", "4", "grow", "model.reference_videos")
    assert workflow_ops.detect_conflict(op_img, op_vid) is False


# --------------------------------------------------------------------------- #
# Review (annehe9) on #812: the guards must cover the CLI-built shape too
# --------------------------------------------------------------------------- #


def test_cli_built_top_level_group_refuses_the_wrong_element_type(graph):
    """``add_node`` writes a ``COMFY_AUTOGROW_V3`` base entry; connecting
    through it must apply the same declared-element-type check as the
    UI-built shape (no base entry) does."""
    wf, au = _add_loader(_empty(), graph, "LoadAudio")
    wf, batch = _add_loader(wf, graph, "BatchImagesNode")
    assert _inputs(wf, batch) == ["images"]
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, au, "AUDIO", batch, "images")
    with pytest.raises(ValueError, match="type mismatch"):
        workflow_ops.connect(wf, graph, au, "AUDIO", batch, "images.image0")


def test_cli_built_top_level_group_enforces_max(graph):
    wf, a = _add_loader(_empty(), graph, "LoadImage")
    wf, batch = _add_loader(wf, graph, "BatchImagesNode")
    for _ in range(50):
        wf, _ = workflow_ops.connect(wf, graph, a, "IMAGE", batch, "images")
    assert "images.image49" in _inputs(wf, batch)
    with pytest.raises(ValueError, match=r"(?i)max(imum)? .*50|50 slots|full"):
        workflow_ops.connect(wf, graph, a, "IMAGE", batch, "images")


def test_full_group_reports_full_before_the_next_key_hint(graph):
    """A gapped key on a FULL group must not be told to use a next key the
    next call would refuse."""
    wf = _minimax_ui_workflow()
    wf, v = _add_loader(wf, graph, "LoadVideo")
    for _ in range(3):
        wf, _ = workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_videos")
    with pytest.raises(ValueError) as e:
        workflow_ops.connect(wf, graph, v, "VIDEO", 4, "model.reference_videos.video_9")
    assert "at most 3" in str(e.value)
    assert "next free key" not in str(e.value)


def test_a_real_widget_name_outranks_the_bare_element_guess(graph):
    """A node whose group's bare element vocabulary collides with a real
    widget name keeps the widget→input conversion (the existing precedence)."""
    oi = json.loads(FIXTURE.read_text())
    oi["BatchImagesNode"]["input"]["required"]["image0"] = ["STRING", {"default": ""}]
    oi["BatchImagesNode"]["input_order"]["required"].append("image0")
    g = Graph.from_object_info(oi)
    wf, prim = _add_loader(_empty(), g, "LoadImage")
    wf, batch = _add_loader(wf, g, "BatchImagesNode")
    _idx, grow = workflow_ops._resolve_input_target(_node(wf, batch), g, "image0", "STRING")
    assert grow == {"name": "image0", "type": "STRING", "widget": "image0"}
