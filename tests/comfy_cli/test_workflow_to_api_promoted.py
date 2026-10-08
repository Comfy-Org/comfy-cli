"""UI→API conversion honors host-owned promoted widget values.

``convert_ui_to_api`` expanded subgraph instances by reattaching *external*
links to interior inputs and otherwise reading the interior node's own
``widgets_values``. That is the pre-ADR 0009 world. A post-migration save
keeps the promoted value on the HOST instance (``widgets_values`` positional
over the widget-backed subgraph inputs), so the interior widget can be stale
or empty — ``audio_minimax_music_3`` ships an interior ``caption`` of ``''``
while the host carries the whole prompt — and the prompt the CLI submitted
(``comfy run`` / ``validate``) was not the prompt the frontend runs.

Precedence, exactly as the frontend serializes it: an external link into the
instance input wins over everything; else the host value when materialized;
else the interior widget.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from comfy_cli import workflow_ops, workflow_to_api
from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api

_FIXTURES = Path(__file__).resolve().parent / "fixtures"
_GALLERY = _FIXTURES / "gallery"


@pytest.fixture(scope="module")
def object_info() -> dict:
    return json.loads((_FIXTURES / "object_info_subgraph_promoted.json").read_text())


@pytest.fixture(scope="module")
def graph(object_info) -> Graph:
    return Graph.from_object_info(object_info)


def _load(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


def _api_node(api: dict, class_type: str, api_id: str | None = None) -> dict:
    if api_id is not None:
        return api[api_id]
    return next(v for v in api.values() if v["class_type"] == class_type)


def test_post_migration_host_values_reach_the_prompt(object_info):
    api = convert_ui_to_api(_load("audio_minimax_music_3.json"), object_info)
    enc = _api_node(api, "MiniMaxMusic3TextEncode", "37:13")["inputs"]
    assert enc["caption"].startswith("Global Metadata: Lo-fi hip-hop")
    assert enc["lyrics"].startswith("[Intro]")
    assert enc["max_duration"] == 60
    assert _api_node(api, "UNETLoader", "37:6")["inputs"]["unet_name"] == "minimax_music3_dit_fp16.safetensors"
    assert _api_node(api, "ComfySwitchNode", "37:43")["inputs"]["switch"] is True


def test_host_value_written_by_set_widget_reaches_the_prompt(object_info, graph):
    wf = _load("image_z_image_turbo.json")
    wf, _ = workflow_ops.set_widget(wf, graph, 57, "width", 768)
    api = convert_ui_to_api(wf, object_info)
    latent = _api_node(api, "EmptySD3LatentImage", "57:13")["inputs"]
    assert latent["width"] == 768
    assert latent["height"] == 1024


def test_interior_value_still_used_when_no_host_value_exists(object_info):
    api = convert_ui_to_api(_load("image_z_image_turbo.json"), object_info)
    assert _api_node(api, "EmptySD3LatentImage", "57:13")["inputs"]["width"] == 1024
    assert _api_node(api, "KSampler", "57:3")["inputs"]["steps"] == 8


def test_external_link_into_a_promoted_input_wins_over_the_host_value(object_info, graph):
    wf = _load("image_z_image_turbo.json")
    wf, _ = workflow_ops.set_widget(wf, graph, 57, "width", 768)
    wf, prim = workflow_ops.add_node(wf, graph, "PrimitiveInt")
    wf, _ = workflow_ops.set_widget(wf, graph, prim["node_id"], "value", 640)
    wf, _ = workflow_ops.connect(wf, graph, prim["node_id"], "INT", 57, "width")
    api = convert_ui_to_api(wf, object_info)
    assert _api_node(api, "EmptySD3LatentImage", "57:13")["inputs"]["width"] == [str(prim["node_id"]), 0]
    assert api[str(prim["node_id"])]["inputs"]["value"] == 640


def test_socket_links_and_host_values_coexist(object_info):
    api = convert_ui_to_api(_load("api_seedance2_5_video_extend.json"), object_info)
    pad = _api_node(api, "PrimitiveBoolean", "39:28")["inputs"]
    assert pad["value"] is False
    resize = _api_node(api, "ResizeAndPadImage", "39:6")["inputs"]
    assert resize["interpolation"] == "lanczos"
    assert resize["padding_color"] == "white"
    # the two VIDEO socket inputs are external links, reattached as before
    comps = _api_node(api, "GetVideoComponents", "39:1")["inputs"]
    assert isinstance(comps["video"], list) and len(comps["video"]) == 2


def test_list_valued_host_values_are_wrapped_not_read_as_links(object_info, graph):
    """A two-item list host value must not be mistaken for a ``[node, slot]``
    link when overlaid onto the interior node."""
    wf = _load("image_z_image_turbo.json")
    from comfy_cli.cql import promoted

    promoted.set_host_value(wf, next(n for n in wf["nodes"] if n["id"] == 57), "text", ["a", "b"], graph)
    api = convert_ui_to_api(wf, object_info)
    text = _api_node(api, "CLIPTextEncode", "57:27")["inputs"]["text"]
    plain = _api_node(convert_ui_to_api(_load("image_z_image_turbo.json"), object_info), "CLIPTextEncode", "57:27")[
        "inputs"
    ]["text"]
    assert isinstance(plain, str)
    assert text != ["a", "b"] or not (isinstance(text, list) and len(text) == 2 and isinstance(text[0], str))
    assert text == workflow_to_api._wrap_widget_value(["a", "b"])


def test_dangling_link_on_a_promoted_input_does_not_drop_the_host_value(object_info):
    """A promoted input whose serialized ``link`` id no longer exists in
    ``links`` is unlinked as far as the frontend is concerned (it drops the
    link on load): the host value must reach the prompt, exactly as
    ``resolve_write`` already treats that shape."""
    wf = _load("audio_minimax_music_3.json")
    inst = next(n for n in wf["nodes"] if n["id"] == 37)
    inst["inputs"].append({"name": "caption", "type": "STRING", "widget": {"name": "caption"}, "link": 999999})
    assert all(link[0] != 999999 for link in wf["links"])
    api = convert_ui_to_api(wf, object_info)
    assert _api_node(api, "MiniMaxMusic3TextEncode", "37:13")["inputs"]["caption"].startswith("Global Metadata")


def test_stale_boundary_target_reattaches_external_value_to_the_actual_holder(object_info, graph):
    wf = _load("image_z_image_turbo.json")
    wf, primitive = workflow_ops.add_node(wf, graph, "PrimitiveInt")
    wf, _ = workflow_ops.set_widget(wf, graph, primitive["node_id"], "value", 640)
    wf, _ = workflow_ops.connect(wf, graph, primitive["node_id"], "INT", 57, "width")
    inst = next(node for node in wf["nodes"] if node["id"] == 57)
    sg = next(definition for definition in wf["definitions"]["subgraphs"] if definition["id"] == inst["type"])
    width = next(item for item in sg["inputs"] if item["name"] == "width")
    link = next(item for item in sg["links"] if item["id"] == width["linkIds"][0])
    actual = next(
        node
        for node in sg["nodes"]
        if any(isinstance(entry, dict) and entry.get("link") == link["id"] for entry in node.get("inputs") or [])
    )
    stale = next(node for node in sg["nodes"] if node is not actual and node.get("inputs"))
    link["target_id"] = stale["id"]
    link["target_slot"] = 0
    duplicate = copy.deepcopy(actual)
    duplicate["id"] = 999
    for entry in duplicate.get("inputs") or []:
        if isinstance(entry, dict):
            entry["link"] = link["id"] if entry.get("name") == "width" else None
    sg["nodes"].append(duplicate)

    api = convert_ui_to_api(wf, object_info)

    expected = [str(primitive["node_id"]), 0]
    assert _api_node(api, "EmptySD3LatentImage", "57:13")["inputs"]["width"] == expected
    assert api["57:999"]["inputs"]["width"] == expected


def test_nested_duplicate_input_holders_resolve_once_per_level():
    ctx = workflow_to_api._SubgraphCtx()
    node_id = "root"
    for _ in range(40):
        ctx.input_targets[node_id] = {0: [(7, 0)] * 8}
        node_id = f"{node_id}:7"

    with mock.patch.object(
        workflow_to_api,
        "_resolve_subgraph_input_all",
        wraps=workflow_to_api._resolve_subgraph_input_all,
    ) as resolver:
        result = workflow_to_api._resolve_subgraph_input_all("root", 0, ctx)

    assert result == [(node_id, 0)]
    assert resolver.call_count <= 41


def test_subgraph_input_resolution_fails_closed_at_materialization_cap():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["root"] = {
        0: [(node_id, 0) for node_id in range(workflow_to_api._MAX_RESOLVED_SUBGRAPH_INPUTS + 1)]
    }

    with pytest.raises(workflow_to_api.WorkflowConversionError, match="input resolution exceeded"):
        workflow_to_api._resolve_subgraph_input_all("root", 0, ctx)
