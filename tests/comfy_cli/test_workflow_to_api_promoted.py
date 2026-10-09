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
from comfy_cli.cql.promoted import PromotionTraversalLimitError
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

    memo: dict = {}
    budget = [workflow_to_api._MAX_RESOLVED_SUBGRAPH_INPUTS]
    first = workflow_to_api._resolve_subgraph_input_all("root", 0, ctx, _memo=memo, _budget=budget)
    remaining = budget[0]
    second = workflow_to_api._resolve_subgraph_input_all("root", 0, ctx, _memo=memo, _budget=budget)
    assert second is first
    assert budget[0] == remaining - len(first)


@pytest.mark.parametrize("slot", [[], {}, True])
def test_malformed_subgraph_target_slots_are_left_unresolved(slot):
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["root"] = {0: [(7, 0)], 1: [(8, 0)]}

    assert workflow_to_api._resolve_subgraph_input_all("root", slot, ctx) == [("root", slot)]


def test_plain_links_do_not_consume_the_subgraph_fanout_reserve():
    count = workflow_to_api._MAX_RESOLVED_SUBGRAPH_INPUTS + 1
    links = [[index, "source", 0, "plain", 0, "*"] for index in range(count)]
    ctx = workflow_to_api._SubgraphCtx()
    # Make the rewrite path live while leaving every serialized row plain.
    ctx.input_targets["unrelated-subgraph"] = {0: [(7, 0)]}

    assert workflow_to_api._rewrite_links_for_subgraphs(links, ctx, []) == links


def test_subgraph_expansion_uses_boundary_link_membership_not_stale_row_slots():
    definition = {
        "inputs": [{"name": "wrong", "linkIds": []}, {"name": "right", "linkIds": [1]}],
        "outputs": [{"name": "wrong_out", "linkIds": []}, {"name": "right_out", "linkIds": [2]}],
        "nodes": [
            {
                "id": 7,
                "type": "Example",
                "inputs": [{"name": "value", "link": 1}],
                "outputs": [{"name": "result"}],
            }
        ],
        "links": [
            {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0},
            {"id": 2, "origin_id": 7, "origin_slot": 0, "target_id": -20, "target_slot": 0},
        ],
    }

    _nodes, _links, input_targets, output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert input_targets == {1: [(7, 0)]}
    assert output_sources == {1: (7, 0)}


@pytest.mark.parametrize("origin_id,origin_slot", [([], 0), ({}, 0), (7, True), (7, 0.0)])
def test_subgraph_expansion_skips_malformed_output_source_coordinates(origin_id, origin_slot):
    definition = {
        "inputs": [],
        "outputs": [{"name": "result", "linkIds": [1]}],
        "nodes": [],
        "links": [
            {
                "id": 1,
                "origin_id": origin_id,
                "origin_slot": origin_slot,
                "target_id": -20,
                "target_slot": 0,
            }
        ],
    }

    _nodes, _links, _input_targets, output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert output_sources == {}


@pytest.mark.parametrize("origin_slot", [-1, 1])
def test_subgraph_expansion_skips_out_of_range_output_sources(origin_slot):
    definition = {
        "inputs": [],
        "outputs": [{"name": "result", "linkIds": [1]}],
        "nodes": [{"id": 7, "type": "Producer", "inputs": [], "outputs": [{"name": "value"}]}],
        "links": [
            {
                "id": 1,
                "origin_id": 7,
                "origin_slot": origin_slot,
                "target_id": -20,
                "target_slot": 0,
            }
        ],
    }

    _nodes, _links, _input_targets, output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert output_sources == {}


def test_one_interior_output_can_feed_multiple_definition_outputs():
    definition = {
        "inputs": [],
        "outputs": [
            {"name": "first", "linkIds": [1]},
            {"name": "second", "linkIds": [1]},
        ],
        "nodes": [{"id": 7, "type": "Producer", "inputs": [], "outputs": [{"name": "value"}]}],
        "links": [{"id": 1, "origin_id": 7, "origin_slot": 0, "target_id": -20, "target_slot": 0}],
    }

    _nodes, _links, _input_targets, output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert output_sources == {0: (7, 0), 1: (7, 0)}
    ctx = workflow_to_api._SubgraphCtx()
    ctx.output_sources["10"] = output_sources
    external = [[11, 10, 0, 20, 0, "*"], [12, 10, 1, 21, 0, "*"]]
    rewritten = workflow_to_api._rewrite_links_for_subgraphs(external, ctx, [])
    assert [link[1:3] for link in rewritten] == [["10:7", 0], ["10:7", 0]]


def test_definition_input_to_output_passthrough_uses_the_outer_input_source():
    definition = {
        "inputs": [{"name": "value", "linkIds": [1]}],
        "outputs": [{"name": "result", "linkIds": [1]}],
        "nodes": [],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": -20, "target_slot": 0}],
    }
    _nodes, _links, input_targets, output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = input_targets
    ctx.output_sources["10"] = output_sources
    ctx.outer_to_input_idx["10"] = {0: 0}
    external = [[11, "producer", 2, 10, 0, "*"], [12, 10, 0, "consumer", 0, "*"]]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs(external, ctx, [])

    assert next(link for link in rewritten if link[0] == 12)[1:3] == ["producer", 2]


def test_passthrough_uses_the_held_outer_input_not_stale_row_order():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.output_sources["10"] = {0: (-10, 1)}
    ctx.held_input_idx["10"] = {2: (1,)}
    ctx.input_holders[2] = [("10", 1)]
    links = [
        [1, "stale", 0, 10, 1, "*"],
        [2, "live", 3, 10, 0, "*"],
        [3, 10, 0, "consumer", 0, "*"],
    ]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs(links, ctx, [])

    assert next(link for link in rewritten if link[0] == 3)[1:3] == ["live", 3]
    assert all(link[0] != 1 for link in rewritten)


def test_nested_input_to_output_passthrough_inherits_the_parent_source():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [(7, 0)]}
    ctx.output_sources["10"] = {0: (7, 0)}
    ctx.input_targets["10:7"] = {}
    ctx.output_sources["10:7"] = {0: ("-10", 0)}
    ctx.held_input_idx["10"] = {1: (0,)}
    ctx.input_holders[1] = [("10", 0)]
    links = [
        [1, "producer", 2, 10, 0, "*"],
        [2, 10, 0, "consumer", 0, "*"],
    ]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs(links, ctx, [])

    assert next(link for link in rewritten if link[0] == 2)[1:3] == ["producer", 2]


def test_passthrough_uses_input_membership_and_accepts_string_proxy_id():
    definition = {
        "inputs": [
            {"name": "wrong", "linkIds": []},
            {"name": "right", "linkIds": [1]},
        ],
        "outputs": [{"name": "result", "linkIds": [1]}],
        "nodes": [],
        "links": [{"id": 1, "origin_id": "-10", "origin_slot": 0, "target_id": -20, "target_slot": 0}],
    }

    _nodes, _links, _inputs, output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert output_sources == {0: (-10, 1)}


def test_large_serialized_boundary_does_not_consume_the_fanout_reserve():
    count = 5_001
    definition = {
        "inputs": [{"name": f"value-{index}", "linkIds": [index]} for index in range(count)],
        "outputs": [],
        "nodes": [
            {"id": index, "type": "Example", "inputs": [{"name": "value", "link": index}], "outputs": []}
            for index in range(count)
        ],
        "links": [
            {"id": index, "origin_id": -10, "origin_slot": index, "target_id": index, "target_slot": 0}
            for index in range(count)
        ],
    }

    _nodes, _links, input_targets, _output_sources = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert len(input_targets) == count
    outer_links = [[10_000 + index, "source", 0, 10, index, "*"] for index in range(count)]
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = input_targets
    assert len(workflow_to_api._rewrite_links_for_subgraphs(outer_links, ctx, [])) == count


def test_subgraph_input_resolution_fails_closed_at_materialization_cap():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["root"] = {
        0: [(node_id, 0) for node_id in range(workflow_to_api._MAX_RESOLVED_SUBGRAPH_INPUTS + 1)]
    }

    with pytest.raises(workflow_to_api.WorkflowConversionError, match="input resolution exceeded"):
        workflow_to_api._resolve_subgraph_input_all("root", 0, ctx)


def test_held_definition_index_is_not_mapped_through_outer_order_twice():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [("a", 0)], 1: [("b", 0)]}
    ctx.outer_to_input_idx["10"] = {0: 1, 1: 0}
    ctx.held_input_idx["10"] = {7: (1,)}
    ctx.input_holders[7] = [("10", 1)]
    nodes = [
        {"id": "10:a", "inputs": [{"name": "value", "link": None}]},
        {"id": "10:b", "inputs": [{"name": "value", "link": None}]},
    ]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, 10, 0, "*"]], ctx, nodes)

    assert rewritten[0][3:5] == ["10:b", 0]
    assert nodes[0]["inputs"][0]["link"] is None
    assert nodes[1]["inputs"][0]["link"] == 7


def test_expanded_holder_recovers_a_stale_declared_target():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [("inside", 0)]}
    ctx.held_input_idx["10"] = {7: (0,)}
    ctx.input_holders[7] = [("10", 0)]
    nodes = [{"id": "10:inside", "inputs": [{"name": "value", "link": None}]}]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, "missing", 9, "*"]], ctx, nodes)

    assert rewritten[0][3:5] == ["10:inside", 0]
    assert nodes[0]["inputs"][0]["link"] == 7


def test_one_outer_link_fans_out_to_every_instance_input_holder():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [("a", 0)], 1: [("b", 0)]}
    ctx.held_input_idx["10"] = {7: (0, 1)}
    ctx.input_holders[7] = [("10", 0), ("10", 1)]
    nodes = [
        {"id": "10:a", "inputs": [{"name": "value", "link": None}]},
        {"id": "10:b", "inputs": [{"name": "value", "link": None}]},
    ]

    workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, 10, 0, "*"]], ctx, nodes)

    assert [node["inputs"][0]["link"] for node in nodes] == [7, 7]


def test_definition_membership_requires_the_boundary_proxy_endpoints():
    definition = {
        "inputs": [{"name": "value", "linkIds": [1]}],
        "outputs": [{"name": "result", "linkIds": [2]}],
        "nodes": [
            {
                "id": 7,
                "type": "Example",
                "inputs": [{"name": "value", "link": 1}],
                "outputs": [{"name": "result"}],
            }
        ],
        "links": [
            {"id": 1, "origin_id": 99, "origin_slot": 0, "target_id": 7, "target_slot": 0},
            {"id": 2, "origin_id": 7, "origin_slot": 0, "target_id": 99, "target_slot": 0},
        ],
    }

    _nodes, _links, inputs, outputs = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert inputs == {}
    assert outputs == {}


def test_first_live_definition_link_for_an_output_wins():
    definition = {
        "inputs": [],
        "outputs": [{"name": "result", "linkIds": [2, 1]}],
        "nodes": [
            {"id": 7, "type": "First", "inputs": [], "outputs": [{"name": "value"}]},
            {"id": 8, "type": "Second", "inputs": [], "outputs": [{"name": "value"}]},
        ],
        "links": [
            {"id": 1, "origin_id": 7, "origin_slot": 0, "target_id": -20, "target_slot": 0},
            {"id": 2, "origin_id": 8, "origin_slot": 0, "target_id": -20, "target_slot": 0},
        ],
    }

    _nodes, _links, _inputs, outputs = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert outputs == {0: (7, 0)}


@pytest.mark.parametrize("field", ["inputs", "outputs", "nodes", "links"])
def test_subgraph_expansion_treats_non_list_containers_as_empty(field):
    definition = {"inputs": [], "outputs": [], "nodes": [], "links": []}
    definition[field] = 5

    nodes, links, inputs, outputs = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert isinstance(nodes, list)
    assert isinstance(links, list)
    assert inputs == {}
    assert outputs == {}


def test_materialization_cap_allows_the_documented_direct_fanout():
    count = 3_434
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [(index, 0) for index in range(count)]}
    nodes = [{"id": f"10:{index}", "inputs": [{"name": "value", "link": None}]} for index in range(count)]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, 10, 0, "*"]], ctx, nodes)

    assert len(rewritten) == 1
    assert all(node["inputs"][0]["link"] == 7 for node in nodes)


def test_repeated_instances_reuse_definition_boundary_indexes():
    subgraph_id = "22222222-3333-4444-5555-666666666666"
    definition = {
        "id": subgraph_id,
        "inputs": [{"name": "value", "linkIds": [1]}],
        "outputs": [],
        "nodes": [{"id": 7, "type": "Example", "inputs": [{"name": "value", "link": 1}], "outputs": []}],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }
    nodes = [
        {"id": 10, "type": subgraph_id, "inputs": [], "outputs": []},
        {"id": 11, "type": subgraph_id, "inputs": [], "outputs": []},
    ]
    from comfy_cli.cql import promoted

    with mock.patch.object(promoted, "_link_holders", wraps=promoted._link_holders) as holders:
        workflow_to_api._expand_subgraphs(nodes, [], {subgraph_id: definition})

    assert holders.call_count == 1


def test_expansion_preserves_raw_input_slots_for_boundary_updates():
    definition = {
        "inputs": [{"name": "value", "linkIds": [1]}],
        "outputs": [],
        "nodes": [
            {
                "id": 7,
                "type": "Example",
                "inputs": ["malformed", {"name": "value", "link": 1}],
                "outputs": [],
            }
        ],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 1}],
    }

    nodes, _links, input_targets, _outputs = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = input_targets
    workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, 10, 0, "*"]], ctx, nodes)

    assert nodes[0]["inputs"][0] == "malformed"
    assert nodes[0]["inputs"][1]["link"] == 7


@pytest.mark.parametrize("target_slot", [[], {}, True, 1.0])
def test_plain_malformed_or_stale_target_slots_do_not_overwrite_holders(target_slot):
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["unrelated-subgraph"] = {0: [(7, 0)]}
    nodes = [
        {
            "id": "plain",
            "inputs": [
                {"name": "other", "link": 9},
                {"name": "actual", "link": 7},
            ],
        }
    ]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, "plain", target_slot, "*"]], ctx, nodes)

    assert nodes[0]["inputs"][0]["link"] == 9
    assert nodes[0]["inputs"][1]["link"] == 7
    assert rewritten[0][3:5] == ["plain", 1]


def test_plain_holder_keeps_a_row_when_an_expanded_holder_resolves_nowhere():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {}
    ctx.input_holders[7] = [("10", 0)]
    nodes = [{"id": "plain", "inputs": [{"name": "value", "link": 7}]}]

    rewritten = workflow_to_api._rewrite_links_for_subgraphs([[7, "source", 0, 10, 0, "*"]], ctx, nodes)

    assert rewritten == [[7, "source", 0, "plain", 0, "*"]]


def test_dangling_interior_link_id_is_cleared_before_outer_scope_indexing():
    assert workflow_to_api._rewrite_internal_input(
        {"name": "value", "link": 7},
        internal_link_map={},
        link_id_remap={},
    ) == {"name": "value", "link": None}


def test_string_input_proxy_row_is_not_expanded_as_an_interior_edge():
    definition = {
        "inputs": [{"name": "value", "linkIds": [1]}],
        "outputs": [],
        "nodes": [{"id": 7, "type": "Example", "inputs": [{"name": "value", "link": 1}], "outputs": []}],
        "links": [{"id": 1, "origin_id": "-10", "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }

    nodes, links, input_targets, _outputs = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert links == []
    assert input_targets == {0: [(7, 0)]}
    assert nodes[0]["inputs"][0]["link"] is None


def test_duplicate_output_rows_resolve_each_link_id_once():
    class CountingNode(dict):
        output_reads = 0

        def get(self, key, default=None):
            if key == "outputs":
                type(self).output_reads += 1
            return super().get(key, default)

    count = 200
    node = CountingNode(id=7, type="Example", inputs=[], outputs=[{"name": "value"}])
    definition = {
        "inputs": [],
        "outputs": [{"name": f"out-{index}", "linkIds": [1]} for index in range(count)],
        "nodes": [node],
        "links": [
            {"id": 1, "origin_id": 7, "origin_slot": 0, "target_id": -20, "target_slot": 0} for _ in range(count)
        ],
    }

    _nodes, _links, _inputs, outputs = workflow_to_api._expand_one_subgraph({"id": 10}, definition, [])

    assert len(outputs) == count
    assert CountingNode.output_reads == 1


def test_duplicate_outer_rows_use_the_link_maps_last_source():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [(7, 0)]}
    ctx.input_holders[5] = [("10", 0)]

    workflow_to_api._rewrite_links_for_subgraphs(
        [[5, "first", 0, 10, 0, "*"], [5, "last", 1, 10, 0, "*"]],
        ctx,
        [],
    )

    assert ctx.input_sources["10"][0] == ("last", 1)


def test_input_source_budget_charges_plain_fanout_targets():
    ctx = workflow_to_api._SubgraphCtx()
    ctx.input_targets["10"] = {0: [(7, 0), (8, 0)]}

    with pytest.raises(workflow_to_api.WorkflowConversionError, match="input-source resolution exceeded"):
        workflow_to_api._record_subgraph_input_source("10", 0, ("source", 0), ctx, budget=[2])


def test_promotion_traversal_limit_is_a_structured_conversion_failure(object_info):
    with (
        mock.patch(
            "comfy_cli.cql.promoted.promoted_inputs",
            side_effect=PromotionTraversalLimitError("promoted input traversal exceeded"),
        ),
        pytest.raises(workflow_to_api.WorkflowConversionError, match="promoted input traversal exceeded"),
    ):
        convert_ui_to_api(_load("audio_minimax_music_3.json"), object_info)
