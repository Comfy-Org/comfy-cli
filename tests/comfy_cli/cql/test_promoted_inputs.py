"""Promoted subgraph widgets: the host-owned value model (the "agent editing subgraph" report, scenarios 1/2).

The frontend (``ComfyUI_frontend`` ADR 0009, ``SubgraphNode.ts``) represents a
promoted widget as a *linked subgraph input*: the subgraph definition declares
an input, a boundary link feeds it into an interior node's widget-backed input,
and the HOST instance owns the value — ``widgets_values[i]`` on the instance,
consumed positionally by the i-th subgraph input that resolves to a widget
(``_applyPromotedWidgetValues`` / ``serializeFromStoreState``). Socket-only
inputs (``VIDEO``, ``MODEL``) own no slot. The interior widget is only a
schema/default provider: *"the host/exterior value wins over the
interior/source value during repair, persistence, and prompt serialization."*

Before this model existed in the CLI, every read and write followed the legacy
``properties.proxyWidgets`` list to the interior node — so ``set-widget
57.width 768`` edited a value the frontend never serializes, and ``comfy run``
submitted the interior prompt on post-migration templates whose real prompt
lives on the host.

Fixtures are verbatim gallery templates (``tests/comfy_cli/fixtures/gallery``):

* ``image_z_image_turbo.json`` — pre-migration save (frontend 0.3.73): proxies
  already backed by linked inputs, no host values materialized.
* ``audio_minimax_music_3.json`` — post-migration save: 8 host values that
  differ from the interior defaults (interior caption is ``''``).
* ``api_seedance2_5_video_extend.json`` — mixed: two ``VIDEO`` socket inputs
  (no slot) and four widget inputs with host values.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest import mock

import pytest

from comfy_cli.cql import promoted
from comfy_cli.cql.engine import Graph, _SubgraphDefs

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_GALLERY = _FIXTURES / "gallery"
OBJECT_INFO = _FIXTURES / "object_info_subgraph_promoted.json"

Z_IMAGE_SG = "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"


def _load(name: str) -> dict:
    return json.loads((_GALLERY / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graph() -> Graph:
    return Graph.from_object_info(json.loads(OBJECT_INFO.read_text()))


def _instance(wf: dict, node_id: int) -> dict:
    return next(n for n in wf["nodes"] if n["id"] == node_id)


def _def_of(wf: dict, instance: dict) -> dict:
    return next(d for d in wf["definitions"]["subgraphs"] if d["id"] == instance["type"])


# --------------------------------------------------------------------------- #
# the model: which subgraph inputs own a host value slot, and where they land
# --------------------------------------------------------------------------- #


def test_z_image_every_declared_input_is_a_promoted_widget():
    wf = _load("image_z_image_turbo.json")
    sg = _def_of(wf, _instance(wf, 57))
    pis = promoted.promoted_inputs(sg, promoted.defs_by_id(wf))
    assert [p.name for p in pis] == ["text", "width", "height", "seed", "steps", "unet_name", "clip_name", "vae_name"]
    assert [p.value_index for p in pis] == list(range(8))
    width = next(p for p in pis if p.name == "width")
    assert width.source_node == "13"
    assert width.source_widget == "width"
    assert width.type == "INT"


def test_socket_inputs_own_no_host_slot():
    wf = _load("api_seedance2_5_video_extend.json")
    inst = _instance(wf, 39)
    pis = promoted.promoted_inputs(_def_of(wf, inst), promoted.defs_by_id(wf))
    assert [(p.name, p.value_index) for p in pis] == [
        ("clip_to_resize", None),
        ("base_video", None),
        ("pad_second_video", 0),
        ("interpolation", 1),
        ("padding_color", 2),
        ("drop_audio", 3),
    ]
    # the captured frontend save has exactly one value per widget-backed input
    assert len(inst["widgets_values"]) == len([p for p in pis if p.value_index is not None])


def test_unheld_boundary_row_does_not_promote_a_widget():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [1]}],
        "nodes": [
            {
                "id": 7,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": None}],
            }
        ],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }

    [item] = promoted.promoted_inputs(sg, {"sg": sg})
    assert item.name == "prompt"
    assert item.value_index is None


def test_every_promotion_resolver_skips_an_unheld_row_before_a_live_one():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [1, 2]}],
        "nodes": [
            {
                "id": 7,
                "type": "PromptNode",
                "inputs": [{"name": "stale", "type": "STRING", "widget": {"name": "stale"}, "link": None}],
            },
            {
                "id": 8,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "type": "STRING", "widget": {"name": "prompt"}, "link": 2}],
            },
        ],
        "links": [
            {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0},
            {"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 8, "target_slot": 0},
        ],
    }
    definitions = {"sg": sg}

    [item] = promoted.promoted_inputs(sg, definitions)

    assert item.source_node == "8"
    assert promoted._promotion_source(sg, sg["inputs"][0], definitions) == ("8", "prompt")
    assert promoted.boundary_widget_targets(sg, item, definitions) == [(["8"], "prompt")]


def test_every_promotion_resolver_follows_a_holder_when_the_row_target_drifted():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [2]}],
        "nodes": [
            {"id": 7, "type": "Other", "inputs": [{"name": "other", "link": None}]},
            {
                "id": 8,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "widget": {"name": "prompt"}, "link": 2}],
            },
        ],
        "links": [{"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }
    definitions = {"sg": sg}

    [item] = promoted.promoted_inputs(sg, definitions)

    assert item.source_node == "8"
    assert promoted._promotion_source(sg, sg["inputs"][0], definitions) == ("8", "prompt")
    assert promoted.boundary_widget_targets(sg, item, definitions) == [(["8"], "prompt")]


def test_boundary_targets_keep_every_duplicate_holder_live():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [2]}],
        "nodes": [
            {
                "id": node_id,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "widget": {"name": "prompt"}, "link": 2}],
            }
            for node_id in (7, 8)
        ],
        "links": [{"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }
    definitions = {"sg": sg}

    [item] = promoted.promoted_inputs(sg, definitions)

    assert item.source_node == "7"
    assert promoted.boundary_widget_targets(sg, item, definitions) == [(["7"], "prompt"), (["8"], "prompt")]


@pytest.mark.parametrize("malformed_slot", [True, False, 0.0])
def test_malformed_target_slot_does_not_exact_match_a_holder(malformed_slot):
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [2]}],
        "nodes": [
            {
                "id": 7,
                "type": "PromptNode",
                "inputs": [
                    {"name": "first", "widget": {"name": "first"}, "link": 2},
                    {"name": "second", "widget": {"name": "second"}, "link": 2},
                ],
            }
        ],
        "links": [{"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": malformed_slot}],
    }
    definitions = {"sg": sg}

    [item] = promoted.promoted_inputs(sg, definitions)

    assert item.source_widget == "first"
    assert promoted.boundary_widget_targets(sg, item, definitions) == [(["7"], "first"), (["7"], "second")]


def test_promotion_resolvers_ignore_unhashable_link_ids():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [[], 2]}],
        "nodes": [
            {
                "id": 8,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "widget": {"name": "prompt"}, "link": 2}],
            }
        ],
        "links": [
            {"id": [], "origin_id": -10, "origin_slot": 0, "target_id": 8, "target_slot": 0},
            {"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 8, "target_slot": 0},
        ],
    }
    definitions = {"sg": sg}

    [item] = promoted.promoted_inputs(sg, definitions)

    assert promoted._promotion_source(sg, sg["inputs"][0], definitions) == ("8", "prompt")
    assert promoted.boundary_widget_targets(sg, item, definitions) == [(["8"], "prompt")]


def test_promotion_resolvers_do_not_alias_boolean_link_ids_to_integers():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": [True]}],
        "nodes": [
            {
                "id": 8,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "widget": {"name": "prompt"}, "link": 1}],
            }
        ],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 8, "target_slot": 0}],
    }
    definitions = {"sg": sg}

    [item] = promoted.promoted_inputs(sg, definitions)

    assert item.value_index is None
    assert promoted._promotion_source(sg, sg["inputs"][0], definitions) is None
    assert promoted.boundary_widget_targets(sg, item, definitions) == []


def test_promotion_resolvers_treat_non_list_link_ids_as_empty():
    sg = {
        "id": "sg",
        "inputs": [{"name": "prompt", "type": "STRING", "linkIds": 2}],
        "nodes": [
            {
                "id": 8,
                "type": "PromptNode",
                "inputs": [{"name": "prompt", "widget": {"name": "prompt"}, "link": 2}],
            }
        ],
        "links": [{"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 8, "target_slot": 0}],
    }

    [item] = promoted.promoted_inputs(sg, {"sg": sg})

    assert item.value_index is None
    assert promoted._promotion_source(sg, sg["inputs"][0], {"sg": sg}) is None
    assert promoted.boundary_widget_targets(sg, item, {"sg": sg}) == []


def test_primitive_targets_ignore_unhashable_listed_link_ids():
    primitive = {"id": 7, "outputs": [{"links": [True, [], 2]}]}
    subgraph = {
        "links": [
            {"id": True, "origin_id": 7, "origin_slot": 0, "target_id": 8, "target_slot": 0},
            {"id": [], "origin_id": 7, "origin_slot": 0, "target_id": 8, "target_slot": 0},
            {"id": 2, "origin_id": 7, "origin_slot": 0, "target_id": 9, "target_slot": 1},
        ]
    }

    assert promoted._primitive_targets(subgraph, primitive) == [("9", 1)]


def test_holder_cache_distinguishes_typed_link_rows_with_different_targets():
    sg = {
        "nodes": [
            {"id": 7, "inputs": [{"name": "first", "link": 1}]},
            {"id": 8, "inputs": [{"name": "second", "link": "1"}]},
        ]
    }
    holders = promoted._link_holders(sg)
    first_link = {"id": 1, "target_id": 7, "target_slot": 0}
    second_link = {"id": "1", "target_id": 8, "target_slot": 0}
    cache: dict = {}

    first = promoted.held_link_targets(sg, 1, first_link, holders, cache)
    second = promoted.held_link_targets(sg, "1", second_link, holders, cache)

    assert first[0][0]["id"] == 7
    assert second[0][0]["id"] == 8


def test_nested_fanout_memoizes_repeated_definition_walks():
    definitions: dict[str, dict] = {}
    depth = 10
    width = 4
    for level in reversed(range(depth)):
        definition_id = f"level-{level}"
        child_id = f"level-{level + 1}"
        nodes = []
        links = []
        for index in range(width):
            link_id = index + 1
            node_type = child_id if level + 1 < depth else "PlainNode"
            nodes.append(
                {
                    "id": index,
                    "type": node_type,
                    "inputs": [{"name": "value", "type": "STRING", "link": link_id}],
                }
            )
            links.append({"id": link_id, "origin_id": -10, "origin_slot": 0, "target_id": index, "target_slot": 0})
        definitions[definition_id] = {
            "id": definition_id,
            "inputs": [{"name": "value", "type": "STRING", "linkIds": list(range(1, width + 1))}],
            "nodes": nodes,
            "links": links,
        }

    with mock.patch.object(promoted, "_nested_definition", wraps=promoted._nested_definition) as resolver:
        [item] = promoted.promoted_inputs(definitions["level-0"], definitions)

    assert item.value_index is None
    assert resolver.call_count == depth * width


def test_distinct_nested_paths_are_bounded_by_definition_graph_size():
    definitions: dict[str, dict] = {}
    depth = 18
    for level in reversed(range(depth)):
        for side in ("left", "right"):
            definition_id = f"{side}-{level}"
            next_level = level + 1
            child_types = (
                (f"left-{next_level}", f"right-{next_level}") if next_level < depth else ("PlainLeft", "PlainRight")
            )
            definitions[definition_id] = {
                "id": definition_id,
                "inputs": [{"name": "value", "type": "STRING", "linkIds": [1, 2]}],
                "nodes": [
                    {
                        "id": index,
                        "type": child_type,
                        "inputs": [{"name": "value", "type": "STRING", "link": index + 1}],
                    }
                    for index, child_type in enumerate(child_types)
                ],
                "links": [
                    {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 0, "target_slot": 0},
                    {"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0},
                ],
            }

    with (
        mock.patch.object(promoted, "_nested_definition", wraps=promoted._nested_definition) as resolver,
        pytest.raises(promoted.PromotionTraversalLimitError, match="input traversal"),
    ):
        promoted.promoted_inputs(definitions["left-0"], definitions)

    assert resolver.call_count <= promoted._promotion_visit_limit(definitions, definitions["left-0"]) * 2

    instance = {"id": 7, "type": "left-0", "widgets_values": ["keep"]}
    workflow = {"nodes": [instance], "definitions": {"subgraphs": list(definitions.values())}}
    with pytest.raises(promoted.PromotionTraversalLimitError, match="input traversal"):
        promoted.set_host_value(workflow, instance, "value", "replace", graph=None)
    assert instance["widgets_values"] == ["keep"]


def test_reused_nested_boundary_paths_charge_each_materialized_copy():
    definitions: dict[str, dict] = {
        "leaf": {
            "id": "leaf",
            "inputs": [{"name": "value", "type": "STRING", "linkIds": [1]}],
            "nodes": [
                {
                    "id": 0,
                    "type": "PlainNode",
                    "inputs": [{"name": "value", "link": 1, "widget": {"name": "value"}}],
                }
            ],
            "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 0, "target_slot": 0}],
        }
    }
    for level in reversed(range(20)):
        child = "leaf" if level == 19 else f"level-{level + 1}"
        definitions[f"level-{level}"] = {
            "id": f"level-{level}",
            "inputs": [{"name": "value", "type": "STRING", "linkIds": [1, 2]}],
            "nodes": [
                {"id": index, "type": child, "inputs": [{"name": "value", "link": index + 1}]} for index in range(2)
            ],
            "links": [
                {"id": index + 1, "origin_id": -10, "origin_slot": 0, "target_id": index, "target_slot": 0}
                for index in range(2)
            ],
        }

    pi = promoted.PromotedInput("value", "STRING", index=0, value_index=0)
    with pytest.raises(promoted.PromotionTraversalLimitError, match="boundary traversal"):
        promoted.boundary_widget_targets(definitions["level-0"], pi, definitions)


def test_repeated_boundary_link_ids_share_holder_work_and_consume_budget():
    sg = {
        "id": "sg",
        "inputs": [{"name": "value", "type": "STRING", "linkIds": [1] * 10_000}],
        "nodes": [{"id": 7, "type": "PlainNode", "inputs": [{"name": "value", "link": 1}]}],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }

    with (
        mock.patch.object(promoted, "_is_slot_index", wraps=promoted._is_slot_index) as slot_check,
        pytest.raises(promoted.PromotionTraversalLimitError, match="input traversal"),
    ):
        promoted.promoted_inputs(sg, {"sg": sg})

    assert slot_check.call_count == 1


def test_holder_materialization_is_charged_before_ordering():
    sg = {"nodes": [{"id": node_id, "inputs": [{"name": "value", "link": 1}]} for node_id in range(3)]}
    holders = promoted._link_holders(sg)

    with pytest.raises(promoted.PromotionTraversalLimitError, match="safe limit"):
        promoted.held_link_targets(
            sg,
            1,
            {"target_id": 0, "target_slot": 0},
            holders,
            budget=[2],
        )


def test_holder_cache_hits_charge_each_target_consumed_by_the_caller():
    sg = {"nodes": [{"id": node_id, "inputs": [{"name": "value", "link": 1}]} for node_id in range(3)]}
    holders = promoted._link_holders(sg)
    link = {"id": 1, "target_id": 0, "target_slot": 0}
    cache: dict = {}
    budget = [6]

    first = promoted.held_link_targets(sg, 1, link, holders, cache, budget)
    assert budget == [3]
    assert promoted.held_link_targets(sg, 1, link, holders, cache, budget) is first
    assert budget == [0]
    with pytest.raises(promoted.PromotionTraversalLimitError, match="safe limit"):
        promoted.held_link_targets(sg, 1, link, holders, cache, budget)


def test_promoted_input_memo_hits_use_constant_budget_and_cached_name_index():
    child = {
        "id": "child",
        "inputs": [{"name": f"value-{index}", "type": "STRING", "linkIds": []} for index in range(50)],
        "nodes": [],
        "links": [],
    }
    memo: dict = {}
    budget = [200]

    first = promoted.promoted_inputs(child, {"child": child}, 1, (123,), memo, budget)
    remaining = budget[0]
    second = promoted.promoted_inputs(child, {"child": child}, 1, (123,), memo, budget)

    assert second is first
    assert budget[0] == remaining - 1


def test_wide_reused_definition_does_not_exhaust_the_linear_budget():
    width = 100
    instances = 300
    child = {
        "id": "child",
        "inputs": [{"name": f"value-{index}", "type": "STRING", "linkIds": []} for index in range(width)],
        "nodes": [],
        "links": [],
    }
    outer = {
        "id": "outer",
        "inputs": [{"name": "value", "type": "STRING", "linkIds": list(range(instances))}],
        "nodes": [
            {
                "id": index,
                "type": "child",
                "inputs": [{"name": "value-0", "link": index}],
            }
            for index in range(instances)
        ],
        "links": [
            {"id": index, "origin_id": -10, "origin_slot": 0, "target_id": index, "target_slot": 0}
            for index in range(instances)
        ],
    }
    definitions = _SubgraphDefs()
    definitions.update({"outer": outer, "child": child})

    [item] = promoted.promoted_inputs(outer, definitions)

    assert item.value_index is None


def test_definition_index_caches_limit_and_holder_scans():
    definition = {
        "id": "sg",
        "inputs": [{"name": "value", "type": "STRING", "linkIds": [1]}],
        "nodes": [{"id": 7, "inputs": [{"name": "value", "link": 1}]}],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }
    definitions = _SubgraphDefs()
    definitions["sg"] = definition

    with (
        mock.patch.object(promoted, "_link_holders", wraps=promoted._link_holders) as holder_scan,
        mock.patch.object(promoted, "_link_rows_by_id", wraps=promoted._link_rows_by_id) as link_scan,
    ):
        promoted.promoted_inputs(definition, definitions)
        promoted.promoted_inputs(definition, definitions)

    assert holder_scan.call_count == 1
    assert link_scan.call_count == 1
    assert definitions.promotion_visit_limit is not None


def test_definition_repairs_invalidate_cached_limits_and_holders():
    definition = {
        "id": "sg",
        "inputs": [{"name": "value", "type": "STRING", "linkIds": [1]}],
        "nodes": [{"id": 7, "inputs": [{"name": "value", "link": 1}]}],
        "links": [{"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 7, "target_slot": 0}],
    }
    definitions = _SubgraphDefs()
    definitions["sg"] = definition
    original = promoted._cached_link_holders(definition, definitions)
    original_links = promoted._cached_link_rows(definition, definitions)
    assert promoted._promotion_visit_limit(definitions, definition) > 0

    definition["nodes"].append({"id": 8, "inputs": [{"name": "value", "link": 1}]})
    promoted._invalidate_promotion_caches(definitions, definition)

    assert promoted._cached_link_holders(definition, definitions) is not original
    assert promoted._cached_link_rows(definition, definitions) is not original_links
    assert len(promoted._cached_link_holders(definition, definitions)["1"]) == 2
    assert definitions.promotion_visit_limit is None


def test_promotion_traversal_limit_is_a_value_error_for_command_boundaries():
    assert issubclass(promoted.PromotionTraversalLimitError, ValueError)


def test_boundary_target_fanout_is_bounded_by_definition_graph_size():
    definitions: dict[str, dict] = {}
    depth = 18
    for level in reversed(range(depth)):
        for side in ("left", "right"):
            definition_id = f"{side}-{level}"
            next_level = level + 1
            child_types = (
                (f"left-{next_level}", f"right-{next_level}") if next_level < depth else ("PlainLeft", "PlainRight")
            )
            definitions[definition_id] = {
                "id": definition_id,
                "inputs": [{"name": "value", "type": "STRING", "linkIds": [1, 2]}],
                "nodes": [
                    {
                        "id": index,
                        "type": child_type,
                        "inputs": [
                            {
                                "name": "value",
                                "type": "STRING",
                                "widget": {"name": "value"},
                                "link": index + 1,
                            }
                        ],
                    }
                    for index, child_type in enumerate(child_types)
                ],
                "links": [
                    {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 0, "target_slot": 0},
                    {"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0},
                ],
            }

    root = definitions["left-0"]
    item = promoted.PromotedInput("value", "STRING", 0, 0)

    with pytest.raises(promoted.PromotionTraversalLimitError, match="boundary traversal"):
        promoted.boundary_widget_targets(root, item, definitions)


# --------------------------------------------------------------------------- #
# reads: host value wins, interior is the fallback
# --------------------------------------------------------------------------- #


def test_effective_value_is_the_host_value_when_materialized(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    caption = promoted.effective_value(wf, inst, "caption", graph)
    assert caption.startswith("Global Metadata: Lo-fi hip-hop")
    assert promoted.effective_value(wf, inst, "max_duration", graph) == 60
    # interior default is NOT what the frontend runs
    interior = next(n for n in _def_of(wf, inst)["nodes"] if n["id"] == 13)
    assert interior["widgets_values"][0] == ""


def test_effective_value_falls_back_to_the_interior_widget(graph):
    wf = _load("image_z_image_turbo.json")
    inst = _instance(wf, 57)
    assert inst["widgets_values"] == []
    assert promoted.effective_value(wf, inst, "width", graph) == 1024
    assert promoted.effective_value(wf, inst, "steps", graph) == 8
    assert promoted.effective_value(wf, inst, "unet_name", graph) == "z_image_turbo_bf16.safetensors"


def test_quarantined_host_value_wins_by_name(graph):
    """ADR 0009: a repaired-but-unresolved legacy entry keeps its host value in
    ``proxyWidgetErrorQuarantine``; the frontend reads it before ``widgets_values``."""
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    inst["properties"]["proxyWidgetErrorQuarantine"] = [
        {
            "originalEntry": ["-1", "max_duration"],
            "reason": "missingSourceWidget",
            "hostValue": 12,
            "attemptedAtVersion": 1,
        }
    ]
    assert promoted.effective_value(wf, inst, "max_duration", graph) == 12


# --------------------------------------------------------------------------- #
# writes: materialize the host array in subgraph-input order, never the interior
# --------------------------------------------------------------------------- #


def test_set_host_value_materializes_the_full_array_in_input_order(graph):
    wf = _load("image_z_image_turbo.json")
    inst = _instance(wf, 57)
    before = copy.deepcopy(_def_of(wf, inst))
    promoted.set_host_value(wf, inst, "width", 768, graph)
    assert inst["widgets_values"] == [
        "Latina female with thick wavy hair, harbor boats and pastel houses behind. Breezy seaside light, warm tones, cinematic close-up. ",
        768,
        1024,
        0,
        8,
        "z_image_turbo_bf16.safetensors",
        "qwen_3_4b.safetensors",
        "ae.safetensors",
    ]
    assert _def_of(wf, inst) == before  # interior untouched


def test_set_host_value_updates_one_slot_of_an_existing_array(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    before = list(inst["widgets_values"])
    promoted.set_host_value(wf, inst, "max_duration", 90, graph)
    assert inst["widgets_values"][2] == 90
    assert inst["widgets_values"][:2] == before[:2]
    assert inst["widgets_values"][3:] == before[3:]


def test_set_host_value_rewrites_a_shadowing_quarantine_value(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    inst["properties"]["proxyWidgetErrorQuarantine"] = [
        {
            "originalEntry": ["-1", "max_duration"],
            "reason": "missingSourceWidget",
            "hostValue": 12,
            "attemptedAtVersion": 1,
        }
    ]
    promoted.set_host_value(wf, inst, "max_duration", 90, graph)
    assert promoted.effective_value(wf, inst, "max_duration", graph) == 90


# --------------------------------------------------------------------------- #
# slots: the advertised surface reports the value the frontend will run
# --------------------------------------------------------------------------- #


def test_slots_report_host_values_on_a_post_migration_template(graph):
    wf = _load("audio_minimax_music_3.json")
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert slots["37.caption"]["current_value"].startswith("Global Metadata: Lo-fi hip-hop")
    assert slots["37.max_duration"]["current_value"] == 60
    assert slots["37.switch"]["current_value"] is True


def test_slots_skip_socket_inputs_and_flag_external_links(graph):
    wf = _load("api_seedance2_5_video_extend.json")
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert "39.clip_to_resize" not in slots
    assert slots["39.pad_second_video"]["current_value"] is False
    assert slots["39.interpolation"]["current_value"] == "lanczos"


def test_slots_do_not_report_a_dangling_link_as_linked_from(graph):
    wf = _load("audio_minimax_music_3.json")
    inst = _instance(wf, 37)
    inst["inputs"].append({"name": "caption", "type": "STRING", "widget": {"name": "caption"}, "link": 999999})
    slots = {s["address"]: s for s in graph.get_template_schema("t", wf)["slots"]}
    assert "linked_from" not in slots["37.caption"]
    assert slots["37.caption"]["current_value"].startswith("Global Metadata")
    assert promoted.live_external_link(wf, inst, "caption") is None


def test_effective_value_for_skips_the_name_lookup(graph):
    """``effective_value_for`` is ``effective_value`` for a caller that already
    holds the ``PromotedInput`` and the definition index: host value when
    materialized, else the interior source value — and it never re-walks
    ``promoted_inputs`` (``find_promoted``) to relocate what it was handed."""
    from unittest import mock

    wf = json.loads((_FIXTURES / "gallery" / "image_z_image_turbo.json").read_text())
    inst = next(n for n in wf["nodes"] if n["id"] == 57)
    defs = promoted.defs_by_id(wf)
    sg = defs[inst["type"]]
    width = next(p for p in promoted.promoted_inputs(sg, defs) if p.name == "width")
    with mock.patch.object(promoted, "promoted_inputs", side_effect=AssertionError("re-walked")):
        assert promoted.effective_value_for(wf, inst, sg, width, graph, defs) == 1024  # interior default
    promoted.set_host_value(wf, inst, "width", 768, graph)
    with mock.patch.object(promoted, "promoted_inputs", side_effect=AssertionError("re-walked")):
        assert promoted.effective_value_for(wf, inst, sg, width, graph, defs) == 768  # host wins
    assert promoted.effective_value_for(wf, inst, sg, width, graph, defs) == promoted.effective_value(
        wf, inst, "width", graph
    )
