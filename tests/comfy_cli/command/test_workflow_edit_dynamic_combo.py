"""Regression coverage for name-keyed edits that change a dynamic-combo schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from comfy_cli import workflow_ops, workflow_to_api
from comfy_cli.cql.engine import Graph

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_OBJECT_INFO = _FIXTURES / "object_info_subgraph_promoted.json"
_SEEDANCE_WORKFLOW = _FIXTURES / "gallery" / "api_seedance2_5_video_extend.json"
_SEEDANCE_NODE_ID = 38

_TRAILING_VALUES = [1672978219, "fixed", False]
_SEEDANCE_20_DEFAULTS = [
    "Seedance 2.0",
    "",
    "480p",
    "adaptive",
    7,
    True,
    True,
    False,
    *_TRAILING_VALUES,
]
_SEEDANCE_25_DEFAULTS = [
    "Seedance 2.5",
    "",
    "720p",
    "16:9",
    5,
    True,
    "auto",
    "mp4",
    True,
    False,
    *_TRAILING_VALUES,
]


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    return json.loads(_OBJECT_INFO.read_text())


@pytest.fixture(scope="module")
def graph(object_info) -> Graph:
    return Graph.from_object_info(object_info)


def _seedance_workflow() -> dict[str, Any]:
    return json.loads(_SEEDANCE_WORKFLOW.read_text())


def _seedance_values(workflow: dict[str, Any]) -> list[Any]:
    return next(node["widgets_values"] for node in workflow["nodes"] if node["id"] == _SEEDANCE_NODE_ID)


def test_set_widget_reshapes_seedance_roster_in_both_directions(graph):
    base = _seedance_workflow()
    changed, op_20 = workflow_ops.set_widget(copy.deepcopy(base), graph, _SEEDANCE_NODE_ID, "model", "Seedance 2.0")

    assert _seedance_values(changed) == _SEEDANCE_20_DEFAULTS
    assert [warning["code"] for warning in op_20["warnings"]] == ["dynamic_combo_roster_rebuilt"]
    replayed = workflow_ops.apply_op(copy.deepcopy(base), op_20, graph)
    assert workflow_ops.canonical(replayed) == workflow_ops.canonical(changed)

    restored, op_25 = workflow_ops.set_widget(
        changed, graph, _SEEDANCE_NODE_ID, "model", "Seedance 2.5", base_version=1
    )
    assert _seedance_values(restored) == _SEEDANCE_25_DEFAULTS
    assert [warning["code"] for warning in op_25["warnings"]] == ["dynamic_combo_roster_rebuilt"]


def test_agent_add_and_name_keyed_writes_copy_seedance_20_values(graph, object_info):
    source_values = [
        "Seedance 2.0",
        "copied prompt",
        "4k",
        "9:16",
        11,
        False,
        False,
        True,
        42,
        "fixed",
        True,
    ]
    workflow, add_op = workflow_ops.add_node(
        {"nodes": [], "links": [], "groups": [], "last_node_id": 0, "last_link_id": 0},
        graph,
        "ByteDance2ReferenceNodeV2",
    )
    node_id = add_op["node_id"]
    copied = {
        "model": source_values[0],
        "model.prompt": source_values[1],
        "model.resolution": source_values[2],
        "model.ratio": source_values[3],
        "model.duration": source_values[4],
        "model.generate_audio": source_values[5],
        "model.auto_downscale": source_values[6],
        "model.auto_upscale": source_values[7],
        "seed": source_values[8],
        "watermark": source_values[10],
    }
    for widget, value in copied.items():
        workflow, _ = workflow_ops.set_widget(workflow, graph, node_id, widget, value)

    target = next(node for node in workflow["nodes"] if node["id"] == node_id)
    assert target["widgets_values"] == source_values
    api_inputs = workflow_to_api.convert_ui_to_api(workflow, object_info)[str(node_id)]["inputs"]
    assert api_inputs["seed"] == 42
    assert "model.task_type" not in api_inputs
    assert "model.output_format" not in api_inputs


def test_selector_conflicts_with_owned_sub_widget_but_not_trailing_seed(graph):
    base = _seedance_workflow()
    _, selector = workflow_ops.set_widget(
        copy.deepcopy(base), graph, _SEEDANCE_NODE_ID, "model", "Seedance 2.0", actor="selector"
    )
    _, prompt = workflow_ops.set_widget(
        copy.deepcopy(base), graph, _SEEDANCE_NODE_ID, "model.prompt", "concurrent", actor="prompt"
    )
    _, seed = workflow_ops.set_widget(copy.deepcopy(base), graph, _SEEDANCE_NODE_ID, "seed", 42, actor="seed")

    assert workflow_ops.detect_conflict(selector, prompt) is True
    assert workflow_ops.detect_conflict(selector, seed) is False
    selector_seed = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), selector, graph), seed, graph)
    seed_selector = workflow_ops.apply_op(workflow_ops.apply_op(copy.deepcopy(base), seed, graph), selector, graph)
    assert workflow_ops.canonical(selector_seed) == workflow_ops.canonical(seed_selector)
    assert _seedance_values(selector_seed)[-3:] == [42, "fixed", False]


@pytest.mark.parametrize("outer_type", ["COMFY_DYNAMICCOMBO_V3", "COMBO"])
@pytest.mark.parametrize("inner_type", ["COMFY_DYNAMICCOMBO_V3", "COMBO"])
def test_structural_selector_switch_preserves_nested_roster_and_suffix(outer_type, inner_type):
    # Regression: reading structural selectors must agree with rebuilding their
    # values on write. https://github.com/Comfy-Org/comfy-cli/pull/914
    quality = [
        inner_type,
        {
            "options": [
                {"key": "fine", "inputs": {"required": {"detail": ["INT", {"default": 17}]}}},
                {"key": "plain", "inputs": {"required": {}}},
            ]
        },
    ]
    required = {
        "before": ["INT", {"default": 0}],
        "mode": [
            outer_type,
            {
                "options": [
                    {"key": "simple", "inputs": {"required": {"score": ["INT", {"default": 11}]}}},
                    {"key": "expanded", "inputs": {"required": {"quality": quality}}},
                ]
            },
        ],
        "after": ["INT", {"default": 0}],
    }
    graph = Graph.from_object_info(
        {"SelectorNode": {"input": {"required": required}, "input_order": {"required": list(required)}, "output": []}}
    )
    base = {
        "nodes": [{"id": 1, "type": "SelectorNode", "widgets_values": [91, "simple", 23, 42]}],
        "links": [],
    }

    expanded, op = workflow_ops.set_widget(copy.deepcopy(base), graph, 1, "mode", "expanded")
    assert expanded["nodes"][0]["widgets_values"] == [91, "expanded", "fine", 17, 42]
    assert [warning["code"] for warning in op["warnings"]] == ["dynamic_combo_roster_rebuilt"]
    replayed = workflow_ops.apply_op(copy.deepcopy(base), op, graph)
    assert workflow_ops.canonical(replayed) == workflow_ops.canonical(expanded)

    edited, _ = workflow_ops.set_widget(expanded, graph, 1, "mode.quality.detail", 63, base_version=1)
    unchanged, _ = workflow_ops.set_widget(edited, graph, 1, "mode", "expanded", base_version=2)
    assert unchanged["nodes"][0]["widgets_values"] == [91, "expanded", "fine", 63, 42]

    plain, _ = workflow_ops.set_widget(unchanged, graph, 1, "mode.quality", "plain", base_version=3)
    assert plain["nodes"][0]["widgets_values"] == [91, "expanded", "plain", 42]
    fine, _ = workflow_ops.set_widget(plain, graph, 1, "mode.quality", "fine", base_version=4)
    assert fine["nodes"][0]["widgets_values"] == [91, "expanded", "fine", 17, 42]

    simple, _ = workflow_ops.set_widget(fine, graph, 1, "mode", "simple", base_version=5)
    assert simple["nodes"][0]["widgets_values"] == [91, "simple", 11, 42]
