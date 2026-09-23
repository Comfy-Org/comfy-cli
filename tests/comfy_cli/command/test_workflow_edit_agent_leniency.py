"""Edit shapes agents send that the CLI refused although their meaning is unambiguous.

Every case here is a real comfy-agent tool input from prod/staging Langfuse
traces (2026-09-22 → 09-23). Each failure cost the agent a round-trip; the
retry that followed was the same edit re-spelled, so the CLI accepts the
first spelling instead.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_OBJECT_INFO = _FIXTURES / "object_info_subgraph_promoted.json"

# The real MeshyTextToModelNode schema (cloud object_info, 2026-09): its
# `should_remesh` is a dynamic combo whose two option keys are the STRINGS
# "true" / "false".
_MESHY = {
    "input": {
        "required": {
            "model": ["COMBO", {"multiselect": False, "options": ["meshy-7", "meshy-6", "latest"]}],
            "prompt": ["STRING", {"default": "", "multiline": True}],
            "style": ["COMBO", {"multiselect": False, "options": ["realistic"]}],
            "should_remesh": [
                "COMFY_DYNAMICCOMBO_V3",
                {
                    "options": [
                        {
                            "key": "true",
                            "inputs": {
                                "required": {
                                    "topology": ["COMBO", {"multiselect": False, "options": ["triangle", "quad"]}],
                                    "target_polycount": ["INT", {"default": 300000, "min": 100, "max": 300000}],
                                }
                            },
                        },
                        {"key": "false", "inputs": {"required": {}}},
                    ]
                },
            ],
            "seed": ["INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}],
        }
    },
    "input_order": {"required": ["model", "prompt", "style", "should_remesh", "seed"]},
    "output": ["STRING", "MESHY_TASK_ID", "FILE_3D_GLB", "FILE_3D_FBX"],
    "output_is_list": [False, False, False, False],
    "output_name": ["model_file", "meshy_task_id", "GLB", "FBX"],
    "name": "MeshyTextToModelNode",
    "display_name": "Meshy: Text to Model",
    "category": "partner/3d/Meshy",
    "output_node": True,
    "api_node": True,
}

# A combo whose options are "true"/"false" but NOT a dynamic combo, plus one
# whose two options merely read as a toggle — the bool mapping must not reach it.
_BOOLISH = {
    "input": {
        "required": {
            "flag": ["COMBO", {"options": ["true", "false"]}],
            "tri": ["COMBO", {"options": ["on", "off"]}],
        }
    },
    "input_order": {"required": ["flag", "tri"]},
    "output": ["STRING"],
    "output_is_list": [False],
    "output_name": ["STRING"],
    "name": "BoolishCombos",
    "display_name": "Boolish Combos",
    "category": "test",
}


@pytest.fixture(scope="module")
def object_info() -> dict[str, Any]:
    info = json.loads(_OBJECT_INFO.read_text())
    info["MeshyTextToModelNode"] = _MESHY
    info["BoolishCombos"] = _BOOLISH
    return info


@pytest.fixture(scope="module")
def graph(object_info) -> Graph:
    return Graph.from_object_info(object_info)


def _empty() -> dict[str, Any]:
    return {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}


def _node(workflow: dict, node_id: Any) -> dict:
    return next(n for n in workflow["nodes"] if n["id"] == node_id)


# ---------------------------------------------------------------------------
# 1. add_node `at` given as an "x,y" string
# ---------------------------------------------------------------------------


class TestPositionString:
    # Trace dbe5e3c6 (prod): {"op":"add_node","class_type":"LoadVideo","as":"load1","at":"40,90"}
    # Trace 65b781cb (stg):  {"op":"add_node","class_type":"UNETLoader","as":"unet","at":"0,0"}
    @pytest.mark.parametrize(
        ("at", "pos"),
        [("40,90", [40, 90]), ("0,0", [0, 0]), ("[430, 90]", [430, 90]), (" -12.5 , 7 ", [-12.5, 7])],
    )
    def test_batch_accepts_string_position(self, graph, at, pos):
        workflow, ops, aliases = workflow_ops.apply_specs(
            _empty(), graph, [{"op": "add_node", "class_type": "LoadVideo", "as": "load1", "at": at}]
        )
        assert ops[0]["pos"] == pos
        assert _node(workflow, aliases["load1"])["pos"] == pos

    def test_string_position_is_frozen_as_numbers_so_replay_converges(self, graph):
        workflow, ops, _ = workflow_ops.apply_specs(
            _empty(), graph, [{"op": "add_node", "class_type": "UNETLoader", "at": "0,0"}]
        )
        assert all(isinstance(v, int) for v in ops[0]["pos"])
        replayed = workflow_ops.apply_op(_empty(), ops[0], graph)
        assert workflow_ops.canonical(replayed) == workflow_ops.canonical(workflow)

    def test_string_position_pins_the_node_among_auto_placed_siblings(self, graph):
        """Layout treats a pinned `at` as an obstacle — a string must not
        crash it when the batch also has nodes that need a position."""
        workflow, ops, _ = workflow_ops.apply_specs(
            _empty(),
            graph,
            [
                {"op": "add_node", "class_type": "UNETLoader", "at": "0,0"},
                {"op": "add_node", "class_type": "LoadVideo"},
            ],
        )
        assert ops[0]["pos"] == [0, 0]
        assert ops[1]["pos"] != [0, 0]

    @pytest.mark.parametrize("at", ["40", "40,90,10", "x,y", "nan,0", "0,inf", "", "40;90"])
    def test_malformed_string_position_still_rejected(self, graph, at):
        with pytest.raises(ValueError, match="node position must be two finite numbers"):
            workflow_ops.apply_specs(_empty(), graph, [{"op": "add_node", "class_type": "LoadVideo", "at": at}])


# ---------------------------------------------------------------------------
# 2. bool written to a combo whose options are exactly 'true' / 'false'
# ---------------------------------------------------------------------------


class TestBoolToTrueFalseCombo:
    # Trace c9552f9f (stg): {"op":"set_widget","node":"$gen","widget":"should_remesh","value":true}
    def test_batch_maps_json_true_to_dynamic_combo_option(self, graph):
        workflow, ops, aliases = workflow_ops.apply_specs(
            _empty(),
            graph,
            [
                {"op": "add_node", "class_type": "MeshyTextToModelNode", "as": "gen"},
                {"op": "set_widget", "node": "$gen", "widget": "should_remesh", "value": True},
                {"op": "set_widget", "node": "$gen", "widget": "should_remesh.topology", "value": "quad"},
            ],
        )
        assert ops[1]["value"] == "true"
        assert any(w.get("code") == "normalized_value" for w in ops[1].get("warnings", []))
        assert "quad" in _node(workflow, aliases["gen"])["widgets_values"]

    def test_false_maps_to_false_option(self, graph):
        workflow, add = workflow_ops.add_node(_empty(), graph, "MeshyTextToModelNode")
        workflow, op = workflow_ops.set_widget(workflow, graph, add["node_id"], "should_remesh", False)
        assert op["value"] == "false"
        assert "false" in _node(workflow, add["node_id"])["widgets_values"]

    def test_plain_true_false_combo_also_maps(self, graph):
        workflow, add = workflow_ops.add_node(_empty(), graph, "BoolishCombos")
        _, op = workflow_ops.set_widget(workflow, graph, add["node_id"], "flag", True)
        assert op["value"] == "true"

    def test_bool_not_mapped_when_options_are_not_exactly_true_false(self, graph):
        workflow, add = workflow_ops.add_node(_empty(), graph, "BoolishCombos")
        with pytest.raises(ValueError, match="got bool"):
            workflow_ops.set_widget(workflow, graph, add["node_id"], "tri", True)

    def test_string_true_still_accepted_unchanged(self, graph):
        workflow, add = workflow_ops.add_node(_empty(), graph, "MeshyTextToModelNode")
        _, op = workflow_ops.set_widget(workflow, graph, add["node_id"], "should_remesh", "true")
        assert op["value"] == "true"
        assert not any(w.get("code") == "normalized_value" for w in op.get("warnings", []))
