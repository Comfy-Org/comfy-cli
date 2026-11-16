"""The "ambiguous widget" refusal stays scoped to DynamicGroup rows.

``_write_widget`` has refused a write whose name occupies two positions since
#973, deliberately only ``if has_group`` — a group's rows are the one place a
single requested name can map to two slots. Threading the saved widget layout
into that lookup (#1081) dropped the ``has_group`` condition along the way,
which widened the refusal onto every node.

Outside a group the one name that legitimately repeats is the frontend's
``control_after_generate`` marker: a node carrying both ``seed`` and
``noise_seed`` gets one marker each. It is unadvertised but writable on purpose
(``_widget_index``: "it carries a real user value"), it has no per-widget
address to disambiguate with, and it has resolved to the first occurrence for as
long as it has been writable. The widened check turned that into a hard
``ambiguous widget 'control_after_generate'`` with no way forward, on a node the
saved-layout fix does not otherwise touch.

Threading the saved order in cannot introduce a duplicate on its own:
``saved_sub_order`` caps every name at the number of occurrences the current
catalog declares, and its reorder consumes each sub exactly once.
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph

_TWO_CONTROL_NODES = {
    "TwoSeedNode": {
        "input": {
            "required": {
                "seed": ["INT", {"default": 0}],
                "noise_seed": ["INT", {"default": 0}],
                "steps": ["INT", {"default": 20}],
            }
        },
        "input_order": {"required": ["seed", "noise_seed", "steps"]},
    }
}

_GROUP_FIXTURE = json.loads((pathlib.Path(__file__).parents[1] / "fixtures" / "dynamic_group.json").read_text())


def _two_seed_workflow() -> dict:
    return {
        "nodes": [{"id": 1, "type": "TwoSeedNode", "widgets_values": [1, "fixed", 2, "fixed", 20]}],
        "links": [],
    }


def test_a_node_with_two_control_markers_repeats_the_name():
    """The premise: this is a real duplicate, not a contrived one."""
    graph = Graph.from_object_info(_TWO_CONTROL_NODES)

    order = graph.widget_order_for_node("TwoSeedNode", [1, "fixed", 2, "fixed", 20])

    assert order.count("control_after_generate") == 2, order


def test_the_control_marker_stays_editable_on_such_a_node():
    graph = Graph.from_object_info(_TWO_CONTROL_NODES)

    out, _op = workflow_ops.set_widget(_two_seed_workflow(), graph, 1, "control_after_generate", "randomize")

    # First occurrence, which is where it has always landed.
    assert out["nodes"][0]["widgets_values"] == [1, "randomize", 2, "fixed", 20]


def test_the_named_widgets_on_such_a_node_are_unaffected():
    graph = Graph.from_object_info(_TWO_CONTROL_NODES)

    out, _op = workflow_ops.set_widget(_two_seed_workflow(), graph, 1, "noise_seed", 7)

    assert out["nodes"][0]["widgets_values"] == [1, "fixed", 7, "fixed", 20]


def test_a_dynamic_group_still_refuses_an_ambiguous_name(monkeypatch):
    """The guard #973 added is kept, not deleted — only re-narrowed."""
    object_info = copy.deepcopy(_GROUP_FIXTURE["object_info"])
    workflow = copy.deepcopy(_GROUP_FIXTURE["cases"]["populated"]["workflow"])
    graph = Graph.from_object_info(object_info)
    real = Graph.widget_order_for_node

    def duplicated(self, class_name, widgets_values, saved_widget_names=None):
        order = real(self, class_name, widgets_values, saved_widget_names)
        return [order[0], *order] if order else order

    first = real(graph, "DevToolsNodeWithDynamicGroup", workflow["nodes"][0]["widgets_values"])[0]
    monkeypatch.setattr(Graph, "widget_order_for_node", duplicated)

    with pytest.raises(ValueError, match=f"ambiguous widget '{first}'"):
        workflow_ops.set_widget(workflow, graph, 1, first, "x")


def test_a_plain_node_takes_the_first_occurrence_instead_of_refusing(monkeypatch):
    graph = Graph.from_object_info(_TWO_CONTROL_NODES)
    real = Graph.widget_order_for_node

    def duplicated(self, class_name, widgets_values, saved_widget_names=None):
        order = real(self, class_name, widgets_values, saved_widget_names)
        return [order[0], *order] if order else order

    monkeypatch.setattr(Graph, "widget_order_for_node", duplicated)

    out, _op = workflow_ops.set_widget(_two_seed_workflow(), graph, 1, "seed", 42)

    assert out["nodes"][0]["widgets_values"][0] == 42


_REPEATED_MARKER_NODES = {
    "ComboNode": {
        "input": {
            "required": {
                "model": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {
                                "key": "A",
                                "inputs": {
                                    "required": {
                                        "prompt": ["STRING", {"default": ""}],
                                        "resolution": [["480p", "720p"], {"default": "720p"}],
                                    }
                                },
                            }
                        ]
                    },
                ],
                "watermark": ["BOOLEAN", {"default": False}],
            }
        },
        "input_order": {"required": ["model", "watermark"]},
    }
}


def test_a_duplicated_saved_marker_does_not_duplicate_a_position():
    """Why narrowing the check is safe: the saved order cannot manufacture one.

    ``saved_sub_order`` takes a repeated marker once, capped at the number of
    occurrences the catalog declares, so threading the saved layout into the
    lookup cannot turn a single-position name into an ambiguous one.
    """
    graph = Graph.from_object_info(_REPEATED_MARKER_NODES)
    node = {
        "id": 1,
        "type": "ComboNode",
        "widgets_values": ["A", "p", "720p", False],
        "inputs": [
            {"name": name, "widget": {"name": name}}
            # `model.prompt` marked twice, as a corrupt or hand-edited save could.
            for name in ("model", "model.prompt", "model.prompt", "model.resolution", "watermark")
        ],
    }

    order = graph.widget_order_for_node(
        "ComboNode", node["widgets_values"], ["model", "model.prompt", "model.prompt", "model.resolution", "watermark"]
    )

    assert len(order) == len(set(order)), order
    out, _op = workflow_ops.set_widget({"nodes": [node], "links": []}, graph, 1, "model.prompt", "edited")
    assert out["nodes"][0]["widgets_values"] == ["A", "edited", "720p", False]
