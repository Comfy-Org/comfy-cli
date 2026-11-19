"""A node whose frontend extension adds a widget object_info never declares.

rgthree's "Display Int (rgthree)" has one ``forceInput`` INT input, so its
catalog ``widget_order`` is empty, but the extension mounts a client-only
``output`` text widget that serializes a value. The frontend fix sends the
node's name-keyed values plus A24 ``widgets_values_form``, and the doc host
projects them back verbatim. The printer resolves name-keyed values through the
catalog order and, for classes without a DynamicGroup input, does not read the form, so a value the catalog cannot name is
dropped rather than shifted onto a catalogued widget. These tests pin how the agent's own
readers treat that projection, so the frontend change cannot silently break
what the agent sees.
"""

from __future__ import annotations

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_print import render_py

DISPLAY_INT = "Display Int (rgthree)"

OBJECT_INFO = {
    DISPLAY_INT: {
        "input": {"required": {"input": ["INT", {"forceInput": True}]}},
        "input_order": {"required": ["input"]},
        "output": [],
        "output_is_list": [],
        "output_name": [],
        "name": DISPLAY_INT,
        "display_name": DISPLAY_INT,
        "description": "",
        "python_module": "custom_nodes.rgthree-comfy",
        "category": "rgthree",
        "output_node": True,
    }
}


def _projected_workflow(node_extra: dict) -> dict:
    node = {
        "id": 7,
        "type": DISPLAY_INT,
        "pos": [0, 0],
        "size": [210, 80],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": "input", "type": "INT", "link": None}],
        "outputs": [],
        **node_extra,
    }
    return {"last_node_id": 7, "last_link_id": 0, "nodes": [node], "links": [], "version": 0.4}


def test_catalog_widget_order_for_a_force_input_class_is_empty():
    graph = Graph.from_object_info(OBJECT_INFO)

    assert graph.widget_order_default(DISPLAY_INT) == []


def test_agent_printer_renders_a_declared_extension_widget_node_like_a_saved_one():
    graph = Graph.from_object_info(OBJECT_INFO)
    declared = _projected_workflow({"widgets_values": {"output": ""}, "widgets_values_form": {"order": ["output"]}})
    saved = _projected_workflow({"widgets_values": [""]})

    declared_result = render_py(declared, graph)
    saved_result = render_py(saved, graph)

    assert declared_result.skipped == []
    assert declared_result.warnings == []
    assert declared_result.source == 'display_int_rgthree = Node["Display Int (rgthree)"](input=None)  # 7\n'
    assert declared_result.source == saved_result.source


def _single_node_workflow(node_extra: dict) -> dict:
    node = {
        "id": 1,
        "pos": [0, 0],
        "size": [210, 130],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [],
        "outputs": [],
        **node_extra,
    }
    return {"last_node_id": 1, "last_link_id": 0, "nodes": [node], "links": [], "version": 0.4}


STEPS_EXTENSION = "Steps With Extension Widget"

STEPS_EXTENSION_OBJECT_INFO = {
    STEPS_EXTENSION: {
        "input": {"required": {"steps": ["INT", {"default": 20}]}},
        "input_order": {"required": ["steps"]},
        "output": [],
        "output_is_list": [],
        "output_name": [],
        "name": STEPS_EXTENSION,
        "display_name": STEPS_EXTENSION,
        "description": "",
        "python_module": "nodes",
        "category": "test",
        "output_node": True,
    }
}


@pytest.mark.parametrize(
    "widgets_values, order",
    [
        ({"steps": 7, "output": ""}, ["steps", "output"]),
        ({"output": "", "steps": 7}, ["output", "steps"]),
    ],
    ids=["catalogued-first", "extension-first"],
)
def test_agent_printer_reads_a_declared_extension_widget_beside_a_catalogued_one(widgets_values, order):
    graph = Graph.from_object_info(STEPS_EXTENSION_OBJECT_INFO)
    workflow = _single_node_workflow(
        {"type": STEPS_EXTENSION, "widgets_values": widgets_values, "widgets_values_form": {"order": order}}
    )

    result = render_py(workflow, graph)

    assert result.warnings == []
    assert result.skipped == []
    assert result.source == 'steps_with_extension_widget = Node["Steps With Extension Widget"](steps=7)  # 1\n'
