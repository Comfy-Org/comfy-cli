"""PM-2003: a node whose frontend extension adds a widget object_info never declares.

rgthree's "Display Int (rgthree)" has one ``forceInput`` INT input, so its
catalog ``widget_order`` is empty, but the extension mounts a client-only
``output`` text widget that serializes a value. The frontend fix declares that
layout as A24 ``widgets_values_form``, and the doc host projects the node back
as ``widgets_values: [""]`` plus the form. These tests pin how the agent's own
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


def test_agent_printer_renders_a_self_described_extension_widget_node_like_a_saved_one():
    graph = Graph.from_object_info(OBJECT_INFO)
    declared = _projected_workflow({"widgets_values": [""], "widgets_values_form": {"order": ["output"]}})
    saved = _projected_workflow({"widgets_values": [""]})

    declared_result = render_py(declared, graph)
    saved_result = render_py(saved, graph)

    assert declared_result.skipped == []
    assert declared_result.warnings == []
    assert declared_result.source == 'display_int_rgthree = Node["Display Int (rgthree)"](input=None)  # 7\n'
    assert declared_result.source == saved_result.source


CONTROLLED_COMBO = "Controlled Combo"

CONTROLLED_COMBO_OBJECT_INFO = {
    CONTROLLED_COMBO: {
        "input": {
            "required": {
                "choice": [["a", "b"], {"control_after_generate": True}],
                "steps": ["INT", {"default": 20}],
            }
        },
        "input_order": {"required": ["choice", "steps"]},
        "output": [],
        "output_is_list": [],
        "output_name": [],
        "name": CONTROLLED_COMBO,
        "display_name": CONTROLLED_COMBO,
        "description": "",
        "python_module": "nodes",
        "category": "test",
        "output_node": True,
    }
}


def _controlled_combo_workflow(node_extra: dict) -> dict:
    node = {
        "id": 1,
        "type": CONTROLLED_COMBO,
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


@pytest.mark.xfail(
    strict=True,
    reason="FE-3470: comfy-cli reads widgets_values by catalog position and ignores widgets_values_form, so the "
    "frontend only declares a form for fixed-layout nodes whose undeclared widgets trail the declared ones",
)
def test_agent_printer_reads_a_declared_form_by_name_when_an_undeclared_widget_precedes_a_declared_one():
    graph = Graph.from_object_info(CONTROLLED_COMBO_OBJECT_INFO)
    workflow = _controlled_combo_workflow(
        {
            "widgets_values": ["a", "randomize", "", 20],
            "widgets_values_form": {"order": ["choice", "control_after_generate", "control_filter_list", "steps"]},
        }
    )

    result = render_py(workflow, graph)

    assert "steps=20" in result.source


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


def test_agent_printer_reads_the_catalog_prefix_of_a_declared_trailing_extension_widget():
    graph = Graph.from_object_info(STEPS_EXTENSION_OBJECT_INFO)
    workflow = _controlled_combo_workflow(
        {"type": STEPS_EXTENSION, "widgets_values": [7, ""], "widgets_values_form": {"order": ["steps", "output"]}}
    )

    result = render_py(workflow, graph)

    assert result.warnings == []
    assert "steps=7" in result.source
