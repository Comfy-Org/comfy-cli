"""Frontend widget types that keep their input socket still own a slot.

``CreateBoundingBoxes.editor_state`` (``BOUNDING_BOXES``) and
``BuildJsonPromptIdeogram.color_palette`` (``COLORS``) are declared with
``socketless: false``. The frontend still renders a widget for them, because
it has a widget constructor for the type (``widgetStore.inputIsWidget``), and
the widget serializes into ``widgets_values`` like an INT. The input only
keeps a socket as well.

Reading the type as a link dropped ``editor_state`` from the widget order, so
the doc host refused every edit of the box editor (``unknown_widget``), and a
saved ``[width, height, editor_state, last_incoming]`` decoded the boxes into
``last_incoming``.
"""

from __future__ import annotations

from typing import Any

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph
from comfy_cli.cql.widget_catalog import build_types
from comfy_cli.workflow_to_api import convert_ui_to_api

_BOX = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5, "metadata": {"type": "obj", "palette": []}}


def _object_info() -> dict[str, Any]:
    # Shapes as ComfyUI's V3 schema emits them (comfy_extras/nodes_bounding_boxes.py,
    # comfy_extras/nodes_json_prompt.py): `socketless` is only pruned when None.
    return {
        "CreateBoundingBoxes": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}],
                    "height": ["INT", {"default": 1024, "min": 64, "max": 16384, "step": 16}],
                    "editor_state": ["BOUNDING_BOXES", {"default": [], "socketless": False}],
                },
                "optional": {
                    "background": ["IMAGE", {}],
                    "bboxes": ["BOUNDING_BOX,ARRAY,STRING", {}],
                    "last_incoming": ["BOUNDING_BOXES", {"default": [], "socketless": True}],
                },
            },
            "input_order": {
                "required": ["width", "height", "editor_state"],
                "optional": ["background", "bboxes", "last_incoming"],
            },
            "output": ["IMAGE", "BOUNDING_BOX", "ARRAY"],
            "output_name": ["preview", "bboxes", "elements"],
            "output_node": True,
            "display_name": "Create Bounding Boxes",
            "category": "utilities",
            "python_module": "comfy_extras.nodes_bounding_boxes",
        },
        "BuildJsonPromptIdeogram": {
            "input": {
                "required": {
                    "element": ["ARRAY", {}],
                    "medium": ["STRING", {"default": ""}],
                    "color_palette": ["COLORS", {"default": [], "socketless": False}],
                }
            },
            "input_order": {"required": ["element", "medium", "color_palette"]},
            "output": ["DICT"],
            "output_name": ["prompt"],
            "display_name": "Build JSON Prompt (Ideogram)",
            "category": "utilities",
            "python_module": "comfy_extras.nodes_json_prompt",
        },
        # Old-style COLOR input (no socketless key), as ComfyUI_LayerStyle
        # declares it; on Comfy Cloud's object_info today.
        "LayerMask: MaskByColor": {
            "input": {
                "required": {
                    "image": ["IMAGE"],
                    "color": ["COLOR", {"default": "#FFFFFF"}],
                    "color_in_HEX": ["STRING", {"default": ""}],
                    "threshold": ["INT", {"default": 50, "min": 0, "max": 100, "step": 1}],
                    "fix_gap": ["INT", {"default": 2, "min": 0, "max": 32, "step": 1}],
                    "fix_threshold": ["FLOAT", {"default": 0.75, "min": 0.01, "max": 0.99, "step": 0.01}],
                    "invert_mask": ["BOOLEAN", {"default": False}],
                },
                "optional": {"mask": ["MASK"]},
            },
            "input_order": {
                "required": ["image", "color", "color_in_HEX", "threshold", "fix_gap", "fix_threshold", "invert_mask"],
                "optional": ["mask"],
            },
            "output": ["MASK"],
            "output_name": ["mask"],
            "display_name": "LayerMask: Mask By Color",
            "category": "LayerMask",
            "python_module": "custom_nodes.comfyui_layerstyle",
        },
        "BoxSource": {
            "input": {"required": {}},
            "input_order": {"required": []},
            "output": ["BOUNDING_BOXES", "IMAGE"],
            "output_name": ["boxes", "image"],
            "display_name": "Box Source",
            "category": "test",
            "python_module": "custom_nodes.test",
        },
        # forceInput still demotes the same type to a pure link.
        "BoxesFromUpstream": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 64}],
                    "boxes": ["BOUNDING_BOXES", {"socketless": False, "forceInput": True}],
                }
            },
            "input_order": {"required": ["width", "boxes"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "display_name": "Boxes From Upstream",
            "category": "test",
            "python_module": "custom_nodes.test",
        },
    }


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _ui(node_type: str, widgets_values: list[Any]) -> dict[str, Any]:
    return {
        "nodes": [
            {
                "id": 7,
                "type": node_type,
                "inputs": [],
                "outputs": [],
                "mode": 0,
                "widgets_values": widgets_values,
            }
        ],
        "links": [],
    }


class TestSocketedWidgetTypesOwnASlot:
    def test_bounding_boxes_editor_is_in_the_widget_order(self, graph: Graph):
        assert graph.widget_order("CreateBoundingBoxes") == ["width", "height", "editor_state", "last_incoming"]

    def test_catalog_lists_the_box_editor_for_the_doc_host(self, graph: Graph):
        types = build_types(graph)
        assert types["CreateBoundingBoxes"]["widget_order"] == ["width", "height", "editor_state", "last_incoming"]

    def test_colors_input_with_a_socket_is_in_the_widget_order(self, graph: Graph):
        assert graph.widget_order("BuildJsonPromptIdeogram") == ["medium", "color_palette"]

    def test_old_style_color_input_is_in_the_widget_order(self, graph: Graph):
        assert graph.widget_order("LayerMask: MaskByColor") == [
            "color",
            "color_in_HEX",
            "threshold",
            "fix_gap",
            "fix_threshold",
            "invert_mask",
        ]

    def test_saved_mask_by_color_keeps_every_value_in_place(self):
        # widgets_values exactly as ComfyUI_LayerStyle's own example workflow
        # (workflow/mask_by_color_example.json) saves the node.
        ui = _ui("LayerMask: MaskByColor", ["#e64549", "", 100, 2, 0.75, False])
        inputs = convert_ui_to_api(ui, _object_info())["7"]["inputs"]
        assert inputs["color"] == "#e64549"
        assert inputs["color_in_HEX"] == ""
        assert inputs["threshold"] == 100
        assert inputs["fix_gap"] == 2
        assert inputs["fix_threshold"] == 0.75
        assert inputs["invert_mask"] is False

    def test_force_input_still_demotes_it_to_a_link(self, graph: Graph):
        assert graph.widget_order("BoxesFromUpstream") == ["width"]

    def test_saved_boxes_convert_to_editor_state_not_last_incoming(self):
        ui = _ui("CreateBoundingBoxes", [800, 600, [_BOX], []])
        inputs = convert_ui_to_api(ui, _object_info())["7"]["inputs"]
        assert inputs["width"] == 800
        assert inputs["height"] == 600
        # A list literal is wrapped so the server cannot read it as a link,
        # exactly as graphToPrompt does.
        assert inputs["editor_state"] == {"__value__": [_BOX]}
        assert inputs["last_incoming"] == {"__value__": []}

    def test_saved_palette_is_kept(self):
        ui = _ui("BuildJsonPromptIdeogram", ["photograph", ["#ff0000", "#00ff00"]])
        inputs = convert_ui_to_api(ui, _object_info())["7"]["inputs"]
        assert inputs["medium"] == "photograph"
        assert inputs["color_palette"] == {"__value__": ["#ff0000", "#00ff00"]}


class TestSocketedWidgetConnect:
    """These widgets keep a typed socket, so a connect is still type-checked."""

    def _two_nodes(self, graph: Graph) -> tuple[dict[str, Any], Any, Any]:
        wf: dict[str, Any] = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        wf, _ = workflow_ops.add_node(wf, graph, "BoxSource")
        wf, _ = workflow_ops.add_node(wf, graph, "CreateBoundingBoxes")
        return wf, wf["nodes"][0]["id"], wf["nodes"][1]["id"]

    def test_a_matching_output_connects_with_the_declared_socket_type(self, graph: Graph):
        wf, src, dst = self._two_nodes(graph)
        wf, _ = workflow_ops.connect(wf, graph, src, "boxes", dst, "editor_state")
        (entry,) = [i for i in wf["nodes"][1]["inputs"] if i["name"] == "editor_state"]
        assert entry["type"] == "BOUNDING_BOXES"
        assert entry["link"] is not None

    def test_a_mismatched_output_is_refused(self, graph: Graph):
        wf, src, dst = self._two_nodes(graph)
        with pytest.raises(ValueError, match="type mismatch"):
            workflow_ops.connect(wf, graph, src, "image", dst, "editor_state")

    def test_a_fresh_node_carries_the_frontends_empty_boxes(self, graph: Graph):
        wf: dict[str, Any] = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
        defaults = _object_info()
        del defaults["CreateBoundingBoxes"]["input"]["required"]["editor_state"][1]["default"]
        g = Graph.from_object_info(defaults)
        wf, _ = workflow_ops.add_node(wf, g, "CreateBoundingBoxes")
        assert wf["nodes"][0]["widgets_values"][2] == []
