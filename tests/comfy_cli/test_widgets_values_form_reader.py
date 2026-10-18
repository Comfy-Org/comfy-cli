"""Reading and editing a node that declares its own ``widgets_values_form``.

The frontend now declares a node's widget layout (comfy-multi-player schema
A24) when a frontend extension adds a value widget the node definition does not
describe. The canonical case is rgthree's ``Display Int``: its only
input is a ``forceInput`` link, so the catalog lists no widgets, but the
extension draws its result in an ``output`` text widget. The doc host stores
such a node with its own layout and projects it back as a name-keyed
``widgets_values`` plus the declaration, so this is what the agent reads.

These pin that the agent's reader and writer handle that projection: the
frontend-only value is never offered as an editable slot, printing does not
treat the class as unknown, and a schema widget write keeps the node's declared
layout instead of flattening it into a positional array the declaration no
longer describes.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph, _extract_frontend_slots
from comfy_cli.workflow_print import render_py

DISPLAY_INT = "Display Int (rgthree)"

_OBJECT_INFO = {
    DISPLAY_INT: {
        "input": {"required": {"input": ["INT", {"forceInput": True}]}},
        "input_order": {"required": ["input"]},
        "output": ["INT"],
        "output_name": ["INT"],
        "category": "rgthree",
        "display_name": DISPLAY_INT,
        "python_module": "custom_nodes.rgthree-comfy",
        "output_node": True,
    },
    "EmptyLatentImage": {
        "input": {
            "required": {
                "width": ["INT", {"default": 512}],
                "height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            },
        },
        "input_order": {"required": ["width", "height", "batch_size"]},
        "output": ["LATENT"],
        "output_name": ["LATENT"],
        "category": "latent",
        "display_name": "Empty Latent Image",
        "python_module": "nodes",
    },
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _display_int() -> dict:
    return {
        "id": 5,
        "type": DISPLAY_INT,
        "inputs": [{"name": "input", "type": "INT", "link": None}],
        "outputs": [{"name": "INT", "type": "INT", "links": []}],
        "widgets_values": {"output": "42"},
        "widgets_values_form": {"order": ["output"]},
    }


def _latent_with_extension_preview() -> dict:
    # A described class whose extension added one widget of its own, so the
    # frontend declared the whole layout.
    return {
        "id": 7,
        "type": "EmptyLatentImage",
        "inputs": [],
        "outputs": [{"name": "LATENT", "type": "LATENT", "links": []}],
        "widgets_values": {"width": 512, "height": 512, "batch_size": 1, "preview": "x"},
        "widgets_values_form": {"order": ["width", "height", "batch_size", "preview"]},
    }


def _workflow(*nodes: dict) -> dict:
    return {"nodes": list(nodes), "links": []}


class TestReadsADeclaredForceInputNode:
    def test_offers_no_editable_slot_for_the_frontend_only_widget(self, graph: Graph):
        slots = _extract_frontend_slots(_workflow(_display_int()), graph)
        assert [s for s in slots if s["instance_id"] == "5"] == []

    def test_prints_the_node_as_a_known_class(self, graph: Graph):
        res = render_py(_workflow(_display_int()), graph)
        assert DISPLAY_INT in res.source
        assert "output" not in res.source
        assert not [w for w in res.warnings if "not in catalog" in w]

    def test_refuses_a_write_to_the_frontend_only_widget(self, graph: Graph):
        with pytest.raises(ValueError, match="all inputs are links"):
            W.set_widget(_workflow(_display_int()), graph, 5, "output", "7")


class TestWritesADeclaredNode:
    def test_reads_the_declared_values_at_their_schema_slots(self, graph: Graph):
        slots = _extract_frontend_slots(_workflow(_latent_with_extension_preview()), graph)
        assert {s["name"]: s["current_value"] for s in slots} == {"width": 512, "height": 512, "batch_size": 1}

    def test_keeps_the_declared_layout_and_the_extension_value(self, graph: Graph):
        new_wf, op = W.set_widget(_workflow(_latent_with_extension_preview()), graph, 7, "batch_size", 4)

        assert op["op"] == "set_widget"
        assert (op["widget"], op["value"]) == ("batch_size", 4)
        node = new_wf["nodes"][0]
        assert node["widgets_values"] == {"width": 512, "height": 512, "batch_size": 4, "preview": "x"}
        assert node["widgets_values_form"] == {"order": ["width", "height", "batch_size", "preview"]}
