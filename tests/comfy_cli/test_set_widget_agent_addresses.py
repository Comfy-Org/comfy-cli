"""set-widget addresses the agent derives from comfy's own output.

Measured on stg-v2/nightly comfy-agent traces (2026-09-20..21):

* ``<instance>/primitive_string_multiline_2.value`` — a ``print_workflow``
  BINDING key (``bindings`` maps it to ``<instance>/18``) used as the node
  part of an address. Refused as "interior node primitive_string_multiline_2
  not found" (22 calls).
* ``51.value`` on a legacy frontend ``PrimitiveNode`` — refused as "widget
  'value' not found on PrimitiveNode; available widgets: (none …)" although a
  write through the node it feeds (``15.text``) already lands on it (8 calls).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_print import render_py

FIXTURES = Path(__file__).parent / "fixtures"


def _load(rel: str) -> dict:
    return json.loads((FIXTURES / rel).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def promoted_graph() -> Graph:
    return Graph.from_object_info(_load("object_info_subgraph_promoted.json"))


def test_set_widget_accepts_a_print_binding_as_the_node(promoted_graph):
    wf = _load("gallery/image_z_image_turbo.json")
    bindings = render_py(wf, promoted_graph).bindings
    assert bindings["57/clip_text_encode"] == "57/27"

    via_binding, _ = workflow_ops.set_widget(
        copy.deepcopy(wf), promoted_graph, "57/clip_text_encode", "text", "a white wolf"
    )
    via_address, _ = workflow_ops.set_widget(copy.deepcopy(wf), promoted_graph, "57/27", "text", "a white wolf")
    # Op ids are minted per call, so compare the graph, not its op bookkeeping.
    assert _graph_only(via_binding) == _graph_only(via_address)


def _graph_only(wf: dict) -> dict:
    return {k: v for k, v in wf.items() if not k.startswith("_")}


def _primitive_canvas() -> dict:
    return {
        "nodes": [
            {
                "id": 51,
                "type": "PrimitiveNode",
                "outputs": [{"name": "STRING", "type": "STRING", "links": [5], "widget": {"name": "text"}}],
                "widgets_values": ["old prompt"],
            },
            {
                "id": 15,
                "type": "CLIPTextEncode",
                "inputs": [
                    {"name": "clip", "type": "CLIP", "link": None},
                    {"name": "text", "type": "STRING", "link": 5, "widget": {"name": "text"}},
                ],
                "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": []}],
                "widgets_values": ["old prompt"],
            },
        ],
        "links": [[5, 51, 0, 15, 1, "STRING"]],
    }


@pytest.mark.parametrize("widget", ["value", "text"])
def test_set_widget_writes_a_legacy_primitive_node_directly(widget, promoted_graph):
    wf, _ = workflow_ops.set_widget(_primitive_canvas(), promoted_graph, 51, widget, "a lighthouse at dusk")
    primitive = next(n for n in wf["nodes"] if n["id"] == 51)
    assert primitive["widgets_values"][0] == "a lighthouse at dusk"
