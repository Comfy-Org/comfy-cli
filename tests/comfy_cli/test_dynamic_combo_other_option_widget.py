"""A dynamic-combo sub-widget that only ANOTHER option of the combo reveals.

`workflow set-widget <id>.model.prompt_expansion_mode` on a
MinimaxHailuo03TextToVideoNode (or MinimaxHailuo03FirstLastFrameNode) still on
its default option was refused with:

    widget 'model.prompt_expansion_mode' not found on MinimaxHailuo03TextToVideoNode;
    available: model, model.prompt, model.resolution, model.ratio, model.duration, seed, ...

The name is real. object_info gives `model` three options, and only
"MiniMax H3 Max" and "MiniMax H3 Max Turbo" reveal `prompt_expansion_mode`;
the default "MiniMax H3" does not. `nodes show` lists it under
`dynamic_options` with those keys. So the refusal was right for a node still
on "MiniMax H3", but it listed the current option's widgets and never said
which option has the one asked for, so a caller could only retry the same
write.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph

_SUB = {
    "prompt": ["STRING", {"default": "", "multiline": True}],
    "duration": ["INT", {"default": 5, "min": 4, "max": 15}],
}
_MAX_SUB = {
    **_SUB,
    "prompt_expansion_mode": ["COMBO", {"default": "balanced", "options": ["balanced", "quality"]}],
}

OBJECT_INFO = {
    "MinimaxHailuo03TextToVideoNode": {
        "input": {
            "required": {
                "model": [
                    "COMFY_DYNAMICCOMBO_V3",
                    {
                        "options": [
                            {"key": "MiniMax H3", "inputs": {"required": _SUB}},
                            {"key": "MiniMax H3 Max", "inputs": {"required": _MAX_SUB}},
                            {"key": "MiniMax H3 Max Turbo", "inputs": {"required": _MAX_SUB}},
                        ]
                    },
                ],
                "watermark": ["BOOLEAN", {"default": False}],
            }
        },
        "input_order": {"required": ["model", "watermark"]},
        "output": ["VIDEO"],
        "output_name": ["VIDEO"],
        "category": "partner/video/MiniMax",
        "display_name": "MiniMax H3 Text to Video",
        "description": "",
        "output_node": False,
        "api_node": True,
        "python_module": "comfy_api_nodes",
    }
}


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(OBJECT_INFO)


def _fresh(graph) -> tuple[dict, int]:
    wf = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0, "version": 0.4}
    wf, op = workflow_ops.add_node(wf, graph, "MinimaxHailuo03TextToVideoNode")
    return wf, op["node_id"]


def test_graph_names_the_options_that_reveal_a_sub_widget(graph):
    assert graph.dynamic_sub_widget_options("MinimaxHailuo03TextToVideoNode", "model.prompt_expansion_mode") == (
        "model",
        ["MiniMax H3 Max", "MiniMax H3 Max Turbo"],
    )
    # Present under every option, or not a sub-widget at all: nothing to say.
    assert graph.dynamic_sub_widget_options("MinimaxHailuo03TextToVideoNode", "model.nope") is None
    assert graph.dynamic_sub_widget_options("MinimaxHailuo03TextToVideoNode", "watermark") is None


def test_refusal_names_the_option_to_select_first(graph):
    wf, nid = _fresh(graph)
    with pytest.raises(ValueError) as exc:
        workflow_ops.set_widget(wf, graph, nid, "model.prompt_expansion_mode", "quality")
    msg = str(exc.value)
    assert "'MiniMax H3 Max'" in msg and "'MiniMax H3 Max Turbo'" in msg, msg
    assert "'MiniMax H3'" in msg, f"must name the CURRENT selection: {msg}"
    assert f"{nid}.model" in msg, f"must name the selector address to set first: {msg}"


def test_write_succeeds_once_the_revealing_option_is_selected(graph):
    wf, nid = _fresh(graph)
    wf, _ = workflow_ops.set_widget(wf, graph, nid, "model", "MiniMax H3 Max")
    wf, op = workflow_ops.set_widget(wf, graph, nid, "model.prompt_expansion_mode", "quality")
    assert op["value"] == "quality"


def test_unknown_sub_widget_keeps_the_plain_refusal(graph):
    wf, nid = _fresh(graph)
    with pytest.raises(ValueError) as exc:
        workflow_ops.set_widget(wf, graph, nid, "model.nope", 1)
    assert "not found" in str(exc.value)
    assert "Max Turbo" not in str(exc.value)


def test_set_slot_warning_names_the_revealing_options(graph):
    """`set-slot` warns (without writing) instead of raising; same missing fact."""
    from comfy_cli.cql.engine import _write_widget

    wf, nid = _fresh(graph)
    node = wf["nodes"][0]
    before = list(node["widgets_values"])
    warnings = _write_widget(node, "model.prompt_expansion_mode", "quality", graph, extend=True)
    w = next(w for w in warnings if w["code"] == "unknown_dynamic_sub_input")
    assert w.get("revealed_by") == ["MiniMax H3 Max", "MiniMax H3 Max Turbo"], w
    assert "MiniMax H3 Max Turbo" in w["hint"], w
    assert node["widgets_values"] == before
