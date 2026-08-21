"""``widgets_values`` may be a NAMED DICT, not a positional list.

VideoHelperSuite's ``VHS_*`` nodes (e.g. ``VHS_LoadVideo``) serialize
``widgets_values`` as a dict:

    "widgets_values": {"video": "wan22-...mp4", "force_rate": 0, ...}

Every widget-slot call site in ``cql/engine.py`` and ``workflow_ops.py`` reads
``node.get("widgets_values") or []`` and then indexes the result by INTEGER
position against the schema's widget order. A non-empty dict is truthy, so the
``or []`` guard never fires, and indexing it with an int raises ``KeyError``
(or ``.extend()`` raises ``AttributeError`` when the write path needs to grow
it).

Prod: 38 failures. Surfaced to the agent as the useless message
"Could not extract slots: 0" (``comfy_cli/command/workflow.py``'s
``except (ValueError, KeyError)`` renders ``str(e)``, and ``str(KeyError(0))``
is just ``"0"``). Reproduced directly against the real failing fixture:

    comfy workflow slots template_purz_wan22_animate_auto_character_replace/workflow.json \
        --input object_info.json
    # -> KeyError: 0 at cql/engine.py's _node_widget_slots

``workflow_to_api.py`` already tolerates non-list ``widgets_values``
(``test_tolerates_non_list_widgets_values`` treats it as no known values); this
fix brings ``cql/engine.py`` and ``workflow_ops.py`` into agreement via one
shared helper, ``_widgets_as_list`` — same semantics: non-list (including a
dict) reads as an empty list of positional values, so the node's widgets show
as unset instead of crashing extraction or a write.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops as W
from comfy_cli.cql.engine import Graph, _apply_one_slot, _extract_frontend_slots

_OBJECT_INFO = {
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


def _dict_widget_node(widgets_values: dict) -> dict:
    return {"nodes": [{"id": 7, "type": "EmptyLatentImage", "widgets_values": widgets_values}], "links": []}


class TestListSlotsToleratesDictWidgetsValues:
    def test_full_dict_does_not_crash(self, graph: Graph):
        # One dict entry per real widget — the shape that used to KeyError at
        # `widgets[idx] if idx < len(widgets) else None` (idx is an int, widgets
        # is a dict).
        wf = _dict_widget_node({"width": 512, "height": 512, "batch_size": 1})
        slots = _extract_frontend_slots(wf, graph)
        names = {s["name"] for s in slots}
        assert names == {"width", "height", "batch_size"}
        # The named-dict values are projected onto their schema positions —
        # slots show the node's REAL values instead of pretending they're unset.
        assert {s["name"]: s["current_value"] for s in slots} == {"width": 512, "height": 512, "batch_size": 1}

    def test_partial_dict_does_not_crash(self, graph: Graph):
        wf = _dict_widget_node({"width": 512})
        slots = _extract_frontend_slots(wf, graph)
        assert {s["name"] for s in slots} == {"width", "height", "batch_size"}

    def test_get_template_schema_end_to_end(self, graph: Graph):
        """The exact call path `comfy workflow slots` uses."""
        wf = _dict_widget_node({"width": 512, "height": 512, "batch_size": 1})
        schema = graph.get_template_schema("t", wf)
        assert {s["name"] for s in schema["slots"]} == {"width", "height", "batch_size"}


class TestSetWidgetToleratesDictWidgetsValues:
    def test_apply_one_slot_grows_past_a_short_dict(self, graph: Graph):
        wf = _dict_widget_node({"width": 512})
        _apply_one_slot(wf, "7.batch_size", 4, graph)
        # The dict is projected onto a real positional list — the known value
        # survives at its schema position and the write lands at batch_size's.
        assert wf["nodes"][0]["widgets_values"] == [512, None, 4]

    def test_apply_one_slot_within_dict_len(self, graph: Graph):
        wf = _dict_widget_node({"width": 512, "height": 512, "batch_size": 1})
        _apply_one_slot(wf, "7.batch_size", 4, graph)
        assert wf["nodes"][0]["widgets_values"][2] == 4

    def test_set_widget_public_api_does_not_crash(self, graph: Graph):
        wf = _dict_widget_node({"width": 512, "height": 512, "batch_size": 1})
        new_wf, op = W.set_widget(wf, graph, 7, "batch_size", 4)
        assert op["op"] == "set_widget"
        assert new_wf["nodes"][0]["widgets_values"][2] == 4

    def test_set_widget_grows_past_a_short_dict(self, graph: Graph):
        wf = _dict_widget_node({"width": 512})
        new_wf, op = W.set_widget(wf, graph, 7, "batch_size", 4)
        assert new_wf["nodes"][0]["widgets_values"] == [512, None, 4]
