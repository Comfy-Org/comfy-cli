"""A save the current catalog cannot reconcile costs its own node, not the read.

``saved_sub_order`` refuses to guess when a saved ``inputs[].widget`` marker
names a dynamic-combo sub-input today's catalog no longer declares: the two
layouts disagree, so no position in that node's ``widgets_values`` is known.
That refusal is right on a WRITE — guessing is how ``set-widget seed`` landed on
a neighbour slot (#1079) — but it was reaching the READ surfaces too, and there
it took the whole workflow with it: ``comfy workflow slots`` returned
``ok:false`` with zero slots and ``comfy workflow print`` returned
``workflow_print_unsupported``, for every node in the file, because one node was
stale. Worse, each error pointed the reader at the other command, and both
failed, so a stale save left no way to read the graph at all.

A read now degrades per node: the stale node advertises no slots (and prints its
values positionally, unnamed), every other node reads normally, and the reason
is reported against the node id. Nothing is read off the catalog order — that
would be the original misalignment wearing a fallback's clothes.

The write paths are unchanged and still refuse, which the last two tests pin.
"""

from __future__ import annotations

import pytest

from comfy_cli import workflow_ops, workflow_print
from comfy_cli.cql import promoted
from comfy_cli.cql.engine import (
    _UNRESOLVED,
    Graph,
    SavedWidgetLayoutError,
    _extract_frontend_slots,
    _resolve_proxy_value,
)

# One option, two sub-inputs. The save below additionally marks
# `model.legacy_flag` — a sub-input this option no longer declares.
_OBJECT_INFO = {
    "DroppedSubNode": {
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
                                        "prompt": ["STRING", {"multiline": True, "default": ""}],
                                        "resolution": [["480p", "720p"], {"default": "720p"}],
                                    }
                                },
                            }
                        ]
                    },
                ],
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "watermark": ["BOOLEAN", {"default": False}],
            }
        },
        "input_order": {"required": ["model", "seed", "watermark"]},
    },
    "CLIPTextEncode": {
        "input": {"required": {"text": ["STRING", {"multiline": True, "default": ""}]}},
        "input_order": {"required": ["text"]},
    },
}

_SAVED = ["A", "a pirate dances", "720p", True, 118383194, "randomize", False]
_MARKERS = (
    "model",
    "model.prompt",
    "model.resolution",
    "model.legacy_flag",
    "seed",
    "watermark",
)


def _graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _stale_node() -> dict:
    return {
        "id": 14,
        "type": "DroppedSubNode",
        "widgets_values": list(_SAVED),
        "inputs": [{"name": name, "widget": {"name": name}} for name in _MARKERS],
    }


def _workflow() -> dict:
    return {
        "nodes": [
            _stale_node(),
            {"id": 16, "type": "CLIPTextEncode", "widgets_values": ["a neighbour node still reads"]},
        ],
        "links": [],
    }


def test_slots_still_lists_every_other_node():
    slots = _extract_frontend_slots(_workflow(), _graph())

    assert [slot["address"] for slot in slots] == ["16.text"]


def test_slots_names_the_node_whose_layout_it_could_not_read():
    warnings: list[dict] = []

    _extract_frontend_slots(_workflow(), _graph(), warnings)

    assert len(warnings) == 1, warnings
    assert warnings[0]["code"] == "saved_widget_layout_unreadable"
    assert warnings[0]["instance_id"] == "14"
    assert warnings[0]["node_type"] == "DroppedSubNode"
    assert "model.legacy_flag" in warnings[0]["message"]


def test_the_slot_manifest_carries_that_warning():
    schema = _graph().get_template_schema("t", _workflow())

    assert [slot["address"] for slot in schema["slots"]] == ["16.text"]
    assert [warning["code"] for warning in schema["warnings"]] == ["saved_widget_layout_unreadable"]


def test_a_clean_workflow_gets_no_warnings_key():
    workflow = {"nodes": [{"id": 16, "type": "CLIPTextEncode", "widgets_values": ["hi"]}], "links": []}

    schema = _graph().get_template_schema("t", workflow)

    assert "warnings" not in schema


def test_slots_never_advertises_a_position_it_had_to_guess():
    """The catalog order would place `seed` at index 3 — `auto`/`legacy` drift is
    exactly the off-by-one this read exists to avoid. No slot is better."""
    slots = _extract_frontend_slots(_workflow(), _graph())

    assert all(not slot["address"].startswith("14.") for slot in slots), slots


def test_print_renders_the_rest_of_the_graph():
    result = workflow_print.render_py(_workflow(), _graph())

    assert "CLIPTextEncode" in result.source
    assert "a neighbour node still reads" in result.source


def test_print_shows_the_stale_node_positionally_and_says_why():
    result = workflow_print.render_py(_workflow(), _graph())

    assert "DroppedSubNode(widgets=" in result.source
    assert any("model.legacy_flag" in warning for warning in result.warnings), result.warnings
    # Unnamed on purpose: naming these from today's order is the misalignment.
    assert "seed=" not in result.source


def test_a_dangling_proxy_into_a_stale_node_resolves_to_the_sentinel():
    instance = {"id": 57, "type": "sg", "properties": {"proxyWidgets": [[14, "seed"]]}}
    subgraph = {"id": "sg", "nodes": [_stale_node()], "inputs": [], "outputs": []}

    assert _resolve_proxy_value(instance, subgraph, "seed", _graph()) is _UNRESOLVED


def test_a_promoted_read_off_a_stale_node_is_unset():
    assert promoted._interior_widget_value(_stale_node(), "seed", _graph()) is promoted.UNSET
    assert promoted._node_widget_names(_stale_node(), _graph()) == []


def test_an_edit_is_still_refused_rather_than_guessed():
    with pytest.raises(SavedWidgetLayoutError, match="model.legacy_flag"):
        workflow_ops.set_widget(_workflow(), _graph(), 14, "seed", 999)


def test_capture_is_still_refused_rather_than_guessed():
    with pytest.raises(workflow_ops.RecipeError, match="absent from the current catalog"):
        workflow_ops.capture_recipe(_workflow(), _graph())
