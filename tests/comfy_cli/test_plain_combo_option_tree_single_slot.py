"""A plain ``COMBO`` keeps one widget slot even when ``options`` is a tree.

The frontend selects its dynamic-combo widget by type name alone
(``COMFY_DYNAMICCOMBO_V3``), so a plain ``COMBO`` carrying ``{key, inputs}``
options renders one selector and serializes one ``widgets_values`` slot.
Expanding that tree here would read later widgets from the wrong slots.
"""

from __future__ import annotations

from typing import Any

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api

_OBJECT_INFO: dict[str, Any] = {
    "PlainTree": {
        "input": {
            "required": {
                "model": [
                    "COMBO",
                    {
                        "options": [
                            {"key": "A", "inputs": {"required": {"value": ["STRING", {"default": ""}]}}},
                            {"key": "B", "inputs": {"required": {"other": ["INT", {"default": 0}]}}},
                        ]
                    },
                ],
                "steps": ["INT", {"default": 0}],
            }
        },
        "input_order": {"required": ["model", "steps"]},
        "output": ["STRING"],
        "output_name": ["text"],
        "output_node": True,
        "display_name": "PlainTree",
    }
}


def test_the_converter_reads_the_widget_after_the_selector_from_the_next_slot() -> None:
    workflow = {
        "nodes": [
            {
                "id": 1,
                "type": "PlainTree",
                "inputs": [],
                "outputs": [],
                "widgets_values": ["A", 7],
                "mode": 0,
            }
        ],
        "links": [],
    }
    assert convert_ui_to_api(workflow, _OBJECT_INFO)["1"]["inputs"] == {"model": "A", "steps": 7}


def test_the_validator_does_not_require_a_sub_input_of_the_selected_option() -> None:
    result = Graph.from_object_info(_OBJECT_INFO).validate_workflow(
        {"1": {"class_type": "PlainTree", "inputs": {"model": "A", "steps": 7}}}
    )
    assert "required_input_missing" not in {finding.get("code") for finding in result["errors"]}, result["errors"]


def test_the_widget_order_has_one_slot_for_the_selector() -> None:
    assert Graph.from_object_info(_OBJECT_INFO).widget_order_for_node("PlainTree", ["A", 7]) == ["model", "steps"]
