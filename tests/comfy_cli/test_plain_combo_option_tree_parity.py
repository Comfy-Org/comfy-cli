"""A ``{key, inputs}`` option tree makes a port a dynamic combo whatever its
type is called — including the plain ``COMBO`` some partner schemas ship it
under. Every surface that expands such a tree must agree about which ports
have one, or the converter, the editors, the ``nodes show`` payload and the
validator describe different nodes.

Regression for the split introduced when ``Port.is_dynamic_combo`` started
treating a populated ``dynamic_options`` as authoritative while the converter
and the payload still gated on the ``COMFY_*COMBO*`` TYPE NAME
(``_is_dynamic_combo_type``). The validator then expanded a tree the converter
had not lowered and reported ``required_input_missing`` for a ``model.value``
that ``convert_ui_to_api``, ``add-node`` and ``set-widget`` never produce —
while ``nodes show`` reported ``choices: []`` with no option tree, so the
sub-input names could not be discovered either.

Each test states the surface it covers; they share one catalog so a
disagreement between two surfaces shows up as two failures on the same schema.
"""

from __future__ import annotations

from typing import Any

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api

# The option tree under a plain ``COMBO``. ``value`` is REQUIRED, which is what
# turns a converter that fails to lower it into a validator error rather than a
# silent omission.
_OPTIONS = [
    {"key": "A", "inputs": {"required": {"value": ["STRING", {"default": ""}]}}},
    {"key": "B", "inputs": {"required": {"other": ["INT", {"default": 0}]}}},
]


@pytest.fixture(params=["COMBO", "COMFY_DYNAMICCOMBO_V3"])
def type_id(request: pytest.FixtureRequest) -> str:
    """Both dialects of the same schema. Every assertion below holds for both —
    that equality IS the invariant, so parameterising rather than writing the
    plain-``COMBO`` case alone keeps the two from drifting apart again.
    """
    return request.param


@pytest.fixture
def object_info(type_id: str) -> dict[str, Any]:
    return {
        "DynamicPass": {
            "input": {"required": {"model": [type_id, {"options": _OPTIONS}], "steps": ["INT", {"default": 0}]}},
            "input_order": {"required": ["model", "steps"]},
            "output": ["STRING"],
            "output_name": ["text"],
            "output_node": True,
            "display_name": "DynamicPass",
        }
    }


@pytest.fixture
def ui_workflow() -> dict[str, Any]:
    """A canvas node on selection ``A``: the selector slot, then the one widget
    slot that option reveals, then ``steps``."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "DynamicPass",
                "inputs": [],
                "outputs": [],
                "widgets_values": ["A", "ready", 7],
                "mode": 0,
            }
        ],
        "links": [],
    }


def test_the_converter_lowers_the_selected_options_sub_inputs(object_info, ui_workflow) -> None:
    """Converter surface: the selected option's sub-input gets its own dotted
    key, and ``steps`` keeps its slot instead of being shifted into it."""
    api = convert_ui_to_api(ui_workflow, object_info)
    assert api["1"]["inputs"]["model"] == "A"
    assert api["1"]["inputs"]["model.value"] == "ready"
    assert api["1"]["inputs"]["steps"] == 7


def test_validate_accepts_exactly_what_the_converter_emits(object_info, ui_workflow) -> None:
    """The headline failure: the validator expanded the tree, the converter did
    not, so a canvas workflow that runs on the server failed validation with
    ``required_input_missing`` for ``model.value``."""
    api = convert_ui_to_api(ui_workflow, object_info)
    result = Graph.from_object_info(object_info).validate_workflow(api)
    assert result["valid"], result["errors"]
    assert "required_input_missing" not in [e.get("code") for e in result["errors"]]


def test_the_widget_walk_gives_the_sub_input_a_slot(object_info) -> None:
    """Editor surface: ``set-widget <id>.model.value`` resolves, and the walk
    agrees with the converter about which names own a positional slot."""
    graph = Graph.from_object_info(object_info)
    assert graph.widget_order_for_node("DynamicPass", ["A", "ready", 7]) == ["model", "model.value", "steps"]
    assert graph.widget_order_for_node("DynamicPass", ["B", 3, 7]) == ["model", "model.other", "steps"]


def test_nodes_show_publishes_the_option_tree(object_info) -> None:
    """Discovery surface: without ``dynamic_options`` an agent cannot learn the
    sub-input names at all, because ``choices`` is empty for a dynamic combo."""
    graph = Graph.from_object_info(object_info)
    payload = graph.morphism_to_dict(graph.node("DynamicPass"))
    model = next(entry for entry in payload["inputs"] if entry["name"] == "model")
    assert model["selection_keys"] == ["A", "B"]
    assert [option["key"] for option in model["dynamic_options"]] == ["A", "B"]
    assert [sub["name"] for sub in model["dynamic_options"][0]["inputs"]] == ["model.value"]


def test_add_node_seeds_the_selected_options_sub_widget(object_info) -> None:
    """``add-node`` surface: a fresh node must carry the sub-widget the default
    selection reveals, or every later value lands one slot early."""
    graph = Graph.from_object_info(object_info)
    defaults = graph.widget_defaults("DynamicPass")
    assert defaults["model"] == "A"
    assert "model.value" in defaults


def test_the_catalog_publishes_every_options_slots(object_info) -> None:
    """``dynamic_combo_options`` is what lets a consumer holding only the
    catalog (the doc host's applier) handle a selection other than the first."""
    combos = Graph.from_object_info(object_info).dynamic_combo_options("DynamicPass")
    assert combos["model"]["options"]["A"]["widgets"] == ["model.value"]
    assert combos["model"]["options"]["B"]["widgets"] == ["model.other"]
