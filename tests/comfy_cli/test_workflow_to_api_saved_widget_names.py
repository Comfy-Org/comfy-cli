"""A dynamic combo's sub-inputs are expanded in the order the SAVE recorded.

``widgets_values`` is positional, and the position it was written against is
the node's widget layout *at save time*. For a ``COMFY_DYNAMICCOMBO_V3``
selector, the chosen option's sub-inputs occupy contiguous slots right after
the selector, and the saved graph names them on its ``inputs[]`` entries,
dotted and in that order. Expanding the option's *current* sub-input list
instead shifts every later slot when the option has since gained one.

Ten of the 392 public Comfy Hub workflows in the 2026-10 conversion audit are
``ByteDance2ReferenceNode`` (Seedance 2.0), whose option gained an optional
``auto_upscale`` after ``auto_downscale``. Those graphs converted with
``model.auto_upscale`` holding the seed and ``seed`` holding the seed's
``control_after_generate`` marker. ``validate`` catches the seed type but not
the boolean, so fixing only the seed still submitted the wrong graph.
"""

from __future__ import annotations

import copy

import pytest

from comfy_cli.workflow_to_api import WorkflowConversionError, _schema_widget_pairs, convert_ui_to_api


def _object_info() -> dict:
    """``model`` as a selector whose option gained an optional ``auto_upscale``.

    ``reference_images`` is a connection-only sub-input: it owns no value slot
    and must not consume one, which is what makes this slot arithmetic fragile.
    """
    option = {
        "key": "Seedance 2.0",
        "inputs": {
            "required": {
                "prompt": ["STRING", {"multiline": True, "default": ""}],
                "resolution": [["480p", "720p", "1080p"], {"default": "720p"}],
                "reference_images": ["COMFY_AUTOGROW_V3", {}],
            },
            "optional": {
                "auto_downscale": ["BOOLEAN", {"default": True}],
                "auto_upscale": ["BOOLEAN", {"default": False}],
            },
        },
    }
    return {
        "ByteDance2ReferenceNode": {
            "input": {
                "required": {
                    "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [option]}],
                    "seed": ["INT", {"default": 0, "control_after_generate": True}],
                    "watermark": ["BOOLEAN", {"default": False}],
                }
            },
            "input_order": {"required": ["model", "seed", "watermark"]},
        }
    }


_SAVED_WIDGET_NAMES = (
    "model",
    "model.prompt",
    "model.resolution",
    "model.auto_downscale",
    "seed",
    "watermark",
)


def _node() -> dict:
    """A save made before the option declared ``auto_upscale``."""
    return {
        "id": 14,
        "type": "ByteDance2ReferenceNode",
        "widgets_values": [
            "Seedance 2.0",
            "a pirate dances in the alley",
            "720p",
            True,
            118383194,
            "randomize",
            False,
        ],
        "inputs": [{"name": n, "widget": {"name": n}} for n in _SAVED_WIDGET_NAMES],
    }


def _convert(node: dict, object_info: dict | None = None) -> dict:
    api = convert_ui_to_api({"nodes": [copy.deepcopy(node)], "links": []}, object_info or _object_info())
    return api[str(node["id"])]["inputs"]


def test_a_sub_input_added_to_the_option_does_not_shift_the_seed():
    inputs = _convert(_node())
    assert inputs["seed"] == 118383194
    assert inputs["model.auto_downscale"] is True
    assert inputs["watermark"] is False


def test_an_option_sub_input_the_save_never_had_owns_no_slot():
    # Not the seed, which is what expanding the CURRENT option order produced.
    assert "model.auto_upscale" not in _convert(_node())


def test_the_control_marker_is_not_emitted_as_a_value():
    inputs = _convert(_node())
    assert "randomize" not in inputs.values()
    assert "control_after_generate" not in inputs


def test_a_connection_only_sub_input_consumes_no_slot():
    inputs = _convert(_node())
    assert "model.reference_images" not in inputs
    assert inputs["model.prompt"] == "a pirate dances in the alley"
    assert inputs["model.resolution"] == "720p"


# --------------------------------------------------------------------------- #
# The saved order is used only where it is unambiguous, so a graph that records
# something unusable expands exactly as it did before.
# --------------------------------------------------------------------------- #


def test_a_save_naming_a_sub_the_option_dropped_is_refused():
    node = _node()
    node["inputs"].insert(4, {"name": "model.retired_knob", "widget": {"name": "model.retired_knob"}})
    schema = _object_info()[node["type"]]
    with pytest.raises(WorkflowConversionError, match="retired_knob.*absent from the current catalog"):
        _schema_widget_pairs(schema, node["widgets_values"], node)


def test_a_saved_widget_that_became_connection_only_is_refused():
    object_info = _object_info()
    option = object_info["ByteDance2ReferenceNode"]["input"]["required"]["model"][1]["options"][0]
    option["inputs"]["optional"]["auto_downscale"] = ["BOOLEAN", {"forceInput": True}]

    with pytest.raises(WorkflowConversionError, match="auto_downscale.*connection-only"):
        node = _node()
        _schema_widget_pairs(object_info[node["type"]], node["widgets_values"], node)


def test_a_node_without_widget_markers_expands_the_current_option():
    node = _node()
    node["inputs"] = []
    inputs = _convert(node)
    assert inputs["model"] == "Seedance 2.0"
    assert inputs["model.prompt"] == "a pirate dances in the alley"


def test_a_malformed_inputs_value_does_not_raise():
    node = _node()
    node["inputs"] = 1  # a truthy scalar, not a list
    inputs = _convert(node)
    assert inputs["model"] == "Seedance 2.0"


def test_a_repeated_marker_is_taken_once_at_its_first_position():
    node = _node()
    node["inputs"].append({"name": "model.prompt", "widget": {"name": "model.prompt"}})
    inputs = _convert(node)
    assert inputs["model.prompt"] == "a pirate dances in the alley"
    assert inputs["seed"] == 118383194


def test_a_save_matching_the_current_option_order_is_unchanged():
    """Nothing to reorder: the save already names the current layout."""
    object_info = _object_info()
    node = {
        "id": 14,
        "type": "ByteDance2ReferenceNode",
        "widgets_values": ["Seedance 2.0", "p", "480p", False, True, 7, "fixed", True],
        "inputs": [
            {"name": n, "widget": {"name": n}}
            for n in (
                "model",
                "model.prompt",
                "model.resolution",
                "model.auto_downscale",
                "model.auto_upscale",
                "seed",
                "watermark",
            )
        ],
    }
    inputs = _convert(node, object_info)
    assert inputs["model.auto_downscale"] is False
    assert inputs["model.auto_upscale"] is True
    assert inputs["seed"] == 7
    assert inputs["watermark"] is True


# --------------------------------------------------------------------------- #
# A nested selector's own descendants are ordered by the recursive walk, with
# the nested selector as their parent. They must not disable the saved order
# for their siblings one level up.
# --------------------------------------------------------------------------- #


def _nested_object_info() -> dict:
    """``model`` whose option reveals a nested ``variant`` selector."""
    variant = {
        "key": "wide",
        "inputs": {"required": {"width": ["INT", {"default": 512}]}},
    }
    option = {
        "key": "Seedance 2.0",
        "inputs": {
            "required": {
                "prompt": ["STRING", {"default": ""}],
                "variant": ["COMFY_DYNAMICCOMBO_V3", {"options": [variant]}],
            },
            "optional": {"auto_downscale": ["BOOLEAN", {"default": True}]},
        },
    }
    return {
        "N": {
            "input": {
                "required": {
                    "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [option]}],
                    "seed": ["INT", {"default": 0, "control_after_generate": True}],
                }
            },
            "input_order": {"required": ["model", "seed"]},
        }
    }


def test_a_nested_selectors_descendants_do_not_disable_the_parents_saved_order():
    """The saved order here DIFFERS from the option's current order, so a
    fallback to schema order is detectable: ``auto_downscale`` was saved first.
    """
    node = {
        "id": 1,
        "type": "N",
        "widgets_values": ["Seedance 2.0", True, "a prompt", "wide", 512, 4242, "randomize"],
        "inputs": [
            {"name": n, "widget": {"name": n}}
            for n in (
                "model",
                "model.auto_downscale",
                "model.prompt",
                "model.variant",
                "model.variant.width",
                "seed",
            )
        ],
    }
    inputs = _convert(node, _nested_object_info())
    # `model.variant.width` is the nested selector's own slot, ordered by the
    # recursive walk with `model.variant` as its parent. Seeing it must not make
    # the walk abandon the saved order for `model`'s direct children, which the
    # current option declares as prompt, variant, auto_downscale.
    assert inputs["model.auto_downscale"] is True
    assert inputs["model.prompt"] == "a prompt"
    assert inputs["model.variant"] == "wide"
    assert inputs["model.variant.width"] == 512
    assert inputs["seed"] == 4242


def test_a_nested_selector_is_retained_when_only_its_descendant_has_a_marker():
    node = {
        "id": 1,
        "type": "N",
        "widgets_values": ["Seedance 2.0", True, "a prompt", "wide", 512, 4242, "randomize"],
        "inputs": [
            {"name": n, "widget": {"name": n}}
            for n in (
                "model",
                "model.auto_downscale",
                "model.prompt",
                "model.variant.width",
                "seed",
            )
        ],
    }

    inputs = _convert(node, _nested_object_info())

    assert inputs["model.variant"] == "wide"
    assert inputs["model.variant.width"] == 512
    assert inputs["seed"] == 4242
