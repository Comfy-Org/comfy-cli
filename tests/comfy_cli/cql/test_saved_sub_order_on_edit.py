"""An edit resolves a dynamic combo's sub-widget slot by the SAVED layout.

A ``COMFY_DYNAMICCOMBO_V3`` selector's chosen option contributes its sub-inputs
as contiguous ``widgets_values`` slots right after the selector. The *saved*
file's layout is the option's sub-input list as it stood when the file was
written, which the node records on its ``inputs[]`` entries. ``Graph``'s
value-aware widget order expanded the option from TODAY's catalog, so a save
made before the option gained a sub-input was off by one from that point on.

``set-widget`` indexes ``widgets_values`` by that order, so the write landed on
a neighbour: on ``ByteDance2ReferenceNode`` (Seedance 2.0), whose option gained
an optional ``auto_upscale`` after ``auto_downscale``, ``set-widget seed`` wrote
the slot holding ``seed``'s ``control_after_generate`` marker and left the seed
itself untouched. The user's edit silently did nothing and the file came back
with a number where a control keyword belongs.

Conversion was fixed for the same reason in #1069; this is the write side of it,
and both now resolve through one helper so they cannot disagree again.
"""

from __future__ import annotations

from comfy_cli import workflow_ops
from comfy_cli.cql.engine import Graph

# `model` is a selector whose "Seedance 2.0" option has since gained an optional
# `auto_upscale` after `auto_downscale`. `reference_images` is connection-only:
# it owns no value slot, which is what makes this arithmetic fragile.
_OPTION = {
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

_OBJECT_INFO = {
    "ByteDance2ReferenceNode": {
        "input": {
            "required": {
                "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [_OPTION]}],
                "seed": ["INT", {"default": 0, "control_after_generate": True}],
                "watermark": ["BOOLEAN", {"default": False}],
            }
        },
        "input_order": {"required": ["model", "seed", "watermark"]},
    }
}

# The save: nine widgets, no `auto_upscale`. Index 6 is auto_downscale, 7 is the
# seed, 8 is its control marker.
_SAVED = ["Seedance 2.0", "a pirate dances", "720p", True, 118383194, "randomize", False]
_MARKERS = (
    "model",
    "model.prompt",
    "model.resolution",
    "model.auto_downscale",
    "seed",
    "watermark",
)


def _graph() -> Graph:
    return Graph.from_object_info(_OBJECT_INFO)


def _workflow() -> dict:
    return {
        "nodes": [
            {
                "id": 14,
                "type": "ByteDance2ReferenceNode",
                "widgets_values": list(_SAVED),
                "inputs": [{"name": n, "widget": {"name": n}} for n in _MARKERS],
            }
        ],
        "links": [],
    }


def _node(wf: dict) -> dict:
    return wf["nodes"][0]


def test_setting_the_seed_writes_the_seed_slot():
    wf, _op = workflow_ops.set_widget(_workflow(), _graph(), 14, "seed", 999)
    values = _node(wf)["widgets_values"]
    assert values[4] == 999, f"seed slot not written; got {values}"


def test_setting_the_seed_does_not_clobber_the_control_marker():
    wf, _op = workflow_ops.set_widget(_workflow(), _graph(), 14, "seed", 999)
    values = _node(wf)["widgets_values"]
    assert values[5] == "randomize", f"control marker overwritten; got {values}"


def test_only_the_targeted_slot_changes():
    wf, _op = workflow_ops.set_widget(_workflow(), _graph(), 14, "seed", 999)
    before, after = list(_SAVED), _node(wf)["widgets_values"]
    changed = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
    assert changed == [4], f"expected only the seed slot to change, changed={changed}"


def test_setting_a_sub_widget_before_the_gap_is_unaffected():
    wf, _op = workflow_ops.set_widget(_workflow(), _graph(), 14, "model.resolution", "1080p")
    values = _node(wf)["widgets_values"]
    assert values[2] == "1080p"
    assert values[4] == 118383194, "the seed must not move"


def test_setting_the_sub_widget_adjacent_to_the_gap_writes_its_own_slot():
    wf, _op = workflow_ops.set_widget(_workflow(), _graph(), 14, "model.auto_downscale", False)
    values = _node(wf)["widgets_values"]
    assert values[3] is False
    assert values[4] == 118383194, "the seed must not be clobbered"


def test_a_save_matching_the_current_option_is_unaffected():
    """No gap to correct for: the save already names today's layout."""
    wf = {
        "nodes": [
            {
                "id": 14,
                "type": "ByteDance2ReferenceNode",
                "widgets_values": ["Seedance 2.0", "p", "480p", True, False, 7, "fixed", True],
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
        ],
        "links": [],
    }
    out, _op = workflow_ops.set_widget(wf, _graph(), 14, "seed", 42)
    assert _node(out)["widgets_values"][5] == 42
    assert _node(out)["widgets_values"][6] == "fixed"


def test_a_node_without_markers_keeps_todays_expansion():
    """No evidence of a saved layout, so the catalog order stands as before."""
    wf = _workflow()
    _node(wf)["inputs"] = []
    out, _op = workflow_ops.set_widget(wf, _graph(), 14, "model.resolution", "1080p")
    assert _node(out)["widgets_values"][2] == "1080p"
