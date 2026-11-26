"""Convert reads ``widgets_values_named`` with a per-widget positional fallback.

Read-side half of Amendment A23 (``converter-program`` plan.md, Item A). The
write side (mirroring the map on every widget write, comfy-cli #1106) already
merged; this is A-1 (the usability guard) and A-2 (resolving names inside the
schema walk, not by post-hoc substitution over already-built pairs).

Mirrors ``@comfyorg/api-to-workflow``'s ``usableNamedValues``/``mapPositionalWidgets``
(``src/converter/widgetMapper.ts`` in ``Comfy-Org/comfy-cloud-mcp-server``):

* A-1 -- ``widgetMapper.ts:832``'s guard: non-object, array, and empty map all
  read as absent.
* A-2 -- ``widgetMapper.ts:1140-1141``'s per-widget fallback and
  ``:1660-1664``'s candidate test (every widget the positional-only walk
  sourced from a real slot must be present, and non-``None``, in the map).
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from comfy_cli.workflow_to_api import _usable_named_values, convert_ui_to_api


@pytest.fixture
def object_info() -> dict:
    return {
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                }
            },
            "input_order": {"required": ["width", "height", "batch_size"]},
            "output_node": False,
            "output": ["LATENT"],
            "display_name": "Empty Latent Image",
        },
    }


def _node(widgets_values, widgets_values_named=None, *, node_id=1):
    node = {
        "id": node_id,
        "type": "EmptyLatentImage",
        "inputs": [],
        "outputs": [],
        "mode": 0,
        "widgets_values": widgets_values,
    }
    if widgets_values_named is not None:
        node["widgets_values_named"] = widgets_values_named
    return node


def _convert(node, object_info):
    workflow = {"nodes": [node], "links": []}
    return convert_ui_to_api(workflow, object_info)[str(node["id"])]["inputs"]


# --------------------------------------------------------------------------- #
# A-1: the usability guard. One test per rejected shape.
# --------------------------------------------------------------------------- #


def test_non_object_named_values_is_absent():
    assert _usable_named_values({"widgets_values_named": "not a map"}) is None


def test_array_named_values_is_absent():
    assert _usable_named_values({"widgets_values_named": ["width", "height"]}) is None


def test_empty_map_named_values_is_absent():
    assert _usable_named_values({"widgets_values_named": {}}) is None


def test_missing_named_values_is_absent():
    assert _usable_named_values({}) is None


def test_non_dict_node_is_absent():
    assert _usable_named_values(None) is None


def test_usable_map_is_returned_as_is():
    named = {"width": 1024, "height": 512, "batch_size": 1}
    assert _usable_named_values({"widgets_values_named": named}) == named


@pytest.mark.parametrize(
    "named",
    [
        "not a map",
        ["width", "height", "batch_size"],
        {},
    ],
)
def test_rejected_shapes_leave_convert_reading_positionally(object_info, named):
    node = _node([1024, 512, 1], named)
    inputs = _convert(node, object_info)
    # Guard rejected the map -- the usual positional read is unaffected.
    assert inputs == {"width": 1024, "height": 512, "batch_size": 1}


# --------------------------------------------------------------------------- #
# A-2: resolve inside the walk with a per-widget fallback.
# --------------------------------------------------------------------------- #


def test_a_complete_map_matches_the_package(object_info):
    """A complete, agreeing map changes nothing observable -- it supplies the
    same values the positional array already holds, exactly as the package's
    own candidate test expects for a frontend-written (#10392) save."""
    node = _node(
        [1024, 768, 4],
        {"width": 1024, "height": 768, "batch_size": 4},
    )
    inputs = _convert(node, object_info)
    assert inputs == {"width": 1024, "height": 768, "batch_size": 4}


def test_a_complete_map_with_a_different_value_is_preferred_over_the_array(object_info):
    """This is the case that actually exercises A-2's mechanism: on
    ``cql/engine.py``'s current write paths the map can disagree with the
    array (not maintained by every writer, see plan.md Item A's audit). A
    COMPLETE map -- one that covers every positionally-sourced name -- is read
    in preference to the stale array, matching ``widgets_values_named``'s role
    as the frontend's own preferred read."""
    node = _node(
        [1024, 512, 1],
        {"width": 2048, "height": 512, "batch_size": 1},
    )
    inputs = _convert(node, object_info)
    assert inputs == {"width": 2048, "height": 512, "batch_size": 1}


def test_a_partial_map_falls_back_per_widget_not_all_or_nothing(object_info):
    """The acceptance case: a map that does not name every widget is still
    used for the widgets it DOES cover, falling back to the positional slot
    only for the one it does not -- not discarded wholesale.

    ``height``'s slot is a ``None`` hole (as ``serialize()`` leaves for a
    widget it does not serialize), so it is excluded from the candidate
    test's requirement (``widgetMapper.ts:1660-1664`` only requires names
    sourced from a REAL slot) and the map need not carry it. ``width`` and
    ``batch_size`` DO come from real slots, the map covers both, so the
    candidate test passes and their values are taken FROM THE MAP -- which
    here deliberately disagrees with the array, so this fails on an
    unmodified converter that ignores the map (1024/4) and only passes once
    the map is actually consulted per widget (2048/8), with ``height`` still
    falling back to its own (``None``) slot rather than being dropped."""
    node = _node(
        [1024, None, 4],
        {"width": 2048, "batch_size": 8},  # height deliberately absent
    )
    inputs = _convert(node, object_info)
    assert inputs == {"width": 2048, "height": None, "batch_size": 8}


def test_a_map_missing_a_real_slots_name_fails_the_candidate_test(object_info):
    """Distinct from the per-widget case above: when the name missing from the
    map DID come from a real (non-``None``) positional slot, the package's
    candidate test (``widgetMapper.ts:1660-1664``) is global, not per-widget --
    the whole map is rejected and the node reads exactly as it would with no
    map at all, rather than mixing map values for the covered names with
    positional for the one real value the map dropped."""
    node = _node(
        [1024, 768, 4],
        {"width": 1024, "batch_size": 4},  # height missing, but its slot (768) is real
    )
    inputs = _convert(node, object_info)
    assert inputs == {"width": 1024, "height": 768, "batch_size": 4}


def test_a_present_but_null_map_entry_also_fails_the_candidate_test(object_info):
    """Same global rejection, via a present-but-``None`` entry for a
    real-slotted name rather than a missing one (the "non-``None``" half of
    ``widgetMapper.ts:1660-1664``'s candidate test). Pins that the guard checks
    ``named[name] is not None``, not just ``name in named`` -- dropping that
    half would let this map pass the candidate test and incorrectly overwrite
    ``height``'s real value (768) with ``None``."""
    node = _node(
        [1024, 768, 4],
        {"width": 1024, "height": None, "batch_size": 4},
    )
    inputs = _convert(node, object_info)
    assert inputs == {"width": 1024, "height": 768, "batch_size": 4}


def test_an_extra_name_in_the_map_is_ignored(object_info):
    """A name outside the node's schema is simply never looked up -- the walk
    only ever asks the map for names it is already visiting."""
    node = _node(
        [1024, 768, 4],
        {"width": 1024, "height": 768, "batch_size": 4, "unrelated_name": "ignored"},
    )
    inputs = _convert(node, object_info)
    assert inputs == {"width": 1024, "height": 768, "batch_size": 4}
    assert "unrelated_name" not in inputs


# --------------------------------------------------------------------------- #
# A-2, structural widgets: a dynamic group's row COUNT and a dynamic combo's
# SELECTOR decide how many further slots the walk consumes, so they are read
# from ``widgets_values`` even when the map covers them.
# --------------------------------------------------------------------------- #


@pytest.fixture
def dynamic_group_fixture() -> dict:
    path = pathlib.Path(__file__).parent / "fixtures" / "dynamic_group.json"
    return json.loads(path.read_text())


def _group_case(dynamic_group_fixture: dict) -> tuple[dict, dict]:
    """The ``populated`` case: 2 LoRA rows between a ``before`` and an ``after``
    STRING, with a ``widgets_values_named`` map the frontend wrote in lockstep.
    """
    return (
        copy.deepcopy(dynamic_group_fixture["cases"]["populated"]["workflow"]),
        dynamic_group_fixture["object_info"],
    )


def test_a_disagreeing_group_count_does_not_shift_the_widgets_after_it(dynamic_group_fixture):
    """The corruption a map-sourced row count causes, and the reason the count
    is excluded from the named read.

    Every name the positional walk sources from a real slot must be in the map
    for it to be used at all — but a slot holding ``None`` is excluded from that
    requirement, so ``after`` can legitimately be absent from the map and fall
    back to its own positional slot. Let the map's ``loras`` (1) override the
    array's row count (2) and the walk expands one row while the array holds
    two, so ``after`` falls back to a slot that is no longer its own: the node's
    trailing STRING input came out holding ``C.safetensors``, a LoRA filename.

    The row count is a statement about ``widgets_values``. A map that disagrees
    with it is not reporting a different value, it is desynchronising the walk
    from the array it is walking.
    """
    workflow, object_info = _group_case(dynamic_group_fixture)
    node = workflow["nodes"][0]
    node["widgets_values"][-1] = None  # `after`'s slot is a hole...
    node["widgets_values_named"].pop("after")  # ...so the map need not carry it
    node["widgets_values_named"]["loras"] = 1  # ...and now disagrees on the count

    inputs = convert_ui_to_api(workflow, object_info)["1"]["inputs"]

    assert inputs["after"] is None
    # Both rows the ARRAY holds are still there.
    assert inputs["loras.1.lora_name"] == "C.safetensors"
    assert inputs["loras.1.strength"] == 0.5


def test_a_group_count_only_the_map_could_satisfy_does_not_fail_the_conversion(dynamic_group_fixture):
    """Same exclusion, the other direction: a map-sourced count ABOVE the
    array's made ``_dynamic_group_row_count`` reject the node outright —
    ``DynamicGroup row count does not match the saved widget values`` — on a
    workflow that converts cleanly without the map."""
    workflow, object_info = _group_case(dynamic_group_fixture)
    workflow["nodes"][0]["widgets_values_named"]["loras"] = 3

    inputs = convert_ui_to_api(workflow, object_info)["1"]["inputs"]

    assert inputs["before"] == "head"
    assert inputs["after"] == "tail"
    assert "loras.2.lora_name" not in inputs


def test_a_groups_own_field_values_are_still_read_from_the_map(dynamic_group_fixture):
    """The exclusion is the count, not the group. A row's own fields are
    ordinary widgets and keep A-2's per-widget named read."""
    workflow, object_info = _group_case(dynamic_group_fixture)
    workflow["nodes"][0]["widgets_values_named"]["loras.1.strength"] = 0.25

    inputs = convert_ui_to_api(workflow, object_info)["1"]["inputs"]

    assert inputs["loras.1.strength"] == 0.25
    assert inputs["loras.0.strength"] == 1
    assert inputs["after"] == "tail"
