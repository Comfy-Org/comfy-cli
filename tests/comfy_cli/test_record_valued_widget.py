"""A record-valued widget must land on its own widget, not trigger the Lora shortcut.

Reported as Comfy-Org/comfy-cli#1011. ``ImageCropV2`` stores its ``crop_region``
(type ``BOUNDING_BOX``, ``socketless: true``) as a record in ``widgets_values``:

    "widgets_values": [{"x": 0, "y": 0, "width": 1669, "height": 2688}, 0, 0, 512, 512]

``_collect_widget_inputs`` sent the whole node through
``_absorb_dict_widget_values`` as soon as ANY value in the list was a dict. That
function only understands the two self-naming shapes the Power Lora stack
serializes -- a dict with a ``type`` key, or a dict with a ``lora`` key -- so
every other record was skipped silently, the input was never set, and
``_collect_default_inputs`` then filled it from the schema.

Nothing warns, because the job runs. In the bundled ``3d_pixal3d_multi_views``
template all four crops became the same 512x512 corner of the input image, the
four "views" fed to the conditioning node were identical, and the mesh came out
radially symmetric. The same workflow queued from the frontend was correct.

This is NOT the missing-slot case of #893: ``_is_widget_input`` does give
``crop_region`` a slot (it is ``socketless``). The slot existed and the value was
discarded by the dict shortcut.

The fix narrows the shortcut's trigger to the entries it can actually absorb, so
a record falls through to the ordinary schema walk and pairs with its widget like
a scalar does. Same shape for any socketless input whose frontend widget
serializes a record -- ``CropByBBoxes.bboxes``, ``VideoTrim.trim``,
``CurveEditor.curve``, ``ImageCompositor.compositor`` and the rest.
"""

from __future__ import annotations

from comfy_cli.workflow_to_api import convert_ui_to_api

_OBJECT_INFO = {
    "LoadImage": {
        "input": {"required": {"image": [["house.png"], {"image_upload": True}]}},
        "input_order": {"required": ["image"]},
        "output": ["IMAGE", "MASK"],
        "output_name": ["IMAGE", "MASK"],
    },
    "ImageCropV2": {
        "input": {
            "required": {
                "image": ["IMAGE", {}],
                "crop_region": [
                    "BOUNDING_BOX",
                    {
                        "default": {"x": 0, "y": 0, "width": 512, "height": 512},
                        "socketless": True,
                        "component": "ImageCrop",
                    },
                ],
            }
        },
        "input_order": {"required": ["image", "crop_region"]},
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    },
    "Power Lora Loader": {
        "input": {"required": {"model": ["MODEL", {}]}},
        "input_order": {"required": ["model"]},
        "output": ["MODEL"],
        "output_name": ["MODEL"],
    },
}

_SAVED_CROP = {"x": 0, "y": 0, "width": 1669, "height": 2688}


def _crop_workflow(widgets_values: list) -> dict:
    return {
        "nodes": [
            {
                "id": 364,
                "type": "LoadImage",
                "widgets_values": ["house.png", "image"],
                "outputs": [{"name": "IMAGE", "type": "IMAGE", "links": [1]}],
            },
            {
                "id": 338,
                "type": "ImageCropV2",
                "inputs": [
                    {"name": "image", "type": "IMAGE", "link": 1},
                    {
                        "name": "crop_region",
                        "type": "BOUNDING_BOX",
                        "widget": {"name": "crop_region"},
                        "link": None,
                    },
                ],
                "widgets_values": widgets_values,
            },
        ],
        "links": [[1, 364, 0, 338, 0, "IMAGE"]],
        "version": 0.4,
    }


def test_record_valued_widget_keeps_its_saved_value():
    """The reported case, with the frontend's Custom-ratio sub-widget trailer."""
    api = convert_ui_to_api(_crop_workflow([_SAVED_CROP, 0, 0, 512, 512]), _OBJECT_INFO)
    assert api["338"]["inputs"]["crop_region"] == _SAVED_CROP
    assert api["338"]["inputs"]["image"] == ["364", 0]


def test_record_valued_widget_without_the_sub_widget_trailer():
    """The reporter checked both; the trailer is not what made the difference."""
    api = convert_ui_to_api(_crop_workflow([_SAVED_CROP]), _OBJECT_INFO)
    assert api["338"]["inputs"]["crop_region"] == _SAVED_CROP


def test_the_schema_default_is_not_silently_substituted():
    """The symptom that made this invisible: a wrong value, never a failure."""
    api = convert_ui_to_api(_crop_workflow([_SAVED_CROP, 0, 0, 512, 512]), _OBJECT_INFO)
    assert api["338"]["inputs"]["crop_region"] != {"x": 0, "y": 0, "width": 512, "height": 512}


def test_the_named_dict_form_still_works():
    """The reporter's own workaround must keep working, unchanged."""
    wf = _crop_workflow([])
    wf["nodes"][1]["widgets_values"] = {"crop_region": _SAVED_CROP}
    api = convert_ui_to_api(wf, _OBJECT_INFO)
    assert api["338"]["inputs"]["crop_region"] == _SAVED_CROP


def test_power_lora_entries_still_absorb_by_name():
    """The shortcut this narrows must still fire for the shapes it understands."""
    wf = {
        "nodes": [
            {
                "id": 7,
                "type": "Power Lora Loader",
                "widgets_values": [
                    {},
                    {"on": True, "lora": "styles/ink.safetensors", "strength": 1.0},
                    {"on": True, "lora": "styles/wash.safetensors", "strength": 0.5},
                    "",
                ],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, _OBJECT_INFO)
    inputs = api["7"]["inputs"]
    assert inputs["lora_1"]["lora"] == "styles/ink.safetensors"
    assert inputs["lora_2"]["lora"] == "styles/wash.safetensors"


def test_type_keyed_entries_still_absorb_by_their_type():
    """The other self-naming shape ``_absorb_dict_widget_values`` handles."""
    wf = {
        "nodes": [
            {
                "id": 9,
                "type": "Power Lora Loader",
                "widgets_values": [{"type": "model", "value": "sd15.safetensors"}],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, _OBJECT_INFO)
    assert api["9"]["inputs"]["model"] == {"type": "model", "value": "sd15.safetensors"}


def test_an_unhashable_type_value_does_not_crash_the_node():
    """A ``type`` key is only a usable name when it is a non-empty string.

    An unhashable value (a list, here) used to reach the ``link_inputs``
    membership test inside the absorber and raise ``TypeError``, dropping the
    whole node from the prompt via the per-node exception handler.
    """
    wf = {
        "nodes": [
            {
                "id": 9,
                "type": "Power Lora Loader",
                "widgets_values": [{"type": ["not", "a", "name"], "value": 1}],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, _OBJECT_INFO)
    assert "9" in api


def test_a_declared_widgets_own_type_keyed_record_is_not_absorbed():
    """A node with a real widget slot must never route through the Lora shortcut.

    ``CurveEditor.curve`` serializes its own value as a record that happens to
    carry a ``type`` key (``{"type": "linear", "points": [...]}``), the same
    shape ``_absorb_dict_widget_values`` uses for a self-naming Power Lora
    entry. Before this guard, the drawn curve landed under a bogus ``linear``
    input and ``curve`` itself fell back to its schema default -- replacing the
    user's curve with the identity curve while reporting nothing.
    """
    object_info = dict(_OBJECT_INFO)
    object_info["CurveEditor"] = {
        "input": {
            "required": {
                "curve": [
                    "CURVE",
                    {"default": {"type": "linear", "points": [[0, 0], [1, 1]]}, "socketless": True},
                ]
            }
        },
        "input_order": {"required": ["curve"]},
        "output": [],
        "output_name": [],
    }
    drawn_curve = {"type": "linear", "points": [[0, 0], [0.5, 0.8], [1, 1]]}
    wf = {
        "nodes": [{"id": 5, "type": "CurveEditor", "widgets_values": [drawn_curve]}],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, object_info)
    assert api["5"]["inputs"]["curve"] == drawn_curve
    assert "linear" not in api["5"]["inputs"]


def test_a_lora_entry_mixed_with_a_record_still_takes_the_lora_path():
    """Deliberate, and stated: one absorbable entry still routes the whole node.

    The trigger is "any entry this absorber understands", not "every entry". A node
    that serializes both a Power Lora stack and an unrelated record is not a shape
    the frontend produces, and guessing per-entry would be a bigger change than the
    bug warrants. If such a node appears, the record is still dropped -- the failure
    is narrowed, not eliminated.
    """
    wf = {
        "nodes": [
            {
                "id": 11,
                "type": "Power Lora Loader",
                "widgets_values": [{"on": True, "lora": "a.safetensors"}, {"x": 1, "y": 2}],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, _OBJECT_INFO)
    assert "lora_1" in api["11"]["inputs"]


def test_a_dropped_record_warns_instead_of_vanishing(caplog):
    """The substitution in the mixed case above must be visible somewhere.

    Routing is all-or-nothing, so the unrelated record in that case is
    discarded with no sign anything happened. A warning is the minimum signal
    that a value was dropped rather than converted.
    """
    wf = {
        "nodes": [
            {
                "id": 11,
                "type": "Power Lora Loader",
                "widgets_values": [{"on": True, "lora": "a.safetensors"}, {"x": 1, "y": 2}],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    with caplog.at_level("WARNING"):
        convert_ui_to_api(wf, _OBJECT_INFO)
    assert any("does not understand" in r.message for r in caplog.records)


def test_a_no_loras_yet_save_does_not_land_on_real_sockets():
    """``[{}, ""]`` -- the legacy "no rows yet" save -- must not misplace values.

    Neither entry names anything ``_is_absorbable_dict`` recognizes, so
    without also routing an all-empty-markers list through the absorber this
    fell to the positional walk and the ``{}``/``""`` landed on the node's
    real ``model``/``clip`` sockets instead of being skipped.
    """
    object_info = dict(_OBJECT_INFO)
    object_info["Power Lora Loader"] = {
        "input": {"required": {"model": ["MODEL", {}], "clip": ["CLIP", {}]}},
        "input_order": {"required": ["model", "clip"]},
        "output": ["MODEL", "CLIP"],
        "output_name": ["MODEL", "CLIP"],
    }
    wf = {
        "nodes": [
            {
                "id": 7,
                "type": "Power Lora Loader",
                "widgets_values": [{}, ""],
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "clip", "type": "CLIP", "link": None},
                ],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, object_info)
    inputs = api["7"]["inputs"]
    assert "model" not in inputs
    assert "clip" not in inputs
    assert inputs == {"➕ Add Lora": ""}


def test_a_mixed_list_with_an_unhashable_type_keeps_the_valid_lora_row():
    """``_is_absorbable_dict`` admits a list when ANY entry qualifies.

    So ``[{"lora": ...}, {"type": ["bad"], "value": 1}]`` reaches the absorber on
    the strength of its lora entry, and the list-valued ``type`` then went into a
    ``name not in link_inputs`` membership test. That raises ``TypeError``, which
    escapes ``_collect_widget_inputs`` and drops the ENTIRE node -- the valid lora
    row with it. A malformed name belongs in the warning branch instead.
    """
    object_info = dict(_OBJECT_INFO)
    object_info["Power Lora Loader"] = {
        "input": {"required": {"model": ["MODEL", {}], "clip": ["CLIP", {}]}},
        "input_order": {"required": ["model", "clip"]},
        "output": ["MODEL", "CLIP"],
        "output_name": ["MODEL", "CLIP"],
    }
    wf = {
        "nodes": [
            {
                "id": 9,
                "type": "Power Lora Loader",
                "widgets_values": [{"lora": "a.safetensors"}, {"type": ["bad"], "value": 1}],
                "inputs": [
                    {"name": "model", "type": "MODEL", "link": None},
                    {"name": "clip", "type": "CLIP", "link": None},
                ],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, object_info)
    # The node survives at all, which is the point.
    assert "9" in api
    assert api["9"]["inputs"]["lora_1"] == {"lora": "a.safetensors"}


def test_a_schema_less_scalar_only_list_keeps_its_value_positionally():
    """A bare-scalar list is not a "no rows yet" lora save and must not be absorbed.

    Routing ``[""]`` through the absorber emits ``{"➕ Add Lora": ""}`` and drops
    the marked widget's value; the positional fallback keeps it under its own
    widget name. The all-empty-markers route therefore requires at least one
    empty DICT, not merely the absence of anything else.
    """
    wf = {
        "nodes": [
            {
                "id": 11,
                "type": "UnknownToTheCatalog",
                "widgets_values": [""],
                "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": None}],
            }
        ],
        "links": [],
        "version": 0.4,
    }
    api = convert_ui_to_api(wf, {})
    assert "11" in api
    # Kept under its own widget name, not under the absorber's "➕ Add Lora".
    assert api["11"]["inputs"] == {"text": ""}
