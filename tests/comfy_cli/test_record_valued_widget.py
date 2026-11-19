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
