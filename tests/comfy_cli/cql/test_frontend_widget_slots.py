"""Frontend-injected and DOM-widget slots in the widget order.

The frontend serializes MORE positional ``widgets_values`` than object_info
declares for a handful of node families:

* ``Comfy.UploadImage`` / ``Comfy.UploadAudio`` append a required ``upload``
  input (the upload button) to every node whose required media combo carries
  an ``image_upload`` / ``animated_image_upload`` / ``video_upload`` /
  ``audio_upload`` flag. Older frontends serialized its value (``"image"``);
  current ones mark it ``serialize: false`` and write nothing.
* ``Comfy.AudioWidget`` appends ``audioUI`` to the audio load/save/preview
  family; ``Comfy.Preview3D`` / ``Comfy.SaveGLB`` append ``image``
  (``PREVIEW_3D``).
* Server-declared DOM-widget inputs (``LOAD_3D``, ``AUDIO_UI``, ...) occupy a
  slot even though their type is an uppercase custom name the engine would
  otherwise read as a link.

The doc host's applier builds a name-keyed widget map from the pinned
catalog's ``widget_order``; a ``widgets_values`` longer than that order is a
hard refusal (``createNodeMap(LoadImage): widgets_values has 2 entries but
widget_order names only 1``), which put every Load* workflow on the v0 path.
Fewer values than names is tolerated, so the order must carry every slot the
frontend CAN write, in the position it writes it: injected inputs come after
every declared one (``getOrderedInputSpecs`` appends unlisted inputs last).
"""

from __future__ import annotations

from typing import Any

import pytest

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api


def _object_info() -> dict[str, Any]:
    files = [["beach.jpg", "example.png"], {"image_upload": True}]
    return {
        "LoadImage": {
            "input": {"required": {"image": files}},
            "input_order": {"required": ["image"]},
            "output": ["IMAGE", "MASK"],
            "output_name": ["IMAGE", "MASK"],
            "display_name": "Load Image",
            "python_module": "nodes",
        },
        "LoadImageMask": {
            "input": {"required": {"image": files, "channel": [["alpha", "red", "green", "blue"]]}},
            "input_order": {"required": ["image", "channel"]},
            "output": ["MASK"],
            "output_name": ["MASK"],
            "display_name": "Load Image (as Mask)",
            "python_module": "nodes",
        },
        "LoadImageWithExif": {
            "input": {
                "required": {"image": files},
                "optional": {"default_focal_mm": ["FLOAT", {"default": 50.0}]},
            },
            "input_order": {"required": ["image"], "optional": ["default_focal_mm"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "display_name": "Load Image With EXIF",
            "python_module": "custom_nodes.exif",
        },
        "LoadVideo": {
            "input": {"required": {"file": [["a.mp4"], {"video_upload": True}]}},
            "input_order": {"required": ["file"]},
            "output": ["VIDEO"],
            "output_name": ["VIDEO"],
            "display_name": "Load Video",
            "python_module": "comfy_extras.nodes_video",
        },
        "LoadAudio": {
            "input": {"required": {"audio": [["a.wav"], {"audio_upload": True}]}},
            "input_order": {"required": ["audio"]},
            "output": ["AUDIO"],
            "output_name": ["AUDIO"],
            "display_name": "Load Audio",
            "python_module": "comfy_extras.nodes_audio",
        },
        "PreviewAudio": {
            "input": {"required": {"audio": ["AUDIO", {}]}},
            "input_order": {"required": ["audio"]},
            "output": [],
            "output_name": [],
            "output_node": True,
            "display_name": "Preview Audio",
            "python_module": "comfy_extras.nodes_audio",
        },
        "LoadAudioUI": {
            "input": {
                "required": {"audio": [["a.wav"], {"audio_upload": True}], "start_time": ["FLOAT", {"default": 0.0}]},
                "optional": {"audioUI": ["AUDIO_UI"]},
            },
            "input_order": {"required": ["audio", "start_time"], "optional": ["audioUI"]},
            "output": ["AUDIO"],
            "output_name": ["AUDIO"],
            "display_name": "Load Audio UI",
            "python_module": "custom_nodes.audio_ui",
        },
        "Painter": {
            # STRING with an upload flag: NOT a media combo, so the frontend
            # does not attach the upload button.
            "input": {
                "required": {
                    "mask": ["STRING", {"widgetType": "PAINTER", "image_upload": True, "default": ""}],
                    "width": ["INT", {"default": 512}],
                }
            },
            "input_order": {"required": ["mask", "width"]},
            "output": ["MASK"],
            "output_name": ["MASK"],
            "display_name": "Painter",
            "python_module": "custom_nodes.painter",
        },
        "Load3D": {
            "input": {
                "required": {
                    "model_file": ["COMBO", {"options": ["none"], "file_upload": True}],
                    "image": ["LOAD_3D", {}],
                    "width": ["INT", {"default": 1024}],
                    "height": ["INT", {"default": 1024}],
                }
            },
            "input_order": {"required": ["model_file", "image", "width", "height"]},
            "output": ["IMAGE", "MASK"],
            "output_name": ["image", "mask"],
            "display_name": "Load 3D",
            "python_module": "comfy_extras.nodes_load_3d",
        },
        "Load3DAdvanced": {
            "input": {
                "required": {
                    "model_file": ["COMBO", {"options": ["none"], "file_upload": True}],
                    "viewport_state": ["LOAD_3D", {}],
                    "width": ["INT", {"default": 1024}],
                }
            },
            "input_order": {"required": ["model_file", "viewport_state", "width"]},
            "output": ["IMAGE"],
            "output_name": ["image"],
            "display_name": "Load 3D Advanced",
            "python_module": "comfy_extras.nodes_load_3d",
        },
        "SaveGLB": {
            "input": {
                "required": {"mesh": ["MESH,FILE_3D_GLB", {}], "filename_prefix": ["STRING", {"default": "3d/ComfyUI"}]}
            },
            "input_order": {"required": ["mesh", "filename_prefix"]},
            "output": [],
            "output_name": [],
            "output_node": True,
            "display_name": "Save GLB",
            "python_module": "comfy_extras.nodes_hunyuan3d",
        },
        "Preview3D": {
            "input": {
                "required": {
                    "model_file": ["STRING,FILE_3D_GLB,FILE_3D_GLTF", {"default": "", "widgetType": "STRING"}]
                },
                "optional": {
                    "camera_info": ["LOAD3D_CAMERA", {"advanced": True}],
                    "bg_image": ["IMAGE", {"advanced": True}],
                },
            },
            "input_order": {"required": ["model_file"], "optional": ["camera_info", "bg_image"]},
            "output": [],
            "output_name": [],
            "output_node": True,
            "display_name": "Preview 3D",
            "python_module": "comfy_extras.nodes_load_3d",
        },
        "MathAbs": {
            "input": {"required": {"value": ["FLOAT,INT", {"default": 0.0, "widgetType": "STRING"}]}},
            "input_order": {"required": ["value"]},
            "output": ["FLOAT"],
            "output_name": ["FLOAT"],
            "display_name": "Math Abs",
            "python_module": "custom_nodes.basic_data_handling",
        },
        "MathAdd": {
            "input": {"required": {"a": ["INT,FLOAT", {"default": 0.0}], "b": ["INT,FLOAT", {"default": 0.0}]}},
            "input_order": {"required": ["a", "b"]},
            "output": ["FLOAT"],
            "output_name": ["FLOAT"],
            "display_name": "Math Add",
            "python_module": "custom_nodes.essentials",
        },
        "KSampler": {
            "input": {
                "required": {
                    "seed": ["INT", {"default": 0, "control_after_generate": True}],
                    "steps": ["INT", {"default": 20}],
                }
            },
            "input_order": {"required": ["seed", "steps"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "display_name": "KSampler",
            "python_module": "nodes",
        },
    }


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_object_info())


# The frontend's serialized shapes these orders must be able to name, slot for
# slot. Captured from real workflows: cloud smoke fixture (LoadImage), the
# inpaint-nodes example (LoadImageMask), AudioTools example (LoadAudio), the
# hunyuan3d template (SaveGLB), DepthAnythingV3 bas_relief (Preview3D).
_FRONTEND_SHAPES = {
    "LoadImage": ["beach.jpg", "image"],
    "LoadImageMask": ["mask.png", "red", "image"],
    "LoadVideo": ["a.mp4", "image"],
    "LoadAudio": ["a.wav", None, None],
    "SaveGLB": ["mesh/ComfyUI", ""],
    "Preview3D": ["out/mesh.glb", ""],
}


class TestInjectedUploadSlot:
    @pytest.mark.parametrize(
        ("cls", "expected"),
        [
            ("LoadImage", ["image", "upload"]),
            ("LoadImageMask", ["image", "channel", "upload"]),
            ("LoadVideo", ["file", "upload"]),
            # Injected inputs come after OPTIONAL declared ones too.
            ("LoadImageWithExif", ["image", "default_focal_mm", "upload"]),
        ],
    )
    def test_upload_button_is_the_last_slot(self, graph: Graph, cls: str, expected: list[str]):
        assert graph.widget_order(cls) == expected
        assert graph.widget_order_default(cls) == expected
        assert graph.widget_order_for_node(cls, _FRONTEND_SHAPES.get(cls)) == expected

    def test_string_input_with_upload_flag_gets_no_button(self, graph: Graph):
        # Comfy.UploadImage only attaches to a media COMBO.
        assert graph.widget_order("Painter") == ["mask", "width"]

    def test_file_upload_flag_alone_gets_no_button(self, graph: Graph):
        # Load3D's model_file carries file_upload; the 3D loader handles its
        # own upload and the frontend attaches no IMAGEUPLOAD button.
        assert "upload" not in graph.widget_order("Load3D")

    def test_upload_has_no_add_node_default(self, graph: Graph):
        # Current frontends mark the button ``serialize: false``; a fresh node
        # must not carry a phantom trailing value for it.
        assert "upload" not in graph.widget_defaults("LoadImage")
        assert graph.widget_defaults("LoadImage") == {"image": "beach.jpg"}


class TestAudioFamily:
    def test_load_audio_carries_audio_ui_then_upload(self, graph: Graph):
        # Comfy.AudioWidget registers before Comfy.UploadAudio, so audioUI is
        # injected first. Older frontends serialized both as null.
        assert graph.widget_order("LoadAudio") == ["audio", "audioUI", "upload"]
        assert graph.widget_order_for_node("LoadAudio", _FRONTEND_SHAPES["LoadAudio"]) == ["audio", "audioUI", "upload"]

    def test_preview_audio_link_input_still_gets_audio_ui(self, graph: Graph):
        assert graph.widget_order("PreviewAudio") == ["audioUI"]

    def test_server_declared_audio_ui_is_not_injected_twice(self, graph: Graph):
        assert graph.widget_order("LoadAudioUI") == ["audio", "start_time", "audioUI", "upload"]

    def test_marker_slots_have_no_defaults(self, graph: Graph):
        assert graph.widget_defaults("LoadAudio") == {"audio": "a.wav"}


class TestDomWidgetInputs:
    def test_load3d_viewport_is_a_slot_in_declared_position(self, graph: Graph):
        assert graph.widget_order("Load3D") == ["model_file", "image", "width", "height"]

    def test_load3d_advanced_server_declared_state_is_not_duplicated(self, graph: Graph):
        assert graph.widget_order("Load3DAdvanced") == ["model_file", "viewport_state", "width"]

    def test_dom_widget_default_keeps_later_slots_aligned(self, graph: Graph):
        # A fresh Load3D must serialize its viewport slot, or width lands in it.
        assert graph.widget_defaults("Load3D") == {"model_file": "none", "image": "", "width": 1024, "height": 1024}

    def test_save_glb_preview_is_injected_last(self, graph: Graph):
        assert graph.widget_order("SaveGLB") == ["filename_prefix", "image"]

    def test_preview3d_widget_type_override_is_a_widget(self, graph: Graph):
        # ``STRING,FILE_3D_GLB,...`` is a link by type; ``widgetType: STRING``
        # makes the frontend render a text widget for it. The camera state is
        # ``serialize: false`` and bg_image is a link.
        assert graph.widget_order("Preview3D") == ["model_file", "image"]

    def test_multi_type_without_widget_type_stays_a_link(self, graph: Graph):
        assert graph.widget_order("MathAdd") == []

    def test_widget_type_override_on_multi_type_input(self, graph: Graph):
        assert graph.widget_order("MathAbs") == ["value"]
        assert graph.widget_defaults("MathAbs") == {"value": 0.0}

    def test_control_after_generate_unaffected(self, graph: Graph):
        assert graph.widget_order("KSampler") == ["seed", "control_after_generate", "steps"]


class TestFrontendShapesFit:
    @pytest.mark.parametrize("cls", sorted(_FRONTEND_SHAPES))
    def test_every_captured_shape_fits_the_order(self, graph: Graph, cls: str):
        # The doc host refuses a widgets_values longer than widget_order.
        assert len(_FRONTEND_SHAPES[cls]) <= len(graph.widget_order_for_node(cls, _FRONTEND_SHAPES[cls]))


class TestConverter:
    def test_load3d_viewport_slot_is_consumed_in_place(self):
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "Load3D",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": ["m.glb", "", 512, 768],
                    "mode": 0,
                }
            ],
            "links": [],
        }
        result = convert_ui_to_api(workflow, _object_info())
        assert result["1"]["inputs"] == {"model_file": "m.glb", "image": "", "width": 512, "height": 768}

    def test_trailing_upload_marker_is_ignored(self):
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "LoadImageMask",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": ["mask.png", "red", "image"],
                    "mode": 0,
                }
            ],
            "links": [],
        }
        result = convert_ui_to_api(workflow, _object_info())
        assert result["1"]["inputs"] == {"image": "mask.png", "channel": "red"}

    def test_preview3d_widget_type_override_is_read_as_a_widget(self):
        workflow = {
            "nodes": [
                {
                    "id": 1,
                    "type": "Preview3D",
                    "inputs": [],
                    "outputs": [],
                    "widgets_values": ["out/mesh.glb", ""],
                    "mode": 0,
                }
            ],
            "links": [],
        }
        result = convert_ui_to_api(workflow, _object_info())
        assert result["1"]["inputs"]["model_file"] == "out/mesh.glb"
