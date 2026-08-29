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

import copy
from typing import Any

import pytest

from comfy_cli import workflow_ops
from comfy_cli.cql import engine
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

    def test_fresh_node_carries_no_phantom_value_for_the_upload_button(self, graph: Graph):
        # Current frontends mark the button ``serialize: false``: a fresh node
        # from add_node must end at the last serialized value — not carry a
        # trailing ``null`` for the injected slot. Asserted on the BUILT node,
        # the layer the claim is about (the defaults dict alone is not).
        assert "upload" not in graph.widget_defaults("LoadImage")
        wf, _ = workflow_ops.add_node(_empty_workflow(), graph, "LoadImage")
        assert wf["nodes"][0]["widgets_values"] == ["beach.jpg"]

    def test_fresh_3d_nodes_carry_the_frontends_empty_preview_slot(self, graph: Graph):
        # The injected PREVIEW_3D ``image`` IS serialized (as ``""``) — the
        # captured shapes are SaveGLB ["mesh/ComfyUI", ""], Preview3D
        # ["out/mesh.glb", ""] — so add_node must write it, not ``null``.
        glb_prefix = graph.widget_defaults("SaveGLB")["filename_prefix"]
        for cls, expected in (("SaveGLB", [glb_prefix, ""]), ("Preview3D", ["", ""])):
            wf, _ = workflow_ops.add_node(_empty_workflow(), graph, cls)
            assert wf["nodes"][0]["widgets_values"] == expected, cls


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
        wf, _ = workflow_ops.add_node(_empty_workflow(), graph, "LoadAudio")
        assert wf["nodes"][0]["widgets_values"] == ["a.wav"]


def _empty_workflow() -> dict[str, Any]:
    return {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "groups": [], "version": 0.4}


class TestCaptureRefusesInjectedSlots:
    def test_capture_recipe_refuses_an_injected_slot_as_a_param_up_front(self, graph: Graph):
        wf, op = workflow_ops.add_node(_empty_workflow(), graph, "LoadImage")
        nid = op["node_id"]
        with pytest.raises(workflow_ops.RecipeError, match="upload"):
            workflow_ops.capture_recipe(wf, graph, lift={(nid, "upload"): "up"})


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


# ---------------------------------------------------------------------------
# Injected slots own a position but are never an edit target.
# ---------------------------------------------------------------------------


def _node(node_id: int, cls: str, widgets: list[Any]) -> dict[str, Any]:
    return {"id": node_id, "type": cls, "inputs": [], "outputs": [], "widgets_values": list(widgets), "mode": 0}


def _workflow(*nodes: dict[str, Any]) -> dict[str, Any]:
    return {"nodes": list(nodes), "links": []}


def _available(msg: str) -> list[str]:
    """The advertised target list out of an edit error, split on the shared
    ``available widgets:`` marker every write surface renders."""
    assert "available widgets: " in msg, msg
    # The not-found path appends a ". Nodes in this workflow: …" hint; names
    # never contain ". ", so the list ends at the first sentence break.
    return msg.split("available widgets: ", 1)[1].split(". ", 1)[0].rstrip(".").split(", ")


# (class, injected name, the only schema-backed widgets ``slots`` advertises)
_INJECTED_TARGETS = [
    ("SaveGLB", "image", ["filename_prefix"]),
    ("Preview3D", "image", ["model_file"]),
    ("LoadImage", "upload", ["image"]),
    ("LoadAudio", "audioUI", ["audio"]),
]


class TestInjectedSlotsAreNotEditable:
    """``frontend_extra_widget_names`` entries have ``port=None``: no schema to
    validate against and no address ``comfy workflow slots`` ever lists. Every
    write surface must refuse them by name — otherwise ``set-widget 1.image``
    on a SaveGLB silently lands an unvalidated value in the viewport slot."""

    @pytest.mark.parametrize(("cls", "name", "avail"), _INJECTED_TARGETS)
    def test_set_widget_refuses_injected_slot(self, graph: Graph, cls: str, name: str, avail: list[str]):
        wf = _workflow(_node(1, cls, _FRONTEND_SHAPES[cls]))
        before = copy.deepcopy(wf)
        with pytest.raises(ValueError) as ei:
            workflow_ops.set_widget(wf, graph, 1, name, "ghost")
        msg = str(ei.value)
        assert "frontend-injected" in msg and "not editable" in msg
        assert _available(msg) == avail
        assert wf == before

    @pytest.mark.parametrize(("cls", "name", "avail"), _INJECTED_TARGETS)
    def test_set_slot_refuses_injected_slot(self, graph: Graph, cls: str, name: str, avail: list[str]):
        # ``set-slot`` / ``vary`` go through ``_write_widget``.
        wf = _workflow(_node(1, cls, _FRONTEND_SHAPES[cls]))
        before = copy.deepcopy(wf)
        with pytest.raises(ValueError) as ei:
            engine._apply_one_slot(wf, f"1.{name}", "ghost", graph)
        msg = str(ei.value)
        assert "frontend-injected" in msg and "not editable" in msg
        assert _available(msg) == avail
        assert wf == before

    @pytest.mark.parametrize(("cls", "name", "avail"), _INJECTED_TARGETS)
    def test_apply_replay_refuses_injected_slot(self, graph: Graph, cls: str, name: str, avail: list[str]):
        # A hand-built op replayed through ``apply`` must not bypass the check.
        wf = _workflow(_node(1, cls, _FRONTEND_SHAPES[cls]))
        before = copy.deepcopy(wf["nodes"])
        op = workflow_ops._new_op("set_widget", "cli", 0, node_id=1, widget=name, value="ghost")
        with pytest.raises(ValueError) as ei:
            workflow_ops.apply_op(wf, op, graph)
        assert "frontend-injected" in str(ei.value)
        assert _available(str(ei.value)) == avail
        assert wf["nodes"] == before

    @pytest.mark.parametrize(("cls", "name", "avail"), _INJECTED_TARGETS)
    def test_not_found_list_omits_injected_slots(self, graph: Graph, cls: str, name: str, avail: list[str]):
        # Both lookups' "available" list is what ``slots`` advertises — never
        # the injected name — so a typo can't be "corrected" onto a ghost.
        wf = _workflow(_node(1, cls, _FRONTEND_SHAPES[cls]))
        with pytest.raises(ValueError) as ei:
            workflow_ops.set_widget(wf, graph, 1, "bogus", "x")
        assert _available(str(ei.value)) == avail
        with pytest.raises(ValueError) as ei:
            engine._apply_one_slot(wf, "1.bogus", "x", graph)
        assert _available(str(ei.value)) == avail

    @pytest.mark.parametrize(("cls", "name", "avail"), _INJECTED_TARGETS)
    def test_positional_order_still_names_the_slot(self, graph: Graph, cls: str, name: str, avail: list[str]):
        # Refusing the write must not drop the slot from the name<->index
        # contract: it still owns its trailing position.
        order = graph.widget_order_for_node(cls, _FRONTEND_SHAPES[cls])
        injected = graph.frontend_injected_widget_names(cls)
        assert name in injected
        # Editable slots first, every injected slot trailing — unchanged.
        assert order == avail + injected
        assert graph.editable_widget_names(cls, _FRONTEND_SHAPES[cls]) == avail

    @pytest.mark.parametrize(("cls", "name", "avail"), _INJECTED_TARGETS)
    def test_error_list_matches_slots(self, graph: Graph, cls: str, name: str, avail: list[str]):
        node = _node(1, cls, _FRONTEND_SHAPES[cls])
        advertised = [s["name"] for s in engine._node_widget_slots(node, "1", graph)]
        assert advertised == avail
        assert name not in advertised


class TestControlAfterGenerateStaysWritable:
    """The seed companion is also a ``port=None`` marker, but unlike the
    injected button/player/viewport slots it carries a real serialized user
    value (``fixed``/``randomize``/...) — pinning a seed is a legitimate edit.
    It stays writable on every surface, and — like ``slots`` — unlisted."""

    def test_set_widget_writes_marker(self, graph: Graph):
        wf = _workflow(_node(1, "KSampler", [0, "randomize", 20]))
        wf, op = workflow_ops.set_widget(wf, graph, 1, "control_after_generate", "fixed")
        assert wf["nodes"][0]["widgets_values"] == [0, "fixed", 20]
        assert op["widget"] == "control_after_generate"

    def test_set_slot_writes_marker(self, graph: Graph):
        wf = _workflow(_node(1, "KSampler", [0, "randomize", 20]))
        assert engine._apply_one_slot(wf, "1.control_after_generate", "fixed", graph) == []
        assert wf["nodes"][0]["widgets_values"] == [0, "fixed", 20]

    def test_marker_is_not_advertised(self, graph: Graph):
        wf = _workflow(_node(1, "KSampler", [0, "randomize", 20]))
        with pytest.raises(ValueError) as ei:
            workflow_ops.set_widget(wf, graph, 1, "bogus", 1)
        assert _available(str(ei.value)) == ["seed", "steps"]
        assert graph.editable_widget_names("KSampler", [0, "randomize", 20]) == ["seed", "steps"]
        assert graph.frontend_injected_widget_names("KSampler") == []
