"""Graphs that validated clean but failed on Comfy Cloud (Langfuse, Sep 2026).

Each test reproduces one observed failure with the schema the cloud catalog
shipped for the node, so validate catches what the worker's ComfyUI would
otherwise reject or crash on.
"""

from typing import Any

from comfy_cli.cql.engine import Graph


def _object_info() -> dict[str, Any]:
    return {
        # VideoHelperSuite: `meta_batch` is a link-only socket whose type is
        # mixed-case, declared BEFORE the `format` widget.
        "VHS_LoadVideo": {
            "input": {
                "required": {
                    "video": [["bedroom.mp4"]],
                    "force_rate": ["FLOAT", {"default": 0}],
                    "custom_width": ["INT", {"default": 0}],
                    "custom_height": ["INT", {"default": 0}],
                    "frame_load_cap": ["INT", {"default": 0}],
                    "skip_first_frames": ["INT", {"default": 0}],
                    "select_every_nth": ["INT", {"default": 1}],
                },
                "optional": {
                    "meta_batch": ["VHS_BatchManager"],
                    "vae": ["VAE"],
                    "format": [["None", "AnimateDiff", "Wan"], {"default": "AnimateDiff"}],
                },
            },
            "input_order": {
                "required": [
                    "video",
                    "force_rate",
                    "custom_width",
                    "custom_height",
                    "frame_load_cap",
                    "skip_first_frames",
                    "select_every_nth",
                ],
                "optional": ["meta_batch", "vae", "format"],
            },
            "output": ["IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO"],
            "output_name": ["IMAGE", "frame_count", "audio", "video_info"],
            "output_node": False,
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": ["IMAGE"],
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                }
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output": [],
            "output_node": True,
        },
        "LoadVideo": {
            "input": {"required": {"file": [["a.mp4", "b.mp4"], {"video_upload": True}]}},
            "input_order": {"required": ["file"]},
            "output": ["VIDEO"],
            "output_node": False,
        },
        "LoadAudio": {
            "input": {"required": {"audio": [["a.wav"], {"audio_upload": True}]}},
            "input_order": {"required": ["audio"]},
            "output": ["AUDIO"],
            "output_node": False,
        },
        "SaveVideo": {
            "input": {
                "required": {
                    "video": ["VIDEO"],
                    "filename_prefix": ["STRING", {"default": "video/ComfyUI"}],
                }
            },
            "input_order": {"required": ["video", "filename_prefix"]},
            "output": [],
            "output_node": True,
        },
        # comfy_extras/nodes_video.py: a prefix autogrow (`videos.video0`, ...).
        "ConcatenateVideo": {
            "input": {
                "required": {
                    "videos": [
                        "COMFY_AUTOGROW_V3",
                        {
                            "template": {
                                "input": {"required": {"video": ["VIDEO", {}]}},
                                "prefix": "video",
                                "min": 1,
                                "max": 100,
                            }
                        },
                    ],
                    "codec": ["COMBO", {"default": "auto", "options": ["auto", "h264", "av1"]}],
                },
                "optional": {"complete_audio": ["AUDIO", {}]},
            },
            "input_order": {"required": ["videos", "codec"], "optional": ["complete_audio"]},
            "output": ["VIDEO"],
            "output_node": False,
        },
        # A names-autogrow nested inside a dynamic combo option.
        "GrokImageEditNodeV2": {
            "input": {
                "required": {
                    "prompt": ["STRING", {"default": "", "multiline": True}],
                    "model": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {
                                    "key": "grok-imagine-image-2.0",
                                    "inputs": {
                                        "required": {
                                            "images": [
                                                "COMFY_AUTOGROW_V3",
                                                {
                                                    "template": {
                                                        "input": {"required": {"image": ["IMAGE", {}]}},
                                                        "names": ["image_1", "image_2", "image_3"],
                                                        "min": 1,
                                                    }
                                                },
                                            ],
                                            "resolution": ["COMBO", {"options": ["1K", "2K"]}],
                                        },
                                        "optional": {"mask": ["MASK", {}]},
                                    },
                                }
                            ]
                        },
                    ],
                }
            },
            "input_order": {"required": ["prompt", "model"]},
            "output": ["IMAGE"],
            "output_node": False,
        },
        # Frontend-extension widget types whose value is a filename.
        "WebcamCapture": {
            "input": {
                "required": {
                    "image": ["WEBCAM", {}],
                    "width": ["INT", {"default": 0}],
                    "height": ["INT", {"default": 0}],
                    "capture_on_queue": ["BOOLEAN", {"default": True}],
                }
            },
            "input_order": {"required": ["image", "width", "height", "capture_on_queue"]},
            "output": ["IMAGE"],
            "output_node": False,
        },
        "RecordAudio": {
            "input": {"required": {"audio": ["AUDIO_RECORD", {}]}},
            "input_order": {"required": ["audio"]},
            "output": ["AUDIO"],
            "output_node": False,
        },
        "LoadImage": {
            "input": {"required": {"image": [["a.png"], {"image_upload": True}]}},
            "input_order": {"required": ["image"]},
            "output": ["IMAGE", "MASK"],
            "output_node": False,
        },
        # comfy_extras/nodes_load_3d.py: `image` is the frontend's viewport
        # capture ({"image": ..., "mask": ..., ...}), read as image['image'].
        "Load3D": {
            "input": {
                "required": {
                    "model_file": ["COMBO", {"options": ["none"], "file_upload": True}],
                    "image": ["LOAD_3D", {}],
                    "width": ["INT", {"default": 1024, "min": 1, "max": 4096}],
                    "height": ["INT", {"default": 1024, "min": 1, "max": 4096}],
                }
            },
            "input_order": {"required": ["model_file", "image", "width", "height"]},
            "output": ["IMAGE", "MASK", "STRING"],
            "output_node": False,
        },
    }


def _graph() -> Graph:
    return Graph.from_object_info(_object_info())


def _codes(result: dict) -> list[tuple[str, Any]]:
    return [(e["code"], e.get("field")) for e in result["errors"]]


class TestLiteralOnLinkInput:
    """VHS_LoadVideo crashed 117x: `'str' object has no attribute 'inputs'`.

    The server type-checks only primitives, so a literal in a custom-type
    socket reaches the node, which then dereferences it as an object.
    """

    def test_literal_value_on_link_only_socket_is_an_error(self):
        vhs_inputs = {
            "video": "bedroom.mp4",
            "force_rate": 0,
            "custom_width": 0,
            "custom_height": 0,
            "frame_load_cap": 0,
            "skip_first_frames": 0,
            "select_every_nth": 1,
            "meta_batch": "None",
            "format": "AnimateDiff",
        }
        wf = {
            "2": {"class_type": "VHS_LoadVideo", "inputs": vhs_inputs},
            "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "x"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("literal_on_link_input", "meta_batch") in _codes(result)


class TestLiteralOnLinkNoFalsePositives:
    """Shapes official templates submit that the server runs fine."""

    def _graph(self) -> Graph:
        info = _object_info()
        info["Painter"] = {
            "input": {
                "required": {
                    "image": ["IMAGE"],
                    # A frontend-registered widget type (ComfyWidgets.COLOR).
                    "bg_color": ["COLOR", {"default": "#000000", "socketless": True}],
                },
                "optional": {
                    # Link socket the converter fills from `default: null`.
                    "mask": ["MASK", {"default": None}],
                    # Extension widget: the schema itself declares a literal.
                    "tint": ["COLORCODE", {"default": "#222222"}],
                },
            },
            "input_order": {"required": ["image", "bg_color"], "optional": ["mask", "tint"]},
            "output": ["IMAGE"],
            "output_node": False,
        }
        return Graph.from_object_info(info)

    def test_widget_types_none_and_declared_defaults_are_not_flagged(self):
        wf = {
            "0": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
            "1": {
                "class_type": "Painter",
                "inputs": {"image": ["0", 0], "bg_color": "#ff0000", "mask": None, "tint": "#123456"},
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        }

        result = self._graph().validate_workflow(wf)

        assert result["valid"], result["errors"]

    def test_webcam_and_record_audio_filenames_are_not_flagged(self):
        # Both nodes read the value as a filename the frontend uploaded.
        info = _object_info()
        info["SaveAudio"] = {
            "input": {"required": {"audio": ["AUDIO"], "filename_prefix": ["STRING", {"default": "audio/ComfyUI"}]}},
            "input_order": {"required": ["audio", "filename_prefix"]},
            "output": [],
            "output_node": True,
        }
        wf = {
            "1": {
                "class_type": "WebcamCapture",
                "inputs": {"image": "webcam/123.png", "width": 0, "height": 0, "capture_on_queue": True},
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
            "3": {"class_type": "RecordAudio", "inputs": {"audio": "recording.wav"}},
            "4": {"class_type": "SaveAudio", "inputs": {"audio": ["3", 0], "filename_prefix": "a"}},
        }

        result = Graph.from_object_info(info).validate_workflow(wf)

        assert result["valid"], result["errors"]


class TestLiteralOnAutogrowSlot:
    """A slot key is a connection too: a filename there reaches the node raw."""

    def test_literal_in_autogrow_slot_is_an_error(self):
        wf = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
            "3": {
                "class_type": "ConcatenateVideo",
                "inputs": {"videos.video0": ["1", 0], "videos.video1": "b.mp4", "codec": "auto"},
            },
            "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0], "filename_prefix": "v"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("literal_on_link_input", "videos.video1") in _codes(result)


class TestAutogrowSlots:
    """ConcatenateVideo 400: `videos.video1` got AUDIO and `video0` was missing."""

    def test_autogrow_slot_link_is_type_checked(self):
        wf = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
            "2": {"class_type": "LoadAudio", "inputs": {"audio": "a.wav"}},
            "3": {
                "class_type": "ConcatenateVideo",
                "inputs": {"videos.video0": ["1", 0], "videos.video1": ["2", 0], "codec": "auto"},
            },
            "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0], "filename_prefix": "v"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("edge_type_mismatch", "videos.video1") in _codes(result)

    def test_first_min_autogrow_slots_are_individually_required(self):
        wf = {
            "1": {"class_type": "LoadVideo", "inputs": {"file": "a.mp4"}},
            "3": {"class_type": "ConcatenateVideo", "inputs": {"videos.video1": ["1", 0], "codec": "auto"}},
            "4": {"class_type": "SaveVideo", "inputs": {"video": ["3", 0], "filename_prefix": "v"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("required_input_missing", "videos.video0") in _codes(result)

    def test_required_autogrow_inside_dynamic_combo_needs_a_slot(self):
        # Prod 9/14: Grok image edit submitted with no image wired.
        wf = {
            "1": {
                "class_type": "GrokImageEditNodeV2",
                "inputs": {"prompt": "make it blue", "model": "grok-imagine-image-2.0", "model.resolution": "1K"},
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("required_input_missing", "model.images.image_1") in _codes(result)

    def test_nested_autogrow_with_first_slot_wired_is_valid(self):
        wf = {
            "0": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
            "1": {
                "class_type": "GrokImageEditNodeV2",
                "inputs": {
                    "prompt": "make it blue",
                    "model": "grok-imagine-image-2.0",
                    "model.resolution": "1K",
                    "model.images.image_1": ["0", 0],
                },
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        }

        result = _graph().validate_workflow(wf)

        assert result["valid"], result["errors"]

    def test_nested_autogrow_slot_link_is_type_checked(self):
        # `model.images.image_1` takes the nested template's IMAGE, not nothing.
        wf = {
            "0": {"class_type": "LoadAudio", "inputs": {"audio": "a.wav"}},
            "1": {
                "class_type": "GrokImageEditNodeV2",
                "inputs": {
                    "prompt": "make it blue",
                    "model": "grok-imagine-image-2.0",
                    "model.resolution": "1K",
                    "model.images.image_1": ["0", 0],
                },
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("edge_type_mismatch", "model.images.image_1") in _codes(result)

    def test_dynamic_combo_sub_input_link_is_type_checked(self):
        wf = {
            "0": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
            "9": {"class_type": "LoadAudio", "inputs": {"audio": "a.wav"}},
            "1": {
                "class_type": "GrokImageEditNodeV2",
                "inputs": {
                    "prompt": "make it blue",
                    "model": "grok-imagine-image-2.0",
                    "model.resolution": "1K",
                    "model.images.image_1": ["0", 0],
                    "model.mask": ["9", 0],
                },
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("edge_type_mismatch", "model.mask") in _codes(result)


class TestLoad3dViewportCapture:
    """Load3D crashed 13x: `string indices must be integers, not 'str'`."""

    def test_load3d_without_viewport_capture_is_an_error(self):
        wf = {
            "1": {
                "class_type": "Load3D",
                "inputs": {"model_file": "m.glb", "image": "", "width": 1024, "height": 1024},
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "x"}},
        }

        result = _graph().validate_workflow(wf)

        assert not result["valid"]
        assert ("frontend_capture_required", "image") in _codes(result)
