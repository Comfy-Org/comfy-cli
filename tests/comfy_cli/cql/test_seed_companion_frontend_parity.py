"""The engine must place a ``control_after_generate`` slot exactly where the frontend does.

The frontend's ``useIntWidget`` adds the companion only when the spec sets
``control_after_generate`` or the input is named exactly ``seed``/``noise_seed``.
A dynamic-combo sub-input is named ``<selector>.<key>`` (``sampling_mode.seed``),
so the name rule never matches it. The engine used to add a slot after every INT
whose name merely CONTAINS ``seed``, so it read and wrote every later widget one
slot off on nodes the frontend saved:

* ``TextGenerate``: the engine read ``sampling_mode.presence_penalty`` from the
  ``thinking`` slot (validate: ``presence_penalty: expected FLOAT, got bool``).
* ``TripoTextToModelNode``: two phantom slots after ``image_seed``/``texture_seed``.
* ``TripoPSeriesTextToModelNode``: ``add_node`` wrote ``"fixed"`` after
  ``model.image_seed`` and ``model.texture_seed``. When the canvas loaded the
  node, every later value moved (validate: ``model.image_seed: expected INT,
  got bool``, ``model.export_uv: expected BOOLEAN, got int``).

The widget values below come from templates the frontend saved
(Comfy-Org/workflow_templates ``api_tripo3_1_text_to_model.json`` and
``llm_gemma4_text_gen.json``). The schemas are cloud ``object_info`` excerpts
(tooltips removed). The P-series schema mirrors ``comfy_api_nodes/nodes_tripo.py``.
"""

from __future__ import annotations

import copy

from comfy_cli import workflow_ops, workflow_to_api
from comfy_cli.cql.engine import Graph

_TRIPO_T2M = {
    "input": {
        "required": {"prompt": ["STRING", {"multiline": True}]},
        "optional": {
            "negative_prompt": ["STRING", {"multiline": True}],
            "model_version": [
                "COMBO",
                {"default": "v2.5-20250123", "options": ["v3.1-20260211", "v3.0-20250812", "v2.5-20250123"]},
            ],
            "style": ["COMBO", {"default": "None", "options": ["object:clay", "gold", "None"]}],
            "texture": ["BOOLEAN", {"default": True}],
            "pbr": ["BOOLEAN", {"default": True}],
            "image_seed": ["INT", {"advanced": True, "default": 42}],
            "model_seed": ["INT", {"advanced": True, "default": 42}],
            "texture_seed": ["INT", {"advanced": True, "default": 42}],
            "texture_quality": [
                "COMBO",
                {"advanced": True, "default": "standard", "options": ["standard", "detailed"]},
            ],
            "face_limit": ["INT", {"advanced": True, "default": -1, "min": -1, "max": 2000000}],
            "quad": ["BOOLEAN", {"advanced": True, "default": False}],
            "geometry_quality": [
                "COMBO",
                {"advanced": True, "default": "standard", "options": ["standard", "detailed"]},
            ],
        },
    },
    "input_order": {
        "required": ["prompt"],
        "optional": [
            "negative_prompt",
            "model_version",
            "style",
            "texture",
            "pbr",
            "image_seed",
            "model_seed",
            "texture_seed",
            "texture_quality",
            "face_limit",
            "quad",
            "geometry_quality",
        ],
    },
    "output": ["MODEL_TASK_ID"],
    "output_name": ["model_file"],
    "category": "api node/3d/Tripo",
    "display_name": "Tripo: Text to Model",
    "python_module": "comfy_api_nodes.nodes_tripo",
}
# api_tripo3_1_text_to_model.json (frontend 0.20.1): three bare 42s, no markers.
_TRIPO_T2M_VALUES = [
    "armored knight kneeling with sword",
    "",
    "v3.1-20260211",
    "None",
    True,
    True,
    42,
    42,
    42,
    "standard",
    -1,
    False,
    "standard",
]

_SAMPLING_ON = {
    "required": {
        "temperature": ["FLOAT", {"default": 0.7, "min": 0.01, "max": 2.0, "step": 1e-06}],
        "top_k": ["INT", {"default": 64, "min": 0, "max": 1000}],
        "top_p": ["FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}],
        "min_p": ["FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}],
        "repetition_penalty": ["FLOAT", {"default": 1.05, "min": 0.0, "max": 5.0, "step": 0.01}],
        "seed": ["INT", {"default": 0, "min": 0, "max": 18446744073709551615}],
    },
    "optional": {"presence_penalty": ["FLOAT", {"default": 0.0, "min": 0.0, "max": 5.0, "step": 0.01}]},
}
_TEXT_GENERATE = {
    "input": {
        "required": {
            "clip": ["CLIP", {}],
            "prompt": ["STRING", {"default": "", "multiline": True, "dynamicPrompts": True}],
            "max_length": ["INT", {"default": 512, "min": 1, "max": 32768}],
            "sampling_mode": [
                "COMFY_DYNAMICCOMBO_V3",
                {"options": [{"key": "on", "inputs": _SAMPLING_ON}, {"key": "off", "inputs": {"required": {}}}]},
            ],
        },
        "optional": {
            "image": ["IMAGE", {}],
            "thinking": ["BOOLEAN", {"default": False}],
            "use_default_template": ["BOOLEAN", {"advanced": True, "default": True}],
        },
    },
    "input_order": {
        "required": ["clip", "prompt", "max_length", "sampling_mode"],
        "optional": ["image", "thinking", "use_default_template"],
    },
    "output": ["STRING"],
    "output_name": ["STRING"],
    "category": "textgen",
    "display_name": "Text Generate",
    "python_module": "comfy_extras.nodes_textgen",
}
# llm_gemma4_text_gen.json (frontend 0.21.0): seed 0 is followed straight by
# presence_penalty 0, then thinking False and use_default_template True.
_TEXT_GENERATE_VALUES = ["Describe the image", 2048, "on", 0.7, 64, 0.95, 0.05, 1.05, 0, 0, False, True]


def _p_series_sub_inputs() -> dict:
    return {
        "required": {
            "prompt": ["STRING", {"multiline": True}],
            "quad": ["BOOLEAN", {"default": False}],
            "face_limit": ["INT", {"default": -1, "min": -1, "max": 50000}],
            "texture": ["COMBO", {"default": "standard", "options": ["standard", "detailed", "extreme", "none"]}],
            "pbr": ["BOOLEAN", {"default": True}],
            "model_seed": ["INT", {"default": 42, "min": 0, "max": 2147483647, "control_after_generate": True}],
            "image_seed": ["INT", {"default": 42, "min": 0, "max": 2147483647, "advanced": True}],
            "texture_seed": ["INT", {"default": 42, "min": 0, "max": 2147483647, "advanced": True}],
            "auto_size": ["BOOLEAN", {"default": False, "advanced": True}],
            "export_uv": ["BOOLEAN", {"default": True, "advanced": True}],
            "compress_geometry": ["BOOLEAN", {"default": False, "advanced": True}],
        },
        "optional": {"negative_prompt": ["STRING", {"multiline": True, "default": ""}]},
    }


_TRIPO_P_SERIES = {
    "input": {
        "required": {
            "model": ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "P2", "inputs": _p_series_sub_inputs()}]}],
        },
    },
    "input_order": {"required": ["model"]},
    "output": ["MODEL_TASK_ID", "FILE_3D_GLB", "FILE_3D_FBX"],
    "output_name": ["model task_id", "GLB", "FBX"],
    "category": "partner/3d/Tripo",
    "display_name": "Tripo P2: Text to Model",
    "python_module": "comfy_api_nodes.nodes_tripo",
}

_OBJECT_INFO = {
    "TripoTextToModelNode": _TRIPO_T2M,
    "TextGenerate": _TEXT_GENERATE,
    "TripoPSeriesTextToModelNode": _TRIPO_P_SERIES,
}


def _graph() -> Graph:
    return Graph.from_object_info(copy.deepcopy(_OBJECT_INFO))


def _workflow(class_type: str, values: list) -> dict:
    return {
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [{"id": 1, "type": class_type, "inputs": [], "outputs": [], "widgets_values": list(values)}],
        "links": [],
        "version": 0.4,
    }


def test_unflagged_seed_like_ints_get_no_slot_on_a_saved_tripo_node():
    order = _graph().widget_order_for_node("TripoTextToModelNode", _TRIPO_T2M_VALUES)

    assert "control_after_generate" not in order
    assert dict(zip(order, _TRIPO_T2M_VALUES)) == dict(
        workflow_to_api._schema_widget_pairs(_TRIPO_T2M, _TRIPO_T2M_VALUES)
    )


def test_set_widget_after_tripo_seeds_writes_the_frontend_slot():
    graph = _graph()
    wf, _op = workflow_ops.set_widget(_workflow("TripoTextToModelNode", _TRIPO_T2M_VALUES), graph, 1, "quad", True)

    values = wf["nodes"][0]["widgets_values"]
    assert values[11] is True
    assert values[10] == -1 and values[12] == "standard"


def test_dynamic_combo_sub_seed_gets_no_slot_on_a_saved_text_generate_node():
    order = _graph().widget_order_for_node("TextGenerate", _TEXT_GENERATE_VALUES)

    assert order == [
        "prompt",
        "max_length",
        "sampling_mode",
        "sampling_mode.temperature",
        "sampling_mode.top_k",
        "sampling_mode.top_p",
        "sampling_mode.min_p",
        "sampling_mode.repetition_penalty",
        "sampling_mode.seed",
        "sampling_mode.presence_penalty",
        "thinking",
        "use_default_template",
    ]


def test_set_presence_penalty_lands_where_validate_reads_it():
    graph = _graph()
    wf, _op = workflow_ops.set_widget(
        _workflow("TextGenerate", _TEXT_GENERATE_VALUES), graph, 1, "sampling_mode.presence_penalty", 0.5
    )

    inputs = workflow_to_api.convert_ui_to_api(wf, copy.deepcopy(_OBJECT_INFO))["1"]["inputs"]
    assert inputs["sampling_mode.presence_penalty"] == 0.5
    assert inputs["thinking"] is False
    assert inputs["use_default_template"] is True


def test_add_node_writes_a_marker_only_after_the_flagged_sub_seed():
    graph = _graph()
    empty = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [], "version": 0.4}
    wf, _op = workflow_ops.add_node(empty, graph, "TripoPSeriesTextToModelNode")

    values = wf["nodes"][0]["widgets_values"]
    # model, prompt, quad, face_limit, texture, pbr, model_seed, <marker>,
    # image_seed, texture_seed, auto_size, export_uv, compress_geometry,
    # negative_prompt: the 14 values the frontend serializes.
    assert values[6:10] == [42, "fixed", 42, 42]
    assert len(values) == 14
    assert values.count("fixed") == 1


def test_exact_seed_name_still_gets_the_implicit_slot():
    graph = Graph.from_object_info(
        {
            "SeedOnly": {
                "input": {"required": {"seed": ["INT", {"default": 0}], "steps": ["INT", {"default": 20}]}},
                "input_order": {"required": ["seed", "steps"]},
                "output": [],
                "output_name": [],
                "category": "test",
                "display_name": "SeedOnly",
                "python_module": "nodes",
            }
        }
    )

    assert graph.widget_order_default("SeedOnly") == ["seed", "control_after_generate", "steps"]
