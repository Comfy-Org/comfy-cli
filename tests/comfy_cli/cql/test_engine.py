"""Tests for comfy_cli.cql.engine — the pure-Python CQL graph engine.

Layer 1: unit tests for Graph methods and slot-editing helpers.
No I/O, no CLI invocation — just the engine in isolation.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import pytest

from comfy_cli.command.run.loader import _classify_api_workflow
from comfy_cli.cql.engine import (
    Graph,
    _apply_one_slot,
    _extract_frontend_slots,
    _write_widget,
)

# ---------------------------------------------------------------------------
# Shared fixture: a small but realistic object_info
# ---------------------------------------------------------------------------


def _object_info() -> dict[str, Any]:
    """Covers: link inputs, widget inputs, COMBO/ENUM, control_after_generate,
    force_input, output_node, api_node, multiple output types."""
    return {
        "CheckpointLoaderSimple": {
            "input": {
                "required": {
                    "ckpt_name": [["sd_xl_base.safetensors", "v1-5-pruned.safetensors"]],
                },
            },
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "category": "loaders",
            "display_name": "Load Checkpoint",
            "description": "Loads a checkpoint.",
            "output_node": False,
            "python_module": "nodes",
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": "MODEL",
                    "positive": "CONDITIONING",
                    "negative": "CONDITIONING",
                    "latent_image": "LATENT",
                    "seed": ["INT", {"default": 0, "min": 0, "max": 2**32, "control_after_generate": True}],
                    "steps": ["INT", {"default": 20, "min": 1, "max": 10000}],
                    "cfg": ["FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}],
                    "sampler_name": [["euler", "euler_ancestral", "dpmpp_2m"]],
                    "scheduler": [["normal", "karras", "simple"]],
                    "denoise": ["FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}],
                },
            },
            "input_order": {
                "required": [
                    "model",
                    "positive",
                    "negative",
                    "latent_image",
                    "seed",
                    "steps",
                    "cfg",
                    "sampler_name",
                    "scheduler",
                    "denoise",
                ]
            },
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "sampling",
            "display_name": "KSampler",
            "description": "Denoise latent via model.",
            "output_node": False,
            "python_module": "nodes",
        },
        "CLIPTextEncode": {
            "input": {
                "required": {
                    "text": ["STRING", {"multiline": True}],
                    "clip": "CLIP",
                },
            },
            "input_order": {"required": ["clip", "text"]},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "category": "conditioning",
            "display_name": "CLIP Text Encode",
            "description": "Encode prompt text.",
            "output_node": False,
            "python_module": "nodes",
        },
        "VAEDecode": {
            "input": {
                "required": {
                    "samples": "LATENT",
                    "vae": "VAE",
                },
            },
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "latent",
            "display_name": "VAE Decode",
            "output_node": False,
            "python_module": "nodes",
        },
        "SaveImage": {
            "input": {
                "required": {
                    "images": "IMAGE",
                    "filename_prefix": ["STRING", {"default": "ComfyUI"}],
                },
            },
            "input_order": {"required": ["images", "filename_prefix"]},
            "output": [],
            "output_name": [],
            "category": "image",
            "display_name": "Save Image",
            "output_node": True,
            "python_module": "nodes",
        },
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512, "min": 16, "max": 8192}],
                    "height": ["INT", {"default": 512, "min": 16, "max": 8192}],
                    "batch_size": ["INT", {"default": 1, "min": 1, "max": 64}],
                },
            },
            "input_order": {"required": ["width", "height", "batch_size"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "category": "latent",
            "display_name": "Empty Latent Image",
            "output_node": False,
            "python_module": "nodes",
        },
        # V3 autogrow node mirroring the live cloud BatchImagesNode shape:
        # one declared input `images` (COMFY_AUTOGROW_V3), but the server
        # expects autogrown slot keys `images.image0`, `images.image1`, …
        "BatchImagesNode": {
            "input": {
                "required": {
                    "images": ["COMFY_AUTOGROW_V3", {}],
                },
            },
            "input_order": {"required": ["images"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "image",
            "display_name": "Batch Images",
            "output_node": False,
            "python_module": "nodes",
        },
        # Partner-API video node mirroring the live cloud shape:
        #  - int-valued combos (`duration`, `fps`) — list-of-ints form
        #  - dict-form combos (`resolution`) — ["COMBO", {"options": [...]}]
        "LtxvApiTextToVideo": {
            "input": {
                "required": {
                    "prompt": ["STRING", {"default": ""}],
                    "duration": [[6, 8, 10, 12], {"default": 8}],
                    "fps": [[25, 50], {"default": 25}],
                    "resolution": ["COMBO", {"options": ["1920x1080", "2560x1440"], "default": "1920x1080"}],
                },
            },
            "input_order": {"required": ["prompt", "duration", "fps", "resolution"]},
            "output": ["VIDEO"],
            "output_name": ["VIDEO"],
            "category": "partner/video/LTXV",
            "display_name": "LTXV Text To Video",
            "output_node": False,
            "api_node": True,
            "python_module": "nodes",
        },
    }


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_object_info())


@pytest.fixture
def graph_sd15() -> Graph:
    """Graph built from the real captured sd15 object_info fixture — the same
    catalog the BE-3349 repro / BE-3357 acceptance criterion runs against."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "sd15_object_info.json"
    return Graph.from_object_info(json.loads(fixture.read_text()))


@pytest.fixture
def graph_path() -> Graph:
    """Graph built from the captured path-search object_info fixture: the sd15
    core nodes, the audio nodes (AUDIO is consumed but never reaches IMAGE), a
    second LATENT->IMAGE decoder, and the partner-API image node whose `model`
    widget is a COMBO of API ids rather than a MODEL input (BE-6857)."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "nodes_path_object_info.json"
    return Graph.from_object_info(json.loads(fixture.read_text()))


# ---------------------------------------------------------------------------
# Direct-mode workflow fixture
# ---------------------------------------------------------------------------


def _direct_workflow():
    """A regular frontend-format workflow — no subgraphs."""
    return {
        "nodes": [
            {
                "id": 3,
                "type": "KSampler",
                "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
            },
            {
                "id": 6,
                "type": "CLIPTextEncode",
                "widgets_values": ["a cat in space"],
            },
            {
                "id": 7,
                "type": "EmptyLatentImage",
                "widgets_values": [512, 512, 1],
            },
        ],
        "links": [],
    }


# ---------------------------------------------------------------------------
# Template-mode workflow fixture
# ---------------------------------------------------------------------------


def _template_workflow():
    """A frontend-format workflow with a subgraph instance."""
    return {
        "nodes": [
            {
                "id": 1,
                "type": "MyTemplate",
                "properties": {
                    "proxyWidgets": [
                        [10, "text"],
                        [11, "seed"],
                    ],
                },
            },
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "name": "MyTemplate",
                    "inputs": [
                        {"name": "text", "type": "STRING"},
                        {"name": "seed", "type": "INT"},
                    ],
                    "nodes": [
                        {
                            "id": 10,
                            "type": "CLIPTextEncode",
                            "widgets_values": ["hello world"],
                        },
                        {
                            "id": 11,
                            "type": "KSampler",
                            "widgets_values": [42, "fixed", 20, 8.0, "euler", "normal", 1.0],
                        },
                    ],
                },
            ],
        },
    }


# ===========================================================================
# TestWidgetOrder
# ===========================================================================


class TestWidgetOrder:
    """Tests graph.widget_order(class_name)."""

    def test_ksampler_order(self, graph: Graph):
        order = graph.widget_order("KSampler")
        assert order == [
            "seed",
            "control_after_generate",
            "steps",
            "cfg",
            "sampler_name",
            "scheduler",
            "denoise",
        ]

    def test_clip_text_encode_order(self, graph: Graph):
        order = graph.widget_order("CLIPTextEncode")
        assert order == ["text"]

    def test_no_widgets_returns_empty(self, graph: Graph):
        order = graph.widget_order("VAEDecode")
        assert order == []

    def test_unknown_node_returns_empty(self, graph: Graph):
        order = graph.widget_order("Nonexistent")
        assert order == []


# ===========================================================================
# TestWidgetOrderDynamicCombo
# ===========================================================================


def _dynamic_combo_object_info() -> dict:
    """A COMFY_DYNAMICCOMBO_V3 node with a nested dynamic combo among one
    option's sub-inputs, plus connection-only subs that own no value slot."""
    return {
        "DynNode": {
            "input": {
                "required": {
                    "prompt": ["STRING", {"default": ""}],
                    "model": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {
                            "options": [
                                {
                                    "key": "alpha",
                                    "inputs": {
                                        "required": {
                                            "size": ["COMBO", {"options": ["S", "M"]}],
                                            "width": ["INT", {"default": 512}],
                                            "images": ["COMFY_AUTOGROW_V3", {"min": 0}],
                                        }
                                    },
                                },
                                {
                                    "key": "beta",
                                    "inputs": {
                                        "required": {
                                            "mode": [
                                                "COMFY_DYNAMICCOMBO_V3",
                                                {
                                                    "options": [
                                                        {
                                                            "key": "fast",
                                                            "inputs": {"required": {"steps": ["INT", {"default": 4}]}},
                                                        },
                                                        {
                                                            "key": "slow",
                                                            "inputs": {
                                                                "required": {
                                                                    "steps": ["INT", {"default": 50}],
                                                                    "refine": ["BOOLEAN", {"default": True}],
                                                                }
                                                            },
                                                        },
                                                    ]
                                                },
                                            ],
                                        }
                                    },
                                },
                            ]
                        },
                    ],
                    "seed": ["INT", {"default": 0, "control_after_generate": True}],
                },
            },
            "input_order": {"required": ["prompt", "model", "seed"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "test",
            "display_name": "DynNode",
            "python_module": "nodes",
        }
    }


class TestWidgetOrderDynamicCombo:
    """Value-aware order expansion for COMFY_DYNAMICCOMBO_V3 ports."""

    @pytest.fixture
    def dyn_graph(self) -> Graph:
        return Graph.from_object_info(_dynamic_combo_object_info())

    def test_dynamic_combo_is_widget_not_link(self, dyn_graph: Graph):
        m = dyn_graph.node("DynNode")
        model = next(p for p in m.inputs if p.name == "model")
        assert model.is_link is False
        assert model.enum_values == ["alpha", "beta"]
        assert len(model.dynamic_options) == 2

    def test_value_independent_order_has_selector_only(self, dyn_graph: Graph):
        assert dyn_graph.widget_order("DynNode") == ["prompt", "model", "seed", "control_after_generate"]

    def test_value_aware_order_expands_selected_option(self, dyn_graph: Graph):
        order = dyn_graph.widget_order_for_node("DynNode", ["p", "alpha", "S", 512, 0, "fixed"])
        # connection-only sub (COMFY_AUTOGROW_V3 images) contributes no slot.
        assert order == ["prompt", "model", "model.size", "model.width", "seed", "control_after_generate"]

    def test_value_aware_order_recurses_nested_dynamic_combo(self, dyn_graph: Graph):
        order = dyn_graph.widget_order_for_node("DynNode", ["p", "beta", "slow", 50, True, 0, "fixed"])
        assert order == [
            "prompt",
            "model",
            "model.mode",
            "model.mode.steps",
            "model.mode.refine",
            "seed",
            "control_after_generate",
        ]

    def test_unknown_selector_expands_nothing(self, dyn_graph: Graph):
        order = dyn_graph.widget_order_for_node("DynNode", ["p", "gone", 0, "fixed"])
        assert order == ["prompt", "model", "seed", "control_after_generate"]

    def test_nested_selector_change_rebuilds_inner_roster(self, dyn_graph: Graph):
        wf = {"nodes": [{"id": 1, "type": "DynNode", "widgets_values": ["p", "beta", "fast", 4, 7, "fixed"]}]}
        out, warnings = dyn_graph.apply_slots(wf, {"1.model.mode": "slow"})
        assert [w["code"] for w in warnings] == ["dynamic_combo_roster_rebuilt"]
        # fast's [steps=4] roster is replaced by slow's defaults [steps=50, refine=True];
        # the trailing seed + control marker stay aligned.
        assert out["nodes"][0]["widgets_values"] == ["p", "beta", "slow", 50, True, 7, "fixed"]

    def test_outer_selector_change_replaces_nested_span(self, dyn_graph: Graph):
        wf = {"nodes": [{"id": 1, "type": "DynNode", "widgets_values": ["p", "beta", "slow", 50, True, 7, "fixed"]}]}
        out, warnings = dyn_graph.apply_slots(wf, {"1.model": "alpha"})
        assert [w["code"] for w in warnings] == ["dynamic_combo_roster_rebuilt"]
        # the whole nested span (mode, mode.steps, mode.refine) is replaced by
        # alpha's defaults (size first-enum, width default).
        assert out["nodes"][0]["widgets_values"] == ["p", "alpha", "S", 512, 7, "fixed"]


def _dynamic_combo_implicit_seed_object_info() -> dict:
    """A COMFY_DYNAMICCOMBO_V3 option whose sub-input is an implicit
    seed/noise_seed INT — the frontend's ``useIntWidget`` composable
    companions it with a control_after_generate marker even without the
    schema's ``control_after_generate`` flag (mirrors
    ``workflow_to_api._has_control_after_generate_companion``)."""
    return {
        "SeedComboNode": {
            "input": {
                "required": {
                    "mode": [
                        "COMFY_DYNAMICCOMBO_V3",
                        {"options": [{"key": "a", "inputs": {"required": {"seed": ["INT", {"default": 0}]}}}]},
                    ],
                },
            },
            "input_order": {"required": ["mode"]},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "category": "test",
            "display_name": "SeedComboNode",
            "python_module": "nodes",
        }
    }


def _prefixed_dynamic_combo_object_info() -> dict:
    """Same as above but with a leading widget, so the combo's selector sits
    at a non-zero positional index — needed to exercise padding-before-write."""
    info = _dynamic_combo_implicit_seed_object_info()
    info["PrefixedDynNode"] = info.pop("SeedComboNode")
    info["PrefixedDynNode"]["display_name"] = "PrefixedDynNode"
    info["PrefixedDynNode"]["input"]["required"] = {
        "prefix": ["STRING", {"default": ""}],
        **info["PrefixedDynNode"]["input"]["required"],
    }
    info["PrefixedDynNode"]["input_order"] = {"required": ["prefix", "mode"]}
    return info


class TestDynamicComboImplicitControlAfterGenerate:
    """Sub-input seed/noise_seed widgets companion a control_after_generate
    marker even without the schema flag — same rule as the UI→API converter."""

    @pytest.fixture
    def seed_graph(self) -> Graph:
        return Graph.from_object_info(_dynamic_combo_implicit_seed_object_info())

    def test_value_aware_order_includes_implicit_marker(self, seed_graph: Graph):
        order = seed_graph.widget_order_for_node("SeedComboNode", ["a", 0, "fixed"])
        assert order == ["mode", "mode.seed", "control_after_generate"]

    def test_roster_rebuild_synthesizes_implicit_marker_default(self, seed_graph: Graph):
        wf = {"nodes": [{"id": 1, "type": "SeedComboNode", "widgets_values": [None]}]}
        out, warnings = seed_graph.apply_slots(wf, {"1.mode": "a"})
        assert [w["code"] for w in warnings] == ["dynamic_combo_roster_rebuilt"]
        assert out["nodes"][0]["widgets_values"] == ["a", 0, "fixed"]

    def test_roster_rebuild_respects_extend_false(self):
        graph = Graph.from_object_info(_prefixed_dynamic_combo_object_info())
        node = {"id": 1, "type": "PrefixedDynNode", "widgets_values": []}
        with pytest.raises(ValueError, match="out of range"):
            _write_widget(node, "mode", "a", graph, extend=False)


# ===========================================================================
# TestTraversal
# ===========================================================================


class TestTraversal:
    """Tests upstream, downstream, find_paths, exact_paths."""

    def test_upstream_ksampler(self, graph: Graph):
        ups = graph.upstream("KSampler")
        ids = {m.id for m in ups}
        assert "CheckpointLoaderSimple" in ids  # produces MODEL
        assert "CLIPTextEncode" in ids  # produces CONDITIONING
        assert "EmptyLatentImage" in ids  # produces LATENT

    def test_downstream_checkpoint(self, graph: Graph):
        downs = graph.downstream("CheckpointLoaderSimple")
        ids = {m.id for m in downs}
        assert "KSampler" in ids  # accepts MODEL
        assert "CLIPTextEncode" in ids  # accepts CLIP
        assert "VAEDecode" in ids  # accepts VAE

    def test_upstream_unknown_returns_empty(self, graph: Graph):
        assert graph.upstream("Ghost") == []

    def test_downstream_unknown_returns_empty(self, graph: Graph):
        assert graph.downstream("Ghost") == []

    def test_find_paths_model_to_image(self, graph: Graph):
        paths = graph.find_paths("MODEL", "IMAGE")
        assert len(paths) >= 1
        # Every path should go from MODEL to IMAGE
        for p in paths:
            assert p["from"] == "MODEL"
            assert p["to"] == "IMAGE"

    def test_exact_paths_model_to_image(self, graph: Graph):
        paths = graph.exact_paths("MODEL", "IMAGE")
        assert len(paths) >= 1
        for p in paths:
            assert p["from"] == "MODEL"
            assert p["to"] == "IMAGE"
            # Each step's node should exist in the graph
            for step in p["steps"]:
                assert graph.node(step["node"]) is not None

    def test_find_paths_same_type_is_searched_not_declined(self, graph: Graph):
        """Same-type queries used to be refused outright; they are now walked
        like any other. This small catalog happens to hold no route back to
        MODEL — nothing here consumes MODEL and emits it — so the empty result
        is a fact about the catalog rather than an abstention. The catalog that
        *does* carry one (`LoraLoaderModelOnly`) is the `graph_path` fixture,
        pinned by `test_same_type_query_finds_the_route` below.
        """
        assert graph.find_paths("MODEL", "MODEL") == []
        result = graph.search_paths("MODEL", "MODEL", exact=False, max_depth=4)
        assert result["paths"] == []
        assert result["not_searched"] is False
        assert result["not_searched_reason"] is None

    def test_find_paths_unreachable_returns_empty(self, graph: Graph):
        # No node consumes IMAGE and produces MODEL in this fixture
        assert graph.find_paths("IMAGE", "MODEL") == []


# ===========================================================================
# TestPathConstraints — BE-6857
# ===========================================================================


def _node_chain(path: dict) -> tuple[str, ...]:
    return tuple(s["node"] for s in path["steps"])


class TestPathConstraints:
    """`nodes path` used to enumerate anything that *produced* the target type,
    ignoring the source type entirely: `AUDIO -> IMAGE` returned the same rows
    as `MODEL -> IMAGE`, every step carried an empty `input_type`, and the
    result was still labelled exact. These pin the source constraint, the depth
    bound, and the honesty of the exhaustiveness claim.
    """

    def test_unreachable_source_type_returns_no_paths(self, graph_path: Graph):
        # In THIS fixture AUDIO is consumed (SaveAudio, PreviewAudio) but never
        # routed to IMAGE, so the correct answer is the empty set — not MODEL's
        # rows. The emptiness is a fact about the catalog, never a hard-coded
        # denial; the next test is the falsifier that pins that distinction.
        assert graph_path.exact_paths("AUDIO", "IMAGE", max_depth=6) == []
        assert graph_path.find_paths("AUDIO", "IMAGE", max_depth=6) == []

    def test_real_audio_to_image_route_is_found_when_the_catalog_has_one(self, graph_path: Graph):
        """`AUDIO -> IMAGE` is NOT inherently impossible, and this walker must
        never treat it that way.

        Current ComfyUI ships `VAEEncodeAudio` (AUDIO + VAE -> LATENT), which
        reaches IMAGE through the ordinary `VAEDecode` hop. Add that real node
        to the catalog and the route has to appear — with the VAE it also needs
        reported as support rather than silently assumed.
        """
        info = copy.deepcopy(graph_path.object_info)
        # Faithful to comfy_extras/nodes_audio.py::VAEEncodeAudio.
        info["VAEEncodeAudio"] = {
            "input": {"required": {"audio": ["AUDIO", {}], "vae": ["VAE", {}]}},
            "input_order": {"required": ["audio", "vae"]},
            "output": ["LATENT"],
            "output_is_list": [False],
            "output_name": ["LATENT"],
            "name": "VAEEncodeAudio",
            "display_name": "VAE Encode Audio",
            "description": "",
            "category": "model/latent",
            "python_module": "comfy_extras.nodes_audio",
            "output_node": False,
            "search_aliases": ["audio to latent"],
        }
        graph = Graph.from_object_info(info)

        paths = graph.exact_paths("AUDIO", "IMAGE", max_depth=6)
        chains = {_node_chain(p) for p in paths}
        assert ("VAEEncodeAudio", "VAEDecode") in chains
        assert ("VAEEncodeAudio", "VAEDecodeTiled") in chains
        # Every hop is a declared link of the type it claims to consume.
        for p in paths:
            assert p["steps"][0]["input_type"] == "AUDIO"
            assert graph.node("VAEEncodeAudio").has_input("AUDIO")
        # The VAE that VAEEncodeAudio also needs is surfaced, not assumed away.
        route = next(p for p in paths if _node_chain(p) == ("VAEEncodeAudio", "VAEDecode"))
        assert "VAE" in {s["type"] for s in route["support"]}

    def test_unknown_source_type_returns_no_paths(self, graph_path: Graph):
        assert graph_path.exact_paths("NOT_A_TYPE", "IMAGE", max_depth=6) == []

    def test_source_type_changes_the_answer(self, graph_path: Graph):
        model = graph_path.exact_paths("MODEL", "IMAGE", max_depth=6)
        audio = graph_path.exact_paths("AUDIO", "IMAGE", max_depth=6)
        assert model, "MODEL -> IMAGE should still route through the sampler"
        assert model != audio

    def test_first_step_consumes_the_declared_source_type(self, graph_path: Graph):
        for from_type in ("MODEL", "LATENT", "CLIP", "CONDITIONING"):
            for p in graph_path.exact_paths(from_type, "IMAGE", max_depth=6):
                first = p["steps"][0]
                assert first["input_type"] == from_type
                assert graph_path.node(first["node"]).has_input(from_type)

    def test_every_step_declares_a_link_input_of_its_from_type(self, graph_path: Graph):
        for p in graph_path.exact_paths("CLIP", "IMAGE", max_depth=6):
            previous_out = "CLIP"
            for step in p["steps"]:
                node = graph_path.node(step["node"])
                assert step["input_type"] == previous_out
                assert step["input_type"], "every step reports the type it consumes"
                assert node.has_input(step["input_type"])
                assert node.has_output(step["output_type"])
                previous_out = step["output_type"]

    def test_widget_named_model_is_not_a_model_input(self, graph_path: Graph):
        """ByteDanceImageNode produces IMAGE and has a *widget* named `model`
        (a COMBO of API ids) — never a MODEL link input, so it is not a routing
        step for MODEL, nor for any other type."""
        bytedance = graph_path.node("ByteDanceImageNode")
        assert bytedance.has_output("IMAGE")
        assert bytedance.input_link_types() == []
        for from_type in ("MODEL", "AUDIO", "CLIP", "LATENT"):
            for p in graph_path.exact_paths(from_type, "IMAGE", max_depth=6):
                assert "ByteDanceImageNode" not in _node_chain(p)

    def test_max_depth_bounds_path_length(self, graph_path: Graph):
        for depth in range(1, 7):
            for p in graph_path.exact_paths("CLIP", "IMAGE", max_depth=depth):
                assert len(p["steps"]) <= depth

    def test_shallower_depth_is_a_subset(self, graph_path: Graph):
        deep = {_node_chain(p) for p in graph_path.exact_paths("CLIP", "IMAGE", max_depth=6)}
        assert deep
        for depth in range(1, 6):
            shallow = {_node_chain(p) for p in graph_path.exact_paths("CLIP", "IMAGE", max_depth=depth)}
            assert shallow <= deep
        # The reported case: depth 1 is a *strict* subset of depth 4.
        shallow = {_node_chain(p) for p in graph_path.exact_paths("MODEL", "IMAGE", max_depth=1)}
        deep = {_node_chain(p) for p in graph_path.exact_paths("MODEL", "IMAGE", max_depth=4)}
        assert shallow < deep

    def test_support_nodes_cover_the_other_required_inputs(self, graph_path: Graph):
        (path,) = [p for p in graph_path.exact_paths("MODEL", "IMAGE", max_depth=6) if "VAEDecode" in _node_chain(p)]
        support = {s["type"]: s["node"] for s in path["support"]}
        # KSampler needs conditioning + an initial latent, VAEDecode needs a VAE
        assert support["CONDITIONING"] == "CLIPTextEncode"
        assert support["LATENT"] == "EmptyLatentImage"
        assert support["VAE"] == "CheckpointLoaderSimple"
        # …and the routed type itself is never listed as support.
        assert "MODEL" not in support

    def test_free_types_excludes_types_nothing_can_produce(self, graph_path: Graph):
        free = graph_path.free_types()
        assert {"MODEL", "LATENT", "IMAGE", "AUDIO"} <= free
        assert "NOT_A_TYPE" not in free

    def test_exhausted_search_reports_no_truncation(self, graph_path: Graph):
        result = graph_path.search_paths("AUDIO", "IMAGE", max_depth=6)
        assert result["paths"] == []
        assert result["truncated"] is False
        assert result["depth_limited"] is False
        assert result["collapsed"] is False

    def test_collapsed_alternate_routes_are_reported(self, graph_path: Graph):
        """The walk explores each intermediate state once, so a second node
        offering the same hop is not re-expanded and the chains through it are
        never printed. That is a real gap in the *listing*, so it has to be
        reported — silently returning a subset while claiming exactness is the
        bug this ticket is about, one level down.
        """
        info = copy.deepcopy(graph_path.object_info)
        # A second MODEL -> LATENT sampler: a genuine alternate first hop.
        info["KSamplerAdvanced"] = copy.deepcopy(info["KSampler"])
        info["KSamplerAdvanced"]["name"] = "KSamplerAdvanced"
        graph = Graph.from_object_info(info)

        result = graph.search_paths("MODEL", "IMAGE", max_depth=3)
        chains = {_node_chain(p) for p in result["paths"]}
        # Both decoders are reported off the surviving sampler...
        assert chains == {("KSampler", "VAEDecode"), ("KSampler", "VAEDecodeTiled")}
        # ...but KSamplerAdvanced's equally valid routes are not, so the result
        # must not be advertised as the complete set.
        assert result["collapsed"] is True
        assert result["truncated"] is False
        assert result["depth_limited"] is False

    def test_max_paths_is_reported_as_truncation(self, graph_path: Graph):
        full = graph_path.search_paths("LATENT", "IMAGE", max_depth=6)
        assert len(full["paths"]) > 1 and full["truncated"] is False
        capped = graph_path.search_paths("LATENT", "IMAGE", max_depth=6, max_paths=1)
        assert len(capped["paths"]) == 1
        assert capped["truncated"] is True
        assert capped["truncated_by"] == "max_paths"

    def test_depth_cut_is_reported(self, graph_path: Graph):
        result = graph_path.search_paths("MODEL", "IMAGE", max_depth=1)
        assert result["paths"] == []
        assert result["depth_limited"] is True

    def test_state_budget_is_reported_as_truncation(self, graph_path: Graph):
        result = graph_path.search_paths("CLIP", "IMAGE", max_depth=6, max_states=1)
        assert result["truncated"] is True
        assert result["truncated_by"] == "max_states"

    def test_degenerate_bounds_return_nothing(self, graph_path: Graph):
        # `MODEL -> MODEL` used to sit here as a third degenerate case. It is no
        # longer degenerate — a same-type query is a real question with a real
        # answer (see `test_same_type_query_finds_the_route`), so only the
        # bounds no path can satisfy remain.
        assert graph_path.search_paths("MODEL", "IMAGE", max_depth=0)["paths"] == []
        assert graph_path.search_paths("MODEL", "IMAGE", max_paths=0)["paths"] == []

    @pytest.mark.parametrize(
        ("kwargs", "reason"),
        [
            ({"from_type": "MODEL", "to_type": "IMAGE", "max_depth": 0}, "degenerate_bounds"),
            ({"from_type": "MODEL", "to_type": "IMAGE", "max_paths": 0}, "degenerate_bounds"),
        ],
    )
    def test_declined_queries_declare_the_abstention(self, graph_path: Graph, kwargs, reason):
        """The query shapes the walk refuses return an empty result. An empty
        result with every limit flag false is this module's proof that no path
        exists, so a refusal that stayed silent would forge that proof. Each
        one says so instead.
        """
        from_type = kwargs.pop("from_type")
        to_type = kwargs.pop("to_type")
        result = graph_path.search_paths(from_type, to_type, **kwargs)
        assert result["paths"] == []
        assert result["not_searched"] is True
        assert result["not_searched_reason"] == reason
        # No limit flag is set — which is exactly why the abstention needs its
        # own signal rather than being inferred from the others.
        assert result["truncated"] is False
        assert result["depth_limited"] is False
        assert result["collapsed"] is False

    def test_same_type_query_finds_the_route(self, graph_path: Graph):
        """`LoraLoaderModelOnly` in the fixture takes a MODEL link input and
        emits MODEL, so `MODEL -> MODEL` is genuinely routable — and is now
        answered rather than declined. The walker used to refuse the query
        outright and report the empty result as an abstention; the no-op rule
        (`out_t == cur_type`) no longer drops the hop that answers it.
        """
        lora = graph_path.node("LoraLoaderModelOnly")
        assert lora is not None and "MODEL" in lora.output_types()
        assert lora.has_input("MODEL")

        result = graph_path.search_paths("MODEL", "MODEL")
        assert ("LoraLoaderModelOnly",) in {_node_chain(p) for p in result["paths"]}
        # A real walk, not an abstention — and the one-step route is a genuine
        # MODEL-in/MODEL-out hop, not a mislabelled edge.
        assert result["not_searched"] is False
        assert result["not_searched_reason"] is None
        one_step = next(p for p in result["paths"] if _node_chain(p) == ("LoraLoaderModelOnly",))
        assert one_step["from"] == "MODEL" and one_step["to"] == "MODEL"
        assert one_step["steps"] == [{"node": "LoraLoaderModelOnly", "input_type": "MODEL", "output_type": "MODEL"}]

    def test_no_op_hops_are_still_dropped(self, graph_path: Graph):
        """The exemption is scoped to the hop that answers a same-type query,
        and to nothing else — a step that hands back the type it consumed is
        still a no-op everywhere it is not the terminal step.

        For a FROM != TO query that means *no* step may do it at all: a step
        whose output equals the target ends the path, so a no-op-looking step
        requires the incoming type to already be the target, which only the
        first frontier item can satisfy.
        """
        for from_type in ("MODEL", "LATENT", "CLIP", "CONDITIONING"):
            for p in graph_path.exact_paths(from_type, "IMAGE", max_depth=6):
                assert all(s["input_type"] != s["output_type"] for s in p["steps"]), (
                    f"no-op hop in {from_type} -> IMAGE via {_node_chain(p)}"
                )
        # And within a same-type query it is the terminal hop only.
        same_type = graph_path.exact_paths("MODEL", "MODEL", max_depth=6)
        assert same_type, "fixture must offer at least one MODEL -> MODEL route"
        for p in same_type:
            for i, step in enumerate(p["steps"]):
                if step["input_type"] == step["output_type"]:
                    assert i == len(p["steps"]) - 1, f"no-op mid-path in {_node_chain(p)}"
                    assert step["output_type"] == "MODEL"

    def test_completed_walks_are_not_marked_as_declined(self, graph_path: Graph):
        """The abstention flag must stay off for searches that actually ran,
        whether they found routes or genuinely exhausted the space."""
        found = graph_path.search_paths("MODEL", "IMAGE", max_depth=4)
        assert found["paths"] and found["not_searched"] is False
        empty = graph_path.search_paths("AUDIO", "IMAGE", max_depth=6)
        assert empty["paths"] == [] and empty["not_searched"] is False
        assert empty["not_searched_reason"] is None


# ===========================================================================
# TestValidateWorkflow
# ===========================================================================


class TestValidateWorkflow:
    """Tests graph.validate_workflow(api_workflow)."""

    @staticmethod
    def _errors_excluding_no_outputs(result: dict) -> list[dict]:
        """Errors other than the workflow-level no-outputs check — for
        single-node fixtures that (deliberately) carry no output node."""
        return [e for e in result["errors"] if e.get("code") != "prompt_no_outputs"]

    def _valid_workflow(self) -> dict:
        # A complete, server-valid pipeline: every required input present and a
        # SaveImage output node (so it passes the required-presence and
        # no-outputs checks, not just the edge/shape checks).
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
            "3": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "positive prompt"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["1", 1], "text": "negative prompt"},
            },
            "5": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 512, "height": 512, "batch_size": 1},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                    "seed": 42,
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
            "6": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["2", 0], "vae": ["1", 2]},
            },
            "7": {
                "class_type": "SaveImage",
                "inputs": {"images": ["6", 0], "filename_prefix": "out"},
            },
        }

    def test_valid_workflow(self, graph: Graph):
        result = graph.validate_workflow(self._valid_workflow())
        assert result["valid"] is True
        assert result["errors"] == []

    def test_non_node_key_warns(self, graph: Graph):
        """An unrecognized non-node key should produce a warning, not an error."""
        wf = {**self._valid_workflow(), "notanode": {"title": "My Workflow"}}
        result = graph.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        non_node = [w for w in result["warnings"] if w["code"] == "non_node_key"]
        assert len(non_node) == 1
        assert non_node[0]["node_id"] == "notanode"
        assert non_node[0]["field"] == "notanode"

    def test_meta_provenance_key_is_not_warned(self, graph: Graph):
        """`_meta` is the compose/run provenance block (stripped before submit),
        not a stray key — validating composed output must not nag about it."""
        wf = {"_meta": {"schema": "compose/1", "blueprint": "blueprints/x.yaml"}, **self._valid_workflow()}
        result = graph.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        assert [w for w in result["warnings"] if w["node_id"] == "_meta"] == []

    def test_non_dict_node_value_warns(self, graph: Graph):
        """A string value for a key should warn, not crash."""
        wf = {**self._valid_workflow(), "_comment": "this is a comment"}
        result = graph.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        non_node = [w for w in result["warnings"] if w["code"] == "non_node_key"]
        assert len(non_node) == 1
        assert non_node[0]["node_id"] == "_comment"

    def test_unknown_class_type(self, graph: Graph):
        wf = {"1": {"class_type": "KSamper", "inputs": {}}}
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        err = result["errors"][0]
        assert err["code"] == "unknown_class_type"
        assert "KSampler" in err["suggestions"]

    def test_shape_mismatch_string_for_int(self, graph: Graph):
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": "hello"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "shape_mismatch"]
        assert len(errs) == 1
        assert errs[0]["field"] == "seed"

    def test_shape_mismatch_bool_for_int(self, graph: Graph):
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"seed": True},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "shape_mismatch"]
        assert len(errs) == 1
        assert errs[0]["field"] == "seed"

    def test_unknown_enum_value(self, graph: Graph):
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"sampler_name": "nonexistent_sampler"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "unknown_enum_value"]
        assert len(errs) == 1
        assert isinstance(errs[0]["suggestions"], list)
        assert "euler" in errs[0]["suggestions"]

    def test_valid_edges_pass(self, graph: Graph):
        """Well-wired edges don't produce errors."""
        result = graph.validate_workflow(self._valid_workflow())
        assert result["valid"] is True
        assert result["errors"] == []

    def test_dangling_edge(self, graph: Graph):
        """Edge to a node that doesn't exist in the workflow."""
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"model": ["99", 0]},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "dangling_edge"]
        assert len(errs) == 1
        assert "99" in errs[0]["message"]

    def test_output_index_out_of_range(self, graph: Graph):
        """Edge references an output index that doesn't exist."""
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                # CheckpointLoaderSimple has 3 outputs (0=MODEL, 1=CLIP, 2=VAE)
                # Index 5 is out of range
                "inputs": {"model": ["1", 5]},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "output_index_out_of_range"]
        assert len(errs) == 1
        assert "3 output" in errs[0]["message"]

    def test_edge_type_mismatch(self, graph: Graph):
        """Edge connects wrong type: CLIP fed into a MODEL input.

        This is advisory (warning, not error) — ComfyUI allows cross-type
        wiring via reroutes and converters; the server is the authority."""
        wf = self._valid_workflow()
        # Output index 1 is CLIP, but the model input expects MODEL — still a
        # present input, so only an advisory warning (not a hard error).
        wf["2"]["inputs"]["model"] = ["1", 1]
        result = graph.validate_workflow(wf)
        # edge_type_mismatch is a warning, not a hard error
        assert result["valid"] is True, result["errors"]
        warns = [w for w in result["warnings"] if w["code"] == "edge_type_mismatch"]
        assert len(warns) == 1
        assert "CLIP" in warns[0]["message"]
        assert "MODEL" in warns[0]["message"]

    def test_int_valued_combo_accepts_int(self, graph: Graph):
        """Server combos can be int-valued (LTXV duration/fps). An int value
        must not be rejected as a shape mismatch."""
        wf = {
            "1": {
                "class_type": "LtxvApiTextToVideo",
                "inputs": {"prompt": "a boat", "duration": 8, "fps": 25, "resolution": "1920x1080"},
            },
        }
        result = graph.validate_workflow(wf)
        # This single-node fixture has no output node, so the only error is the
        # workflow-level no-outputs one; the int-valued combo itself is clean.
        assert self._errors_excluding_no_outputs(result) == []

    def test_int_valued_combo_unknown_option_is_enum_error(self, graph: Graph):
        """An int outside the combo's options is an unknown_enum_value (same as
        a bad string combo) — caught by membership, not mislabeled as a shape
        mismatch."""
        wf = {
            "1": {
                "class_type": "LtxvApiTextToVideo",
                "inputs": {"prompt": "a boat", "duration": 7, "fps": 25, "resolution": "1920x1080"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        # 7 is not in [6, 8, 10, 12]; it's an enum error, not a shape error.
        errs = [e for e in result["errors"] if e.get("field") == "duration"]
        assert len(errs) == 1
        assert errs[0]["code"] == "unknown_enum_value"

    def test_combo_rejects_wrong_shape(self, graph: Graph):
        """A list/dict for a COMBO is still a hard shape mismatch."""
        wf = {
            "1": {
                "class_type": "LtxvApiTextToVideo",
                "inputs": {"prompt": "x", "duration": {"bad": 1}, "fps": 25, "resolution": "1920x1080"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "shape_mismatch"]
        assert any(e["field"] == "duration" for e in errs)

    def test_dict_form_combo_keeps_options(self, graph: Graph):
        """Dict-form COMBO specs (["COMBO", {"options": [...]}]) must retain
        their enum so unknown values are caught — the partner-node case."""
        m = graph._nodes["LtxvApiTextToVideo"]
        resolution = next(p for p in m.inputs if p.name == "resolution")
        assert resolution.enum_values == ["1920x1080", "2560x1440"]

        wf = {
            "1": {
                "class_type": "LtxvApiTextToVideo",
                "inputs": {"prompt": "x", "duration": 8, "fps": 25, "resolution": "640x480"},
            },
        }
        result = graph.validate_workflow(wf)
        errs = [e for e in result["errors"] if e["code"] == "unknown_enum_value"]
        assert any(e["field"] == "resolution" for e in errs)

    def test_int_combo_preserves_int_type(self, graph: Graph):
        """object_info int-valued combos must keep their int type so `nodes show`
        tells the truth and agents pass 8 (not "8") — the cloud rejects the
        string form (the Sora-2 `duration` bug)."""
        m = graph._nodes["LtxvApiTextToVideo"]
        duration = next(p for p in m.inputs if p.name == "duration")
        assert duration.enum_values == [6, 8, 10, 12]
        assert all(isinstance(v, int) for v in duration.enum_values)

    def test_enum_error_carries_full_valid_options(self, graph: Graph):
        """A rejection must surface the FULL valid list (typed), not a truncated
        preview — so an agent can pick a real value instead of guessing."""
        wf = {
            "1": {
                "class_type": "LtxvApiTextToVideo",
                "inputs": {"prompt": "x", "duration": 7, "fps": 25, "resolution": "1920x1080"},
            },
        }
        result = graph.validate_workflow(wf)
        err = next(e for e in result["errors"] if e["field"] == "duration")
        assert err["valid_options"] == [6, 8, 10, 12]

    def test_int_combo_accepts_string_form_leniently(self, graph: Graph):
        """Local validate stays lenient on type (string "8" still matches int 8)
        so it never false-warns; truthfulness comes from the displayed schema,
        not from stricter local validation."""
        wf = {
            "1": {
                "class_type": "LtxvApiTextToVideo",
                "inputs": {"prompt": "x", "duration": "8", "fps": 25, "resolution": "1920x1080"},
            },
        }
        result = graph.validate_workflow(wf)
        # Lenient on type (no combo error); the only error is the no-outputs one.
        assert self._errors_excluding_no_outputs(result) == []

    def test_wildcard_type_compatible(self, graph: Graph):
        """'*' type on either side should not trigger a mismatch."""
        # Add a wildcard node to the graph for this test
        from comfy_cli.cql.engine import Graph as G

        oi = _object_info()
        oi["Reroute"] = {
            "input": {"required": {"input": "*"}},
            "input_order": {"required": ["input"]},
            "output": ["*"],
            "output_name": ["output"],
            "category": "utils",
        }
        g = G.from_object_info(oi)
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
            "2": {
                "class_type": "Reroute",
                "inputs": {"input": ["1", 0]},  # MODEL into *
            },
            "3": {
                "class_type": "KSampler",
                "inputs": {"model": ["2", 0]},  # * into MODEL
            },
        }
        result = g.validate_workflow(wf)
        edge_errs = [e for e in result["errors"] if e["code"] == "edge_type_mismatch"]
        assert edge_errs == []

    def test_multiple_edge_errors_reported(self, graph: Graph):
        """All edge errors are reported, not just the first."""
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["99", 0],  # dangling
                    "positive": ["98", 0],  # dangling
                    "latent_image": ["97", 0],  # dangling
                },
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        dangling = [e for e in result["errors"] if e["code"] == "dangling_edge"]
        assert len(dangling) == 3

    def test_below_min_error(self, graph: Graph):
        """A value below the catalog min is a hard error (the server rejects it
        with value_smaller_than_min) — was a warning before BE-3357. Node "1" is
        wired to a SaveImage output so it is server-reachable (BE-3406); an
        unreachable node would be pruned and the range demoted to a warning."""
        wf = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 0, "height": 512, "batch_size": 1},
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "out"}},
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "below_min"]
        assert len(errs) == 1
        assert errs[0]["field"] == "width"
        # No longer surfaced as a warning.
        assert "below_min" not in [w["code"] for w in result["warnings"]]

    def test_above_max_error(self, graph: Graph):
        """A value above the catalog max is a hard error (value_bigger_than_max).
        Node "1" is wired to a SaveImage output so it is server-reachable."""
        wf = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 999999, "height": 512, "batch_size": 1},
            },
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "out"}},
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "above_max"]
        assert len(errs) == 1
        assert errs[0]["field"] == "width"
        assert "above_max" not in [w["code"] for w in result["warnings"]]


class TestAutogrowInputs:
    """COMFY_AUTOGROW_V3 inputs (e.g. BatchImagesNode.images): the schema
    declares ONE input, the server expects autogrown slot keys
    `images.image0`, `images.image1`, … — one per connection."""

    def _loaders(self) -> dict:
        # Two IMAGE producers from the fixture catalog (VAEDecode → IMAGE).
        return {
            "10": {"class_type": "VAEDecode", "inputs": {}},
            "11": {"class_type": "VAEDecode", "inputs": {}},
        }

    def test_dotted_slots_validate_clean(self, graph: Graph):
        # A fully server-valid workflow: two IMAGE producers with all their
        # required inputs wired, autogrown into BatchImagesNode, terminating in
        # a SaveImage output node.
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "sd_xl_base.safetensors"}},
            "2": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "10": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 2]}},
            "11": {"class_type": "VAEDecode", "inputs": {"samples": ["2", 0], "vae": ["1", 2]}},
            "20": {
                "class_type": "BatchImagesNode",
                "inputs": {"images.image0": ["10", 0], "images.image1": ["11", 0]},
            },
            "30": {"class_type": "SaveImage", "inputs": {"images": ["20", 0], "filename_prefix": "out"}},
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        # The dotted slots must not trip type-mismatch or unknown-input noise.
        assert result["warnings"] == []

    def test_bare_link_wiring_errors_with_slot_hint(self, graph: Graph):
        wf = {
            **self._loaders(),
            "20": {
                "class_type": "BatchImagesNode",
                "inputs": {"images": ["10", 0]},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        err = next(e for e in result["errors"] if e["code"] == "autogrow_bare_input")
        assert err["node_id"] == "20"
        assert "images.image0" in err["hint"]

    def test_required_autogrow_with_no_slots_errors(self, graph: Graph):
        # BatchImagesNode "20" is wired to a SaveImage output so it is
        # server-reachable (BE-3406) — an unreachable node would be pruned.
        wf = {
            "20": {"class_type": "BatchImagesNode", "inputs": {}},
            "30": {"class_type": "SaveImage", "inputs": {"images": ["20", 0], "filename_prefix": "out"}},
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        err = next(e for e in result["errors"] if e["code"] == "autogrow_no_slots")
        assert err["node_id"] == "20"
        assert "images.image0" in err["hint"]

    def test_dangling_dotted_slot_still_checked(self, graph: Graph):
        wf = {
            "20": {
                "class_type": "BatchImagesNode",
                "inputs": {"images.image0": ["99", 0]},
            },
        }
        result = graph.validate_workflow(wf)
        codes = [e["code"] for e in result["errors"]]
        assert "dangling_edge" in codes

    def test_describe_marks_autogrow(self, graph: Graph):
        desc = graph.morphism_to_dict(graph.node("BatchImagesNode"))
        images = next(i for i in desc["inputs"] if i["name"] == "images")
        assert images["autogrow"] is True
        assert "images.image0" in images["wire_as"]
        # Non-autogrow inputs don't carry the keys.
        ks = graph.morphism_to_dict(graph.node("KSampler"))
        assert all("autogrow" not in i for i in ks["inputs"])


# ===========================================================================
# TestValidateServerParity — BE-3357: presence, no-outputs, range = errors
# ===========================================================================


class TestValidateServerParity:
    """Validate mirrors the three server-side rejections that `validate` used to
    pass silently (BE-3349 / BE-3357), against the captured sd15 catalog:
    required-input presence, the no-outputs check, and range violations."""

    def _sd15_full(self) -> dict:
        """A complete, server-valid sd15 txt2img graph (SaveImage output, every
        required input present)."""
        return {
            "4": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "v1-5-pruned-emaonly-fp16.safetensors"},
            },
            "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "a cat"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "blurry"}},
            "5": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512, "batch_size": 1}},
            "3": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["4", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                    "seed": 42,
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "simple",
                    "denoise": 1.0,
                },
            },
            "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
            "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "ComfyUI"}},
        }

    def test_full_workflow_is_valid(self, graph_sd15: Graph):
        """Regression guard: a KSampler with all 10 required inputs present, in a
        graph with an output node, validates clean."""
        result = graph_sd15.validate_workflow(self._sd15_full())
        assert result["valid"] is True, result["errors"]
        assert result["errors"] == []

    def test_missing_widget_inputs_each_error(self, graph_sd15: Graph):
        """KSampler missing `seed`/`steps` → one required_input_missing per
        missing input, `field` set to the input name."""
        wf = self._sd15_full()
        del wf["3"]["inputs"]["seed"]
        del wf["3"]["inputs"]["steps"]
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        missing = [e for e in result["errors"] if e["code"] == "required_input_missing"]
        assert {e["field"] for e in missing} == {"seed", "steps"}
        assert len(missing) == 2

    def test_missing_required_link_errors(self, graph_sd15: Graph):
        """A missing required *link* input (`model`) is also required_input_missing,
        and its hint tells you to wire a link."""
        wf = self._sd15_full()
        del wf["3"]["inputs"]["model"]
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        err = next(e for e in result["errors"] if e["code"] == "required_input_missing" and e["field"] == "model")
        assert "wire" in err["hint"] and "MODEL" in err["hint"]

    def test_be3349_repro_only_links_wired(self, graph_sd15: Graph):
        """The BE-3349 acceptance case: a KSampler with only its four link inputs
        wired is missing all six widget inputs → six required_input_missing errors."""
        wf = self._sd15_full()
        wf["3"]["inputs"] = {
            "model": ["4", 0],
            "positive": ["6", 0],
            "negative": ["7", 0],
            "latent_image": ["5", 0],
        }
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        missing = [e for e in result["errors"] if e["code"] == "required_input_missing" and e["node_id"] == "3"]
        assert {e["field"] for e in missing} == {"seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"}
        assert len(missing) == 6

    def test_optional_inputs_absent_no_error(self):
        """A required input that is absent errors, but an *optional* input that is
        absent does not."""
        object_info = {
            "OptNode": {
                "input": {
                    "required": {"needed": ["STRING", {}]},
                    "optional": {"maybe": ["STRING", {}]},
                },
                "output": [],
                "output_name": [],
                "output_node": True,
                "python_module": "nodes",
            },
        }
        g = Graph.from_object_info(object_info)
        # Optional absent, required present → clean.
        clean = g.validate_workflow({"1": {"class_type": "OptNode", "inputs": {"needed": "x"}}})
        assert clean["valid"] is True, clean["errors"]
        # Required absent → error; the optional one is never flagged.
        missing = g.validate_workflow({"1": {"class_type": "OptNode", "inputs": {}}})
        codes = {(e["field"], e["code"]) for e in missing["errors"]}
        assert ("needed", "required_input_missing") in codes
        assert not any(e["field"] == "maybe" for e in missing["errors"])

    def test_no_output_node_errors(self, graph_sd15: Graph):
        """A workflow of recognized nodes with no output node is rejected
        (prompt_no_outputs); adding SaveImage clears it."""
        wf = self._sd15_full()
        del wf["9"]  # remove the only output node (SaveImage)
        result = graph_sd15.validate_workflow(wf)
        no_out = [e for e in result["errors"] if e["code"] == "prompt_no_outputs"]
        assert len(no_out) == 1

        # With SaveImage present, no such error.
        result2 = graph_sd15.validate_workflow(self._sd15_full())
        assert [e for e in result2["errors"] if e["code"] == "prompt_no_outputs"] == []

    def test_no_output_error_emitted_once(self, graph_sd15: Graph):
        """The no-outputs error is appended once, not per node."""
        wf = self._sd15_full()
        del wf["9"]
        result = graph_sd15.validate_workflow(wf)
        assert len([e for e in result["errors"] if e["code"] == "prompt_no_outputs"]) == 1

    def test_all_unknown_nodes_no_false_no_outputs(self, graph_sd15: Graph):
        """An unknown node could itself be the (custom) output node — we can't
        see it — so we don't pile a no-outputs error on top of the
        unknown-class errors the user must resolve first."""
        result = graph_sd15.validate_workflow({"1": {"class_type": "TotallyMadeUp", "inputs": {}}})
        assert [e for e in result["errors"] if e["code"] == "prompt_no_outputs"] == []

    def test_empty_workflow_is_no_outputs(self, graph_sd15: Graph):
        """An empty prompt has zero output nodes, which the server rejects
        (prompt_no_outputs); a node-less prompt must not slip through as valid."""
        result = graph_sd15.validate_workflow({})
        no_out = [e for e in result["errors"] if e["code"] == "prompt_no_outputs"]
        assert len(no_out) == 1
        assert result["valid"] is False
        # workflow-level error still carries the node_id/field schema keys.
        assert no_out[0]["node_id"] is None
        assert no_out[0]["field"] is None

    def test_meta_only_workflow_is_no_outputs(self, graph_sd15: Graph):
        """A prompt that is only a `_meta` block (no nodes) has no outputs."""
        result = graph_sd15.validate_workflow({"_meta": {"schema": "x"}})
        assert len([e for e in result["errors"] if e["code"] == "prompt_no_outputs"]) == 1

    def test_unknown_output_node_no_double_no_outputs(self, graph_sd15: Graph):
        """A recognized non-output node plus an unknown node (which could be the
        real output) must not stack prompt_no_outputs on the unknown-class
        error — the fix is installing the custom node, not adding an output."""
        wf = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "v1-5.safetensors"}},
            "2": {"class_type": "MyCustomSaver", "inputs": {"images": ["1", 0]}},
        }
        result = graph_sd15.validate_workflow(wf)
        assert any(e["code"] == "unknown_class_type" for e in result["errors"])
        assert [e for e in result["errors"] if e["code"] == "prompt_no_outputs"] == []

    def test_width_below_min_is_error(self, graph_sd15: Graph):
        """EmptyLatentImage width below the catalog min is a hard error (was a
        warning before BE-3357)."""
        wf = self._sd15_full()
        wf["5"]["inputs"]["width"] = 1  # sd15 min is 16
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "below_min" and e["field"] == "width"]
        assert len(errs) == 1
        assert "16" in errs[0]["hint"]

    def test_meta_key_still_exempt(self, graph_sd15: Graph):
        """`_meta` provenance is still ignored — not counted as a node, never a
        required/no-outputs trigger."""
        wf = {"_meta": {"schema": "compose/1"}, **self._sd15_full()}
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        assert [e for e in result["errors"] if e.get("node_id") == "_meta"] == []

    # -- Output-reachability pruning (BE-3406): match the server, which only
    # validates output nodes and their transitive input ancestors. --

    def test_disconnected_node_missing_required_is_pruned(self, graph_sd15: Graph):
        """Acceptance (a): a disconnected KSampler missing all its required
        inputs, alongside a valid connected output chain, does not fail
        validation — the server prunes it (never reachable from an output), so
        we must not hard-reject the whole prompt on it."""
        wf = self._sd15_full()
        # A stray KSampler wired to nothing and referenced by nothing: not
        # reachable from SaveImage, so the server never validates it.
        wf["99"] = {"class_type": "KSampler", "inputs": {"seed": 1}}
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        assert [e for e in result["errors"] if e["node_id"] == "99"] == []

    def test_reachable_node_missing_required_still_errors(self, graph_sd15: Graph):
        """Acceptance (b): a node ON the output chain that is missing a required
        input still hard-errors — reachability doesn't weaken real validation."""
        wf = self._sd15_full()
        del wf["3"]["inputs"]["seed"]  # KSampler feeds VAEDecode → SaveImage
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        missing = [e for e in result["errors"] if e["code"] == "required_input_missing" and e["node_id"] == "3"]
        assert {e["field"] for e in missing} == {"seed"}

    def test_transitive_ancestor_missing_required_still_errors(self, graph_sd15: Graph):
        """A *transitive* ancestor (CLIPTextEncode, two hops upstream of the
        SaveImage output) is reachable and its missing required input errors —
        proving the backward walk follows link edges, not just direct parents."""
        wf = self._sd15_full()
        del wf["6"]["inputs"]["text"]  # "6" → KSampler.positive → VAEDecode → SaveImage
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        missing = [e for e in result["errors"] if e["code"] == "required_input_missing" and e["node_id"] == "6"]
        assert {e["field"] for e in missing} == {"text"}

    def test_output_node_missing_required_still_errors(self, graph_sd15: Graph):
        """The output node itself seeds the reachable set, so a required input
        missing on SaveImage still errors."""
        wf = self._sd15_full()
        del wf["9"]["inputs"]["filename_prefix"]
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        assert any(
            e["code"] == "required_input_missing" and e["node_id"] == "9" and e["field"] == "filename_prefix"
            for e in result["errors"]
        )

    def test_disconnected_out_of_range_demoted_to_warning(self, graph_sd15: Graph):
        """A below_min value on a disconnected node is a warning, not a hard
        error — the server never range-checks a pruned node. The connected chain
        stays valid."""
        wf = self._sd15_full()
        # A stray EmptyLatentImage (all required inputs present) with an
        # out-of-range width, wired to nothing.
        wf["99"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1, "height": 512, "batch_size": 1}}
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is True, result["errors"]
        assert [e for e in result["errors"] if e["node_id"] == "99"] == []
        warned = [w for w in result["warnings"] if w.get("code") == "below_min" and w.get("node_id") == "99"]
        assert len(warned) == 1

    def test_reachable_out_of_range_still_errors(self, graph_sd15: Graph):
        """The connected EmptyLatentImage feeding the output chain still
        hard-errors on an out-of-range width (reachability preserves the #551
        promotion where it matters)."""
        wf = self._sd15_full()
        wf["5"]["inputs"]["width"] = 1  # "5" → KSampler.latent_image → … → SaveImage
        result = graph_sd15.validate_workflow(wf)
        assert result["valid"] is False
        assert [e for e in result["errors"] if e["code"] == "below_min" and e["node_id"] == "5"]

    def test_demoted_range_warning_field_is_qualified(self, graph_sd15: Graph):
        """A range violation demoted to a warning on a pruned node uses the same
        fully-qualified `field` (`node.class.input`) as every other warning, so
        consumers (e.g. preflight renders w["field"]) see one schema."""
        wf = self._sd15_full()
        wf["99"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 1, "height": 512, "batch_size": 1}}
        result = graph_sd15.validate_workflow(wf)
        warned = [w for w in result["warnings"] if w.get("code") == "below_min" and w.get("node_id") == "99"]
        assert len(warned) == 1
        assert warned[0]["field"] == "99.EmptyLatentImage.width"


class TestValidateMalformedInputs:
    """Malformed workflow JSON must yield structured output, never an unhandled
    traceback (BE-3406 hardening) — the validator's whole contract is to catch
    bad prompts, so it may not crash on the shapes it's meant to reject."""

    def test_non_dict_inputs_does_not_crash(self, graph: Graph):
        """A truthy non-dict `inputs` (string/list from malformed JSON) slips
        past `or {}` and would crash `.items()`/`.values()`; validation must
        instead return a result. Node is wired to a SaveImage so it's reachable
        (exercises both the per-input loop and the reachability walk)."""
        wf = {
            "1": {"class_type": "EmptyLatentImage", "inputs": "not-a-dict"},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "out"}},
        }
        result = graph.validate_workflow(wf)  # must not raise
        assert isinstance(result["errors"], list)
        assert isinstance(result["warnings"], list)

    def test_unhashable_class_type_does_not_crash(self, graph: Graph):
        """An unhashable class_type (list/dict) would raise TypeError in the
        `self._nodes.get(class_type)` lookup and the reachability walk's
        `graph.node(...)`; both are screened so validation returns a result."""
        wf = {
            "1": {"class_type": ["EmptyLatentImage"], "inputs": {"width": 512}},
            "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0], "filename_prefix": "out"}},
        }
        result = graph.validate_workflow(wf)  # must not raise
        assert isinstance(result["errors"], list)


class TestValidateEmptyCombo:
    """A COMBO whose option list is declared but EMPTY means the server has zero
    files installed for that field — it rejects every value against it — so it
    must be reported, not skipped (BE-6585).

    Before this, the membership check was gated on ``self.enum_values`` being
    truthy, so detection got *worse* the emptier the install: ``VAELoader``
    ships one built-in option and its missing model was caught, while
    ``UNETLoader``/``CLIPLoader`` (no built-ins, and the two largest downloads)
    were silent on a fresh install — the exact user this check exists to serve.
    """

    def _object_info(self, **extra) -> dict[str, Any]:
        oi = {
            "UNETLoader": {
                # Bare install: `folder_paths.get_filename_list("diffusion_models")`
                # is empty and the node ships no built-in option.
                "input": {"required": {"unet_name": [[]], "weight_dtype": [["default", "fp8_e4m3fn"]]}},
                "input_order": {"required": ["unet_name", "weight_dtype"]},
                "output": ["MODEL"],
                "output_name": ["MODEL"],
                "python_module": "nodes",
            },
            "VAELoader": {
                # One built-in option (`pixel_space`) — the loader that was
                # already caught, kept here as the contrast case.
                "input": {"required": {"vae_name": [["pixel_space"]]}},
                "input_order": {"required": ["vae_name"]},
                "output": ["VAE"],
                "output_name": ["VAE"],
                "python_module": "nodes",
            },
            "SaveImage": {
                "input": {"required": {"images": "IMAGE"}},
                "output": [],
                "output_name": [],
                "output_node": True,
                "python_module": "nodes",
            },
        }
        oi.update(extra)
        return oi

    def _graph(self, **extra) -> Graph:
        return Graph.from_object_info(self._object_info(**extra))

    def test_empty_combo_flags_the_missing_model(self):
        """The regression: a value against a zero-option loader is an error, not
        silence."""
        g = self._graph()
        result = g.validate_workflow(
            {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"},
                },
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "no_options_available"]
        assert len(errs) == 1
        assert errs[0]["field"] == "unet_name"
        assert errs[0]["node_id"] == "1"
        assert "flux1-dev.safetensors" in errs[0]["message"]
        assert errs[0]["valid_options"] == []
        assert "UNETLoader" in errs[0]["hint"]

    def test_populated_and_empty_loaders_are_both_reported(self):
        """The ticket's count bug: with one loader populated and one empty, only
        the populated one used to be reported. Both are now."""
        g = self._graph()
        result = g.validate_workflow(
            {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"},
                },
                "2": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
                "3": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        codes = {(e["field"], e["code"]) for e in result["errors"]}
        assert ("unet_name", "no_options_available") in codes
        assert ("vae_name", "unknown_enum_value") in codes

    def test_same_loader_with_one_option_installed_flags_membership(self):
        """The ticket's counter-experiment, at the unit level: drop one file into
        the folder and the SAME missing model is caught by the membership check.
        Proves the mechanism was the empty list, not the loader."""
        oi = self._object_info()
        oi["UNETLoader"]["input"]["required"]["unet_name"] = [["some-other-model.safetensors"]]
        g = Graph.from_object_info(oi)
        result = g.validate_workflow(
            {
                "1": {
                    "class_type": "UNETLoader",
                    "inputs": {"unet_name": "flux1-dev.safetensors", "weight_dtype": "default"},
                },
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        errs = [e for e in result["errors"] if e["field"] == "unet_name"]
        assert len(errs) == 1
        assert errs[0]["code"] == "unknown_enum_value"

    def test_installed_value_on_populated_loader_still_passes(self):
        """No false positive on the field that IS populated."""
        g = self._graph()
        result = g.validate_workflow(
            {
                "1": {"class_type": "VAELoader", "inputs": {"vae_name": "pixel_space"}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        assert result["valid"] is True, result["errors"]

    def test_dict_form_empty_options_is_flagged(self):
        """The partner-node dialect (``["COMBO", {"options": [...]}]``) declares
        its choices in the options dict — an empty list there is the same
        statement as an empty list-form combo."""
        g = self._graph(
            PartnerNode={
                "input": {"required": {"model_name": ["COMBO", {"options": []}]}},
                "output": ["MODEL"],
                "output_name": ["MODEL"],
                "python_module": "nodes",
            }
        )
        result = g.validate_workflow(
            {
                "1": {"class_type": "PartnerNode", "inputs": {"model_name": "seedream-5"}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        errs = [e for e in result["errors"] if e["code"] == "no_options_available"]
        assert [e["field"] for e in errs] == ["model_name"]

    def test_remote_combo_stays_unconstrained(self):
        """A combo whose options the frontend fetches at runtime ships NO
        ``options`` key (``prune_dict`` drops it). That is "unknown", not "zero
        installed" — validating against it would false-positive on every
        remote-backed field, so it stays silent."""
        g = self._graph(
            RemoteNode={
                "input": {"required": {"model": ["COMBO", {"remote": {"route": "/api/models"}}]}},
                "output": ["MODEL"],
                "output_name": ["MODEL"],
                "python_module": "nodes",
            }
        )
        port = next(p for p in g.node("RemoteNode").inputs if p.name == "model")
        assert port.enum_declared is False
        result = g.validate_workflow(
            {
                "1": {"class_type": "RemoteNode", "inputs": {"model": "whatever-the-route-returns"}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        assert result["valid"] is True, result["errors"]

    def test_empty_combo_is_still_a_widget_not_a_link(self):
        """``enum_declared`` is deliberately separate from ``is_enum`` so the
        empty case cannot move a port between widget and link wiring."""
        g = self._graph(
            PartnerNode={
                "input": {"required": {"model_name": ["COMBO", {"options": []}]}},
                "output": ["MODEL"],
                "output_name": ["MODEL"],
                "python_module": "nodes",
            }
        )
        list_form = next(p for p in g.node("UNETLoader").inputs if p.name == "unet_name")
        dict_form = next(p for p in g.node("PartnerNode").inputs if p.name == "model_name")
        assert (list_form.is_link, list_form.enum_declared, list_form.enum_values) == (False, True, [])
        assert (dict_form.is_link, dict_form.enum_declared, dict_form.enum_values) == (False, True, [])

    def test_absent_input_is_not_reported_as_unavailable(self):
        """The check only fires on a value the workflow actually supplies — an
        input that is missing entirely stays the existing required_input_missing
        error, so the two never double-report the same field."""
        g = self._graph()
        result = g.validate_workflow(
            {
                "1": {"class_type": "UNETLoader", "inputs": {"weight_dtype": "default"}},
                "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
            }
        )
        by_field = {(e["field"], e["code"]) for e in result["errors"]}
        assert ("unet_name", "required_input_missing") in by_field
        assert ("unet_name", "no_options_available") not in by_field


class TestValidateDynamicCombo:
    """Validate expands a ``COMFY_DYNAMICCOMBO_V3`` selector's chosen option and
    checks the dotted sub-inputs the server will actually require (BE-3777).

    object_info declares only the selector (``model``); the frontend and
    ``convert_ui_to_api`` lower the selected option's own INPUT_TYPES into
    ``model.width`` / ``model.size_preset`` / … keys. Before this, none of those
    keys were reachable from the schema, so every one of the cases below came
    back ``valid: true`` and was then rejected at ``/prompt`` with
    ``required_input_missing`` / ``shape_mismatch``.
    """

    @pytest.fixture
    def seedream_graph(self) -> Graph:
        """The captured ByteDance Seedream catalog plus a minimal output node, so
        a converted single-node workflow isn't drowned in prompt_no_outputs."""
        import json
        from pathlib import Path

        fixture = Path(__file__).parent.parent / "fixtures" / "object_info_bytedance_seedream_v2.json"
        object_info = json.loads(fixture.read_text())
        object_info["SaveImageAdvanced"] = {
            "input": {"required": {"images": ["IMAGE", {}]}},
            "output": [],
            "output_name": [],
            "output_node": True,
            "python_module": "nodes",
        }
        return Graph.from_object_info(object_info)

    @pytest.fixture
    def seedream_api(self, seedream_graph: Graph) -> dict:
        """The ticket's repro payload: the pristine Seedream 5.0 Pro UI export,
        lowered to API format by the same converter ``comfy run`` / ``comfy
        validate`` use — i.e. the exact bytes that reach ``/prompt``."""
        import json
        from pathlib import Path

        from comfy_cli.workflow_to_api import convert_ui_to_api

        ui = json.loads((Path(__file__).parent.parent / "fixtures" / "seedream_5_0_pro_t2i_ui.json").read_text())
        return convert_ui_to_api(ui, seedream_graph.object_info)

    def test_converted_seedream_workflow_is_valid(self, seedream_graph: Graph, seedream_api: dict):
        """Guard against over-strictness: the pristine template — whose
        ``model.images`` autogrow sub-input is declared *required* with
        ``min: 0`` and legitimately emits no key — must still validate clean."""
        result = seedream_graph.validate_workflow(seedream_api)
        assert result["valid"] is True, result["errors"]

    def test_missing_sub_input_errors(self, seedream_graph: Graph, seedream_api: dict):
        wf = copy.deepcopy(seedream_api)
        del wf["1"]["inputs"]["model.width"]
        result = seedream_graph.validate_workflow(wf)
        assert result["valid"] is False
        err = next(e for e in result["errors"] if e["field"] == "model.width")
        assert err["code"] == "required_input_missing"
        assert err["node_id"] == "1"

    def test_sub_input_shape_mismatch_errors(self, seedream_graph: Graph, seedream_api: dict):
        wf = copy.deepcopy(seedream_api)
        wf["1"]["inputs"]["model.width"] = "wide"  # INT declared, non-numeric string given
        result = seedream_graph.validate_workflow(wf)
        assert result["valid"] is False
        err = next(e for e in result["errors"] if e["field"] == "model.width")
        assert err["code"] == "shape_mismatch"

    def test_sub_input_range_and_enum_checked(self, seedream_graph: Graph, seedream_api: dict):
        """Sub-inputs go through the same Port machinery as top-level inputs, so
        they inherit the range and enum-membership hard errors."""
        wf = copy.deepcopy(seedream_api)
        wf["1"]["inputs"]["model.width"] = 8  # catalog min is 1024
        wf["1"]["inputs"]["model.size_preset"] = "(9K) nope"
        result = seedream_graph.validate_workflow(wf)
        codes = {(e["field"], e["code"]) for e in result["errors"]}
        assert ("model.width", "below_min") in codes
        assert ("model.size_preset", "unknown_enum_value") in codes

    def test_selecting_another_option_switches_the_required_set(self, seedream_graph: Graph, seedream_api: dict):
        """Flipping the selector to ``seedream 5.0 lite`` — which declares
        ``max_images``/``fail_on_partial`` the pro option does not — makes those
        sub-inputs required, exactly as the server would."""
        wf = copy.deepcopy(seedream_api)
        wf["1"]["inputs"]["model"] = "seedream 5.0 lite"
        result = seedream_graph.validate_workflow(wf)
        assert result["valid"] is False
        missing = {e["field"] for e in result["errors"] if e["code"] == "required_input_missing"}
        assert missing == {"model.max_images", "model.fail_on_partial"}

    def test_unknown_selector_errors_with_options(self, seedream_graph: Graph, seedream_api: dict):
        """A selector naming no option is where the converter silently misaligns
        the following widget values, so it must not pass as valid."""
        wf = copy.deepcopy(seedream_api)
        wf["1"]["inputs"]["model"] = "seedream 9.9 ultra"
        result = seedream_graph.validate_workflow(wf)
        assert result["valid"] is False
        err = next(e for e in result["errors"] if e["field"] == "model")
        assert err["code"] == "unknown_enum_value"
        assert "seedream 5.0 pro" in err["valid_options"]

    def test_unresolvable_selector_messages_say_execution_not_prompt(self, seedream_graph: Graph, seedream_api: dict):
        """An unresolvable selector is a hard error, but the message must NOT
        claim a ``/prompt`` rejection: with nothing to expand, the server never
        adds the selector to its finalized input set, so ``/prompt`` accepts the
        prompt and the node dies at execution — after a paid node is entered.
        Only the *sub-input* errors are genuine ``/prompt`` rejections."""
        for bad in ({"model": "seedream 9.9 ultra"}, {}):
            wf = copy.deepcopy(seedream_api)
            wf["1"]["inputs"].pop("model")
            wf["1"]["inputs"].update(bad)
            err = next(e for e in seedream_graph.validate_workflow(wf)["errors"] if e["field"] == "model")
            assert "execution" in err["message"]
            assert "will reject" not in err["message"]

    def test_missing_selector_errors_once(self, seedream_graph: Graph, seedream_api: dict):
        """The absent selector is reported by the dynamic path only —
        ``_check_required_present`` exempts it, so exactly one error, and it
        carries the valid options."""
        wf = copy.deepcopy(seedream_api)
        del wf["1"]["inputs"]["model"]
        result = seedream_graph.validate_workflow(wf)
        errs = [e for e in result["errors"] if e["field"] == "model"]
        assert len(errs) == 1
        assert errs[0]["code"] == "required_input_missing"
        assert "seedream 5.0 pro" in errs[0]["valid_options"]
        # No sub-input errors pile on top — the option set is unknown.
        assert [e for e in result["errors"] if e["field"].startswith("model.")] == []

    # -- synthetic catalogs for the shapes the captured fixture doesn't cover --

    @staticmethod
    def _dyn_object_info(options: list[dict]) -> dict:
        return {
            "DynNode": {
                "input": {"required": {"mode": ["COMFY_DYNAMICCOMBO_V3", {"options": options}]}},
                "output": [],
                "output_name": [],
                "output_node": True,
                "python_module": "nodes",
            },
        }

    def test_optional_sub_input_absent_is_clean(self):
        """A sub-input in the option's ``optional`` section may be absent."""
        g = Graph.from_object_info(
            self._dyn_object_info(
                [{"key": "go", "inputs": {"required": {"a": ["INT", {}]}, "optional": {"b": ["INT", {}]}}}]
            )
        )
        result = g.validate_workflow({"1": {"class_type": "DynNode", "inputs": {"mode": "go", "mode.a": 1}}})
        assert result["valid"] is True, result["errors"]

    def test_optional_sub_input_present_is_still_shape_checked(self):
        g = Graph.from_object_info(
            self._dyn_object_info(
                [{"key": "go", "inputs": {"required": {"a": ["INT", {}]}, "optional": {"b": ["INT", {}]}}}]
            )
        )
        result = g.validate_workflow(
            {"1": {"class_type": "DynNode", "inputs": {"mode": "go", "mode.a": 1, "mode.b": "nope"}}}
        )
        assert result["valid"] is False
        assert next(e for e in result["errors"] if e["field"] == "mode.b")["code"] == "shape_mismatch"

    def test_nested_dynamic_combo_expands(self):
        """An option's sub-input can itself be a dynamic combo — the inner
        option's own required sub-inputs are checked at ``mode.inner.leaf``."""
        inner = ["COMFY_DYNAMICCOMBO_V3", {"options": [{"key": "deep", "inputs": {"required": {"leaf": ["INT", {}]}}}]}]
        g = Graph.from_object_info(self._dyn_object_info([{"key": "go", "inputs": {"required": {"inner": inner}}}]))
        wf = {"1": {"class_type": "DynNode", "inputs": {"mode": "go", "mode.inner": "deep"}}}
        result = g.validate_workflow(wf)
        assert result["valid"] is False
        assert {e["field"] for e in result["errors"]} == {"mode.inner.leaf"}

        wf["1"]["inputs"]["mode.inner.leaf"] = 3
        assert g.validate_workflow(wf)["valid"] is True

    def test_wired_selector_skips_expansion(self):
        """A selector wired as a link resolves at execution time, so there is no
        static option to expand — no phantom missing-sub-input errors."""
        object_info = self._dyn_object_info([{"key": "go", "inputs": {"required": {"a": ["INT", {}]}}}])
        object_info["IntSource"] = {
            "input": {"required": {}},
            "output": ["INT"],
            "output_name": ["INT"],
            "output_node": False,
            "python_module": "nodes",
        }
        g = Graph.from_object_info(object_info)
        result = g.validate_workflow(
            {
                "1": {"class_type": "DynNode", "inputs": {"mode": ["2", 0]}},
                "2": {"class_type": "IntSource", "inputs": {}},
            }
        )
        assert result["valid"] is True, result["errors"]

    def test_wired_sub_input_is_not_presence_or_shape_flagged(self):
        """A sub-input satisfied by a link counts as present, and its value is a
        ``[node_id, index]`` pair — not a shape violation."""
        object_info = self._dyn_object_info([{"key": "go", "inputs": {"required": {"a": ["INT", {}]}}}])
        object_info["IntSource"] = {
            "input": {"required": {}},
            "output": ["INT"],
            "output_name": ["INT"],
            "output_node": False,
            "python_module": "nodes",
        }
        g = Graph.from_object_info(object_info)
        result = g.validate_workflow(
            {
                "1": {"class_type": "DynNode", "inputs": {"mode": "go", "mode.a": ["2", 0]}},
                "2": {"class_type": "IntSource", "inputs": {}},
            }
        )
        assert result["valid"] is True, result["errors"]

    def test_malformed_option_block_does_not_crash(self):
        """object_info is server-supplied — a junk option block degrades to
        'no sub-inputs to check', never an exception."""
        g = Graph.from_object_info(self._dyn_object_info([{"key": "go", "inputs": "not-a-dict"}, "junk"]))
        result = g.validate_workflow({"1": {"class_type": "DynNode", "inputs": {"mode": "go"}}})
        assert result["valid"] is True, result["errors"]

    def test_optional_selector_absent_is_clean(self):
        """An *optional* dynamic combo that is absent is not a missing input."""
        object_info = {
            "OptDyn": {
                "input": {
                    "optional": {
                        "mode": [
                            "COMFY_DYNAMICCOMBO_V3",
                            {"options": [{"key": "go", "inputs": {"required": {"a": ["INT", {}]}}}]},
                        ]
                    }
                },
                "output": [],
                "output_name": [],
                "output_node": True,
                "python_module": "nodes",
            },
        }
        g = Graph.from_object_info(object_info)
        result = g.validate_workflow({"1": {"class_type": "OptDyn", "inputs": {}}})
        assert result["valid"] is True, result["errors"]


# ===========================================================================
# TestDirectModeSlots
# ===========================================================================


class TestDirectModeSlots:
    """Tests _extract_frontend_slots and _apply_one_slot in direct mode."""

    def test_extract_finds_all_widget_inputs(self, graph: Graph):
        wf = _direct_workflow()
        slots = _extract_frontend_slots(wf, graph)
        # KSampler: seed, steps, cfg, sampler_name, scheduler, denoise (6)
        # CLIPTextEncode: text (1)
        # EmptyLatentImage: width, height, batch_size (3)
        # Total: 10
        assert len(slots) == 10
        names = {s["name"] for s in slots}
        # No link inputs should appear
        assert "model" not in names
        assert "positive" not in names
        assert "negative" not in names
        assert "latent_image" not in names
        assert "clip" not in names
        assert "samples" not in names
        assert "vae" not in names

    def test_extract_addresses_are_node_id_dot_name(self, graph: Graph):
        wf = _direct_workflow()
        slots = _extract_frontend_slots(wf, graph)
        for slot in slots:
            assert slot["address"] == f"{slot['instance_id']}.{slot['name']}"

    def test_extract_current_values(self, graph: Graph):
        wf = _direct_workflow()
        slots = _extract_frontend_slots(wf, graph)
        by_addr = {s["address"]: s for s in slots}
        assert by_addr["6.text"]["current_value"] == "a cat in space"
        assert by_addr["3.seed"]["current_value"] == 42

    def test_apply_slot_updates_value(self, graph: Graph):
        wf = _direct_workflow()
        _apply_one_slot(wf, "3.seed", 999, graph)
        assert wf["nodes"][0]["widgets_values"][0] == 999

    def test_apply_slot_text(self, graph: Graph):
        wf = _direct_workflow()
        _apply_one_slot(wf, "6.text", "a dog", graph)
        assert wf["nodes"][1]["widgets_values"][0] == "a dog"

    def test_apply_slot_shape_rejection(self, graph: Graph):
        wf = _direct_workflow()
        with pytest.raises(ValueError):
            _apply_one_slot(wf, "3.seed", "not_an_int", graph)

    def test_apply_slot_unknown_node(self, graph: Graph):
        wf = _direct_workflow()
        with pytest.raises(ValueError, match="not found"):
            _apply_one_slot(wf, "99.seed", 1, graph)

    def test_apply_slot_unknown_widget(self, graph: Graph):
        wf = _direct_workflow()
        with pytest.raises(ValueError, match="not found on KSampler"):
            _apply_one_slot(wf, "3.nonexistent", 1, graph)

    def test_apply_returns_catalog_warnings(self, graph: Graph):
        wf = _direct_workflow()
        warnings = _apply_one_slot(wf, "3.steps", 99999, graph)
        codes = [w["code"] for w in warnings]
        assert "above_max" in codes


# ===========================================================================
# TestTemplateModeSlots
# ===========================================================================


class TestTemplateModeSlots:
    """Tests _extract_frontend_slots and _apply_one_slot in template/subgraph mode."""

    def test_extract_template_slots(self, graph: Graph):
        wf = _template_workflow()
        slots = _extract_frontend_slots(wf, graph)
        assert len(slots) == 2
        by_addr = {s["address"]: s for s in slots}
        assert "1.text" in by_addr
        assert "1.seed" in by_addr
        assert by_addr["1.text"]["current_value"] == "hello world"
        assert by_addr["1.seed"]["current_value"] == 42

    def test_template_mode_takes_priority(self, graph: Graph):
        wf = _template_workflow()
        slots = _extract_frontend_slots(wf, graph)
        # Only the 2 template-declared inputs, not direct widget slots
        assert len(slots) == 2
        assert all(s["node_type"] == "MyTemplate" for s in slots)

    def test_apply_template_slot_text(self, graph: Graph):
        wf = _template_workflow()
        _apply_one_slot(wf, "1.text", "new prompt", graph)
        interior_nodes = wf["definitions"]["subgraphs"][0]["nodes"]
        clip_node = next(n for n in interior_nodes if n["id"] == 10)
        assert clip_node["widgets_values"][0] == "new prompt"

    def test_apply_template_slot_seed(self, graph: Graph):
        wf = _template_workflow()
        _apply_one_slot(wf, "1.seed", 999, graph)
        interior_nodes = wf["definitions"]["subgraphs"][0]["nodes"]
        ks_node = next(n for n in interior_nodes if n["id"] == 11)
        assert ks_node["widgets_values"][0] == 999


# ===========================================================================
# TestSubgraphIsolation
# ===========================================================================


class TestSubgraphIsolation:
    """Two instances of one subgraph definition must not alias on interior write."""

    def test_nested_slot_write_isolates_instances(self, graph: Graph):
        """Writing 10/9.text must not affect instance 12's interior node."""
        from comfy_cli.cql.engine import _apply_one_slot

        wf = {
            "nodes": [
                {"id": 10, "type": "uuid-def-1"},
                {"id": 12, "type": "uuid-def-1"},
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "uuid-def-1",
                        "name": "Sub",
                        "nodes": [
                            {"id": 9, "type": "CLIPTextEncode", "widgets_values": ["orig"]},
                        ],
                    },
                ]
            },
        }
        _apply_one_slot(wf, "10/9.text", "VALUE-FOR-10", graph)

        # Rebuild the definitions index from the (potentially mutated) workflow
        defs = {d["id"]: d for d in wf["definitions"]["subgraphs"]}

        # Instance 12 must still read 'orig'
        inst12 = next(n for n in wf["nodes"] if n["id"] == 12)
        inst12_def = defs[inst12["type"]]
        assert inst12_def["nodes"][0]["widgets_values"][0] == "orig"

        # Instance 10 got the new value
        inst10 = next(n for n in wf["nodes"] if n["id"] == 10)
        inst10_def = defs[inst10["type"]]
        assert inst10_def["nodes"][0]["widgets_values"][0] == "VALUE-FOR-10"

    def test_second_write_to_same_instance_no_extra_fork(self, graph: Graph):
        """A second write to the same instance must not create yet another fork."""
        from comfy_cli.cql.engine import _apply_one_slot

        wf = {
            "nodes": [
                {"id": 10, "type": "uuid-def-1"},
                {"id": 12, "type": "uuid-def-1"},
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "uuid-def-1",
                        "name": "Sub",
                        "nodes": [
                            {"id": 9, "type": "CLIPTextEncode", "widgets_values": ["orig"]},
                        ],
                    },
                ]
            },
        }
        _apply_one_slot(wf, "10/9.text", "FIRST", graph)
        _apply_one_slot(wf, "10/9.text", "SECOND", graph)

        # Should still be exactly 2 definitions total (one fork + one original)
        assert len(wf["definitions"]["subgraphs"]) == 2

        # Instance 10 has the latest value
        defs = {d["id"]: d for d in wf["definitions"]["subgraphs"]}
        inst10 = next(n for n in wf["nodes"] if n["id"] == 10)
        inst10_def = defs[inst10["type"]]
        assert inst10_def["nodes"][0]["widgets_values"][0] == "SECOND"

        # Instance 12 still has the original value
        inst12 = next(n for n in wf["nodes"] if n["id"] == 12)
        inst12_def = defs[inst12["type"]]
        assert inst12_def["nodes"][0]["widgets_values"][0] == "orig"

    def test_single_instance_no_fork(self, graph: Graph):
        """When only one instance of a def exists, no fork is created."""
        from comfy_cli.cql.engine import _apply_one_slot

        wf = {
            "nodes": [
                {"id": 10, "type": "uuid-def-1"},
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "uuid-def-1",
                        "name": "Sub",
                        "nodes": [
                            {"id": 9, "type": "CLIPTextEncode", "widgets_values": ["orig"]},
                        ],
                    },
                ]
            },
        }
        _apply_one_slot(wf, "10/9.text", "NEW", graph)

        # No extra definition was appended
        assert len(wf["definitions"]["subgraphs"]) == 1
        # The single def got the new value directly
        assert wf["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"][0] == "NEW"


# ===========================================================================
# TestExpandVariations
# ===========================================================================


class TestExpandVariations:
    """Tests graph.expand_variations."""

    def test_produces_n_independent_copies(self, graph: Graph):
        wf = _direct_workflow()
        variations = [
            {"3.seed": 100},
            {"3.seed": 200},
            {"3.seed": 300},
        ]
        results, _ = graph.expand_variations(wf, variations)
        assert len(results) == 3
        # Each has its own seed
        assert results[0]["nodes"][0]["widgets_values"][0] == 100
        assert results[1]["nodes"][0]["widgets_values"][0] == 200
        assert results[2]["nodes"][0]["widgets_values"][0] == 300
        # Mutating one doesn't affect others
        results[0]["nodes"][0]["widgets_values"][0] = -1
        assert results[1]["nodes"][0]["widgets_values"][0] == 200

    def test_original_unchanged(self, graph: Graph):
        wf = _direct_workflow()
        original_seed = wf["nodes"][0]["widgets_values"][0]
        graph.expand_variations(wf, [{"3.seed": 999}])
        assert wf["nodes"][0]["widgets_values"][0] == original_seed

    def test_multi_slot_sweep(self, graph: Graph):
        wf = _direct_workflow()
        variations = [
            {"3.seed": 10, "6.text": "cat"},
            {"3.seed": 20, "6.text": "dog"},
            {"3.seed": 30, "6.text": "fox"},
        ]
        results, _ = graph.expand_variations(wf, variations)
        assert len(results) == 3
        for i, (seed, text) in enumerate([(10, "cat"), (20, "dog"), (30, "fox")]):
            assert results[i]["nodes"][0]["widgets_values"][0] == seed
            assert results[i]["nodes"][1]["widgets_values"][0] == text


# ===========================================================================
# TestBrowse
# ===========================================================================


class TestBrowse:
    """Quick tests for list_types, category_tree, node_count, all_nodes."""

    def test_list_types(self, graph: Graph):
        types = graph.list_types()
        for t in ["CLIP", "CONDITIONING", "IMAGE", "LATENT", "MODEL", "VAE"]:
            assert t in types
        # Should be sorted
        assert types == sorted(types)

    def test_category_tree_has_root(self, graph: Graph):
        tree = graph.category_tree()
        assert "Root" in tree

    def test_node_count(self, graph: Graph):
        assert graph.node_count() == 8

    def test_all_nodes_sorted(self, graph: Graph):
        nodes = graph.all_nodes()
        ids = [m.id for m in nodes]
        assert ids == sorted(ids)


# ===========================================================================
# TestClassifyApiWorkflow
# ===========================================================================


class TestClassifyApiWorkflow:
    def test_meta_key_first_still_ok(self):
        wf = {
            "_meta": {"title": "test"},
            "1": {"class_type": "KSampler", "inputs": {}},
        }
        kind, _ = _classify_api_workflow(wf)
        assert kind == "ok"

    def test_no_nodes_invalid(self):
        wf = {"_meta": {"title": "test"}}
        kind, _ = _classify_api_workflow(wf)
        assert kind == "invalid"

    def test_empty_dict_is_empty(self):
        kind, _ = _classify_api_workflow({})
        assert kind == "empty"

    def test_meta_with_class_type_is_ok(self):
        """A metadata key that happens to have class_type passes shape check.
        Downstream validate_workflow catches unknown class_types."""
        wf = {"_meta": {"class_type": "NotARealNode"}}
        kind, _ = _classify_api_workflow(wf)
        assert kind == "ok"


# ===========================================================================
# TestNullValuedProxy
# ===========================================================================


class TestNullValuedProxy:
    """A proxy that resolves to a legitimately-null widget value must keep the
    curated address and must NOT explode into interior-node slots."""

    def test_null_valued_proxy_stays_curated(self, graph: Graph):
        """CLIPTextEncode.text at index 0 is resolvable; widgets_values=[None]
        means the widget exists but its value is null — the slot must remain
        curated with address '10.text' and current_value None."""
        wf = {
            "nodes": [
                {
                    "id": 10,
                    "type": "uuid-def-2",
                    "properties": {"proxyWidgets": [["9", "text"]]},
                }
            ],
            "definitions": {
                "subgraphs": [
                    {
                        "id": "uuid-def-2",
                        "name": "Sub",
                        "inputs": [{"name": "text", "type": "STRING"}],
                        "nodes": [
                            {
                                "id": 9,
                                "type": "CLIPTextEncode",
                                "widgets_values": [None],
                            }
                        ],
                    }
                ]
            },
        }
        slots = _extract_frontend_slots(wf, graph)
        addrs = [s["address"] for s in slots]
        # Curated address preserved despite null value
        assert "10.text" in addrs
        # Value is explicitly None (not missing)
        by_addr = {s["address"]: s for s in slots}
        assert by_addr["10.text"]["current_value"] is None
        # Did NOT explode into interior slots
        assert not any(a.startswith("10/") for a in addrs)


# ===========================================================================
# TestDottedInputName
# ===========================================================================


def test_slot_address_with_dotted_input_name(graph, monkeypatch):
    """Input names may contain dots; the node path never does, so parse on the
    FIRST dot: 10/9.images.image0 -> node_path '10/9', input 'images.image0'."""
    from comfy_cli.cql import engine

    class _FakeMeta:
        inputs = []  # no declared inputs -> _write_widget skips shape/catalog validation

    real_node = graph.node
    real_order = graph.widget_order_for_node
    monkeypatch.setattr(graph, "node", lambda nt: _FakeMeta() if nt == "DottedWidgetNode" else real_node(nt))
    monkeypatch.setattr(
        graph,
        "widget_order_for_node",
        lambda nt, wv: ["images.image0"] if nt == "DottedWidgetNode" else real_order(nt, wv),
    )

    wf = {
        "nodes": [{"id": 10, "type": "uuid-dot"}],
        "definitions": {
            "subgraphs": [
                {
                    "id": "uuid-dot",
                    "name": "Sub",
                    "nodes": [
                        {"id": 9, "type": "DottedWidgetNode", "widgets_values": [None]},
                    ],
                },
            ]
        },
    }
    # Must NOT raise "interior node images not found"; value lands on node 9's dotted widget.
    engine._apply_one_slot(wf, "10/9.images.image0", "X", graph)
    assert wf["definitions"]["subgraphs"][0]["nodes"][0]["widgets_values"][0] == "X"


# ===========================================================================
# SSRF loopback guard on the local object_info fetch
# ===========================================================================


def test_load_from_target_refuses_non_loopback_local_host():
    from comfy_cli.cql.engine import LoadError, _load_from_target

    with pytest.raises(LoadError, match="non-loopback"):
        _load_from_target(mode="local", host="example.com", port=8188)


# ===========================================================================
# `--input <dump>` is an offline path — annotation lookup must not reach out
# ===========================================================================


def test_input_path_load_does_not_touch_the_network(tmp_path, monkeypatch):
    """``comfy nodes ls --input dump.json`` reads a local file by the caller's
    explicit choice. Resolving annotations is incidental to that and must not be
    the thing that turns an offline command into a network round-trip."""
    from comfy_cli.cql import annotations_source
    from comfy_cli.cql.engine import Graph

    dump = tmp_path / "object_info.json"
    dump.write_text(json.dumps(_object_info()))

    monkeypatch.setattr(
        annotations_source,
        "fetch_pair",
        lambda **kw: pytest.fail("annotation fetch attempted on the --input path"),
    )
    monkeypatch.setenv("COMFY_CLI_NO_REMOTE_REFRESH", "0")  # network would otherwise be allowed

    seen: dict = {}
    real_load = annotations_source.load_annotation_bytes

    def spy(**kwargs):
        seen.update(kwargs)
        return real_load(**kwargs)

    monkeypatch.setattr(annotations_source, "load_annotation_bytes", spy)

    g = Graph.load(input_path=str(dump))
    assert g.node_count() > 0
    assert seen == {"allow_network": False}
