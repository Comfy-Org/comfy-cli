"""Tests for comfy_cli.cql.engine — the pure-Python CQL graph engine.

Layer 1: unit tests for Graph methods and slot-editing helpers.
No I/O, no CLI invocation — just the engine in isolation.
"""

from __future__ import annotations

from typing import Any

import pytest

from comfy_cli.command.run.loader import _classify_api_workflow
from comfy_cli.cql.engine import (
    Graph,
    _apply_one_slot,
    _extract_frontend_slots,
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

    def test_find_paths_same_type_returns_empty(self, graph: Graph):
        assert graph.find_paths("MODEL", "MODEL") == []

    def test_find_paths_unreachable_returns_empty(self, graph: Graph):
        # No node consumes IMAGE and produces MODEL in this fixture
        assert graph.find_paths("IMAGE", "MODEL") == []


# ===========================================================================
# TestValidateWorkflow
# ===========================================================================


class TestValidateWorkflow:
    """Tests graph.validate_workflow(api_workflow)."""

    def _valid_workflow(self) -> dict:
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
        }

    def test_valid_workflow(self, graph: Graph):
        result = graph.validate_workflow(self._valid_workflow())
        assert result["valid"] is True
        assert result["errors"] == []

    def test_non_node_key_warns(self, graph: Graph):
        """An unrecognized non-node key should produce a warning, not an error."""
        wf = {
            "notanode": {"title": "My Workflow"},
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is True
        non_node = [w for w in result["warnings"] if w["code"] == "non_node_key"]
        assert len(non_node) == 1
        assert non_node[0]["node_id"] == "notanode"
        assert non_node[0]["field"] == "notanode"

    def test_meta_provenance_key_is_not_warned(self, graph: Graph):
        """`_meta` is the compose/run provenance block (stripped before submit),
        not a stray key — validating composed output must not nag about it."""
        wf = {
            "_meta": {"schema": "compose/1", "blueprint": "blueprints/x.yaml"},
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is True
        assert [w for w in result["warnings"] if w["node_id"] == "_meta"] == []

    def test_non_dict_node_value_warns(self, graph: Graph):
        """A string value for a key should warn, not crash."""
        wf = {
            "_comment": "this is a comment",
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is True
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
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],  # MODEL from CheckpointLoaderSimple[0]
                    "seed": 42,
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 1.0,
                },
            },
        }
        result = graph.validate_workflow(wf)
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
        wf = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "sd_xl_base.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                # Output index 1 is CLIP, but model input expects MODEL
                "inputs": {"model": ["1", 1]},
            },
        }
        result = graph.validate_workflow(wf)
        # edge_type_mismatch is a warning, not a hard error
        assert result["valid"] is True
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
        assert result["valid"] is True
        assert result["errors"] == []

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
        errs = [e for e in result["errors"] if e["field"] == "duration"]
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
        assert result["valid"] is True

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

    def test_below_min_warning(self, graph: Graph):
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"steps": 0},
            },
        }
        result = graph.validate_workflow(wf)
        # below_min is a warning, not an error
        codes = [w["code"] for w in result["warnings"]]
        assert "below_min" in codes

    def test_above_max_warning(self, graph: Graph):
        wf = {
            "1": {
                "class_type": "KSampler",
                "inputs": {"steps": 99999},
            },
        }
        result = graph.validate_workflow(wf)
        codes = [w["code"] for w in result["warnings"]]
        assert "above_max" in codes


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
        wf = {
            **self._loaders(),
            "20": {
                "class_type": "BatchImagesNode",
                "inputs": {"images.image0": ["10", 0], "images.image1": ["11", 0]},
            },
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
        wf = {
            "20": {"class_type": "BatchImagesNode", "inputs": {}},
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
    real_order = graph.widget_order
    monkeypatch.setattr(graph, "node", lambda nt: _FakeMeta() if nt == "DottedWidgetNode" else real_node(nt))
    monkeypatch.setattr(
        graph,
        "widget_order",
        lambda nt: ["images.image0"] if nt == "DottedWidgetNode" else real_order(nt),
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
