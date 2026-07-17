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


@pytest.fixture
def graph_sd15() -> Graph:
    """Graph built from the real captured sd15 object_info fixture — the same
    catalog the BE-3349 repro / BE-3357 acceptance criterion runs against."""
    import json
    from pathlib import Path

    fixture = Path(__file__).parent.parent / "fixtures" / "sd15_object_info.json"
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
        with value_smaller_than_min) — was a warning before BE-3357."""
        wf = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 0, "height": 512, "batch_size": 1},
            },
        }
        result = graph.validate_workflow(wf)
        assert result["valid"] is False
        errs = [e for e in result["errors"] if e["code"] == "below_min"]
        assert len(errs) == 1
        assert errs[0]["field"] == "width"
        # No longer surfaced as a warning.
        assert "below_min" not in [w["code"] for w in result["warnings"]]

    def test_above_max_error(self, graph: Graph):
        """A value above the catalog max is a hard error (value_bigger_than_max)."""
        wf = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": 999999, "height": 512, "batch_size": 1},
            },
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
