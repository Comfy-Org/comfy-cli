"""Template model files the server lacks, and two validate false positives.

A template can name a model build the server does not carry: the MiniMax H3
templates name ``minimax_h3_video_vae_int8_convrot.safetensors`` while a server
may carry ``minimax_h3_video_vae_fp16.safetensors``. Without help, ``validate``
fails on the template and the caller must find the variant by hand.

* ``templates fetch`` with an offline catalog swaps a missing file for its ONE
  same-model, other-precision sibling and reports the swap; a file with no such
  sibling is reported with its closest options and left alone.
* ``validate`` names that sibling on the ``unknown_enum_value`` finding.
* ``CustomCombo.choice`` (frontend-defined options, never checked by the
  server) and a BOOLEAN saved as the string ``"True"`` (the server runs it
  through ``bool()``) no longer fail validate.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import templates as templates_cmd
from comfy_cli.cql.engine import Graph
from comfy_cli.model_variants import (
    ModelVariantResolutionError,
    precision_key,
    precision_sibling,
    resolve_workflow_models,
)
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

VAES = [
    "ae.safetensors",
    "minimax_h3_video_vae_fp16.safetensors",
    "minimax_h3_audio_vae_fp32.safetensors",
    "qwen3vl_8b_bf16.safetensors",
    "qwen3vl_8b_fp8_scaled.safetensors",
]


def _object_info() -> dict[str, Any]:
    def node(inputs: dict, outputs: list[str], *, output_node: bool = False) -> dict:
        return {
            "input": {"required": inputs},
            "input_order": {"required": list(inputs)},
            "output": outputs,
            "output_name": outputs,
            "output_node": output_node,
            "category": "test",
            "display_name": "x",
            "python_module": "nodes",
        }

    return {
        "VAELoader": node({"vae_name": [list(VAES)]}, ["VAE"]),
        "SaveAny": node({"vae": ["VAE"], "flag": ["BOOLEAN", {"default": "True"}]}, [], output_node=True),
        "CustomCombo": node({"choice": ["COMBO", {"multiselect": False, "options": []}]}, ["STRING", "INT"]),
        "TakeString": node({"text": ["STRING", {"forceInput": True}]}, [], output_node=True),
        "Prompt": node({"text": ["STRING", {"multiline": True}]}, ["STRING"]),
    }


@pytest.fixture
def graph() -> Graph:
    return Graph.from_object_info(_object_info())


class TestPrecisionSibling:
    def test_int8_convrot_matches_the_installed_fp16(self):
        assert (
            precision_sibling("minimax_h3_video_vae_int8_convrot.safetensors", VAES)
            == "minimax_h3_video_vae_fp16.safetensors"
        )

    @pytest.mark.parametrize(
        "value",
        [
            "minimax_h3_video_vae_fp16.safetensors",  # already installed
            "minimax_h3_music_vae_int8.safetensors",  # a different model
            "qwen3vl_8b_int8_convrot.safetensors",  # two candidate precisions
            "minimax_h3_video_vae_int8.ckpt",  # different file format
            "minimax_h3_video_vaeINT8.safetensors",  # tag not a separate token
            "trellis_2_shape_vae_bf16.safetensors",  # nothing similar installed
        ],
    )
    def test_never_a_different_model_or_a_choice(self, value):
        assert precision_sibling(value, VAES) is None

    def test_key_drops_only_whole_precision_tokens(self):
        assert precision_key("wan2.2_t2v_14B_fp8_e4m3fn_scaled.safetensors") == ("", "wan2.2_t2v_14B", "safetensors")
        assert precision_key("pruned_int8.safetensors") == ("", "pruned", "safetensors")
        assert precision_key("readme.txt") is None


class TestPrecisionTokensAreTags:
    """A word that only MODIFIES a precision
    ("scaled", "convrot", an fp8 format) is not a precision tag on its own,
    and a tag that leads the name is the model's name, not its precision."""

    @pytest.mark.parametrize(
        ("value", "options"),
        [
            ("realesrgan_x4_scaled.pth", ["realesrgan_x4.pth"]),  # a lone modifier is part of the name
            ("nvfp4_block.safetensors", ["block.safetensors"]),  # a leading tag is the name
            ("upscaler_convrot.safetensors", ["upscaler.safetensors"]),
            ("model_e4m3fn.safetensors", ["model.safetensors"]),
        ],
    )
    def test_a_modifier_or_a_leading_tag_is_not_a_precision(self, value, options):
        assert precision_sibling(value, options) is None

    def test_modifiers_still_drop_beside_a_precision(self):
        assert precision_key("wan_fp8_e4m3fn_scaled.safetensors") == ("", "wan", "safetensors")
        assert precision_key("vae_int8_convrot.safetensors") == ("", "vae", "safetensors")
        assert precision_key("realesrgan_x4_scaled.pth") == ("", "realesrgan_x4_scaled", "pth")


class TestGgufQuantTags:
    def test_one_gguf_quant_is_a_sibling_of_another(self):
        opts = ["wan2.2_t2v_14b-Q8_0.gguf", "flux1-dev-Q4_K_S.gguf"]
        assert precision_sibling("wan2.2_t2v_14b-Q4_K_M.gguf", opts) == "wan2.2_t2v_14b-Q8_0.gguf"

    @pytest.mark.parametrize(
        "name",
        ["m-Q4_0.gguf", "m-q4_1.gguf", "m-Q5_K_M.gguf", "m-Q6_K.gguf", "m-Q3_K_L.gguf", "m-IQ4_XS.gguf", "m-F16.gguf"],
    )
    def test_quant_tags_drop_on_a_gguf(self, name):
        assert precision_key(name) == ("", "m", "gguf")

    def test_a_leading_quant_tag_is_the_name_and_keeps_its_identity(self):
        assert precision_key("Q4_K_M_block.gguf") != precision_key("Q8_0_block.gguf")
        assert precision_sibling("Q4_K_M_block.gguf", ["Q8_0_block.gguf"]) is None

    def test_a_quant_tag_is_not_a_precision_outside_a_gguf(self):
        assert precision_sibling("m-Q8_0.safetensors", ["m.safetensors"]) is None
        assert precision_sibling("m-Q8_0.gguf", ["m-fp16.safetensors"]) is None


class TestSiblingIsTheSameFileInAnotherPrecision:
    """A sibling sits in the same folder, differs only in the stem's TRAILING
    precision run, and really differs in precision."""

    @pytest.mark.parametrize(
        ("value", "options"),
        [
            # another folder is another model, whatever the stem says
            ("SDXL/lightning_fp16.safetensors", ["SD15/lightning.safetensors"]),
            ("wan/lightx2v_lora_fp8.safetensors", ["hunyuan/lightx2v_lora.safetensors"]),
            ("SDXL/lightning_fp16.safetensors", ["sdxl/lightning.safetensors"]),
            ("lightning_fp16.safetensors", ["SDXL/lightning.safetensors"]),
            ("SDXL/lightning_fp16.safetensors", ["lightning.safetensors"]),
            # a precision word inside the name is not the build's precision
            ("flux_fp8_e4m3fn_lora.safetensors", ["flux_lora.safetensors"]),
            ("model_fp16_lora_fp8.safetensors", ["model_lora.safetensors"]),
            ("realesrgan_x4_scaled_fp16.pth", ["realesrgan_x4.pth"]),
            # a different spelling is a different file, not another precision
            ("flux1-dev.safetensors", ["flux1_dev.safetensors"]),
            ("sd_xl_base_1.0.safetensors", ["sd_xl_base_1_0.safetensors"]),
            ("Flux1-Dev.safetensors", ["flux1-dev.safetensors"]),
            ("flux1-dev-fp8.safetensors", ["flux1_dev_fp16.safetensors"]),
            ("flux1-dev-fp8.safetensors", ["flux1-dev_fp8.safetensors"]),
            ("m-Q4_K_M.gguf", ["m-q4_k_m.gguf"]),
        ],
    )
    def test_not_a_sibling(self, value, options):
        assert precision_sibling(value, options) is None

    @pytest.mark.parametrize(
        ("value", "options", "expected"),
        [
            (
                "SDXL/lightning_fp16.safetensors",
                ["SD15/lightning.safetensors", "SDXL/lightning.safetensors"],
                "SDXL/lightning.safetensors",
            ),
            ("SDXL\\lightning_fp16.safetensors", ["SDXL/lightning.safetensors"], "SDXL/lightning.safetensors"),
            (
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                ["umt5_xxl_fp16.safetensors"],
                "umt5_xxl_fp16.safetensors",
            ),
            (
                "umt5_xxl_fp16.safetensors",
                ["umt5_xxl_fp8_e4m3fn_scaled.safetensors"],
                "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
            ),
            ("flux1-dev-fp8.safetensors", ["flux1-dev.safetensors"], "flux1-dev.safetensors"),
            ("flux1-dev.safetensors", ["flux1-dev-fp8.safetensors"], "flux1-dev-fp8.safetensors"),
            ("flux1-dev-Q4_K_M.gguf", ["flux1-dev-Q8_0.gguf"], "flux1-dev-Q8_0.gguf"),
            ("wan2.1-t2v-14b-Q8_0.gguf", ["wan2.1-t2v-14b-Q4_K_M.gguf"], "wan2.1-t2v-14b-Q4_K_M.gguf"),
            ("realesrgan_x4_scaled_fp16.pth", ["realesrgan_x4_scaled.pth"], "realesrgan_x4_scaled.pth"),
        ],
    )
    def test_sibling(self, value, options, expected):
        assert precision_sibling(value, options) == expected

    def test_directory_is_part_of_the_key(self):
        assert precision_key("SDXL\\lightning_fp16.safetensors") == ("SDXL", "lightning", "safetensors")


def _template() -> dict[str, Any]:
    """A MiniMax-H3-shaped template: the VAE loader sits inside a subgraph
    definition, the instance carries a promoted copy of the value, and the
    loader's ``properties.models`` lists the download."""
    missing = "minimax_h3_video_vae_int8_convrot.safetensors"
    return {
        "id": "tpl",
        "nodes": [
            # Promoted: [vae_name, text]. The text widget happens to hold the same string.
            {"id": 105, "type": "sg-1", "inputs": [], "outputs": [], "widgets_values": [missing, missing]},
            {"id": 3, "type": "VAELoader", "outputs": [], "widgets_values": ["trellis_2_shape_vae_bf16.safetensors"]},
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": "sg-1",
                    "inputs": [
                        {"name": "vae_name", "type": "COMBO", "linkIds": [223]},
                        {"name": "text", "type": "STRING", "linkIds": [224]},
                    ],
                    "links": [
                        {"id": 223, "origin_id": -10, "origin_slot": 0, "target_id": 11, "target_slot": 0},
                        {"id": 224, "origin_id": -10, "origin_slot": 1, "target_id": 12, "target_slot": 0},
                    ],
                    "nodes": [
                        {
                            "id": 12,
                            "type": "Prompt",
                            "inputs": [{"name": "text", "type": "STRING", "widget": {"name": "text"}, "link": 224}],
                            "outputs": [],
                            "widgets_values": ["a prompt"],
                        },
                        {
                            "id": 11,
                            "type": "VAELoader",
                            "inputs": [
                                {"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}, "link": 223}
                            ],
                            "outputs": [],
                            "widgets_values": [missing],
                            "properties": {
                                "models": [
                                    {
                                        "name": missing,
                                        "url": "https://example.invalid/vae.safetensors",
                                        "directory": "vae",
                                    }
                                ]
                            },
                        },
                    ],
                }
            ]
        },
    }


class TestResolveWorkflowModels:
    def test_swaps_interior_promoted_and_download_list(self, graph):
        wf = _template()
        subs, unavailable = resolve_workflow_models(wf, graph)

        interior = wf["definitions"]["subgraphs"][0]["nodes"][1]
        assert interior["widgets_values"] == ["minimax_h3_video_vae_fp16.safetensors"]
        # Only the promoted slot bound to the swapped widget follows; the text
        # widget holding the same string is not a model selector.
        assert wf["nodes"][0]["widgets_values"] == [
            "minimax_h3_video_vae_fp16.safetensors",
            "minimax_h3_video_vae_int8_convrot.safetensors",
        ]
        assert interior["properties"]["models"] == [
            {"name": "minimax_h3_video_vae_fp16.safetensors", "directory": "vae"}
        ]
        assert [(s["code"], s["node_id"], s["subgraph"], s["field"], s["from"], s["to"]) for s in subs] == [
            (
                "normalized_value",
                11,
                "sg-1",
                "vae_name",
                "minimax_h3_video_vae_int8_convrot.safetensors",
                "minimax_h3_video_vae_fp16.safetensors",
            )
        ]

        # No sibling: left as is, reported with the closest options.
        assert wf["nodes"][1]["widgets_values"] == ["trellis_2_shape_vae_bf16.safetensors"]
        assert [(u["code"], u["node_id"], u["value"]) for u in unavailable] == [
            ("model_unavailable", 3, "trellis_2_shape_vae_bf16.safetensors")
        ]

    def test_only_instances_of_the_changed_definition_follow(self, graph):
        wf = _template()
        missing = "minimax_h3_video_vae_int8_convrot.safetensors"
        wf["nodes"].append({"id": 7, "type": "sg-other", "widgets_values": [missing]})
        resolve_workflow_models(wf, graph)
        assert wf["nodes"][-1]["widgets_values"] == [missing]

    def test_follows_a_widget_promoted_through_nested_subgraphs(self, graph):
        missing = "minimax_h3_video_vae_int8_convrot.safetensors"
        wf = _template()
        wf["nodes"] = [{"id": 200, "type": "sg-outer", "widgets_values": [missing]}]
        wf["definitions"]["subgraphs"].append(
            {
                "id": "sg-outer",
                "inputs": [{"name": "vae", "type": "COMBO", "linkIds": [300]}],
                "links": [{"id": 300, "origin_id": -10, "origin_slot": 0, "target_id": 50, "target_slot": 0}],
                "nodes": [
                    {
                        "id": 50,
                        "type": "sg-1",
                        "inputs": [{"name": "vae_name", "type": "COMBO", "widget": {"name": "vae_name"}, "link": 300}],
                        "widgets_values": [missing, missing],
                    }
                ],
            }
        )
        resolve_workflow_models(wf, graph)
        inner_instance = wf["definitions"]["subgraphs"][1]["nodes"][0]
        assert inner_instance["widgets_values"] == ["minimax_h3_video_vae_fp16.safetensors", missing]
        assert wf["nodes"][0]["widgets_values"] == ["minimax_h3_video_vae_fp16.safetensors"]

    def test_installed_files_are_untouched(self, graph):
        wf = _template()
        wf["definitions"]["subgraphs"][0]["nodes"][1]["widgets_values"] = ["ae.safetensors"]
        wf["nodes"] = []
        before = copy.deepcopy(wf)
        assert resolve_workflow_models(wf, graph) == ([], [])
        assert wf == before

    def test_top_level_only_resolution_preserves_nested_identity_without_deepcopy(self, graph, monkeypatch):
        missing = "minimax_h3_video_vae_int8_convrot.safetensors"
        node = {"id": 1, "type": "VAELoader", "widgets_values": [missing]}
        workflow = {"nodes": [node]}

        def unexpected_copy(_workflow):
            raise AssertionError("top-level model resolution must not deepcopy the workflow")

        monkeypatch.setattr("comfy_cli.model_variants.copy.deepcopy", unexpected_copy)

        substitutions, unavailable = resolve_workflow_models(workflow, graph)

        assert substitutions and not unavailable
        assert workflow["nodes"][0] is node
        assert node["widgets_values"] == ["minimax_h3_video_vae_fp16.safetensors"]

    def test_promoted_resolution_reports_deepcopy_recursion_without_mutation(self, graph, monkeypatch):
        workflow = _template()
        before = copy.deepcopy(workflow)

        def recursive_copy(_workflow):
            raise RecursionError("workflow nesting is too deep")

        monkeypatch.setattr("comfy_cli.model_variants.copy.deepcopy", recursive_copy)

        with pytest.raises(ModelVariantResolutionError, match="workflow nesting is too deep"):
            resolve_workflow_models(workflow, graph)

        assert workflow == before

    def test_promotion_budget_failure_happens_before_model_mutation(self, graph, monkeypatch):
        definitions: dict[str, dict] = {}
        depth = 18
        for level in reversed(range(depth)):
            for side in ("left", "right"):
                definition_id = f"{side}-{level}"
                next_level = level + 1
                child_types = (
                    (f"left-{next_level}", f"right-{next_level}") if next_level < depth else ("PlainLeft", "PlainRight")
                )
                definitions[definition_id] = {
                    "id": definition_id,
                    "inputs": [{"name": "value", "type": "STRING", "linkIds": [1, 2]}],
                    "nodes": [
                        {
                            "id": index,
                            "type": child_type,
                            "inputs": [{"name": "value", "type": "STRING", "link": index + 1}],
                        }
                        for index, child_type in enumerate(child_types)
                    ],
                    "links": [
                        {"id": 1, "origin_id": -10, "origin_slot": 0, "target_id": 0, "target_slot": 0},
                        {"id": 2, "origin_id": -10, "origin_slot": 0, "target_id": 1, "target_slot": 0},
                    ],
                }
        definitions["left-0"]["nodes"].append(
            {
                "id": 99,
                "type": "VAELoader",
                "widgets_values": ["minimax_h3_video_vae_int8_convrot.safetensors"],
            }
        )
        workflow = {
            "nodes": [{"id": 100, "type": "left-0", "widgets_values": []}],
            "definitions": {"subgraphs": list(definitions.values())},
        }
        before = copy.deepcopy(workflow)

        # Keep this transactional-failure regression independent of traversal
        # optimizations: shared acyclic DAGs are intentionally memoized now.
        monkeypatch.setattr("comfy_cli.cql.promoted._promotion_visit_limit", lambda *_args: 1)

        with pytest.raises(ModelVariantResolutionError, match="input traversal"):
            resolve_workflow_models(workflow, graph)

        assert workflow == before

    def test_unrelated_definition_is_not_preflighted(self, graph, monkeypatch):
        from comfy_cli.cql import promoted

        wf = _template()
        wf["definitions"]["subgraphs"].append({"id": "unrelated", "inputs": [], "nodes": [], "links": []})
        original = promoted.promoted_inputs
        visited: list[str] = []

        def guarded(definition, definitions, *args, **kwargs):
            definition_id = str(definition.get("id"))
            visited.append(definition_id)
            if definition_id == "unrelated":
                raise promoted.PromotionTraversalLimitError("unrelated definition was traversed")
            return original(definition, definitions, *args, **kwargs)

        monkeypatch.setattr(promoted, "promoted_inputs", guarded)

        substitutions, unavailable = resolve_workflow_models(wf, graph)

        assert substitutions
        assert unavailable
        assert "sg-1" in visited
        assert "unrelated" not in visited


# ---------------------------------------------------------------------------
# templates fetch
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


@pytest.fixture
def patched_fetch(monkeypatch: pytest.MonkeyPatch):
    row = {"name": "video_minimax_h3_i2v", "title": "H3", "output_type": "video", "tags": [], "models": []}
    monkeypatch.setattr(templates_cmd, "_load_gallery", lambda *a, **kw: [{"templates": []}])
    monkeypatch.setattr(templates_cmd, "_flatten_templates", lambda cats: [dict(row)])
    monkeypatch.setattr(templates_cmd, "_fetch_template_workflow", lambda name, **kw: json.dumps(_template()).encode())
    monkeypatch.delenv("COMFY_OBJECT_INFO_FILE", raising=False)


def _run(args: list[str], capsys) -> dict[str, Any]:
    r = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    r.mode = OutputMode.JSON
    set_renderer(r)
    result = CliRunner().invoke(templates_cmd.app, args, standalone_mode=False)
    out = capsys.readouterr().out or result.stdout or ""
    for line in reversed(out.strip().splitlines()):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    raise AssertionError(f"no envelope: {result.exception} {out[:400]}")


class TestTemplatesFetch:
    def test_catalog_from_env_swaps_the_precision_variant(self, tmp_path: Path, patched_fetch, monkeypatch, capsys):
        info = tmp_path / "object_info.json"
        info.write_text(json.dumps(_object_info()))
        monkeypatch.setenv("COMFY_OBJECT_INFO_FILE", str(info))
        out = tmp_path / "workflow.json"

        env = _run(["fetch", "video_minimax_h3_i2v", "-o", str(out)], capsys)

        assert env["ok"] is True, env
        written = json.loads(out.read_text())
        assert written["definitions"]["subgraphs"][0]["nodes"][1]["widgets_values"] == [
            "minimax_h3_video_vae_fp16.safetensors"
        ]
        data = env["data"]
        assert [s["to"] for s in data["model_substitutions"]] == ["minimax_h3_video_vae_fp16.safetensors"]
        assert [u["value"] for u in data["unavailable_models"]] == ["trellis_2_shape_vae_bf16.safetensors"]
        assert "different model" in data["unavailable_models_hint"]

    def test_input_flag_works_without_env(self, tmp_path: Path, patched_fetch, capsys):
        info = tmp_path / "object_info.json"
        info.write_text(json.dumps(_object_info()))
        out = tmp_path / "workflow.json"
        env = _run(["fetch", "video_minimax_h3_i2v", "-o", str(out), "--input", str(info)], capsys)
        assert env["data"]["model_substitutions"][0]["from"] == "minimax_h3_video_vae_int8_convrot.safetensors"

    def test_no_catalog_writes_the_template_verbatim(self, tmp_path: Path, patched_fetch, capsys):
        out = tmp_path / "workflow.json"
        env = _run(["fetch", "video_minimax_h3_i2v", "-o", str(out)], capsys)
        assert env["ok"] is True
        assert "model_substitutions" not in env["data"]
        assert json.loads(out.read_text()) == _template()

    def test_unreadable_catalog_does_not_fail_the_fetch(self, tmp_path: Path, patched_fetch, capsys):
        out = tmp_path / "workflow.json"
        env = _run(["fetch", "video_minimax_h3_i2v", "-o", str(out), "--input", str(tmp_path / "nope.json")], capsys)
        assert env["ok"] is True
        assert "could not load object_info" in env["data"]["model_check_skipped"]
        assert json.loads(out.read_text()) == _template()

    def test_promotion_limit_skips_model_check_with_structured_reason(self, tmp_path: Path, monkeypatch):
        info = tmp_path / "object_info.json"
        info.write_text(json.dumps(_object_info()))
        workflow = _template()
        before = copy.deepcopy(workflow)

        def fail_resolution(*_args, **_kwargs):
            raise ModelVariantResolutionError("promoted input traversal exceeded its safe limit")

        monkeypatch.setattr("comfy_cli.model_variants.resolve_workflow_models", fail_resolution)

        notes = templates_cmd._resolve_template_models(workflow, str(info))

        assert "promoted input traversal exceeded" in notes["model_check_skipped"]
        assert workflow == before


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


def _api(nodes: dict) -> dict:
    return nodes


class TestValidate:
    def test_unknown_model_names_its_precision_sibling(self, graph):
        res = graph.validate_workflow(
            {
                "1": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"},
                },
                "2": {"class_type": "SaveAny", "inputs": {"vae": ["1", 0], "flag": True}},
            }
        )
        (err,) = [e for e in res["errors"] if e["code"] == "unknown_enum_value"]
        assert "closest: minimax_h3_video_vae_fp16.safetensors," in err["message"]
        assert "'minimax_h3_video_vae_fp16.safetensors' is the same model in another precision" in err["message"]
        # The structured finding the message comes from carries it as a field.
        port = graph.node("VAELoader").inputs[0]
        (finding,) = port.validate_catalog("minimax_h3_video_vae_int8_convrot.safetensors")
        assert finding["precision_sibling"] == "minimax_h3_video_vae_fp16.safetensors"

    def test_custom_combo_choice_is_frontend_defined(self, graph):
        res = graph.validate_workflow(
            {
                "1": {"class_type": "CustomCombo", "inputs": {"choice": "Music"}},
                "2": {"class_type": "TakeString", "inputs": {"text": ["1", 0]}},
            }
        )
        assert res["errors"] == [], res["errors"]

    @pytest.mark.parametrize(("value", "ok"), [("True", True), ("true", True), ("False", False), ("yes", False)])
    def test_boolean_string(self, graph, value, ok):
        res = graph.validate_workflow(
            {
                "1": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
                "2": {"class_type": "SaveAny", "inputs": {"vae": ["1", 0], "flag": value}},
            }
        )
        errs = [e for e in res["errors"] if e.get("field") == "flag"]
        assert (errs == []) is ok, errs
        if value == "False":
            assert "any non-empty string as true" in errs[0]["message"]
