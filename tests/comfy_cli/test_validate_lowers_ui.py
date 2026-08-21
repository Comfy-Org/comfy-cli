"""`comfy validate` must lower a frontend/canvas graph to API format first.

Regression for the bug where ``validate_workflow`` (which only inspects the
API/prompt shape ``{id: {class_type, inputs}}``) never iterated the nodes of a
frontend ``{nodes: [...], links: [...]}`` graph and therefore returned
``valid:true`` for a structurally broken canvas workflow. The ``validate``
command now converts a frontend graph with the SAME converter the ``run`` path
uses before calling ``validate_workflow``.

Layered:
  * direct convert + ``Graph.validate_workflow`` (the empirical core), and
  * CLI-level envelope tests via ``CliRunner``.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from comfy_cli.cql.engine import Graph
from comfy_cli.workflow_to_api import convert_ui_to_api, is_api_format

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Fixtures: a small /object_info covering the SD1.5 text-to-image fixture, plus
# helpers to load and break that frontend workflow.
# ---------------------------------------------------------------------------


def _object_info() -> dict:
    """Schemas for every node type in ``sd15_ui_workflow.json``.

    ``ckpt_name`` lists the exact checkpoint the fixture uses so the valid
    graph validates clean (no spurious ``unknown_enum_value``).
    """
    return {
        "CheckpointLoaderSimple": {
            "input": {"required": {"ckpt_name": [["v1-5-pruned-emaonly-fp16.safetensors", "sd_xl_base.safetensors"]]}},
            "input_order": {"required": ["ckpt_name"]},
            "output": ["MODEL", "CLIP", "VAE"],
            "output_name": ["MODEL", "CLIP", "VAE"],
            "display_name": "Load Checkpoint",
            "output_node": False,
        },
        "KSampler": {
            "input": {
                "required": {
                    "model": "MODEL",
                    "positive": "CONDITIONING",
                    "negative": "CONDITIONING",
                    "latent_image": "LATENT",
                    "seed": ["INT", {"default": 0, "control_after_generate": True}],
                    "steps": ["INT", {"default": 20}],
                    "cfg": ["FLOAT", {"default": 8.0}],
                    "sampler_name": [["euler", "euler_ancestral"]],
                    "scheduler": [["normal", "karras"]],
                    "denoise": ["FLOAT", {"default": 1.0}],
                }
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
            "display_name": "KSampler",
            "output_node": False,
        },
        "CLIPTextEncode": {
            "input": {"required": {"text": ["STRING", {"multiline": True}], "clip": "CLIP"}},
            "input_order": {"required": ["clip", "text"]},
            "output": ["CONDITIONING"],
            "output_name": ["CONDITIONING"],
            "display_name": "CLIP Text Encode",
            "output_node": False,
        },
        "VAEDecode": {
            "input": {"required": {"samples": "LATENT", "vae": "VAE"}},
            "output": ["IMAGE"],
            "output_name": ["IMAGE"],
            "display_name": "VAE Decode",
            "output_node": False,
        },
        "SaveImage": {
            "input": {"required": {"images": "IMAGE", "filename_prefix": ["STRING", {"default": "ComfyUI"}]}},
            "input_order": {"required": ["images", "filename_prefix"]},
            "output": [],
            "output_name": [],
            "display_name": "Save Image",
            "output_node": True,
        },
        "EmptyLatentImage": {
            "input": {
                "required": {
                    "width": ["INT", {"default": 512}],
                    "height": ["INT", {"default": 512}],
                    "batch_size": ["INT", {"default": 1}],
                }
            },
            "input_order": {"required": ["width", "height", "batch_size"]},
            "output": ["LATENT"],
            "output_name": ["LATENT"],
            "display_name": "Empty Latent Image",
            "output_node": False,
        },
    }


def _sd15_ui() -> dict:
    return json.loads((FIXTURES / "sd15_ui_workflow.json").read_text(encoding="utf-8"))


def _break_model_link(wf: dict) -> dict:
    """Delete the litegraph link (id 1) that feeds KSampler(id=3).model.

    Mirrors a user deleting a required-input wire on the canvas: the link is
    removed from ``links`` and the target input's ``link`` is nulled. After
    conversion the ``model`` input becomes ABSENT (not a dangling reference).
    """
    wf = json.loads(json.dumps(wf))
    wf["links"] = [link for link in wf["links"] if link[0] != 1]
    for node in wf["nodes"]:
        if node.get("id") == 3:
            for inp in node.get("inputs", []):
                if inp.get("name") == "model":
                    inp["link"] = None
    return wf


# ---------------------------------------------------------------------------
# Layer 1 — the empirical core: convert a frontend graph, then validate.
# ---------------------------------------------------------------------------


class TestConvertThenValidate:
    def test_deleted_required_link_becomes_absent_and_is_flagged(self):
        """A deleted required-input wire → absent input → required_input_missing.

        This is the exact real-world failure the fix targets: the agent's
        canvas graph had a required KSampler input unwired, yet the API-only
        validator passed it. Verifies (a) the converter drops the input rather
        than emitting a dangling edge, and (b) validate_workflow catches the
        absent required input.
        """
        oi = _object_info()
        api = convert_ui_to_api(_break_model_link(_sd15_ui()), oi)

        # The required input is gone from the lowered node (not a dangling ref).
        assert "model" not in api["3"]["inputs"]

        result = Graph.from_object_info(oi).validate_workflow(api)
        assert result["valid"] is False
        missing = [e for e in result["errors"] if e["code"] == "required_input_missing"]
        assert any(e["node_id"] == "3" and e["field"] == "model" for e in missing)

    def test_valid_frontend_graph_lowers_and_validates_clean(self):
        oi = _object_info()
        api = convert_ui_to_api(_sd15_ui(), oi)
        # Lowered form is API-shaped; the KSampler's model IS wired.
        assert is_api_format(api)
        assert api["3"]["inputs"]["model"] == ["4", 0]

        result = Graph.from_object_info(oi).validate_workflow(api)
        assert result["valid"] is True, result["errors"]


# ---------------------------------------------------------------------------
# Layer 2 — CLI envelope tests: `comfy validate` end-to-end.
# ---------------------------------------------------------------------------


def _run_validate(tmp_path: Path, workflow: dict, object_info: dict):
    from comfy_cli.cmdline import app

    wf_path = tmp_path / "workflow.json"
    wf_path.write_text(json.dumps(workflow), encoding="utf-8")
    oi_path = tmp_path / "object_info.json"
    oi_path.write_text(json.dumps(object_info), encoding="utf-8")

    return CliRunner().invoke(
        app,
        ["validate", "--workflow", str(wf_path), "--input", str(oi_path), "--where", "local"],
        env={"COMFY_OUTPUT": "json"},
    )


def _envelope(result) -> dict:
    lines = [ln for ln in result.stdout.splitlines() if ln.strip().startswith("{")]
    assert lines, f"no JSON envelope in output: {result.stdout!r}"
    return json.loads(lines[-1])


_SG_UUID = "f2fdebf6-dfaf-43b6-9eb2-7f70613cfdc1"


def _subgraph_ui_workflow() -> dict:
    """A frontend graph whose only real node is a SaveImage INSIDE a subgraph
    instance (id 57), with none of its link inputs wired — so lowering expands
    it to the composite id ``57:3`` and validation flags its missing inputs.

    An OUTPUT node specifically: BE-3406 prunes output-unreachable nodes before
    the required-input check, so a non-output interior node would produce no
    errors for this test to inspect the ids of."""
    return {
        "last_node_id": 60,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 57,
                "type": _SG_UUID,
                "pos": [0, 0],
                "inputs": [],
                "outputs": [],
                "properties": {},
            }
        ],
        "links": [],
        "definitions": {
            "subgraphs": [
                {
                    "id": _SG_UUID,
                    "name": "Sampler",
                    "inputs": [],
                    "outputs": [],
                    "nodes": [
                        {
                            "id": 3,
                            "type": "SaveImage",
                            "inputs": [],
                            "widgets_values": ["ComfyUI"],
                        }
                    ],
                    "links": [],
                }
            ]
        },
    }


class TestValidateSubgraphIdTranslation:
    """Validate errors on subgraph interiors must be keyed by the EDITABLE
    address (`57/3` — what set-widget/slots speak), not the flattened API id
    (`57:3` — what lowering mints). Callers only ever saw the pre-flatten ids;
    feeding `57:3` back into the edit surface used to dead-end with
    `node 57:3 not found in workflow`."""

    def test_lowered_subgraph_error_uses_editable_address(self, tmp_path):
        result = _run_validate(tmp_path, _subgraph_ui_workflow(), _object_info())
        assert result.exit_code == 1
        env = _envelope(result)
        missing = [e for e in env["data"]["errors"] if e["code"] == "required_input_missing"]
        assert missing, env["data"]["errors"]
        for e in missing:
            assert e["node_id"] == "57/3", e
            assert e["api_node_id"] == "57:3", e

    def test_api_format_input_keeps_raw_ids(self, tmp_path):
        # A caller that hands us an already-lowered API doc addresses THAT doc;
        # its ids pass through untouched (no api_node_id annotation).
        api = {"57:3": {"class_type": "SaveImage", "inputs": {}}}  # output node: reachable (BE-3406)
        result = _run_validate(tmp_path, api, _object_info())
        assert result.exit_code == 1
        env = _envelope(result)
        missing = [e for e in env["data"]["errors"] if e["code"] == "required_input_missing"]
        assert missing
        for e in missing:
            assert e["node_id"] == "57:3", e
            assert "api_node_id" not in e, e


class TestValidateCLI:
    def test_broken_frontend_graph_is_flagged(self, tmp_path):
        result = _run_validate(tmp_path, _break_model_link(_sd15_ui()), _object_info())
        assert result.exit_code == 1
        env = _envelope(result)
        assert env["ok"] is False
        assert env["data"]["valid"] is False
        codes = {(e["code"], e["node_id"], e["field"]) for e in env["data"]["errors"]}
        assert ("required_input_missing", "3", "model") in codes

    def test_valid_frontend_graph_passes(self, tmp_path):
        result = _run_validate(tmp_path, _sd15_ui(), _object_info())
        assert result.exit_code == 0
        env = _envelope(result)
        assert env["ok"] is True
        assert env["data"]["valid"] is True
        assert env["data"]["error_count"] == 0

    def test_already_api_format_is_validated_unchanged(self, tmp_path):
        # A pre-lowered (already-API) graph must still be validated, and must
        # NOT be double-converted. Feeding the lowered valid graph stays valid.
        api = convert_ui_to_api(_sd15_ui(), _object_info())
        assert is_api_format(api)
        result = _run_validate(tmp_path, api, _object_info())
        assert result.exit_code == 0
        assert _envelope(result)["data"]["valid"] is True

    def test_already_api_format_broken_is_still_flagged(self, tmp_path):
        # Drop the model input from the already-API KSampler node: since the
        # graph is already API-shaped, no conversion happens, but validation
        # still runs and catches the absent required input.
        api = convert_ui_to_api(_sd15_ui(), _object_info())
        del api["3"]["inputs"]["model"]
        result = _run_validate(tmp_path, api, _object_info())
        assert result.exit_code == 1
        env = _envelope(result)
        assert env["data"]["valid"] is False
        assert any(
            e["code"] == "required_input_missing" and e["node_id"] == "3" and e["field"] == "model"
            for e in env["data"]["errors"]
        )
