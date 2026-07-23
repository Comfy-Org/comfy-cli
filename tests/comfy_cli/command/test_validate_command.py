"""Tests for `comfy validate` — frontend-format (UI-export) auto-conversion.

`comfy validate --workflow <ui-export.json>` used to validate vacuously: a
UI-export file's wrapper keys (`nodes`, `links`, `groups`, `config`, …) each
emitted a `non_node_key` warning, zero nodes were checked, and the verdict was
`valid:true`. The command now detects UI format (`is_ui_workflow`) and lowers it
to API format with `convert_ui_to_api` — exactly as `comfy run` does — before
validating, so the verdict reflects the real graph and the payload carries
`converted_from_ui: true` plus the converted node count.

Offline mode (`--input <object_info.json>`) is used throughout so no server is
needed: the same file supplies both the graph and the converter's object_info.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app

FIXTURES = Path(__file__).parent.parent / "fixtures"
OBJECT_INFO = FIXTURES / "sd15_object_info.json"
UI_WORKFLOW = FIXTURES / "sd15_ui_workflow.json"


@pytest.fixture
def runner():
    return CliRunner()


def _write(tmp_path: Path, name: str, obj) -> Path:
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _envelope(result) -> dict:
    """Parse the final JSON envelope line emitted in `--json` mode."""
    return json.loads(result.stdout.strip().splitlines()[-1])


def _validate(runner: CliRunner, workflow: Path):
    """Invoke `comfy --json validate` offline against the sd15 object_info."""
    return runner.invoke(
        app,
        ["--json", "validate", "--workflow", str(workflow), "--input", str(OBJECT_INFO)],
        env={"COMFY_WHERE": "local"},
    )


def test_ui_export_is_converted_and_validated(runner):
    """A UI-export fixture validates against the CONVERTED graph: a truthful
    verdict, `converted_from_ui: true`, the converted node count, and zero
    `non_node_key` wrapper-key noise."""
    result = _validate(runner, UI_WORKFLOW)

    assert result.exit_code == 0, result.stdout
    data = _envelope(result)["data"]
    assert data["valid"] is True
    assert data["converted_from_ui"] is True
    # The sd15 UI workflow lowers to 7 API nodes.
    assert data["converted_node_count"] == 7
    # The wrapper keys (nodes/links/groups/config/…) are gone after conversion,
    # so none of them can produce the old vacuous-pass warnings.
    assert [w for w in data["warnings"] if w.get("code") == "non_node_key"] == []


def test_ui_export_surfaces_real_problems(runner, tmp_path):
    """Acceptance: the converted graph is really validated — an unknown node
    type surfaces as `valid:false` (not a vacuous pass), while still flagging
    the file as UI-converted."""
    bad = {
        "nodes": [{"id": 1, "type": "TotallyMadeUpNode", "mode": 0, "inputs": [], "outputs": [], "widgets_values": []}],
        "links": [],
    }
    wf = _write(tmp_path, "bad_ui.json", bad)

    result = _validate(runner, wf)

    assert result.exit_code == 1
    data = _envelope(result)["data"]
    assert data["valid"] is False
    assert data["converted_from_ui"] is True
    assert any(e["code"] == "unknown_class_type" for e in data["errors"])


def test_ui_export_that_converts_to_nothing_is_rejected(runner, tmp_path):
    """A UI file whose nodes carry no usable `type` converts to zero executable
    nodes → structured `workflow_not_api_format` error, exit 1, message naming
    the conversion."""
    empty_convert = {"nodes": [{"id": 1, "mode": 0, "inputs": [], "outputs": []}], "links": []}
    wf = _write(tmp_path, "no_exec_ui.json", empty_convert)

    result = _validate(runner, wf)

    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "workflow_not_api_format"
    assert "convert" in error["message"].lower()


def test_api_format_unchanged(runner, tmp_path):
    """An API-format file behaves exactly as before: validated directly, no
    `converted_from_ui` key in the payload. The fixture is a complete sd15
    txt2img graph — it carries a SaveImage output node, as any real API-format
    export does, so it clears the server-parity no-outputs check (BE-3357)."""
    api = {
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
    wf = _write(tmp_path, "api.json", api)

    result = _validate(runner, wf)

    assert result.exit_code == 0
    data = _envelope(result)["data"]
    assert data["valid"] is True
    assert "converted_from_ui" not in data


def test_non_dict_payload_unchanged(runner, tmp_path):
    """A non-dict JSON payload keeps its existing `workflow_not_api_format`
    error (the UI-detection branch never runs for it)."""
    wf = _write(tmp_path, "list.json", [1, 2, 3])

    result = _validate(runner, wf)

    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "workflow_not_api_format"


def test_empty_dict_payload_not_converted_and_rejected(runner, tmp_path):
    """An empty dict is not UI format and is left to the existing validator
    (no conversion, no `converted_from_ui` key) — which now rejects it: the
    server refuses any prompt with zero output nodes, including a node-less
    one, so validate mirrors that as `prompt_no_outputs` (BE-3357)."""
    wf = _write(tmp_path, "empty.json", {})

    result = _validate(runner, wf)

    assert result.exit_code == 1
    data = _envelope(result)["data"]
    assert "converted_from_ui" not in data
    assert data["valid"] is False
    assert any(e["code"] == "prompt_no_outputs" for e in data["errors"])
