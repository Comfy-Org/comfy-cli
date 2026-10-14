from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app

SCHEMA = Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas" / "build_inventory.json"


@pytest.fixture(autouse=True)
def quiet_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)


def _invoke(*args: str):
    result = CliRunner().invoke(
        cli_app,
        ["build", "inventory", *args],
        env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"},
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return result, json.loads(lines[-1])


def _write_workflow(tmp_path: Path) -> Path:
    workflow = {
        "nodes": [
            {"id": 1, "type": "CheckpointLoaderSimple", "mode": 0, "widgets_values": ["sdxl/base.safetensors"]},
            {"id": 2, "type": "LoraLoaderModelOnly", "mode": 0, "widgets_values": ["private_v1.safetensors", 1.0]},
            {"id": 3, "type": "UpscaleModelLoader", "mode": 2, "widgets_values": ["muted.pth"]},
        ]
    }
    path = tmp_path / "wf.json"
    path.write_text(json.dumps(workflow), encoding="utf-8")
    return path


def test_inventory_reports_scope_and_local_files(tmp_path: Path):
    workflow = _write_workflow(tmp_path)
    models = tmp_path / "models" / "loras"
    models.mkdir(parents=True)
    (models / "private_v1.safetensors").write_bytes(b"LORA")

    result, envelope = _invoke(str(workflow), "--models-root", str(tmp_path / "models"), "--hash")

    assert result.exit_code == 0, result.stdout
    assert envelope["ok"] is True
    assert envelope["command"] == "build inventory"
    data = envelope["data"]
    jsonschema.validate(data, json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert [m["path"] for m in data["models"]] == ["sdxl/base.safetensors", "private_v1.safetensors"]
    assert data["skipped"][0]["why"] == "muted"
    assert data["models_not_on_disk"] == ["sdxl/base.safetensors"]
    assert data["models"][1]["local"][0]["type_dir"] == "loras"
    assert "sha256" in data["models"][1]["local"][0]


def test_inventory_cloud_uses_the_cloud_catalog(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workflow = _write_workflow(tmp_path)
    catalog = {
        "CheckpointLoaderSimple": {"inputs": [{"name": "ckpt_name", "choices": ["base.safetensors"]}]},
        "LoraLoaderModelOnly": {"inputs": [{"name": "lora_name", "choices": ["public.safetensors"]}]},
    }

    class FakeGraph:
        def node(self, name):
            return name if name in catalog else None

        def morphism_to_dict(self, node):
            return catalog[node]

    seen = {}

    def fake_get_graph(input_path, host, port, where=None, on_stale=None):
        seen["where"] = where
        return FakeGraph()

    monkeypatch.setattr("comfy_cli.command.nodes._get_graph", fake_get_graph)

    result, envelope = _invoke(str(workflow), "--cloud")

    assert result.exit_code == 0, result.stdout
    assert seen["where"] == "cloud"
    cloud = envelope["data"]["cloud"]
    assert cloud["verdict"] == "needs_build"
    assert cloud["blocking"] == ["private_v1.safetensors"]
    assert cloud["models"][0]["set_widget"] == "1.ckpt_name"
    jsonschema.validate(envelope["data"], json.loads(SCHEMA.read_text(encoding="utf-8")))


def test_inventory_unreachable_server_is_reported_not_fatal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workflow = _write_workflow(tmp_path)

    def refuse(url, **kwargs):
        raise ConnectionRefusedError(61, "Connection refused")

    monkeypatch.setattr("comfy_cli.http.plain_urlopen", refuse)

    result, envelope = _invoke(str(workflow), "--server", "http://127.0.0.1:8000")

    assert result.exit_code == 0, result.stdout
    section = envelope["data"]["local_server"]
    assert section["server"] == "http://127.0.0.1:8000"
    assert "ConnectionRefusedError" in section["error"]


def test_inventory_rejects_a_missing_or_malformed_workflow(tmp_path: Path):
    result, envelope = _invoke(str(tmp_path / "nope.json"))
    assert result.exit_code == 1
    assert envelope["error"]["code"] == "workflow_not_found"

    bad = tmp_path / "bad.json"
    bad.write_text("<html>", encoding="utf-8")
    result, envelope = _invoke(str(bad))
    assert result.exit_code == 1
    assert envelope["error"]["code"] == "workflow_invalid_json"

    listed = tmp_path / "list.json"
    listed.write_text("[]", encoding="utf-8")
    result, envelope = _invoke(str(listed))
    assert envelope["error"]["code"] == "workflow_invalid_json"
