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
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda where=None: object())

    result, envelope = _invoke(str(workflow), "--cloud")

    assert result.exit_code == 0, result.stdout
    assert seen["where"] == "cloud"
    cloud = envelope["data"]["cloud"]
    assert cloud["verdict"] == "needs_build_or_lossy_conversion"
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


def test_inventory_cloud_compares_local_weights_and_hashes_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    workflow = tmp_path / "wf.json"
    workflow.write_text(
        json.dumps(
            {
                "nodes": [
                    {"id": 1, "type": "UpscaleModelLoader", "mode": 0, "widgets_values": ["4x-UltraSharp.pth"]},
                    {"id": 2, "type": "LoadImage", "mode": 0, "widgets_values": ["face.png", "image"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    models = tmp_path / "models" / "upscale_models"
    models.mkdir(parents=True)
    (models / "4x-UltraSharp.pth").write_bytes(b"x" * 10)
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "face.png").write_bytes(b"PNG")
    from comfy_cli.hashing import blake3_file

    uploaded = blake3_file(inputs / "face.png").split(":", 1)[1] + ".png"
    catalog = {
        "UpscaleModelLoader": {"inputs": [{"name": "model_name", "choices": ["4x-UltraSharp.pth"]}]},
        "LoadImage": {"inputs": [{"name": "image", "choices": [uploaded]}]},
    }

    class FakeGraph:
        def node(self, name):
            return name if name in catalog else None

        def morphism_to_dict(self, node):
            return catalog[node]

    monkeypatch.setattr("comfy_cli.command.nodes._get_graph", lambda *args, **kwargs: FakeGraph())
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda where=None: object())
    monkeypatch.setattr("comfy_cli.command.build._inventory_cloud_model_size", lambda target, name: 66_961_958)

    result, envelope = _invoke(
        str(workflow), "--cloud", "--models-root", str(tmp_path / "models"), "--input-root", str(inputs)
    )

    assert result.exit_code == 0, result.stdout
    data = envelope["data"]
    assert data["cloud"]["models"][0]["weights"] == "different_size"
    assert data["cloud"]["verdict"] == "needs_build_or_lossy_conversion"
    assert data["inputs"][0]["cloud_name"] == uploaded
    assert data["inputs"][0]["on_cloud"] is True
    jsonschema.validate(data, json.loads(SCHEMA.read_text(encoding="utf-8")))


@pytest.mark.parametrize("payload", [{}, {"nodes": "not-a-list"}, {"nodes": [None]}, {"not": "a workflow"}])
def test_inventory_rejects_objects_that_are_not_workflows(tmp_path: Path, payload):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result, envelope = _invoke(str(path))

    assert result.exit_code == 1
    assert envelope["error"]["code"] == "workflow_invalid_json"


def test_api_inventory_lists_the_deployment_scanners_complete_upload_plan(tmp_path: Path, monkeypatch):
    from comfy_cli.command.deploy_workflow import scan_workflow_inputs

    private = tmp_path / "private-face.png"
    private.write_bytes(b"PRIVATE")
    graph = {"1": {"class_type": "FaceReference", "inputs": {"reference": private.name}}}
    workflow = tmp_path / "api.json"
    workflow.write_text(json.dumps(graph), encoding="utf-8")
    monkeypatch.setattr("comfy_cli.command.deploy_workflow.resolve_asset_roots", lambda *a, **kw: (tmp_path,))

    result, envelope = _invoke(str(workflow), "--input-root", str(tmp_path))

    assert result.exit_code == 0
    expected = scan_workflow_inputs(graph, asset_roots=(tmp_path,))
    assert envelope["data"]["deployment_assets"] == [
        {"local_path": str(asset.local_path), "file_path": asset.file_path, "bytes": asset.local_path.stat().st_size}
        for asset in expected.assets
    ]
    assert envelope["data"]["deployment_asset_roots"] == [str(tmp_path)]


@pytest.mark.parametrize("root_option", ["--models-root", "--input-root"])
def test_inventory_reports_nonexistent_explicit_roots(tmp_path, root_option):
    workflow = _write_workflow(tmp_path)
    missing = tmp_path / "does-not-exist"
    result, envelope = _invoke(str(workflow), root_option, str(missing))
    assert result.exit_code == 0
    assert any("root_not_found" in warning and str(missing) in warning for warning in envelope["data"]["warnings"])


def test_desktop_inventory_uses_only_running_instance_models_and_input(tmp_path, monkeypatch):
    from comfy_cli import desktop_install

    app = tmp_path / "Comfy Desktop"
    app.mkdir()
    records = []
    for name in ("running", "other"):
        code = tmp_path / name / "ComfyUI"
        code.mkdir(parents=True)
        (code / "main.py").write_text("")
        (code / "models").mkdir()
        (code / "input").mkdir()
        records.append({"id": name, "installPath": str(code.parent)})
    (app / "installations.json").write_text(json.dumps(records))
    monkeypatch.setattr(desktop_install, "desktop_app_data_dir", lambda: app)

    def fetch(url):
        return (
            {"system": {"argv": [str(tmp_path / "running" / "ComfyUI" / "main.py")]}}
            if url.endswith("system_stats")
            else {}
        )

    monkeypatch.setattr("comfy_cli.command.build._inventory_fetch_json", fetch)
    result, envelope = _invoke(str(_write_workflow(tmp_path)), "--desktop", "--server", "127.0.0.1:8000")
    assert result.exit_code == 0
    data = envelope["data"]
    assert data["models_roots"] == [str(tmp_path / "running" / "ComfyUI" / "models")]
    assert data["desktop"]["install_id"] == "running"
    assert data["input_roots"] == [str(tmp_path / "running" / "ComfyUI" / "input")]


def test_desktop_inventory_settings_only_modern_registry_is_supported(tmp_path, monkeypatch):
    from comfy_cli import desktop_install

    app = tmp_path / "comfyui-desktop-2"
    app.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (app / "settings.json").write_text(json.dumps({"modelsDirs": [str(models)]}))
    monkeypatch.setattr(desktop_install, "desktop_app_data_dir", lambda: app)
    roots, warning = __import__("comfy_cli.command.build", fromlist=["_desktop_model_roots"])._desktop_model_roots()
    assert roots == [str(models)]
    assert warning is None


@pytest.mark.parametrize("value", ["/not-a-model-root", {"/": "models"}, True, False, 1, 0, {}])
def test_desktop_inventory_settings_only_rejects_non_list_model_roots(tmp_path, monkeypatch, value):
    from comfy_cli import desktop_install
    from comfy_cli.command.build import _desktop_model_roots

    app = tmp_path / "desktop-settings"
    app.mkdir()
    (app / "settings.json").write_text(json.dumps({"modelsDirs": value}))
    monkeypatch.setattr(desktop_install, "app_data_dirs", lambda: (app,))

    roots, warning = _desktop_model_roots()

    assert roots == []
    assert "modelsDirs must be a list" in warning
    assert "--models-root" in warning


def test_desktop_inventory_settings_only_ignores_empty_and_non_string_roots(tmp_path, monkeypatch):
    from comfy_cli import desktop_install
    from comfy_cli.command.build import _desktop_model_roots

    app = tmp_path / "desktop-settings"
    app.mkdir()
    models = tmp_path / "models"
    models.mkdir()
    (app / "settings.json").write_text(json.dumps({"modelsDirs": ["", None, 7, str(models)]}))
    monkeypatch.setattr(desktop_install, "app_data_dirs", lambda: (app,))

    roots, warning = _desktop_model_roots()

    assert roots == [str(models)]
    assert warning is None


def test_desktop_inventory_plans_only_referenced_file_from_effective_output_root(tmp_path, monkeypatch):
    from comfy_cli import desktop_install

    code = tmp_path / "ComfyUI"
    for directory in (code, code / "models", code / "input"):
        directory.mkdir(exist_ok=True)
    (code / "main.py").write_text("")
    app = tmp_path / "Comfy Desktop"
    app.mkdir()
    (app / "installations.json").write_text(json.dumps([{"id": "live", "installPath": str(code)}]))
    output = tmp_path / "actual-output"
    output.mkdir()
    required = output / "generated.png"
    required.write_bytes(b"one required synthetic input")
    (output / "unrelated-private.png").write_bytes(b"do not plan this file")
    workflow = tmp_path / "api.json"
    workflow.write_text(json.dumps({"1": {"class_type": "FaceReference", "inputs": {"reference": "generated.png"}}}))
    monkeypatch.setattr(desktop_install, "desktop_app_data_dir", lambda: app)

    def fetch(url):
        return (
            {
                "system": {
                    "argv": [str(code / "main.py"), "--output-directory", str(output), "--api-token", "secret-not-path"]
                }
            }
            if url.endswith("system_stats")
            else {}
        )

    monkeypatch.setattr("comfy_cli.command.build._inventory_fetch_json", fetch)
    result, envelope = _invoke(str(workflow), "--desktop", "--server", "127.0.0.1:8000")
    assert result.exit_code == 0, result.output
    data = envelope["data"]
    assert data["desktop"]["output_dir"] == str(output)
    assert str(output) in data["deployment_asset_roots"]
    assert [entry["local_path"] for entry in data["deployment_assets"]] == [str(required)]
    assert data["local_server"]["launch_paths"]["output_directory"] == [str(output)]
    assert "secret-not-path" not in result.output
