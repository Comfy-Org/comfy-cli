"""`comfy workflow check <workflow> --where cloud`: the command around comfy_cli.workflow_coverage.

Cloud's node list and model list are mocked; the comparing rules are covered in
tests/comfy_cli/test_workflow_coverage.py.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Any

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.command import workflow as workflow_cmd
from comfy_cli.output.renderer import OutputMode, Renderer, reset_renderer_for_testing, set_renderer

_SCHEMA = json.loads(
    (Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas" / "workflow_check.json").read_text()
)

OBJECT_INFO = {
    "CheckpointLoaderSimple": {"input": {"required": {"ckpt_name": [["sd15.safetensors"], {}]}}},
    "LoraLoader": {"input": {"required": {"lora_name": [["styles/ink.safetensors"], {}]}}},
    "KSampler": {"input": {"required": {"seed": ["INT", {}]}}},
}
SD15 = b"sd15 weights"


@pytest.fixture(autouse=True)
def reset_singleton():
    reset_renderer_for_testing()
    yield
    reset_renderer_for_testing()


def _force_json_renderer():
    r = Renderer.resolve(
        is_stdout_tty=False,
        env={},
        caller=Caller(kind="user", agentic=False, source_env=None),
        json_flag=True,
    )
    r.mode = OutputMode.JSON
    set_renderer(r)


def _run(args: list[str], capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    _force_json_renderer()
    result = CliRunner().invoke(workflow_cmd.app, ["check", *args], standalone_mode=False)
    captured = capsys.readouterr().out or result.stdout or ""
    assert captured.strip(), f"no envelope on stdout (rc={result.exit_code}, exc={result.exception})"
    return json.loads(captured.strip().splitlines()[-1])


def _cloud_target(**creds):
    from comfy_cli.target import Target

    return Target(kind="cloud", base_url="https://cloud.example.com", path_prefix="/api", **creds)


@pytest.fixture
def cloud(monkeypatch: pytest.MonkeyPatch):
    """A signed-in Cloud: OBJECT_INFO as its node list, ``state["assets"]`` as its model list."""
    state: dict[str, Any] = {"assets": [], "requests": []}
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: _cloud_target(api_key="k"))
    monkeypatch.setattr("comfy_cli.cql.loader.resilient_load_object_info", lambda **kw: OBJECT_INFO)

    def fake_http(url, target, **kw):
        query = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
        state["requests"].append(query)
        public = query["include_public"] == ["true"]
        needle = query["name_contains"][0]
        rows = [a for a in state["assets"] if needle in a["name"] and (public or not a.get("public"))]
        return 200, {"assets": rows, "has_more": False}

    monkeypatch.setattr("comfy_cli.command.cloud_http.http_request", fake_http)
    return state


def _write(tmp_path: Path, data: Any, name: str = "wf.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


def _ui(*nodes):
    return {"nodes": [{"id": i, "type": t, "mode": 0, "widgets_values": v} for i, t, v in nodes], "links": []}


def _ok(env: dict[str, Any]) -> dict[str, Any]:
    assert env["ok"] is True, env
    assert env["command"] == "workflow check"
    assert env["where"] == "cloud"
    jsonschema.validate(env["data"], _SCHEMA)
    return env["data"]


# --- the four verdicts -----------------------------------------------------


def test_needs_check_when_models_are_not_proven(cloud, tmp_path, capsys):
    wf = _write(tmp_path, _ui((1, "CheckpointLoaderSimple", ["sd15.safetensors"]), (2, "KSampler", [1])))
    data = _ok(_run([str(wf), "--where", "cloud"], capsys))
    assert data["verdict"] == "covered_needs_check"
    assert data["blocking"] == []
    [model] = data["models"]
    assert model["status"] == "present" and model["source"] == "cloud" and model["identity"] == "unverified"
    assert model["folder"] == "checkpoints" and model["filename"] == "sd15.safetensors"
    assert {n["class_type"]: n["on_cloud"] for n in data["nodes"]} == {"CheckpointLoaderSimple": True, "KSampler": True}
    assert data["out"] is None and data["remaps_applied"] == 0


def test_covered_when_a_local_copy_hashes_the_same(cloud, tmp_path, capsys):
    from comfy_cli.hashing import blake3_file

    models = tmp_path / "models" / "checkpoints"
    models.mkdir(parents=True)
    (models / "sd15.safetensors").write_bytes(SD15)
    digest = blake3_file(models / "sd15.safetensors")
    cloud["assets"] = [
        {"name": "sd15.safetensors", "hash": digest, "size": len(SD15), "tags": ["models"], "public": True}
    ]
    wf = _write(tmp_path, _ui((1, "CheckpointLoaderSimple", ["sd15.safetensors"])))
    data = _ok(_run([str(wf), "--where", "cloud", "--models-dir", str(tmp_path / "models")], capsys))
    assert data["verdict"] == "covered"
    assert data["models"][0]["identity"] == "same"
    assert data["models"][0]["local_paths"] == [str(models / "sd15.safetensors")]


def test_a_different_local_file_blocks(cloud, tmp_path, capsys):
    (tmp_path / "sd15.safetensors").write_bytes(SD15)
    cloud["assets"] = [
        {"name": "sd15.safetensors", "hash": "f" * 64, "size": len(SD15), "tags": ["models"], "public": True}
    ]
    wf = _write(tmp_path, _ui((1, "CheckpointLoaderSimple", ["sd15.safetensors"])))
    data = _ok(_run([str(wf), "--where", "cloud", "--models-dir", str(tmp_path)], capsys))
    assert data["verdict"] == "needs_build"
    assert [(b["kind"], b["reason"]) for b in data["blocking"]] == [("model", "different_file")]


def test_remaps_and_apply_remaps_writes_a_copy(cloud, tmp_path, capsys):
    original = _ui((1, "LoraLoader", ["ink.safetensors", 1.0, 1.0]))
    wf = _write(tmp_path, original)
    before = wf.read_bytes()
    out = tmp_path / "out" / "remapped.json"
    data = _ok(_run([str(wf), "--where", "cloud", "--apply-remaps", "--out", str(out)], capsys))
    assert data["verdict"] == "covered_with_remaps"
    [model] = data["models"]
    assert (model["status"], model["value"], model["cloud_value"]) == (
        "remap",
        "ink.safetensors",
        "styles/ink.safetensors",
    )
    assert data["out"] == str(out.resolve()) and data["remaps_applied"] == 1
    assert wf.read_bytes() == before
    assert json.loads(out.read_text())["nodes"][0]["widgets_values"][0] == "styles/ink.safetensors"


def test_needs_build_lists_missing_nodes_and_models_with_their_folder(cloud, tmp_path, capsys):
    (tmp_path / "local" / "loras").mkdir(parents=True)
    (tmp_path / "local" / "loras" / "mine.safetensors").write_bytes(b"x")
    wf = _write(
        tmp_path,
        _ui((1, "LoraLoader", ["mine.safetensors", 1, 1]), (2, "MyCustomNode", [])),
    )
    data = _ok(_run([str(wf), "--where", "cloud", "--models-dir", str(tmp_path / "local")], capsys))
    assert data["verdict"] == "needs_build"
    node, model = data["blocking"]
    assert node == {"kind": "node", "class_type": "MyCustomNode", "node_ids": ["2"], "reason": "not_on_cloud"}
    assert model["reason"] == "not_on_cloud" and model["folder"] == "loras"
    assert model["local_paths"] == [str(tmp_path / "local" / "loras" / "mine.safetensors")]
    assert {n["class_type"]: n["on_cloud"] for n in data["nodes"]}["MyCustomNode"] is False


def test_the_persons_upload_counts_and_only_private_assets_are_asked(cloud, tmp_path, capsys):
    cloud["assets"] = [{"name": "mine.safetensors", "hash": "a" * 64, "size": 1, "tags": ["models", "loras"]}]
    wf = _write(tmp_path, {"1": {"class_type": "LoraLoader", "inputs": {"lora_name": "mine.safetensors"}}})
    data = _ok(_run([str(wf), "--where", "cloud"], capsys))
    assert data["format"] == "api"
    assert data["verdict"] == "covered_needs_check"
    assert data["models"][0]["source"] == "yours" and data["models"][0]["widget"] == "lora_name"
    assert [q["include_public"] for q in cloud["requests"]] == [["false"]]


def test_muted_nodes_are_skipped(cloud, tmp_path, capsys):
    wf = _write(tmp_path, {"nodes": [{"id": 1, "type": "MyCustomNode", "mode": 4, "widgets_values": []}]})
    data = _ok(_run([str(wf), "--where", "cloud"], capsys))
    assert data["verdict"] == "covered"
    assert data["skipped"] == [{"node_id": "1", "class_type": "MyCustomNode", "why": "bypassed"}]


# --- refusals ----------------------------------------------------------------


def _err(env: dict[str, Any], code: str) -> dict[str, Any]:
    assert env["ok"] is False, env
    assert env["error"]["code"] == code, env["error"]
    return env["error"]


def test_out_without_apply_remaps_and_the_reverse(cloud, tmp_path, capsys):
    wf = _write(tmp_path, _ui())
    _err(_run([str(wf), "--where", "cloud", "--out", str(tmp_path / "o.json")], capsys), "missing_argument")
    _err(_run([str(wf), "--where", "cloud", "--apply-remaps"], capsys), "missing_argument")


def test_out_may_not_name_the_input(cloud, tmp_path, capsys):
    wf = _write(tmp_path, _ui((1, "LoraLoader", ["ink.safetensors", 1, 1])))
    before = wf.read_bytes()
    _err(_run([str(wf), "--where", "cloud", "--apply-remaps", "--out", str(wf)], capsys), "workflow_out_is_input")
    assert wf.read_bytes() == before


@pytest.mark.parametrize(
    ("content", "code"),
    [("{not json", "workflow_invalid_json"), ('{"hello": "world"}', "workflow_format_unknown")],
)
def test_bad_files(cloud, tmp_path, capsys, content, code):
    wf = tmp_path / "wf.json"
    wf.write_text(content)
    _err(_run([str(wf), "--where", "cloud"], capsys), code)


def test_missing_file(cloud, tmp_path, capsys):
    _err(_run([str(tmp_path / "nope.json"), "--where", "cloud"], capsys), "workflow_not_found")


def test_local_target_is_refused(monkeypatch, tmp_path, capsys):
    from comfy_cli.target import Target

    local = Target(kind="local", base_url="http://127.0.0.1:8188", host="127.0.0.1", port=8188)
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: local)
    _err(_run([str(_write(tmp_path, _ui())), "--where", "local"], capsys), "cloud_only_command")


def test_signed_out_is_refused(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: _cloud_target())
    _err(_run([str(_write(tmp_path, _ui())), "--where", "cloud"], capsys), "auth_not_signed_in")


def test_node_list_failure(monkeypatch, tmp_path, capsys):
    from comfy_cli.cql.engine import LoadError

    def boom(**kw):
        raise LoadError("down")

    monkeypatch.setattr("comfy_cli.target.resolve_target", lambda **kw: _cloud_target(api_key="k"))
    monkeypatch.setattr("comfy_cli.cql.loader.resilient_load_object_info", boom)
    _err(_run([str(_write(tmp_path, _ui())), "--where", "cloud"], capsys), "cql_no_graph")


def test_model_list_failure_is_an_error_not_a_missing_model(cloud, monkeypatch, tmp_path, capsys):
    def unauthorized(url, target, **kw):
        raise urllib.error.HTTPError(url, 401, "Unauthorized", {}, None)

    monkeypatch.setattr("comfy_cli.command.cloud_http.http_request", unauthorized)
    wf = _write(tmp_path, _ui((1, "LoraLoader", ["mine.safetensors", 1, 1])))
    error = _err(_run([str(wf), "--where", "cloud"], capsys), "cloud_unauthorized")
    assert error["details"]["operation"] == "model lookup"
