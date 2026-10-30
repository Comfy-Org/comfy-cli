"""Stable file UUID verification cannot be inferred from workflow-record data."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
from typer.testing import CliRunner

from comfy_cli.command import workflow as cmd
from comfy_cli.command import workflow_ui as ui
from comfy_cli.target import Target

FILE_ID = "978e5f58-49dc-4a81-8235-a0b91c204a1c"
WORKSPACE = "w-12345678-abcd-4abc-abcd-123456789abc"
GRAPH = {"version": 0.4, "nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
SNAPSHOT_PATH = f"/api/userdata-files/{FILE_ID}"
SNAPSHOT = {
    "id": FILE_ID,
    "path": "workflows/sub/My saved workflow.json",
    "content": json.dumps(GRAPH),
    "modified": 1793320000000,
    "size": len(json.dumps(GRAPH).encode()),
}


@pytest.fixture
def saved_server(monkeypatch):
    from comfy_cli.output.renderer import reset_renderer_for_testing

    reset_renderer_for_testing()
    target = Target(kind="cloud", base_url="https://cloud.test", path_prefix="/api", auth_token="oauth-test")
    monkeypatch.setattr(ui, "resolve_target", lambda **kwargs: target)
    bodies = {
        "/api/workspaces/current": {"id": WORKSPACE},
        "/api/features": {"private_saved_file_links_enabled": True},
        SNAPSHOT_PATH: copy.deepcopy(SNAPSHOT),
    }
    calls = []

    def request(url, request_target, **kwargs):
        assert kwargs.get("method", "GET") == "GET"
        assert request_target is target
        calls.append(urlsplit(url).path)
        value = bodies[urlsplit(url).path]
        return value if isinstance(value, tuple) else (200, copy.deepcopy(value))

    monkeypatch.setattr(ui, "request_json", request)
    yield bodies, calls
    reset_renderer_for_testing()


def test_one_uuid_snapshot_without_name_or_content_association(saved_server):
    bodies, calls = saved_server
    bodies[f"/api/workflows/{FILE_ID}"] = {"id": FILE_ID, "name": "sub/My saved workflow"}
    result = ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")
    assert result["savedFileId"] == FILE_ID
    assert "workflowId" not in result
    assert result["filename"] == SNAPSHOT["path"]
    assert parse_qs(urlsplit(result["url"]).query) == {
        "savedFile": [FILE_ID],
        "workspace": [WORKSPACE],
        "environment": ["standard"],
    }
    assert calls == ["/api/workspaces/current", "/api/features", SNAPSHOT_PATH]
    assert "content" not in result and "workflow_json" not in result


@pytest.mark.parametrize("flag", [None, False, 1, "true", {}, []])
def test_only_literal_new_capability_can_publish(saved_server, flag):
    bodies, calls = saved_server
    bodies["/api/features"] = {"private_workflow_links_enabled": True}
    if flag is not None:
        bodies["/api/features"]["private_saved_file_links_enabled"] = flag
    with pytest.raises(ui.WorkflowUiUnavailable) as failure:
        ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")
    assert failure.value.reason == "capability_unavailable"
    assert SNAPSHOT_PATH not in calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", None),
        ("id", "other"),
        ("id", FILE_ID.upper()),
        ("path", None),
        ("path", "workflows/../file.json"),
        ("path", "other/file.json"),
        ("path", "workflows/other.txt"),
        ("path", "workflows/sub//file.json"),
        ("modified", True),
        ("modified", -1),
        ("size", True),
        ("size", -1),
        ("size", 999),
        ("content", GRAPH),
        ("content", "not JSON"),
        ("content", "NaN"),
    ],
)
def test_snapshot_fields_are_validated_without_fallback(saved_server, field, value):
    bodies, calls = saved_server
    bodies[SNAPSHOT_PATH][field] = value
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")
    assert calls[-1] == SNAPSHOT_PATH
    assert not any("/workflows/" in call or "/userdata/" in call for call in calls)


@pytest.mark.parametrize("field", ["id", "path", "content", "modified", "size"])
def test_missing_snapshot_fields_are_refused(saved_server, field):
    bodies, _ = saved_server
    bodies[SNAPSHOT_PATH].pop(field)
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize("status", [401, 403, 404, 412, 422, 500])
def test_endpoint_errors_never_fall_back_to_same_name_equal_content(saved_server, status):
    bodies, calls = saved_server
    bodies[f"/api/workflows/{FILE_ID}"] = {"id": FILE_ID, "name": "sub/My saved workflow"}
    bodies[f"/api/workflows/{FILE_ID}/content"] = {"workflow_json": GRAPH}
    bodies["/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json"] = GRAPH
    bodies[SNAPSHOT_PATH] = (status, {"SECRET": "private prompt"})
    with pytest.raises(ui.WorkflowUiUnavailable) as failure:
        ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")
    assert failure.value.status == status
    assert "SECRET" not in str(failure.value)
    assert calls[-1] == SNAPSHOT_PATH


@pytest.mark.parametrize("legacy,mixed", [(True, False), (True, True), (False, False)])
def test_legacy_record_missing_or_mixed_invocation_refuses_before_network(saved_server, legacy, mixed):
    _, calls = saved_server
    args = ["ui"] + ([FILE_ID] if legacy else []) + (["--saved-file", FILE_ID] if mixed else [])
    result = CliRunner().invoke(cmd.app, args + ["--workspace", WORKSPACE, "--environment", "standard"])
    assert result.exit_code == 1, result.output
    assert "saved_file" in result.output or "saved-file" in result.output
    assert calls == []


@pytest.mark.parametrize("mode", ["json", "ndjson"])
def test_explicit_saved_file_cli_machine_output(saved_server, mode):
    from comfy_cli.caller import Caller
    from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

    renderer = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    renderer.mode = OutputMode.JSON if mode == "json" else OutputMode.NDJSON
    set_renderer(renderer)
    result = CliRunner().invoke(
        cmd.app, ["ui", "--saved-file", FILE_ID, "--workspace", WORKSPACE, "--environment", "standard"]
    )
    assert result.exit_code == 0, result.output
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is True
    assert envelope["data"]["savedFileId"] == FILE_ID
    assert "savedFile=" in envelope["data"]["url"]


def test_renamed_unicode_path_comes_from_same_snapshot(saved_server):
    bodies, calls = saved_server
    bodies[SNAPSHOT_PATH]["path"] = "workflows/新しい folder/雪 workflow.json"
    result = ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")
    assert result["filename"] == bodies[SNAPSHOT_PATH]["path"]
    assert calls[-1] == SNAPSHOT_PATH


# Replay the actual consumer corpus through the new raw-text snapshot route too.
# This retains both canvas versions and legacy Unicode/space model-URL behavior.

_CORPUS = json.loads((Path(__file__).parents[1] / "fixtures/cloud_workflow_consumer/corpus.json").read_text())


@pytest.mark.parametrize("case", _CORPUS["cases"], ids=lambda case: case["id"])
def test_saved_snapshot_matches_existing_consumer_compatibility(saved_server, case):
    bodies, _ = saved_server
    text = json.dumps(case["input"])
    bodies[SNAPSHOT_PATH].update(content=text, size=len(text.encode()))
    if case["accepted"]:
        assert ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")["verified"] is True
    else:
        with pytest.raises(ui.WorkflowUiUnavailable):
            ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")


def test_original_utf8_text_size_and_opaque_unicode_are_accepted(saved_server):
    bodies, _ = saved_server
    graph = {**GRAPH, "extra": {"title": "雪の workflow"}}
    text = " \n" + json.dumps(graph, ensure_ascii=False) + "\n"
    bodies[SNAPSHOT_PATH].update(content=text, size=len(text.encode("utf-8")))
    assert ui.verify_workflow_link(FILE_ID, workspace=WORKSPACE, environment="standard")["verified"] is True


@pytest.mark.parametrize("mode", ["json", "ndjson"])
def test_legacy_invocation_machine_error_has_regeneration_hint(saved_server, mode):
    from comfy_cli.caller import Caller
    from comfy_cli.output.renderer import OutputMode, Renderer, set_renderer

    renderer = Renderer.resolve(
        is_stdout_tty=False, env={}, caller=Caller(kind="user", agentic=False, source_env=None), json_flag=True
    )
    renderer.mode = OutputMode.JSON if mode == "json" else OutputMode.NDJSON
    set_renderer(renderer)
    result = CliRunner().invoke(cmd.app, ["ui", FILE_ID, "--workspace", WORKSPACE, "--environment", "standard"])
    assert result.exit_code == 1
    envelope = json.loads(result.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is False
    assert envelope["error"]["details"]["reason"] == "legacy_identity_unsupported"
    assert "--saved-file" in envelope["error"]["message"]
    assert "Workflows" in envelope["error"]["hint"]
