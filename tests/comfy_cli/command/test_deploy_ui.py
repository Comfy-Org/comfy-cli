from __future__ import annotations

import importlib
import json
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.comfy_client import Client, HTTPError
from comfy_cli.target import Target

DEPLOYMENT_ID = "dep-12345678-1234-1234-1234-123456789abc"


@pytest.fixture
def editor(monkeypatch):
    ui = importlib.import_module("comfy_cli.command.deploy_ui")
    deployment = {"id": DEPLOYMENT_ID, "status": "ready", "releaseId": "release-1"}
    listing = {"items": [{"deployment_id": DEPLOYMENT_ID, "release_id": "release-1", "status": "ready"}]}
    state = {"workspace": {"id": "ws-1"}, "listing": listing, "deployment": deployment, "failure": None}
    calls = []

    class Deploy:
        def get_deployment(self, deployment_id):
            calls.append(("deploy", deployment_id))
            return state["deployment"]

        def list_all_deployments(self):
            pytest.fail("explicit id must not list deployments")

        def get_deployment_logs(self, deployment_id):
            pytest.fail("UI discovery must not fetch logs")

        def get_deployment_events(self, deployment_id):
            pytest.fail("UI discovery must not fetch events")

    monkeypatch.setattr("comfy_cli.command.deploy_read._command_clients", lambda: (object(), Deploy()))
    target = Target(kind="cloud", base_url="https://cloud.test/api", path_prefix="/api", auth_token="secret-token")
    monkeypatch.setattr(ui, "resolve_target", lambda **kwargs: state.get("target", target))

    def request(self, method, parts, **kwargs):
        assert method == "GET"
        calls.append((method, parts))
        if state["failure"] is not None:
            raise state["failure"]
        return state["workspace"] if parts == ("workspaces", "current") else state["listing"]

    monkeypatch.setattr(Client, "_request", request)
    return state, calls


def invoke():
    result = CliRunner().invoke(app, ["--json", "deploy", "ui", "--deployment", DEPLOYMENT_ID])
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return result, json.loads(lines[-1])


def test_deploy_ui_returns_verified_workspace_target_release_link(editor):
    state, calls = editor
    result, envelope = invoke()
    assert result.exit_code == 0, result.output
    data = envelope["data"]
    assert data == {
        "workspaceId": "ws-1",
        "deploymentId": DEPLOYMENT_ID,
        "releaseId": "release-1",
        "url": f"https://cloud.test/?workspace=ws-1&deployment={DEPLOYMENT_ID}",
    }
    assert envelope["changed"] is False
    schema = Path(__file__).resolve().parents[3] / "comfy_cli/schemas/deploy_ui.json"
    jsonschema.validate(data, json.loads(schema.read_text()))
    assert calls == [
        ("deploy", DEPLOYMENT_ID),
        ("GET", ("workspaces", "current")),
        ("GET", ("workspaces", "ws-1", "deployments")),
    ]
    assert "secret-token" not in result.output


@pytest.mark.parametrize("status", ["stopped", "provisioning", "unhealthy", "failed"])
def test_deploy_ui_refuses_nonready_control_record(editor, status):
    state, calls = editor
    state["deployment"]["status"] = status
    result, envelope = invoke()
    assert result.exit_code == 1
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "deploy_ui_unavailable"
    assert calls == [("deploy", DEPLOYMENT_ID)]


@pytest.mark.parametrize(
    "listing",
    [
        None,
        {},
        {"items": []},
        {"items": [None]},
        {"items": [{"deployment_id": DEPLOYMENT_ID, "release_id": "release-other", "status": "ready"}]},
        {"items": [{"deployment_id": DEPLOYMENT_ID, "release_id": "release-1", "status": "stopped"}]},
    ],
)
def test_deploy_ui_refuses_unverified_listing(editor, listing):
    state, _ = editor
    state["listing"] = listing
    result, envelope = invoke()
    assert result.exit_code == 1
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "deploy_ui_unavailable"


@pytest.mark.parametrize("status", [401, 403, 404, 503])
def test_deploy_ui_reports_cloud_refusal_without_guessing_a_url(editor, status):
    state, _ = editor
    state["failure"] = HTTPError(status, "refused", "Bearer secret-token")
    result, envelope = invoke()
    assert result.exit_code == 1
    assert envelope["data"] is None
    assert envelope["error"]["code"] == "deploy_ui_unavailable"
    assert envelope["error"]["details"]["status"] == status
    assert "secret-token" not in result.output


def test_deploy_ui_requires_session_instead_of_api_key(editor):
    state, calls = editor
    state["target"] = Target(kind="cloud", base_url="https://cloud.test", api_key="comfyui-secret")
    result, envelope = invoke()
    assert result.exit_code == 1
    assert envelope["error"]["code"] == "deploy_ui_unavailable"
    assert "comfy cloud login" in envelope["error"]["hint"]
    assert not any(method == "GET" for method, _ in calls)


def test_deploy_ui_refuses_mismatching_control_id(editor):
    state, _ = editor
    state["deployment"]["id"] = "dep-other"
    result, envelope = invoke()
    assert result.exit_code == 1
    assert envelope["data"] is None


def test_deploy_ui_refuses_workspace_path_injection(editor):
    state, _ = editor
    state["workspace"] = {"id": "ws-1/../other"}
    result, envelope = invoke()
    assert result.exit_code == 1
    assert envelope["data"] is None


def test_deploy_ui_is_registered_as_a_read_only_command():
    result = CliRunner().invoke(app, ["deploy", "ui", "--help"])
    assert result.exit_code == 0, result.output
    assert "read-only" in result.output.lower()


def test_editor_reads_use_authenticated_bounded_cloud_transport(monkeypatch):
    import io

    from comfy_cli import comfy_client

    requests = []

    class Response(io.BytesIO):
        status = 200

    def open_request(request, timeout):
        requests.append(request)
        return Response(b'{"id":"ws-1","items":[]}')

    monkeypatch.setattr(comfy_client._OPENER, "open", open_request)
    client = Client(Target(kind="cloud", base_url="https://cloud.test", path_prefix="/api", auth_token="test-token"))
    assert client.get_current_workspace()["id"] == "ws-1"
    assert client.list_workspace_deployments("ws/one")["items"] == []
    assert [request.full_url for request in requests] == [
        "https://cloud.test/api/workspaces/current",
        "https://cloud.test/api/workspaces/ws%2Fone/deployments",
    ]
    assert all(request.get_method() == "GET" for request in requests)
    assert all(request.get_header("Authorization") == "Bearer test-token" for request in requests)


def test_deploy_ui_reports_recognized_backend_session_refusal(editor):
    state, _ = editor
    reason = "listing deployments needs a signed-in session, not an API key"
    state["failure"] = HTTPError(403, "refused", json.dumps({"code": "FORBIDDEN", "message": reason}))
    result, envelope = invoke()
    assert result.exit_code == 1
    assert reason in envelope["error"]["message"]
