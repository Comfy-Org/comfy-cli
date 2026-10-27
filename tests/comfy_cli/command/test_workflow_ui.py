"""Read-only capability-gated links preserve normal private workflow identity."""

from __future__ import annotations

import copy
import json
import urllib.error
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
from click import unstyle
from typer.testing import CliRunner

from comfy_cli.command import workflow as cmd
from comfy_cli.target import Target

WID = "c327cae8-2acf-4bfa-bb88-97948a3c008b"
WORKSPACE = "w-12345678-abcd-4abc-abcd-123456789abc"
DEPLOYMENT = "dep-12345678-abcd-4abc-abcd-123456789abc"
GRAPH = {"version": 0.4, "nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
META = {
    "id": WID,
    "name": "sub/My saved workflow",
    "latest_version": 1,
    "created_by": "user-1",
    "created_at": "2026-10-24T00:00:00Z",
    "updated_at": "2026-10-24T00:00:00Z",
}


def test_cli_ui_command_exists():
    result = CliRunner().invoke(cmd.app, ["ui", "--help"])
    assert result.exit_code == 0, result.output
    help_text = unstyle(result.output)
    assert "--workspace" in help_text
    assert "--environment" in help_text
    assert "--deployment" in help_text


@pytest.fixture
def server(monkeypatch):
    from comfy_cli.command import workflow_ui as ui

    target = Target(kind="cloud", base_url="https://cloud.test", path_prefix="/api", auth_token="oauth-test")
    monkeypatch.setattr(ui, "resolve_target", lambda **kwargs: target)
    bodies = {
        "/api/workspaces/current": {"id": WORKSPACE, "name": "Test workspace"},
        "/api/features": {"private_workflow_links_enabled": True},
        f"/api/workflows/{WID}": copy.deepcopy(META),
        f"/api/workflows/{WID}/content": {
            "id": "version-1",
            "version": 1,
            "created_by": "user-1",
            "created_at": META["created_at"],
            "workflow_json": copy.deepcopy(GRAPH),
        },
        "/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json": copy.deepcopy(GRAPH),
    }
    calls = []

    def request(url, request_target, **kwargs):
        assert kwargs.get("method", "GET") == "GET"
        assert kwargs["max_bytes"] > 0
        assert request_target is target
        calls.append(urlsplit(url).path)
        value = bodies[urlsplit(url).path]
        if isinstance(value, Exception):
            raise value
        return 200, copy.deepcopy(value)

    monkeypatch.setattr(ui, "request_json", request)
    return ui, bodies, calls, target


def test_standard_link_verifies_exact_record_content_and_userdata(server):
    ui, _, calls, _ = server
    result = ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    assert result["workflowId"] == WID
    assert result["filename"] == "workflows/sub/My saved workflow.json"
    assert result["verified"] is True
    query = parse_qs(urlsplit(result["url"]).query)
    assert query == {"workflow": [WID], "workspace": [WORKSPACE], "environment": ["standard"]}
    assert "/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json" in calls
    assert not urlsplit(result["url"]).fragment
    assert "workflow_json" not in result


@pytest.mark.parametrize("flag", [None, False, 1, "true", {}, []])
def test_missing_or_nonboolean_capability_cannot_produce_url(server, flag):
    ui, bodies, calls, _ = server
    bodies["/api/features"] = {} if flag is None else {"private_workflow_links_enabled": flag}
    with pytest.raises(ui.WorkflowUiUnavailable) as failure:
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    assert failure.value.reason == "capability_unavailable"
    assert f"/api/workflows/{WID}" not in calls


@pytest.mark.parametrize("workspace", ["other-workspace", "", "../x", "bad&deployment=other"])
def test_wrong_or_unsafe_workspace_is_refused(server, workspace):
    ui, _, _, _ = server
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=workspace, environment="standard")


@pytest.mark.parametrize(
    "name",
    [
        None,
        "",
        "../private",
        "sub/../private",
        "/rooted",
        "sub//empty",
        "sub/./file",
        "sub\\file",
        "bad\nfile",
        "~hidden",
        "sub/.. /file",
    ],
)
def test_missing_or_unsafe_record_name_never_becomes_userdata_path(server, name):
    ui, bodies, calls, _ = server
    bodies[f"/api/workflows/{WID}"]["name"] = name
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    assert not any("/userdata/" in path for path in calls)


@pytest.mark.parametrize("bad_id", ["not-a-uuid", "../secret", WID + "?x=1", "", "a" * 32])
def test_bad_workflow_identity_rejected_before_network(server, bad_id):
    ui, _, calls, _ = server
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(bad_id, workspace=WORKSPACE, environment="standard")
    assert calls == []


def test_metadata_other_id_is_refused(server):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}"]["id"] = "other-id"
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize("field", ["created_by", "created_at", "updated_at", "latest_version"])
def test_malformed_metadata_is_refused(server, field):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}"].pop(field)
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


def test_stale_version_or_userdata_content_is_refused(server):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}/content"]["version"] = 2
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    bodies[f"/api/workflows/{WID}/content"]["version"] = 1
    bodies["/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json"]["nodes"] = [{"id": 1}]
    with pytest.raises(ui.WorkflowUiUnavailable) as failure:
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    assert failure.value.reason == "file_mismatch"


def test_boolean_number_difference_is_not_matching_json(server):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}/content"]["workflow_json"]["extra"] = {"flag": True}
    bodies["/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json"]["extra"] = {"flag": 1}
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize("point", ["metadata", "content", "file"])
def test_old_record_only_or_inaccessible_workflow_has_safe_fallback(server, point):
    ui, bodies, _, _ = server
    paths = {
        "metadata": f"/api/workflows/{WID}",
        "content": f"/api/workflows/{WID}/content",
        "file": "/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json",
    }
    bodies[paths[point]] = urllib.error.HTTPError("https://cloud.test", 404, "missing", {}, None)
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize(
    "environment,deployment", [("standard", DEPLOYMENT), ("custom", None), ("invalid", None), ("custom", "../escape")]
)
def test_environment_deployment_conflicts_refused_before_network(server, environment, deployment):
    ui, _, calls, _ = server
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment=environment, deployment_id=deployment)
    assert calls == []


def test_oauth_session_required(server, monkeypatch):
    ui, _, calls, _ = server
    monkeypatch.setattr(
        ui, "resolve_target", lambda **kw: Target(kind="cloud", base_url="https://cloud.test", api_key="test-key")
    )
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    assert calls == []


def test_custom_link_reuses_exact_verified_deployment_target(server, monkeypatch):
    ui, _, _, _ = server
    control = Mock()
    control.get_deployment.return_value = {"id": DEPLOYMENT, "releaseId": "release-1", "status": "ready"}
    monkeypatch.setattr(ui, "_deployment_client", lambda: control)
    verified = Mock(
        return_value={
            "workspaceId": WORKSPACE,
            "deploymentId": DEPLOYMENT,
            "releaseId": "release-1",
            "url": "https://cloud.test/?workspace=" + WORKSPACE + "&deployment=" + DEPLOYMENT,
        }
    )
    monkeypatch.setattr(ui, "verify_deployment_editor_target", verified)
    result = ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="custom", deployment_id=DEPLOYMENT)
    assert parse_qs(urlsplit(result["url"]).query) == {
        "workflow": [WID],
        "workspace": [WORKSPACE],
        "deployment": [DEPLOYMENT],
    }
    assert result["releaseId"] == "release-1"
    control.get_deployment.assert_called_once_with(DEPLOYMENT)
    assert verified.call_args.kwargs["workspace_id"] == WORKSPACE


def test_error_response_never_leaks_private_prompts_or_credentials(server, monkeypatch):
    ui, _, _, _ = server
    renderer = Mock()
    monkeypatch.setattr(ui, "get_renderer", lambda: renderer)
    monkeypatch.setattr(ui, "request_json", Mock(side_effect=ValueError("SECRET PRIVATE PROMPT oauth-token")))
    with pytest.raises(Exception):
        ui.run_ui(WID, workspace=WORKSPACE, environment="standard")
    error = renderer.error.call_args.kwargs
    assert error["code"] == "workflow_ui_unavailable"
    assert "SECRET" not in json.dumps(error)
    assert "oauth-token" not in json.dumps(error)


def test_cli_environment_is_a_real_parser_choice(monkeypatch):
    from comfy_cli.command import workflow_ui as ui

    action = Mock()
    monkeypatch.setattr(ui, "run_ui", action)
    result = CliRunner().invoke(cmd.app, ["ui", WID, "--workspace", WORKSPACE, "--environment", "bogus"])
    assert result.exit_code == 2
    action.assert_not_called()


@pytest.mark.parametrize("change", ["stopped", "wrong-release", "missing", "duplicate", "editor-disabled"])
def test_custom_link_uses_real_shared_deployment_verification(server, monkeypatch, change):
    ui, _, _, target = server
    control = Mock()
    control.get_deployment.return_value = {
        "id": DEPLOYMENT,
        "releaseId": "release-1",
        "status": "stopped" if change == "stopped" else "ready",
    }
    if change == "editor-disabled":
        control.get_deployment.return_value["editor"] = False
    monkeypatch.setattr(ui, "_deployment_client", lambda: control)
    row = {
        "deployment_id": DEPLOYMENT,
        "release_id": "other" if change == "wrong-release" else "release-1",
        "status": "ready",
    }
    rows = [] if change == "missing" else [row, row] if change == "duplicate" else [row]
    cloud = Mock()
    cloud.target = target
    cloud.list_workspace_deployments.return_value = {"items": rows}
    monkeypatch.setattr(ui, "Client", lambda selected: cloud)
    with pytest.raises(ui.WorkflowUiUnavailable) as failure:
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="custom", deployment_id=DEPLOYMENT)
    assert failure.value.reason == "deployment_unverified"


def test_custom_link_real_shared_verifier_passes_and_schema_matches(server, monkeypatch):
    from pathlib import Path

    import jsonschema

    ui, _, _, target = server
    control = Mock()
    control.get_deployment.return_value = {"id": DEPLOYMENT, "releaseId": "release-1", "status": "ready"}
    monkeypatch.setattr(ui, "_deployment_client", lambda: control)
    cloud = Mock()
    cloud.target = target
    cloud.list_workspace_deployments.return_value = {
        "items": [{"deployment_id": DEPLOYMENT, "release_id": "release-1", "status": "ready"}]
    }
    monkeypatch.setattr(ui, "Client", lambda selected: cloud)
    schema = json.loads((Path(cmd.__file__).parent.parent / "schemas/workflow_ui.json").read_text())
    custom = ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="custom", deployment_id=DEPLOYMENT)
    standard = ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    for result in [custom, standard]:
        jsonschema.Draft202012Validator(schema).validate(result)
    cloud.list_workspace_deployments.assert_called_once_with(WORKSPACE)


@pytest.mark.parametrize("location", ["metadata", "content", "userdata", "features"])
def test_malformed_bodies_are_refused_without_content_in_errors(server, location):
    ui, bodies, _, _ = server
    paths = {
        "metadata": f"/api/workflows/{WID}",
        "content": f"/api/workflows/{WID}/content",
        "userdata": "/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json",
        "features": "/api/features",
    }
    bodies[paths[location]] = ["SECRET PRIVATE PROMPT"]
    with pytest.raises(ui.WorkflowUiUnavailable) as failure:
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
    assert "SECRET" not in str(failure.value)


def test_actual_read_only_http_transport_and_single_segment_userdata_encoding(server, monkeypatch):
    import io

    from comfy_cli import http

    ui, bodies, _, _ = server
    observed = []

    class Response(io.BytesIO):
        status = 200

    def open_request(request, *, timeout):
        assert request.get_method() == "GET"
        assert request.get_header("Authorization") == "Bearer oauth-test"
        assert timeout == 15
        observed.append(urlsplit(request.full_url).path)
        return Response(json.dumps(bodies[observed[-1]]).encode())

    monkeypatch.setattr(http._AUTHED_OPENER, "open", open_request)
    monkeypatch.setattr(ui, "request_json", http.request_json)
    assert ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")["verified"] is True
    assert observed[-1] == "/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json"


@pytest.mark.parametrize("field", ["created_by", "created_at"])
def test_content_requires_the_frontend_response_contract(server, field):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}/content"].pop(field)
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
def test_frontend_datetime_metadata_must_be_valid(server, field):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}"][field] = "not-a-timestamp"
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize(
    "field,value", [("default_view", "unknown"), ("description", 12), ("forked_from", {"workflow_id": 12})]
)
def test_optional_metadata_still_obeys_frontend_schema(server, field, value):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}"][field] = value
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


def test_malformed_content_asset_references_cannot_produce_a_link(server):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}/content"]["dependency_asset_ids"] = [42]
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")


@pytest.mark.parametrize(
    "graph", [{"nodes": []}, {"version": 0.4, "last_node_id": 1, "last_link_id": 0, "nodes": [{}], "links": []}]
)
def test_actual_consumer_invalid_graph_is_not_a_verified_normal_link(server, graph):
    ui, bodies, _, _ = server
    bodies[f"/api/workflows/{WID}/content"]["workflow_json"] = graph
    bodies["/api/userdata/workflows%2Fsub%2FMy%20saved%20workflow.json"] = graph
    with pytest.raises(ui.WorkflowUiUnavailable):
        ui.verify_workflow_link(WID, workspace=WORKSPACE, environment="standard")
