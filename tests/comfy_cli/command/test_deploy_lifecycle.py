from __future__ import annotations

import copy
import importlib
import io
import json
import urllib.error
from email.message import Message

import pytest
from deploy_up_support import FakeBuilder, deployment
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api import DeployClient


class LifecycleDeploy:
    def __init__(self, row: JsonObject | None = None) -> None:
        selected = row or deployment("dep-1")
        self.row = copy.deepcopy(selected)
        self.calls: list[tuple[str, str]] = []

    def list_all_deployments(self) -> list[JsonObject]:
        self.calls.append(("list", ""))
        return [copy.deepcopy(self.row)]

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(("get", deployment_id))
        return copy.deepcopy({**self.row, "id": deployment_id})

    def update_deployment(self, deployment_id: str, compute_config: JsonObject) -> JsonObject:
        self.calls.append(("scale", deployment_id))
        self.row["computeConfig"] = copy.deepcopy(compute_config)
        return copy.deepcopy(self.row)

    def stop_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(("stop", deployment_id))
        self.row["status"] = "stopping"
        return copy.deepcopy(self.row)

    def start_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(("start", deployment_id))
        self.row["status"] = "queued"
        return copy.deepcopy(self.row)

    def delete_deployment(self, deployment_id: str) -> None:
        self.calls.append(("delete", deployment_id))


class APIResponder:
    def __init__(self, row: JsonObject, *, status: int | None = None, message: str = "") -> None:
        self.row = copy.deepcopy(row)
        self.status = status
        self.message = message
        self.bodies: list[JsonObject] = []

    def __call__(self, url, _target, *, method="GET", body=None, **_kwargs):
        if method == "GET":
            return 200, copy.deepcopy(self.row)
        if self.status is not None:
            payload = io.BytesIO(json.dumps({"error": "rejected", "message": self.message}).encode())
            raise urllib.error.HTTPError(url, self.status, self.message, Message(), payload)
        assert isinstance(body, dict)
        self.bodies.append(copy.deepcopy(body))
        compute = body.get("computeConfig")
        if isinstance(compute, dict):
            self.row["computeConfig"] = copy.deepcopy(compute)
        return 202, copy.deepcopy(self.row)


def _install_client(monkeypatch, client: LifecycleDeploy | DeployClient) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_lifecycle")
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))


def _invoke(command: str, *args: str, pretty: bool = False):
    output_flag = "--no-json" if pretty else "--json"
    return CliRunner(mix_stderr=False).invoke(app, [output_flag, "deploy", command, *args], env={"COLUMNS": "400"})


def _envelope(result) -> JsonObject:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _object(value: JsonObject, key: str) -> JsonObject:
    field = value.get(key)
    assert isinstance(field, dict)
    return field


def _string(value: JsonObject, key: str) -> str:
    field = value.get(key)
    assert isinstance(field, str)
    return field


@pytest.mark.parametrize("command", ["scale", "stop", "start", "delete"])
def test_lifecycle_command_is_registered(command: str) -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", command, "--help"])

    # Then
    assert result.exit_code == 0
    assert "--deployment" in result.stdout


def test_scale_merges_unsupplied_compute_fields_and_lowers_without_a_gate(monkeypatch) -> None:
    # Given
    responder = APIResponder(deployment("dep-1", minimum=2, maximum=4))
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", responder)
    _install_client(monkeypatch, DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke("scale", "--deployment", "dep-1", "--max", "2")

    # Then
    assert result.exit_code == 0
    assert responder.bodies == [{"computeConfig": {"gpuClass": "l4", "region": "US-MO-2", "min": 2, "max": 2}}]


def test_ready_gpu_change_maps_to_immutable_compute_with_ordered_remedy(monkeypatch) -> None:
    # Given
    responder = APIResponder(
        deployment("dep-1"),
        status=409,
        message="stop the deployment before changing gpuClass or region",
    )
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", responder)
    _install_client(monkeypatch, DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke("scale", "--deployment", "dep-1", "--gpu", "a100")

    # Then
    error = _object(_envelope(result), "error")
    assert result.exit_code == 1
    assert error["code"] == "deploy_immutable_compute"
    hint = _string(error, "hint")
    assert hint.index("comfy deploy stop") < hint.index("comfy deploy scale") < hint.index("comfy deploy start")


def test_midflight_scale_conflict_names_the_fetched_status(monkeypatch) -> None:
    # Given
    responder = APIResponder(
        deployment("dep-1", status="provisioning"),
        status=409,
        message="wait until the deployment is ready or stopped before editing",
    )
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", responder)
    _install_client(monkeypatch, DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke("scale", "--deployment", "dep-1", "--max", "2")

    # Then
    error = _object(_envelope(result), "error")
    assert result.exit_code == 1
    assert error["code"] == "deploy_conflict"
    assert "provisioning" in _string(error, "message")
    assert _object(error, "details")["currentStatus"] == "provisioning"


@pytest.mark.parametrize(
    ("status", "code"),
    [(402, "deploy_payment_required"), (429, "deploy_quota_exceeded")],
)
def test_raising_bounds_maps_server_gates(status: int, code: str, monkeypatch) -> None:
    # Given
    responder = APIResponder(deployment("dep-1"), status=status, message="worker increase rejected")
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", responder)
    _install_client(monkeypatch, DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke("scale", "--deployment", "dep-1", "--max", "2")

    # Then
    assert result.exit_code == 1
    assert _object(_envelope(result), "error")["code"] == code


def test_stopped_gpu_provisioning_failure_maps_to_compute_unavailable(monkeypatch) -> None:
    # Given
    responder = APIResponder(
        deployment("dep-1", status="stopped"),
        status=400,
        message="a100 is not available in region US-MO-2",
    )
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", responder)
    _install_client(monkeypatch, DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke("scale", "--deployment", "dep-1", "--gpu", "a100")

    # Then
    assert result.exit_code == 1
    assert _object(_envelope(result), "error")["code"] == "deploy_compute_unavailable"


def test_stop_never_confirms_for_an_agentic_caller(monkeypatch) -> None:
    # Given
    client = LifecycleDeploy()
    _install_client(monkeypatch, client)
    monkeypatch.setattr("comfy_cli.interaction._ask_confirm", lambda _question: pytest.fail("stop prompted"))

    # When
    result = _invoke("stop", "--deployment", "dep-1")

    # Then
    assert result.exit_code == 0
    assert client.calls == [("stop", "dep-1")]


def test_delete_without_yes_refuses_agentic_caller(monkeypatch) -> None:
    # Given
    client = LifecycleDeploy()
    _install_client(monkeypatch, client)

    # When
    result = _invoke("delete", "--deployment", "dep-1")

    # Then
    assert result.exit_code == 1
    assert _object(_envelope(result), "error")["code"] == "deploy_delete_needs_confirm"
    assert client.calls == []


def test_delete_yes_reports_accepted_teardown_and_retained_record(monkeypatch) -> None:
    # Given
    client = LifecycleDeploy()
    _install_client(monkeypatch, client)

    # When
    result = _invoke("delete", "--deployment", "dep-1", "-y", pretty=True)

    # Then
    normalized = " ".join(result.stdout.lower().split())
    assert result.exit_code == 0
    assert "accepted" in normalized
    assert "teardown" in normalized
    assert "record remains" in normalized
    assert client.calls == [("delete", "dep-1")]


def test_start_on_deleted_row_maps_to_deploy_deleted(monkeypatch) -> None:
    # Given
    responder = APIResponder(deployment("dep-1"), status=409, message="deployment is deleted")
    monkeypatch.setattr("comfy_cli.deploy_api.request_json", responder)
    _install_client(monkeypatch, DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke("start", "--deployment", "dep-1")

    # Then
    assert result.exit_code == 1
    assert _object(_envelope(result), "error")["code"] == "deploy_deleted"
