from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest
from deploy_up_support import FakeBuilder, deployment, option_names, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject


class ReadDeploy:
    def __init__(self, rows: list[JsonObject] | None = None) -> None:
        self.rows = copy.deepcopy(rows or [])
        self.calls: list[tuple[str, str | None]] = []
        self.detail = {
            **deployment("dep-read"),
            "releaseId": "release-5",
            "endpointUrl": "https://dep-read.run.comfy.app",
            "error": None,
            "serving": None,
            "stopReason": None,
        }
        self.logs: JsonObject = {
            "capturedAt": "2026-08-23T12:05:00Z",
            "comfyuiLog": "ComfyUI started",
            "deploymentId": "dep-read",
        }
        self.events: JsonObject = {
            "deploymentId": "dep-read",
            "events": [
                {"at": "2026-08-23T12:00:00Z", "message": None, "status": "provisioning"},
                {"at": "2026-08-23T12:01:00Z", "message": "worker online", "status": "starting"},
                {"at": "2026-08-23T12:02:00Z", "message": None, "status": "ready"},
            ],
        }

    def list_all_deployments(self) -> list[JsonObject]:
        self.calls.append(("list", None))
        return copy.deepcopy(self.rows)

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(("show", deployment_id))
        return copy.deepcopy({**self.detail, "id": deployment_id})

    def get_deployment_logs(self, deployment_id: str) -> JsonObject:
        self.calls.append(("logs", deployment_id))
        return copy.deepcopy({**self.logs, "deploymentId": deployment_id})

    def get_deployment_events(self, deployment_id: str) -> JsonObject:
        self.calls.append(("events", deployment_id))
        return copy.deepcopy({**self.events, "deploymentId": deployment_id})


def _install_clients(monkeypatch, builder: FakeBuilder, client: ReadDeploy) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_read")
    monkeypatch.setattr(module, "_command_clients", lambda: (builder, client))


def _invoke_json(command: str, *args: str):
    return CliRunner(mix_stderr=False).invoke(app, ["--json", "deploy", command, *args])


def _envelope(result) -> JsonObject:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


@pytest.mark.parametrize("command", ["show", "logs", "events"])
def test_deploy_read_command_is_registered(command: str) -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", command, "--help"])

    # Then
    assert result.exit_code == 0
    options = option_names(command)
    assert "--deployment" in options
    if command == "logs":
        assert "--follow" not in options


def test_show_with_explicit_id_ignores_path_and_renders_raw_record(monkeypatch, tmp_path: Path) -> None:
    # Given
    builder = FakeBuilder()
    client = ReadDeploy()
    _install_clients(monkeypatch, builder, client)

    # When
    result = _invoke_json("show", str(tmp_path / "missing"), "--deployment", "dep-explicit")

    # Then
    assert result.exit_code == 0
    assert _envelope(result)["data"] == {**client.detail, "id": "dep-explicit"}
    assert builder.calls == []
    assert client.calls == [("show", "dep-explicit")]


@pytest.mark.parametrize("command", ["show", "logs", "events"])
def test_read_commands_fall_back_to_the_spec_deployment(
    command: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Given
    builder = FakeBuilder()
    client = ReadDeploy([deployment("dep-read")])
    _install_clients(monkeypatch, builder, client)

    # When
    result = _invoke_json(command, str(write_spec(tmp_path)))

    # Then
    assert result.exit_code == 0
    assert builder.calls == [("list_releases", "build-1")]
    assert client.calls == [("list", None), (command, "dep-read")]


@pytest.mark.parametrize("command", ["show", "logs", "events"])
def test_missing_spec_deployment_maps_to_deploy_not_found_with_remedy(
    command: str,
    monkeypatch,
    tmp_path: Path,
) -> None:
    # Given
    client = ReadDeploy()
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = _invoke_json(command, str(write_spec(tmp_path)))

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert isinstance(error, dict)
    hint = error.get("hint")
    assert isinstance(hint, str)
    assert error["code"] == "deploy_not_found"
    assert "comfy deploy up" in hint
    assert "comfy deploy status" in hint
    assert client.calls == [("list", None)]


def test_logs_rejects_follow_as_an_unknown_typer_option() -> None:
    # Given / When
    result = CliRunner(mix_stderr=False).invoke(app, ["deploy", "logs", "--follow"])

    # Then
    assert result.exit_code != 0
    assert "No such option" in result.stderr
    assert "--follow" not in option_names("logs")


def test_logs_always_renders_capture_vintage(monkeypatch) -> None:
    # Given
    client = ReadDeploy()
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = CliRunner(mix_stderr=False).invoke(
        app,
        ["--no-json", "deploy", "logs", "--deployment", "dep-read"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 0
    assert "capturedAt: 2026-08-23T12:05:00Z" in result.stdout
    assert "ComfyUI started" in result.stdout


def test_unprobed_logs_explicitly_say_not_captured_yet(monkeypatch) -> None:
    # Given
    client = ReadDeploy()
    client.logs = {"capturedAt": None, "comfyuiLog": "", "deploymentId": "dep-read"}
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = CliRunner(mix_stderr=False).invoke(
        app,
        ["--no-json", "deploy", "logs", "--deployment", "dep-read"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 0
    assert "capturedAt: not captured yet" in result.stdout


def test_events_render_in_server_order_without_false_complete_framing(monkeypatch) -> None:
    # Given
    client = ReadDeploy()
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = CliRunner(mix_stderr=False).invoke(
        app,
        ["--no-json", "deploy", "events", "--deployment", "dep-read"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 0
    assert result.stdout.index("provisioning") < result.stdout.index("starting") < result.stdout.index("ready")
    assert "since creation" not in result.stdout.lower()
    assert "beginning of history" not in result.stdout.lower()


@pytest.mark.parametrize("output_flag", ["--json", "--no-json"])
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"capturedAt": "2026-08-23T12:05:00Z", "comfyuiLog": "started"}, id="missing-deploymentId"),
        pytest.param(
            {"capturedAt": 7, "comfyuiLog": "started", "deploymentId": "dep-read"}, id="non-string-capturedAt"
        ),
        pytest.param({"capturedAt": "2026-08-23T12:05:00Z", "deploymentId": "dep-read"}, id="missing-comfyuiLog"),
    ],
)
def test_both_output_modes_reject_the_same_malformed_logs(payload, output_flag, monkeypatch) -> None:
    """`deploy_logs.json` requires deploymentId, so forwarding a response that
    omits it emits an envelope that fails its own published schema."""
    # Given
    client = ReadDeploy()
    monkeypatch.setattr(ReadDeploy, "get_deployment_logs", lambda self, _id: copy.deepcopy(payload))
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = CliRunner(mix_stderr=False).invoke(
        app,
        [output_flag, "deploy", "logs", "--deployment", "dep-read"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 1
    assert "deploy_server_error" in result.stdout
    if output_flag == "--json":
        assert _envelope(result)["error"]["code"] == "deploy_server_error"


@pytest.mark.parametrize("output_flag", ["--json", "--no-json"])
@pytest.mark.parametrize(
    "payload",
    [
        pytest.param({"deploymentId": "dep-read", "events": ["not-an-object"]}, id="non-object-event"),
        pytest.param(
            {"deploymentId": "dep-read", "events": [{"at": "2026-08-23T12:00:00Z", "status": "ready", "message": 7}]},
            id="non-string-message",
        ),
        pytest.param({"deploymentId": "dep-read", "events": [{"status": "ready"}]}, id="missing-at"),
        pytest.param({"deploymentId": "dep-read", "events": [{"at": "2026-08-23T12:00:00Z"}]}, id="missing-status"),
        pytest.param({"events": []}, id="missing-deploymentId"),
    ],
)
def test_both_output_modes_reject_the_same_malformed_events(payload, output_flag, monkeypatch) -> None:
    """Validation must not sit behind an ``is_pretty()`` branch: a `--json`
    consumer has to be refused exactly the responses a human is refused."""
    # Given
    client = ReadDeploy()
    monkeypatch.setattr(ReadDeploy, "get_deployment_events", lambda self, _id: copy.deepcopy(payload))
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = CliRunner(mix_stderr=False).invoke(
        app,
        [output_flag, "deploy", "events", "--deployment", "dep-read"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 1
    assert "deploy_server_error" in result.stdout
    if output_flag == "--json":
        assert _envelope(result)["error"]["code"] == "deploy_server_error"
