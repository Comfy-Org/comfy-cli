from __future__ import annotations

import importlib
import json
from pathlib import Path

from deploy_up_support import FakeBuilder, FakeDeploy, deployment, option_names, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_runtime import DEPLOY_POLL_SECONDS


class RecordingDeploy(FakeDeploy):
    def __init__(self, rows: list[JsonObject] | None = None, *, get_statuses: list[str] | None = None) -> None:
        super().__init__(rows, get_statuses=get_statuses)
        self.list_calls = 0
        self.get_calls: list[str] = []

    def list_all_deployments(self) -> list[JsonObject]:
        self.list_calls += 1
        return super().list_all_deployments()

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.get_calls.append(deployment_id)
        return super().get_deployment(deployment_id)


def _release(version: int) -> JsonObject:
    return {"id": f"release-{version}", "buildId": "build-1", "version": version, "deployable": True}


def _status_deployment(release_id: str = "release-5", status: str = "ready") -> JsonObject:
    row = deployment("dep-status", release_id=release_id, status=status, maximum=2)
    row.update(
        {
            "releaseId": release_id,
            "endpointUrl": "https://dep-status.run.comfy.app" if status == "ready" else None,
            "error": None,
            "serving": None,
            "stopReason": None,
        }
    )
    return row


def _serving(*, idle: int = 0, unhealthy: int = 0) -> JsonObject:
    return {
        "workers": {
            "idle": idle,
            "initializing": 0,
            "ready": 0,
            "running": 0,
            "throttled": 0,
            "unhealthy": unhealthy,
        },
        "jobsInQueue": 0,
        "sampledAt": "2026-08-21T09:12:03Z",
    }


def _install_clients(monkeypatch, builder: FakeBuilder, client: RecordingDeploy, sleeps: list[float]) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_status")
    monkeypatch.setattr(module, "_command_clients", lambda: (builder, client))
    monkeypatch.setattr(module, "_sleep", sleeps.append)


def _invoke_json(path: Path, *args: str):
    return CliRunner(mix_stderr=False).invoke(app, ["--json", "deploy", "status", str(path), *args])


def _invoke_pretty(path: Path):
    return CliRunner(mix_stderr=False).invoke(
        app,
        ["--no-json", "deploy", "status", str(path)],
        env={"COLUMNS": "400"},
    )


def _json_envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def test_deploy_status_is_a_registered_real_command() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "status", "--help"])

    # Then
    assert result.exit_code == 0
    options = option_names("status")
    assert "--watch" in options
    assert "--deployment" not in options
    assert "--release" not in options


def test_no_deployment_exits_zero_with_nullable_payload_and_up_hint(tmp_path, monkeypatch) -> None:
    # Given
    builder = FakeBuilder([_release(5)])
    client = RecordingDeploy()
    _install_clients(monkeypatch, builder, client, [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    payload = _json_envelope(result)["data"]
    assert payload == {
        "build": {"id": "build-1", "name": "example"},
        "deployment": None,
        "release": None,
        "serving": None,
    }
    assert "comfy deploy up" in result.stderr
    assert client.list_calls == 1
    assert client.get_calls == []
    assert builder.calls == [("list_releases", "build-1")]


def test_older_release_reports_behind_with_latest_deployable_and_new_url_hint(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment("release-3")
    row["serving"] = _serving(idle=1)
    builder = FakeBuilder([_release(3), _release(5)])
    _install_clients(monkeypatch, builder, RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    payload = _json_envelope(result)["data"]
    assert payload["release"] == {
        "id": "release-3",
        "version": 3,
        "behind": True,
        "latestDeployable": {"id": "release-5", "version": 5},
    }
    assert payload["serving"]["sampledAt"] == "2026-08-21T09:12:03Z"
    assert "creates a new deployment" in result.stderr.lower()
    assert "new url" in result.stderr.lower()


def test_null_serving_renders_not_sampled_yet(tmp_path, monkeypatch) -> None:
    # Given
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([_status_deployment()]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "not sampled yet" in result.stdout.lower()


def test_all_zero_serving_renders_healthy_scale_to_zero_idle(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment()
    row["serving"] = _serving()
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "healthy idle" in result.stdout.lower()
    assert "scale-to-zero" in result.stdout.lower()
    assert "not sampled yet" not in result.stdout.lower()


def test_serving_renders_sample_vintage_beside_worker_counts(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment()
    row["serving"] = _serving(idle=1)
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "idle=1" in serving_line
    assert "2026-08-21T09:12:03Z" in serving_line


def test_unhealthy_is_recoverable_and_not_an_error(tmp_path, monkeypatch) -> None:
    # Given
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([_status_deployment(status="unhealthy")]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "recoverable" in result.stdout.lower()
    assert "failed" not in result.stdout.lower()


def test_stop_failed_is_loud_and_names_retry_stop_remedy(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment(status="stop_failed")
    row["error"] = "provider release failed"
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 1
    assert _json_envelope(result)["data"]["deployment"]["status"] == "stop_failed"
    assert "may still be billing" in result.stderr.lower()
    assert "comfy deploy stop" in result.stderr


def test_credit_stop_is_not_attributed_to_the_user(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment(status="stopped")
    row["stopReason"] = "credits"
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    rendered = f"{result.stdout}\n{result.stderr}".lower()
    assert result.exit_code == 0
    assert "insufficient credits" in rendered
    assert "stopped by user" not in rendered
    assert "user-initiated" not in rendered


def test_watch_continues_after_first_unhealthy_sample(tmp_path, monkeypatch) -> None:
    # Given
    client = RecordingDeploy([_status_deployment(status="queued")], get_statuses=["unhealthy", "ready"])
    sleeps: list[float] = []
    _install_clients(monkeypatch, FakeBuilder(), client, sleeps)

    # When
    result = _invoke_json(write_spec(tmp_path), "--watch")

    # Then
    assert result.exit_code == 0
    assert _json_envelope(result)["data"]["deployment"]["status"] == "ready"
    assert client.get_calls == ["dep-status", "dep-status"]
    assert sleeps == [DEPLOY_POLL_SECONDS]


def test_watch_exits_promptly_on_stop_failed_with_retry_stop_hint(tmp_path, monkeypatch) -> None:
    # Given
    client = RecordingDeploy([_status_deployment(status="stopping")], get_statuses=["stop_failed", "ready"])
    sleeps: list[float] = []
    _install_clients(monkeypatch, FakeBuilder(), client, sleeps)

    # When
    result = _invoke_json(write_spec(tmp_path), "--watch")

    # Then
    assert result.exit_code == 1
    assert _json_envelope(result)["data"]["deployment"]["status"] == "stop_failed"
    assert "comfy deploy stop" in result.stderr
    assert client.get_calls == ["dep-status"]
    assert client.get_statuses == ["ready"]
    assert sleeps == []
