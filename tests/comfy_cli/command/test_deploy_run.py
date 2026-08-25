from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import NotRequired, TypedDict

import click
import pytest
import typer
from click.testing import Result
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command import deploy, deploy_run
from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.deploy_assets import AssetResolveRequest, AssetResolveResult
from comfy_cli.deploy_events import JobWatchResult
from comfy_cli.deploy_jobs import JobSubmitRequest
from comfy_cli.target import Target


class Envelope(TypedDict):
    data: NotRequired[JsonObject]
    error: NotRequired[JsonObject]


class FakeBuilder:
    def list_releases(self, _build_id: str) -> list[JsonObject]:
        return [{"id": "release-1", "deployable": True, "version": 1}]


class FakeControl:
    def __init__(self, endpoint_url: str | None, *, status: str = "ready") -> None:
        self.target = Target(
            kind="cloud", base_url="https://control.test", path_prefix="/v1", auth_token="cloud-secret"
        )
        self.row: JsonObject = {"id": "dep-id", "status": status, "endpointUrl": endpoint_url}
        self.calls: list[str] = []

    def list_all_deployments(self) -> list[JsonObject]:
        self.calls.append("list")
        return [copy.deepcopy(self.row)]

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.calls.append(f"get:{deployment_id}")
        return copy.deepcopy({**self.row, "id": deployment_id})


class FakeAssetClient:
    def __init__(self, result: AssetResolveResult | None = None) -> None:
        self.result = result
        self.requests: list[AssetResolveRequest] = []

    def resolve_asset(self, request: AssetResolveRequest) -> AssetResolveResult:
        self.requests.append(request)
        assert self.result is not None
        return self.result


class FakeJobClient:
    def __init__(self, response: JsonObject) -> None:
        self.response = response
        self.requests: list[JobSubmitRequest] = []

    def submit_job(self, request: JobSubmitRequest, control_plane: FakeControl) -> JsonObject:
        self.requests.append(request)
        assert control_plane is not None
        return copy.deepcopy(self.response)


def job(status: str = "queued", *, outputs: list[JsonObject] | None = None) -> JsonObject:
    output_values: list[JsonValue] = [*(outputs or [])]
    return {
        "id": "job-1",
        "status": status,
        "outputs": output_values,
        "metrics": {"queue_ms": 9, "execution_ms": 42} if status == "succeeded" else {},
        "urls": {
            "self": "/api/v2/jobs/job-1",
            "events": "/api/v2/jobs/job-1/events",
            "cancel": "/api/v2/jobs/job-1/cancel",
        },
    }


def write_workflow(path: Path, *, local_asset: bool = False) -> Path:
    inputs: JsonObject = {"text": "hello"}
    if local_asset:
        (path.parent / "input.png").write_bytes(b"input")
        inputs["image"] = "input.png"
    path.write_text(json.dumps({"1": {"class_type": "Test", "inputs": inputs}}), encoding="utf-8")
    return path


def envelope(result: Result) -> Envelope:
    parsed = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    assert isinstance(parsed, dict)
    resolved: Envelope = {}
    data = parsed.get("data")
    if data is not None:
        assert isinstance(data, dict)
        resolved["data"] = data
    error = parsed.get("error")
    if error is not None:
        assert isinstance(error, dict)
        resolved["error"] = error
    return resolved


def envelope_data(result: Result) -> JsonObject:
    data = envelope(result).get("data")
    assert isinstance(data, dict)
    return data


def envelope_error(result: Result) -> JsonObject:
    error = envelope(result).get("error")
    assert isinstance(error, dict)
    return error


def install_run(
    monkeypatch: pytest.MonkeyPatch,
    control: FakeControl,
    job_client: FakeJobClient,
    *,
    asset_client: FakeAssetClient | None = None,
    watched: JobWatchResult | None = None,
) -> None:
    selected_asset_client = asset_client or FakeAssetClient()
    monkeypatch.setattr(deploy_run, "_command_clients", lambda: (FakeBuilder(), control))
    monkeypatch.setattr(deploy_run, "DeployAssetClient", lambda *_args, **_kwargs: selected_asset_client)
    monkeypatch.setattr(deploy_run, "DeployJobClient", lambda *_args, **_kwargs: job_client)
    if watched is not None:
        monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: watched)


def invoke(
    workflow: Path | None,
    *args: str,
    input_text: str | None = None,
    agentic: bool = True,
) -> Result:
    command = ["--json" if agentic else "--no-json", "deploy", "run", *args]
    if workflow is not None:
        command.extend(["--workflow", str(workflow)])
    return CliRunner(mix_stderr=False).invoke(
        app,
        command,
        input=input_text,
        env={"AI_AGENT": "1" if agentic else None, "NO_COLOR": "1", "COLUMNS": "400"},
    )


def test_deploy_run_is_registered_with_the_complete_option_surface() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "run", "--help"])

    # Then
    assert result.exit_code == 0
    assert "--workflow" in result.stdout
    assert "--deployment" in result.stdout
    assert "--wait" in result.stdout
    assert "--no-wait" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--timeout" in result.stdout
    assert "--no-upload" in result.stdout


def test_deploy_tree_is_exactly_the_twelve_designed_commands() -> None:
    # Given / When
    command = typer.main.get_command(deploy.app)
    assert isinstance(command, click.Group)
    command_names = set(command.commands)

    # Then
    assert command_names == {
        "up",
        "status",
        "ls",
        "show",
        "logs",
        "events",
        "scale",
        "stop",
        "start",
        "delete",
        "run",
        "refs",
    }


def test_happy_wait_resolves_assets_downloads_outputs_and_emits_the_full_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json", local_asset=True)
    output: JsonObject = {
        "node_id": "9",
        "name": "image.png",
        "type": "image",
        "id": "output-1",
        "url": "https://storage.googleapis.com/signed",
    }
    asset_client = FakeAssetClient(AssetResolveResult({"id": "asset-1"}, uploaded=0, deduped=1, bytes=0))
    job_client = FakeJobClient(job())
    watched = JobWatchResult(job("succeeded", outputs=[output]), [output])
    install_run(
        monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client, asset_client=asset_client, watched=watched
    )
    monkeypatch.setattr(
        deploy_run,
        "download_job_outputs",
        lambda *_args, **_kwargs: [
            {"node_id": "9", "name": "image.png", "type": "image", "id": "output-1", "path": "outputs/image.png"}
        ],
    )

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    data = envelope_data(result)
    assert result.exit_code == 0, result.stderr
    assert data["assets"] == {"uploaded": 0, "deduped": 1, "bytes": 0}
    assert data["job"] == {"id": "job-1", "status": "succeeded"}
    assert data["metrics"] == {"queue_ms": 9, "execution_ms": 42}
    output_values = data["outputs"]
    assert isinstance(output_values, list)
    first_output = output_values[0]
    assert isinstance(first_output, dict)
    assert first_output["path"] == "outputs/image.png"
    submitted_node = job_client.requests[0].workflow["1"]
    assert isinstance(submitted_node, dict)
    submitted_inputs = submitted_node["inputs"]
    assert isinstance(submitted_inputs, dict)
    submitted_image = submitted_inputs["image"]
    assert isinstance(submitted_image, dict)
    submitted_info = submitted_image["info"]
    assert isinstance(submitted_info, dict)
    assert submitted_info["id"] == "asset-1"


def test_no_wait_returns_after_submission_without_watch_or_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client)
    monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: pytest.fail("watch called"))
    monkeypatch.setattr(deploy_run, "download_job_outputs", lambda *_args, **_kwargs: pytest.fail("download called"))

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    data = envelope_data(result)
    assert result.exit_code == 0
    assert data["job"] == {"id": "job-1", "status": "queued"}
    assert data["outputs"] == []


def test_each_invocation_uses_a_fresh_job_idempotency_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client)

    # When
    first = invoke(workflow, "--deployment", "dep-id", "--no-wait")
    second = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert first.exit_code == second.exit_code == 0
    assert len({request.idempotency_key for request in job_client.requests}) == 2
