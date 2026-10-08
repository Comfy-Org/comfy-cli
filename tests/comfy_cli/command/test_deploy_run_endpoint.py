from __future__ import annotations

from pathlib import Path

import pytest
from test_deploy_run import (
    FakeControl,
    FakeJobClient,
    envelope_data,
    envelope_error,
    install_run,
    invoke,
    job,
    write_workflow,
)

from comfy_cli.caller import Caller
from comfy_cli.command import deploy_run


@pytest.mark.parametrize(
    "endpoint_url",
    [
        "https://api.runpod.ai/v2/xyz",
        "https://dep-id.attacker.example",
        "https://dep-id.attacker.run.comfy.app",
        "https://dep-id.run.comfy.app/path",
        "https://dep-id.run.comfy.app?query=1",
    ],
)
def test_untrusted_endpoint_stops_before_any_data_plane_request(
    endpoint_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    control = FakeControl(endpoint_url)
    job_client = FakeJobClient(job())
    install_run(monkeypatch, control, job_client)

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert result.exit_code == 1
    assert envelope_error(result)["code"] == "deploy_endpoint_unknown"
    assert control.calls == ["get:dep-id"]
    assert job_client.requests == []


@pytest.mark.parametrize("endpoint_url", ["https://dep-id.run.comfy.app", "https://dep-id.stg.run.comfy.app"])
def test_trusted_endpoint_submits_to_the_server_returned_origin(
    endpoint_url: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl(endpoint_url), job_client)

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    assert result.exit_code == 0, result.stderr
    assert len(job_client.requests) == 1
    assert envelope_data(result)["deployment"] == {"id": "dep-id", "endpointUrl": endpoint_url}


def test_non_ready_deployment_reports_the_actual_status_without_data_plane_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    job_client = FakeJobClient(job())
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app", status="provisioning"), job_client)

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--no-wait")

    # Then
    error = envelope_error(result)
    assert result.exit_code == 1
    assert error["code"] == "deploy_not_ready"
    details = error["details"]
    assert isinstance(details, dict)
    assert details["status"] == "provisioning"
    assert job_client.requests == []


def test_job_links_are_resolved_against_and_confined_to_the_endpoint_origin(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    response = job()
    urls = response["urls"]
    assert isinstance(urls, dict)
    urls["self"] = "https://attacker.example/jobs/job-1"
    job_client = FakeJobClient(response)
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), job_client)

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 1
    assert envelope_error(result)["code"] == "deploy_endpoint_unknown"


def test_missing_workflow_is_one_error_envelope_and_zero_plane_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    monkeypatch.setattr(deploy_run, "_command_clients", lambda: pytest.fail("plane client constructed"))

    # When
    result = invoke(None, "--deployment", "dep-id")

    # Then
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert result.exit_code == 1
    assert len(lines) == 1
    assert envelope_error(result)["code"] == "deploy_missing_input"


def test_missing_workflow_prompts_a_human_for_a_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()))
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", False, None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)

    # When
    result = invoke(
        None,
        "--deployment",
        "dep-id",
        "--no-wait",
        input_text=f"{workflow}\n",
        agentic=False,
    )

    # Then
    assert result.exit_code == 0, result.stderr


def test_ui_workflow_maps_to_the_deploy_format_error_before_any_plane_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = tmp_path / "ui.json"
    workflow.write_text('{"nodes": [], "links": []}', encoding="utf-8")
    monkeypatch.setattr(deploy_run, "_command_clients", lambda: pytest.fail("plane client constructed"))

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 1
    assert envelope_error(result)["code"] == "deploy_workflow_format_ui"


@pytest.mark.parametrize(
    ("payload", "code"),
    [
        ("[1, 2]", "deploy_workflow_not_api_format"),
        ('"a string"', "deploy_workflow_not_api_format"),
        ("42", "deploy_workflow_not_api_format"),
        ("null", "deploy_workflow_not_api_format"),
        ("true", "deploy_workflow_not_api_format"),
        ("{}", "deploy_workflow_empty"),
    ],
    ids=lambda value: str(value).replace(" ", ""),
)
def test_a_workflow_that_is_not_an_object_of_nodes_is_refused_with_an_envelope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
    code: str,
) -> None:
    """`_load_workflow_file` returns whatever `json.load` produced.

    The node scan then met `.items()` on a list, a string or `None`, so an
    `AttributeError` escaped with no envelope at all — the one outcome a
    `--json` caller cannot act on. `{}` is the neighbouring case: a valid empty
    object, which used to scan to an empty plan and proceed to submit nothing.
    """
    # Given
    workflow = tmp_path / "not-a-workflow.json"
    workflow.write_text(payload, encoding="utf-8")
    monkeypatch.setattr(deploy_run, "_command_clients", lambda: pytest.fail("plane client constructed"))

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 1
    assert envelope_error(result)["code"] == code
    assert not isinstance(result.exception, AttributeError)
