from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest
from test_deploy_run import (
    FakeControl,
    FakeJobClient,
    envelope_error,
    install_run,
    invoke,
    job,
    write_workflow,
)

from comfy_cli.command import deploy_run
from comfy_cli.deploy_events import JobWatchResult


class InterruptedSubmitClient(FakeJobClient):
    def submit_job(self, request, control_plane):
        raise KeyboardInterrupt


def test_sigint_before_submission_has_no_cancel_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(
        monkeypatch,
        FakeControl("https://dep-id.run.comfy.app"),
        InterruptedSubmitClient(job()),
    )
    cancels: list[str] = []
    monkeypatch.setattr(deploy_run, "request_json", lambda url, *_args, **_kwargs: cancels.append(url))

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 130
    assert cancels == []


def test_sigint_after_submission_issues_exactly_one_cancel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()))
    monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))
    cancels: list[tuple[str, str]] = []

    def cancel(url, _target, *, method="GET", **_kwargs):
        cancels.append((url, method))
        return 202, {}

    monkeypatch.setattr(deploy_run, "request_json", cancel)

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 130
    assert cancels == [("https://dep-id.run.comfy.app/api/v2/jobs/job-1/cancel", "POST")]


def test_failed_cancel_warns_but_keeps_sigint_exit_130(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()))
    monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))

    def fail_cancel(*_args, **_kwargs):
        raise urllib.error.URLError("cancel offline")

    monkeypatch.setattr(deploy_run, "request_json", fail_cancel)

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 130
    assert "cancel offline" in result.stderr


def test_second_sigint_during_cancel_exits_without_a_second_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()))
    monkeypatch.setattr(deploy_run, "watch_job", lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt))
    attempts = 0

    def interrupt_cancel(*_args, **_kwargs):
        nonlocal attempts
        attempts += 1
        raise KeyboardInterrupt

    monkeypatch.setattr(deploy_run, "request_json", interrupt_cancel)

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 130
    assert attempts == 1


def test_timeout_issues_one_cancel_and_exits_nonzero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    install_run(monkeypatch, FakeControl("https://dep-id.run.comfy.app"), FakeJobClient(job()))
    monkeypatch.setattr(
        deploy_run,
        "_watch_with_timeout",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(deploy_run.DeployRunTimeoutError(1.0)),
    )
    cancels: list[str] = []

    def cancel(url, _target, **_kwargs):
        cancels.append(url)
        return 202, {}

    monkeypatch.setattr(deploy_run, "request_json", cancel)

    # When
    result = invoke(workflow, "--deployment", "dep-id", "--timeout", "1")

    # Then
    assert result.exit_code == 1
    assert envelope_error(result)["code"] == "deploy_server_error"
    assert cancels == ["https://dep-id.run.comfy.app/api/v2/jobs/job-1/cancel"]


def test_canceled_authoritative_job_uses_the_distinct_error_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given
    workflow = write_workflow(tmp_path / "workflow.json")
    canceled = JobWatchResult(job("canceled"), [])
    install_run(
        monkeypatch,
        FakeControl("https://dep-id.run.comfy.app"),
        FakeJobClient(job()),
        watched=canceled,
    )

    # When
    result = invoke(workflow, "--deployment", "dep-id")

    # Then
    assert result.exit_code == 1
    assert envelope_error(result)["code"] == "deploy_job_canceled"
