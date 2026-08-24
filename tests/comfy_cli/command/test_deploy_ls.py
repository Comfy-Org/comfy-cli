from __future__ import annotations

import copy
import importlib
import io
import json
import urllib.error
from collections.abc import Iterator
from email.message import Message
from pathlib import Path

import pytest
from deploy_up_support import FakeBuilder, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject, JsonValue
from comfy_cli.deploy_api import DeployClient


def _summary(
    deployment_id: str,
    release_id: str,
    *,
    status: str = "ready",
    deleted_at: str | None = None,
    deleted_by: str | None = None,
) -> JsonObject:
    return {
        "buildVersionId": release_id,
        "computeConfig": None,
        "createdAt": "2026-08-23T12:00:00Z",
        "deletedAt": deleted_at,
        "deletedBy": deleted_by,
        "distributionVersionId": release_id,
        "endpointUrl": None,
        "id": deployment_id,
        "status": status,
        "updatedAt": "2026-08-23T12:01:00Z",
    }


class PagedDeploy:
    def __init__(self, pages: list[list[JsonObject]]) -> None:
        self.pages = pages
        self.calls: list[tuple[str | None, int | None]] = []
        self.pages_yielded = 0

    def iter_deployments(self, *, status: str | None = None, limit: int | None = None) -> Iterator[JsonObject]:
        self.calls.append((status, limit))
        for rows in self.pages:
            self.pages_yielded += 1
            page_rows: list[JsonValue] = [*copy.deepcopy(rows)]
            yield {"deployments": page_rows}


def _install_clients(monkeypatch, builder: FakeBuilder, client: PagedDeploy | DeployClient) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_ls")
    monkeypatch.setattr(module, "_command_clients", lambda: (builder, client))


def _invoke_json(path: Path, *args: str):
    return CliRunner(mix_stderr=False).invoke(app, ["--json", "deploy", "ls", str(path), *args])


def _payload(result) -> JsonObject:
    envelope = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    return envelope["data"]


def _deployments(result) -> list[JsonObject]:
    rows = _payload(result).get("deployments")
    assert isinstance(rows, list)
    deployments: list[JsonObject] = []
    for row in rows:
        assert isinstance(row, dict)
        deployments.append(row)
    return deployments


def test_deploy_ls_is_a_registered_real_command() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "ls", "--help"])

    # Then
    assert result.exit_code == 0
    assert "--all" in result.stdout
    assert "--workspace" in result.stdout
    assert "--status" in result.stdout
    assert "--limit" in result.stdout


@pytest.mark.parametrize(
    ("args", "expected_ids"),
    [
        ((), {"build-live", "build-stopped"}),
        (("-a",), {"build-live", "build-stopped", "build-deleted"}),
        (("--workspace",), {"build-live", "build-stopped", "other-live"}),
        (
            ("--workspace", "--all"),
            {"build-live", "build-stopped", "build-deleted", "other-live", "other-deleted"},
        ),
    ],
    ids=("build-live", "build-all", "workspace-live", "workspace-all"),
)
def test_build_scope_and_deleted_visibility_are_independent_axes(
    tmp_path: Path,
    monkeypatch,
    args: tuple[str, ...],
    expected_ids: set[str],
) -> None:
    # Given
    rows = [
        _summary("build-live", "release-5"),
        _summary("build-stopped", "release-5", status="stopped"),
        _summary("build-deleted", "release-5", deleted_at="2026-08-23T13:00:00Z", deleted_by="user-1"),
        _summary("other-live", "release-other"),
        _summary("other-deleted", "release-other", deleted_at="2026-08-23T14:00:00Z", deleted_by="user-2"),
    ]
    _install_clients(monkeypatch, FakeBuilder(), PagedDeploy([rows]))

    # When
    result = _invoke_json(write_spec(tmp_path), *args)

    # Then
    assert result.exit_code == 0
    deployments = _deployments(result)
    actual_ids: set[str] = set()
    for row in deployments:
        deployment_id = row.get("id")
        assert isinstance(deployment_id, str)
        actual_ids.add(deployment_id)
    assert actual_ids == expected_ids
    if "build-deleted" in expected_ids:
        deleted = next(row for row in deployments if row["id"] == "build-deleted")
        assert deleted["deletedAt"] == "2026-08-23T13:00:00Z"
        assert deleted["deletedBy"] == "user-1"


def test_workspace_scope_ignores_path_entirely(tmp_path: Path, monkeypatch) -> None:
    # Given
    _install_clients(monkeypatch, FakeBuilder(), PagedDeploy([[_summary("other-live", "release-other")]]))

    # When
    result = _invoke_json(tmp_path / "does-not-exist", "--workspace")

    # Then
    assert result.exit_code == 0
    assert [row["id"] for row in _deployments(result)] == ["other-live"]


def test_stopped_non_deleted_row_is_never_hidden(tmp_path: Path, monkeypatch) -> None:
    # Given
    _install_clients(
        monkeypatch, FakeBuilder(), PagedDeploy([[_summary("dep-stopped", "release-5", status="stopped")]])
    )

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert [row["status"] for row in _deployments(result)] == ["stopped"]


def test_pretty_output_marks_deleted_rows(tmp_path: Path, monkeypatch) -> None:
    # Given
    row = _summary("dep-deleted", "release-5", deleted_at="2026-08-23T13:00:00Z", deleted_by="user-1")
    _install_clients(monkeypatch, FakeBuilder(), PagedDeploy([[row]]))

    # When
    result = CliRunner(mix_stderr=False).invoke(
        app,
        ["--no-json", "deploy", "ls", str(write_spec(tmp_path)), "--all"],
        env={"COLUMNS": "400"},
    )

    # Then
    assert result.exit_code == 0
    assert "dep-deleted" in result.stdout
    assert "(deleted)" in result.stdout


def test_limit_continues_to_a_second_page_until_enough_visible_rows(tmp_path: Path, monkeypatch) -> None:
    # Given
    client = PagedDeploy(
        [
            [
                _summary("deleted", "release-5", deleted_at="2026-08-23T13:00:00Z"),
                _summary("other-build", "release-other"),
                _summary("visible-1", "release-5"),
            ],
            [_summary("visible-2", "release-5"), _summary("visible-3", "release-5")],
        ]
    )
    _install_clients(monkeypatch, FakeBuilder(), client)

    # When
    result = _invoke_json(write_spec(tmp_path), "--limit", "2")

    # Then
    assert result.exit_code == 0
    assert [row["id"] for row in _deployments(result)] == ["visible-1", "visible-2"]
    assert client.calls == [(None, 2)]
    assert client.pages_yielded == 2


def test_invalid_status_is_sent_to_server_and_maps_to_deploy_bad_request(tmp_path: Path, monkeypatch) -> None:
    # Given
    requested_urls: list[str] = []

    def reject_status(url, *_args, **_kwargs):
        requested_urls.append(url)
        body = io.BytesIO(
            json.dumps({"error": "invalid_query", "message": "status query parameter is invalid"}).encode()
        )
        raise urllib.error.HTTPError(url, 400, "Bad Request", Message(), body)

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", reject_status)
    _install_clients(monkeypatch, FakeBuilder(), DeployClient("https://deploy.test", "token"))

    # When
    result = _invoke_json(tmp_path / "ignored", "--workspace", "--status", "not-a-status")

    # Then
    assert result.exit_code == 1
    envelope = json.loads(result.stdout)
    assert envelope["error"]["code"] == "deploy_bad_request"
    assert "status" in envelope["error"]["message"]
    assert "status=not-a-status" in requested_urls[0]
