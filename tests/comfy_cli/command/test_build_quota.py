from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build, build_quota
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError

SCHEMA = Path(__file__).resolve().parents[3] / "comfy_cli" / "schemas" / "build_quota.json"


class Builder:
    def __init__(self, limits: JsonObject | Exception) -> None:
        self.limits = limits

    def list_builds(self) -> list[JsonObject]:
        return [{"id": "build-1", "name": "portraits"}, {"id": "build-2", "name": "upscale"}]

    def list_releases(self, build_id: str) -> list[JsonObject]:
        count = {"build-1": 3, "build-2": 1}[build_id]
        return [{"id": f"{build_id}-r{n}", "version": n, "createdAt": "2026-08-20T00:00:00Z"} for n in range(count)]

    def get_limits(self) -> JsonObject:
        if isinstance(self.limits, Exception):
            raise self.limits
        return self.limits


class Deploy:
    def __init__(self, limits: JsonObject | Exception) -> None:
        self.limits = limits

    def list_all_deployments(self) -> list[JsonObject]:
        return [
            {"id": "dep-1", "releaseId": "build-1-r0", "status": "ready", "deletedAt": None},
            {"id": "dep-2", "releaseId": "build-1-r1", "status": "stopping", "deletedAt": None},
            {"id": "dep-3", "releaseId": "build-2-r0", "status": "stopped", "deletedAt": None},
            {"id": "dep-4", "releaseId": "build-2-r0", "status": "ready", "deletedAt": "2026-08-21T00:00:00Z"},
        ]

    def get_limits(self) -> JsonObject:
        if isinstance(self.limits, Exception):
            raise self.limits
        return self.limits


@pytest.fixture(autouse=True)
def stable_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("comfy_cli.tracking.prompt_tracking_consent", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.tracking.track_event", lambda *args, **kwargs: None)
    monkeypatch.setattr("comfy_cli.credentials.get_session", lambda *args, **kwargs: None)


def _quota(monkeypatch: pytest.MonkeyPatch, builder: Builder, deploy: Deploy) -> dict:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: builder)
    monkeypatch.setattr(build_quota, "deploy_client", lambda: deploy)
    result = CliRunner().invoke(
        cli_app, ["build", "quota"], env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"}
    )
    assert result.exit_code == 0, result.stderr
    data = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])["data"]
    jsonschema.Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8"))).validate(data)
    return data


def test_quota_reports_releases_and_active_deployments_against_their_limits(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given
    builder = Builder({"maxBuilds": 50, "maxRunningBuilds": 5, "maxTotalReleases": 10})
    deploy = Deploy({"maxActiveDeployments": 3, "maxModelStorageGb": 100})

    # When
    data = _quota(monkeypatch, builder, deploy)

    # Then a stopping deployment holds a slot; a stopped or deleted one does not
    assert data["releases"] == {"used": 4, "limit": 10, "limitKnown": True}
    assert data["activeDeployments"] == {"used": 2, "limit": 3, "limitKnown": True}
    assert data["builds"] == {"used": 2, "limit": 50, "limitKnown": True}
    assert "Builds you can see" in data["note"]


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(urllib.error.URLError("down"), id="unreachable"),
        pytest.param(DeployAPIError("deploy_payment_required", "no credit", status=402), id="refused"),
    ],
)
def test_quota_still_counts_when_the_limits_cannot_be_read(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    # When
    data = _quota(monkeypatch, Builder(failure), Deploy(failure))

    # Then
    assert data["releases"] == {"used": 4, "limit": None, "limitKnown": False}
    assert data["activeDeployments"] == {"used": 2, "limit": None, "limitKnown": False}
    assert data["builds"] == {"used": 2, "limit": None, "limitKnown": False}


def test_a_cap_the_limits_leave_out_is_no_limit_rather_than_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the builder applies no release or build cap
    data = _quota(monkeypatch, Builder({"maxRunningBuilds": 5}), Deploy({"maxActiveDeployments": 3}))

    # Then
    assert data["releases"] == {"used": 4, "limit": None, "limitKnown": True}
