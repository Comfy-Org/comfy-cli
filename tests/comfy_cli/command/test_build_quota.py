from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import jsonschema
import pytest
from typer.testing import CliRunner

from comfy_cli.cmdline import app as cli_app
from comfy_cli.command import build, build_quota, deploy_runtime
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
        self.target = SimpleNamespace(base_url="https://control.test")

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


def _quota(monkeypatch: pytest.MonkeyPatch, builder: Builder, deploy: Deploy, *args: str) -> dict:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: builder)
    monkeypatch.setattr(deploy_runtime, "deploy_client", lambda: deploy)
    result = CliRunner().invoke(
        cli_app, ["build", "quota", *args], env={"AI_AGENT": "1", "COMFY_OUTPUT": "json", "NO_COLOR": "1"}
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
    assert data["activeDeployments"] == {"used": 2, "limit": 3, "limitKnown": True, "exact": False}
    assert "Builds you can see" in data["note"]


def test_quota_reports_only_the_release_and_active_deployment_caps(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a builder that also sends a build cap
    builder = Builder({"maxBuilds": 50, "maxRunningBuilds": 5, "maxTotalReleases": 10})

    # When
    data = _quota(monkeypatch, builder, Deploy({"maxActiveDeployments": 3}))

    # Then
    assert set(data) == {"releases", "activeDeployments", "note"}


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param(urllib.error.URLError("down"), id="unreachable"),
        pytest.param(DeployAPIError("deploy_payment_required", "no credit", status=402), id="refused"),
    ],
)
def test_quota_still_counts_when_the_limits_cannot_be_read(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    # When
    data = _quota(monkeypatch, Builder(failure), Deploy(failure))

    # Then
    assert data["releases"] == {"used": 4, "limit": None, "limitKnown": False}
    assert data["activeDeployments"] == {"used": 2, "limit": None, "limitKnown": False, "exact": False}


def test_a_cap_the_limits_leave_out_is_no_limit_rather_than_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    # When the builder applies no release or build cap
    data = _quota(monkeypatch, Builder({"maxRunningBuilds": 5}), Deploy({"maxActiveDeployments": 3}))

    # Then
    assert data["releases"] == {"used": 4, "limit": None, "limitKnown": True}


@pytest.mark.parametrize(
    "body",
    [pytest.param({}, id="empty-body"), pytest.param({"unexpected": True}, id="other-shape")],
)
def test_a_limits_body_without_its_expected_shape_is_unknown_rather_than_no_limit(
    monkeypatch: pytest.MonkeyPatch, body: JsonObject
) -> None:
    # Given limits reads that succeed but carry neither maxRunningBuilds nor maxActiveDeployments,
    # as an empty or unparsable body reaches the command
    data = _quota(monkeypatch, Builder(body), Deploy(body))

    # Then no count claims the workspace is uncapped
    assert data["releases"] == {"used": 4, "limit": None, "limitKnown": False}
    assert data["activeDeployments"]["limit"] is None
    assert data["activeDeployments"]["limitKnown"] is False


def test_the_active_deployment_count_is_marked_a_lower_bound(monkeypatch: pytest.MonkeyPatch) -> None:
    """The deploy list shows each deployment's serving copy, while the limit counts
    every undeleted copy, so a stopped deployment whose replacement is queued holds
    a slot the list does not show."""
    # When
    data = _quota(monkeypatch, Builder({"maxRunningBuilds": 5}), Deploy({"maxActiveDeployments": 3}))

    # Then
    assert data["activeDeployments"] == {"used": 2, "limit": 3, "limitKnown": True, "exact": False}


class ManyBuilds(Builder):
    def list_builds(self) -> list[JsonObject]:
        return [{"id": f"build-{n}"} for n in range(build_quota.MAX_BUILDS + 1)]

    def list_releases(self, build_id: str) -> list[JsonObject]:
        return []


def _pretty_quota(monkeypatch: pytest.MonkeyPatch, builder: Builder, deploy: Deploy, *args: str) -> str:
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: builder)
    monkeypatch.setattr(deploy_runtime, "deploy_client", lambda: deploy)
    result = CliRunner().invoke(cli_app, ["build", "quota", *args], env={"COMFY_OUTPUT": "pretty", "NO_COLOR": "1"})
    assert result.exit_code == 0, result.stderr
    return result.stdout + result.stderr


def test_the_json_release_count_is_marked_truncated_past_the_builds_it_reads(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given one more visible Build than the command reads releases for
    builder = ManyBuilds({"maxRunningBuilds": 5, "maxTotalReleases": 10})

    # When
    data = _quota(monkeypatch, builder, Deploy({"maxActiveDeployments": 3}))

    # Then
    assert data["releases"] == {"used": 0, "limit": 10, "limitKnown": True, "truncated": True}


def test_pretty_output_says_when_the_release_count_skipped_builds(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given more visible Builds than the command reads releases for
    builder = ManyBuilds({"maxRunningBuilds": 5, "maxTotalReleases": 10})

    # When
    output = _pretty_quota(monkeypatch, builder, Deploy({"maxActiveDeployments": 3}))

    # Then the release line carries the truncation, and the active count reads as a lower bound
    release_line = next(line for line in output.splitlines() if line.strip().startswith("Releases:"))
    assert f"first {build_quota.MAX_BUILDS} Builds" in release_line
    active_line = next(line for line in output.splitlines() if line.strip().startswith("Active deployments:"))
    assert "at least 2 of 3" in active_line
    assert not any(line.strip().startswith("Builds:") for line in output.splitlines())


_OVERRIDES = [
    pytest.param(("--builder-url", "https://staging.test/builder"), {}, id="flag"),
    pytest.param((), {"COMFY_BUILDER_URL": "https://staging.test/builder"}, id="env"),
]


@pytest.mark.parametrize(("args", "env"), _OVERRIDES)
def test_an_overridden_builder_url_is_named_beside_where_deployments_were_read(
    monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], env: dict[str, str]
) -> None:
    # Given the builder pointed somewhere the deploy commands are not
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # When
    data = _quota(monkeypatch, Builder({"maxRunningBuilds": 5}), Deploy({"maxActiveDeployments": 3}), *args)

    # Then
    assert "https://control.test, the deploy commands' environment" in data["note"]
    assert "not the overridden builder URL" in data["note"]


@pytest.mark.parametrize(("args", "env"), _OVERRIDES)
def test_pretty_output_names_an_overridden_builder_url_on_its_own_line(
    monkeypatch: pytest.MonkeyPatch, args: tuple[str, ...], env: dict[str, str]
) -> None:
    # Given
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # When
    output = _pretty_quota(monkeypatch, Builder({"maxRunningBuilds": 5}), Deploy({"maxActiveDeployments": 3}), *args)

    # Then a note of its own, apart from the counting note (the terminal may wrap it)
    text = " ".join(output.split())
    assert "Note: deployments were read from https://control.test" in text
    assert "not the overridden builder URL." in text


def test_without_an_override_the_note_says_nothing_of_the_builder_url(monkeypatch: pytest.MonkeyPatch) -> None:
    # When
    data = _quota(monkeypatch, Builder({"maxRunningBuilds": 5}), Deploy({"maxActiveDeployments": 3}))

    # Then
    assert "builder URL" not in data["note"]


def test_quota_reads_deployments_with_the_deploy_commands_credential_not_the_builder_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a builder token handed in beside the Cloud token the deploy commands send
    monkeypatch.setattr(build, "_builder_client", lambda renderer, builder_url: Builder({"maxRunningBuilds": 5}))
    sent: list[tuple[str, str | None]] = []

    def deploy_transport(url: str, target, **_: object) -> tuple[int, JsonObject]:
        path = urlsplit(url).path
        sent.append((path, target.auth_token))
        return 200, ({"deployments": []} if path == "/v1/deployments" else {"maxActiveDeployments": 3})

    monkeypatch.setattr("comfy_cli.deploy_api.request_json", deploy_transport)

    # When
    result = CliRunner().invoke(
        cli_app,
        ["build", "quota"],
        env={
            "AI_AGENT": "1",
            "COMFY_OUTPUT": "json",
            "NO_COLOR": "1",
            "COMFY_BUILDER_TOKEN": "builder-jwt",
            "COMFY_CLOUD_AUTH_TOKEN": "cloud-jwt",
            "COMFY_DEPLOY_URL": "https://control.test",
        },
    )

    # Then
    assert result.exit_code == 0, result.stderr
    assert sent and {token for _, token in sent} == {"cloud-jwt"}


class TimedBuilds(Builder):
    """Five visible Builds, each releases read taking four seconds on ``clock``."""

    def __init__(self, clock: list[float]) -> None:
        super().__init__({"maxRunningBuilds": 5})
        self.clock = clock
        self.read: list[str] = []

    def list_builds(self) -> list[JsonObject]:
        return [{"id": f"build-{n}"} for n in range(5)]

    def list_releases(self, build_id: str) -> list[JsonObject]:
        self.read.append(build_id)
        self.clock[0] += 4
        return [{"id": f"{build_id}-r0", "createdAt": "2026-08-20T00:00:00Z"}]


class TimedDeploy(Deploy):
    def __init__(self, clock: list[float]) -> None:
        super().__init__({"maxActiveDeployments": 3})
        self.clock = clock

    def list_all_deployments(self) -> list[JsonObject]:
        self.clock[0] += 1
        return [{"id": "dep-1", "releaseId": "build-0-r0", "status": "stopped", "deletedAt": None}]


def test_the_reads_after_a_refusal_stop_at_one_deadline_and_return_what_was_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a deploy read of one second and Build reads of four each, on a fake clock
    clock = [0.0]
    builder = TimedBuilds(clock)
    monkeypatch.setattr(deploy_runtime, "deploy_client", lambda: TimedDeploy(clock))

    # When
    details = build_quota.release_limit_details(builder, now=lambda: clock[0])

    # Then the Builds begun before the deadline are listed, and the list says it is cut short
    assert build_quota.FOLLOW_UP_DEADLINE_SECONDS == 10
    assert builder.read == ["build-0", "build-1", "build-2"]
    assert [row["id"] for row in details["builds"]] == ["build-0", "build-1", "build-2"]
    assert details["builds"][0]["inUse"] is True
    assert details["buildsTruncated"] is True


def test_reads_inside_the_deadline_are_not_marked_truncated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Given a clock that never moves
    builder = TimedBuilds([0.0])
    monkeypatch.setattr(deploy_runtime, "deploy_client", lambda: TimedDeploy([0.0]))

    # When
    details = build_quota.release_limit_details(builder, now=lambda: 0.0)

    # Then
    assert len(details["builds"]) == 5
    assert "buildsTruncated" not in details


def test_a_failed_best_effort_read_returns_none_and_logs_it_at_debug(caplog: pytest.LogCaptureFixture) -> None:
    # Given a read that raises
    def broken() -> JsonObject:
        raise RuntimeError("limits broke")

    # When
    with caplog.at_level("DEBUG", logger=build_quota.__name__):
        result = build_quota.best_effort(broken)

    # Then
    assert result is None
    record = next(r for r in caplog.records if r.name == build_quota.__name__)
    assert record.levelname == "DEBUG"
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], RuntimeError)
