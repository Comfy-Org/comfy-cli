"""Every deploy command finding a deployment by its name, or as `<build>/<name>`."""

from __future__ import annotations

import importlib
import json
import urllib.error

import pytest
from deploy_up_support import FakeBuilder, FakeDeploy, deployment, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_resolve import (
    AmbiguousBuildError,
    BuildNotFoundError,
    DeploymentNameNotFoundError,
    NameOutsideBuildError,
    UnrelatedDeploymentError,
    deployment_id_for,
)

_RELEASES = [
    {"id": "release-4", "buildId": "build-1", "version": 4, "deployable": True},
    {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
]
_BUILDS = [{"id": "build-1", "name": "flux-pipeline"}, {"id": "build-2", "name": "sdxl"}]


def _live(deployment_id: str, name: str | None, *, release_id: str = "release-4", **changes) -> JsonObject:
    row = deployment(deployment_id, release_id=release_id, name=name, **changes)
    row["revision"] = 3
    return row


class _WorkspaceBuilder(FakeBuilder):
    """comfy-builder outside enterprise: the Build list holds only the caller's own
    Builds, while any Build of the workspace serves its releases."""

    def __init__(self, workspace: set[str], builds: list[JsonObject] | None = None) -> None:
        sdxl = {"id": "release-9", "buildId": "build-2", "version": 1, "deployable": True}
        super().__init__([*_RELEASES, sdxl], builds or [{"id": "build-1", "name": "flux-pipeline"}])
        self.workspace = workspace

    def list_releases(self, build_id: str) -> list[JsonObject]:
        if build_id not in self.workspace:
            self.calls.append(("list_releases", build_id))
            raise urllib.error.HTTPError(f"https://builder/v1/builds/{build_id}/releases", 404, "Not Found", {}, None)
        return [release for release in super().list_releases(build_id) if release["buildId"] == build_id]


def _staging_and_production() -> FakeDeploy:
    return FakeDeploy([_live("dep-a1", "staging", release_id="release-5"), _live("dep-b2", "production")])


def _envelope(result) -> JsonObject:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _invoke(monkeypatch, client: FakeDeploy, *args: str, builder: FakeBuilder | None = None):
    clients = (builder or FakeBuilder(_RELEASES, _BUILDS), client)
    for module in ("comfy_cli.command.deploy", "comfy_cli.command.deploy_read"):
        monkeypatch.setattr(importlib.import_module(module), "_command_clients", lambda: clients)
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy"), "_sleep", lambda _: None)
    return CliRunner().invoke(app, ["--json", "deploy", *args], env={"COLUMNS": "400"})


@pytest.mark.parametrize("value", ["dep-0f3c9a7e-1b2d-4c5e-8f9a-0b1c2d3e4f5a", "dep_0f3c9a7e"])
def test_an_id_is_passed_through_without_a_lookup(value: str) -> None:
    # Given a service that lists nothing
    builder = FakeBuilder(_RELEASES, _BUILDS)

    # When
    resolved = deployment_id_for(builder, FakeDeploy(), value)

    # Then comfy-deploy refuses a name starting dep-, so the value is an id
    assert resolved == value
    assert builder.calls == []


def test_a_name_is_found_among_the_live_deployments_of_the_folders_build(tmp_path) -> None:
    # Given
    spec = write_spec(tmp_path)

    # When
    resolved = deployment_id_for(FakeBuilder(_RELEASES), _staging_and_production(), "staging", path=str(spec))

    # Then
    assert resolved == "dep-a1"


def test_a_deleted_deployment_never_answers_to_its_old_name() -> None:
    # Given staging was deleted and a new deployment took the name
    client = FakeDeploy(
        [
            _live("dep-old", "staging", deleted_at="2026-08-23T12:30:00Z"),
            _live("dep-new", "staging", release_id="release-5"),
        ]
    )

    # When
    resolved = deployment_id_for(FakeBuilder(_RELEASES), client, "staging", build_id="build-1")

    # Then
    assert resolved == "dep-new"


def test_a_deployment_of_another_build_never_answers_to_the_name() -> None:
    # Given build-2's staging, on a release build-1 does not list
    client = FakeDeploy([_live("dep-a1", "staging", release_id="release-9")])

    # When / Then
    with pytest.raises(DeploymentNameNotFoundError):
        deployment_id_for(FakeBuilder(_RELEASES), client, "staging", build_id="build-1")


@pytest.mark.parametrize("build", ["flux-pipeline", "build-1"])
def test_build_slash_name_finds_the_build_by_its_name_or_id(build: str) -> None:
    # When
    resolved = deployment_id_for(FakeBuilder(_RELEASES, _BUILDS), _staging_and_production(), f"{build}/staging")

    # Then
    assert resolved == "dep-a1"


@pytest.mark.parametrize("build", ["my build", "a?b", "x#y", "..", ""])
def test_an_unknown_build_is_refused_by_the_value_typed(build: str) -> None:
    # Given a builder that records the Build ids it is asked for
    builder = _WorkspaceBuilder({"build-1"})

    # When / Then
    with pytest.raises(BuildNotFoundError) as caught:
        deployment_id_for(builder, _staging_and_production(), f"{build}/staging")
    assert caught.value.details == {"build": build}
    # and an empty or dot-only one never reaches comfy-builder
    asked = [value for call, value in builder.calls if call == "list_releases"]
    assert asked == ([] if not build.strip(".") else [build])


def test_the_same_name_in_two_builds_answers_within_the_build_named() -> None:
    # Given a staging in flux-pipeline and another in sdxl
    client = FakeDeploy([*_staging_and_production().rows.values(), _live("dep-x9", "staging", release_id="release-9")])
    builder = _WorkspaceBuilder({"build-1", "build-2"}, _BUILDS)

    # When
    flux = deployment_id_for(builder, client, "flux-pipeline/staging")
    sdxl = deployment_id_for(builder, client, "sdxl/staging")

    # Then
    assert (flux, sdxl) == ("dep-a1", "dep-x9")


def test_build_slash_name_refuses_a_build_name_two_builds_share() -> None:
    # Given
    builds = [{"id": "build-1", "name": "flux-pipeline"}, {"id": "build-7", "name": "flux-pipeline"}]

    # When / Then
    with pytest.raises(AmbiguousBuildError) as caught:
        deployment_id_for(FakeBuilder(_RELEASES, builds), _staging_and_production(), "flux-pipeline/staging")
    assert caught.value.code == "deploy_ambiguous_build"
    assert caught.value.details["buildIds"] == ["build-1", "build-7"]


def test_build_slash_name_finds_a_teammates_build_by_its_id() -> None:
    # Given build-2, a teammate's, which the caller's Build list leaves out
    builder = _WorkspaceBuilder({"build-1", "build-2"})
    client = FakeDeploy([*_staging_and_production().rows.values(), _live("dep-x9", "staging", release_id="release-9")])

    # When
    resolved = deployment_id_for(builder, client, "build-2/staging")

    # Then
    assert resolved == "dep-x9"


def test_build_slash_name_refuses_a_build_nobody_has() -> None:
    # When / Then
    with pytest.raises(BuildNotFoundError) as caught:
        deployment_id_for(_WorkspaceBuilder({"build-1"}), _staging_and_production(), "wan-video/staging")
    assert caught.value.code == "deploy_build_not_found"
    assert caught.value.details == {"build": "wan-video"}


def test_a_name_none_holds_is_refused_with_the_names_held() -> None:
    # Given staging, production, one deployment from before names, and one
    # whose name is malformed, which reads as none as ls and status read it
    client = FakeDeploy([*_staging_and_production().rows.values(), _live("dep-c3", None), _live("dep-d4", "")])

    # When / Then
    with pytest.raises(DeploymentNameNotFoundError) as caught:
        deployment_id_for(FakeBuilder(_RELEASES), client, "canary", build_id="build-1")
    error = caught.value
    assert error.code == "deploy_name_not_found"
    assert error.details == {"buildId": "build-1", "name": "canary", "names": ["production", "staging"]}
    assert "production, staging" in error.hint


def test_deployments_with_no_name_field_are_unnamed_and_the_refusal_asks_for_the_id() -> None:
    # Given rows with no name field: comfy-deploy omits an unset name, and one
    # that predates names sends none at all
    client = FakeDeploy([deployment("dep-a1", release_id="release-5")])

    # When / Then
    with pytest.raises(DeploymentNameNotFoundError) as caught:
        deployment_id_for(FakeBuilder(_RELEASES), client, "staging", build_id="build-1")
    assert caught.value.details["names"] == []
    assert "predate names" in caught.value.hint
    assert "comfy deploy ls" in caught.value.hint


def test_a_build_with_no_live_deployment_says_so() -> None:
    # When / Then
    with pytest.raises(DeploymentNameNotFoundError) as caught:
        deployment_id_for(FakeBuilder(_RELEASES), FakeDeploy(), "staging", build_id="build-1")
    assert str(caught.value) == "Build build-1 has no live deployment, so none is named staging"
    assert "drop `--deployment`" in caught.value.hint


def test_an_id_after_the_build_is_one_of_its_live_deployments() -> None:
    # When
    resolved = deployment_id_for(FakeBuilder(_RELEASES, _BUILDS), _staging_and_production(), "flux-pipeline/dep-b2")

    # Then
    assert resolved == "dep-b2"


def test_an_id_after_the_build_that_is_not_its_deployment_is_refused() -> None:
    # Given sdxl's deployment, on a release flux-pipeline does not list
    client = FakeDeploy([*_staging_and_production().rows.values(), _live("dep-x9", "staging", release_id="release-9")])

    # When / Then the Build named is held to, as it is for a name
    with pytest.raises(UnrelatedDeploymentError) as caught:
        deployment_id_for(FakeBuilder(_RELEASES, _BUILDS), client, "flux-pipeline/dep-x9")
    assert caught.value.details["candidateIds"] == ["dep-a1", "dep-b2"]


def test_a_bare_name_outside_a_builds_folder_points_to_build_slash_name(tmp_path, monkeypatch) -> None:
    # Given a folder with no Build in it
    monkeypatch.chdir(tmp_path)

    # When / Then
    with pytest.raises(NameOutsideBuildError) as caught:
        deployment_id_for(FakeBuilder(_RELEASES), _staging_and_production(), "staging")
    assert caught.value.code == "deploy_build_not_found"
    assert "<build>/staging" in caught.value.hint


def test_a_bare_name_with_a_path_holding_no_build_names_the_path(tmp_path) -> None:
    # Given a PATH, mistyped
    missing = tmp_path / "flux-pipline"

    # When / Then the path is the mistake, as it is without --deployment
    with pytest.raises(BuildSpecNotFoundError):
        deployment_id_for(FakeBuilder(_RELEASES), _staging_and_production(), "staging", path=str(missing))


def test_show_reads_the_deployment_named(tmp_path, monkeypatch) -> None:
    # Given
    client = _staging_and_production()

    # When
    result = _invoke(monkeypatch, client, "show", str(write_spec(tmp_path)), "--deployment", "staging")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["id"] == "dep-a1"


def test_show_finds_build_slash_name_outside_any_folder(tmp_path, monkeypatch) -> None:
    # Given
    monkeypatch.chdir(tmp_path)

    # When
    result = _invoke(monkeypatch, _staging_and_production(), "show", "--deployment", "flux-pipeline/staging")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["name"] == "staging"


def test_a_name_none_holds_exits_with_its_own_code(tmp_path, monkeypatch) -> None:
    # When
    result = _invoke(monkeypatch, _staging_and_production(), "show", str(write_spec(tmp_path)), "--deployment", "qa")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_name_not_found"


def test_run_reads_the_deployment_named(tmp_path, monkeypatch) -> None:
    # Given staging, still provisioning, beside a ready production
    client = FakeDeploy([_live("dep-a1", "staging", status="provisioning"), _live("dep-b2", "production")])
    client.target = None
    run_module = importlib.import_module("comfy_cli.command.deploy_run")
    monkeypatch.setattr(run_module, "_command_clients", lambda: (FakeBuilder(_RELEASES, _BUILDS), client))
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps({"1": {"class_type": "Test", "inputs": {"text": "hello"}}}), encoding="utf-8")

    # When
    args = [
        "--json",
        "deploy",
        "run",
        str(write_spec(tmp_path)),
        "--deployment",
        "staging",
        "--workflow",
        str(workflow),
    ]
    result = CliRunner().invoke(app, args, env={"COLUMNS": "400"})

    # Then run read staging, and stopped because it is not ready
    assert result.exit_code == 1
    assert _envelope(result)["error"]["details"] == {"deployment_id": "dep-a1", "status": "provisioning"}


def test_promote_takes_two_names(tmp_path, monkeypatch) -> None:
    # Given a Build's folder holding staging on v5 and production on v4
    write_spec(tmp_path).rename(tmp_path / "comfy-build.yaml")
    monkeypatch.chdir(tmp_path)
    client = _staging_and_production()

    # When
    result = _invoke(monkeypatch, client, "promote", "staging", "production")

    # Then production moved onto staging's release by id
    assert result.exit_code == 0, result.stderr
    assert client.promote_calls == [("dep-b2", 3, "dep-a1")]


def test_promote_refuses_a_bare_name_outside_a_builds_folder(tmp_path, monkeypatch) -> None:
    # Given
    monkeypatch.chdir(tmp_path)
    client = _staging_and_production()

    # When
    result = _invoke(monkeypatch, client, "promote", "staging", "flux-pipeline/production")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_build_not_found"
    assert "<build>/staging" in error["hint"]
    assert client.promote_calls == []


def test_up_moves_the_deployment_named(tmp_path, monkeypatch) -> None:
    # Given staging on v4 and production on v5
    client = FakeDeploy([_live("dep-a1", "staging"), _live("dep-b2", "production", release_id="release-5")])
    spec = str(write_spec(tmp_path))

    # When
    result = _invoke(monkeypatch, client, "up", spec, "--release", "release-5", "--deployment", "staging")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == [("dep-a1", 3, "release-5")]


def test_rollback_moves_the_deployment_named(tmp_path, monkeypatch) -> None:
    # Given staging, which ran v4 then v5
    client = _staging_and_production()
    client.rows["dep-a1"]["revision"] = 2
    client.revisions["dep-a1"] = [
        {
            "revision": 1,
            "releaseId": "release-4",
            "kind": "create",
            "createdBy": "user-1",
            "createdAt": "2026-08-23T12:00:00Z",
        },
        {
            "revision": 2,
            "releaseId": "release-5",
            "kind": "update",
            "createdBy": "user-1",
            "createdAt": "2026-08-23T13:00:00Z",
        },
    ]

    # When
    result = _invoke(monkeypatch, client, "rollback", str(write_spec(tmp_path)), "--deployment", "staging")

    # Then
    assert result.exit_code == 0, result.stderr
    assert [call[0] for call in client.rollback_calls] == ["dep-a1"]


def test_the_deployment_option_says_it_takes_a_name() -> None:
    # Given every command's --deployment, which they all share
    command = importlib.import_module("typer.main").get_command(importlib.import_module("comfy_cli.command.deploy").app)
    option = next(param for param in command.commands["show"].params if "--deployment" in param.opts)

    # Then
    assert "name or id" in (option.help or "")
