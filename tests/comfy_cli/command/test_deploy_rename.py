"""Naming a deployment: `up --name` on a create, `comfy deploy rename`, and the
refusals that list a Build's deployments by name."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import jsonschema
import pytest
from deploy_up_support import FakeBuilder, FakeDeploy, deployment, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError

_RELEASES = [
    {"id": "release-4", "buildId": "build-1", "version": 4, "deployable": True},
    {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
]
_COMPUTE = ("--gpu", "l4", "--region", "US-MO-2")


def _schema(name: str) -> dict:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _live(deployment_id: str, name: str, *, release_id: str = "release-4", **changes) -> JsonObject:
    row = deployment(deployment_id, release_id=release_id, name=name, **changes)
    row["revision"] = 3
    return row


def _invoke(tmp_path, monkeypatch, client: FakeDeploy, *args: str, output: str = "--json"):
    clients = (FakeBuilder(_RELEASES), client)
    for module in ("comfy_cli.command.deploy", "comfy_cli.command.deploy_read", "comfy_cli.command.deploy_rename"):
        monkeypatch.setattr(importlib.import_module(module), "_command_clients", lambda: clients)
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy"), "_sleep", lambda _: None)
    write_spec(tmp_path).rename(tmp_path / "comfy-build.yaml")
    monkeypatch.chdir(tmp_path)
    return CliRunner().invoke(app, [output, "deploy", *args], env={"COLUMNS": "400"})


def test_a_first_up_sends_no_name_and_reports_the_one_comfy_deploy_gave(tmp_path, monkeypatch) -> None:
    # Given a Build with no deployment
    client = FakeDeploy(serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "up", "--release", "release-5", *_COMPUTE, "--no-watch")

    # Then comfy-deploy picks the name, and the CLI reports what it answered
    assert result.exit_code == 0, result.stderr
    assert client.create_names == [None]
    data = _envelope(result)["data"]
    assert data["deployment"]["name"] == client.rows["dep-1"]["name"]
    jsonschema.Draft202012Validator(_schema("deploy_up.json")).validate(data)


def test_a_named_create_sends_the_key_its_name_joins(tmp_path, monkeypatch) -> None:
    # Given the same Build state twice
    named, unnamed = FakeDeploy(serves_names=True), FakeDeploy(serves_names=True)

    # When one create names its deployment and the other does not
    base = ("up", "--release", "release-5", "--create", *_COMPUTE, "--no-watch")
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _invoke(tmp_path / "a", monkeypatch, named, *base, "--name", "staging")
    _invoke(tmp_path / "b", monkeypatch, unnamed, *base)

    # Then
    module = importlib.import_module("comfy_cli.command.deploy_up")
    assert unnamed.create_keys == [module._idempotency_key("build-1", "release-5", 0)]
    assert named.create_keys == [module._idempotency_key("build-1", "release-5", 0, name="staging")]


def test_up_create_sends_the_name_given(tmp_path, monkeypatch) -> None:
    # Given production on v4
    client = FakeDeploy([_live("dep-a1", "production")], serves_names=True)

    # When
    args = ("up", "--release", "release-5", "--create", "--name", "staging", *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.create_names == ["staging"]
    assert _envelope(result)["data"]["deployment"]["name"] == "staging"


def test_a_name_joins_the_create_key_so_another_name_is_another_create(tmp_path, monkeypatch) -> None:
    # Given
    module = importlib.import_module("comfy_cli.command.deploy_up")

    # Then a create with no name keeps the key it always had, which an older
    # comfy-cli's retry still sends
    assert module._idempotency_key("build-1", "release-5", 0) == "4680af49-8785-5442-b360-47d47544aee2"
    assert module._idempotency_key("build-1", "release-5", 0, 2) == "bc276a12-c084-5cfb-aecc-1fa8dcad7222"
    assert module._idempotency_key("build-1", "release-5", 0, name="staging") != module._idempotency_key(
        "build-1", "release-5", 0
    )
    assert module._idempotency_key("build-1", "release-5", 0, name="2") != module._idempotency_key(
        "build-1", "release-5", 0, 2
    )


def test_up_names_the_name_comfy_deploy_says_is_taken(tmp_path, monkeypatch) -> None:
    # Given production on v4
    client = FakeDeploy([_live("dep-a1", "production")], serves_names=True)

    # When
    args = ("up", "--release", "release-5", "--create", "--name", "production", *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_name_taken"
    assert "production" in error["message"]
    assert error["details"]["name"] == "production"


@pytest.mark.parametrize("name", ["Bad_Name", "dep-staging", "-staging", "staging-", "a" * 41, ""])
def test_a_name_outside_the_rule_is_refused_before_any_call(tmp_path, monkeypatch, name: str) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "production")], serves_names=True)

    # When
    args = ("up", "--release", "release-5", "--create", "--name", name, *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_invalid_name"
    assert "1 to 40" in error["message"]
    assert client.create_names == []


def test_a_create_that_loses_the_default_name_to_a_concurrent_one_is_sent_again(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy whose concurrent creates take the default name twice
    client = FakeDeploy(serves_names=True, name_races=2)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "up", "--release", "release-5", *_COMPUTE, "--no-watch")

    # Then the third send, under the same key, creates it
    assert result.exit_code == 0, result.stderr
    assert len(set(client.create_keys)) == 1
    assert len(client.create_keys) == 3
    assert _envelope(result)["data"]["deployment"]["name"] == "production"


def test_a_create_that_keeps_losing_the_default_name_says_to_run_it_again(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy(serves_names=True, name_races=5)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "up", "--release", "release-5", *_COMPUTE, "--no-watch")

    # Then nothing was created
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_conflict"
    assert "run it again" in error["hint"]
    assert client.rows == {}


def test_up_refuses_a_name_when_it_would_move_an_existing_deployment(tmp_path, monkeypatch) -> None:
    # Given production on v4, which a bare `up` on v5 would move
    client = FakeDeploy([_live("dep-a1", "production")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "up", "--release", "release-5", "--name", "staging", "--no-watch")

    # Then nothing moved, and the refusal points to the two ways that work
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_bad_request"
    assert "--create" in error["hint"]
    assert "rename" in error["hint"]
    assert client.move_calls == []


def test_up_run_again_with_the_same_name_reconciles_the_deployment_it_named(tmp_path, monkeypatch) -> None:
    # Given staging, which an earlier `up --name staging` created on v5
    client = FakeDeploy([_live("dep-a1", "staging", release_id="release-5")], serves_names=True)

    # When the same command runs again
    result = _invoke(tmp_path, monkeypatch, client, "up", "--release", "release-5", "--name", "staging", "--no-watch")

    # Then it finds staging rather than refusing or creating
    assert result.exit_code == 0, result.stderr
    assert client.create_names == []
    assert _envelope(result)["data"]["deployment"]["id"] == "dep-a1"


def test_up_names_the_deployment_it_creates_where_updates_are_off(tmp_path, monkeypatch) -> None:
    # Given production on v4, in a workspace outside the rollout of deployment
    # updates, where `up` on v5 adds a deployment rather than moving one
    row = deployment("dep-a1", release_id="release-4", name="production")
    client = FakeDeploy([row], serves_names=True)

    # When
    args = ("up", "--release", "release-5", "--name", "staging", *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.create_names == ["staging"]


def test_a_named_create_beside_a_live_deployment_holding_the_name_is_refused_before_the_call(
    tmp_path, monkeypatch
) -> None:
    # Given staging, created on v5 and since moved to v4, so nothing is live on v5
    # and a create would send the key staging's first create sent
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "staging")], serves_names=True)

    # When
    args = ("up", "--release", "release-5", "--create", "--name", "staging", *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_name_taken"
    assert client.create_keys == []


def test_a_named_create_after_its_namesake_was_renamed_and_moved_creates_anew(tmp_path, monkeypatch) -> None:
    # Given staging, created on v5, since renamed canary and moved to v4
    client = FakeDeploy(serves_names=True)
    args = ("up", "--release", "release-5", "--create", "--name", "staging", *_COMPUTE, "--no-watch")
    _invoke(tmp_path, monkeypatch, client, *args)
    client.rows["dep-1"].update({"name": "canary", "releaseId": "release-4"})

    # When the same create runs again
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then it sends a key of its own, so comfy-deploy makes a new deployment
    assert result.exit_code == 0, result.stderr
    assert len(set(client.create_keys)) == 2
    assert _envelope(result)["data"]["deployment"]["id"] == "dep-2"


@pytest.mark.parametrize("create", [(), ("--create",)])
def test_a_named_up_rerun_while_the_first_creates_gets_the_first_runs_deployment(tmp_path, monkeypatch, create) -> None:
    # Given the key a first `up --name staging` sends on a Build with no deployment
    args = ("up", "--release", "release-5", *create, "--name", "staging", *_COMPUTE, "--no-watch")
    first = FakeDeploy(serves_names=True)
    (tmp_path / "first").mkdir()
    _invoke(tmp_path / "first", monkeypatch, first, *args)
    client = FakeDeploy(serves_names=True)
    list_all = client.list_all_deployments
    lists = []

    def first_run_lands_between_lists() -> list[JsonObject]:
        lists.append(1)
        if len(lists) == 2:
            client.create_deployment("release-5", {}, idempotency_key=first.create_keys[0], name="staging")
        return list_all()

    client.list_all_deployments = first_run_lands_between_lists

    # When a rerun reads the Build before that create lands and creates after it
    (tmp_path / "rerun").mkdir()
    result = _invoke(tmp_path / "rerun", monkeypatch, client, *args)

    # Then the rerun sends the same key and gets the first run's deployment back
    assert result.exit_code == 0, result.stderr
    assert set(client.create_keys) == {first.create_keys[0]}
    assert _envelope(result)["data"]["deployment"]["id"] == "dep-1"


def test_up_refuses_a_name_for_the_deployment_already_on_the_release(tmp_path, monkeypatch) -> None:
    # Given production, already on v5
    client = FakeDeploy([_live("dep-a1", "production", release_id="release-5")], serves_names=True)

    # When
    args = ("up", "--release", "release-5", "--deployment", "production", "--name", "staging", "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_bad_request"
    assert "rename" in _envelope(result)["error"]["hint"]


def test_up_with_a_new_name_beside_two_deployments_points_to_create_not_a_pick(tmp_path, monkeypatch) -> None:
    # Given production and canary, neither named staging
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "canary")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "up", "--release", "release-5", "--name", "staging", "--no-watch")

    # Then the refusal is about the name, since picking one of them would refuse again
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_bad_request"
    assert "--create" in error["hint"]


def test_a_create_answered_without_a_name_is_reported_by_its_id(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy that predates names
    client = FakeDeploy()

    # When
    args = ("up", "--release", "release-5", "--create", "--name", "staging", *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then the deployment it bills for is still reported, and the name is said to be lost
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["deployment"] == {"id": "dep-1", "name": None, "status": "ready", "created": True}
    assert "dep-1" in result.stderr
    assert "staging" in result.stderr


def test_a_create_answered_under_another_name_is_reported_with_a_warning(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy that creates the deployment under a name of its own
    client = FakeDeploy(serves_names=True)
    create = client.create_deployment

    def create_unnamed(*args, **kwargs) -> JsonObject:
        created = create(*args, **kwargs)
        client.rows[created["id"]]["name"] = "deployment-2"
        return {**created, "name": "deployment-2"}

    client.create_deployment = create_unnamed

    # When
    args = ("up", "--release", "release-5", "--create", "--name", "staging", *_COMPUTE, "--no-watch")
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then the deployment it bills for is reported, and the name it took is said
    assert result.exit_code == 0, result.stderr
    assert "named deployment-2, not staging" in " ".join(result.stderr.split())


@pytest.mark.parametrize("command", ["up", "rollback"])
def test_a_bare_command_refusing_two_deployments_lists_each_by_name(tmp_path, monkeypatch, command: str) -> None:
    # Given production on v4 and canary on v5, both running
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "canary", release_id="release-5")])
    args = ("up", "--release", "release-5", "--no-watch") if command == "up" else ("rollback", "--no-watch")

    # When
    result = _invoke(tmp_path, monkeypatch, client, *args)

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_ambiguous_deployment"
    assert error["details"]["candidateIds"] == ["dep-a1", "dep-b2"]
    assert error["details"]["candidates"] == [
        {"id": "dep-a1", "name": "production", "release": "v4", "status": "ready"},
        {"id": "dep-b2", "name": "canary", "release": "v5", "status": "ready"},
    ]
    assert "production (v4, ready)" in error["message"]
    assert "canary (v5, ready)" in error["message"]


def test_rename_changes_the_deployments_name(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "staging")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "--deployment", "staging", "canary")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.rename_calls == [("dep-b2", "canary")]
    envelope = _envelope(result)
    assert envelope["changed"] is True
    assert envelope["data"] == {"deployment": {"id": "dep-b2", "name": "canary"}, "previousName": "staging"}
    jsonschema.Draft202012Validator(_schema("deploy_rename.json")).validate(envelope["data"])


def test_a_renamed_deployment_answers_to_its_new_name(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "staging")], serves_names=True)
    _invoke(tmp_path, monkeypatch, client, "rename", "--deployment", "staging", "canary")

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "show", "--deployment", "canary"])

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["id"] == "dep-b2"


def test_rename_reads_two_arguments_as_the_folder_then_the_name(tmp_path, monkeypatch) -> None:
    # Given a Build whose only deployment is production
    client = FakeDeploy([_live("dep-a1", "production")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", str(tmp_path), "canary")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.rename_calls == [("dep-a1", "canary")]


def test_rename_without_a_deployment_refuses_two_listing_them_by_name(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "staging")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "canary")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_ambiguous_deployment"
    assert [row["name"] for row in error["details"]["candidates"]] == ["production", "staging"]
    assert client.rename_calls == []


def test_rename_refuses_a_name_outside_the_rule_and_keeps_the_old_one(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "staging")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "--deployment", "staging", "Bad_Name")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_invalid_name"
    assert client.rows["dep-a1"]["name"] == "staging"


def test_rename_to_a_name_another_deployment_holds_names_it(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "production"), _live("dep-b2", "staging")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "--deployment", "staging", "production")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_name_taken"
    assert error["details"]["name"] == "production"


def test_renaming_to_the_name_held_says_nothing_changed(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "staging")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "staging", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    assert "already named staging" in result.stdout
    assert "was staging" not in result.stdout


def test_a_rename_answered_without_the_name_is_confirmed_by_a_read(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy that renames and answers with no body
    client = FakeDeploy([_live("dep-a1", "staging")], serves_names=True)
    renamed = client.rename_deployment
    client.rename_deployment = lambda deployment_id, name: (renamed(deployment_id, name), {})[1]

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "canary")

    # Then the read shows the new name, so the rename is reported
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["deployment"] == {"id": "dep-a1", "name": "canary"}


def test_a_rename_answered_without_the_name_and_not_applied_says_names_are_unavailable(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy that answers 200 and keeps the old name
    client = FakeDeploy([_live("dep-a1", "staging")], serves_names=True)
    client.rename_deployment = lambda deployment_id, name: {"id": deployment_id}

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "canary")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_names_unavailable"


def test_a_rename_refused_for_another_reason_keeps_comfy_deploys_words(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy serving names that refuses this rename on other grounds
    client = FakeDeploy([_live("dep-a1", "staging")], serves_names=True)

    def refuse(deployment_id: str, name: str) -> JsonObject:
        raise DeployAPIError(
            "deploy_conflict",
            "a deployment with no build cannot take a name",
            status=409,
            details={"server_code": "CONFLICT"},
        )

    client.rename_deployment = refuse

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "canary")

    # Then
    assert result.exit_code == 1
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_conflict"
    assert error["message"] == "a deployment with no build cannot take a name"


def test_rename_against_a_comfy_deploy_without_names_says_so(tmp_path, monkeypatch) -> None:
    # Given a comfy-deploy that predates names, so the id is the only handle
    client = FakeDeploy([_live("dep-a1", "staging")])

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "--deployment", "dep-a1", "canary")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_names_unavailable"


def test_rename_with_a_third_argument_is_a_usage_error_like_a_missing_one(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-a1", "production")], serves_names=True)

    # When
    result = _invoke(tmp_path, monkeypatch, client, "rename", "a", "b", "staging")

    # Then
    assert result.exit_code == 2
    assert _envelope(result)["error"]["code"] == "usage_error"
    assert client.rename_calls == []
