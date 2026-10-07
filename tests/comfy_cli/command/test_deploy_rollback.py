"""`comfy deploy rollback` and `comfy deploy history`."""

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
    {"id": f"release-{version}", "buildId": "build-1", "version": version, "deployable": True} for version in (1, 2, 3)
]


def _schema(name: str) -> dict:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _revision(number: int, release: int, kind: str, **extra) -> JsonObject:
    return {
        "revision": number,
        "releaseId": f"release-{release}",
        "kind": kind,
        "createdBy": "user-1",
        "createdAt": f"2026-10-0{number}T12:00:00Z",
        **extra,
    }


def _moved_through(*releases: int, deployment_id: str = "dep-prod", **client_options) -> FakeDeploy:
    """A deployment that ran each release in turn, one revision per release."""
    row = deployment(deployment_id, release_id=f"release-{releases[-1]}")
    row["revision"] = len(releases)
    client = FakeDeploy([row], **client_options)
    client.revisions[deployment_id] = [
        _revision(number, release, "create" if number == 1 else "update")
        for number, release in enumerate(releases, start=1)
    ]
    return client


class _ForgetfulBuilder(FakeBuilder):
    """A Builder that no longer finds a deleted release by id."""

    def get_release(self, release_id: str) -> JsonObject:
        if not any(release["id"] == release_id for release in self.releases):
            raise DeployAPIError("deploy_not_found", "release not found", status=404)
        return super().get_release(release_id)


def _invoke(monkeypatch, client: FakeDeploy, *args: str, output: str = "--json"):
    module = importlib.import_module("comfy_cli.command.deploy")
    clients = (_ForgetfulBuilder(_RELEASES), client)
    monkeypatch.setattr(module, "_command_clients", lambda: clients)
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy_read"), "_command_clients", lambda: clients)
    monkeypatch.setattr(module, "_sleep", lambda _: None)
    return CliRunner().invoke(app, [output, "deploy", *args], env={"COLUMNS": "400"})


def test_rollback_returns_the_deployment_to_the_release_before(monkeypatch) -> None:
    # Given a deployment that moved from v1 to v2
    client = _moved_through(1, 2)

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # Then it serves v1 again under the same id
    assert result.exit_code == 0, result.stderr
    assert client.rollback_calls == [("dep-prod", 2, None)]
    envelope = _envelope(result)
    assert envelope["changed"] is True
    data = envelope["data"]
    assert data["deployment"] == {"id": "dep-prod", "status": "ready", "revision": 3}
    assert data["release"] == {"id": "release-1", "version": 1}
    assert data["previousRelease"] == {"id": "release-2", "version": 2}
    assert data["waiting"] is False
    jsonschema.Draft202012Validator(_schema("deploy_rollback")).validate(data)


def test_a_second_rollback_undoes_the_first(monkeypatch) -> None:
    # Given a rollback from v2 to v1
    client = _moved_through(1, 2)
    _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # When it rolls back again
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # Then it is back on v2
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["release"] == {"id": "release-2", "version": 2}


def test_rollback_to_a_version_picks_the_latest_revision_that_ran_it(monkeypatch) -> None:
    # Given v1, v2, v3
    client = _moved_through(1, 2, 3)

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", "--to", "v1")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.rollback_calls == [("dep-prod", 3, 1)]
    assert _envelope(result)["data"]["release"] == {"id": "release-1", "version": 1}


def test_rollback_to_a_version_that_ran_twice_picks_the_later_revision(monkeypatch) -> None:
    # Given v1 at r1 and again at r3, then v2
    client = _moved_through(1, 2, 1, 2)

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", "--to", "v1")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.rollback_calls == [("dep-prod", 4, 3)]


def test_rollback_to_a_release_id_the_build_no_longer_lists(monkeypatch) -> None:
    # Given a deployment that once ran a release its Build no longer lists
    client = _moved_through(1, 2)
    client.revisions["dep-prod"][0]["releaseId"] = "release-gone"

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", "--to", "release-gone")

    # Then it returns to it, named by id
    assert result.exit_code == 0, result.stderr
    assert client.rollback_calls == [("dep-prod", 2, 1)]
    assert _envelope(result)["data"]["release"] == {"id": "release-gone"}


def test_rollback_to_a_version_the_build_does_not_list_refuses_without_a_move(monkeypatch) -> None:
    # Given
    client = _moved_through(1, 2, 3)

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", "--to", "v9")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_bad_request"
    assert error["message"] == "Build build-1 lists no release v9; it may have been deleted"
    assert client.rollback_calls == []


def test_rollback_to_a_release_never_run_refuses_without_a_move(monkeypatch) -> None:
    # Given a deployment that ran v1 then v2, never v3
    client = _moved_through(1, 2)

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", "--to", "v3")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_bad_request"
    assert "never ran release v3" in error["message"]
    assert client.rollback_calls == []


def test_a_deployment_with_one_revision_has_nothing_to_roll_back_to(monkeypatch) -> None:
    # Given
    client = _moved_through(1)

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_conflict"
    assert error["message"] == "deployment dep-prod has no earlier release to roll back to"


class _Refused(FakeDeploy):
    server_code = ""

    def rollback_deployment(self, deployment_id: str, base_revision: int, to_revision: int | None = None):
        raise DeployAPIError("deploy_conflict", "refused", status=409, details={"server_code": self.server_code})


@pytest.mark.parametrize(
    ("server_code", "message", "hint"),
    [
        ("STALE_REVISION", "changed after it was read, so the rollback was refused", "run the rollback again"),
        (
            "PENDING_UPDATE_EXISTS",
            "is waiting on an update, so the rollback was refused",
            "wait until `comfy deploy status --deployment dep-prod` shows no update",
        ),
        (
            "RELEASE_DELETED",
            "cannot go back to that release, because it was deleted",
            "run `comfy deploy history --deployment dep-prod`",
        ),
    ],
)
def test_a_refused_rollback_says_why_plainly(monkeypatch, server_code: str, message: str, hint: str) -> None:
    # Given a service that refuses the rollback
    row = deployment("dep-prod", release_id="release-2")
    row["revision"] = 2
    client = _Refused([row])
    client.server_code = server_code
    client.revisions["dep-prod"] = [_revision(1, 1, "create"), _revision(2, 2, "update")]

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_conflict"
    assert error["message"] == f"deployment dep-prod {message}"
    assert hint in error["hint"]
    assert error["details"]["server_code"] == server_code


def test_a_stopping_deployment_is_not_rolled_back(monkeypatch) -> None:
    # Given
    client = _moved_through(1, 2)
    client.rows["dep-prod"]["status"] = "stop_failed"

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_conflict"
    assert "`rollback` will not move it" in error["message"]
    assert client.rollback_calls == []


def test_rollback_outside_the_rollout_says_updates_are_off(monkeypatch) -> None:
    # Given
    client = FakeDeploy([deployment("dep-prod", release_id="release-2")])

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_updates_unavailable"
    assert client.rollback_calls == []


def test_a_waiting_rollback_is_followed_until_it_lands(monkeypatch) -> None:
    # Given v1's copy starting again, and landing on the next read
    landed = {"pendingUpdate": None, "releaseId": "release-1", "revision": 3}
    client = _moved_through(1, 2, move="pending", get_patches=[{}, landed])

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    assert "moves to release v1 once it is ready; release v2 serves until then" in result.stdout
    assert "now serves release v1 (was release v2)" in result.stdout


def test_a_waiting_rollback_not_followed_says_it_waits(monkeypatch) -> None:
    # Given v1's copy starting again
    client = _moved_through(1, 2, move="pending")

    # When
    result = _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod", "--no-watch")

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["waiting"] is True
    assert data["release"] == {"id": "release-1", "version": 1}
    jsonschema.Draft202012Validator(_schema("deploy_rollback")).validate(data)


def test_rollback_picks_the_builds_only_running_deployment(tmp_path, monkeypatch) -> None:
    # Given a Build whose one running deployment moved from v1 to v2
    client = _moved_through(1, 2)

    # When
    result = _invoke(monkeypatch, client, "rollback", str(write_spec(tmp_path)))

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.rollback_calls == [("dep-prod", 2, None)]


def test_rollback_refuses_to_pick_between_two_running_deployments(tmp_path, monkeypatch) -> None:
    # Given
    client = _moved_through(1, 2)
    client.rows["dep-other"] = deployment("dep-other", release_id="release-2")

    # When
    result = _invoke(monkeypatch, client, "rollback", str(write_spec(tmp_path)))

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_ambiguous_deployment"
    assert sorted(error["details"]["candidateIds"]) == ["dep-other", "dep-prod"]
    assert client.rollback_calls == []


def test_history_lists_each_revision_newest_first(monkeypatch) -> None:
    # Given v1, v2, then a rollback to v1
    client = _moved_through(1, 2)
    _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # When
    result = _invoke(monkeypatch, client, "history", "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    jsonschema.Draft202012Validator(_schema("deploy_history")).validate(data)
    assert [(row["revision"], row["releaseVersion"], row["kind"], row["current"]) for row in data["revisions"]] == [
        (3, 1, "rollback", True),
        (2, 2, "update", False),
        (1, 1, "create", False),
    ]
    assert data["revisions"][0]["fromRevision"] == 1


def test_history_marks_the_revision_the_deployment_serves(monkeypatch) -> None:
    # Given a revisions list that runs ahead of the deployment's own revision
    client = _moved_through(1, 2)
    client.rows["dep-prod"]["revision"] = 1

    # When
    result = _invoke(monkeypatch, client, "history", "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    assert [row["current"] for row in _envelope(result)["data"]["revisions"]] == [False, True]


def test_history_prints_the_current_revision_marked(monkeypatch) -> None:
    # Given
    client = _moved_through(1, 2)
    _invoke(monkeypatch, client, "rollback", "--deployment", "dep-prod")

    # When
    result = _invoke(monkeypatch, client, "history", "--deployment", "dep-prod", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    assert lines[0].startswith("* r3  v1  rollback to r1  user-1")
    assert lines[2].startswith("  r1  v1  create  user-1")


def test_history_outside_the_rollout_says_updates_are_off(monkeypatch) -> None:
    # Given
    client = FakeDeploy([deployment("dep-prod", release_id="release-2")])

    # When
    result = _invoke(monkeypatch, client, "history", "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_updates_unavailable"
