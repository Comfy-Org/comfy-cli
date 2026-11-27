"""`comfy deploy cancel`."""

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
    {"id": f"release-{version}", "buildId": "build-1", "version": version, "deployable": True} for version in (4, 5)
]


def _schema() -> dict:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / "deploy_cancel.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _waiting(kind: str = "update", **served: str | None) -> FakeDeploy:
    """A deployment serving v4 at revision 3, waiting on an update to v5."""
    row = deployment("dep-prod", release_id="release-4", **served)
    row["revision"] = 3
    row["pendingUpdate"] = {
        "releaseId": "release-5",
        "baseRevision": 3,
        "status": "provisioning",
        "since": "2026-10-07T12:00:00Z",
        "kind": kind,
    }
    return FakeDeploy([row])


def _invoke(monkeypatch, client: FakeDeploy, *args: str, output: str = "--json"):
    clients = (FakeBuilder(_RELEASES), client)
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy_cancel"), "_command_clients", lambda: clients)
    return CliRunner().invoke(app, [output, "deploy", "cancel", *args], env={"COLUMNS": "400"})


def test_a_cancel_ends_the_waiting_update_and_keeps_the_release(monkeypatch) -> None:
    # Given
    client = _waiting()

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["changed"] is True
    data = envelope["data"]
    assert data["release"] == {"id": "release-4", "version": 4}
    assert data["cancelledUpdate"] == {
        "release": {"id": "release-5", "version": 5},
        "baseRevision": 3,
        "kind": "update",
    }
    assert data["deployment"]["revision"] == 3
    jsonschema.Draft202012Validator(_schema()).validate(data)
    assert client.cancel_calls == ["dep-prod"]


class _NoRevisionRead(FakeDeploy):
    """A comfy-deploy whose rollout check fails on the read, so it leaves revision and pendingUpdate out."""

    def get_deployment(self, deployment_id: str) -> JsonObject:
        row = super().get_deployment(deployment_id)
        row.pop("revision", None)
        row.pop("pendingUpdate", None)
        return row


def test_a_read_without_a_revision_still_sends_the_cancel(monkeypatch) -> None:
    # Given an update waiting that the read leaves out
    client = _NoRevisionRead(list(_waiting().rows.values()))

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    # Then the service, not the read, says whether one waited
    assert result.exit_code == 0, result.stderr
    assert client.cancel_calls == ["dep-prod"]
    assert _envelope(result)["data"]["cancelledUpdate"]["release"] == {"id": "release-5", "version": 5}


def test_a_cancel_says_what_it_ended_and_what_still_serves(monkeypatch) -> None:
    # Given a named deployment waiting on a rollback
    client = _waiting(kind="rollback", name="production")

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    assert "Cancelled the rollback to release v5. production (dep-prod) keeps serving release v4." in result.stdout


def test_a_cancel_with_nothing_waiting_changes_nothing(monkeypatch) -> None:
    # Given a deployment with no update waiting
    row = deployment("dep-prod", release_id="release-4")
    row["revision"] = 3
    client = FakeDeploy([row])

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["changed"] is False
    assert envelope["data"]["cancelledUpdate"] is None
    assert envelope["data"]["release"] == {"id": "release-4", "version": 4}
    jsonschema.Draft202012Validator(_schema()).validate(envelope["data"])


class _Unrouted(FakeDeploy):
    """A comfy-deploy whose cancel answers 404, with or without its own code."""

    server_code: str | None = None

    def cancel_pending_update(self, deployment_id: str) -> JsonObject:
        self.cancel_calls.append(deployment_id)
        details = {"server_code": self.server_code} if self.server_code else None
        raise DeployAPIError("deploy_not_found", "not found", status=404, details=details)


@pytest.mark.parametrize(
    ("revision", "server_code", "code", "reason", "cancels"),
    [
        (None, None, "deploy_updates_unavailable", "service_too_old", 1),
        (3, None, "deploy_updates_unavailable", "service_too_old", 1),
        (3, "NOT_FOUND", "deploy_not_found", None, 1),
    ],
    ids=["no_revision_in_the_read", "service_too_old", "gone_since_the_read"],
)
def test_a_cancel_the_service_cannot_take_says_why(
    monkeypatch, revision: int | None, server_code: str | None, code: str, reason: str | None, cancels: int
) -> None:
    # Given a deployment the read finds, and a cancel the service cannot take
    row = deployment("dep-prod", release_id="release-4")
    row["revision"] = revision
    client = _Unrouted([row])
    client.server_code = server_code

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == code
    assert (error.get("details") or {}).get("reason") == reason
    assert len(client.cancel_calls) == cancels


class _LandedFirst(FakeDeploy):
    """A deployment whose waiting update lands between the cancel's read and its call."""

    def cancel_pending_update(self, deployment_id: str) -> JsonObject:
        self.rows[deployment_id].update({"pendingUpdate": None, "releaseId": "release-5", "revision": 4})
        return super().cancel_pending_update(deployment_id)


def test_a_cancel_the_landing_beat_reports_the_release_that_landed(monkeypatch) -> None:
    # Given the update to v5 landing before the cancel reaches the service
    client = _waiting()
    client = _LandedFirst(list(client.rows.values()))

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["cancelledUpdate"] is None
    assert data["release"] == {"id": "release-5", "version": 5}
    assert data["deployment"]["revision"] == 4


class _Missing(FakeDeploy):
    """A comfy-deploy that holds no deployment by the id asked."""

    def get_deployment(self, deployment_id: str) -> JsonObject:
        raise DeployAPIError("deploy_not_found", "no deployment with that id", status=404)


def test_a_deployment_that_does_not_exist_is_not_found(monkeypatch) -> None:
    # Given an id no deployment has
    client = _Missing([])

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-typo")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_not_found"
    assert client.cancel_calls == []


def test_a_kind_this_client_does_not_know_reads_as_an_update(monkeypatch) -> None:
    # Given a waiting update of a kind the service added later
    client = _waiting(kind="promote")

    # When
    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["cancelledUpdate"]["kind"] == "update"


def test_cancel_picks_the_builds_only_running_deployment(tmp_path, monkeypatch) -> None:
    # Given a Build whose one deployment waits on an update
    client = _waiting()

    # When
    result = _invoke(monkeypatch, client, str(write_spec(tmp_path)))

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.cancel_calls == ["dep-prod"]


def test_cancel_refuses_to_pick_between_two_running_deployments(tmp_path, monkeypatch) -> None:
    # Given
    client = _waiting()
    client.rows["dep-other"] = deployment("dep-other", release_id="release-5")

    # When
    result = _invoke(monkeypatch, client, str(write_spec(tmp_path)))

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_ambiguous_deployment"
    assert client.cancel_calls == []
