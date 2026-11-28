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


def _waiting(kind: str = "update", release_id: str = "release-5", **served: str | None) -> FakeDeploy:
    """A deployment serving v4 at revision 3, waiting on an update to ``release_id``."""
    row = deployment("dep-prod", release_id="release-4", **served)
    row["revision"] = 3
    row["pendingUpdate"] = {
        "releaseId": release_id,
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


def test_a_cancel_stops_the_release_the_update_waits_on_and_keeps_the_release(monkeypatch) -> None:
    client = _waiting()

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

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
    assert client.stop_release_calls == [("dep-prod", "release-5")]


def test_a_cancel_says_what_it_ended_and_what_still_serves(monkeypatch) -> None:
    client = _waiting()

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output="--no-json")

    assert result.exit_code == 0, result.stderr
    assert "Cancelled the update to release v5. dep-prod keeps serving release v4." in result.stdout


def test_a_cancelled_rollback_is_named_a_rollback(monkeypatch) -> None:
    client = _waiting(kind="rollback", name="production")

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output="--no-json")

    assert result.exit_code == 0, result.stderr
    assert "Cancelled the rollback to release v5. production (dep-prod) keeps serving release v4." in result.stdout
    assert client.stop_release_calls == [("dep-prod", "release-5")]


def test_a_gpu_or_region_move_on_the_serving_release_cancels_the_same_way(monkeypatch) -> None:
    client = _waiting(kind="compute", release_id="release-4")

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-4")]
    envelope = _envelope(result)
    assert envelope["changed"] is True
    data = envelope["data"]
    assert data["cancelledUpdate"] == {
        "release": {"id": "release-4", "version": 4},
        "baseRevision": 3,
        "kind": "update",
    }
    assert data["release"] == {"id": "release-4", "version": 4}
    jsonschema.Draft202012Validator(_schema()).validate(data)


def test_a_gpu_or_region_move_cancel_ends_once_its_update_reads_failed(monkeypatch) -> None:
    client = _waiting(kind="compute", release_id="release-4")
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy_cancel"), "_sleep", pytest.fail)

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-4")]
    assert client.rows["dep-prod"]["pendingUpdate"]["status"] == "failed"
    envelope = _envelope(result)
    assert envelope["changed"] is True
    assert envelope["data"]["cancelledUpdate"]["release"] == {"id": "release-4", "version": 4}


def test_a_cancelled_gpu_or_region_move_is_named_a_move(monkeypatch) -> None:
    client = _waiting(kind="compute", release_id="release-4")

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output="--no-json")

    assert result.exit_code == 0, result.stderr
    assert (
        "Cancelled the GPU/region move. dep-prod keeps serving release v4 on its current GPU and region."
        in result.stdout
    )
    assert "to release v4" not in result.stdout


@pytest.mark.parametrize("output", ["--json", "--no-json"])
def test_a_cancel_with_nothing_waiting_sends_nothing(monkeypatch, output: str) -> None:
    row = deployment("dep-prod", release_id="release-4")
    row["revision"] = 3
    client = FakeDeploy([row])

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output=output)

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == []
    if output == "--no-json":
        assert "Deployment dep-prod has no update waiting; it serves release v4." in result.stdout
        return
    envelope = _envelope(result)
    assert envelope["changed"] is False
    assert envelope["data"]["cancelledUpdate"] is None
    assert envelope["data"]["release"] == {"id": "release-4", "version": 4}
    jsonschema.Draft202012Validator(_schema()).validate(envelope["data"])


class _LandedFirst(FakeDeploy):
    """A deployment whose waiting update lands between the cancel's read and its stop."""

    def stop_kept_release(self, deployment_id: str, release_id: str) -> JsonObject:
        self.rows[deployment_id].update({"pendingUpdate": None, "releaseId": "release-5", "revision": 4})
        return super().stop_kept_release(deployment_id, release_id)


@pytest.mark.parametrize("output", ["--json", "--no-json"])
def test_a_cancel_the_landing_beat_says_the_update_already_landed(monkeypatch, output: str) -> None:
    client = _LandedFirst(list(_waiting().rows.values()))

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output=output)

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-5")]
    if output == "--no-json":
        assert "The update to release v5 already landed; dep-prod serves release v5." in result.stdout
        return
    envelope = _envelope(result)
    assert envelope["changed"] is False
    data = envelope["data"]
    assert data["cancelledUpdate"] is None
    assert data["release"] == {"id": "release-5", "version": 5}
    assert data["deployment"]["revision"] == 4
    jsonschema.Draft202012Validator(_schema()).validate(data)


class _MovedFirst(FakeDeploy):
    def stop_kept_release(self, deployment_id: str, release_id: str) -> JsonObject:
        self.rows[deployment_id].update({"pendingUpdate": None, "revision": 4})
        return super().stop_kept_release(deployment_id, release_id)


@pytest.mark.parametrize("output", ["--json", "--no-json"])
def test_a_gpu_or_region_move_that_landed_first_says_the_move_landed(monkeypatch, output: str) -> None:
    client = _MovedFirst(list(_waiting(kind="compute", release_id="release-4").rows.values()))

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output=output)

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-4")]
    if output == "--no-json":
        assert "The GPU/region move already landed; dep-prod serves release v4." in result.stdout
        return
    envelope = _envelope(result)
    assert envelope["changed"] is False
    data = envelope["data"]
    assert data["cancelledUpdate"] is None
    assert data["release"] == {"id": "release-4", "version": 4}
    assert data["deployment"]["revision"] == 4
    jsonschema.Draft202012Validator(_schema()).validate(data)


class _CancelledFirst(FakeDeploy):
    def stop_kept_release(self, deployment_id: str, release_id: str) -> JsonObject:
        self.rows[deployment_id]["pendingUpdate"] = None
        return super().stop_kept_release(deployment_id, release_id)


@pytest.mark.parametrize("output", ["--json", "--no-json"])
def test_a_cancel_an_earlier_cancellation_beat_does_not_say_the_move_landed(monkeypatch, output: str) -> None:
    client = _CancelledFirst(list(_waiting(kind="compute", release_id="release-4").rows.values()))
    client.rows["dep-prod"]["pendingUpdate"]["status"] = "failed"

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod", output=output)

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-4")]
    if output == "--no-json":
        assert "landed" not in result.stdout
        assert "The GPU/region move is no longer waiting; dep-prod serves release v4." in result.stdout
        return
    envelope = _envelope(result)
    assert envelope["changed"] is False
    data = envelope["data"]
    assert data["cancelledUpdate"] is None
    assert data["release"] == {"id": "release-4", "version": 4}
    assert data["deployment"]["revision"] == 3
    jsonschema.Draft202012Validator(_schema()).validate(data)


class _SlowStop(FakeDeploy):
    reads = 2

    def stop_kept_release(self, deployment_id: str, release_id: str) -> JsonObject:
        with self._lock:
            self.stop_release_calls.append((deployment_id, release_id))
            return {"releaseId": release_id, "status": "stopping"}

    def get_deployment(self, deployment_id: str) -> JsonObject:
        if self.stop_release_calls:
            if self.reads == 0:
                self.rows[deployment_id]["pendingUpdate"] = None
            self.reads -= 1
        return super().get_deployment(deployment_id)


def test_an_update_still_waiting_after_the_stop_is_waited_on(monkeypatch) -> None:
    client = _SlowStop(list(_waiting().rows.values()))
    sleeps: list[float] = []
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy_cancel"), "_sleep", sleeps.append)

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-5")]
    assert len(sleeps) == 2
    envelope = _envelope(result)
    assert envelope["changed"] is True
    assert envelope["data"]["cancelledUpdate"]["release"] == {"id": "release-5", "version": 5}


def test_an_unfinished_cancellation_is_stopped_again_before_it_is_reported(monkeypatch) -> None:
    client = _waiting(kind="compute", release_id="release-4")
    client.rows["dep-prod"]["pendingUpdate"]["status"] = "failed"

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-4")]
    envelope = _envelope(result)
    assert envelope["changed"] is True
    assert envelope["data"]["cancelledUpdate"]["release"] == {"id": "release-4", "version": 4}


class _Replaced(FakeDeploy):
    def stop_kept_release(self, deployment_id: str, release_id: str) -> JsonObject:
        reply = super().stop_kept_release(deployment_id, release_id)
        self.rows[deployment_id]["pendingUpdate"] = {
            "releaseId": "release-4",
            "baseRevision": 3,
            "status": "provisioning",
            "since": "2026-10-07T12:05:00Z",
            "kind": "rollback",
        }
        return reply


def test_a_newer_update_waiting_ends_the_wait_and_names_the_one_stopped(monkeypatch) -> None:
    client = _Replaced(list(_waiting().rows.values()))
    monkeypatch.setattr(importlib.import_module("comfy_cli.command.deploy_cancel"), "_sleep", pytest.fail)

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    assert result.exit_code == 0, result.stderr
    assert client.stop_release_calls == [("dep-prod", "release-5")]
    assert _envelope(result)["data"]["cancelledUpdate"]["release"] == {"id": "release-5", "version": 5}


def test_the_fake_copy_stop_countdown_keeps_a_newer_update_waiting() -> None:
    client = _waiting(kind="compute", release_id="release-4")
    client.move = "pending"
    client.stop_kept_release("dep-prod", "release-4")
    client.move_deployment("dep-prod", 3, "release-5")

    for _ in range(client.copy_stop_reads + 1):
        row = client.get_deployment("dep-prod")

    assert row["pendingUpdate"]["releaseId"] == "release-5"
    assert row["pendingUpdate"]["status"] == "provisioning"


def test_a_cancellation_that_does_not_finish_hands_the_watch_back(monkeypatch) -> None:
    client = _SlowStop(list(_waiting().rows.values()))
    client.reads = 1000
    module = importlib.import_module("comfy_cli.command.deploy_cancel")
    monkeypatch.setattr(module, "CANCEL_WATCH_SECONDS", 0.0)
    monkeypatch.setattr(module, "_sleep", lambda _: None)

    result = _invoke(monkeypatch, client, "--deployment", "dep-prod")

    assert result.exit_code == 75
    error = _envelope(result)["error"]
    assert error["code"] == "deploy_watch_lost"
    assert "comfy deploy cancel --deployment dep-prod" in error["hint"]
    assert client.stop_release_calls == [("dep-prod", "release-5")]


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
    assert client.stop_release_calls == []


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
    assert client.stop_release_calls == [("dep-prod", "release-5")]


def test_cancel_refuses_to_pick_between_two_running_deployments(tmp_path, monkeypatch) -> None:
    # Given
    client = _waiting()
    client.rows["dep-other"] = deployment("dep-other", release_id="release-5")

    # When
    result = _invoke(monkeypatch, client, str(write_spec(tmp_path)))

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_ambiguous_deployment"
    assert client.stop_release_calls == []
