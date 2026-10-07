"""`comfy deploy up` moving a deployment between releases.

A deployment carries `revision` only in a workspace with deployment updates on,
so every row here that should move sets one, and a row without it plays a
workspace outside that rollout.
"""

from __future__ import annotations

import functools
import importlib
import json
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest
from deploy_up_support import FakeBuilder, FakeDeploy, deployment, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_up import move_text
from comfy_cli.deploy_api_errors import DeployAPIError

_RELEASES = [
    {"id": "release-4", "buildId": "build-1", "version": 4, "deployable": True},
    {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
]


def _deploy() -> ModuleType:
    return importlib.import_module("comfy_cli.command.deploy")


def _schema(name: str) -> dict:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _live(deployment_id: str, *, release_id: str = "release-4", revision: int | None = 3, **changes) -> JsonObject:
    row = deployment(deployment_id, release_id=release_id, **changes)
    if revision is not None:
        row["revision"] = revision
    return row


def _up(tmp_path, monkeypatch, client: FakeDeploy, *args: str):
    module = _deploy()
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(_RELEASES), client))
    monkeypatch.setattr(module, "_sleep", lambda _: None)
    return CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path)), *args])


def test_up_moves_the_one_deployment_onto_the_new_release(tmp_path, monkeypatch) -> None:
    # Given one deployment on v4 in a workspace with deployment updates on
    client = FakeDeploy([_live("dep-1")])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then it moved in place: same id, the next revision, no second deployment
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == [("dep-1", 3, "release-5")]
    assert client.create_keys == []
    envelope = _envelope(result)
    data = envelope["data"]
    assert envelope["changed"] is True
    assert data["deployment"] == {"id": "dep-1", "status": "ready", "created": False, "revision": 4}
    assert data["release"] == {"id": "release-5", "version": 5}
    assert data["previousRelease"] == {"id": "release-4", "version": 4}
    # The moved deployment no longer serves v4, so it is not reported as billing beside the new one.
    assert data["supersedes"] == []
    jsonschema.Draft202012Validator(_schema("deploy_up.json")).validate(data)


def test_up_outside_the_rollout_still_creates_a_deployment_for_the_new_release(tmp_path, monkeypatch) -> None:
    # Given the same deployment, read without a revision
    client = FakeDeploy([_live("dep-1", revision=None)])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--gpu", "l4", "--region", "US-MO-2")

    # Then today's behaviour holds: a new deployment, and the old one reported as billing
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == []
    assert len(client.create_keys) == 1
    assert [row["id"] for row in _envelope(result)["data"]["supersedes"]] == ["dep-1"]


def test_up_refuses_to_guess_between_two_deployments_of_the_build(tmp_path, monkeypatch) -> None:
    # Given two deployments of the Build, inside the rollout
    client = FakeDeploy([_live("dep-1"), _live("dep-2", release_id="release-5")])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_ambiguous_deployment"
    assert error["details"]["candidateIds"] == ["dep-1", "dep-2"]
    assert "--create" in error["hint"]
    assert client.move_calls == [] and client.create_keys == []


def test_up_moves_the_deployment_named_among_several(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-1"), _live("dep-2", release_id="release-5")])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--deployment", "dep-1")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == [("dep-1", 3, "release-5")]


def test_create_adds_a_deployment_beside_the_one_already_on_the_release(tmp_path, monkeypatch) -> None:
    # Given a live deployment already on the release
    client = FakeDeploy([_live("dep-live", release_id="release-5")])

    # When
    result = _up(
        tmp_path, monkeypatch, client, "--release", "release-5", "--create", "--gpu", "l4", "--region", "US-MO-2"
    )

    # Then a second deployment exists, under a key the plain create never uses
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == []
    assert len(client.rows) == 2
    module = _deploy()
    assert client.create_keys == [module._idempotency_key("build-1", "release-5", 0, 1)]
    assert client.create_keys[0] != module._idempotency_key("build-1", "release-5", 0)


def test_create_takes_no_deployment(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-1")])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--create", "--deployment", "dep-1")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_bad_request"
    assert client.create_keys == []


@pytest.mark.parametrize("status", ["stopping", "stop_failed"])
def test_up_will_not_move_a_deployment_that_is_stopping_or_failed_to_stop(tmp_path, monkeypatch, status: str) -> None:
    # Given
    client = FakeDeploy([_live("dep-1", status=status)])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--no-watch")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_conflict"
    assert "comfy deploy status --deployment dep-1" in error["hint"]
    assert client.move_calls == [] and client.start_calls == []


def test_up_follows_a_waiting_move_until_it_lands(tmp_path, monkeypatch) -> None:
    # Given a move that waits on v5's copy for two reads, then lands
    client = FakeDeploy(
        [_live("dep-1")],
        move="pending",
        get_patches=[
            {},
            {"pendingUpdate": None, "releaseId": "release-5", "revision": 4},
        ],
    )

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["deployment"]["revision"] == 4
    assert client.get_ids.count("dep-1") == 3


def test_a_read_caught_as_the_move_lands_does_not_fail_it(tmp_path, monkeypatch) -> None:
    # Given one read taken mid-landing: the update no longer waits, but the
    # old release still serves at the old revision, then the move has landed
    client = FakeDeploy(
        [_live("dep-1")],
        move="pending",
        get_patches=[
            {"pendingUpdate": None},
            {"releaseId": "release-5", "revision": 4},
        ],
    )

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["deployment"]["revision"] == 4


_WAITING = {"pendingUpdate": {"releaseId": "release-5", "baseRevision": 3, "status": "provisioning", "since": "x"}}


class _CountingWatch(FakeDeploy):
    """Counts the reads made once the move was sent."""

    watched = 0

    def get_deployment(self, deployment_id: str) -> JsonObject:
        if self.move_calls:
            self.watched += 1
        return super().get_deployment(deployment_id)


@pytest.mark.parametrize(
    ("patches", "reads"),
    [
        ([{"pendingUpdate": {"releaseId": "release-5", "baseRevision": 3, "status": "failed", "since": "x"}}], 1),
        ([{"pendingUpdate": None}], 2),
        ([{"pendingUpdate": None}, {"revision": None}, {"revision": 3}], 3),
    ],
    ids=["copy_failed_at_once", "dropped_twice", "no_revision_read_between"],
)
def test_a_watch_settles_a_failure_after_the_reads_it_needs(tmp_path, monkeypatch, patches, reads) -> None:
    # Given
    client = _CountingWatch([_live("dep-1")], move="pending", get_patches=patches)

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then the watch read only as often as the failure needed
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_update_failed"
    assert client.watched == reads


def test_a_dropped_read_then_a_waiting_one_starts_the_count_again(tmp_path, monkeypatch) -> None:
    # Given dropped, waiting, dropped, then landed
    client = FakeDeploy(
        [_live("dep-1")],
        move="pending",
        get_patches=[
            {"pendingUpdate": None},
            _WAITING,
            {"pendingUpdate": None},
            {"releaseId": "release-5", "revision": 4},
        ],
    )

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["deployment"]["revision"] == 4


def test_a_watch_interrupted_on_a_dropped_read_does_not_claim_the_move() -> None:
    # Given a read that shows the old release with nothing waiting
    deployment = {"id": "dep-1", "status": "ready", "releaseId": "release-4", "revision": 3, "pendingUpdate": None}

    # When
    text = move_text("dep-1", deployment, {"id": "release-5", "version": 5}, {"id": "release-4", "version": 4}, True)

    # Then
    assert "now serves" not in text
    assert "read as serving release v4, with no update to release v5 waiting" in text


def test_a_watch_interrupted_after_another_move_names_the_release_the_read_shows() -> None:
    # Given a read showing a third release, which another move put there
    deployment = {"id": "dep-1", "status": "ready", "releaseId": "release-9", "revision": 4, "pendingUpdate": None}

    # When
    text = move_text("dep-1", deployment, {"id": "release-5", "version": 5}, {"id": "release-4", "version": 4}, True)

    # Then
    assert "read as serving release release-9" in text


def test_up_without_a_watch_returns_while_the_move_waits(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-1")], move="pending")

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--no-watch")

    # Then
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["changed"] is True
    assert envelope["data"]["deployment"]["revision"] == 3


@pytest.mark.parametrize(
    "patch",
    [
        {"pendingUpdate": {"releaseId": "release-5", "baseRevision": 3, "status": "failed", "since": "x"}},
        {"pendingUpdate": None},
    ],
    ids=["copy_failed", "update_dropped"],
)
def test_a_move_that_fails_exits_1_naming_the_release_still_serving(tmp_path, monkeypatch, patch) -> None:
    # Given
    client = FakeDeploy([_live("dep-1")], move="pending", get_patches=[patch])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_update_failed"
    assert error["details"]["serving_release_id"] == "release-4"
    assert error["details"]["release_id"] == "release-5"
    assert "still serves release v4" in error["message"]


def test_bounds_on_a_move_are_applied_once_it_lands(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-1")])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--min", "1", "--max", "3")

    # Then the move went first, and the bounds followed as their own edit
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == [("dep-1", 3, "release-5")]
    assert client.update_calls == ["dep-1"]
    assert _envelope(result)["data"]["computeConfig"]["max"] == 3


def test_bounds_on_a_move_without_a_watch_are_refused_before_moving(tmp_path, monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-1")])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--min", "1", "--max", "3", "--no-watch")

    # Then
    assert result.exit_code == 1
    assert _envelope(result)["error"]["code"] == "deploy_bad_request"
    assert client.move_calls == [] and client.update_calls == []


def test_a_rerun_on_the_release_already_served_reads_nothing_extra(tmp_path, monkeypatch) -> None:
    # Given the one deployment already on v5
    client = FakeDeploy([_live("dep-1", release_id="release-5", revision=4)])

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--no-watch")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.get_ids == []
    assert client.move_calls == []
    assert _envelope(result)["changed"] is False
    assert "previousRelease" not in _envelope(result)["data"]


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_a_release_cut_to_fix_a_down_deployment_moves_it(tmp_path, monkeypatch, status: str) -> None:
    """The service starts the new release's copy for a deployment that is down,
    so the fix keeps the URL rather than needing `--create`."""
    # Given the one deployment, down on v4, and a move waiting on v5's copy
    client = FakeDeploy(
        [_live("dep-1", status=status)],
        move="pending",
        get_patches=[{}, {"pendingUpdate": None, "releaseId": "release-5", "revision": 4, "status": "ready"}],
    )

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then the deployment's own status did not end the watch before the move landed
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == [("dep-1", 3, "release-5")]
    assert client.start_calls == []
    assert _envelope(result)["data"]["deployment"]["status"] == "ready"


def test_a_read_without_a_revision_mid_watch_does_not_fail_the_move(tmp_path, monkeypatch) -> None:
    """The service leaves revision and pendingUpdate out when its rollout check
    fails, which is a missing answer about the move, not a dropped move."""
    # Given a read during the wait that carries neither
    client = FakeDeploy(
        [_live("dep-1")],
        move="pending",
        get_patches=[
            {"pendingUpdate": None, "revision": None},
            {"pendingUpdate": None, "releaseId": "release-5", "revision": 4},
        ],
    )

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.get_ids.count("dep-1") == 3


def test_a_move_another_change_overtook_is_a_failure(tmp_path, monkeypatch) -> None:
    # Given a later revision that serves neither release up asked about
    client = FakeDeploy(
        [_live("dep-1")], move="pending", get_patches=[{"pendingUpdate": None, "releaseId": "release-3", "revision": 5}]
    )

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_update_failed"
    assert error["details"]["serving_release_id"] == "release-3"


def test_a_refused_bounds_edit_after_a_landed_move_still_reports_the_move(tmp_path, monkeypatch) -> None:
    # Given a move that lands, and a bounds edit the service refuses
    client = FakeDeploy([_live("dep-1")])
    client.update_error = DeployAPIError("deploy_payment_required", "spend blocked", status=402)

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--min", "1", "--max", "3")

    # Then the move is reported, and the bounds as having had no effect
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["ok"] is True
    assert envelope["data"]["release"]["id"] == "release-5"
    assert envelope["data"]["computeConfig"]["max"] == 1
    assert "--min and --max had no effect" in result.stderr


def test_bounds_with_a_move_answered_at_the_same_revision_are_applied_at_once(tmp_path, monkeypatch) -> None:
    # Given a service that already serves v5 by the time the move arrives
    client = FakeDeploy([_live("dep-1")], move="unchanged")

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--min", "1", "--max", "3")

    # Then
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["changed"] is False
    assert client.update_calls == ["dep-1"]
    assert envelope["data"]["computeConfig"]["max"] == 3


def test_a_deployment_moved_since_the_list_is_reconciled_not_moved_again(tmp_path, monkeypatch) -> None:
    # Given a list row on v4 whose fresh read already serves v5
    client = FakeDeploy([_live("dep-1")])
    original = client.get_deployment

    def read(deployment_id: str) -> JsonObject:
        row = original(deployment_id)
        row["releaseId"] = "release-5"
        return row

    client.get_deployment = read

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--no-watch")

    # Then
    assert result.exit_code == 0, result.stderr
    assert client.move_calls == [] and client.create_keys == []


def test_stopping_a_move_watch_says_the_bounds_were_not_applied(tmp_path, monkeypatch) -> None:
    # Given a move still waiting when the user presses Ctrl-C
    module = _deploy()
    client = FakeDeploy([_live("dep-1")], move="pending")
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(_RELEASES), client))

    def interrupt(_: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_sleep", interrupt)

    # When
    result = CliRunner().invoke(
        app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--release", "release-5", "--min", "1", "--max", "3"]
    )

    # Then
    assert result.exit_code == 130
    assert "--min and --max were not applied" in result.stderr
    assert "comfy deploy scale --deployment dep-1" in result.stderr
    assert client.update_calls == []


@pytest.mark.parametrize("target", ["release-5", "release-6"], ids=["already_served", "new_release"])
def test_a_stopped_leftover_never_makes_up_ambiguous(tmp_path, monkeypatch, target: str) -> None:
    """Every workspace that used `up` before this change has stopped deployments
    on old releases beside the one it runs."""
    # Given one ready deployment on v5 and a stopped one an older `up` left on v4
    releases = [*_RELEASES, {"id": "release-6", "buildId": "build-1", "version": 6, "deployable": True}]
    client = FakeDeploy([_live("dep-1", release_id="release-5", revision=4), _live("dep-2", status="stopped")])
    module = _deploy()
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(releases), client))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--release", target])

    # Then the running one is reconciled or moved, and the leftover is left alone
    assert result.exit_code == 0, result.stderr
    expected = [] if target == "release-5" else [("dep-1", 4, "release-6")]
    assert client.move_calls == expected
    assert client.create_keys == []


@pytest.mark.parametrize("status", ["failed", "stopped"])
def test_a_waiting_move_of_a_down_deployment_without_a_watch_is_not_a_failure(tmp_path, monkeypatch, status) -> None:
    # Given
    client = FakeDeploy([_live("dep-1", status=status)], move="pending")

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--no-watch")

    # Then
    assert result.exit_code == 0, result.stderr
    envelope = _envelope(result)
    assert envelope["ok"] is True
    assert envelope["changed"] is True


def test_a_move_reply_without_a_revision_is_followed_not_failed(tmp_path, monkeypatch) -> None:
    """The service drops revision and pendingUpdate from any answer its rollout check fails."""
    # Given a move accepted with both fields missing from its reply, which then lands
    client = FakeDeploy(
        [_live("dep-1")], move="pending", get_patches=[{"pendingUpdate": None, "releaseId": "release-5", "revision": 4}]
    )
    original = client.move_deployment

    def move(*args) -> JsonObject:
        reply = original(*args)
        reply.pop("revision")
        reply.pop("pendingUpdate")
        return reply

    client.move_deployment = move

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["deployment"]["revision"] == 4


def test_an_unanswered_bounds_edit_after_a_landed_move_says_the_move_landed(tmp_path, monkeypatch) -> None:
    # Given a landed move, then a bounds edit that gets no answer
    client = FakeDeploy([_live("dep-1")])
    client.update_error = DeployAPIError("deploy_server_error", "timed out")

    # When
    result = _up(tmp_path, monkeypatch, client, "--release", "release-5", "--min", "1", "--max", "3")

    # Then the outcome is unknown, so it fails, and says the move itself landed
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_server_error"
    assert "release v5 landed" in error["message"]


def test_a_move_that_never_lands_hands_the_watch_back_after_the_limit(tmp_path, monkeypatch) -> None:
    # Given a move that keeps waiting, and a clock that passes the limit
    module = _deploy()
    runtime = importlib.import_module("comfy_cli.command.deploy_runtime")
    client = FakeDeploy([_live("dep-1")], move="pending")
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(_RELEASES), client))
    monkeypatch.setattr(module, "_sleep", lambda _: None)
    ticks = iter(range(0, 10_000, 1000))
    poll = functools.partial(runtime.poll_deployment, clock=lambda: float(next(ticks)))
    monkeypatch.setattr(module, "_poll_deployment", poll)

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--release", "release-5"])

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 75
    assert error["code"] == "deploy_watch_lost"
    assert "still updating" in error["message"]
    assert "comfy deploy show --deployment dep-1" in error["hint"]


def test_stopping_a_move_watch_before_any_read_still_mentions_the_bounds(tmp_path, monkeypatch) -> None:
    # Given a Ctrl-C during the first read after the move
    module = _deploy()
    client = FakeDeploy([_live("dep-1")], move="pending")
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(_RELEASES), client))
    original = client.get_deployment

    def read(deployment_id: str) -> JsonObject:
        if client.move_calls:
            raise KeyboardInterrupt
        return original(deployment_id)

    client.get_deployment = read

    # When
    result = CliRunner().invoke(
        app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--release", "release-5", "--min", "1", "--max", "3"]
    )

    # Then
    assert result.exit_code == 130
    assert "--min and --max were not applied" in result.stderr
