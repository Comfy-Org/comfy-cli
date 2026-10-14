"""`comfy deploy promote` moving one deployment onto the release another serves."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import jsonschema
from deploy_up_support import FakeBuilder, FakeDeploy, deployment
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject

_RELEASES = [
    {"id": "release-4", "buildId": "build-1", "version": 4, "deployable": True},
    {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
    {"id": "release-6", "buildId": "build-1", "version": 6, "deployable": True},
]


def _schema() -> dict:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / "deploy_promote.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def _live(deployment_id: str, *, release_id: str = "release-4", revision: int | None = 3, **changes) -> JsonObject:
    row = deployment(deployment_id, release_id=release_id, **changes)
    if revision is not None:
        row["revision"] = revision
    return row


def _promote(monkeypatch, client: FakeDeploy, *args: str, output: str = "--json", builder: FakeBuilder | None = None):
    module = importlib.import_module("comfy_cli.command.deploy")
    chosen = builder or FakeBuilder(_RELEASES)
    monkeypatch.setattr(module, "_command_clients", lambda: (chosen, client))
    monkeypatch.setattr(module, "_sleep", lambda _: None)
    return CliRunner().invoke(app, [output, "deploy", "promote", *args], env={"COLUMNS": "400"})


def _staging_and_production(**client_options) -> FakeDeploy:
    return FakeDeploy(
        [_live("dep-staging", release_id="release-5", revision=1), _live("dep-prod")],
        **client_options,
    )


def test_promote_moves_the_target_onto_the_source_release(monkeypatch) -> None:
    # Given staging on v5 and production on v4
    client = _staging_and_production()

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then the service resolved the release from the source, and production kept its id
    assert result.exit_code == 0, result.stderr
    assert client.promote_calls == [("dep-prod", 3, "dep-staging")]
    assert client.move_calls == []
    envelope = _envelope(result)
    assert envelope["changed"] is True
    data = envelope["data"]
    assert data["deployment"] == {"id": "dep-prod", "status": "ready", "revision": 4}
    assert data["source"] == {"id": "dep-staging"}
    assert data["release"] == {"id": "release-5", "version": 5}
    assert data["previousRelease"] == {"id": "release-4", "version": 4}
    jsonschema.Draft202012Validator(_schema()).validate(data)


def test_a_waiting_promote_is_followed_until_it_lands(monkeypatch) -> None:
    # Given a move that waits on v5's copy, which the next read shows landed
    landed = {"pendingUpdate": None, "releaseId": "release-5", "revision": 4}
    client = _staging_and_production(move="pending", get_patches=[landed])

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["data"]["deployment"]["revision"] == 4


def test_no_watch_returns_while_the_promote_waits(monkeypatch) -> None:
    # Given
    client = _staging_and_production(move="pending")

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", "--no-watch", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    assert "moves to release v5 once it is ready; release v4 serves until then" in result.stdout


def test_promote_outside_the_rollout_says_updates_are_off(monkeypatch) -> None:
    # Given
    client = FakeDeploy([_live("dep-staging", release_id="release-5", revision=None), _live("dep-prod", revision=None)])

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_updates_unavailable"
    assert error["details"] == {"deployment_id": "dep-prod"}
    assert "`comfy deploy up --create --release release-5 --gpu <class> --region <region>`" in error["hint"]
    assert client.promote_calls == []


def test_a_repeated_promote_changes_nothing(monkeypatch) -> None:
    # Given a promote that landed
    client = _staging_and_production()
    first = _promote(monkeypatch, client, "dep-staging", "dep-prod")
    assert _envelope(first)["changed"] is True

    # When it runs again
    again = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then the service answered at the same revision
    assert again.exit_code == 0, again.stderr
    envelope = _envelope(again)
    assert envelope["changed"] is False
    assert envelope["data"]["deployment"]["revision"] == 4


def test_a_reply_without_a_revision_is_followed_onto_the_source_release(monkeypatch) -> None:
    # Given a waiting move whose reply, and the read after it, lost revision
    # and pendingUpdate, so both still name the old release
    unanswered = {"pendingUpdate": None, "revision": None}
    landed = {"releaseId": "release-5", "revision": 4}
    client = _staging_and_production(move="pending", get_patches=[unanswered, landed], strip_move_reply=True)

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then the watch followed v5, which the source served, rather than v4
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["release"] == {"id": "release-5", "version": 5}
    assert data["deployment"]["revision"] == 4


def test_a_bare_reply_is_read_again_so_an_unwatched_promote_says_it_waits(monkeypatch) -> None:
    # Given a reply with no revision, and a read that shows the move waiting
    client = _staging_and_production(move="pending", get_patches=[{}], strip_move_reply=True)

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", "--no-watch")

    # Then
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["waiting"] is True
    assert data["release"] == {"id": "release-5", "version": 5}


class _SourceMovesFirst(FakeDeploy):
    def promote_deployment(self, deployment_id: str, base_revision: int, from_deployment_id: str) -> JsonObject:
        self.rows[from_deployment_id]["releaseId"] = "release-6"
        return super().promote_deployment(deployment_id, base_revision, from_deployment_id)


class _BuilderDownAfterTheMove(FakeBuilder):
    def get_release(self, release_id: str) -> JsonObject:
        if release_id == "release-6":
            raise TimeoutError("the builder did not answer")
        return super().get_release(release_id)


def test_an_unreadable_release_after_the_move_still_watches_it(monkeypatch) -> None:
    # Given the source moving to v6 just before the promote, and the builder
    # failing to read v6 once the promote is accepted
    landed = {"pendingUpdate": None, "releaseId": "release-6", "revision": 4}
    client = _SourceMovesFirst(
        [_live("dep-staging", release_id="release-5", revision=1), _live("dep-prod")],
        move="pending",
        get_patches=[landed],
    )

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", builder=_BuilderDownAfterTheMove(_RELEASES))

    # Then the move is followed and reported by its release id
    assert result.exit_code == 0, result.stderr
    data = _envelope(result)["data"]
    assert data["release"] == {"id": "release-6"}
    jsonschema.Draft202012Validator(_schema()).validate(data)


def test_a_stopped_target_starts_on_the_source_release(monkeypatch) -> None:
    # Given production stopped on v4
    client = FakeDeploy(
        [_live("dep-staging", release_id="release-5", revision=1), _live("dep-prod", status="stopped")],
        move="pending",
    )

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", "--no-watch", output="--no-json")

    # Then nothing is said to serve meanwhile
    assert result.exit_code == 0, result.stderr
    assert "Deployment dep-prod starts on release v5." in result.stdout
    assert "serves until" not in result.stdout


def test_a_promote_that_lands_at_once_says_so_once(monkeypatch) -> None:
    # Given
    client = _staging_and_production()

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    assert result.stdout.count("now serves release v5 (was release v4)") == 1
    assert "serves until" not in result.stdout


def test_a_landed_promote_onto_an_unhealthy_target_exits_1(monkeypatch) -> None:
    # Given v5 landing on a deployment whose endpoint then degrades
    landed = {"pendingUpdate": None, "releaseId": "release-5", "revision": 4, "status": "unhealthy"}
    client = _staging_and_production(move="pending", get_patches=[landed])

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    envelope = _envelope(result)
    assert result.exit_code == 1
    assert envelope["ok"] is False
    assert envelope["error"]["code"] == "deploy_status_terminal"


def test_a_failed_promote_exits_1_naming_the_release_still_serving(monkeypatch) -> None:
    # Given v5's copy failing
    failed = {"pendingUpdate": {"releaseId": "release-5", "baseRevision": 3, "status": "failed", "since": "x"}}
    client = _staging_and_production(move="pending", get_patches=[failed])

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_update_failed"
    assert error["details"]["serving_release_id"] == "release-4"


def test_a_promote_a_newer_update_replaced_exits_1_as_replaced(monkeypatch) -> None:
    # Given a newer update to v6 replacing the promote of v5 while it waits
    replaced = {
        "pendingUpdate": {
            "releaseId": "release-6",
            "baseRevision": 3,
            "status": "provisioning",
            "since": "x",
        }
    }
    client = _staging_and_production(move="pending", get_patches=[replaced])

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_update_replaced"
    assert error["details"]["replacing_release_id"] == "release-6"
    assert error["message"].endswith("was replaced by an update to release v6")


def test_a_stopping_target_is_refused_as_promote(monkeypatch) -> None:
    # Given
    client = FakeDeploy(
        [_live("dep-staging", release_id="release-5", revision=1), _live("dep-prod", status="stopping")]
    )

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    error = _envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_conflict"
    assert "`promote` will not move it" in error["message"]
    assert client.promote_calls == []


def test_a_repeat_whose_reply_lost_its_revision_still_changes_nothing(monkeypatch) -> None:
    # Given both on v5, and a reply and a read the rollout check left bare
    client = FakeDeploy(
        [_live("dep-staging", release_id="release-5", revision=1), _live("dep-prod", release_id="release-5")],
        get_patches=[{"revision": None}],
        strip_move_reply=True,
    )

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _envelope(result)["changed"] is False


def test_a_promote_that_leaves_a_stopped_target_unchanged_reports_it(monkeypatch) -> None:
    # Given production stopped, already on the source's release
    client = FakeDeploy(
        [
            _live("dep-staging", release_id="release-5", revision=1),
            _live("dep-prod", release_id="release-5", status="stopped"),
        ]
    )

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", output="--no-json")

    # Then it is not judged a failure, and the way to serve it is named
    assert result.exit_code == 0, result.stderr
    output = " ".join((result.stdout + result.stderr).split())
    assert "Deployment dep-prod already serves release v5." in output
    assert "`comfy deploy start --deployment dep-prod`" in output


def test_an_unconfirmed_promote_under_no_watch_says_it_waits(monkeypatch) -> None:
    # Given a reply and a read the rollout check left bare, still on v4
    client = _staging_and_production(move="pending", get_patches=[{"pendingUpdate": None, "revision": None}])
    client.strip_move_reply = True

    # When
    result = _promote(monkeypatch, client, "dep-staging", "dep-prod", "--no-watch", output="--no-json")

    # Then
    assert result.exit_code == 0, result.stderr
    assert "moves to release v5 once it is ready; release v4 serves until then" in result.stdout
    assert "now serves" not in result.stdout
