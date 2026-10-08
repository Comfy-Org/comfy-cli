from __future__ import annotations

import importlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import jsonschema
import pytest
from deploy_up_support import FakeBuilder, FakeDeploy, deployment, option_names, write_spec
from typer.testing import CliRunner

from comfy_cli.cmdline import app
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError


class RecordingDeploy(FakeDeploy):
    def __init__(self, rows: list[JsonObject] | None = None, *, get_statuses: list[str] | None = None) -> None:
        super().__init__(rows, get_statuses=get_statuses)
        self.list_calls = 0
        self.get_calls: list[str] = []

    def list_all_deployments(self) -> list[JsonObject]:
        self.list_calls += 1
        return super().list_all_deployments()

    def get_deployment(self, deployment_id: str) -> JsonObject:
        self.get_calls.append(deployment_id)
        return super().get_deployment(deployment_id)


def _release(version: int) -> JsonObject:
    return {"id": f"release-{version}", "buildId": "build-1", "version": version, "deployable": True}


def _status_deployment(release_id: str = "release-5", status: str = "ready") -> JsonObject:
    row = deployment("dep-status", release_id=release_id, status=status, maximum=2)
    row.update(
        {
            "releaseId": release_id,
            "endpointUrl": "https://dep-status.run.comfy.app" if status == "ready" else None,
            "error": None,
            "serving": None,
            "stopReason": None,
        }
    )
    return row


def _serving(*, idle: int = 0, unhealthy: int = 0) -> JsonObject:
    """A sample as the deploy service sends it now: capacity, and the provider's
    own counts beside it while they are deprecated."""
    return {
        "capacity": {"ready": idle, "busy": 0, "starting": 0},
        "workers": {
            "idle": idle,
            "initializing": 0,
            "ready": 0,
            "running": 0,
            "throttled": 0,
            "unhealthy": unhealthy,
        },
        "jobsInQueue": 0,
        "sampledAt": "2026-08-21T09:12:03Z",
    }


def _install_clients(monkeypatch, builder: FakeBuilder, client: RecordingDeploy, sleeps: list[float]) -> None:
    module = importlib.import_module("comfy_cli.command.deploy_status")
    monkeypatch.setattr(module, "_command_clients", lambda: (builder, client))
    monkeypatch.setattr(module, "_sleep", sleeps.append)


def _invoke_json(path: Path, *args: str):
    return CliRunner().invoke(app, ["--json", "deploy", "status", str(path), *args])


def _invoke_pretty(path: Path):
    return CliRunner().invoke(
        app,
        ["--no-json", "deploy", "status", str(path)],
        env={"COLUMNS": "400"},
    )


def _json_envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def test_deploy_status_is_a_registered_real_command() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "status", "--help"])

    # Then
    assert result.exit_code == 0
    options = option_names("status")
    assert "--watch" in options
    # `status` reaches the same ambiguous-deployment refusal the lifecycle verbs
    # do, and that error's hint names `--deployment`, so it has to accept one.
    assert "--deployment" in options
    assert "--release" not in options


def _schema(name: str) -> JsonObject:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_bounds_free_deployment_validates_against_the_published_status_schema(tmp_path, monkeypatch) -> None:
    """A deployment the web UI created stores no `min`/`max`, and `compute_config`
    carries only what the service stored. Requiring the bounds in the schema made
    `deploy status --json` fail its own contract on a perfectly ordinary row."""
    # Given
    row = _status_deployment()
    row["computeConfig"] = {"gpuClass": "l4", "region": "US-MO-2"}
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    data = _json_envelope(result)["data"]
    assert data["deployment"]["computeConfig"] == {"gpuClass": "l4", "region": "US-MO-2"}
    jsonschema.Draft202012Validator(_schema("deploy_status.json")).validate(data)


def test_a_named_deployment_is_the_one_status_reports_on(tmp_path, monkeypatch) -> None:
    # Given two live deployments of the same Build
    rows = [_status_deployment(), _status_deployment()]
    rows[1]["id"] = "dep-other"
    rows[1]["createdAt"] = "2026-08-24T12:00:00Z"
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), RecordingDeploy(rows), [])

    # When
    result = _invoke_json(write_spec(tmp_path), "--deployment", "dep-status")

    # Then the named row wins over the newer one the ranking would have picked
    assert result.exit_code == 0, result.stderr
    assert _json_envelope(result)["data"]["deployment"]["id"] == "dep-status"


def test_no_deployment_exits_zero_with_nullable_payload_and_up_hint(tmp_path, monkeypatch) -> None:
    # Given
    builder = FakeBuilder([_release(5)])
    client = RecordingDeploy()
    _install_clients(monkeypatch, builder, client, [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    payload = _json_envelope(result)["data"]
    assert payload == {
        "build": {"id": "build-1", "name": "example"},
        "deployment": None,
        "release": None,
        "serving": None,
    }
    assert "comfy deploy up" in result.stderr
    assert client.list_calls == 1
    assert client.get_calls == []
    assert builder.calls == [("list_releases", "build-1")]


def test_older_release_reports_behind_with_latest_deployable_and_new_url_hint(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment("release-3")
    row["serving"] = _serving(idle=1)
    builder = FakeBuilder([_release(3), _release(5)])
    _install_clients(monkeypatch, builder, RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    payload = _json_envelope(result)["data"]
    assert payload["release"] == {
        "id": "release-3",
        "version": 3,
        "behind": True,
        "latestDeployable": {"id": "release-5", "version": 5},
    }
    assert payload["serving"]["sampledAt"] == "2026-08-21T09:12:03Z"
    assert "creates a new deployment" in result.stderr.lower()
    assert "new url" in result.stderr.lower()


def test_a_deployment_inside_the_rollout_is_told_up_moves_it_keeping_its_url(tmp_path, monkeypatch) -> None:
    # Given a deployment whose read carries a revision, so the workspace has updates
    row = _status_deployment("release-3")
    row["revision"] = 4
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    stderr = " ".join(result.stderr.split())
    assert "moves this deployment to it, keeping its URL" in stderr
    assert "new url" not in stderr.lower()


def _waiting(row: JsonObject, release_id: str = "release-5", kind: object = "update") -> JsonObject:
    row["revision"] = 4
    row["pendingUpdate"] = {
        "releaseId": release_id,
        "baseRevision": 4,
        "status": "starting",
        "since": "2026-10-07T16:20:00Z",
        "kind": kind,
        "fromRevision": None,
    }
    return row


def test_status_shows_the_update_a_deployment_waits_on(tmp_path, monkeypatch) -> None:
    # Given a deployment on release 3 waiting for release 5's copy
    row = _waiting(_status_deployment("release-3"))
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then the update names its release by version, and the release still serving stays the current one
    assert result.exit_code == 0, result.stderr
    data = _json_envelope(result)["data"]
    assert data["update"] == {
        "release": {"id": "release-5", "version": 5},
        "status": "starting",
        "since": "2026-10-07T16:20:00Z",
        "kind": "update",
    }
    assert data["release"]["id"] == "release-3"
    jsonschema.Draft202012Validator(_schema("deploy_status.json")).validate(data)


def test_the_update_line_names_the_release_and_its_copy(tmp_path, monkeypatch) -> None:
    # Given
    row = _waiting(_status_deployment("release-3"), kind="rollback")
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    assert "Rolling back to release v5: its copy is starting" in result.stdout


def test_an_update_to_a_release_the_build_does_not_list_still_reads(tmp_path, monkeypatch) -> None:
    # Given a waiting update whose release the Build's list omits, and an unknown kind
    row = _waiting(_status_deployment("release-3"), release_id="release-9", kind=None)
    _install_clients(monkeypatch, FakeBuilder([_release(3)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then the release is named by id and the kind reads as an update
    assert result.exit_code == 0, result.stderr
    update = _json_envelope(result)["data"]["update"]
    assert update["release"] == {"id": "release-9"}
    assert update["kind"] == "update"


def test_a_settled_deployment_has_no_update_field(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment()
    row["revision"] = 2
    row["pendingUpdate"] = None
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    assert "update" not in _json_envelope(result)["data"]


class SummaryListDeploy(RecordingDeploy):
    """The list endpoint's summary shape: no `error` and no `serving`."""

    def list_all_deployments(self) -> list[JsonObject]:
        rows = super().list_all_deployments()
        for row in rows:
            row.pop("error", None)
            row.pop("serving", None)
        return rows


def test_status_reads_the_failed_deployments_error_the_list_omits(tmp_path, monkeypatch) -> None:
    """A failed deployment carries `error` and no `serving`: the control plane drops
    the sample once the deployment no longer claims compute, and `failed` never
    does. So the full read is what the failure reason rides in on, not the counts."""
    # Given
    row = _status_deployment(status="failed")
    row["error"] = "the model download was refused"
    client = SummaryListDeploy([row])
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), client, [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    data = _json_envelope(result)["data"]
    assert client.get_calls == ["dep-status"]
    assert data["deployment"]["error"] == "the model download was refused"
    assert data["serving"] is None


def test_status_reads_a_ready_deployments_worker_counts_the_list_omits(tmp_path, monkeypatch) -> None:
    """The other half of the full read: a ready deployment is sampled, and the
    counts live only on the single-deployment reply."""
    # Given
    row = _status_deployment()
    row["serving"] = _serving(idle=2)
    client = SummaryListDeploy([row])
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), client, [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    data = _json_envelope(result)["data"]
    assert client.get_calls == ["dep-status"]
    assert data["serving"]["capacity"] == {"ready": 2, "busy": 0, "starting": 0}
    assert data["serving"]["workers"]["idle"] == 2


def test_failed_deployment_prints_its_reason_in_the_terminal(tmp_path, monkeypatch) -> None:
    """The point of the ticket: someone reading the terminal, not `--json`, is the
    one who cannot tell why their deploy died."""
    # Given
    row = _status_deployment(status="failed")
    row["error"] = "model staging made no progress for 2m0s"
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), SummaryListDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 1
    output = result.stdout + result.stderr
    assert "Reason: model staging made no progress for 2m0s" in output


def test_a_healthy_deployment_prints_no_reason_line(tmp_path, monkeypatch) -> None:
    # Given
    _install_clients(monkeypatch, FakeBuilder([_release(5)]), SummaryListDeploy([_status_deployment()]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "Reason:" not in result.stdout + result.stderr


def test_null_serving_renders_not_sampled_yet(tmp_path, monkeypatch) -> None:
    # Given
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([_status_deployment()]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "not sampled yet" in result.stdout.lower()


def test_all_zero_serving_renders_healthy_scale_to_zero_idle(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment()
    row["serving"] = _serving()
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "healthy idle" in result.stdout.lower()
    assert "scale-to-zero" in result.stdout.lower()
    assert "not sampled yet" not in result.stdout.lower()


def test_serving_renders_sample_vintage_beside_worker_counts(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment()
    row["serving"] = _serving(idle=1)
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "ready=1" in serving_line
    assert "2026-08-21T09:12:03Z" in serving_line


def test_unhealthy_workers_are_named_and_never_read_as_healthy_idle(tmp_path, monkeypatch) -> None:
    """Capacity counts no failing worker, so a deployment whose workers all fail
    reads all-zero; the line must say so rather than call it healthy."""
    # Given
    row = _status_deployment()
    row["serving"] = _serving(unhealthy=3)
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "unhealthy=3" in serving_line
    assert "healthy idle" not in serving_line


def test_a_thinned_out_workers_object_beside_capacity_still_reads(tmp_path, monkeypatch) -> None:
    """The old counts are deprecated: beside capacity, a missing state in them
    must not fail the command."""
    # Given
    row = _status_deployment()
    row["serving"] = {
        "capacity": {"ready": 1, "busy": 0, "starting": 0},
        "workers": {"idle": 1},
        "jobsInQueue": 0,
        "sampledAt": "2026-08-21T09:12:03Z",
    }
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    assert _json_envelope(result)["data"]["serving"]["capacity"]["ready"] == 1
    jsonschema.Draft202012Validator(_schema("deploy_status.json")).validate(_json_envelope(result)["data"])


def test_an_unhealthy_deployment_at_zero_is_never_healthy_idle(tmp_path, monkeypatch) -> None:
    """Once the service stops sending the old counts, only the status can say a
    deployment is unhealthy, so the label must read it."""
    # Given
    row = _status_deployment(status="unhealthy")
    row["serving"] = {
        "capacity": {"ready": 0, "busy": 0, "starting": 0},
        "jobsInQueue": 0,
        "sampledAt": "2026-08-21T09:12:03Z",
    }
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "healthy idle" not in serving_line


@pytest.mark.parametrize(
    "serving",
    [
        {"capacity": {"ready": -1, "busy": 0, "starting": 0}},
        {"capacity": {"ready": True, "busy": 0, "starting": 0}},
        {"workers": {"idle": -1, "initializing": 0, "ready": 0, "running": 0, "throttled": 0, "unhealthy": 0}},
        {"capacity": {"ready": 0, "busy": 0, "starting": 0}, "workers": {"unhealthy": "3"}},
        {"capacity": {"ready": 0, "busy": 0, "starting": 0}, "workers": "unhealthy"},
    ],
    ids=[
        "negative capacity",
        "boolean capacity",
        "negative old count",
        "malformed present old count",
        "old counts not an object",
    ],
)
def test_a_malformed_count_is_a_shape_error_not_a_quiet_zero(tmp_path, monkeypatch, serving) -> None:
    """A count the schema would reject is refused, never dropped or passed on,
    since a dropped unhealthy count would read as healthy."""
    # Given
    row = _status_deployment()
    row["serving"] = {**serving, "jobsInQueue": 0, "sampledAt": "2026-08-21T09:12:03Z"}
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code != 0
    assert _json_envelope(result)["ok"] is False


def test_serving_prints_capacity_and_no_provider_state(tmp_path, monkeypatch) -> None:
    """The counts read the same on every GPU provider: RunPod's own states, such
    as a scale-to-zero endpoint's throttled slots, never reach the terminal."""
    # Given
    row = _status_deployment()
    row["serving"] = {
        "capacity": {"ready": 1, "busy": 2, "starting": 3},
        "workers": {"idle": 1, "initializing": 3, "ready": 1, "running": 2, "throttled": 3, "unhealthy": 0},
        "jobsInQueue": 4,
        "sampledAt": "2026-08-21T09:12:03Z",
    }
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "ready=1 busy=2 starting=3 queued=4" in serving_line
    for state in ("idle", "initializing", "running", "throttled", "unhealthy"):
        assert state not in serving_line


def test_serving_without_capacity_is_worked_out_from_the_old_counts(tmp_path, monkeypatch) -> None:
    """A deploy service older than the capacity field still answers: the three
    counts come from the provider's, with a warm worker RunPod counts as both
    idle and ready counted once."""
    # Given a sample with only the old counts
    row = _status_deployment()
    row["serving"] = {
        "workers": {"idle": 1, "initializing": 2, "ready": 1, "running": 3, "throttled": 4, "unhealthy": 0},
        "jobsInQueue": 0,
        "sampledAt": "2026-08-21T09:12:03Z",
    }
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    serving = _json_envelope(result)["data"]["serving"]
    assert serving["capacity"] == {"ready": 1, "busy": 3, "starting": 2}
    assert serving["workers"]["throttled"] == 4
    jsonschema.Draft202012Validator(_schema("deploy_status.json")).validate(_json_envelope(result)["data"])


def test_serving_without_the_old_counts_still_reads(tmp_path, monkeypatch) -> None:
    """Once the deploy service drops the deprecated counts, status keeps working."""
    # Given
    row = _status_deployment()
    row["serving"] = {
        "capacity": {"ready": 0, "busy": 1, "starting": 0},
        "jobsInQueue": 2,
        "sampledAt": "2026-08-21T09:12:03Z",
    }
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    serving = _json_envelope(result)["data"]["serving"]
    assert serving == {
        "capacity": {"ready": 0, "busy": 1, "starting": 0},
        "jobsInQueue": 2,
        "sampledAt": "2026-08-21T09:12:03Z",
    }
    jsonschema.Draft202012Validator(_schema("deploy_status.json")).validate(_json_envelope(result)["data"])


def test_serving_says_how_old_the_sample_is(tmp_path, monkeypatch) -> None:
    """A bare timestamp does not say whether the counts still describe now: the
    control plane refreshes on its own schedule, and a deployment whose endpoint
    stopped answering keeps the last sample it got. The age is what says so."""
    # Given a sample taken two hours ago
    row = _status_deployment()
    sampled = datetime.now(timezone.utc) - timedelta(hours=2, minutes=5)
    row["serving"] = {**_serving(idle=1), "sampledAt": sampled.strftime("%Y-%m-%dT%H:%M:%SZ")}
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "2h 5m ago" in serving_line


def test_a_sample_stamped_in_the_future_reads_as_just_now(tmp_path, monkeypatch) -> None:
    """A server clock a little ahead of ours must not print a negative age."""
    # Given
    row = _status_deployment()
    sampled = datetime.now(timezone.utc) + timedelta(minutes=3)
    row["serving"] = {**_serving(idle=1), "sampledAt": sampled.strftime("%Y-%m-%dT%H:%M:%SZ")}
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "(just now)" in serving_line


def test_an_unreadable_sampled_at_still_prints_the_counts(tmp_path, monkeypatch) -> None:
    """`sampledAt` is rendered, not just carried, so a timestamp the parser cannot
    read must degrade to an unknown age rather than raise out of the renderer,
    where nothing catches ValueError and the whole command would traceback."""
    # Given
    row = _status_deployment()
    row["serving"] = {**_serving(idle=1), "sampledAt": "last tuesday"}
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0, result.stderr
    serving_line = next(line for line in result.stdout.splitlines() if "sampled" in line.lower())
    assert "ready=1" in serving_line
    assert "age unknown" in serving_line


def test_unhealthy_is_recoverable_and_not_an_error(tmp_path, monkeypatch) -> None:
    # Given
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([_status_deployment(status="unhealthy")]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    assert result.exit_code == 0
    assert "recoverable" in result.stdout.lower()
    assert "failed" not in result.stdout.lower()


def test_stop_failed_is_loud_and_names_retry_stop_remedy(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment(status="stop_failed")
    row["error"] = "provider release failed"
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 1
    assert _json_envelope(result)["data"]["deployment"]["status"] == "stop_failed"
    assert "may still be billing" in result.stderr.lower()
    assert "comfy deploy stop --deployment" in result.stderr


def test_credit_stop_is_not_attributed_to_the_user(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment(status="stopped")
    row["stopReason"] = "credits"
    _install_clients(monkeypatch, FakeBuilder(), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    rendered = f"{result.stdout}\n{result.stderr}".lower()
    assert result.exit_code == 0
    assert "insufficient credits" in rendered
    assert "stopped by user" not in rendered
    assert "user-initiated" not in rendered


def test_watch_stops_at_unhealthy_and_calls_it_recoverable(tmp_path, monkeypatch) -> None:
    """`unhealthy` only follows `ready`: the deployment came up. Waiting on it
    for `ready` would wait silently for as long as the endpoint is degraded."""
    # Given
    client = RecordingDeploy([_status_deployment(status="queued")], get_statuses=["unhealthy", "ready"])
    sleeps: list[float] = []
    _install_clients(monkeypatch, FakeBuilder(), client, sleeps)

    # When
    result = _invoke_json(write_spec(tmp_path), "--watch")

    # Then
    assert result.exit_code == 0
    assert _json_envelope(result)["data"]["deployment"]["status"] == "unhealthy"
    assert client.get_calls == ["dep-status"]
    assert sleeps == []


def test_watch_rides_out_a_brief_503_from_the_deploy_api(tmp_path, monkeypatch) -> None:
    """The same rollout blip `up` rides out: a watch that ended on it exited 1
    for a deployment that went on to ready."""

    # Given a watch whose first two reads are refused with a 503
    class Flaky(RecordingDeploy):
        def get_deployment(self, deployment_id: str) -> JsonObject:
            if len(self.get_calls) < 2:
                self.get_calls.append(deployment_id)
                raise DeployAPIError(
                    "deploy_server_error",
                    "Service Unavailable",
                    status=503,
                    details={"operation": "get", "status": 503},
                )
            return super().get_deployment(deployment_id)

    client = Flaky([_status_deployment(status="provisioning")], get_statuses=["ready"])
    sleeps: list[float] = []
    _install_clients(monkeypatch, FakeBuilder(), client, sleeps)

    # When
    result = _invoke_json(write_spec(tmp_path), "--watch")

    # Then
    assert result.exit_code == 0, result.stderr
    assert _json_envelope(result)["data"]["deployment"]["status"] == "ready"
    assert result.stderr.count("deploy API is not answering") == 1
    assert client.get_calls == ["dep-status"] * 3
    assert sleeps == [2.0, 4.0]


def test_a_watch_the_deploy_api_never_answers_ends_with_the_watch_lost_exit_code(tmp_path, monkeypatch) -> None:
    # Given a watch whose reads are never answered
    class Down(RecordingDeploy):
        def get_deployment(self, deployment_id: str) -> JsonObject:
            self.get_calls.append(deployment_id)
            raise ConnectionResetError(54, "Connection reset by peer")

    client = Down([_status_deployment(status="provisioning")])
    sleeps: list[float] = []
    _install_clients(monkeypatch, FakeBuilder(), client, sleeps)

    # When
    result = _invoke_json(write_spec(tmp_path), "--watch")

    # Then
    assert result.exit_code == 75, result.stderr
    error = _json_envelope(result)["error"]
    assert error["code"] == "deploy_watch_lost"
    assert "comfy deploy status --deployment dep-status --watch" in error["hint"]
    assert len(client.get_calls) == 7


def test_a_watch_that_gives_up_takes_the_live_line_down_before_the_error_prints(tmp_path, monkeypatch) -> None:
    """An error printed while the live line is up lands under a spinner that is still turning."""
    # Given a watch that draws its line, then never hears back. CliRunner's
    # stream is not a terminal, so the reporter records its line, not draws it.
    module = importlib.import_module("comfy_cli.command.deploy_status")
    events: list[str] = []

    class Reporter(module.DeployWatchReporter):
        def snapshot(self, deployment: JsonObject) -> None:
            events.append("line drawn")

        def close(self) -> None:
            events.append("line closed")
            super().close()

    exit_watch_lost = module.exit_watch_lost

    def recorded_exit_watch_lost(renderer, error):
        events.append("error printed")
        exit_watch_lost(renderer, error)

    class DownAfterOneRead(RecordingDeploy):
        def get_deployment(self, deployment_id: str) -> JsonObject:
            if self.get_calls:
                self.get_calls.append(deployment_id)
                raise ConnectionResetError(54, "Connection reset by peer")
            return super().get_deployment(deployment_id)

    monkeypatch.setattr(module, "DeployWatchReporter", Reporter)
    monkeypatch.setattr(module, "exit_watch_lost", recorded_exit_watch_lost)
    client = DownAfterOneRead([_status_deployment(status="provisioning")])
    _install_clients(monkeypatch, FakeBuilder(), client, [])

    # When
    result = CliRunner().invoke(
        app, ["--no-json", "deploy", "status", str(write_spec(tmp_path)), "--watch"], env={"COLUMNS": "400"}
    )

    # Then
    assert result.exit_code == 75, result.output
    assert "may still be coming up" in result.output
    printed = events.index("error printed")
    assert "line drawn" in events[:printed]
    assert events[printed - 1] == "line closed"


def test_watch_exits_promptly_on_stop_failed_with_retry_stop_hint(tmp_path, monkeypatch) -> None:
    # Given
    client = RecordingDeploy([_status_deployment(status="stopping")], get_statuses=["stop_failed", "ready"])
    sleeps: list[float] = []
    _install_clients(monkeypatch, FakeBuilder(), client, sleeps)

    # When
    result = _invoke_json(write_spec(tmp_path), "--watch")

    # Then
    assert result.exit_code == 1
    assert _json_envelope(result)["data"]["deployment"]["status"] == "stop_failed"
    assert "comfy deploy stop --deployment" in result.stderr
    assert client.get_calls == ["dep-status"]
    assert client.get_statuses == ["ready"]
    assert sleeps == []


def test_no_behind_hint_while_the_move_to_the_newest_release_waits(tmp_path, monkeypatch) -> None:
    # Given a deployment on release 3 already moving to release 5, the newest
    row = _waiting(_status_deployment("release-3"))
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then the update line stands alone, with no hint to start it
    assert result.exit_code == 0, result.stderr
    assert "Updating to release v5" in result.stdout
    assert "is deployable" not in result.stdout + result.stderr


def test_a_waiting_update_with_an_unreadable_since_is_a_shape_error(tmp_path, monkeypatch) -> None:
    # Given
    row = _waiting(_status_deployment("release-3"))
    row["pendingUpdate"]["since"] = "yesterday"
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert result.exit_code == 1
    assert _json_envelope(result)["error"]["code"] == "deploy_server_error"


def test_the_interrupt_fallback_survives_a_malformed_waiting_update() -> None:
    # Given a last read whose pendingUpdate made the full result fail
    module = importlib.import_module("comfy_cli.command.deploy_status")
    row = _waiting(_status_deployment("release-3"))
    row["pendingUpdate"]["since"] = "yesterday"
    target = module.StatusTarget("build-1", "example", row)

    # When
    result = module._interrupted_result(FakeBuilder([_release(3), _release(5)]), target)

    # Then the interrupt still reports the deployment, without the update
    assert result.deployment["id"] == "dep-status"
    assert result.update is None


def test_the_behind_hint_waits_for_an_update_to_an_older_release(tmp_path, monkeypatch) -> None:
    # Given a deployment on release 3 waiting on release 4, with release 5 the newest
    row = _waiting(_status_deployment("release-3"), release_id="release-4")
    builder = FakeBuilder([_release(3), _release(4), _release(5)])
    _install_clients(monkeypatch, builder, RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then the hint does not send someone to a move the service would refuse now
    output = result.stdout + result.stderr
    assert result.exit_code == 0, result.stderr
    assert "once the waiting update lands or fails" in output
    assert "keeping its URL" not in output


def test_the_behind_hint_names_the_deployment_up_should_move(tmp_path, monkeypatch) -> None:
    # Given
    row = _status_deployment("release-3")
    row["revision"] = 4
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_json(write_spec(tmp_path))

    # Then
    assert "`comfy deploy up --deployment dep-status` moves this deployment" in " ".join(result.stderr.split())


def test_a_failed_update_to_the_newest_release_says_so_and_keeps_the_hint(tmp_path, monkeypatch) -> None:
    # Given the waiting update to release 5, the newest, failed
    row = _waiting(_status_deployment("release-3"))
    row["pendingUpdate"]["status"] = "failed"
    _install_clients(monkeypatch, FakeBuilder([_release(3), _release(5)]), RecordingDeploy([row]), [])

    # When
    result = _invoke_pretty(write_spec(tmp_path))

    # Then
    output = " ".join((result.stdout + result.stderr).split())
    assert result.exit_code == 0, result.stderr
    assert "The update to release v5 failed" in output
    assert "release v5 is deployable" in output
    assert "the waiting update failed: read `comfy deploy events --deployment dep-status`" in output
    assert "lands or fails" not in output
