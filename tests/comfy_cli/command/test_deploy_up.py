from __future__ import annotations

import http.client
import importlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType

import jsonschema
import pytest
from deploy_up_support import FakeBuilder, FakeDeploy, deployment, option_names, write_spec
from typer.testing import CliRunner

from comfy_cli.caller import Caller
from comfy_cli.cmdline import app
from comfy_cli.deploy_api import DeployClient, _validate_compute_config
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge


def _deploy() -> ModuleType:
    return importlib.import_module("comfy_cli.command.deploy")


def _request(module: ModuleType, **changes):
    values = {
        "release": {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
        "build_id": "build-1",
        "gpu": "l4",
        "region": "US-MO-2",
        "minimum": 0,
        "maximum": 1,
    }
    values.update(changes)
    return module.UpRequest(**values)


def _json_envelope(result) -> dict:
    return json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])


def test_deploy_up_is_a_registered_real_command() -> None:
    # Given / When
    result = CliRunner().invoke(app, ["deploy", "up", "--help"])

    # Then
    assert result.exit_code == 0
    assert {"--gpu", "--region", "--deployment"} <= option_names("up")


@pytest.mark.parametrize(("flag", "missing"), [("--min", "--max"), ("--max", "--min")], ids=["min_alone", "max_alone"])
def test_up_refuses_one_worker_bound_without_the_other(tmp_path, monkeypatch, flag: str, missing: str) -> None:
    """Refused before any client is built, so a bad pair costs no round trip."""
    # Given
    module = _deploy()

    def _unreachable():
        raise AssertionError("the refusal must precede any API call")

    monkeypatch.setattr(module, "_command_clients", _unreachable)

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path)), flag, "3"])

    # Then
    error = _json_envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_missing_input"
    assert error["details"]["missing"] == [missing]


def _schema(name: str) -> dict:
    path = Path(__file__).parent.parent.parent.parent / "comfy_cli" / "schemas" / name
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_bounds_free_deployment_validates_against_the_published_up_schema(tmp_path, monkeypatch) -> None:
    """`up` reconciling a web-UI deployment emits the bounds the service stored,
    which for that row is neither. Requiring them made `deploy up --json` fail
    its own published contract on an otherwise ordinary reconcile."""
    # Given a live deployment the service stored without worker bounds
    module = _deploy()
    row = deployment("dep-live", status="ready")
    row["computeConfig"] = {"gpuClass": "l4", "region": "US-MO-2"}
    client = FakeDeploy([row])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 0, result.stderr
    data = _json_envelope(result)["data"]
    assert data["computeConfig"] == {"gpuClass": "l4", "region": "US-MO-2"}
    jsonschema.Draft202012Validator(_schema("deploy_up.json")).validate(data)


def test_up_edits_the_named_deployment_not_the_highest_ranked_one(tmp_path, monkeypatch) -> None:
    """`--deployment` has to reach the selection, not merely be accepted.

    Two live deployments on one release is exactly the state
    `deploy_ambiguous_deployment` tells the user to resolve this way. If the id
    stops being forwarded, ranking silently picks the newer row and the edit
    lands on a deployment the user did not name, at exit 0.
    """
    # Given two ready deployments on the reconciled release, `dep-2` the newer
    module = _deploy()
    client = FakeDeploy(
        [
            deployment("dep-1", status="ready", minimum=0, maximum=1),
            deployment("dep-2", status="ready", minimum=0, maximum=1),
        ]
    )
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When the older one is named explicitly
    result = CliRunner().invoke(
        app,
        ["--json", "deploy", "up", str(write_spec(tmp_path)), "--deployment", "dep-1", "--min", "2", "--max", "4"],
    )

    # Then the edit lands on it, and the ranking's pick is untouched
    assert result.exit_code == 0, result.stderr
    assert _json_envelope(result)["data"]["deployment"]["id"] == "dep-1"
    assert client.update_calls == ["dep-1"]
    assert client.rows["dep-2"]["computeConfig"] == {"gpuClass": "l4", "region": "US-MO-2", "min": 0, "max": 1}


def test_a_named_deployment_that_matches_nothing_never_creates_one(tmp_path, monkeypatch) -> None:
    """Naming a deployment is not a request to make one.

    The release filter can be empty while `--deployment` still names something,
    and returning "no deployment" there fell through to the create branch — so a
    typo'd or stale id answered with a second, billable deployment at exit 0.
    """
    # Given a Build whose selected release has no live deployment
    module = _deploy()
    client = FakeDeploy([])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When
    # When compute is supplied, so the create branch would otherwise succeed
    result = CliRunner().invoke(
        app,
        [
            "--json",
            "deploy",
            "up",
            str(write_spec(tmp_path)),
            "--deployment",
            "dep-typo",
            "--gpu",
            "l4",
            "--region",
            "US-MO-2",
        ],
    )

    # Then
    error = _json_envelope(result)["error"]
    assert result.exit_code == 1
    assert error["code"] == "deploy_unrelated_deployment"
    assert error["details"]["deploymentId"] == "dep-typo"
    assert error["details"]["scope"] == "the live deployments of release release-5 of Build build-1"
    assert error["details"]["candidateIds"] == []
    assert "drop `--deployment`" in error["hint"]
    assert client.create_keys == []
    assert client.rows == {}


def test_idempotency_key_is_pinned_for_a_known_generation() -> None:
    # Given
    module = _deploy()

    # When
    key = module._idempotency_key("build-1", "release-2", 3)

    # Then
    assert str(module._IDEMPOTENCY_NAMESPACE) == "86e81377-21c8-5a10-9db8-33797ad495f1"
    assert key == "a7f98f5d-9168-5357-840b-bf42b8677e85"
    assert (
        module._soft_deleted_generation(
            [deployment("dep-failed", status="failed"), deployment("dep-stopped", status="stopped")],
            "release-5",
        )
        == 0
    )


def test_second_identical_up_reconciles_without_a_second_post() -> None:
    # Given
    module = _deploy()
    builder = FakeBuilder()
    client = FakeDeploy()
    request = _request(module)

    # When
    first = module.reconcile_up(builder, client, request)
    second = module.reconcile_up(builder, client, request)

    # Then
    assert first.created is True
    assert second.created is False
    assert first.deployment["id"] == second.deployment["id"]
    assert len(client.create_keys) == 1
    assert second.supersedes == []


def test_two_concurrent_identical_calls_use_one_generation_and_one_row() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(generation_barrier=threading.Barrier(2))
    builder = FakeBuilder()
    request = _request(module)

    # When
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: module.reconcile_up(builder, client, request), range(2)))

    # Then
    assert {result.deployment["id"] for result in results} == {"dep-1"}
    assert len(client.rows) == 1
    assert client.create_keys == [module._idempotency_key("build-1", "release-5", 0)] * 2
    assert client.generation_deleted_counts == [0, 0]


def test_delete_then_up_creates_a_distinct_generation() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy()
    request = _request(module)
    first = module.reconcile_up(FakeBuilder(), client, request)
    client.soft_delete(first.deployment["id"])

    # When
    second = module.reconcile_up(FakeBuilder(), client, request)

    # Then
    assert second.deployment["id"] != first.deployment["id"]
    assert client.create_keys == [
        module._idempotency_key("build-1", "release-5", 0),
        module._idempotency_key("build-1", "release-5", 1),
    ]


def test_tombstone_race_recomputes_generation_and_retries_to_a_live_row() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(tombstone_first_create=True)

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module))

    # Then
    assert result.deployment["deletedAt"] is None
    assert client.create_keys == [
        module._idempotency_key("build-1", "release-5", 0),
        module._idempotency_key("build-1", "release-5", 1),
    ]
    assert client.generation_deleted_counts[-2:] == [0, 1]


def test_three_concurrent_delete_invalidations_fail_with_deploy_conflict() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(tombstone_all_creates=True)

    # When / Then
    with pytest.raises(DeployAPIError) as raised:
        module.reconcile_up(FakeBuilder(), client, _request(module))
    assert raised.value.code == "deploy_conflict"
    assert raised.value.details["attempts"] == 3
    assert len(client.create_keys) == 3


def test_failed_deployment_is_started_and_never_recreated() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-failed", status="failed")])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module))

    # Then
    assert result.created is False
    assert client.start_calls == ["dep-failed"]
    assert client.create_keys == []


def test_reconcile_patches_only_mutable_worker_bounds() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", minimum=1, maximum=2)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=0, maximum=4))

    # Then
    assert result.created is False
    assert result.changed is True
    assert client.update_calls == ["dep-live"]
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 0, "max": 4}


def test_omitted_worker_bounds_keep_the_live_scale() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", minimum=2, maximum=8)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=None, maximum=None))

    # Then
    assert result.changed is False
    assert client.update_calls == []
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 2, "max": 8}


def test_explicit_zero_and_one_still_unscale_a_live_deployment() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", minimum=2, maximum=8)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=0, maximum=1))

    # Then
    assert result.changed is True
    assert client.update_calls == ["dep-live"]
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 0, "max": 1}


def test_one_omitted_bound_leaves_only_that_bound_alone() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", minimum=2, maximum=8)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=None, maximum=4))

    # Then
    assert client.update_calls == ["dep-live"]
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 2, "max": 4}


def test_a_created_deployment_falls_back_to_the_documented_bounds() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy()

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=None, maximum=None))

    # Then
    assert result.created is True
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 0, "max": 1}


def test_a_created_deployment_lifts_the_default_ceiling_to_the_requested_floor() -> None:
    """The library-level contract behind the CLI's paired bounds.

    `_require_paired_bounds` stops a lone `--min` reaching `reconcile_up` from
    the command line, so this pins the guarantee for direct callers: an omitted
    ceiling clears the requested floor rather than letting the create be refused
    against the placeholder maximum of 1."""
    # Given
    module = _deploy()
    client = FakeDeploy()

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=3, maximum=None))

    # Then
    assert result.created is True
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 3, "max": 3}
    _validate_compute_config(result.compute_config)


def test_a_bare_up_after_a_scale_does_not_unscale_the_deployment(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", minimum=2, maximum=8)])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 0
    envelope = _json_envelope(result)
    assert envelope["changed"] is False
    assert envelope["data"]["computeConfig"] == {"gpuClass": "l4", "region": "US-MO-2", "min": 2, "max": 8}
    assert client.update_calls == []


@pytest.mark.parametrize("status", ["stopped", "failed", "stop_failed"])
def test_bounds_supplied_to_a_restart_are_reported_rather_than_dropped(status: str) -> None:
    """These branches hand back the live compute untouched, so a bound the caller
    actually typed is discarded. Silently discarding explicit input is the same
    defect as silently resetting it."""
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", status=status, minimum=2, maximum=8)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=3, maximum=5))

    # Then
    assert result.dropped_bounds == ("--min", "--max")
    assert result.compute_config == {"gpuClass": "l4", "region": "US-MO-2", "min": 2, "max": 8}
    assert client.update_calls == []


def test_a_restart_reports_only_the_bound_that_would_have_changed() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", status="stopped", minimum=2, maximum=8)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=2, maximum=5))

    # Then
    assert result.dropped_bounds == ("--max",)


def test_a_restart_without_bounds_reports_nothing() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", status="stopped", minimum=2, maximum=8)])

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module, minimum=None, maximum=None))

    # Then
    assert result.dropped_bounds == ()


def test_the_dropped_bound_warning_reaches_a_json_caller_on_stderr(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", status="stopped", minimum=2, maximum=8)])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When
    # --no-watch because this is about the warning, not the wait. Restarting a
    # stopped deployment leaves it `queued`, and watch (now the default) polls a
    # fake that never leaves that status, so the run would never end.
    result = CliRunner().invoke(
        app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--min", "3", "--max", "8", "--no-watch"]
    )

    # Then
    assert "--min had no effect" in result.stderr
    assert "comfy deploy scale --deployment" in result.stderr
    assert _json_envelope(result)["data"]["computeConfig"]["min"] == 2


def test_a_stop_failed_deployment_is_not_pointed_at_a_scale_that_would_bounce(tmp_path, monkeypatch) -> None:
    """`deploy scale` is rejected unless the deployment is ready or stopped, so
    offering it here would send the user into a `deploy_conflict`. The stop
    remedy warned about just above is the actionable one."""
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", status="stop_failed", minimum=2, maximum=8)])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--min", "3", "--max", "8"])

    # Then
    assert "--min had no effect" in result.stderr
    assert "comfy deploy scale" not in result.stderr
    assert "comfy deploy stop --deployment" in result.stderr


def test_up_warns_that_an_older_release_deployment_is_still_billing(tmp_path, monkeypatch) -> None:
    """A new release gets a new deployment and the old one keeps billing. Only
    the JSON `supersedes` array said so; the terminal has to say it too."""
    # Given a ready deployment on release 3 and the live one on release 5
    module = _deploy()
    releases = [
        {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
        {"id": "release-3", "buildId": "build-1", "version": 3, "deployable": True},
    ]
    client = FakeDeploy([deployment("dep-live"), deployment("dep-old", release_id="release-3", status="ready")])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(releases), client))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path))])

    # Then the old deployment is named with the command that stops it
    assert result.exit_code == 0, result.stderr
    assert "Deployment dep-old (release v3, ready) is still running and billing." in result.stderr
    assert "comfy deploy stop --deployment dep-old" in result.stderr
    assert [row["id"] for row in _json_envelope(result)["data"]["supersedes"]] == ["dep-old"]


def test_a_person_at_the_terminal_sees_the_billing_warning(tmp_path, monkeypatch) -> None:
    # Given a ready deployment on release 3 and the live one on release 5
    module = _deploy()
    releases = [
        {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
        {"id": "release-3", "buildId": "build-1", "version": 3, "deployable": True},
    ]
    client = FakeDeploy([deployment("dep-live"), deployment("dep-old", release_id="release-3", status="ready")])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(releases), client))

    # When run with human output
    result = CliRunner().invoke(app, ["deploy", "up", str(write_spec(tmp_path))], env={"COMFY_OUTPUT": "pretty"})

    # Then the warning is on stdout, after the deployment's own line
    assert result.exit_code == 0, result.stderr
    assert result.stdout.index("Deployment dep-live: ready") < result.stdout.index(
        "Deployment dep-old (release v3, ready) is still running and billing."
    )
    assert "comfy deploy stop --deployment dep-old" in result.stdout


def test_up_with_nothing_else_running_prints_no_billing_warning(tmp_path, monkeypatch) -> None:
    # Given only the live deployment of this release
    module = _deploy()
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), FakeDeploy([deployment("dep-live")])))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 0, result.stderr
    assert "still running and billing" not in result.stderr
    assert _json_envelope(result)["data"]["supersedes"] == []


def test_reconcile_rejects_an_immutable_gpu_change() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live")])

    # When / Then
    with pytest.raises(DeployAPIError) as raised:
        module.reconcile_up(FakeBuilder(), client, _request(module, gpu="a100"))
    assert raised.value.code == "deploy_immutable_compute"
    assert client.update_calls == []
    assert client.create_keys == []


def test_supersedes_uses_only_the_server_committed_worker_statuses() -> None:
    # Given
    module = _deploy()
    releases = [
        {"id": "release-5", "buildId": "build-1", "version": 5, "deployable": True},
        {"id": "release-3", "buildId": "build-1", "version": 3, "deployable": True},
    ]
    statuses = [
        "queued",
        "provisioning",
        "starting",
        "ready",
        "unhealthy",
        "stopping",
        "stop_failed",
        "stopped",
        "failed",
    ]
    rows = [deployment(f"dep-{index}", release_id="release-3", status=status) for index, status in enumerate(statuses)]
    rows.append(deployment("dep-deleted", release_id="release-3", status="ready", deleted_at="2026-08-23T13:00:00Z"))

    # When
    result = module.reconcile_up(FakeBuilder(releases), FakeDeploy(rows), _request(module))

    # Then
    assert {row["status"] for row in result.supersedes} == {
        "queued",
        "provisioning",
        "starting",
        "ready",
        "unhealthy",
    }
    assert "dep-deleted" not in {row["id"] for row in result.supersedes}


def test_agentic_create_names_both_missing_compute_options(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), FakeDeploy()))

    # When
    result = CliRunner().invoke(app, ["--json", "deploy", "up", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 1
    envelope = _json_envelope(result)
    assert envelope["error"]["code"] == "deploy_missing_input"
    assert envelope["error"]["details"]["missing"] == ["--gpu", "--region"]


def test_tty_create_prompts_with_compute_catalog_choices(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy()
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))
    monkeypatch.setattr("comfy_cli.interaction.detect_caller", lambda: Caller("user", agentic=False, source_env=None))
    monkeypatch.setattr("comfy_cli.interaction._skip_prompt_flag", lambda: False)
    selected = iter(["l4", "US-MO-2"])
    questions: list[str] = []

    def choose(question, choices, default="", force_prompting=False):
        questions.append(question)
        assert choices
        return next(selected)

    monkeypatch.setattr("comfy_cli.ui.prompt_select", choose)

    # When
    result = CliRunner().invoke(app, ["--no-json", "deploy", "up", str(write_spec(tmp_path))])

    # Then
    assert result.exit_code == 0
    assert questions == ["GPU class", "Region"]
    assert client.catalog_calls == 2


def test_watch_exits_immediately_on_stop_failed_with_stop_remedy(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(get_statuses=["queued", "stop_failed"])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))
    sleeps: list[float] = []
    monkeypatch.setattr(module, "_sleep", sleeps.append)

    # When
    result = CliRunner().invoke(
        app,
        ["--json", "deploy", "up", str(write_spec(tmp_path)), "--gpu", "l4", "--region", "US-MO-2", "--watch"],
    )

    # Then
    assert result.exit_code == 1
    assert _json_envelope(result)["data"]["deployment"]["status"] == "stop_failed"
    assert "comfy deploy stop --deployment" in result.stderr
    assert sleeps == []


def test_watch_stops_at_unhealthy_and_reports_it_as_not_ok(tmp_path, monkeypatch) -> None:
    # Given a deployment the watch reads as unhealthy
    module = _deploy()
    client = FakeDeploy(get_statuses=["queued", "unhealthy", "ready"])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))
    sleeps: list[float] = []
    monkeypatch.setattr(module, "_sleep", sleeps.append)

    # When
    result = CliRunner().invoke(
        app,
        ["--json", "deploy", "up", str(write_spec(tmp_path)), "--gpu", "l4", "--region", "US-MO-2", "--watch"],
        env={"COLUMNS": "400"},
    )

    # Then: `unhealthy` only follows `ready`, so the watch ends there, loudly
    assert result.exit_code == 1
    envelope = _json_envelope(result)
    assert envelope["data"]["deployment"]["status"] == "unhealthy"
    assert envelope["error"]["code"] == "deploy_status_terminal"
    assert envelope["error"]["details"]["status"] == "unhealthy"
    assert "still billing" in result.stderr
    assert "comfy deploy stop --deployment dep-1" in result.stderr
    assert sleeps == []


def test_up_on_a_deployment_already_unhealthy_does_not_wait_for_ever(tmp_path, monkeypatch) -> None:
    """`up` leaves an unhealthy deployment as it is, so a watch that waited for
    `ready` would poll for as long as the endpoint stays degraded, saying
    nothing."""
    # Given the release's deployment is unhealthy and stays so
    module = _deploy()
    client = FakeDeploy([deployment("dep-1", status="unhealthy")])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) > 5:
            raise AssertionError("the watch kept polling an unhealthy deployment")

    monkeypatch.setattr(module, "_sleep", sleep)

    # When
    result = CliRunner().invoke(app, ["--no-json", "deploy", "up", str(write_spec(tmp_path))], env={"COLUMNS": "400"})

    # Then
    assert result.exception is None or isinstance(result.exception, SystemExit), result.exception
    assert result.exit_code == 1
    assert sleeps == []
    assert "Deployment dep-1 is unhealthy" in result.stdout + result.stderr
    assert client.start_calls == [] and client.update_calls == []


_GIB = 1024**3
_ESTIMATE = {
    "enabled": True,
    "bytesTotal": 42 * _GIB,
    "bytesHeld": 0,
    "bytesToFetch": 42 * _GIB,
    "measured": True,
    "unsizedModelCount": 0,
    "atLeast": False,
    "stagingSecondsLow": 444,
    "stagingSecondsHigh": 4260,
    "startupSecondsLow": 60,
    "startupSecondsHigh": 600,
    "etaSecondsLow": 504,
    "etaSecondsHigh": 4860,
}


def test_a_create_asks_for_the_estimate_with_its_release_and_compute() -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(estimate=_ESTIMATE)

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module))

    # Then
    assert client.estimate_calls == [("release-5", "l4", "US-MO-2")]
    assert result.estimate == _ESTIMATE
    assert result.payload()["estimate"] == _ESTIMATE


@pytest.mark.parametrize("status", ["ready", "stopped"])
def test_a_reconcile_that_creates_nothing_asks_for_no_estimate(status: str) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy([deployment("dep-live", status=status)], estimate=_ESTIMATE)

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module))

    # Then
    assert client.estimate_calls == []
    assert result.estimate is None
    assert "estimate" not in result.payload()


@pytest.mark.parametrize(
    "answer",
    [
        DeployAPIError("deploy_not_found", "no route", status=404),
        DeployAPIError("deploy_server_error", "boom", status=503),
        {**_ESTIMATE, "etaSecondsHigh": None},
        {**_ESTIMATE, "bytesToFetch": -1},
        {**_ESTIMATE, "etaSecondsLow": True},
        {**_ESTIMATE, "etaSecondsLow": 3600, "etaSecondsHigh": 60},
        {**_ESTIMATE, "bytesTotal": -1},
        {**_ESTIMATE, "bytesHeld": "0"},
        {**_ESTIMATE, "atLeast": "yes"},
        ConnectionResetError("peer reset"),
        ResponseTooLarge("https://deploy.test/v1/deploy-estimate is over the cap"),
        http.client.IncompleteRead(b'{"etaSecondsLow": 5', 40),
    ],
    ids=[
        "no_route",
        "server_error",
        "missing_number",
        "negative",
        "bool",
        "runs_backwards",
        "negative_total",
        "held_a_string",
        "flag_a_string",
        "connection_reset",
        "too_large",
        "cut_short",
    ],
)
def test_a_create_the_service_cannot_estimate_still_deploys(answer) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(estimate=answer)

    # When
    result = module.reconcile_up(FakeBuilder(), client, _request(module))

    # Then
    assert result.created is True
    assert result.estimate is None
    assert len(client.create_keys) == 1


class _CutShortEstimate(BaseHTTPRequestHandler):
    # Promises a 64-byte chunk, sends part of it, then hangs up.
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        self.wfile.write(b'40\r\n{"etaSecondsLow": 5')
        self.wfile.flush()
        self.close_connection = True

    def log_message(self, *args) -> None:
        pass


class _EstimateFromServer(FakeDeploy):
    def __init__(self, base_url: str) -> None:
        super().__init__()
        self._real = DeployClient(base_url, "token")

    def get_deploy_estimate(self, release_id: str, gpu_class: str, region: str):
        return self._real.get_deploy_estimate(release_id, gpu_class, region)


def test_an_estimate_response_cut_short_on_the_wire_still_deploys() -> None:
    # Given
    module = _deploy()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _CutShortEstimate)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = _EstimateFromServer(f"http://127.0.0.1:{server.server_address[1]}")

        # When
        result = module.reconcile_up(FakeBuilder(), client, _request(module))
    finally:
        server.shutdown()
        server.server_close()

    # Then
    assert result.created is True
    assert result.estimate is None
    assert len(client.create_keys) == 1


def test_the_estimate_validates_against_the_published_up_schema(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(estimate=_ESTIMATE)
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))

    # When
    result = CliRunner().invoke(
        app, ["--json", "deploy", "up", str(write_spec(tmp_path)), "--gpu", "l4", "--region", "US-MO-2"]
    )

    # Then
    assert result.exit_code == 0, result.stderr
    data = _json_envelope(result)["data"]
    assert data["estimate"]["etaSecondsHigh"] == 4860
    jsonschema.Draft202012Validator(_schema("deploy_up.json")).validate(data)


def test_a_person_sees_the_estimate_before_the_watch_starts(tmp_path, monkeypatch) -> None:
    # Given
    module = _deploy()
    client = FakeDeploy(estimate=_ESTIMATE, get_statuses=["queued", "ready"])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), client))
    monkeypatch.setattr(module, "_sleep", lambda _: None)

    # When
    result = CliRunner().invoke(
        app,
        ["--no-json", "deploy", "up", str(write_spec(tmp_path)), "--gpu", "l4", "--region", "US-MO-2", "--watch"],
    )

    # Then
    assert result.exit_code == 0, result.output
    assert "Expected ready in 8-81 min (42.0 GB of models to download)." in result.output
    assert result.output.index("Expected ready") < result.output.index("Deployment dep-1: ready")


@pytest.mark.parametrize(
    ("changes", "line"),
    [
        ({}, "Expected ready in 8-81 min (42.0 GB of models to download)."),
        ({"atLeast": True}, "Expected ready in 8-81 min (at least 42.0 GB of models to download)."),
        (
            {"bytesToFetch": 0, "etaSecondsLow": 120, "etaSecondsHigh": 900},
            "Expected ready in 2-15 min (nothing to download).",
        ),
        (
            {"etaSecondsLow": 3000, "etaSecondsHigh": 4 * 3600 + 60},
            "Expected ready in 0.5-4.5 h (42.0 GB of models to download).",
        ),
        (
            {"etaSecondsLow": 840, "etaSecondsHigh": 7800},
            "Expected ready in 14-130 min (42.0 GB of models to download).",
        ),
        ({"etaSecondsLow": 60, "etaSecondsHigh": 60}, "Expected ready in 1 min (42.0 GB of models to download)."),
        ({"etaSecondsLow": 30, "etaSecondsHigh": 45}, "Expected ready in 0-1 min (42.0 GB of models to download)."),
        (
            {"measured": False, "atLeast": True, "bytesToFetch": 0},
            "Expected ready in 8-81 min (the release's models were never measured, so this counts starting the endpoint alone).",
        ),
    ],
    ids=[
        "sized",
        "at_least",
        "nothing_to_fetch",
        "hours",
        "short_end_under_half_an_hour",
        "one_number",
        "short_end_under_a_minute",
        "unmeasured",
    ],
)
def test_estimate_line_rounds_outward_and_says_what_downloads(changes, line) -> None:
    from comfy_cli.command.deploy_up import estimate_line

    assert estimate_line({**_ESTIMATE, **changes}) == line


@pytest.mark.parametrize(
    "answer",
    [
        {"enabled": False},
        {**_ESTIMATE, "enabled": False},
        {k: v for k, v in _ESTIMATE.items() if k != "enabled"},
        {**_ESTIMATE, "enabled": None},
        {**_ESTIMATE, "enabled": "true"},
    ],
    ids=["bare", "with_numbers", "enabled_missing", "enabled_null", "enabled_a_string"],
)
def test_a_switched_off_estimate_prints_nothing_and_adds_nothing(tmp_path, monkeypatch, answer) -> None:
    """``enabled: false`` is the service saying the estimate is off, not a failure:
    no line for a person, no ``estimate`` under --json, no warning either way. An
    ``enabled`` that is not ``true`` reads the same way, as the site reads it."""
    # Given
    module = _deploy()
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), FakeDeploy(estimate=answer)))
    args = ["deploy", "up", str(write_spec(tmp_path)), "--gpu", "l4", "--region", "US-MO-2"]

    # When
    pretty = CliRunner().invoke(app, ["--no-json", *args])
    monkeypatch.setattr(module, "_command_clients", lambda: (FakeBuilder(), FakeDeploy(estimate=answer)))
    as_json = CliRunner().invoke(app, ["--json", *args])

    # Then
    assert pretty.exit_code == 0, pretty.output
    assert "Expected ready" not in pretty.output
    assert "estimate" not in pretty.output.lower()
    assert as_json.exit_code == 0, as_json.stderr
    assert "estimate" not in _json_envelope(as_json)["data"]
    assert as_json.stderr.strip() == ""
