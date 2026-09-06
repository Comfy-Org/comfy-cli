"""Resolve and render the current Build's deployment status."""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass, replace
from typing import Final

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError, resolve_build_paths
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, read_build_spec
from comfy_cli.command.deploy_resolve import (
    _STATUS_RANK,
    BuilderReleaseClient,
    BuildNotPushedError,
    DeployResolveError,
    resolve_deployment,
)
from comfy_cli.command.deploy_runtime import (
    command_clients as _command_clients,
)
from comfy_cli.command.deploy_runtime import (
    poll_deployment,
    render_spec_error,
    terminal_status_error,
)
from comfy_cli.command.deploy_runtime import (
    sleep as _sleep,
)
from comfy_cli.command.deploy_types import (
    DeployUpClient,
    compute_config,
    release_summary,
    required_int,
    required_string,
    server_shape_error,
)
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer
from comfy_cli.output.renderer import Renderer

_WORKER_STATES: Final = ("idle", "initializing", "ready", "running", "throttled", "unhealthy")
_STOP_REASONS: Final = frozenset({"user", "credits", "policy"})


@dataclass(frozen=True, slots=True)
class StatusTarget:
    build_id: str
    build_name: str
    deployment: JsonObject | None


@dataclass(frozen=True, slots=True)
class StatusResult:
    build_id: str
    build_name: str
    deployment: JsonObject | None
    release: JsonObject | None
    serving: JsonObject | None

    def payload(self) -> JsonObject:
        return {
            "build": {"id": self.build_id, "name": self.build_name},
            "deployment": self.deployment,
            "release": self.release,
            "serving": self.serving,
        }


def _nullable_string(value: JsonObject, key: str) -> str | None:
    field = value.get(key)
    if field is None:
        return None
    if not isinstance(field, str) or not field:
        raise server_shape_error(f"the deploy service returned an invalid {key}", field=key)
    return field


def _stop_reason(deployment: JsonObject) -> str | None:
    reason = _nullable_string(deployment, "stopReason")
    if reason is not None and reason not in _STOP_REASONS:
        raise server_shape_error("the deployment has an unknown stopReason", stopReason=reason)
    return reason


def _normalized_deployment(deployment: JsonObject) -> JsonObject:
    status = required_string(deployment, "status")
    if status not in _STATUS_RANK:
        raise server_shape_error("the deployment has an unknown status", status=status)
    return {
        "id": required_string(deployment, "id"),
        "status": status,
        "endpointUrl": _nullable_string(deployment, "endpointUrl"),
        "computeConfig": compute_config(deployment),
        "stopReason": _stop_reason(deployment),
        "error": _nullable_string(deployment, "error"),
    }


def _normalized_serving(deployment: JsonObject) -> JsonObject | None:
    serving = deployment.get("serving")
    if serving is None:
        return None
    if not isinstance(serving, dict):
        raise server_shape_error("the deployment has an invalid serving sample")
    workers = serving.get("workers")
    if not isinstance(workers, dict):
        raise server_shape_error("the deployment serving sample has no workers")
    return {
        "workers": {state: required_int(workers, state) for state in _WORKER_STATES},
        "jobsInQueue": required_int(serving, "jobsInQueue"),
        "sampledAt": required_string(serving, "sampledAt"),
    }


def resolve_status(
    builder: BuilderReleaseClient, client: DeployUpClient, path: str | None, deployment_id: str | None = None
) -> StatusTarget:
    paths = resolve_build_paths(path)
    spec = read_build_spec(paths.spec_file)
    build_id = spec.get("id")
    if not isinstance(build_id, str) or not build_id:
        raise BuildNotPushedError
    build_name = spec.get("name")
    if not isinstance(build_name, str) or not build_name:
        raise KeyError("name")
    deployment = resolve_deployment(builder, client, build_id, deployment_id=deployment_id)
    return StatusTarget(build_id, build_name, deployment)


def _release_by_id(releases: list[JsonObject], release_id: str) -> JsonObject:
    for release in releases:
        if required_string(release, "id") == release_id:
            return release
    raise server_shape_error("the deployment release is missing from its Build", releaseId=release_id)


def status_result(builder: BuilderReleaseClient, target: StatusTarget) -> StatusResult:
    deployment = target.deployment
    if deployment is None:
        return StatusResult(target.build_id, target.build_name, None, None, None)

    releases = builder.list_releases(target.build_id)
    release_id = required_string(deployment, "releaseId")
    current = _release_by_id(releases, release_id)
    current_version = required_int(current, "version")
    deployable = [release for release in releases if release.get("deployable") is True]
    latest = max(deployable, key=lambda release: required_int(release, "version")) if deployable else None
    behind = latest is not None and required_int(latest, "version") > current_version
    release = {
        **release_summary(current),
        "behind": behind,
        "latestDeployable": release_summary(latest) if behind and latest is not None else None,
    }
    return StatusResult(
        target.build_id,
        target.build_name,
        _normalized_deployment(deployment),
        release,
        _normalized_serving(deployment),
    )


def _render_serving(renderer: Renderer, serving: JsonObject | None) -> None:
    if serving is None:
        renderer.info("Serving: not sampled yet.")
        return
    workers = serving["workers"]
    if not isinstance(workers, dict):
        raise server_shape_error("the normalized serving sample has no workers")
    counts = " ".join(f"{state}={required_int(workers, state)}" for state in _WORKER_STATES)
    queue = required_int(serving, "jobsInQueue")
    sampled_at = required_string(serving, "sampledAt")
    suffix = " — healthy idle (scale-to-zero)" if queue == 0 and all(value == 0 for value in workers.values()) else ""
    renderer.info(f"Serving: {counts} queued={queue}; sampledAt={sampled_at}{suffix}")


def _render_stop_reason(renderer: Renderer, deployment: JsonObject) -> None:
    match deployment.get("stopReason"):
        case None:
            return
        case "user":
            renderer.info("Stop reason: stopped by user.")
        case "credits":
            renderer.warn("Stop reason: stopped for insufficient credits.")
        case "policy":
            renderer.warn("Stop reason: stopped by policy.")
        case reason:
            raise server_shape_error("the deployment has an unknown stopReason", stopReason=str(reason))


def _render_deployment(renderer: Renderer, deployment: JsonObject) -> str:
    deployment_id = required_string(deployment, "id")
    status = required_string(deployment, "status")
    match status:
        case "ready":
            if renderer.is_pretty():
                renderer.success(f"Deployment {deployment_id}: ready")
        case "unhealthy":
            if renderer.is_pretty():
                renderer.info(f"Deployment {deployment_id}: unhealthy (recoverable)")
        case "stop_failed":
            renderer.warn(
                f"Deployment {deployment_id} could not stop and may still be billing.",
                hint=f"run `comfy deploy stop --deployment {deployment_id}` again",
            )
        case "failed":
            if renderer.is_pretty():
                renderer.warn(f"Deployment {deployment_id}: failed")
        case "stopped":
            if renderer.is_pretty():
                renderer.info(f"Deployment {deployment_id}: stopped")
        case "queued" | "provisioning" | "starting" | "stopping":
            if renderer.is_pretty():
                renderer.info(f"Deployment {deployment_id}: {status}")
        case _:
            raise server_shape_error("the deployment has an unknown status", status=status)
    return status


def render_status(renderer: Renderer, result: StatusResult) -> None:
    deployment = result.deployment
    if deployment is None:
        renderer.info(
            f"Build {result.build_id} ({result.build_name}) has no deployment.",
            hint="run `comfy deploy up` to create one",
        )
        renderer.emit(result.payload(), command="deploy status", changed=False)
        return

    status = _render_deployment(renderer, deployment)
    if renderer.is_pretty():
        _render_stop_reason(renderer, deployment)
        _render_serving(renderer, result.serving)
    release = result.release
    if release is not None and release.get("behind") is True:
        latest = release.get("latestDeployable")
        if not isinstance(latest, dict):
            raise server_shape_error("a behind release has no latestDeployable")
        renderer.warn(
            f"Deployment {required_string(deployment, 'id')} runs release v{required_int(release, 'version')}; "
            f"release v{required_int(latest, 'version')} is deployable.",
            hint="running `comfy deploy up` creates a new deployment with a new URL",
        )
    terminal = status in {"failed", "stop_failed"}
    renderer.emit(
        result.payload(),
        command="deploy status",
        changed=False,
        ok=not terminal,
        error=terminal_status_error(required_string(deployment, "id"), status) if terminal else None,
    )
    if terminal:
        raise typer.Exit(code=1)


def run_status(path: str | None, *, deployment_id: str | None = None, watch: bool) -> None:
    renderer = get_renderer()
    try:
        builder, client = _command_clients()
        target = resolve_status(builder, client, path, deployment_id)
        if watch and target.deployment is not None:
            watched = poll_deployment(client, required_string(target.deployment, "id"), _sleep)
            target = replace(target, deployment=watched)
        render_status(renderer, status_result(builder, target))
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        render_spec_error(renderer, error)
        raise typer.Exit(code=1) from error
    except DeployResolveError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except DeployAPIError as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except BuilderAuthError as error:
        renderer.error(code="deploy_not_signed_in", message=str(error))
        raise typer.Exit(code=1) from error
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
