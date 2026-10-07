"""Resolve, fetch, and render one deployment resource."""

from __future__ import annotations

import json
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError, resolve_build_paths
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, JsonValue, read_build_spec
from comfy_cli.command.deploy_resolve import (
    BuilderReleaseClient,
    BuildNotPushedError,
    DeploymentListClient,
    DeployResolveError,
    ReleaseNotInBuildError,
    find_build_release,
    release_version_selector,
    resolve_deployment,
)
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import (
    optional_revision,
    release_label,
    release_summary,
    required_string,
    server_shape_error,
)
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer
from comfy_cli.output.renderer import Renderer


@dataclass(frozen=True, slots=True)
class ReadRequest:
    path: str | None
    deployment_id: str | None


@runtime_checkable
class DeploymentReadClient(DeploymentListClient, Protocol):
    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def get_deployment_logs(self, deployment_id: str, /) -> JsonObject: ...

    def get_deployment_events(self, deployment_id: str, /) -> JsonObject: ...


class UpdatesUnavailableError(DeployResolveError):
    code = "deploy_updates_unavailable"
    hint = "run `comfy deploy events` without `--release`; its events do not say which release made them"

    def __init__(self, deployment_id: str) -> None:
        self.details = {"deployment_id": deployment_id}
        super().__init__(
            f"deployment {deployment_id} carries no revision, so its events do not say which release made them; "
            "deployment updates are most likely not on for this workspace yet"
        )


class EmptyReleaseSelectorError(DeployResolveError):
    code = "deploy_bad_request"
    hint = "pass a version such as v5, or a release id"

    def __init__(self) -> None:
        self.details = {"release": ""}
        super().__init__("--release is empty")


class DeploymentNotFoundError(DeployResolveError):
    code = "deploy_not_found"
    hint = "run `comfy deploy up` to create one, or `comfy deploy status` to inspect this Build"

    def __init__(self, build_id: str) -> None:
        self.details = {"buildId": build_id}
        super().__init__(f"Build {build_id} has no deployment")


def resolve_deployment_id(
    builder: BuilderReleaseClient,
    deploy: DeploymentListClient,
    request: ReadRequest,
) -> str:
    if request.deployment_id is not None:
        return request.deployment_id
    paths = resolve_build_paths(request.path)
    spec = read_build_spec(paths.spec_file)
    build_id = spec.get("id")
    if not isinstance(build_id, str) or not build_id:
        raise BuildNotPushedError
    deployment = resolve_deployment(builder, deploy, build_id)
    if deployment is None:
        raise DeploymentNotFoundError(build_id)
    return required_string(deployment, "id")


ReadAction: TypeAlias = Callable[[Renderer, BuilderReleaseClient, DeploymentReadClient, str], None]


def _show(renderer: Renderer, builder: BuilderReleaseClient, client: DeploymentReadClient, deployment_id: str) -> None:
    deployment = client.get_deployment(deployment_id)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(deployment))
    renderer.emit(deployment, command="deploy show", changed=False)


def _logs(renderer: Renderer, builder: BuilderReleaseClient, client: DeploymentReadClient, deployment_id: str) -> None:
    logs = client.get_deployment_logs(deployment_id)
    captured_at = logs.get("capturedAt")
    if captured_at is not None and not isinstance(captured_at, str):
        raise server_shape_error("the deployment logs have an invalid capturedAt")
    log = logs.get("comfyuiLog")
    if not isinstance(log, str):
        raise server_shape_error("the deployment logs have no comfyuiLog")
    # Validated before any branch on output mode, as in `_events`: `--json` must
    # reject exactly the malformed responses pretty mode rejects, and the schema
    # requires deploymentId of both.
    required_string(logs, "deploymentId")
    if renderer.is_pretty():
        renderer.info(f"capturedAt: {captured_at if captured_at is not None else 'not captured yet'}")
        if log:
            renderer.print(log)
        elif captured_at is None:
            # The log is written once the health check finishes, so it is absent
            # in two different states: a deployment still coming up has one on
            # the way, and one that failed before a container ran never will.
            # Saying only the second would tell a booting deployment's owner
            # their deploy is dead.
            renderer.info(
                "No log yet. The log is captured when the health check finishes, so a deployment that is "
                "still coming up has none yet, and one that failed before a container ran never will. "
                f"`comfy deploy events --deployment {deployment_id}` shows which of the two this is."
            )
    renderer.emit(logs, command="deploy logs", changed=False)


def _event_line(event: JsonValue) -> str:
    if not isinstance(event, dict):
        raise server_shape_error("the deployment events response contains a non-object event")
    message = event.get("message")
    if message is not None and not isinstance(message, str):
        raise server_shape_error("a deployment event has an invalid message")
    suffix = f"  {message}" if message else ""
    # After a move the events hold every copy's transitions; say whose each is.
    release_id = event.get("releaseId")
    if "releaseId" in event and (not isinstance(release_id, str) or not release_id):
        raise server_shape_error("a deployment event has an invalid releaseId")
    release = f"  release {release_id}" if release_id else ""
    return f"  {required_string(event, 'at')}  {required_string(event, 'status')}{release}{suffix}"


def _build_releases(
    builder: BuilderReleaseClient, client: DeploymentReadClient, deployment_id: str
) -> tuple[str, list[JsonObject]]:
    """The Build the deployment runs and its releases.

    Only a deployment inside the updates rollout tags each event with its
    release, and its read is the one that carries a revision.
    """
    deployment = client.get_deployment(deployment_id)
    if optional_revision(deployment) is None:
        raise UpdatesUnavailableError(deployment_id)
    current = builder.get_release(required_string(deployment, "releaseId"))
    build_id = required_string(current, "buildId")
    return build_id, builder.list_releases(build_id)


def _named_release(build_id: str, releases: list[JsonObject], selector: str, events: list[JsonValue]) -> JsonObject:
    """The release `--release` names. An id the Build no longer lists, because
    its release was deleted, still names the events that carry it."""
    listed = find_build_release(releases, selector)
    if listed is not None:
        version = listed.get("version")
        return release_summary(listed) if isinstance(version, int) else {"id": required_string(listed, "id")}
    version = release_version_selector(selector)
    if version is None and any(isinstance(event, dict) and event.get("releaseId") == selector for event in events):
        return {"id": selector}
    raise ReleaseNotInBuildError(build_id, f"v{version}" if version is not None else selector)


def _events(
    renderer: Renderer,
    builder: BuilderReleaseClient,
    client: DeploymentReadClient,
    deployment_id: str,
    release_selector: str | None = None,
) -> None:
    if release_selector is not None:
        release_selector = release_selector.strip()
        if not release_selector:
            raise EmptyReleaseSelectorError
    build = _build_releases(builder, client, deployment_id) if release_selector is not None else None
    result = client.get_deployment_events(deployment_id)
    events = result.get("events")
    if not isinstance(events, list):
        raise server_shape_error("the deployment events response has no events array")
    # Validated before any branch on output mode: `--json` must reject exactly
    # the malformed responses pretty mode rejects, not silently forward them.
    required_string(result, "deploymentId")
    lines = [_event_line(event) for event in events]
    release = None
    if build is not None and release_selector is not None:
        release = _named_release(*build, release_selector, events)
        kept = [
            (event, line)
            for event, line in zip(events, lines)
            if isinstance(event, dict) and event.get("releaseId") == release["id"]
        ]
        lines = [line for _, line in kept]
        result = {**result, "events": [event for event, _ in kept], "release": release}
    if renderer.is_pretty():
        if not lines:
            renderer.info("No deployment events." if release is None else f"No events from {release_label(release)}.")
        for line in lines:
            renderer.print(line)
    renderer.emit(result, command="deploy events", changed=False)


def _run_read(request: ReadRequest, action: ReadAction) -> None:
    renderer = get_renderer()
    try:
        builder, candidate = _command_clients()
        if not isinstance(candidate, DeploymentReadClient):
            raise server_shape_error("the deploy client cannot read deployment resources")
        deployment_id = resolve_deployment_id(builder, candidate, request)
        action(renderer, builder, candidate, deployment_id)
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


def run_show(request: ReadRequest) -> None:
    _run_read(request, _show)


def run_logs(request: ReadRequest) -> None:
    _run_read(request, _logs)


def run_events(request: ReadRequest, release: str | None = None) -> None:
    _run_read(
        request,
        lambda renderer, builder, client, deployment_id: _events(renderer, builder, client, deployment_id, release),
    )
