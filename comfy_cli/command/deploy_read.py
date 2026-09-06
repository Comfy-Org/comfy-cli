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
    resolve_deployment,
)
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import required_string, server_shape_error
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


ReadAction: TypeAlias = Callable[[Renderer, DeploymentReadClient, str], None]


def _show(renderer: Renderer, client: DeploymentReadClient, deployment_id: str) -> None:
    deployment = client.get_deployment(deployment_id)
    if renderer.is_pretty():
        renderer.console().print_json(json.dumps(deployment))
    renderer.emit(deployment, command="deploy show", changed=False)


def _logs(renderer: Renderer, client: DeploymentReadClient, deployment_id: str) -> None:
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
    renderer.emit(logs, command="deploy logs", changed=False)


def _event_line(event: JsonValue) -> str:
    if not isinstance(event, dict):
        raise server_shape_error("the deployment events response contains a non-object event")
    message = event.get("message")
    if message is not None and not isinstance(message, str):
        raise server_shape_error("a deployment event has an invalid message")
    suffix = f"  {message}" if message else ""
    return f"  {required_string(event, 'at')}  {required_string(event, 'status')}{suffix}"


def _events(renderer: Renderer, client: DeploymentReadClient, deployment_id: str) -> None:
    result = client.get_deployment_events(deployment_id)
    events = result.get("events")
    if not isinstance(events, list):
        raise server_shape_error("the deployment events response has no events array")
    # Validated before any branch on output mode: `--json` must reject exactly
    # the malformed responses pretty mode rejects, not silently forward them.
    required_string(result, "deploymentId")
    lines = [_event_line(event) for event in events]
    if renderer.is_pretty():
        if not lines:
            renderer.info("No deployment events.")
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
        action(renderer, candidate, deployment_id)
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


def run_events(request: ReadRequest) -> None:
    _run_read(request, _events)
