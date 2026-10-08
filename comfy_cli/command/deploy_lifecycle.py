"""Resolve and run deployment lifecycle mutations."""

from __future__ import annotations

import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeAlias, runtime_checkable

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject
from comfy_cli.command.deploy_read import ReadRequest, resolve_deployment_id
from comfy_cli.command.deploy_resolve import DeploymentListClient, DeployResolveError
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import compute_config, required_string, server_shape_error
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.interaction import confirm
from comfy_cli.output import get_renderer
from comfy_cli.output.renderer import Renderer


@dataclass(frozen=True, slots=True)
class ScaleRequest:
    target: ReadRequest
    minimum: int | None
    maximum: int | None
    gpu: str | None
    region: str | None


@runtime_checkable
class DeploymentLifecycleClient(DeploymentListClient, Protocol):
    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def update_deployment(self, deployment_id: str, compute: JsonObject, /) -> JsonObject: ...

    def stop_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def start_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def delete_deployment(self, deployment_id: str, /) -> None: ...


LifecycleAction: TypeAlias = Callable[[Renderer, DeploymentLifecycleClient, str], None]


def _run_lifecycle(request: ReadRequest, action: LifecycleAction) -> None:
    renderer = get_renderer()
    try:
        builder, candidate = _command_clients()
        if not isinstance(candidate, DeploymentLifecycleClient):
            raise server_shape_error("the deploy client cannot mutate deployment lifecycle")
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


def run_scale(request: ScaleRequest) -> None:
    def scale(renderer: Renderer, client: DeploymentLifecycleClient, deployment_id: str) -> None:
        deployment = client.get_deployment(deployment_id)
        current = compute_config(deployment)
        status = required_string(deployment, "status")
        merged: JsonObject = {
            "gpuClass": request.gpu if request.gpu is not None else current["gpuClass"],
            "region": request.region if request.region is not None else current["region"],
        }
        for bound, requested in (("min", request.minimum), ("max", request.maximum)):
            value = requested if requested is not None else current.get(bound)
            if value is not None:
                merged[bound] = value
        try:
            result = client.update_deployment(deployment_id, merged)
        except DeployAPIError as error:
            if error.code != "deploy_conflict":
                raise
            details: JsonObject = {**error.details, "currentStatus": status}
            raise DeployAPIError(
                error.code,
                f"{error} (current status: {status})",
                status=error.status,
                details=details,
            ) from error
        if renderer.is_pretty():
            renderer.success(
                f"Scaled deployment {deployment_id} to "
                f"min={merged.get('min', 'unset')}, max={merged.get('max', 'unset')}"
            )
        renderer.emit(result, command="deploy scale", changed=merged != current)

    _run_lifecycle(request.target, scale)


def _stop(renderer: Renderer, client: DeploymentLifecycleClient, deployment_id: str) -> None:
    result = client.stop_deployment(deployment_id)
    if renderer.is_pretty():
        renderer.success(f"Stop accepted for deployment {deployment_id}")
    renderer.emit(result, command="deploy stop", changed=True)


def run_stop(request: ReadRequest) -> None:
    _run_lifecycle(request, _stop)


def _start(renderer: Renderer, client: DeploymentLifecycleClient, deployment_id: str) -> None:
    result = client.start_deployment(deployment_id)
    if renderer.is_pretty():
        renderer.success(f"Start accepted for deployment {deployment_id}")
    renderer.emit(result, command="deploy start", changed=True)


def run_start(request: ReadRequest) -> None:
    _run_lifecycle(request, _start)


def run_delete(request: ReadRequest, *, yes: bool, ctx: typer.Context) -> None:
    def delete(renderer: Renderer, client: DeploymentLifecycleClient, deployment_id: str) -> None:
        if not confirm(
            f"Enqueue teardown and soft-delete deployment {deployment_id}?",
            yes=yes,
            error_code="deploy_delete_needs_confirm",
            details={"deploymentId": deployment_id},
            ctx=ctx,
        ):
            # Only a prompt can decline, and `interaction._may_prompt` allows one
            # only in pretty mode, where `emit` is a no-op — so there is no
            # envelope to write here. A machine caller is refused before this
            # point with `deploy_delete_needs_confirm`.
            renderer.info("Aborted.")
            return
        client.delete_deployment(deployment_id)
        if renderer.is_pretty():
            renderer.success(
                f"Delete accepted for deployment {deployment_id}; teardown is queued. "
                "The record remains visible with `comfy deploy ls -a` once deletion settles."
            )
        renderer.emit(
            {"deploymentId": deployment_id, "accepted": True, "recordRetained": True},
            command="deploy delete",
            changed=True,
        )

    _run_lifecycle(request, delete)
