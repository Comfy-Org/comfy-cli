"""Give a deployment a new name, keeping its id, URL and everything else."""

from __future__ import annotations

import urllib.error
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject
from comfy_cli.command.deploy_read import DeploymentNotFoundError
from comfy_cli.command.deploy_resolve import (
    BuilderReleaseClient,
    ChoiceRefusedError,
    DeploymentListClient,
    DeployResolveError,
    NamesUnavailableError,
    deployment_id_for,
    name_refusal,
    sole_deployment,
    valid_name,
)
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import deployment_name, server_shape_error
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer


@dataclass(frozen=True, slots=True)
class RenameRequest:
    path: str | None
    deployment_id: str | None
    name: str


@runtime_checkable
class DeploymentRenameClient(DeploymentListClient, Protocol):
    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def rename_deployment(self, deployment_id: str, name: str, /) -> JsonObject: ...


class RenameAmbiguousError(ChoiceRefusedError):
    hint = "pass `--deployment <name|id>` to rename one of them"

    def __init__(self, build_id: str, rows: list[JsonObject], releases: list[JsonObject]) -> None:
        super().__init__(build_id, "rename", rows, releases)


def _picked_deployment(builder: BuilderReleaseClient, client: DeploymentRenameClient, request: RenameRequest) -> str:
    """The deployment named, else the Build's only live one, stopped or not, since any can be renamed."""
    if request.deployment_id is not None:
        return deployment_id_for(builder, client, request.deployment_id, path=request.path)
    return sole_deployment(builder, client, request.path, none=DeploymentNotFoundError, many=RenameAmbiguousError)[0]


def rename(builder: BuilderReleaseClient, client: DeploymentRenameClient, request: RenameRequest) -> JsonObject:
    """Rename the deployment and return what changed, refusing a name comfy-deploy would refuse first."""
    name = valid_name(request.name)
    deployment_id = _picked_deployment(builder, client, request)
    previous = deployment_name(client.get_deployment(deployment_id))
    try:
        renamed = client.rename_deployment(deployment_id, name)
    except DeployAPIError as error:
        # comfy-deploy before names finds neither compute nor a move in the body.
        if (error.details or {}).get("server_code") == "INVALID_REQUEST":
            raise NamesUnavailableError(name) from error
        refusal = name_refusal(error, name)
        if refusal is error:
            raise
        raise refusal from error
    # An answer without the name is read again before it is called a refusal.
    if deployment_name(renamed) != name and deployment_name(client.get_deployment(deployment_id)) != name:
        raise NamesUnavailableError(name)
    return {"deployment": {"id": deployment_id, "name": name}, "previousName": previous}


def run_rename(request: RenameRequest) -> None:
    renderer = get_renderer()
    try:
        builder, candidate = _command_clients()
        if not isinstance(candidate, DeploymentRenameClient):
            raise server_shape_error("the deploy client cannot rename deployments")
        result = rename(builder, candidate, request)
        previous = result["previousName"]
        if renderer.is_pretty():
            deployment_id = result["deployment"]["id"]
            if previous == request.name:
                renderer.success(f"Deployment {deployment_id} is already named {request.name}.")
            else:
                was = f" (was {previous})" if previous else ""
                renderer.success(f"Deployment {deployment_id} is now named {request.name}{was}.")
        renderer.emit(result, command="deploy rename", changed=previous != request.name)
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        render_spec_error(renderer, error)
        raise typer.Exit(code=1) from error
    except (DeployResolveError, DeployAPIError) as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except BuilderAuthError as error:
        renderer.error(code="deploy_not_signed_in", message=str(error))
        raise typer.Exit(code=1) from error
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
