"""End the update a deployment waits on, leaving it on the release it serves."""

from __future__ import annotations

import urllib.error
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.command.build_paths import BuildSpecNotFoundError
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject
from comfy_cli.command.deploy_resolve import (
    BuilderReleaseClient,
    ChoiceRefusedError,
    DeploymentListClient,
    DeployResolveError,
    deployment_id_for,
    sole_deployment,
)
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import render_spec_error
from comfy_cli.command.deploy_types import (
    deployment_label,
    deployment_name,
    optional_revision,
    release_label,
    release_summary,
    required_int,
    required_string,
    server_shape_error,
)
from comfy_cli.command.deploy_up import release_or_id, running_first
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer


@dataclass(frozen=True, slots=True)
class CancelRequest:
    path: str | None
    deployment_id: str | None


@runtime_checkable
class DeploymentCancelClient(DeploymentListClient, Protocol):
    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def cancel_pending_update(self, deployment_id: str, /) -> JsonObject: ...


class CancelUnavailableError(DeployResolveError):
    """A deploy service too old for the cancel route: it ends once that ships."""

    code = "deploy_updates_unavailable"
    hint = "wait until the update lands or fails; `comfy deploy status` shows it"

    def __init__(self, deployment_id: str) -> None:
        self.details = {"deployment_id": deployment_id, "reason": "service_too_old"}
        super().__init__(f"the deploy service cannot cancel the update of deployment {deployment_id} yet")


class NoDeploymentToCancelError(DeployResolveError):
    code = "deploy_not_found"
    hint = "run `comfy deploy ls` to find one, then pass `--deployment <id>`"

    def __init__(self, build_id: str) -> None:
        self.details = {"buildId": build_id}
        super().__init__(f"Build {build_id} has no deployment to cancel an update of")


class CancelAmbiguousError(ChoiceRefusedError):
    hint = "pass `--deployment <name|id>` to cancel the update of one of them"

    def __init__(self, build_id: str, rows: Sequence[JsonObject], releases: Sequence[JsonObject]) -> None:
        super().__init__(build_id, "cancel", rows, releases)


@dataclass(frozen=True, slots=True)
class CancelResult:
    deployment: JsonObject
    release: JsonObject
    # The update the cancel ended, or None where nothing waited.
    cancelled: JsonObject | None

    def payload(self) -> JsonObject:
        deployment: JsonObject = {
            "id": required_string(self.deployment, "id"),
            "status": required_string(self.deployment, "status"),
        }
        name, revision = deployment_name(self.deployment), optional_revision(self.deployment)
        if name is not None:
            deployment["name"] = name
        if revision is not None:
            deployment["revision"] = revision
        return {"deployment": deployment, "release": self.release, "cancelledUpdate": self.cancelled}


def _picked_deployment(
    builder: BuilderReleaseClient, client: DeploymentCancelClient, request: CancelRequest
) -> tuple[str, Sequence[JsonObject]]:
    """The deployment named, else the Build's only one up, as `up` and `rollback`
    pick it, with the Build's releases where picking listed them."""
    if request.deployment_id is not None:
        return deployment_id_for(builder, client, request.deployment_id, path=request.path), ()
    deployment_id, _, releases = sole_deployment(
        builder,
        client,
        request.path,
        none=NoDeploymentToCancelError,
        many=CancelAmbiguousError,
        prefer=running_first,
    )
    return deployment_id, releases


def _release(builder: BuilderReleaseClient, releases: Sequence[JsonObject], release_id: str) -> JsonObject:
    """The release's summary, from the Build's releases where listed, else read."""
    for listed in releases:
        if listed.get("id") == release_id and isinstance(listed.get("version"), int):
            return release_summary(listed)
    return release_or_id(builder, release_id)


def _cancelled(builder: BuilderReleaseClient, releases: Sequence[JsonObject], answer: JsonObject) -> JsonObject:
    ended = answer.get("cancelledUpdate")
    if not isinstance(ended, dict):
        raise server_shape_error("the cancel response names no cancelledUpdate", field="cancelledUpdate")
    kind = ended.get("kind")
    return {
        "release": _release(builder, releases, required_string(ended, "releaseId")),
        "baseRevision": required_int(ended, "baseRevision"),
        # Open on the service's side: a kind this client does not know reads as an update.
        "kind": "rollback" if kind == "rollback" else "update",
    }


def cancel(builder: BuilderReleaseClient, client: DeploymentCancelClient, request: CancelRequest) -> CancelResult:
    deployment_id, releases = _picked_deployment(builder, client, request)
    # Read first, as rollback does: a missing deployment answers here. A read
    # with no revision still sends the cancel, since the service leaves it out
    # whenever its rollout check fails, while an update may still wait.
    client.get_deployment(deployment_id)
    try:
        answer = client.cancel_pending_update(deployment_id)
    except DeployAPIError as error:
        server_code = (error.details or {}).get("server_code")
        if error.status == 409 and server_code == "NO_PENDING_UPDATE":
            # Nothing waits, so nothing changed. Read again: the update may
            # have landed since the first read.
            deployment = client.get_deployment(deployment_id)
            return CancelResult(deployment, _release(builder, releases, required_string(deployment, "releaseId")), None)
        # The deployment was just read, so a 404 without the service's own
        # code is a comfy-deploy too old to know the route; with it, the
        # deployment went since, and the error says so.
        if error.status == 404 and server_code != "NOT_FOUND":
            raise CancelUnavailableError(deployment_id) from error
        raise
    cancelled = _cancelled(builder, releases, answer)
    return CancelResult(answer, _release(builder, releases, required_string(answer, "releaseId")), cancelled)


def _cancel_text(result: CancelResult) -> str:
    label = deployment_label(result.deployment)
    serving = release_label(result.release)
    if result.cancelled is None:
        return f"Deployment {label} has no update waiting; it serves {serving}."
    noun = result.cancelled["kind"]
    return f"Cancelled the {noun} to {release_label(result.cancelled['release'])}. {label} keeps serving {serving}."


def run_cancel(request: CancelRequest) -> None:
    renderer = get_renderer()
    try:
        builder, candidate = _command_clients()
        if not isinstance(candidate, DeploymentCancelClient):
            raise server_shape_error("the deploy client cannot cancel updates")
        result = cancel(builder, candidate, request)
        if renderer.is_pretty():
            renderer.success(_cancel_text(result))
        renderer.emit(result.payload(), command="deploy cancel", changed=result.cancelled is not None)
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
