"""End the update a deployment waits on, leaving it on the release it serves."""

from __future__ import annotations

import urllib.error
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Literal, Protocol, TypeAlias, runtime_checkable

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
from comfy_cli.command.deploy_runtime import (
    DeployWatchLostError,
    exit_watch_lost,
    poll_deployment,
    render_spec_error,
)
from comfy_cli.command.deploy_runtime import command_clients as _command_clients
from comfy_cli.command.deploy_runtime import sleep as _sleep
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

CANCEL_WATCH_SECONDS: Final = 600.0
_SAME_UPDATE: Final = ("releaseId", "baseRevision", "since", "instanceGroupId")
_MOVE: Final = "GPU/region move"

Outcome: TypeAlias = Literal["idle", "cancelled", "landed", "gone"]


@dataclass(frozen=True, slots=True)
class CancelRequest:
    path: str | None
    deployment_id: str | None


@runtime_checkable
class DeploymentCancelClient(DeploymentListClient, Protocol):
    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...

    def stop_kept_release(self, deployment_id: str, release_id: str, /) -> JsonObject: ...


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
    outcome: Outcome
    update: JsonObject | None = None
    move: bool = False

    @property
    def changed(self) -> bool:
        return self.outcome == "cancelled"

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
        cancelled = self.update if self.changed else None
        return {"deployment": deployment, "release": self.release, "cancelledUpdate": cancelled}


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


def _cancelled(builder: BuilderReleaseClient, releases: Sequence[JsonObject], pending: JsonObject) -> JsonObject:
    kind = pending.get("kind")
    return {
        "release": _release(builder, releases, required_string(pending, "releaseId")),
        "baseRevision": required_int(pending, "baseRevision"),
        # Open on the service's side: a kind this client does not know reads as an update.
        "kind": "rollback" if kind == "rollback" else "update",
    }


def _waiting(deployment: JsonObject) -> JsonObject | None:
    pending = deployment.get("pendingUpdate")
    return pending if isinstance(pending, dict) else None


def _ended(pending: JsonObject) -> Callable[[JsonObject], bool]:
    def settled(snapshot: JsonObject) -> bool:
        now = _waiting(snapshot)
        if now is None or now.get("status") == "failed":
            return True
        return any(now.get(field) != pending.get(field) for field in _SAME_UPDATE)

    return settled


def cancel(builder: BuilderReleaseClient, client: DeploymentCancelClient, request: CancelRequest) -> CancelResult:
    deployment_id, releases = _picked_deployment(builder, client, request)
    deployment = client.get_deployment(deployment_id)
    pending = _waiting(deployment)
    if pending is None:
        return CancelResult(deployment, _release(builder, releases, required_string(deployment, "releaseId")), "idle")
    update = _cancelled(builder, releases, pending)
    release_id = required_string(pending, "releaseId")
    move = release_id == deployment.get("releaseId")
    try:
        client.stop_kept_release(deployment_id, release_id)
    except DeployAPIError as error:
        if error.status != 409 or (error.details or {}).get("server_code") != "CURRENT_RELEASE":
            raise
        deployment = client.get_deployment(deployment_id)
        serving = _release(builder, releases, required_string(deployment, "releaseId"))
        revision = optional_revision(deployment)
        outcome: Outcome = "landed" if revision is not None and revision > update["baseRevision"] else "gone"
        return CancelResult(deployment, serving, outcome, update, move)
    deployment = poll_deployment(client, deployment_id, _sleep, settled=_ended(pending), limit=CANCEL_WATCH_SECONDS)
    serving = _release(builder, releases, required_string(deployment, "releaseId"))
    return CancelResult(deployment, serving, "cancelled", update, move)


def _cancel_text(result: CancelResult) -> str:
    label = deployment_label(result.deployment)
    serving = release_label(result.release)
    update = result.update
    if result.outcome == "idle" or update is None:
        return f"Deployment {label} has no update waiting; it serves {serving}."
    what = _MOVE if result.move else f"{update['kind']} to {release_label(update['release'])}"
    if result.outcome == "landed":
        return f"The {what} already landed; {label} serves {serving}."
    if result.outcome == "gone":
        return f"The {what} is no longer waiting; {label} serves {serving}."
    where = " on its current GPU and region" if result.move else ""
    return f"Cancelled the {what}. {label} keeps serving {serving}{where}."


def run_cancel(request: CancelRequest) -> None:
    renderer = get_renderer()
    try:
        builder, candidate = _command_clients()
        if not isinstance(candidate, DeploymentCancelClient):
            raise server_shape_error("the deploy client cannot cancel updates")
        result = cancel(builder, candidate, request)
        if renderer.is_pretty():
            renderer.success(_cancel_text(result))
        renderer.emit(result.payload(), command="deploy cancel", changed=result.changed)
    except (BuildSpecNotFoundError, BuildSpecInvalidError) as error:
        render_spec_error(renderer, error)
        raise typer.Exit(code=1) from error
    except (DeployResolveError, DeployAPIError) as error:
        renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        raise typer.Exit(code=1) from error
    except DeployWatchLostError as error:
        hint = (
            f"run `comfy deploy cancel --deployment {error.deployment_id}` again: it stops the update's copy "
            "again while its cancellation has not finished"
        )
        exit_watch_lost(renderer, error, hint)
    except BuilderAuthError as error:
        renderer.error(code="deploy_not_signed_in", message=str(error))
        raise typer.Exit(code=1) from error
    except (ResponseTooLarge, TimeoutError, urllib.error.URLError, KeyError) as error:
        renderer.error(code="deploy_server_error", message=str(error))
        raise typer.Exit(code=1) from error
