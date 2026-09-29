"""Shared authentication, resolution, and polling for deploy commands."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, TypeVar

from typing_extensions import assert_never

from comfy_cli.builder_api import BuilderClient, BuilderCredentialRefused
from comfy_cli.command.build import DEFAULT_BUILDER_URL
from comfy_cli.command.build_paths import BuildSpecNotFoundError, resolve_build_paths
from comfy_cli.command.build_spec import BuildSpecInvalidError, JsonObject, read_build_spec
from comfy_cli.command.deploy_resolve import (
    _STATUS_RANK,
    BuilderReleaseClient,
    DeploymentReference,
    ReleaseReference,
    ReleaseResolveRequest,
    resolve_release,
)
from comfy_cli.command.deploy_types import DeployUpClient, UpRequest, required_string, server_shape_error
from comfy_cli.deploy_api import DeployAPIError, DeployClient
from comfy_cli.output.renderer import Renderer

T = TypeVar("T")

DEPLOY_POLL_SECONDS: Final = 2.0
# Where a watch stops. `unhealthy` is here although the service can still move
# it back to `ready`: it only ever follows `ready`, so a deployment in it has
# already come up, and a watch that waited on it would wait silently for as
# long as the endpoint stays degraded.
_WATCH_TERMINAL: Final = frozenset({"ready", "unhealthy", "failed", "stopped", "stop_failed"})


@dataclass(frozen=True, slots=True)
class _BuilderReleaseAdapter:
    client: BuilderClient

    def get_release(self, release_id: str) -> JsonObject:
        return _refusal_as_deploy_error(lambda: self.client.get_release(release_id))

    def list_releases(self, build_id: str) -> list[JsonObject]:
        return _refusal_as_deploy_error(lambda: self.client.list_releases(build_id))


def _refusal_as_deploy_error(call: Callable[[], T]) -> T:
    """Report the builder refusing this command's credential the way the deploy service would.

    Every deploy command catches a builder ``HTTPError`` as a network failure, so
    without this a revoked key reads as an unreachable service.
    """
    try:
        return call()
    except BuilderCredentialRefused as error:
        raise DeployAPIError(
            "deploy_not_signed_in", "the builder refused the credential (401)", status=401, hint=error.hint
        ) from error


def command_clients() -> tuple[BuilderReleaseClient, DeployUpClient]:
    deploy = DeployClient.from_credentials()
    builder_url = os.environ.get("COMFY_BUILDER_URL") or DEFAULT_BUILDER_URL
    return _BuilderReleaseAdapter(BuilderClient.from_credentials(builder_url)), deploy


def sleep(seconds: float) -> None:
    time.sleep(seconds)


def render_spec_error(renderer: Renderer, error: BuildSpecNotFoundError | BuildSpecInvalidError) -> None:
    match error:
        case BuildSpecNotFoundError():
            renderer.error(code=error.code, message=str(error), hint=error.hint, details=error.details)
        case BuildSpecInvalidError():
            details = {"path": str(error.path)} if error.path is not None else None
            renderer.error(code=error.code, message=str(error), details=details)
        case unreachable:
            assert_never(unreachable)


def terminal_status_error(deployment_id: str, status: str) -> JsonObject:
    """The error block a not-ok deployment envelope carries.

    The read itself succeeded, so ``data`` still holds the whole payload; this
    is the machine-readable statement of the verdict that sat beside it as
    ``error: null``.
    """
    from comfy_cli import error_codes

    registered = error_codes.get("deploy_status_terminal")
    return {
        "code": "deploy_status_terminal",
        "message": f"deployment {deployment_id} is {status}",
        "hint": registered.hint if registered is not None else None,
        "details": {"deployment_id": deployment_id, "status": status},
    }


def poll_deployment(
    client: DeployUpClient,
    deployment_id: str,
    sleep_fn: Callable[[float], None],
    on_snapshot: Callable[[JsonObject], None] | None = None,
) -> JsonObject:
    """Read the deployment until it settles, handing each read to ``on_snapshot``.

    The progress a watcher shows rides the same read the loop already makes, so
    watching costs the service nothing it was not already answering.
    """
    while True:
        snapshot = client.get_deployment(deployment_id)
        if on_snapshot is not None:
            on_snapshot(snapshot)
        status = required_string(snapshot, "status")
        if status in _WATCH_TERMINAL:
            return snapshot
        if status not in _STATUS_RANK:
            raise server_shape_error("the deployment has an unknown status", status=status)
        sleep_fn(DEPLOY_POLL_SECONDS)


def resolved_up_request(builder: BuilderReleaseClient, path: str | None, release_id: str | None) -> UpRequest:
    spec = None
    if release_id is None:
        paths = resolve_build_paths(path)
        spec = read_build_spec(paths.spec_file)
    resolved = resolve_release(builder, ReleaseResolveRequest(release_id=release_id, spec=spec))
    match resolved:
        case ReleaseReference(release=release, build_id=build_id) if build_id is not None:
            return UpRequest(release, build_id, None, None, None, None)
        case ReleaseReference():
            raise server_shape_error("the Builder release has no buildId")
        case DeploymentReference():
            raise server_shape_error("deploy up resolved a deployment instead of a release")
        case unreachable:
            assert_never(unreachable)
