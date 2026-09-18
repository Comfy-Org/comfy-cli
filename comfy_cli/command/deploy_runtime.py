"""Shared authentication, resolution, and polling for deploy commands."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

from typing_extensions import assert_never

from comfy_cli.builder_api import BuilderClient
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
from comfy_cli.deploy_api import DeployClient
from comfy_cli.output.renderer import Renderer

DEPLOY_POLL_SECONDS: Final = 2.0
_WATCH_TERMINAL: Final = frozenset({"ready", "failed", "stopped", "stop_failed"})


@dataclass(frozen=True, slots=True)
class _BuilderReleaseAdapter:
    client: BuilderClient

    def get_release(self, release_id: str) -> JsonObject:
        return self.client.get_release(release_id)

    def list_releases(self, build_id: str) -> list[JsonObject]:
        return self.client.list_releases(build_id)


def command_clients() -> tuple[BuilderReleaseClient, DeployUpClient]:
    deploy = DeployClient.from_session()
    builder_url = os.environ.get("COMFY_BUILDER_URL") or DEFAULT_BUILDER_URL
    return _BuilderReleaseAdapter(BuilderClient.from_session(builder_url)), deploy


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


def poll_deployment(client: DeployUpClient, deployment_id: str, sleep_fn: Callable[[float], None]) -> JsonObject:
    while True:
        snapshot = client.get_deployment(deployment_id)
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
