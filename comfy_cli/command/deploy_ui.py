"""Read-only discovery of an authenticated deployment editor link."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
from collections.abc import Callable
from typing import Any

from comfy_cli.comfy_client import Client, HTTPError, Unauthenticated
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.command.deploy_read import DeploymentReadClient, ReadRequest, _run_read
from comfy_cli.command.deploy_resolve import BuilderReleaseClient, DeployResolveError
from comfy_cli.command.deploy_types import required_string
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output.renderer import Renderer
from comfy_cli.target import Target, resolve_target


class EditorUnavailableError(DeployResolveError):
    code = "deploy_ui_unavailable"
    hint = "run `comfy cloud login`, verify the workspace and ready deployment, or use `comfy deploy run`"

    def __init__(self, message: str, *, status: int | None = None) -> None:
        self.details: JsonObject = {"status": status} if status is not None else {}
        super().__init__(message)


def _ready_release(deployment: JsonObject, deployment_id: str) -> str:
    if (
        deployment.get("id") != deployment_id
        or deployment.get("status") != "ready"
        or deployment.get("deletedAt") is not None
    ):
        raise EditorUnavailableError("the selected deployment is not a verified ready target")
    if "editor" in deployment and deployment["editor"] is not True:
        raise EditorUnavailableError("the selected deployment's editor is disabled or could not be verified")
    return required_string(deployment, "releaseId")


def _editor_origin(target: Target) -> urllib.parse.SplitResult:
    if not target.auth_token or target.api_key:
        raise EditorUnavailableError("opening a custom environment requires a signed-in Cloud session, not an API key")
    origin = urllib.parse.urlsplit(target.base_url)
    if origin.username is not None or origin.password is not None or not origin.hostname:
        raise EditorUnavailableError("the configured Cloud origin is not a safe editor URL")
    return origin


def _cloud_read(request: Callable[[], Any]) -> Any:
    try:
        return request()
    except HTTPError as error:
        try:
            body = json.loads(error.body)
        except ValueError:
            body = None
        reason = body.get("message") if isinstance(body, dict) else None
        recognized = {
            "listing deployments needs a signed-in session, not an API key",
            "running Cloud on a Build is not enabled for this workspace yet",
            "deployment picking is not available in this environment",
            "Workspace not found",
            "Authentication required",
        }
        message = (
            reason
            if isinstance(reason, str) and reason in recognized
            else "Cloud refused or could not verify editor access"
        )
        raise EditorUnavailableError(f"{message} (HTTP {error.status})", status=error.status) from error
    except Unauthenticated as error:
        raise EditorUnavailableError("the Cloud sign-in could not be refreshed; run `comfy cloud login`") from error
    except (ResponseTooLarge, urllib.error.URLError, OSError, ValueError) as error:
        raise EditorUnavailableError("could not verify editor access with Cloud; no link was generated") from error


def verify_deployment_editor_target(
    cloud: Client, *, workspace_id: str, deployment_id: str, deployment: JsonObject
) -> dict[str, str]:
    """Verify one ready target against the authenticated workspace's editor listing; mutate nothing."""
    release_id = _ready_release(deployment, deployment_id)
    origin = _editor_origin(cloud.target)
    if not isinstance(workspace_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", workspace_id):
        raise EditorUnavailableError("Cloud did not identify the authenticated workspace")
    listing = _cloud_read(lambda: cloud.list_workspace_deployments(workspace_id))
    rows = listing.get("items") if isinstance(listing, dict) else None
    matches = (
        [row for row in rows if isinstance(row, dict) and row.get("deployment_id") == deployment_id]
        if isinstance(rows, list)
        else []
    )
    if len(matches) != 1 or matches[0].get("status") != "ready" or matches[0].get("release_id") != release_id:
        raise EditorUnavailableError("Cloud did not confirm this workspace's ready deployment and served release")
    url = urllib.parse.urlunsplit(
        (
            origin.scheme,
            origin.netloc,
            "/",
            urllib.parse.urlencode({"workspace": workspace_id, "deployment": deployment_id}),
            "",
        )
    )
    return {"workspaceId": workspace_id, "deploymentId": deployment_id, "releaseId": release_id, "url": url}


def _ui(renderer: Renderer, builder: BuilderReleaseClient, client: DeploymentReadClient, deployment_id: str) -> None:
    deployment = client.get_deployment(deployment_id)
    _ready_release(deployment, deployment_id)
    target = resolve_target(where="cloud")
    _editor_origin(target)
    cloud = Client(target)
    workspace = _cloud_read(cloud.get_current_workspace)
    workspace_id = workspace.get("id") if isinstance(workspace, dict) else None
    if not isinstance(workspace_id, str):
        raise EditorUnavailableError("Cloud did not identify the authenticated workspace")
    result = verify_deployment_editor_target(
        cloud, workspace_id=workspace_id, deployment_id=deployment_id, deployment=deployment
    )
    if renderer.is_pretty():
        renderer.print(result["url"])
    renderer.emit(result, command="deploy ui", changed=False)


def run_ui(request: ReadRequest) -> None:
    _run_read(request, _ui)
