"""Read-only, capability-gated URLs for normal private saved-workflow identity."""

from __future__ import annotations

import json
import re
import urllib.parse
from uuid import UUID

import typer

from comfy_cli.builder_api import BuilderAuthError
from comfy_cli.comfy_client import Client, HTTPError, Unauthenticated
from comfy_cli.command.deploy_resolve import DeployResolveError
from comfy_cli.command.deploy_ui import EditorUnavailableError, verify_deployment_editor_target
from comfy_cli.deploy_api_errors import DeployAPIError
from comfy_cli.http import ResponseTooLarge, assert_safe_url, request_json
from comfy_cli.output import get_renderer
from comfy_cli.target import resolve_target

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]+")
_MAX_CONTROL_BYTES = 256 * 1024
_MAX_WORKFLOW_BYTES = 16 * 1024 * 1024
_HINT = (
    "If sidebar read-back succeeded, open the saved name from Cloud → Workflows. "
    "If the normal sidebar file is unverified or record-only, retain the local workflow and explain that limitation. "
    "For a custom environment, use verified deployment API commands until the complete private handoff is verified."
)


class WorkflowUiUnavailable(Exception):
    def __init__(self, reason: str, message: str, *, status: int | None = None):
        super().__init__(message)
        self.reason, self.status = reason, status


def _refuse(reason: str, message: str) -> None:
    raise WorkflowUiUnavailable(reason, message)


def _get(target, *parts: str, max_bytes: int = _MAX_CONTROL_BYTES):
    status, body = request_json(target.url(*parts), target, method="GET", max_bytes=max_bytes, timeout=15)
    if status != 200:
        raise WorkflowUiUnavailable("unavailable", "Cloud could not verify this saved workflow.", status=status)
    return body


def _safe_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and 0 < len(name) <= 1024
        and not name.startswith(("/", "~"))
        and "\\" not in name
        and not any(ord(c) < 32 or ord(c) == 127 for c in name)
        and all(part.rstrip(" .") not in ("", ".", "..") for part in name.split("/"))
    )


def _deployment_client():
    from comfy_cli.deploy_api import DeployClient

    return DeployClient.from_credentials()


def verify_workflow_link(
    saved_file_id: str, *, workspace: str, environment: str, deployment_id: str | None = None
) -> dict:
    """Verify current access, normal file-backed identity and selected environment.

    No save/share operation, browser launch, environment mutation or compute
    operation occurs. A capability absent on an older release is a hard refusal.
    """
    try:
        canonical_id = str(UUID(saved_file_id))
    except (ValueError, AttributeError, TypeError):
        _refuse("invalid_identity", "Use the exact saved_file_id UUID from the userdata save receipt.")
    if canonical_id != saved_file_id.lower():
        _refuse("invalid_identity", "Use the canonical saved_file_id UUID.")
    if not isinstance(workspace, str) or not _SAFE_ID.fullmatch(workspace):
        _refuse("invalid_workspace", "Use the exact destination workspace ID.")
    if environment not in ("standard", "custom"):
        _refuse("invalid_environment", "Choose standard or custom explicitly.")
    if (environment == "standard" and deployment_id is not None) or (environment == "custom" and not deployment_id):
        _refuse("environment_conflict", "A custom environment requires its deployment ID; standard excludes it.")
    if deployment_id is not None and (not isinstance(deployment_id, str) or not _SAFE_ID.fullmatch(deployment_id)):
        _refuse("invalid_deployment", "Use an exact deployment ID, not a path or name selector.")

    try:
        target = resolve_target(where="cloud", allow_clear=False)
        if not target.is_cloud or not target.auth_token or target.api_key:
            _refuse("authentication_required", "A private workflow link requires an authenticated Cloud OAuth session.")
        origin = urllib.parse.urlsplit(target.base_url)
        if (
            origin.scheme not in ("https", "http")
            or not origin.hostname
            or origin.username is not None
            or origin.password is not None
        ):
            _refuse("invalid_origin", "The configured Cloud origin cannot produce a safe editor URL.")
        assert_safe_url(target.base_url)
        current = _get(target, "workspaces", "current")
        if not isinstance(current, dict) or current.get("id") != workspace:
            _refuse("workspace_mismatch", "The approved workspace is not the current authenticated workspace.")
        features = _get(target, "features")
        if not isinstance(features, dict) or features.get("private_saved_file_links_enabled") is not True:
            _refuse(
                "capability_unavailable", "This Cloud release has not enabled verified private saved-workflow links."
            )
        snapshot = _get(target, "userdata-files", canonical_id, max_bytes=_MAX_WORKFLOW_BYTES)
        if not isinstance(snapshot, dict) or snapshot.get("id") != canonical_id:
            _refuse("invalid_identity", "Cloud did not return the exact saved file identity.")
        filename = snapshot.get("path")
        if (
            not _safe_name(filename)
            or not filename.startswith("workflows/")
            or not filename.endswith(".json")
            or filename == "workflows/.json"
        ):
            _refuse("file_unavailable", "Cloud did not return a safe normal Workflows file path.")
        for key in ("modified", "size"):
            value = snapshot.get(key)
            if type(value) is not int or not 0 <= value <= 9007199254740991:
                _refuse("metadata_unverified", "Cloud did not return valid saved-file metadata.")
        text = snapshot.get("content")
        if not isinstance(text, str) or len(text.encode("utf-8")) != snapshot["size"]:
            _refuse("content_unverified", "Cloud did not return the exact saved-file content.")
        graph = json.loads(text)
        from comfy_cli.cloud_workflow_schema import is_supported_canvas

        if not is_supported_canvas(graph):
            _refuse("content_unverified", "Cloud did not return a supported saved canvas.")
        # Browser JSON.parse rejects NaN/Infinity, including opaque metadata.
        json.dumps(graph, allow_nan=False)
        query = {"savedFile": canonical_id, "workspace": workspace}
        result = {
            "savedFileId": canonical_id,
            "workspaceId": workspace,
            "environment": environment,
            "filename": filename,
            "verified": True,
        }
        if environment == "custom":
            deployment = _deployment_client().get_deployment(deployment_id)
            selected = verify_deployment_editor_target(
                Client(target), workspace_id=workspace, deployment_id=deployment_id, deployment=deployment
            )
            if selected.get("workspaceId") != workspace or selected.get("deploymentId") != deployment_id:
                _refuse("deployment_unverified", "The selected custom environment was not verified.")
            result.update({"deploymentId": deployment_id, "releaseId": selected["releaseId"]})
            query["deployment"] = deployment_id
        else:
            query["environment"] = "standard"
        result["url"] = urllib.parse.urlunsplit((origin.scheme, origin.netloc, "/", urllib.parse.urlencode(query), ""))
        return result
    except WorkflowUiUnavailable:
        raise
    except EditorUnavailableError as error:
        raise WorkflowUiUnavailable(
            "deployment_unverified", "The selected custom environment could not be verified."
        ) from error
    except (
        HTTPError,
        Unauthenticated,
        BuilderAuthError,
        DeployResolveError,
        DeployAPIError,
        OSError,
        ValueError,
        ResponseTooLarge,
        RecursionError,
    ) as error:
        # Workflow content, prompts and transport bodies never enter errors.
        raise WorkflowUiUnavailable(
            "unavailable", "Cloud could not verify this private workflow and target; no link was generated."
        ) from error


def run_ui(
    workflow_id: str | None = None,
    *,
    saved_file_id: str | None = None,
    workspace: str,
    environment: str,
    deployment_id: str | None = None,
) -> None:
    renderer = get_renderer()
    try:
        if workflow_id is not None or not saved_file_id:
            _refuse(
                "legacy_identity_unsupported" if workflow_id is not None else "invalid_identity",
                "Workflow-record links are unavailable. Regenerate with `workflow ui --saved-file <saved_file_id>` "
                "using the userdata UUID returned by a supported save, or open the verified file in Cloud → Workflows. "
                "Do not repeat saves or uploads solely to obtain an ID.",
            )
        result = verify_workflow_link(
            saved_file_id, workspace=workspace, environment=environment, deployment_id=deployment_id
        )
    except WorkflowUiUnavailable as error:
        renderer.error(
            code="workflow_ui_unavailable",
            message=str(error),
            hint=_HINT,
            details={"reason": error.reason, "status": error.status},
            where="cloud",
            command="workflow ui",
        )
        raise typer.Exit(code=1) from error
    if renderer.is_pretty():
        renderer.print(result["url"])
    renderer.emit(result, command="workflow ui", where="cloud", changed=False)
