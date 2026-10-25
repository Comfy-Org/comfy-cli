"""Read-only, capability-gated URLs for normal private saved-workflow identity."""

from __future__ import annotations

import json
import re
import urllib.parse
from datetime import datetime
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


def _valid_datetime(value: object) -> bool:
    # Match the frontend's z.iso.datetime() UTC contract and validate the
    # calendar, including impossible month/day/time values.
    match = (
        re.fullmatch(r"([0-9]{4})-([0-9]{2})-([0-9]{2})T([0-9]{2}):([0-9]{2}):([0-9]{2})(?:\.[0-9]+)?Z", value)
        if isinstance(value, str)
        else None
    )
    if not match:
        return False
    try:
        datetime(*(int(part) for part in match.groups()))
    except ValueError:
        return False
    return True


def _optional_metadata_valid(metadata: dict) -> bool:
    if "default_view" in metadata and metadata["default_view"] not in ("workflow", "app"):
        return False
    if "description" in metadata and not isinstance(metadata["description"], str):
        return False
    if "forked_from" in metadata:
        fork = metadata["forked_from"]
        if not isinstance(fork, dict) or any(
            key in fork and not isinstance(fork[key], str) for key in ("workflow_id", "workflow_version_id")
        ):
            return False
    return True


def _content_references_valid(content: dict) -> bool:
    if "dependency_asset_ids" not in content:
        return True
    refs = content["dependency_asset_ids"]
    return isinstance(refs, list) and all(isinstance(ref, str) for ref in refs)


def _safe_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and 0 < len(name) <= 1024
        and not name.startswith(("/", "~"))
        and "\\" not in name
        and not any(ord(c) < 32 or ord(c) == 127 for c in name)
        and all(part.rstrip(" .") not in ("", ".", "..") for part in name.split("/"))
    )


def _same_json(left, right) -> bool:
    # Python considers True == 1, unlike the frontend's JSON deep equality.
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same_json(value, right[key]) for key, value in left.items())
    if isinstance(left, list):
        return len(left) == len(right) and all(_same_json(a, b) for a, b in zip(left, right))
    return left == right


def _deployment_client():
    from comfy_cli.deploy_api import DeployClient

    return DeployClient.from_credentials()


def verify_workflow_link(
    workflow_id: str, *, workspace: str, environment: str, deployment_id: str | None = None
) -> dict:
    """Verify current access, normal file-backed identity and selected environment.

    No save/share operation, browser launch, environment mutation or compute
    operation occurs. A capability absent on an older release is a hard refusal.
    """
    try:
        canonical_id = str(UUID(workflow_id))
    except (ValueError, AttributeError, TypeError):
        _refuse("invalid_identity", "Use the exact saved workflow UUID.")
    if canonical_id != workflow_id.lower():
        _refuse("invalid_identity", "Use the canonical saved workflow UUID.")
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
        if not isinstance(features, dict) or features.get("private_workflow_links_enabled") is not True:
            _refuse(
                "capability_unavailable", "This Cloud release has not enabled verified private saved-workflow links."
            )
        metadata = _get(target, "workflows", canonical_id)
        if (
            not isinstance(metadata, dict)
            or metadata.get("id") != canonical_id
            or not all(
                isinstance(metadata.get(key), str) and metadata[key]
                for key in ("created_by", "created_at", "updated_at")
            )
            or not _valid_datetime(metadata.get("created_at"))
            or not _valid_datetime(metadata.get("updated_at"))
            or not isinstance(metadata.get("latest_version"), int)
            or isinstance(metadata.get("latest_version"), bool)
            or metadata["latest_version"] < 1
            or not _optional_metadata_valid(metadata)
        ):
            _refuse("metadata_unverified", "Cloud did not identify the exact private saved workflow and version.")
        name = metadata.get("name")
        if not _safe_name(name):
            _refuse("file_unavailable", "The saved record does not identify a safe normal Workflows file.")
        content = _get(target, "workflows", canonical_id, "content", max_bytes=_MAX_WORKFLOW_BYTES)
        graph = content.get("workflow_json") if isinstance(content, dict) else None
        from comfy_cli.cloud_workflow_schema import is_supported_canvas

        if (
            not isinstance(content, dict)
            or content.get("version") != metadata["latest_version"]
            or isinstance(content.get("version"), bool)
            or not isinstance(content.get("id"), str)
            or not content["id"]
            or not isinstance(content.get("created_by"), str)
            or not content["created_by"]
            or not _valid_datetime(content.get("created_at"))
            or not _content_references_valid(content)
            or not is_supported_canvas(graph)
        ):
            _refuse("content_unverified", "Cloud did not return the current saved canvas content.")
        # Ensure both content bodies are real JSON, excluding NaN/Infinity that
        # Python's permissive decoder accepts but the browser cannot parse.
        json.dumps(graph, allow_nan=False)
        from comfy_cli.command.workflow import _userdata_file_url

        filename = f"workflows/{name}.json"
        file_url = _userdata_file_url(target, f"{name}.json")
        status, stored = request_json(file_url, target, method="GET", max_bytes=_MAX_WORKFLOW_BYTES, timeout=15)
        if status != 200 or not isinstance(stored, dict):
            _refuse("file_unavailable", "The normal Workflows sidebar file could not be verified.")
        json.dumps(stored, allow_nan=False)
        if not _same_json(graph, stored):
            _refuse("file_mismatch", "The private record and normal Workflows file contain different content.")

        query = {"workflow": canonical_id, "workspace": workspace}
        result = {
            "workflowId": canonical_id,
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


def run_ui(workflow_id: str, *, workspace: str, environment: str, deployment_id: str | None = None) -> None:
    renderer = get_renderer()
    try:
        result = verify_workflow_link(
            workflow_id, workspace=workspace, environment=environment, deployment_id=deployment_id
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
