"""V2 data-plane job submission with reject-on-duplicate idempotency."""

from __future__ import annotations

import json
import time
import urllib.error
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Protocol

from comfy_cli import credentials
from comfy_cli.command.build_spec import JsonObject
from comfy_cli.deploy_api_errors import DeployAPIError, assert_safe_deploy_url
from comfy_cli.http import ResponseTooLarge, read_capped, request_json
from comfy_cli.target import Target

_MAX_JSON: Final = 5 * 1024 * 1024
_MAX_ATTEMPTS: Final = 3
_RETRIABLE_CODES: Final = frozenset({"deployment_not_ready", "queue_full"})
# Matches `deploy_events.MAX_IDLE_INTERVAL_SECONDS`: the ceiling on how long a
# server-sent `Retry-After` may hold a foreground command.
_MAX_RETRY_AFTER_SECONDS: Final = 10.0
_UNKNOWN_MESSAGE: Final = (
    "The job may exist, but the API has no job-list endpoint, no lookup by idempotency key, "
    "and no client-supplied job id, so there is no way to find the possibly-created job."
)

_SUBMIT_ERRORS: Final = {
    (401, "unauthorized"): {"code": "deploy_not_signed_in", "message": "the data-plane request is not authenticated"},
    (402, "insufficient_credits"): {
        "code": "deploy_payment_required",
        "message": "the job requires available credit",
    },
    (403, "forbidden"): {"code": "deploy_forbidden", "message": "the job submission is forbidden"},
    (404, "not_found"): {"code": "deploy_not_found", "message": "the deployment endpoint was not found"},
    (422, "deployment_stopped"): {
        "code": "deploy_conflict",
        "message": "the deployment is stopped and cannot accept jobs",
    },
    (422, "idempotency_key_reuse"): {
        "code": "deploy_idempotency_reuse",
        "message": "the idempotency key has already been used",
    },
    (422, "missing_asset"): {
        "code": "deploy_asset_missing",
        "message": "the workflow references an asset this account cannot mint",
    },
    (422, "invalid_workflow"): {
        "code": "deploy_workflow_invalid",
        "message": "the workflow is invalid",
    },
    (422, "workflow_format_ui"): {
        "code": "deploy_workflow_invalid",
        "message": "the workflow is not in API format",
    },
    (429, "deployment_not_ready"): {
        "code": "deploy_not_ready",
        "message": "the deployment is not ready to accept jobs",
    },
    (429, "queue_full"): {"code": "deploy_rate_limited", "message": "the deployment job queue is full"},
}

_STATUS_ERRORS: Final = {
    400: {"code": "deploy_bad_request", "message": "the job submission request is invalid"},
    401: {"code": "deploy_not_signed_in", "message": "the data-plane request is not authenticated"},
    402: {"code": "deploy_payment_required", "message": "the job requires available credit"},
    403: {"code": "deploy_forbidden", "message": "the job submission is forbidden"},
    404: {"code": "deploy_not_found", "message": "the deployment endpoint was not found"},
    409: {"code": "deploy_conflict", "message": "the job submission conflicts with the deployment state"},
    422: {"code": "deploy_workflow_invalid", "message": "the workflow is invalid"},
    429: {"code": "deploy_rate_limited", "message": "the deployment rejected the job rate"},
}


class DeploymentReader(Protocol):
    def get_deployment(self, deployment_id: str, /) -> JsonObject: ...


@dataclass(frozen=True, slots=True)
class JobSubmitRequest:
    workflow: JsonObject
    idempotency_key: str
    deployment_id: str


@dataclass(frozen=True, slots=True)
class _ServerError:
    code: str | None
    details: JsonObject


def _server_error(error: urllib.error.HTTPError, url: str, secret: str | None) -> _ServerError:
    try:
        raw = read_capped(error, url, max_bytes=_MAX_JSON)
    except ResponseTooLarge:
        return _ServerError(code=None, details={})
    if not raw:
        return _ServerError(code=None, details={})
    try:
        text = raw.decode("utf-8")
        parsed = json.loads(text.replace(secret, "[redacted]") if secret else text)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return _ServerError(code=None, details={})
    if not isinstance(parsed, dict):
        return _ServerError(code=None, details={})
    payload = parsed.get("error")
    if not isinstance(payload, dict):
        return _ServerError(code=None, details={})
    code = payload.get("code")
    details = payload.get("details")
    return _ServerError(
        code=code if isinstance(code, str) else None,
        details=details if isinstance(details, dict) else {},
    )


def _retry_after(error: urllib.error.HTTPError) -> float | None:
    """The server's requested backoff, bounded.

    A `Retry-After` is a hint, not an instruction to hand the process over: an
    unbounded one parks a foreground command for as long as the header says —
    `Retry-After: 86400` on a 429 would sleep a day inside `submit_job`, with no
    output and nothing to interrupt but the terminal. `deploy_events` already
    clamps its own; this is the same ceiling.
    """
    value = error.headers.get("Retry-After") if error.headers is not None else None
    if value is None:
        return None
    try:
        seconds = int(value)
    except ValueError:
        return None
    return min(float(seconds), _MAX_RETRY_AFTER_SECONDS) if seconds >= 0 else None


class DeployJobClient:
    def __init__(self, base_url: str, token: str, *, sleep: Callable[[float], None] = time.sleep):
        resolved_url = base_url.rstrip("/")
        assert_safe_deploy_url(resolved_url, source="the deployment endpointUrl")
        self.target = Target(kind="cloud", base_url=resolved_url, path_prefix="/api/v2", auth_token=token)
        self._sleep = sleep

    def submit_job(self, request: JobSubmitRequest, control_plane: DeploymentReader) -> JsonObject:
        if not request.idempotency_key.strip():
            raise DeployAPIError("deploy_bad_request", "Idempotency-Key is required")
        partner_key = credentials.find_api_key(purpose="partner")
        api_key = partner_key.value if partner_key is not None else None
        body: JsonObject = {"workflow": request.workflow}
        if api_key is not None:
            body["extra_data"] = {"api_key_comfy_org": api_key}
        url = self.target.url("jobs")
        headers = {"Idempotency-Key": request.idempotency_key}

        for attempt in range(_MAX_ATTEMPTS):
            try:
                status, parsed = request_json(
                    url,
                    self.target,
                    method="POST",
                    body=body,
                    headers=headers,
                    max_bytes=_MAX_JSON,
                )
            except urllib.error.HTTPError as error:
                if 500 <= error.code <= 599:
                    raise self._unknown(request.deployment_id) from error
                server = _server_error(error, url, api_key)
                delay = _retry_after(error)
                if delay is not None and error.code == 429 and server.code in _RETRIABLE_CODES:
                    if attempt + 1 < _MAX_ATTEMPTS:
                        self._sleep(delay)
                        continue
                raise self._mapped(error.code, server, request, control_plane) from error
            except (TimeoutError, urllib.error.URLError) as error:
                raise self._unknown(request.deployment_id) from error

            if status == 201 and isinstance(parsed, dict):
                return parsed
            raise DeployAPIError(
                "deploy_server_error",
                "the data plane returned an invalid job-submission response",
                status=status,
                details={"http_status": status, "deployment_id": request.deployment_id},
            )
        raise AssertionError("unreachable submission attempt state")

    @staticmethod
    def _unknown(deployment_id: str) -> DeployAPIError:
        return DeployAPIError(
            code="deploy_job_submit_unknown",
            message=_UNKNOWN_MESSAGE,
            details={"deployment_id": deployment_id},
        )

    @staticmethod
    def _mapped(
        status: int,
        server: _ServerError,
        request: JobSubmitRequest,
        control_plane: DeploymentReader,
    ) -> DeployAPIError:
        rule = _STATUS_ERRORS.get(status, _STATUS_ERRORS[400])
        if server.code is not None:
            rule = _SUBMIT_ERRORS.get((status, server.code), rule)
        details: JsonObject = {"http_status": status}
        if server.code is not None:
            details["server_code"] = server.code
        if server.code == "invalid_workflow" and "node_errors" in server.details:
            details["node_errors"] = server.details["node_errors"]
        if status == 429 and server.code == "deployment_not_ready":
            deployment = control_plane.get_deployment(request.deployment_id)
            deployment_status = deployment.get("status")
            if not isinstance(deployment_status, str) or not deployment_status:
                raise DeployAPIError(
                    "deploy_server_error",
                    "the control plane returned no deployment status",
                    details={"deployment_id": request.deployment_id},
                )
            details["status"] = deployment_status
            details["deployment_id"] = request.deployment_id
        code = rule["code"]
        message = rule["message"]
        if not isinstance(code, str) or not isinstance(message, str):
            raise AssertionError("invalid job-submission error rule")
        return DeployAPIError(code, message, status=status, details=details)
