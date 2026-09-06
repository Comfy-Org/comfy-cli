from __future__ import annotations

import json
import urllib.error
from typing import Final

from comfy_cli.http import assert_safe_url, read_capped, tls_trust_hint, tls_verification_failed

_MAX_JSON: Final = 5 * 1024 * 1024

_BAD_REQUEST: Final = {"code": "deploy_bad_request", "message": "the deploy request is invalid"}
_SERVER_ERROR: Final = {"code": "deploy_server_error", "message": "the deploy service is unavailable"}
_NOT_SIGNED_IN: Final = {"code": "deploy_not_signed_in", "message": "the deploy request is not authenticated"}
_NOT_FOUND: Final = {"code": "deploy_not_found", "message": "the deployment was not found"}
_FORBIDDEN: Final = {"code": "deploy_forbidden", "message": "the deploy operation is forbidden"}
_CONFLICT: Final = {"code": "deploy_conflict", "message": "the deployment state conflicts with this operation"}
_PAYMENT_REQUIRED: Final = {"code": "deploy_payment_required", "message": "the deploy operation requires payment"}
_QUOTA_EXCEEDED: Final = {"code": "deploy_quota_exceeded", "message": "the deploy quota was exceeded"}
_COMPUTE_UNAVAILABLE: Final = {"code": "deploy_compute_unavailable", "message": "the requested compute is unavailable"}
_IMMUTABLE_COMPUTE: Final = {"code": "deploy_immutable_compute", "message": "ready deployment compute is immutable"}
_DELETED: Final = {"code": "deploy_deleted", "message": "the deployment was deleted"}
_INSECURE_URL: Final = {"code": "deploy_insecure_url", "message": "the deploy endpoint is not https"}

STATUS_ERRORS: Final = {
    ("create", 400): _COMPUTE_UNAVAILABLE,
    ("create", 401): _NOT_SIGNED_IN,
    ("create", 402): _PAYMENT_REQUIRED,
    ("create", 403): _FORBIDDEN,
    ("create", 404): _NOT_FOUND,
    ("create", 409): _CONFLICT,
    ("create", 429): _QUOTA_EXCEEDED,
    ("create", 500): _SERVER_ERROR,
    ("list", 400): _BAD_REQUEST,
    ("list", 401): _NOT_SIGNED_IN,
    ("list", 402): _PAYMENT_REQUIRED,
    ("list", 403): _FORBIDDEN,
    ("list", 404): _NOT_FOUND,
    ("list", 409): _CONFLICT,
    ("list", 429): _QUOTA_EXCEEDED,
    ("list", 500): _SERVER_ERROR,
    ("get", 400): _BAD_REQUEST,
    ("get", 401): _NOT_SIGNED_IN,
    ("get", 402): _PAYMENT_REQUIRED,
    ("get", 403): _FORBIDDEN,
    ("get", 404): _NOT_FOUND,
    ("get", 409): _CONFLICT,
    ("get", 429): _QUOTA_EXCEEDED,
    ("get", 500): _SERVER_ERROR,
    ("scale", 400): _COMPUTE_UNAVAILABLE,
    ("scale", 401): _NOT_SIGNED_IN,
    ("scale", 402): _PAYMENT_REQUIRED,
    ("scale", 403): _FORBIDDEN,
    ("scale", 404): _NOT_FOUND,
    ("scale", 409): _IMMUTABLE_COMPUTE,
    ("scale", 429): _QUOTA_EXCEEDED,
    ("scale", 500): _SERVER_ERROR,
    ("delete", 400): _BAD_REQUEST,
    ("delete", 401): _NOT_SIGNED_IN,
    ("delete", 402): _PAYMENT_REQUIRED,
    ("delete", 403): _FORBIDDEN,
    ("delete", 404): _NOT_FOUND,
    ("delete", 409): _CONFLICT,
    ("delete", 429): _QUOTA_EXCEEDED,
    ("delete", 500): _SERVER_ERROR,
    ("start", 400): _COMPUTE_UNAVAILABLE,
    ("start", 401): _NOT_SIGNED_IN,
    ("start", 402): _PAYMENT_REQUIRED,
    ("start", 403): _FORBIDDEN,
    ("start", 404): _NOT_FOUND,
    ("start", 409): _DELETED,
    ("start", 429): _QUOTA_EXCEEDED,
    ("start", 500): _SERVER_ERROR,
    ("stop", 400): _BAD_REQUEST,
    ("stop", 401): _NOT_SIGNED_IN,
    ("stop", 402): _PAYMENT_REQUIRED,
    ("stop", 403): _FORBIDDEN,
    ("stop", 404): _NOT_FOUND,
    ("stop", 409): _CONFLICT,
    ("stop", 429): _QUOTA_EXCEEDED,
    ("stop", 500): _SERVER_ERROR,
    ("events", 400): _BAD_REQUEST,
    ("events", 401): _NOT_SIGNED_IN,
    ("events", 402): _PAYMENT_REQUIRED,
    ("events", 403): _FORBIDDEN,
    ("events", 404): _NOT_FOUND,
    ("events", 409): _CONFLICT,
    ("events", 429): _QUOTA_EXCEEDED,
    ("events", 500): _SERVER_ERROR,
    ("logs", 400): _BAD_REQUEST,
    ("logs", 401): _NOT_SIGNED_IN,
    ("logs", 402): _PAYMENT_REQUIRED,
    ("logs", 403): _FORBIDDEN,
    ("logs", 404): _NOT_FOUND,
    ("logs", 409): _CONFLICT,
    ("logs", 429): _QUOTA_EXCEEDED,
    ("logs", 500): _SERVER_ERROR,
    ("compute", 400): _BAD_REQUEST,
    ("compute", 401): _NOT_SIGNED_IN,
    ("compute", 402): _PAYMENT_REQUIRED,
    ("compute", 403): _FORBIDDEN,
    ("compute", 404): _NOT_FOUND,
    ("compute", 409): _CONFLICT,
    ("compute", 429): _QUOTA_EXCEEDED,
    ("compute", 500): _SERVER_ERROR,
}


class DeployAPIError(Exception):
    """A deploy failure, with the navigation its handler renders.

    ``hint`` is specific to THIS failure rather than to its code — the CA store
    actually in use, say. ``None`` for almost every instance, and
    ``Renderer.error`` substitutes the registered hint for ``None``.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        status: int | None = None,
        details: dict | None = None,
        hint: str | None = None,
    ) -> None:
        self.code = code
        self.status = status
        self.details = details or {}
        self.hint = hint
        super().__init__(message)


def _compute_or_bad_request(message: str) -> dict[str, str]:
    lowered = message.lower()
    markers = (
        " is not available in region ",
        " is not currently available",
        "cannot be used for a deployment",
    )
    return _COMPUTE_UNAVAILABLE if any(marker in lowered for marker in markers) else _BAD_REQUEST


def _immutable_or_conflict(message: str) -> dict[str, str]:
    return _IMMUTABLE_COMPUTE if "changing gpuclass or region" in message.lower() else _CONFLICT


def _deleted_or_conflict(message: str) -> dict[str, str]:
    return _DELETED if "deleted" in message.lower() else _CONFLICT


_CONTEXTUAL_ERRORS: Final = {
    ("create", 400): _compute_or_bad_request,
    ("scale", 400): _compute_or_bad_request,
    ("start", 400): _compute_or_bad_request,
    ("scale", 409): _immutable_or_conflict,
    ("start", 409): _deleted_or_conflict,
}


def _error_body(error: urllib.error.HTTPError, url: str) -> tuple[str | None, str | None]:
    raw = read_capped(error, url, max_bytes=_MAX_JSON)
    if not raw:
        return None, None
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
        return None, None
    if not isinstance(parsed, dict):
        return None, None
    server_code = parsed.get("error")
    message = parsed.get("message")
    return (
        server_code if isinstance(server_code, str) else None,
        message if isinstance(message, str) else None,
    )


def mapped_error(operation: str, error: urllib.error.HTTPError, url: str) -> DeployAPIError:
    server_code, server_message = _error_body(error, url)
    status = error.code
    lookup_status = 500 if 500 <= status <= 599 else status
    rule = STATUS_ERRORS.get((operation, lookup_status), _SERVER_ERROR)
    classifier = _CONTEXTUAL_ERRORS.get((operation, status))
    message = server_message or str(error.reason) or rule["message"]
    if classifier is not None:
        rule = classifier(message)
    details = {"operation": operation, "status": status}
    if server_code is not None:
        details["server_code"] = server_code
    return DeployAPIError(rule["code"], message, status=status, details=details)


def bad_request(message: str) -> DeployAPIError:
    return DeployAPIError(_BAD_REQUEST["code"], message)


def assert_safe_deploy_url(url: str, *, source: str) -> None:
    """Refuse an insecure deploy endpoint as a typed error, not a traceback.

    ``http.assert_safe_url`` signals refusal with a bare ``ValueError``, and no
    deploy command lists ``ValueError`` in its ``except`` tuple — so a
    misconfigured endpoint escaped the command layer entirely and ``--json``
    emitted no envelope at all, breaking the machine-output contract every
    other deploy failure honours. Raising ``DeployAPIError`` puts the refusal
    back inside the handler that already renders it.

    ``source`` names the setting actually in play. The underlying message
    points at ``COMFY_CLOUD_BASE_URL``, which is never what configured a deploy
    endpoint: the control plane reads ``COMFY_DEPLOY_URL`` and the data plane
    takes the deployment's own ``endpointUrl``.
    """
    try:
        assert_safe_url(url)
    except ValueError as error:
        raise DeployAPIError(
            _INSECURE_URL["code"],
            f"refusing to send credentials to non-https, non-loopback URL: {url} (from {source})",
            details={"url": url, "source": source},
        ) from error


def transport_error(operation: str, error: TimeoutError | urllib.error.URLError) -> DeployAPIError:
    # A local trust problem, not the unavailable service `_SERVER_ERROR`
    # describes — the two send the reader to opposite places.
    if tls_verification_failed(error):
        return DeployAPIError(
            "tls_verify_failed",
            f"TLS certificate verification failed: {error}",
            details={"operation": operation},
            hint=tls_trust_hint(),
        )
    return DeployAPIError(_SERVER_ERROR["code"], str(error), details={"operation": operation})
