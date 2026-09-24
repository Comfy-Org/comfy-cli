"""Shared cloud HTTP/URL error → structured-envelope mapping.

``comfy workflow`` and ``comfy jobs`` both talk to Comfy Cloud over ``urllib``
and must turn transport failures into the same structured error envelopes. This
helper is the single source of truth for that mapping so the two call sites
can't drift — historically only ``workflow`` mapped 401/403 to
``cloud_unauthorized`` (with an actionable ``comfy cloud login`` hint) while
``jobs`` emitted a generic ``cloud_http_error``; routing both through here
closes that gap.

The 404 branch is deliberately caller-parameterized: ``workflow`` surfaces
``workflow_not_found`` and ``jobs`` surfaces ``prompt_not_found``, each with its
own message/hint. Everything else — the bounded body read, the
401/403 → ``cloud_unauthorized`` branch, the 429 → ``cloud_rate_limited``
branch, the generic ``cloud_http_error``, and the ``URLError``/``OSError``
network hint — is shared. :func:`emit_status_error` is the 429-vs-generic split
on its own, for callers (``run``, ``cloud_http``) that map the other statuses
with their own vocabulary.
"""

from __future__ import annotations

import urllib.error

import typer

# Cap on the server error body we surface in ``details.body``. ``base_url`` is
# env-configurable, so a hostile or misbehaving endpoint must not be able to
# OOM the CLI with an unbounded error response.
_MAX_ERROR_BODY_BYTES = 1000

# A 401 is unambiguously an authentication failure. A 403 is not — it is also
# how the server denies a forbidden resource, a quota, or an already-finished
# job, so pointing the user straight at re-login would be a misleading
# remediation. Send them to the server's own explanation instead.
_UNAUTHORIZED_HINTS = {
    401: "re-run `comfy cloud login`",
    403: "re-run `comfy cloud login` if your session expired; otherwise check `details.body` — the server may be denying access to this resource",
}


def _read_error_body(e: urllib.error.HTTPError) -> str:
    """Best-effort read of a bounded slice of the server's error body.

    A read that raises (reset or truncated stream) must not pre-empt the
    structured envelope this module exists to emit, so failures degrade to an
    empty body rather than escaping as an unhandled traceback.
    """
    try:
        return (e.read(_MAX_ERROR_BODY_BYTES) or b"").decode("utf-8", "replace")
    except Exception:
        return ""


def retry_after_from_headers(headers) -> float | None:
    """The server's ``Retry-After`` (delta-seconds) from a urllib error's headers."""
    from comfy_cli.comfy_client import _parse_retry_after

    return _parse_retry_after(headers)


def emit_status_error(
    renderer,
    *,
    status: int,
    retry_after: float | None,
    operation: str,
    message: str,
    hint: str | None,
    details: dict,
    rate_limited_next_step: str = "retry it unchanged",
) -> None:
    """Emit the envelope for a cloud HTTP status that has no caller-specific code.

    A 429 is throttling, not a rejection: the request was never judged, so the
    generic ``cloud_http_error`` (whose callers' hints say "check the workflow
    is valid", "check `details.body`") sends an agent off to rewrite a request
    that was fine. It gets ``cloud_rate_limited`` and a retry hint instead,
    with the server's ``Retry-After`` in ``details.retry_after`` when it sent
    one. Every other status keeps the caller's ``cloud_http_error`` envelope
    exactly as given. ``rate_limited_next_step`` finishes the 429 hint for a
    caller whose request already had an effect (``run``'s poll: the job was
    submitted, so re-running would submit a second one).
    """
    if status != 429:
        renderer.error(code="cloud_http_error", message=message, hint=hint, details=details)
        return
    rate_details = {**details, "status": 429}
    if retry_after is not None:
        rate_details["retry_after"] = int(retry_after) if float(retry_after).is_integer() else retry_after
        wait = f"wait {rate_details['retry_after']}s (`details.retry_after`)"
    else:
        wait = "wait a few seconds"
    renderer.error(
        code="cloud_rate_limited",
        message=f"Comfy Cloud rate-limited the {operation} request (HTTP 429): too many requests",
        hint=f"the request was throttled, not rejected — {wait} and {rate_limited_next_step}",
        details=rate_details,
    )


def handle_cloud_http_error(
    renderer,
    e: Exception,
    *,
    operation: str,
    not_found_code: str,
    not_found_message: str,
    not_found_hint: str,
    id_label: str,
    resource_id: str | None = None,
    not_found_details: dict | None = None,
) -> typer.Exit:
    """Map a cloud HTTP/URL failure to a structured error envelope.

    Emits the error via ``renderer.error`` and returns a ``typer.Exit`` for the
    caller to ``raise ... from e`` so the original traceback is preserved.

    Args:
        operation: short verb naming what failed (``"get"``, ``"cancel"``, …);
            used in messages and detail payloads.
        not_found_code / not_found_message / not_found_hint: the 404 envelope,
            which differs per caller (``workflow_not_found`` vs
            ``prompt_not_found``).
        id_label: detail key for ``resource_id`` (``"workflow_id"`` /
            ``"prompt_id"``).
        resource_id: the id being operated on, or ``None`` for id-less
            operations (e.g. ``list``).
        not_found_details: extra ``details`` keys for the 404 envelope only.
    """
    id_detail = {id_label: resource_id}
    if isinstance(e, urllib.error.HTTPError):
        if e.code == 404:
            renderer.error(
                code=not_found_code,
                message=not_found_message,
                hint=not_found_hint,
                details={**id_detail, "operation": operation, **(not_found_details or {})},
            )
        elif e.code in (401, 403):
            renderer.error(
                code="cloud_unauthorized",
                message=f"HTTP {e.code} during {operation}",
                hint=_UNAUTHORIZED_HINTS[e.code],
                details={"status": e.code, "body": _read_error_body(e), "operation": operation, **id_detail},
            )
        else:
            emit_status_error(
                renderer,
                status=e.code,
                retry_after=retry_after_from_headers(getattr(e, "headers", None)),
                operation=operation,
                message=f"HTTP {e.code} during {operation}",
                hint="check `details.body` for the server's message",
                details={"status": e.code, "body": _read_error_body(e), "operation": operation, **id_detail},
            )
    else:
        renderer.error(
            code="cloud_http_error",
            message=f"{operation} failed: {e}",
            hint="check network / `comfy cloud whoami`",
        )
    return typer.Exit(code=1)
