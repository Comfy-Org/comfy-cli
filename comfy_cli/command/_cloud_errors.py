"""Shared cloud HTTP/URL error → structured-envelope mapping (BE-3266).

``comfy workflow`` and ``comfy jobs`` both talk to Comfy Cloud over ``urllib``
and must turn transport failures into the same structured error envelopes. This
helper is the single source of truth for that mapping so the two call sites
can't drift — historically only ``workflow`` mapped 401/403 to
``cloud_unauthorized`` (with an actionable ``comfy cloud login`` hint) while
``jobs`` emitted a generic ``cloud_http_error``; routing both through here
closes that gap.

The 404 branch is deliberately caller-parameterized: ``workflow`` surfaces
``workflow_not_found`` and ``jobs`` surfaces ``prompt_not_found``, each with its
own message/hint. Everything else — the identical body truncation, the
401/403 → ``cloud_unauthorized`` branch, the generic ``cloud_http_error``, and
the ``URLError``/``OSError`` network hint — is shared.
"""

from __future__ import annotations

import urllib.error

import typer


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
    """
    id_detail = {id_label: resource_id}
    if isinstance(e, urllib.error.HTTPError):
        body = (e.read() or b"")[:1000].decode("utf-8", "replace")
        if e.code == 404:
            renderer.error(
                code=not_found_code,
                message=not_found_message,
                hint=not_found_hint,
                details={**id_detail, "operation": operation},
            )
        elif e.code in (401, 403):
            renderer.error(
                code="cloud_unauthorized",
                message=f"HTTP {e.code} during {operation}",
                hint="re-run `comfy cloud login`",
                details={"status": e.code},
            )
        else:
            renderer.error(
                code="cloud_http_error",
                message=f"HTTP {e.code} during {operation}",
                hint="check `details.body` for the server's message",
                details={"status": e.code, "body": body, "operation": operation, **id_detail},
            )
    else:
        renderer.error(
            code="cloud_http_error",
            message=f"{operation} failed: {e}",
            hint="check network / `comfy auth whoami`",
        )
    return typer.Exit(code=1)
