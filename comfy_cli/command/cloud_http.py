"""Shared cloud-HTTP helpers used by ``comfy workflow``'s cloud-saved-workflow
subcommands (and by other commands that talk to Comfy Cloud's ``/api/*``
surface, e.g. ``comfy assets library``).

Extracted verbatim from ``comfy_cli/command/workflow.py`` so call sites in
multiple command modules can share one implementation instead of importing
underscore-prefixed privates across module boundaries.
"""

from __future__ import annotations

import json

import typer

# The most of a response body ``http_request`` reads.
_MAX_BODY_BYTES = 64 * 1024 * 1024


def cloud_target_or_local_error(where: str | None, renderer):
    """Resolve a cloud Target, or emit ``cloud_only_command`` for a non-cloud target."""
    from comfy_cli.target import resolve_target

    target = resolve_target(where=where)
    if not target.is_cloud:
        renderer.error(
            code="cloud_only_command",
            message="This command requires Comfy Cloud; there is no local equivalent.",
            hint="sign in with `comfy cloud login` and re-run with `--where cloud`",
        )
        raise typer.Exit(code=1)
    return target


def _authed_request(
    url: str, target, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None
):
    """Build an authenticated urllib Request. The return type is annotated
    loosely to keep urllib out of the module's top-level imports."""
    import urllib.request

    from comfy_cli.http import USAGE_SOURCE_HEADERS

    req = urllib.request.Request(url, data=data, method=method, headers=USAGE_SOURCE_HEADERS)
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")
    if content_type:
        req.add_header("Content-Type", content_type)
    return req


class ResponseUnparseable(Exception):
    """A non-empty response body that was not valid JSON.

    Only raised when a caller opts in via ``http_request(..., strict_json=True)``;
    otherwise a decode failure keeps collapsing to ``None`` as it always has.
    """


def http_request(
    url: str,
    target,
    *,
    method: str = "GET",
    body: dict | None = None,
    timeout: float = 30.0,
    strict_json: bool = False,
) -> tuple[int, dict | None]:
    """Authed HTTP call returning (status, parsed_json_or_none). Raises
    urllib errors verbatim so callers can surface the right error code.

    An *empty* body is reported as ``None``. A *non-empty* body that fails to
    decode is reported as ``None`` too, unless ``strict_json`` is set — then it
    raises ``ResponseUnparseable``. The two are different answers ("nothing to
    say" vs. "a malformed answer"), and a listing caller that conflates them
    reports a proxy or captive-portal error page as a successful empty listing.

    A connection that closes before the body it declared has arrived raises
    ``http.client.IncompleteRead``, never a decode failure: the answer was cut
    off in transit, not malformed, and a retry may get all of it.

    A body longer than ``_MAX_BODY_BYTES``, chunked or not, is never parsed; it
    gets the decode-failure outcome, since the bytes read are not the whole answer.
    """
    import http.client
    import urllib.request

    data = json.dumps(body).encode("utf-8") if body is not None else None
    ct = "application/json" if data is not None else None
    req = _authed_request(url, target, method=method, data=data, content_type=ct)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        # One byte past the cap tells a body longer than the cap apart from one
        # exactly the cap, with or without a Content-Length.
        raw = resp.read(_MAX_BODY_BYTES + 1)
        # A chunked body cut short raises IncompleteRead inside `read`, but a
        # bounded read of a Content-Length body returns whatever arrived;
        # http.client still counts the declared bytes that did not (`length`).
        unread = getattr(resp, "length", None)
    over_cap = len(raw) > _MAX_BODY_BYTES
    if isinstance(unread, int) and unread > 0 and not over_cap:
        raise http.client.IncompleteRead(raw, unread)
    # `.strip()` so a whitespace-only body counts as empty rather than as a
    # decode failure: identical to the old behaviour on the default path (it
    # collapsed to `None` either way), but it keeps `strict_json` from calling a
    # server that answers `b"\n"` malformed.
    if not over_cap and not raw.strip():
        return status, None
    try:
        if over_cap:
            # The body runs past the cap: the prefix may happen to be valid
            # JSON, but it is not the answer.
            raise ValueError(f"response body larger than {_MAX_BODY_BYTES} bytes")
        return status, json.loads(raw)
    except (ValueError, RecursionError) as e:
        # Catch the ``ValueError`` BASE, not ``JSONDecodeError``: handed raw bytes,
        # ``json.loads`` rejects a malformed body with three different exceptions and
        # only one of them is a ``JSONDecodeError``. Non-UTF-8 bytes (``b"\xff"``, a
        # binary error page) raise ``UnicodeDecodeError``; a JSON integer past
        # CPython's 4300-digit int/str limit raises a bare ``ValueError``; both
        # subclass ``ValueError``. Pathologically nested input raises
        # ``RecursionError``, which does not — the 64 MiB read permits deep nesting.
        # Narrowing to ``JSONDecodeError`` let those escape past ``strict_json`` and
        # past every call site's ``except``, crashing the CLI with a raw traceback
        # for exactly the malformed response this helper exists to classify. Same
        # reasoning, and the same catch, as the sibling helper in ``workflow.py``.
        if strict_json:
            raise ResponseUnparseable(f"non-JSON response body from {url}") from e
        return status, None


def handle_cloud_http_error(renderer, e, *, operation: str, workflow_id: str | None = None) -> typer.Exit:
    """Map HTTP failures to envelope codes. Returns an Exit to ``raise from``."""
    import http.client
    import urllib.error

    if isinstance(e, urllib.error.HTTPError):
        # Best effort: a server that drops the connection partway through its
        # error body must not keep the status from being classified.
        try:
            body = (e.read() or b"")[:1000].decode("utf-8", "replace")
        except (http.client.HTTPException, OSError):
            body = ""
        if e.code == 404:
            renderer.error(
                code="workflow_not_found",
                message=f"no saved workflow with id {workflow_id!r}"
                if workflow_id
                else f"workflow not found ({operation})",
                hint="list available workflows via `comfy --json workflow list`",
                details={"workflow_id": workflow_id, "operation": operation},
            )
        elif e.code in (401, 403):
            renderer.error(
                code="cloud_unauthorized",
                message=f"HTTP {e.code} during {operation}",
                hint="re-run `comfy cloud login`",
                details={"status": e.code},
            )
        else:
            from comfy_cli.command._cloud_errors import emit_status_error, retry_after_from_headers

            emit_status_error(
                renderer,
                status=e.code,
                retry_after=retry_after_from_headers(getattr(e, "headers", None)),
                operation=operation,
                message=f"HTTP {e.code} during {operation}",
                hint="check `details.body` for the server's message",
                details={"status": e.code, "body": body, "operation": operation},
            )
    else:
        renderer.error(
            code="cloud_http_error",
            message=f"{operation} failed: {e}",
            hint="check network / `comfy cloud whoami`",
        )
    return typer.Exit(code=1)
