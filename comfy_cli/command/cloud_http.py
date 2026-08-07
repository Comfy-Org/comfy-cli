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

    req = urllib.request.Request(url, data=data, method=method)
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")
    if content_type:
        req.add_header("Content-Type", content_type)
    return req


def http_request(
    url: str, target, *, method: str = "GET", body: dict | None = None, timeout: float = 30.0
) -> tuple[int, dict | None]:
    """Authed HTTP call returning (status, parsed_json_or_none). Raises
    urllib errors verbatim so callers can surface the right error code."""
    import urllib.request

    data = json.dumps(body).encode("utf-8") if body is not None else None
    ct = "application/json" if data is not None else None
    req = _authed_request(url, target, method=method, data=data, content_type=ct)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        raw = resp.read(64 * 1024 * 1024)  # 64 MiB cap
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, None


def handle_cloud_http_error(renderer, e, *, operation: str, workflow_id: str | None = None) -> typer.Exit:
    """Map HTTP failures to envelope codes. Returns an Exit to ``raise from``."""
    import urllib.error

    if isinstance(e, urllib.error.HTTPError):
        body = (e.read() or b"")[:1000].decode("utf-8", "replace")
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
            renderer.error(
                code="cloud_http_error",
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
