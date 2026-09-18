"""``comfy assets library`` — browse and borrow assets from Comfy Cloud's
asset library.

Mirrors the cloud-saved-workflow subcommands in ``workflow.py`` (``list``,
``get``, ...): thin Typer commands over ``cloud_http``'s shared helpers,
emitting a JSON envelope via the renderer. Cloud-only — there is no local
``/api/assets`` surface.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.command.cloud_http import (
    ResponseUnparseable,
    cloud_target_or_local_error,
    handle_cloud_http_error,
    http_request,
)
from comfy_cli.output.renderer import get_renderer

app = typer.Typer(help="Browse your Comfy Cloud asset library (list, borrow).")


@app.command("ls", help="List your assets on Comfy Cloud.")
@tracking.track_command("assets")
def ls_cmd(
    name: Annotated[
        str | None,
        typer.Option("--name", show_default=False, help="Case-insensitive substring match on asset name."),
    ] = None,
    tags: Annotated[
        str | None,
        typer.Option(
            "--tags", show_default=False, help="Comma-separated tags; assets must have ALL of them (e.g. input,output)."
        ),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap rows returned (max 500).")] = 20,
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.error
    import urllib.parse

    renderer = get_renderer()
    target = cloud_target_or_local_error(where, renderer)

    params: list[tuple[str, Any]] = [("limit", min(max(limit, 1), 500))]
    if name:
        params.append(("name_contains", name))
    for t in tags.split(",") if tags else []:
        t = t.strip()
        if t:
            params.append(("include_tags", t))
    url = target.url("assets") + "?" + urllib.parse.urlencode(params)

    try:
        # `strict_json` so a 200 carrying a proxy/captive-portal error page is
        # reported as malformed rather than collapsed to `None`, which is
        # indistinguishable here from a genuinely empty body — and would render
        # a broken response as a successful EMPTY library, the same masquerade
        # the shape guards below refuse to make.
        _, body = http_request(url, target, strict_json=True)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        raise handle_cloud_http_error(renderer, e, operation="list") from e
    except ResponseUnparseable as e:
        renderer.error(
            code="cloud_http_error",
            message="unexpected response from /api/assets (body was not valid JSON)",
            hint="check whether a proxy or captive portal is intercepting the request",
            details={"operation": "list"},
        )
        raise typer.Exit(code=1) from e

    # A non-empty body decodes here only if it was valid JSON, but valid JSON is
    # not necessarily the shape we asked for: a proxy or error page can answer 200
    # with an array or a scalar, and `b.get(...)` would then raise a raw
    # `AttributeError` past the `except` above — a traceback instead of an error
    # envelope. Guard it the way `workflow list` guards the same failure on the
    # sibling endpoint. An empty body stays a legitimate `None` (→ no rows).
    if body is not None and not isinstance(body, dict):
        renderer.error(
            code="cloud_http_error",
            message="unexpected response shape from /api/assets (expected a JSON object)",
            details={"got_type": type(body).__name__},
        )
        raise typer.Exit(code=1)

    b = body or {}
    # A missing/empty `assets` is a legitimately-empty listing; a present non-list
    # `assets` is malformed the same way, and `len(rows)` would raise `TypeError`.
    rows = b.get("assets")
    if rows is None:
        rows = []
    elif not isinstance(rows, list):
        renderer.error(
            code="cloud_http_error",
            message="unexpected response shape from /api/assets (assets must be a JSON array)",
            details={"got_type": type(rows).__name__},
        )
        raise typer.Exit(code=1)
    assets = [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "hash": r.get("hash"),
            "mime_type": r.get("mime_type"),
            "size": r.get("size"),
            "tags": r.get("tags"),
            "preview_url": r.get("preview_url"),
            "job_id": r.get("job_id"),
            "created_at": r.get("created_at"),
        }
        for r in rows
        if isinstance(r, dict)
    ]
    payload = {
        # `count` describes the array actually emitted, not the raw server rows.
        # A non-dict row is dropped from `assets` above, so counting raw rows
        # reported more items than the payload carries — and a knowingly-skewed
        # `count` sitting next to a forwarded `total` undercuts the point of
        # forwarding an authoritative count at all. Identical to the old value
        # for every well-formed response; it differs only where the old value
        # was simply wrong. Matches the repo convention in
        # `comfy_cli/command/nodes.py`, where `count` is over the emitted rows.
        "count": len(assets),
        "assets": assets,
    }
    # Forward the server's own truncation signal instead of making callers infer
    # it from an exactly-full page: `has_more`/`total` are both `required` on the
    # cloud API's `ListAssetsResponse`, and `has_more` comes off a limit+1
    # sentinel row rather than off the returned row count, so it stays correct
    # on a page that comes back short. Only forward what the server actually
    # sent — an older or local server may omit them, and a JSON `null` in the
    # envelope would poison a consumer's type assertion, so omit the key rather
    # than emitting None.
    #
    # Each field is checked against its OWN type, not a shared bool-or-int test:
    # `bool` is a subclass of `int` in Python, so one shared check would forward
    # `has_more: 0` and `total: false` and emit an envelope that violates
    # schemas/assets_library.json — the very contract this command publishes.
    if isinstance(b.get("has_more"), bool):
        payload["has_more"] = b["has_more"]
    total = b.get("total")
    # `>= 0` because the schema publishes `total` as a non-negative integer, and a
    # guard looser than the declared contract is how the envelope ends up violating
    # its own schema.
    if isinstance(total, int) and not isinstance(total, bool) and total >= 0:
        payload["total"] = total
    renderer.emit(payload, command="assets library ls", where="cloud")


@app.command("ensure", help="Ensure you own an asset by content hash (borrows public/shared bytes, no re-upload).")
@tracking.track_command("assets")
def ensure_cmd(
    hash: Annotated[str, typer.Option("--hash", help="Asset content hash (as returned by `assets library ls`).")],
    tags: Annotated[
        str,
        typer.Option("--tags", help="Comma-separated tags to attach (>=1 required by the API)."),
    ] = "input",
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.error

    renderer = get_renderer()
    target = cloud_target_or_local_error(where, renderer)

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] or ["input"]
    url = target.url("assets/from-hash")
    try:
        status, body = http_request(url, target, method="POST", body={"hash": hash, "tags": tag_list})
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        # The parameterized helper, not `cloud_http`'s: that one hardcodes the
        # saved-workflow vocabulary, so a 404 here read "workflow not found
        # (ensure)" with a hint to list workflows — for a request that never
        # named a workflow. Seen in prod when an agent passed a file name where
        # the content hash belongs.
        from comfy_cli.command._cloud_errors import handle_cloud_http_error as _handle_cloud_http_error

        raise _handle_cloud_http_error(
            renderer,
            e,
            operation="ensure",
            not_found_code="asset_not_found",
            not_found_message=f"no asset with content hash {hash!r} in your Comfy Cloud library",
            not_found_hint=(
                "pass the `hash` from `comfy --json assets library ls --name <file>` "
                "(a file name is not a hash), or upload the file first with `comfy upload <file> --where cloud`"
            ),
            id_label="hash",
            resource_id=hash,
        ) from e

    b = body or {}
    payload = {
        "id": b.get("id"),
        "hash": b.get("hash", hash),
        "created_new": status == 201,
    }
    renderer.emit(payload, command="assets library ensure", where="cloud")
