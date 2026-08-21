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
        _, body = http_request(url, target)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        raise handle_cloud_http_error(renderer, e, operation="list") from e

    rows = (body or {}).get("assets") or []
    payload = {
        "count": len(rows),
        "assets": [
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
        ],
    }
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
        raise handle_cloud_http_error(renderer, e, operation="ensure") from e

    b = body or {}
    payload = {
        "id": b.get("id"),
        "hash": b.get("hash", hash),
        "created_new": status == 201,
    }
    renderer.emit(payload, command="assets library ensure", where="cloud")
