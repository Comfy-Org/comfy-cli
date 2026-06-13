"""``comfy models`` — live model discovery against local or cloud.

Four subcommands, all routed by ``--where`` (cloud auto-detect by default):

    comfy models list-folders           # GET /api/experiment/models  | /models
    comfy models list-folder <folder>   # GET /api/experiment/models/<folder> | /models/<folder>
    comfy models search [--text] [--type] [--limit]  # cloud: /api/assets; local: /models/<folder>
    comfy models show <name>            # exact-match name across the catalog

The search surface mirrors the asset→model extraction used by Comfy-Org's
cloud tooling: prefer ``user_metadata`` over ``metadata`` for any given key,
treat ``tags`` as the canonical type/role signal, and surface the
densely-populated fields (``source_url``, ``preview_url``, ``size``) as
first-class result columns. Sparse fields (``base_model``, ``trained_words``)
ride along when present.

Local-mode caveats:
  * ``/models/<folder>`` returns ``[{name, pathIndex}, ...]`` — filenames only,
    no enrichment. ``search`` on local degrades to substring on the listing
    of the resolved folder.
  * The cloud asset catalog (``/api/assets``) has no local equivalent —
    local search is intentionally simpler.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
import urllib.request
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(no_args_is_help=True, help="Discover models — folders, files, and the cloud asset catalog.")

# Cap response reads from cloud/local. The largest legitimate response we see
# in practice is `/api/object_info` (~9 MB on cloud), but the model endpoints
# are far smaller. 64 MiB is generous headroom; anything beyond this is either
# a misconfigured backend or hostile and would only serve to OOM the CLI.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024

# Folder + asset-name arguments are interpolated into URL paths or query
# strings. We disallow path-traversal sequences and control characters so a
# crafted argument can't escape the intended endpoint. The set of legitimate
# folder names (loras, checkpoints, …) is alphanumeric + ``_``/`-`/`.`; the
# regex below is permissive enough for real-world filenames while rejecting
# the obvious attack shapes.
_PATH_SAFE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-]*$")


def _reject_unsafe_path_segment(value: str, *, kind: str, renderer) -> None:
    """Exit with an `invalid_argument` error if ``value`` isn't safe as a path segment."""
    if not value or ".." in value or "/" in value or "\\" in value or not _PATH_SAFE.match(value):
        renderer.error(
            code="invalid_argument",
            message=f"{kind} {value!r} contains characters that aren't allowed in a path segment",
            hint=f"valid {kind} names are alphanumeric with `_`, `-`, or `.`",
        )
        raise typer.Exit(code=1)


# Maps a friendly --type to the folder used on both backends.
_TYPE_TO_FOLDER = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "lora": "loras",
    "loras": "loras",
    "vae": "vae",
    "controlnet": "controlnet",
    "upscale": "upscale_models",
    "upscale_models": "upscale_models",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "unet": "diffusion_models",
    "diffusion": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "style": "style_models",
    "style_models": "style_models",
    "embeddings": "embeddings",
    "hypernetworks": "hypernetworks",
    "gligen": "gligen",
}
# Unknown values pass through verbatim — the backend rejects bad folder
# names with 404, which the caller surfaces as `folder_not_found`.


def _models_path_parts(target) -> tuple[str, ...]:
    """Return the URL path parts for the model-listing endpoints.

    Cloud uses the ``/api/experiment/models`` family (the legacy ``/api/models``
    explicitly 404s by design). Local stays on ``/models``.
    """
    return ("experiment", "models") if target.is_cloud else ("models",)


def _authed_request(url: str, target) -> urllib.request.Request:
    req = urllib.request.Request(url)
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")
    return req


def _http_get_json(url: str, target, timeout: float = 30.0) -> Any:
    """Issue an authenticated GET and decode JSON. Raises urllib/JSON errors verbatim.

    Response body is capped at ``_MAX_RESPONSE_BYTES`` to bound memory use on a
    misbehaving server. A ``ValueError`` is raised if the cap is exceeded.
    """
    req = _authed_request(url, target)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        # ``read(N)`` returns up to N bytes; reading N+1 lets us distinguish
        # "fits exactly" from "exceeds cap" without buffering the whole stream
        # twice on the happy path.
        body = resp.read(_MAX_RESPONSE_BYTES + 1)
        if len(body) > _MAX_RESPONSE_BYTES:
            raise ValueError(f"response from {url} exceeds {_MAX_RESPONSE_BYTES} byte cap")
        return json.loads(body)


# ---------------------------------------------------------------------------
# list-folders / list-folder — runtime introspection
# ---------------------------------------------------------------------------


@app.command(
    "list-folders",
    help="List model folders available to the resolved backend (cloud: /api/experiment/models, local: /models).",
)
@tracking.track_command("models")
def list_folders_cmd(
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Override the resolved routing mode."),
    ] = None,
):
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    target = resolve_target(where=where)
    url = target.url(*_models_path_parts(target))

    try:
        data = _http_get_json(url, target)
    except urllib.error.HTTPError as e:
        renderer.error(
            code="cloud_http_error" if target.is_cloud else "server_not_running",
            message=f"HTTP {e.code} from {url}",
            hint="run `comfy auth whoami` to verify auth"
            if target.is_cloud
            else "run `comfy launch` to start a local server",
            details={"status": e.code, "body": (e.read() or b"")[:1000].decode("utf-8", "replace")},
        )
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(
            code="server_not_running" if not target.is_cloud else "cloud_http_error",
            message=f"failed to fetch {url}: {e}",
            hint="check `--where` and network connectivity",
        )
        raise typer.Exit(code=1) from e

    # Cloud returns [{folders: [...], name: ...}, ...]; local returns a flat list of folder names.
    # Normalize both into [{name, subfolders}] so the envelope shape is identical.
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                rows.append({"name": entry.get("name", ""), "subfolders": list(entry.get("folders") or [])})
            elif isinstance(entry, str):
                rows.append({"name": entry, "subfolders": []})
    payload = {
        "mode": "cloud" if target.is_cloud else "local",
        "url": url,
        "count": len(rows),
        "folders": rows,
    }

    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("folder")
        tbl.add_column("subfolders", style="dim")
        for r in rows[:200]:
            tbl.add_row(r["name"], ", ".join(r["subfolders"]) if r["subfolders"] else "")
        renderer.console().print(tbl)
        rprint(f"[dim]{len(rows)} folders ({payload['mode']})[/dim]")
    renderer.emit(payload, command="models list-folders")


@app.command(
    "list-folder",
    help="List model files in a specific folder. Returns name + pathIndex per entry — no enrichment.",
)
@tracking.track_command("models")
def list_folder_cmd(
    folder: Annotated[str, typer.Argument(help="Folder name (e.g. 'loras', 'checkpoints').")],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Override the resolved routing mode."),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option("--limit", show_default=False, help="Cap output to N rows."),
    ] = None,
):
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    _reject_unsafe_path_segment(folder, kind="folder", renderer=renderer)
    target = resolve_target(where=where)
    url = target.url(*_models_path_parts(target), folder)

    try:
        data = _http_get_json(url, target)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            renderer.error(
                code="folder_not_found",
                message=f"HTTP 404 fetching {url}",
                hint=f"try `comfy models list-folders --where {'cloud' if target.is_cloud else 'local'}`",
                details={"status": 404, "folder": folder},
            )
        elif target.is_cloud:
            renderer.error(
                code="cloud_http_error",
                message=f"HTTP {e.code} fetching {url}",
                hint="check auth / connectivity",
                details={"status": e.code, "folder": folder},
            )
        else:
            renderer.error(
                code="server_not_running",
                message=f"HTTP {e.code} fetching {url}",
                hint="run `comfy launch` to start a local server",
                details={"status": e.code, "folder": folder},
            )
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(
            code="cloud_http_error" if target.is_cloud else "server_not_running",
            message=f"failed to fetch {url}: {e}",
            hint="check `--where` and network connectivity",
        )
        raise typer.Exit(code=1) from e

    files = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                files.append({"name": entry.get("name", ""), "pathIndex": entry.get("pathIndex", 0)})
            elif isinstance(entry, str):
                files.append({"name": entry, "pathIndex": 0})
    total = len(files)
    if limit is not None:
        files = files[: max(0, limit)]

    payload = {
        "mode": "cloud" if target.is_cloud else "local",
        "url": url,
        "folder": folder,
        "total": total,
        "shown": len(files),
        "files": files,
    }
    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("name")
        tbl.add_column("pathIndex", style="dim", justify="right")
        for f in files:
            tbl.add_row(f["name"], str(f["pathIndex"]))
        renderer.console().print(tbl)
        tail = f" (of {total})" if total != len(files) else ""
        rprint(f"[dim]{len(files)} files in {folder!r}{tail} ({payload['mode']})[/dim]")
    renderer.emit(payload, command="models list-folder")


# ---------------------------------------------------------------------------
# search / show — enriched catalog (cloud), filename listing (local)
# ---------------------------------------------------------------------------


def _meta_str(asset: dict[str, Any], key: str) -> str | None:
    """First-wins lookup across user_metadata then metadata. Returns None if absent."""
    for bag_key in ("user_metadata", "metadata"):
        bag = asset.get(bag_key) or {}
        val = bag.get(key)
        if isinstance(val, str) and val:
            return val
        if isinstance(val, list) and val:
            joined = ", ".join(str(x) for x in val if x)
            if joined:
                return joined
    return None


def _meta_list(asset: dict[str, Any], key: str) -> list[str]:
    """First-wins list lookup across user_metadata then metadata."""
    for bag_key in ("user_metadata", "metadata"):
        bag = asset.get(bag_key) or {}
        val = bag.get(key)
        if isinstance(val, list):
            return [str(x) for x in val if x]
        if isinstance(val, str) and val:
            return [val]
    return []


def _asset_to_row(asset: dict[str, Any]) -> dict[str, Any]:
    """Project an Asset dict into the agent-friendly model row.

    ``name`` is the on-disk filename — what goes into a LoraLoader / UNETLoader
    combo. ``display_name`` is the human-readable label kept as a separate
    field so consumers can choose which to show.
    """
    tags = [t for t in (asset.get("tags") or []) if t not in ("models", "missing")]
    type_ = tags[0] if tags else "unknown"
    return {
        "name": asset.get("name", ""),
        "display_name": asset.get("display_name") or asset.get("name", ""),
        "type": type_,
        "tags": tags + _meta_list(asset, "additional_tags"),
        "base_model": _meta_str(asset, "base_model"),
        "trained_words": _meta_list(asset, "trained_words") or None,
        "source_url": _meta_str(asset, "repo_url") or _meta_str(asset, "source_url") or _meta_str(asset, "source_arn"),
        "preview_url": asset.get("preview_url"),
        "size": asset.get("size"),
        "is_public": asset.get("is_immutable") is True,
        "id": asset.get("id"),
    }


def _cloud_search(
    target,
    *,
    text: str | None,
    type_: str | None,
    limit: int,
    include_public: bool,
) -> tuple[list[dict[str, Any]], int]:
    """Page through /api/assets, returning (rows, total).

    The ``include_tags`` parameter is comma-separated, not repeated — cloud
    rejects the repeated-key form (``include_tags=a&include_tags=b``) per its
    OpenAPI spec (``style: form, explode: false``).
    """
    tags = ["models"]
    if type_:
        tags.append(_TYPE_TO_FOLDER.get(type_, type_))
    params: dict[str, Any] = {
        "include_tags": ",".join(tags),
        "limit": min(max(limit, 1), 500),
        "include_public": str(include_public).lower(),
    }
    if text:
        params["name_contains"] = text

    qs = urllib.parse.urlencode(params)
    url = target.url("assets") + "?" + qs
    body = _http_get_json(url, target)
    assets = body.get("assets") or []
    rows = [_asset_to_row(a) for a in assets if isinstance(a, dict)]
    return rows, int(body.get("total") or len(rows))


def _local_search(
    target,
    *,
    text: str | None,
    type_: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Filename listing from /models/<folder>. No enrichment available on local."""
    if not type_:
        # No tag-based filtering on local — pick a default that's almost always
        # populated rather than scanning every folder (which would be slow).
        folder = "checkpoints"
    else:
        folder = _TYPE_TO_FOLDER.get(type_, type_)
    url = target.url(*_models_path_parts(target), folder)
    data = _http_get_json(url, target)
    items = []
    if isinstance(data, list):
        for entry in data:
            name = entry.get("name", "") if isinstance(entry, dict) else (entry if isinstance(entry, str) else "")
            if not name:
                continue
            if text and text.lower() not in name.lower():
                continue
            items.append(
                {
                    "name": name,
                    "type": folder,
                    "tags": [folder],
                    "base_model": None,
                    "trained_words": None,
                    "source_url": None,
                    "preview_url": None,
                    "size": None,
                    "is_public": False,
                    "id": None,
                }
            )
    total = len(items)
    return items[:limit], total


@app.command(
    "search",
    help="Search models. Cloud: enriched via /api/assets. Local: filename substring on /models/<folder>.",
)
@tracking.track_command("models")
def search_cmd(
    text: Annotated[
        str | None,
        typer.Option(
            "--text", "-t", show_default=False, help="Substring on the model name (case-insensitive on cloud)."
        ),
    ] = None,
    type_: Annotated[
        str | None,
        typer.Option("--type", show_default=False, help="Model type: lora, checkpoint, vae, controlnet, …"),
    ] = None,
    limit: Annotated[
        int,
        typer.Option("--limit", help="Cap results."),
    ] = 20,
    include_public: Annotated[
        bool,
        typer.Option(
            "--include-public/--mine-only",
            help="Cloud only: include public/shared assets (default true). On local this flag is ignored.",
        ),
    ] = True,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Override the resolved routing mode."),
    ] = None,
):
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    if type_ is not None:
        _reject_unsafe_path_segment(type_, kind="type", renderer=renderer)
    target = resolve_target(where=where)

    try:
        if target.is_cloud:
            rows, total = _cloud_search(target, text=text, type_=type_, limit=limit, include_public=include_public)
        else:
            rows, total = _local_search(target, text=text, type_=type_, limit=limit)
    except urllib.error.HTTPError as e:
        renderer.error(
            code="cloud_http_error" if target.is_cloud else "server_not_running",
            message=f"HTTP {e.code} during models search",
            hint="check auth (`comfy auth whoami`) or network",
            details={"status": e.code, "body": (e.read() or b"")[:1000].decode("utf-8", "replace")},
        )
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(
            code="cloud_http_error" if target.is_cloud else "server_not_running",
            message=f"models search failed: {e}",
            hint="check connectivity / auth",
        )
        raise typer.Exit(code=1) from e

    payload = {
        "mode": "cloud" if target.is_cloud else "local",
        "filters": {"text": text, "type": type_, "include_public": include_public if target.is_cloud else None},
        "total": total,
        "shown": len(rows),
        "rows": rows,
    }
    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("name")
        tbl.add_column("type", style="dim")
        tbl.add_column("base_model", style="dim")
        tbl.add_column("source", style="dim")
        for r in rows:
            tbl.add_row(
                r["name"][:60],
                r["type"] or "",
                r["base_model"] or "",
                (r["source_url"] or "")[:48],
            )
        renderer.console().print(tbl)
        tail = f" (of {total} total)" if total != len(rows) else ""
        rprint(f"[dim]{len(rows)} model(s){tail} ({payload['mode']})[/dim]")
    renderer.emit(payload, command="models search")


@app.command(
    "show",
    help="Show one model by exact name. Surfaces both metadata bags verbatim alongside the projected row.",
)
@tracking.track_command("models")
def show_cmd(
    name: Annotated[str, typer.Argument(help="Exact model filename (e.g. 'wan2.2_vae.safetensors').")],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Override the resolved routing mode."),
    ] = None,
):
    from comfy_cli.target import resolve_target

    renderer = get_renderer()
    target = resolve_target(where=where)

    if not target.is_cloud:
        # On local there's no asset catalog. We can confirm the file exists by
        # scanning the folders, but there's no enrichment to show. Surface that
        # honestly rather than returning a misleadingly empty record.
        renderer.error(
            code="models_show_local_unsupported",
            message="`models show` requires the cloud asset catalog and isn't available on local.",
            hint="for filename-only listing on local, use `comfy models list-folder <folder>`",
        )
        raise typer.Exit(code=1)

    # `name_contains` is a server-side substring filter, so for a common
    # substring the requested exact name can land on page 2+. Page through the
    # results (honoring the server's `has_more` flag) and run the exact-name
    # check client-side on every page until we find it or the server runs out.
    candidates: list[dict] = []
    match = None
    offset = 0
    page_size = 200
    max_pages = 50  # safety cap (10k results) so a misbehaving server can't loop forever
    for _ in range(max_pages):
        qs = urllib.parse.urlencode(
            {"include_tags": "models", "name_contains": name, "limit": page_size, "offset": offset}
        )
        url = target.url("assets") + "?" + qs
        try:
            body = _http_get_json(url, target)
        except urllib.error.HTTPError as e:
            renderer.error(
                code="cloud_http_error",
                message=f"HTTP {e.code} from {url}",
                hint="check auth and network",
                details={"status": e.code},
            )
            raise typer.Exit(code=1) from e
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            renderer.error(code="cloud_http_error", message=f"models show failed: {e}")
            raise typer.Exit(code=1) from e

        page = [a for a in (body.get("assets") or []) if isinstance(a, dict)]
        candidates.extend(page)
        # First exact match wins (name or display_name).
        match = next((a for a in page if a.get("name") == name or a.get("display_name") == name), None)
        if match is not None:
            break
        offset += len(page)
        if not page or not body.get("has_more"):
            break
    if match is None:
        renderer.error(
            code="model_not_found",
            message=f"no asset with exact name {name!r} ({len(candidates)} substring matches)",
            hint="try `comfy models search --text <substring>` to find candidates",
            details={
                "close_matches": [a.get("name") for a in candidates[:10] if isinstance(a, dict)],
            },
        )
        raise typer.Exit(code=1)

    payload = {
        "row": _asset_to_row(match),
        "asset": match,  # full Asset object verbatim
    }
    if renderer.is_pretty():
        row = payload["row"]
        rprint(f"[bold]{row['name']}[/bold]")
        rprint(f"  type:        {row['type']}")
        if row.get("base_model"):
            rprint(f"  base_model:  {row['base_model']}")
        if row.get("tags"):
            rprint(f"  tags:        {', '.join(row['tags'])}")
        if row.get("source_url"):
            rprint(f"  source:      {row['source_url']}")
        if row.get("preview_url"):
            rprint(f"  preview:     {row['preview_url']}")
        if row.get("size"):
            rprint(f"  size:        {row['size']:,} bytes")
        trained = row.get("trained_words")
        if trained:
            rprint(f"  trained:     {', '.join(trained)}")
    renderer.emit(payload, command="models show")
