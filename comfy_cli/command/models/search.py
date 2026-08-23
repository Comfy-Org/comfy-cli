"""``comfy models`` — live model discovery against local or cloud.

Four subcommands, all routed by ``--where`` (cloud auto-detect by default):

    comfy models list-folders           # GET /api/experiment/models  | /models
    comfy models list-folder <folder>   # GET /api/experiment/models/<folder> | /models/<folder>
    comfy models search [--text] [--type] [--limit]  # cloud: /api/assets; local: /models/<folder>, all folders w/o --type
    comfy models show <name>            # exact-match name across the catalog

The search surface mirrors the asset→model extraction used by Comfy-Org's
cloud tooling: prefer ``user_metadata`` over ``metadata`` for any given key,
treat ``tags`` as the canonical type/role signal, and surface the
densely-populated fields (``source_url``, ``preview_url``, ``size``) as
first-class result columns. Sparse fields (``base_model``, ``trained_words``)
ride along when present.

Local-mode caveats:
  * ``/models/<folder>`` returns ``[{name, pathIndex}, ...]`` — filenames only,
    no enrichment. ``search`` on local degrades to a filename match: without
    ``--type`` it walks *every* folder reported by ``/models`` (so a
    diffusion_models/vae/lora file is findable by name), and ``--type`` scopes
    the walk to that single folder. ``--text`` is token-AND and
    separator-insensitive there (see ``_name_matches``).
  * The cloud asset catalog (``/api/assets``) has no local equivalent —
    local search is intentionally simpler.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.parse
from typing import Annotated, Any, NoReturn

import typer

from comfy_cli import knowledge, tracking
from comfy_cli.http import ResponseTooLarge
from comfy_cli.output import get_renderer, rprint
from comfy_cli.output.sanitize import sanitize_markup

app = typer.Typer(no_args_is_help=True, help="Discover models — folders, files, and the cloud asset catalog.")

# Cap response reads from cloud/local. The largest legitimate response we see
# in practice is `/api/object_info` (~9 MB on cloud), but the model endpoints
# are far smaller. 64 MiB is generous headroom; anything beyond this is either
# a misconfigured backend or hostile and would only serve to OOM the CLI.
_MAX_RESPONSE_BYTES = 64 * 1024 * 1024

# Folder names are interpolated into URL paths, so they must stay a *single*
# path segment. Only genuine traversal shapes are refused; everything else is
# percent-encoded before it reaches the URL (see `_is_walkable_folder_name`).


def _is_walkable_folder_name(value: str) -> bool:
    """True if ``value`` is usable as a single URL path segment.

    Applies to both server-advertised folder names (the ``models search
    --where local`` walk) and user-supplied ones (``models list-folder <folder>``,
    ``models search --type``). Real installs configure folders like ``my loras``,
    ``SDXL (base)``, or non-ASCII names; holding those to a strict-ASCII regex
    silently skipped them on the walk and rejected them outright on user input,
    leaving every model inside them unreachable even though ComfyUI serves the
    folder fine. Only genuine traversal shapes are refused here; every call site
    percent-encodes the segment with ``quote(..., safe="")`` before it reaches
    the URL, so spaces, ``?``/``#``, and control characters can't alter the
    request.
    """
    if not value or "/" in value or "\\" in value:
        return False
    # Only the *exact* segments `.` and `..` are rewritten by a URL resolver
    # (RFC 3986 remove_dot_segments); `..` merely *inside* a name (`model..v2`)
    # is an ordinary run of characters and must stay usable. `quote` leaves `.`
    # unencoded, so a bare `.` would otherwise reach the server as `/models/.`
    # and normalize back to the `/models` collection.
    if value in (".", ".."):
        return False
    try:
        # argv can carry undecodable bytes as lone surrogates (PEP 383
        # `surrogateescape`), and a JSON `"\udcff"` escape does the same for
        # server-advertised names. `quote(..., safe="")` raises
        # `UnicodeEncodeError` on those — a `ValueError` that no call site's
        # handler catches, so it would surface as an uncaught traceback.
        # Rejecting costs no reachable capability: `quote(..., errors=
        # "surrogateescape")` would encode the raw bytes instead, but a backend's
        # folder names are config-defined `str` keys (`folder_names_and_paths`,
        # `extra_model_paths.yaml`) that are always valid UTF-8, so such a segment
        # can never match one — it would only turn this clear error into a 404.
        value.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def _reject_unsafe_path_segment(value: str, *, kind: str, renderer) -> None:
    """Exit with an `invalid_argument` error if ``value`` isn't safe as a path segment."""
    if not _is_walkable_folder_name(value):
        renderer.error(
            code="invalid_argument",
            message=f"{kind} {value!r} is not usable as a single path segment",
            hint=(
                f"a {kind} name must be non-empty, must not be `.` or `..`, must not contain "
                "`/` or `\\`, and must be valid UTF-8"
            ),
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


def _http_get_json(url: str, target, timeout: float = 30.0) -> Any:
    """Issue an authenticated GET and decode JSON. Raises urllib/JSON errors verbatim.

    Response body is capped at ``_MAX_RESPONSE_BYTES`` to bound memory use on a
    misbehaving server; exceeding it raises ``ResponseTooLarge``, which every
    caller routes to an envelope error alongside the urllib/JSON families.
    """
    from comfy_cli.http import request_json

    _, body = request_json(url, target, timeout=timeout, max_bytes=_MAX_RESPONSE_BYTES)
    if body is None:
        # Callers route JSONDecodeError to an envelope error; an empty or
        # unparseable body must surface the same way, not crash on body.get().
        raise json.JSONDecodeError("empty or unparseable response body", "", 0)
    return body


def _emit_http_error(e: urllib.error.HTTPError, *, renderer, target, message: str, hint: str) -> NoReturn:
    """Emit a renderer error for an ``HTTPError`` and exit with code 1.

    Shared by the ``list-folders`` and ``search`` handlers, whose HTTPError
    branches are identical: the error ``code`` is cloud/local-routed, and the
    response body is truncated to 1 KiB and decoded (lossily) into ``details``
    for debugging. Only ``message`` and ``hint`` differ per caller. The
    single-hint ``show`` handler and the 404-special-casing ``list-folder``
    handler deliberately do NOT use this — their shapes differ.
    """
    renderer.error(
        code="cloud_http_error" if target.is_cloud else "server_not_running",
        message=message,
        hint=hint,
        details={"status": e.code, "body": (e.read() or b"")[:1000].decode("utf-8", "replace")},
    )
    raise typer.Exit(code=1) from e


def _resolve_and_stamp(renderer, where: str | None):
    """Resolve the routing Target for a ``models`` verb and stamp it on the renderer.

    Every verb here calls this at the point it decides local-vs-cloud, so the
    error envelopes raised downstream carry ``where`` instead of ``null``.
    Errors raised *before* this (an unsafe path segment) keep ``where: null``,
    which is correct — nothing had routed yet. Explicit
    ``emit(..., where=...)`` arguments still take precedence over the stamp.
    """
    from comfy_cli.target import resolve_target

    target = resolve_target(where=where)
    renderer.where = target.kind
    return target


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
    renderer = get_renderer()
    target = _resolve_and_stamp(renderer, where)
    url = target.url(*_models_path_parts(target))

    try:
        data = _http_get_json(url, target)
    except urllib.error.HTTPError as e:
        _emit_http_error(
            e,
            renderer=renderer,
            target=target,
            message=f"HTTP {e.code} from {url}",
            hint="run `comfy cloud whoami` to verify auth"
            if target.is_cloud
            else "run `comfy launch` to start a local server",
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ResponseTooLarge) as e:
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
            # ``add_row`` parses markup in ``str`` cells; these are server names.
            tbl.add_row(
                sanitize_markup(r["name"]),
                sanitize_markup(", ".join(r["subfolders"])) if r["subfolders"] else "",
            )
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
    renderer = get_renderer()
    _reject_unsafe_path_segment(folder, kind="folder", renderer=renderer)
    target = _resolve_and_stamp(renderer, where)
    # Percent-encoded for the same reason `_local_folder_matches` does it: the
    # relaxed validation above admits spaces, `?`/`#`, and non-ASCII, none of
    # which may be allowed to alter the request. Error payloads below carry the
    # decoded `folder` so the user sees what they typed.
    url = target.url(*_models_path_parts(target), urllib.parse.quote(folder, safe=""))

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
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ResponseTooLarge) as e:
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
            tbl.add_row(sanitize_markup(f["name"]), sanitize_markup(f["pathIndex"]))
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
    if not isinstance(body, dict):
        # Callers route JSONDecodeError to an envelope error; a non-object
        # top-level body (list/scalar) must surface the same way, not crash
        # on body.get().
        raise json.JSONDecodeError(f"unexpected response shape (not an object) from {url}", "", 0)
    assets = body.get("assets") or []
    rows = [_asset_to_row(a) for a in assets if isinstance(a, dict)]
    return rows, int(body.get("total") or len(rows))


def _local_folder_names(target) -> list[str]:
    """Fetch the backend's model-folder list, normalized to plain folder names.

    Same endpoint (and same tolerant normalizer) as ``list_folders_cmd``: local
    serves a flat list of folder-name strings, but the dict-entry shape is
    accepted too so both server generations work.
    """
    data = _http_get_json(target.url(*_models_path_parts(target)), target)
    names: list[str] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                name = entry.get("name", "")
            elif isinstance(entry, str):
                name = entry
            else:
                continue
            if isinstance(name, str) and name:
                names.append(name)
    return names


# Characters that separate the "words" of a model filename. Model files are
# named with any of them, inconsistently, in the same catalog:
# `sd_xl_base_1.0.safetensors`, `flux1-dev.safetensors`, `ltx-video-2b-v0.9`.
_NAME_SEPARATORS = re.compile(r"[-_. ]+")


def _search_tokens(text: str | None) -> list[tuple[str, str]] | None:
    """Split ``--text`` into ``(token, separator-squashed token)`` pairs.

    Returns ``None`` when there is no filter at all — ``--text`` omitted, or
    the falsy ``--text ""``. That is the list-everything case, and it matches
    what the cloud branch does with the same input (``if text:`` is false, so
    no ``name_contains`` is sent).

    Returns an *empty list* when ``--text`` was given but held nothing
    matchable: whitespace only, or tokens that are purely separators. Those
    are a filter nothing satisfies, not the absence of a filter — a token that
    squashes to ``""`` is a substring of every name, so keeping it would turn
    ``--text "."`` into a wildcard, and testing it raw would make it one
    anyway (every name with an extension contains ``.``). Dropping such tokens
    also stops a stray separator in a real query (``--text "sd - xl"``) from
    ANDing in a literal ``-`` that ``sd_xl_base_1.0.safetensors`` fails.
    """
    if not text:
        return None
    tokens = []
    for t in text.lower().split():
        t_squashed = _NAME_SEPARATORS.sub("", t)
        if t_squashed:
            tokens.append((t, t_squashed))
    return tokens


def _name_matches(name: str, tokens: list[tuple[str, str]] | None) -> bool:
    """Token-AND, separator-insensitive filename match.

    Every whitespace-separated token of the query must be present, in any
    order, either in the raw lowercased filename or in the filename with
    ``- _ . space`` runs squashed out. The squashed pass is what lets
    ``--text "sdxl base"`` find ``sd_xl_base_1.0.safetensors``: real model
    filenames put separators wherever the uploader felt like it, so a
    whole-string contiguous substring test (what this used to be) could not
    match a multi-word query at all.

    The token is squashed too, so the match is symmetric — ``sd-xl``,
    ``sd_xl`` and ``sdxl`` all find the same file.

    ``tokens`` carries the two no-token cases apart, per ``_search_tokens``:
    ``None`` is "no filter" (everything matches), ``[]`` is "a filter nothing
    can satisfy" (nothing matches).
    """
    if tokens is None:
        return True
    if not tokens:
        return False
    name_l = name.lower()
    name_squashed = _NAME_SEPARATORS.sub("", name_l)
    return all(t in name_l or t_squashed in name_squashed for t, t_squashed in tokens)


def _local_folder_matches(target, folder: str, *, text: str | None) -> list[dict[str, Any]]:
    """Rows for one ``/models/<folder>`` listing, client-side filtered by ``text``.

    ``folder`` is percent-encoded into the path so folder names with spaces or
    non-ASCII characters resolve correctly; the emitted rows carry the decoded
    name so ``type``/``tags`` stay human-readable.
    """
    tokens = _search_tokens(text)
    segment = urllib.parse.quote(folder, safe="")
    data = _http_get_json(target.url(*_models_path_parts(target), segment), target)
    rows: list[dict[str, Any]] = []
    if isinstance(data, list):
        for entry in data:
            name = entry.get("name", "") if isinstance(entry, dict) else (entry if isinstance(entry, str) else "")
            # A dict entry's `name` is server-controlled and may not be a string;
            # `name.lower()` below (and the cross-folder sort in `_local_search`)
            # would blow up on a non-str, so drop those the way the folder-name
            # normalizer does.
            if not isinstance(name, str) or not name:
                continue
            if not _name_matches(name, tokens):
                continue
            rows.append(
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
    return rows


def _local_search(
    target,
    *,
    text: str | None,
    type_: str | None,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Filename listing from /models/<folder>. No enrichment available on local.

    With ``--type`` this is a single-folder fetch. Without it we walk every
    folder ``/models`` reports: local has no tag-based filtering, so a
    single-folder default (historically ``checkpoints``) made every model
    outside it invisible to text search. Walking is cheap — filename-only
    listings against a localhost server, ~20 small GETs.
    """
    # `--limit -1` would otherwise become a negative slice that silently drops
    # the last N rows while `total` still reports the full count. Clamp like
    # `_cloud_search` and `list_folder_cmd` already do.
    limit = max(0, limit)

    if type_:
        scoped = _local_folder_matches(target, _TYPE_TO_FOLDER.get(type_, type_), text=text)
        return scoped[:limit], len(scoped)

    all_matches: list[dict[str, Any]] = []
    seen: set[str] = set()
    for folder in _local_folder_names(target):
        # Folder names are server-provided, so they can't be trusted into a URL
        # path. Skip traversal shapes silently — it isn't user input to reject.
        if not _is_walkable_folder_name(folder) or folder in seen:
            continue
        seen.add(folder)
        try:
            all_matches.extend(_local_folder_matches(target, folder, text=text))
        except (OSError, ValueError):
            # One misbehaving folder must not sink the whole walk: a folder the
            # listing advertises but doesn't serve (HTTPError 404), a hung or
            # refused fetch (URLError/OSError — HTTPError and URLError are both
            # OSError subclasses), a proxy serving HTML instead of JSON
            # (json.JSONDecodeError, a ValueError), or a body over the 64 MiB
            # cap (ValueError).
            continue
    all_matches.sort(key=lambda r: (r["type"], r["name"]))
    return all_matches[:limit], len(all_matches)


@app.command(
    "search",
    help=(
        "Search models. Cloud: enriched via /api/assets. "
        "Local: filename match across every /models folder (--type scopes it to one)."
    ),
)
@tracking.track_command("models")
def search_cmd(
    text: Annotated[
        str | None,
        typer.Option(
            "--text",
            "-t",
            show_default=False,
            help=(
                "Match the model name (case-insensitive). Cloud: substring. "
                "Local: every word must appear, in any order, ignoring - _ . separators."
            ),
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
    renderer = get_renderer()
    if type_ is not None:
        _reject_unsafe_path_segment(type_, kind="type", renderer=renderer)
    target = _resolve_and_stamp(renderer, where)

    try:
        if target.is_cloud:
            rows, total = _cloud_search(target, text=text, type_=type_, limit=limit, include_public=include_public)
        else:
            rows, total = _local_search(target, text=text, type_=type_, limit=limit)
    except urllib.error.HTTPError as e:
        _emit_http_error(
            e,
            renderer=renderer,
            target=target,
            message=f"HTTP {e.code} during models search",
            hint="check auth (`comfy cloud whoami`) or network",
        )
    except (urllib.error.URLError, OSError, json.JSONDecodeError, ResponseTooLarge) as e:
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
            # Truncate first, then sanitize: escaping last keeps the markup
            # escapes balanced, and a sequence cut in half by the slice is
            # cleaned up rather than left dangling.
            tbl.add_row(
                sanitize_markup(r["name"][:60]),
                sanitize_markup(r["type"] or ""),
                sanitize_markup(r["base_model"] or ""),
                sanitize_markup((r["source_url"] or "")[:48]),
            )
        renderer.console().print(tbl)
        tail = f" (of {total} total)" if total != len(rows) else ""
        rprint(f"[dim]{len(rows)} model(s){tail} ({payload['mode']})[/dim]")
    knowledge.attach(
        payload,
        command="models search",
        queries=[text] if text else [],
        thin=(total == 0 and bool(text)),
        qualified=bool(text),
    )
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
    renderer = get_renderer()
    target = _resolve_and_stamp(renderer, where)

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
            if not isinstance(body, dict):
                raise json.JSONDecodeError(f"unexpected response shape (not an object) from {url}", "", 0)
        except urllib.error.HTTPError as e:
            renderer.error(
                code="cloud_http_error",
                message=f"HTTP {e.code} from {url}",
                hint="check auth and network",
                details={"status": e.code},
            )
            raise typer.Exit(code=1) from e
        except (urllib.error.URLError, OSError, json.JSONDecodeError, ResponseTooLarge) as e:
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
        # Every field below is catalog text the server chose, interpolated into
        # a markup-parsing sink — sanitize each one (see comfy_cli.output.sanitize).
        rprint(f"[bold]{sanitize_markup(row['name'])}[/bold]")
        rprint(f"  type:        {sanitize_markup(row['type'])}")
        if row.get("base_model"):
            rprint(f"  base_model:  {sanitize_markup(row['base_model'])}")
        if row.get("tags"):
            rprint(f"  tags:        {sanitize_markup(', '.join(row['tags']))}")
        if row.get("source_url"):
            rprint(f"  source:      {sanitize_markup(row['source_url'])}")
        if row.get("preview_url"):
            rprint(f"  preview:     {sanitize_markup(row['preview_url'])}")
        if row.get("size"):
            rprint(f"  size:        {row['size']:,} bytes")
        trained = row.get("trained_words")
        if trained:
            rprint(f"  trained:     {sanitize_markup(', '.join(trained))}")
    renderer.emit(payload, command="models show")
