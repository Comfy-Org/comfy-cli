"""``comfy templates`` — workflow-template gallery introspection.

Mirrors the shape of ``comfy nodes`` but queries the curated
**workflow-template gallery** from ``Comfy-Org/workflow_templates``
(the same content that drives comfy.org/workflows). Three primitives:

    comfy templates ls   [--type T] [--category PAT] [--tag T] [--model M]
                         [--provider P] [--name SUB] [--limit N]
    comfy templates show <name>
    comfy templates refresh                            # re-fetch index.json

The gallery file ``templates/index.json`` is cached under
``~/.cache/comfy-cli/gallery/index.json``. The CLI side here parses the
index in Python (no WASM needed); for the full CQL grammar over templates
use the flag-based filters for browsing.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(no_args_is_help=True, help="Browse the Comfy workflow-template gallery.")

GALLERY_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/index.json"


# ---------------------------------------------------------------------------
# Gallery loading + caching
# ---------------------------------------------------------------------------


def _cache_path() -> Path:
    """Where the gallery index lives on disk. XDG-respecting."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "gallery" / "index.json"


def _fetch_gallery(url: str = GALLERY_URL, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"gallery fetch failed: HTTP {resp.status}")
        return resp.read()


def _load_gallery(
    explicit_path: str | None,
    *,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Resolve the gallery index. Precedence: explicit --gallery > cache > fetch.

    Returns the raw decoded JSON (a list of category dicts). The CLI does
    its own filtering on top.
    """
    if explicit_path:
        return json.loads(Path(explicit_path).read_bytes())

    cache = _cache_path()
    if refresh or not cache.exists():
        data = _fetch_gallery()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
        return json.loads(data)
    return json.loads(cache.read_bytes())


def _flatten_templates(categories: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Walk the nested (category → templates) shape and flatten to a list.

    Each row gets a few extras: ``category_title``, ``group_category``, and
    ``output_type`` (from the parent category's ``type`` — the per-template
    ``mediaType`` is actually the thumbnail format and is misleading).
    Providers from ``logos[].provider`` are flattened to a flat string list
    that tolerates the scalar-or-array variance in real data.
    """
    rows: list[dict[str, Any]] = []
    for cat in categories:
        if not isinstance(cat, dict):
            continue
        output_type = cat.get("type") or ""
        for t in cat.get("templates", []) or []:
            if not isinstance(t, dict):
                continue
            rows.append(
                {
                    "name": t.get("name") or "",
                    "title": (t.get("title") or "").strip(),
                    "description": t.get("description") or "",
                    "output_type": output_type,
                    "category_title": cat.get("title") or "",
                    "group_category": cat.get("category") or "",
                    "tags": list(t.get("tags") or []),
                    "models": list(t.get("models") or []),
                    "requires_custom_nodes": list(t.get("requiresCustomNodes") or []),
                    "providers": _flatten_providers(t.get("logos") or []),
                    "date": t.get("date") or "",
                    "open_source": bool(t.get("openSource", False)),
                    "usage": int(t.get("usage") or 0),
                    "media_subtype": t.get("mediaSubtype") or "",
                    "io": t.get("io") or {},
                }
            )
    return rows


def _flatten_providers(logos: list[Any]) -> list[str]:
    """``logos[].provider`` may be a string or a list-of-strings. Coalesce."""
    out: list[str] = []
    seen: set[str] = set()
    for logo in logos:
        if not isinstance(logo, dict):
            continue
        prov = logo.get("provider")
        if isinstance(prov, str):
            if prov and prov not in seen:
                seen.add(prov)
                out.append(prov)
        elif isinstance(prov, list):
            for p in prov:
                if isinstance(p, str) and p and p not in seen:
                    seen.add(p)
                    out.append(p)
    return out


# ---------------------------------------------------------------------------
# Filters — Python equivalents of nodegraph/gallery_search.go predicates
# ---------------------------------------------------------------------------


def _matches(
    row: dict[str, Any],
    *,
    type_: str | None,
    category: str | None,
    tag: str | None,
    model: str | None,
    provider: str | None,
    name_sub: str | None,
) -> bool:
    if type_ and (row.get("output_type") or "").lower() != type_.lower():
        return False
    if category and (row.get("category_title") or "").lower() != category.lower():
        return False
    if tag and not any((t or "").lower() == tag.lower() for t in row.get("tags") or []):
        return False
    if model and not any(model.lower() in (m or "").lower() for m in row.get("models") or []):
        return False
    if provider and not any(provider.lower() in (p or "").lower() for p in row.get("providers") or []):
        return False
    if name_sub and name_sub.lower() not in (row.get("name") or "").lower():
        return False
    return True


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def _ls_via_query(
    renderer,
    query: str,
    gallery_path: str | None,
    refresh: bool,
    limit: int | None,
) -> None:
    """CQL grammar queries over the template gallery are not available.
    Emit an actionable error pointing the user at the flag-based filters instead.
    """
    renderer.error(
        code="cql_query_invalid",
        message="CQL grammar queries are not available. Use flag-based filtering instead.",
        hint="comfy templates ls --type image --tag API --model Flux",
    )
    raise typer.Exit(code=1)


@app.command(
    "ls",
    help="List gallery templates. Filter by type/category/tag/model/provider/name, or pass --query for the full CQL grammar.",
)
@tracking.track_command("templates")
def ls_cmd(
    type_: Annotated[
        str | None,
        typer.Option("--type", help="Output kind: image, video, audio, 3d."),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", help="Exact category title (e.g. 'Image', 'Video')."),
    ] = None,
    tag: Annotated[
        str | None,
        typer.Option("--tag", help="Tag (case-insensitive exact match, e.g. 'API')."),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option("--model", help="Model name substring (e.g. 'Flux')."),
    ] = None,
    provider: Annotated[
        str | None,
        typer.Option("--provider", help="Provider substring (e.g. 'Kling', 'Black Forest Labs')."),
    ] = None,
    name_sub: Annotated[
        str | None,
        typer.Option("--name", help="Substring match on template name."),
    ] = None,
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            show_default=False,
            help="A CQL grammar query (e.g. 'templates type video | sort name | limit 5'). Bypasses the flag filters.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(show_default=False, help="Cap output to N rows."),
    ] = None,
    gallery_path: Annotated[
        str | None,
        typer.Option(
            "--gallery",
            show_default=False,
            help="Path to a local templates/index.json (skips the cache + fetch).",
        ),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch index.json from GitHub before listing."),
    ] = False,
):
    renderer = get_renderer()

    # CQL grammar path — routes through WASM with the gallery loaded.
    if query is not None:
        return _ls_via_query(renderer, query, gallery_path, refresh, limit)

    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(
            code="gallery_load_failed",
            message=str(e),
            hint="check your network, or pass --gallery <path> to a local index.json",
        )
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    total = len(rows)
    rows = [
        r
        for r in rows
        if _matches(
            r,
            type_=type_,
            category=category,
            tag=tag,
            model=model,
            provider=provider,
            name_sub=name_sub,
        )
    ]
    matched = len(rows)
    if limit is not None:
        rows = rows[: max(0, limit)]

    payload = {
        "total_in_gallery": total,
        "matched": matched,
        "shown": len(rows),
        "filters": {
            "type": type_,
            "category": category,
            "tag": tag,
            "model": model,
            "provider": provider,
            "name": name_sub,
        },
        "rows": [
            {
                "name": r["name"],
                "title": r["title"],
                "output_type": r["output_type"],
                "category_title": r["category_title"],
                "tags": r["tags"],
                "models": r["models"],
                "providers": r["providers"],
                "description": r["description"][:120],
            }
            for r in rows
        ],
    }

    if renderer.is_pretty():
        from rich.table import Table

        if not rows:
            rprint("[dim]0 templates matched.[/dim]")
        else:
            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("type", style="dim")
            tbl.add_column("title")
            tbl.add_column("tags", style="dim")
            for r in rows:
                tbl.add_row(
                    r["name"],
                    r["output_type"],
                    r["title"] or "(untitled)",
                    ", ".join(r["tags"]),
                )
            renderer.console().print(tbl)
            tail = f" (of {matched} matched, {total} in gallery)" if (matched != len(rows) or matched != total) else ""
            rprint(f"[dim]{len(rows)} template(s){tail}[/dim]")
    renderer.emit(payload, command="templates ls")


@app.command(
    "show",
    help="Show full details for a single template by name.",
)
@tracking.track_command("templates")
def show_cmd(
    name: Annotated[str, typer.Argument(help="Template name (e.g. 'image_flux2').")],
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch from GitHub before showing."),
    ] = False,
):
    renderer = get_renderer()
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    match = next((r for r in rows if r["name"] == name), None)
    if match is None:
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r}",
            hint="try `comfy templates ls --name <substring>` to search",
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        rprint(f"[bold]{match['name']}[/bold]")
        if match["title"]:
            rprint(f"  [dim]{match['title']}[/dim]")
        rprint(f"  type:        {match['output_type']}")
        rprint(f"  category:    {match['category_title']} ({match['group_category']})")
        if match["tags"]:
            rprint(f"  tags:        {', '.join(match['tags'])}")
        if match["models"]:
            rprint(f"  models:      {', '.join(match['models'])}")
        if match["providers"]:
            rprint(f"  providers:   {', '.join(match['providers'])}")
        if match["date"]:
            rprint(f"  date:        {match['date']}")
        if match["description"]:
            rprint("")
            rprint(match["description"])
    renderer.emit({"template": match}, command="templates show")


@app.command("refresh", help="Re-download templates/index.json into the local cache.")
@tracking.track_command("templates")
def refresh_cmd():
    renderer = get_renderer()
    try:
        data = _fetch_gallery()
    except (urllib.error.URLError, OSError) as e:
        renderer.error(code="gallery_fetch_failed", message=str(e))
        raise typer.Exit(code=1) from e
    cache = _cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    payload = {"path": str(cache), "bytes": len(data)}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] cached gallery to {cache} ({len(data)} bytes)")
    renderer.emit(payload, command="templates refresh")


# Where the per-template workflow JSONs live on GitHub. The gallery index lists
# each template by ``name``; the corresponding workflow is at
# ``Comfy-Org/workflow_templates/templates/<name>.json``.
_TEMPLATE_WORKFLOW_URL = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/{name}.json"


def _fetch_template_workflow(name: str, *, timeout: float = 15.0) -> bytes:
    """Pull a single template's workflow JSON from the canonical GitHub raw URL."""
    url = _TEMPLATE_WORKFLOW_URL.format(name=urllib.parse.quote(name, safe=""))
    req = urllib.request.Request(url, headers={"User-Agent": "comfy-cli"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if resp.status != 200:
            raise RuntimeError(f"template workflow fetch failed: HTTP {resp.status}")
        return resp.read()


@app.command(
    "fetch",
    help=(
        "Fetch a template's workflow JSON from the curated gallery. "
        "Verifies the name against the gallery index first, then pulls "
        "templates/<name>.json from Comfy-Org/workflow_templates."
    ),
)
@tracking.track_command("templates")
def fetch_cmd(
    name: Annotated[str, typer.Argument(help="Template name (matches `comfy templates ls` rows).")],
    out: Annotated[
        str | None,
        typer.Option("--out", "-o", show_default=False, help="Write to this file instead of stdout."),
    ] = None,
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json (skips the cache + fetch)."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch the gallery index from GitHub before resolving."),
    ] = False,
):
    renderer = get_renderer()

    # Resolve against the gallery index first so we surface "no such template"
    # with the same close_matches affordance the rest of the CLI uses, instead
    # of letting the user hit a raw GitHub 404.
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    match = next((r for r in rows if r["name"] == name), None)
    if match is None:
        # Build a small list of close matches so the agent can self-correct.
        lower = name.lower()
        close = [r["name"] for r in rows if lower in r["name"].lower()][:5]
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r} in the gallery",
            hint="try `comfy templates ls --name <substring>` to search",
            details={"close_matches": close},
        )
        raise typer.Exit(code=1)

    try:
        body = _fetch_template_workflow(name)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        status = getattr(e, "code", None)
        renderer.error(
            code="template_fetch_failed",
            message=f"failed to fetch workflow for {name!r}: {e}",
            hint=(
                "the gallery index references a template whose workflow JSON "
                "is missing upstream — report at "
                "https://github.com/Comfy-Org/workflow_templates/issues"
                if status == 404
                else "check network connectivity"
            ),
            details={"status": status} if status else None,
        )
        raise typer.Exit(code=1) from e

    # Parse so we (a) validate it's well-formed JSON and (b) can report the
    # node count in the envelope without re-reading.
    try:
        wf = json.loads(body)
    except json.JSONDecodeError as e:
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"upstream returned non-JSON for {name!r}: {e}",
            hint="report at https://github.com/Comfy-Org/workflow_templates/issues",
        )
        raise typer.Exit(code=1) from e

    if out:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(body)
        target_repr = str(out_path)
    else:
        # In JSON mode, the renderer's emit() is the only thing on stdout — the
        # raw workflow goes into the envelope under data.workflow. In pretty
        # mode we print it to stdout so the user can pipe it.
        if renderer.is_pretty():
            import sys

            sys.stdout.write(body.decode("utf-8"))
            sys.stdout.write("\n")
        target_repr = "stdout" if out is None else str(Path(out).expanduser())

    payload = {
        "name": name,
        "title": match["title"],
        "output_type": match["output_type"],
        "out": target_repr,
        "bytes": len(body),
        # `nodes` count is the only field the agent needs to confirm the
        # workflow loaded; the full JSON ride-along bloats every envelope.
        "node_count": len(wf) if isinstance(wf, dict) else None,
    }
    if renderer.is_pretty() and out:
        rprint(f"[green]✓[/green] wrote {len(body):,} bytes ({payload['node_count']} nodes) to {target_repr}")
    renderer.emit(payload, command="templates fetch")


# ---------------------------------------------------------------------------
# templates check — per-template runnable/missing/api-required/unknown verdict
# ---------------------------------------------------------------------------


def _template_workflow_cache_path(name: str) -> Path:
    """Where a fetched per-template workflow JSON is cached. Same base-dir logic
    as :func:`_cache_path`, one file per template under ``gallery/templates``."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return Path(base) / "comfy-cli" / "gallery" / "templates" / f"{name}.json"


def _iter_workflow_nodes(wf: dict[str, Any]):
    """Yield every node dict in a UI-format workflow: top-level ``nodes`` first,
    then the nodes inside each subgraph *definition*.

    The subgraph walk is mandatory — modern templates (e.g. ``image_z_image_turbo``)
    carry their model references only inside subgraph definitions, so a top-level
    walk alone would report them as having zero model requirements.
    """
    if not isinstance(wf, dict):
        return
    for node in wf.get("nodes") or []:
        if isinstance(node, dict):
            yield node
    for sg in (wf.get("definitions") or {}).get("subgraphs") or []:
        if isinstance(sg, dict):
            for node in sg.get("nodes") or []:
                if isinstance(node, dict):
                    yield node


def _collect_model_requirements(wf: dict[str, Any]) -> list[dict[str, str]]:
    """Gather every ``node["properties"]["models"]`` entry across top-level and
    subgraph nodes. Each entry is normalized to ``{name, directory, url}`` and
    deduped by ``(directory, name)`` preserving first-seen order.
    """
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for node in _iter_workflow_nodes(wf):
        props = node.get("properties")
        if not isinstance(props, dict):
            continue
        models = props.get("models")
        if not isinstance(models, list):
            continue
        for m in models:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "")
            directory = str(m.get("directory") or "")
            if not name and not directory:
                continue
            key = (directory, name)
            if key in seen:
                continue
            seen.add(key)
            out.append({"name": name, "directory": directory, "url": str(m.get("url") or "")})
    return out


def _collect_node_class_types(wf: dict[str, Any]) -> list[str]:
    """Distinct ``node["type"]`` values across top-level + subgraph nodes.

    For a regular node this is the ComfyUI class name (``CheckpointLoaderSimple``);
    for a subgraph *instance* it's the definition UUID (never loader-ish, never in
    object_info), which is why the loader/api heuristics simply skip those.
    """
    seen: set[str] = set()
    out: list[str] = []
    for node in _iter_workflow_nodes(wf):
        t = node.get("type")
        if isinstance(t, str) and t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _basename(path: str) -> str:
    """Last path segment of a folder-relative listing entry. Model folder
    listings return paths that may include subdirectories (``sdxl/model.safetensors``)
    and always use ``/`` on the wire; also tolerate a stray backslash defensively.
    """
    return path.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]


def _list_local_folder(target, folder: str) -> list[str] | None:
    """List file paths in ``/models/<folder>`` on the local server.

    Returns the folder-relative names, or ``None`` if the folder 404s (a
    custom-node folder like ``SEEDVR2`` that ComfyUI doesn't register). Connection
    errors (server not running) and other HTTP errors propagate to the caller.
    """
    # Reuse the exact target/URL plumbing `comfy models list-folder` uses.
    from comfy_cli.command.models.search import _http_get_json, _models_path_parts

    url = target.url(*_models_path_parts(target), folder)
    try:
        data = _http_get_json(url, target)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    names: list[str] = []
    if isinstance(data, list):
        for entry in data:
            if isinstance(entry, dict):
                n = entry.get("name", "")
            elif isinstance(entry, str):
                n = entry
            else:
                n = ""
            if n:
                names.append(n)
    return names


def _template_is_api_by_index(name: str, tags: list[str]) -> bool:
    """Index-only API heuristic: an ``api_*`` template name, or an ``API`` tag."""
    if name.startswith("api_"):
        return True
    return any(str(t).strip().lower() == "api" for t in tags)


def _compute_verdict(*, api_dependent: bool, missing: list, required_count: int, node_types: list[str]) -> str:
    """Verdict precedence: api-required > missing-models > runnable > unknown."""
    if api_dependent:
        return "api-required"
    if missing:
        return "missing-models"
    if required_count > 0:
        # Every referenced model resolved on disk.
        return "runnable"
    # Zero model references: runnable only if nothing looks like it *should* load
    # a model; otherwise we genuinely can't tell (loader with no declared models).
    loaderish = any("loader" in (t or "").lower() for t in node_types)
    return "unknown" if loaderish else "runnable"


@app.command(
    "check",
    help=(
        "Report whether a gallery template is runnable on THIS install: which of "
        "its models are present vs missing locally, whether it needs partner-API "
        "access, and any custom nodes it declares. Resolves the name against the "
        "gallery, fetches the workflow, and intersects its model refs with the "
        "local server's model folders."
    ),
)
@tracking.track_command("templates")
def check_cmd(
    name: Annotated[str, typer.Argument(help="Template name (matches `comfy templates ls` rows).")],
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json (skips the cache + fetch)."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch the gallery index AND the template workflow before checking."),
    ] = False,
):
    from comfy_cli.target import resolve_target

    renderer = get_renderer()

    # 1. Resolve the name against the gallery index (same affordance as `fetch`).
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    match = next((r for r in rows if r["name"] == name), None)
    if match is None:
        lower = name.lower()
        close = [r["name"] for r in rows if lower in r["name"].lower()][:5]
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r} in the gallery",
            hint="try `comfy templates ls --name <substring>` to search",
            details={"close_matches": close},
        )
        raise typer.Exit(code=1)

    # 2. Fetch (or read from cache) the per-template workflow JSON.
    cache_path = _template_workflow_cache_path(name)
    body: bytes | None = None
    if not refresh and cache_path.exists():
        try:
            body = cache_path.read_bytes()
        except OSError:
            body = None
    if body is None:
        try:
            body = _fetch_template_workflow(name)
        except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
            status = getattr(e, "code", None)
            renderer.error(
                code="template_fetch_failed",
                message=f"failed to fetch workflow for {name!r}: {e}",
                hint=(
                    "the gallery index references a template whose workflow JSON "
                    "is missing upstream — report at "
                    "https://github.com/Comfy-Org/workflow_templates/issues"
                    if status == 404
                    else "check network connectivity"
                ),
                details={"status": status} if status else None,
            )
            raise typer.Exit(code=1) from e
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(body)
        except OSError:
            pass  # a non-writable cache dir must not fail the check

    try:
        wf = json.loads(body)
    except json.JSONDecodeError as e:
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"template workflow for {name!r} is not valid JSON: {e}",
            hint="re-run with --refresh to re-fetch, or report upstream",
        )
        raise typer.Exit(code=1) from e
    if not isinstance(wf, dict):
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"template workflow for {name!r} is not a JSON object",
            hint="re-run with --refresh to re-fetch, or report upstream",
        )
        raise typer.Exit(code=1)

    # 3. Model requirements (top-level + subgraph walk) and node class types.
    required = _collect_model_requirements(wf)
    node_types = _collect_node_class_types(wf)

    # 4. API dependence. Tier (a) is the index heuristic; tier (b) upgrades it with
    #    object_info when a local server is reachable (best-effort — a down server
    #    just leaves the index answer in place).
    api_dependent = _template_is_api_by_index(name, match.get("tags") or [])
    api_source = "index"
    api_nodes: list[str] = []
    try:
        from comfy_cli.cql.engine import Graph

        graph = Graph.load(mode="local")
        found = [cls for cls in node_types if (m := graph.node(cls)) is not None and m.is_api_node]
        api_nodes = found
        api_source = "object_info"
        api_dependent = api_dependent or bool(api_nodes)
    except Exception:
        # Server down / object_info unavailable → index heuristic alone decides.
        # Discard any partial object_info result so `source` and `api_nodes` agree.
        api_nodes = []
        api_source = "index"

    # 5. Installed intersection: list each distinct folder once on the local server
    #    and match required files by basename.
    warnings: list[str] = []
    present: list[str] = []
    missing: list[dict[str, str]] = []
    distinct_dirs = list(dict.fromkeys(req["directory"] for req in required))
    if required:
        target = resolve_target(where="local")
        listings: dict[str, list[str] | None] = {}
        try:
            for directory in distinct_dirs:
                if not directory or ".." in directory or "/" in directory or "\\" in directory:
                    # Not addressable as a `/models/<folder>` segment — treat as absent.
                    listings[directory] = None
                    warnings.append(
                        f"model directory {directory!r} isn't a valid model folder — its files are reported missing"
                    )
                    continue
                folder_files = _list_local_folder(target, directory)
                listings[directory] = folder_files
                if folder_files is None:
                    warnings.append(
                        f"model folder {directory!r} not found on the local server "
                        f"(custom-node folder?) — its files are reported missing"
                    )
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError) as e:
            renderer.error(
                code="server_not_running",
                message=f"local ComfyUI server is unreachable, cannot check installed models: {e}",
                hint="run `comfy launch` to start a local server",
            )
            raise typer.Exit(code=1) from e

        for req in required:
            listing = listings.get(req["directory"])
            if listing and any(_basename(entry) == req["name"] for entry in listing):
                present.append(req["name"])
            else:
                missing.append(dict(req))

    # 6. Custom nodes are report-only in v1 (surfaced verbatim, not verified).
    custom_nodes_required = list(match.get("requires_custom_nodes") or [])

    # 7. Verdict.
    verdict = _compute_verdict(
        api_dependent=api_dependent,
        missing=missing,
        required_count=len(required),
        node_types=node_types,
    )

    # 8. Emit.
    payload = {
        "name": name,
        "title": match["title"],
        "verdict": verdict,
        "models": {"required": len(required), "present": present, "missing": missing},
        "api": {"dependent": api_dependent, "source": api_source, "api_nodes": api_nodes},
        "custom_nodes_required": custom_nodes_required,
        "warnings": warnings,
    }

    if renderer.is_pretty():
        mark = "[bold green]✓[/bold green]" if verdict == "runnable" else "[bold red]✗[/bold red]"
        rprint(f"{mark} {name} — [bold]{verdict}[/bold]")
        if api_dependent:
            extra = f": {', '.join(api_nodes)}" if api_nodes else ""
            rprint(f"  [yellow]needs partner-API access[/yellow] (via {api_source}){extra}")
        if custom_nodes_required:
            rprint(f"  custom nodes: {', '.join(custom_nodes_required)} [dim](not verified)[/dim]")
        if missing:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("missing model")
            tbl.add_column("directory", style="dim")
            tbl.add_column("download url")
            for m in missing:
                tbl.add_row(m["name"], m["directory"], m["url"] or "[dim](none)[/dim]")
            renderer.console().print(tbl)
        elif required:
            rprint(f"  [dim]{len(present)}/{len(required)} model(s) present[/dim]")
        for w in warnings:
            rprint(f"  [yellow]⚠[/yellow] {w}")
    renderer.emit(payload, command="templates check")
