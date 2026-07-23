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
# run-template — fetch → fill params → spend-gate → run via the run path
# ---------------------------------------------------------------------------


def _parse_param_value(raw: str) -> Any:
    """Parse a ``--param`` value as JSON; fall back to the literal string.

    Mirrors ``comfy workflow set-slot`` semantics so `--param seed=42` writes
    an int and `--param prompt="a cat"` writes a string.
    """
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _workflow_node_types(workflow: Any) -> set[str]:
    """Collect node class names from a workflow in either format.

    Frontend format: ``nodes[].type`` plus every ``definitions.subgraphs[].nodes[].type``
    (gallery templates routinely hide partner nodes inside UUID subgraphs).
    API format: ``values()[].class_type``.
    """
    types: set[str] = set()
    if not isinstance(workflow, dict):
        return types
    if isinstance(workflow.get("nodes"), list):
        node_lists = [workflow.get("nodes") or []]
        subgraphs = (workflow.get("definitions") or {}).get("subgraphs") or []
        for sg in subgraphs:
            if isinstance(sg, dict):
                node_lists.append(sg.get("nodes") or [])
        for nodes in node_lists:
            for node in nodes:
                if isinstance(node, dict) and isinstance(node.get("type"), str):
                    types.add(node["type"])
        return types
    for node in workflow.values():
        if isinstance(node, dict) and isinstance(node.get("class_type"), str):
            types.add(node["class_type"])
    return types


def _detect_paid_nodes(workflow: Any, object_info: dict) -> list[str]:
    """Sorted node class names in ``workflow`` that are partner-API (paid) nodes.

    Same signals as ``comfy_cli.command.run``'s partner detection — the
    authoritative ``api_node: true`` flag with a ``partner/`` category-prefix
    fallback — but format-agnostic so it works on the frontend-format JSON
    gallery templates ship as (run's detector only reads API format).
    """
    from comfy_cli.command.run import PARTNER_NODE_CATEGORY_PREFIXES

    out: list[str] = []
    for ct in _workflow_node_types(workflow):
        info = object_info.get(ct) or {}
        if not isinstance(info, dict):
            continue
        if info.get("api_node") is True:
            out.append(ct)
            continue
        category = info.get("category")
        if isinstance(category, str) and category.startswith(PARTNER_NODE_CATEGORY_PREFIXES):
            out.append(ct)
    return sorted(out)


def _gallery_paid_signals(row: dict[str, Any]) -> list[str]:
    """Gallery-index evidence that a template runs partner/API nodes.

    Belt-and-suspenders for the spend gate: when the local server's
    object_info is unavailable (fail-open fetch) node detection can miss, but
    the curated gallery still marks paid templates with the ``API`` tag and
    provider logos. Returns human-readable signal strings, empty = no signal.
    """
    signals: list[str] = []
    for tag in row.get("tags") or []:
        if isinstance(tag, str) and tag.lower() == "api":
            signals.append("tag:API")
    for prov in row.get("providers") or []:
        if isinstance(prov, str) and prov:
            signals.append(f"provider:{prov}")
    return signals


def _resolve_param_addresses(
    renderer,
    overrides: dict[str, Any],
    slots: list[dict],
) -> dict[str, Any]:
    """Map ``--param KEY=VALUE`` keys onto slot addresses.

    KEY may be a full slot address (``6.text``, ``62/34.text``) or a bare slot
    name (``prompt``) when exactly one slot carries that name. Ambiguous or
    unknown keys error with the candidate list so agents can self-correct.
    """
    addresses = {s.get("address") for s in slots if isinstance(s, dict)}
    by_name: dict[str, list[str]] = {}
    for s in slots:
        if not isinstance(s, dict):
            continue
        n = s.get("name")
        if isinstance(n, str) and n:
            by_name.setdefault(n.lower(), []).append(str(s.get("address")))

    resolved: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in addresses:
            resolved[key] = value
            continue
        candidates = by_name.get(key.lower(), [])
        if len(candidates) == 1:
            resolved[candidates[0]] = value
            continue
        if len(candidates) > 1:
            renderer.error(
                code="workflow_slot_invalid",
                message=f"--param key {key!r} is ambiguous: {len(candidates)} slots share that name",
                hint="use the full slot address instead: " + ", ".join(f"{a}={key}" for a in candidates[:5]),
                details={"key": key, "candidates": candidates},
            )
            raise typer.Exit(code=1)
        sample = sorted(a for a in addresses if a)[:10]
        renderer.error(
            code="workflow_slot_invalid",
            message=f"--param key {key!r} matches no slot in this template",
            hint=(
                "valid addresses include: " + ", ".join(sample)
                if sample
                else "this template exposes no tweakable slots"
            ),
            details={"key": key, "available": sample},
        )
        raise typer.Exit(code=1)
    return resolved


@tracking.track_command("templates")
def run_template_cmd(
    name: Annotated[str, typer.Argument(help="Template name (matches `comfy templates ls` rows).")],
    params: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            "-p",
            metavar="KEY=VALUE",
            show_default=False,
            help=(
                "Fill a parameterized input before running (repeatable). KEY is a slot "
                "address (`6.text`) or a unique slot name (`prompt`); VALUE parses as "
                "JSON with string fallback. List slots with `comfy templates fetch "
                "<name> -o wf.json && comfy workflow slots wf.json`."
            ),
        ),
    ] = None,
    allow_spend: Annotated[
        bool,
        typer.Option(
            "--allow-spend",
            help=(
                "Consent to running partner-API (paid) nodes that spend Comfy credits. "
                "Required for API templates when not confirming interactively."
            ),
        ),
    ] = False,
    async_: Annotated[
        bool,
        typer.Option(
            "--async",
            show_default=False,
            help="Submit and return immediately instead of waiting for completion.",
        ),
    ] = False,
    host: Annotated[
        str | None,
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
    ] = None,
    timeout: Annotated[
        int,
        typer.Option(help="Per-event timeout in seconds (same semantics as `comfy run --timeout`)."),
    ] = 120,
    verbose: Annotated[
        bool,
        typer.Option(help="Verbose execution output."),
    ] = False,
    api_key: Annotated[
        str | None,
        typer.Option(
            "--api-key",
            envvar="COMFY_API_KEY",
            help="Comfy API key for partner-API nodes (prefer the COMFY_API_KEY env var).",
        ),
    ] = None,
    gallery_path: Annotated[
        str | None,
        typer.Option("--gallery", show_default=False, help="Path to a local index.json (skips the cache + fetch)."),
    ] = None,
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch the gallery index from GitHub before resolving."),
    ] = False,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Stream NDJSON run events to stdout (same dialect as `comfy run --json`).",
        ),
    ] = False,
):
    """Fetch a gallery template, fill its parameterized inputs, and run it on local ComfyUI.

    OSS templates need their referenced models installed locally first
    (`comfy model download`); missing models surface through the normal run
    validation errors. Templates that embed partner-API nodes spend Comfy
    credits and are gated behind --allow-spend / an interactive confirmation.
    """
    import sys
    import tempfile

    from comfy_cli.command import run as run_module
    from comfy_cli.env_checker import check_comfy_server_running

    renderer = get_renderer()
    if json_output:
        renderer.force_stream()

    # -- Parse --param pairs up front so syntax errors fail before any I/O.
    overrides: dict[str, Any] = {}
    for raw in params or []:
        if "=" not in raw:
            renderer.error(
                code="workflow_slot_invalid",
                message=f"Expected `--param KEY=VALUE`, got {raw!r}",
                hint='example: --param 6.text="a cat" or --param prompt="a cat"',
            )
            raise typer.Exit(code=1)
        key, _, val = raw.partition("=")
        overrides[key.strip()] = _parse_param_value(val)

    # -- Resolve the template against the gallery index (close-matches on miss).
    try:
        cats = _load_gallery(gallery_path, refresh=refresh)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = _flatten_templates(cats)
    row = next((r for r in rows if r["name"] == name), None)
    if row is None:
        lower = name.lower()
        close = [r["name"] for r in rows if lower in r["name"].lower()][:5]
        renderer.error(
            code="template_not_found",
            message=f"no template named {name!r} in the gallery",
            hint="try `comfy templates ls --name <substring>` to search",
            details={"close_matches": close},
        )
        raise typer.Exit(code=1)

    # -- Fetch + parse the template's workflow JSON.
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
        workflow = json.loads(body)
    except json.JSONDecodeError as e:
        renderer.error(
            code="template_workflow_invalid_json",
            message=f"upstream returned non-JSON for {name!r}: {e}",
            hint="report at https://github.com/Comfy-Org/workflow_templates/issues",
        )
        raise typer.Exit(code=1) from e

    # -- Resolve host/port through the shared resolver, exactly like `comfy
    # run`'s local branch (cmdline.py). This validates the host (rejecting
    # URL-injection characters), brackets IPv6 literals, and honors
    # config.background — behavior the old hand-rolled block lacked.
    from comfy_cli.host_port import parse_host_port_arg, resolve_host_port

    if host:
        host, parsed_port = parse_host_port_arg(host)
        if not port and parsed_port is not None:
            port = parsed_port
    host, port = resolve_host_port(host, port)

    if not check_comfy_server_running(port, host, timeout=timeout):
        renderer.error(
            code="server_not_running",
            message=f"ComfyUI not running on specified address ({host}:{port})",
            hint="run: comfy launch",
            details={"host": host, "port": port},
        )
        raise typer.Exit(code=1)

    # object_info powers both slot filling and paid-node detection. Fail-open
    # ({}) keeps template runs working against bare servers — the gallery-index
    # signals below still gate paid templates in that case.
    object_info = run_module._fetch_object_info(host, port)

    # -- Fill parameterized inputs via the CQL slot engine.
    if overrides:
        if not isinstance(workflow.get("nodes"), list):
            renderer.error(
                code="workflow_slot_invalid",
                message=f"template {name!r} is not a frontend-format workflow; --param is not supported for it",
                hint="run it without --param, or fetch + edit it directly",
            )
            raise typer.Exit(code=1)
        if not object_info:
            renderer.error(
                code="object_info_unavailable",
                message="could not fetch /object_info from the server; --param needs the node catalog to fill slots",
                hint="check the ComfyUI server logs, or run without --param",
            )
            raise typer.Exit(code=1)
        from comfy_cli.cql.engine import Graph

        graph = Graph.from_object_info(object_info)
        graph._try_default_annotations()
        try:
            schema = graph.get_template_schema(name, workflow)
        except (ValueError, KeyError) as e:
            renderer.error(code="workflow_slot_invalid", message=f"Could not extract slots: {e}")
            raise typer.Exit(code=1) from e
        resolved = _resolve_param_addresses(renderer, overrides, schema.get("slots") or [])
        try:
            workflow, warnings = graph.apply_slots(workflow, resolved)
        except ValueError as e:
            renderer.error(
                code="workflow_slot_invalid",
                message=str(e),
                hint="fetch the template and run `comfy workflow slots <file>` to see valid addresses + types",
            )
            raise typer.Exit(code=1) from e
        if renderer.is_pretty():
            rprint(f"[bold green]✓[/bold green] filled {len(resolved)} parameter(s)")
            for addr in resolved:
                rprint(f"  [dim]·[/dim] {addr}")
            for w in warnings:
                rprint(f"  [yellow]warning:[/yellow] {w}")

    # -- Spend gate (BE-4113): partner-API nodes spend Comfy credits. Require
    # explicit consent before submitting anything that would burn them.
    paid_nodes = _detect_paid_nodes(workflow, object_info)
    gallery_signals = _gallery_paid_signals(row)
    if (paid_nodes or gallery_signals) and not allow_spend:
        evidence = {
            "template": name,
            "partner_nodes": paid_nodes,
            "gallery_signals": gallery_signals,
        }
        if renderer.is_pretty() and sys.stdin.isatty():
            rprint(f"[yellow]⚠ Template [bold]{name}[/bold] uses partner-API nodes that spend Comfy credits.[/yellow]")
            if paid_nodes:
                rprint(f"  [dim]nodes:[/dim] {', '.join(paid_nodes)}")
            if gallery_signals:
                rprint(f"  [dim]gallery:[/dim] {', '.join(gallery_signals)}")
            if not typer.confirm("Run anyway and spend credits?", default=False):
                renderer.error(
                    code="spend_consent_required",
                    message="declined — template not submitted, no credits spent",
                    details=evidence,
                )
                raise typer.Exit(code=1)
        else:
            renderer.error(
                code="spend_consent_required",
                message=(
                    f"template {name!r} uses partner-API (paid) nodes; "
                    "re-run with --allow-spend to consent to spending Comfy credits"
                ),
                hint="paid nodes only run with explicit consent; OSS templates run without this flag",
                details=evidence,
            )
            raise typer.Exit(code=1)

    # -- Hand off to the existing run path (UI→API conversion, partner
    # credential injection, preflight validation, execution, jobs state).
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    fd, tmp_path = tempfile.mkstemp(prefix=f"comfy_template_{safe}_", suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(workflow, f)
        run_module.execute(
            tmp_path,
            host,
            port,
            wait=not async_,
            verbose=verbose,
            timeout=timeout,
            api_key=api_key,
        )
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
