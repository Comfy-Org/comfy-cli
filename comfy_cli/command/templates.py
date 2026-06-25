"""``comfy templates`` — workflow-template gallery introspection.

Mirrors the shape of ``comfy nodes`` but queries the curated
**workflow-template gallery** from ``Comfy-Org/workflow_templates``
(the same content that drives comfy.org/workflows). Three primitives:

    comfy templates ls   [--type T] [--category PAT] [--tag T] [--model M]
                         [--provider P] [--name SUB] [--limit N]
    comfy templates show <name>
    comfy templates refresh                            # re-fetch index.json

The gallery file ``templates/index.json`` is cached under
``~/.cache/comfy-cli/gallery/index.json`` and parsed in Python. Browsing is
flag-based (``--type``/``--category``/``--tag``/``--model``/``--provider``/
``--name``); there is no separate query grammar.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path
from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.cql import gallery
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(no_args_is_help=True, help="Browse the Comfy workflow-template gallery.")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command(
    "ls",
    help="List gallery templates. Filter by type/category/tag/model/provider/name.",
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

    try:
        cats = gallery.load_gallery(gallery_path, refresh=refresh)
    except (gallery.GalleryError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(
            code="gallery_load_failed",
            message=str(e),
            hint="check your network, or pass --gallery <path> to a local index.json",
        )
        raise typer.Exit(code=1) from e

    rows = gallery.flatten_templates(cats)
    total = len(rows)
    rows = gallery.filter_rows(
        rows,
        type_=type_,
        category=category,
        tag=tag,
        model=model,
        provider=provider,
        name_sub=name_sub,
    )
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
        cats = gallery.load_gallery(gallery_path, refresh=refresh)
    except (gallery.GalleryError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = gallery.flatten_templates(cats)
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
        data = gallery.fetch_gallery()
    except (gallery.GalleryError, urllib.error.URLError, OSError) as e:
        renderer.error(code="gallery_fetch_failed", message=str(e))
        raise typer.Exit(code=1) from e
    cache = gallery.cache_path()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(data)
    payload = {"path": str(cache), "bytes": len(data)}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] cached gallery to {cache} ({len(data)} bytes)")
    renderer.emit(payload, command="templates refresh")


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
        cats = gallery.load_gallery(gallery_path, refresh=refresh)
    except (gallery.GalleryError, urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        renderer.error(code="gallery_load_failed", message=str(e))
        raise typer.Exit(code=1) from e

    rows = gallery.flatten_templates(cats)
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
        body = gallery.fetch_template_workflow(name)
    except (gallery.GalleryError, urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
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
        "node_count": len(wf) if isinstance(wf, dict) else None,
    }
    # When there's no file destination, ride the full workflow along in the
    # envelope so a JSON consumer can actually retrieve it (pretty mode already
    # wrote it to stdout above). With --out the workflow is on disk, so we keep
    # the envelope lean and omit it.
    if out is None:
        payload["workflow"] = wf
    if renderer.is_pretty() and out:
        rprint(f"[green]✓[/green] wrote {len(body):,} bytes ({payload['node_count']} nodes) to {target_repr}")
    renderer.emit(payload, command="templates fetch")
