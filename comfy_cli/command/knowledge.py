"""``comfy knowledge`` — inspect the curated model-knowledge bundle.

    comfy knowledge status [--refresh]
    comfy knowledge resolve <alias-or-id>
    comfy knowledge pick <capability>

Backed by :mod:`comfy_cli.knowledge`. JSON mode is the contract; pretty mode
is a short courtesy view.
"""

from __future__ import annotations

import difflib
import os
from typing import Annotated, Any

import typer

from comfy_cli import knowledge, tracking
from comfy_cli.output import get_renderer, rprint
from comfy_cli.output.sanitize import sanitize_markup

app = typer.Typer(no_args_is_help=True, help="Inspect the curated model-knowledge bundle.")


def _env_context() -> dict[str, Any]:
    return {
        "env_file": os.environ.get(knowledge.ENV_FILE, "").strip() or None,
        "url": os.environ.get(knowledge.ENV_URL, "").strip() or None,
        "ttl_seconds": knowledge.ttl_seconds(),
        "cache_path": str(knowledge.cache_paths()[0]),
    }


def _require_bundle(renderer) -> knowledge.Bundle:
    bundle = knowledge.load_bundle()
    if bundle is None:
        renderer.error(
            code="knowledge_unavailable",
            message="no knowledge bundle is loaded",
            hint="set COMFY_KNOWLEDGE_FILE to a knowledge.json, or COMFY_KNOWLEDGE_URL to fetch one; see `comfy knowledge status`",
        )
        raise typer.Exit(code=1)
    return bundle


@app.command("status", help="Report which knowledge bundle is loaded, where it came from, and how big it is.")
@tracking.track_command("knowledge")
def status_cmd(
    refresh: Annotated[
        bool,
        typer.Option("--refresh", help="Re-fetch from COMFY_KNOWLEDGE_URL, ignoring the cache TTL."),
    ] = False,
):
    renderer = get_renderer()
    bundle = knowledge.load_bundle(force_fetch=refresh)
    if bundle is None:
        payload: dict[str, Any] = {"loaded": False, "reason": knowledge.last_reason(), **_env_context()}
    else:
        payload = {
            "loaded": True,
            "source": bundle.source,
            "stale": bundle.stale,
            "version": bundle.version,
            "schema_version": knowledge.SCHEMA_VERSION,
            "as_of": bundle.as_of,
            "path": bundle.path,
            **_env_context(),
            "counts": {
                "models": len(bundle.models),
                "capabilities": len(bundle.capabilities),
                "aliases": len(bundle.aliases),
                "deprecations": len(bundle.deprecations),
            },
        }
    if renderer.is_pretty():
        for key, value in payload.items():
            if isinstance(value, dict):
                value = ", ".join(f"{k}={v}" for k, v in value.items())
            rprint(f"[bold]{key}[/bold]: {sanitize_markup(value)}")
    renderer.emit(payload, command="knowledge status")


@app.command("resolve", help="Resolve a model alias or id (e.g. 'Kling 3.0', minimax-h3) to its knowledge row.")
@tracking.track_command("knowledge")
def resolve_cmd(
    query: Annotated[str, typer.Argument(help="Model alias or id; case- and whitespace-insensitive.")],
):
    renderer = get_renderer()
    bundle = _require_bundle(renderer)
    q = query.strip().lower()
    row = knowledge.resolve(bundle, query)
    if row is None:
        renderer.error(
            code="knowledge_unknown_model",
            message=f"no knowledge row matches {query!r}",
            hint="run `comfy knowledge status` to confirm a bundle is loaded; try a different alias",
            details={
                "query": query,
                "close_matches": difflib.get_close_matches(q, list(bundle.aliases), n=5, cutoff=0.6),
            },
        )
        raise typer.Exit(code=1)
    model_id = bundle.aliases[q]
    payload = {
        "query": query,
        "id": model_id,
        "model": row,
        "deprecation": bundle.deprecations.get(model_id),
        "bundle_version": bundle.version,
        "stale": bundle.stale,
    }
    if renderer.is_pretty():
        status, route = sanitize_markup(row.get("status")), sanitize_markup(row.get("route"))
        rprint(f"[bold]{sanitize_markup(model_id)}[/bold]  status={status}  route={route}")
        for line in row.get("best_for") or []:
            rprint(f"  [green]+[/green] {sanitize_markup(line)}")
        for pitfall in row.get("pitfalls") or []:
            text = pitfall.get("text") if isinstance(pitfall, dict) else pitfall
            rprint(f"  [yellow]![/yellow] {sanitize_markup(text)}")
    renderer.emit(payload, command="knowledge resolve")


@app.command("pick", help="Ranked model picks for a capability (e.g. lipsync, text-to-video).")
@tracking.track_command("knowledge")
def pick_cmd(
    capability: Annotated[str, typer.Argument(help="Capability id; see `details.known` on a miss.")],
):
    renderer = get_renderer()
    bundle = _require_bundle(renderer)
    cap = knowledge.pick(bundle, capability)
    if cap is None:
        renderer.error(
            code="knowledge_unknown_capability",
            message=f"no knowledge capability matches {capability!r}",
            hint="pick one of the listed capabilities",
            details={"capability": capability, "known": sorted(bundle.capabilities)},
        )
        raise typer.Exit(code=1)
    picks = []
    for p in cap["picks"]:
        model_id = p.get("model")
        row = bundle.models.get(model_id, {}) if isinstance(model_id, str) else {}
        picks.append(
            {
                "rank": p.get("rank"),
                "model": model_id,
                "route": p.get("route"),
                "template": p.get("template"),
                "caveat": p.get("caveat"),
                "status": row.get("status"),
                "superseded_by": row.get("superseded_by"),
            }
        )
    payload = {
        "capability": capability.strip().lower(),
        "description": cap.get("description"),
        "as_of": cap.get("as_of"),
        "picks": picks,
        "bundle_version": bundle.version,
        "stale": bundle.stale,
    }
    if renderer.is_pretty():
        from rich.table import Table

        columns = ("rank", "model", "route", "template", "status", "caveat")
        tbl = Table(show_header=True, header_style="bold")
        for col in columns:
            tbl.add_column(col)
        for p in picks:
            tbl.add_row(*(sanitize_markup("" if p[c] is None else p[c]) for c in columns))
        renderer.console().print(tbl)
    renderer.emit(payload, command="knowledge pick")
