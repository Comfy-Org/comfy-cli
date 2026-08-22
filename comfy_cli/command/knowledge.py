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
    url = os.environ.get(knowledge.ENV_URL, "").strip()
    return {
        "env_file": os.environ.get(knowledge.ENV_FILE, "").strip() or None,
        # Userinfo, query and fragment can carry a token; the path still shows which bundle is configured.
        "url": tracking._scrub_value(url) if url else None,
        "ttl_seconds": knowledge.ttl_seconds(),
        "cache_path": str(knowledge.cache_paths()[0]),
    }


def _items(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


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
    query: Annotated[
        str,
        typer.Argument(
            help="Model alias or id. Case, spacing, punctuation, and leading zeros are ignored ('Hailuo 3' == 'hailuo-03')."
        ),
    ],
):
    renderer = get_renderer()
    bundle = _require_bundle(renderer)
    model_id = knowledge.resolve_id(bundle, query)
    if model_id is None:
        q = query.strip().lower()
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
    row = bundle.models[model_id]
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
        for line in _items(row.get("best_for")):
            rprint(f"  [green]+[/green] {sanitize_markup(line)}")
        for pitfall in _items(row.get("pitfalls")):
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
        model_id = model_id if isinstance(model_id, str) else None
        row = bundle.models.get(model_id) or {}
        picks.append(
            {
                "rank": knowledge.pick_rank(p),
                "model": model_id,
                "route": _text(p.get("route")),
                "template": _text(p.get("template")),
                "caveat": _text(p.get("caveat")),
                "status": _text(row.get("status")),
                "superseded_by": _text(row.get("superseded_by")),
            }
        )
    payload = {
        "capability": _text(cap.get("id")) or capability.strip().lower(),
        "description": _text(cap.get("description")),
        "as_of": _text(cap.get("as_of")),
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
