"""``comfy knowledge`` — inspect the curated model-knowledge bundle.

    comfy knowledge status [--refresh]
    comfy knowledge resolve <alias-or-id>
    comfy knowledge pick [capability] [--check-local]

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
        # Userinfo, query and fragment can carry a token; the path still shows which bundle is configured.
        "url": tracking._scrub_value(knowledge.bundle_url()),
        "ttl_seconds": knowledge.ttl_seconds(),
        "cache_path": str(knowledge.cache_paths()[0]),
    }


def _items(value: Any) -> list:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) else None


_UNAVAILABLE_HINTS = {
    knowledge.REASON_SIGNED_OUT: "sign in with `comfy cloud login` so the bundle can be fetched, or set COMFY_KNOWLEDGE_FILE to a knowledge.json",
    knowledge.REASON_FETCH_FAILED: "check COMFY_KNOWLEDGE_URL, or set COMFY_KNOWLEDGE_FILE to a knowledge.json",
    knowledge.REASON_ENV_FILE: "check the COMFY_KNOWLEDGE_FILE path",
}


def _require_bundle(renderer) -> knowledge.Bundle:
    bundle = knowledge.load_bundle()
    if bundle is None:
        reason = knowledge.last_reason()
        hint = _UNAVAILABLE_HINTS.get(reason, "set COMFY_KNOWLEDGE_FILE to a knowledge.json")
        renderer.error(
            code="knowledge_unavailable",
            message=f"no knowledge bundle is loaded: {reason}",
            hint=f"{hint}; see `comfy knowledge status`",
        )
        raise typer.Exit(code=1)
    return bundle


@app.command("status", help="Report which knowledge bundle is loaded, where it came from, and how big it is.")
@tracking.track_command("knowledge")
def status_cmd(
    refresh: Annotated[
        bool,
        typer.Option(
            "--refresh",
            help="Reload the bundle, ignoring the cache TTL (re-reads COMFY_KNOWLEDGE_FILE when set, else re-fetches).",
        ),
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
            "compiled_at": bundle.compiled_at,
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
    logged = [query.strip()[: knowledge.MAX_QUERY_CHARS]]
    model_id = knowledge.resolve_id(bundle, query)
    if model_id is None:
        knowledge.log_query("knowledge resolve", logged, hit_ids=[], zero_hit=True, bundle=bundle)
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
    knowledge.log_query("knowledge resolve", logged, hit_ids=[model_id], zero_hit=False, bundle=bundle)
    row = bundle.models[model_id]
    payload = {
        "query": query,
        "id": model_id,
        "model": row,
        "deprecation": bundle.deprecations.get(model_id),
        "bundle_version": bundle.version,
        "compiled_at": bundle.compiled_at,
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


def _emit_capabilities(renderer, bundle: knowledge.Bundle, *, query: str | None = None) -> None:
    """The capability list, either as the bare listing or as ``query``'s miss.

    A miss is an answer, not a failure: it reports the same ``zero_hit`` and
    ``nudge`` an enrichment block uses, so one concept keeps one shape.
    """
    caps = [
        {"id": cid, "description": _text((bundle.capabilities.get(cid) or {}).get("description"))}
        for cid in sorted(bundle.capabilities)
    ]
    payload: dict[str, Any] = {
        "capabilities": caps,
        "zero_hit": query is not None,
        "bundle_version": bundle.version,
        "compiled_at": bundle.compiled_at,
        "stale": bundle.stale,
    }
    if query is not None:
        payload["query"] = query
        payload["nudge"] = f"no curated knowledge for {query!r}; query one of the listed capability ids"
    if renderer.is_pretty():
        if query is not None:
            rprint(f"[yellow]no curated knowledge for[/yellow] {sanitize_markup(query)}")
        for cap in caps:
            tail = f"  {sanitize_markup(cap['description'])}" if cap["description"] else ""
            rprint(f"[bold]{sanitize_markup(cap['id'])}[/bold]{tail}")
    renderer.emit(payload, command="knowledge pick")


def _check_picks_locally(picks: list[dict[str, Any]]) -> str:
    """Flag each ``oss`` pick whose template the local ComfyUI cannot run.

    Returns ``ok`` when the check ran. A pick that could not be checked carries
    its own ``local_check`` instead of a flag. A failure that would repeat for
    every pick (server down, no gallery) returns its error code instead and
    marks nothing.
    """
    from comfy_cli.command import templates as templates_cmd

    marks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    listings: dict[str, list[str] | None] = {}
    rows: list[dict[str, Any]] | None = None
    index_fresh = False
    try:
        for pick in picks:
            template = pick.get("template")
            if pick.get("route") != "oss" or not template:
                continue
            if rows is None:
                rows = templates_cmd._gallery_rows(None, refresh=False)
                # A failed re-fetch serves the stale cache, which cannot show that
                # a template is absent upstream.
                index_fresh = not templates_cmd._cache_is_stale(templates_cmd._cache_path())
            try:
                _row, wf = templates_cmd._template_workflow(template, rows, refresh=False)
            except templates_cmd.TemplateCheckError as e:
                if e.code == "template_not_found" and index_fresh:
                    flag = {"available_locally": False, "unavailable_reason": knowledge.UNAVAILABLE_TEMPLATE_NOT_FOUND}
                    marks.append((pick, flag))
                else:
                    marks.append((pick, {"local_check": e.code}))
                continue
            required = templates_cmd._collect_model_requirements(wf)
            _present, missing, _warnings = templates_cmd._match_local_models(required, listings)
            verdict = templates_cmd._compute_verdict(
                api_dependent=False,
                missing=missing,
                required_count=len(required),
                node_types=templates_cmd._collect_node_class_types(wf),
            )
            if verdict == "unknown":
                marks.append((pick, {"local_check": "unknown"}))
            elif missing:
                if all(listings.get(m["directory"]) is None for m in missing):
                    reason = knowledge.UNAVAILABLE_MODEL_FOLDER_NOT_FOUND
                else:
                    reason = knowledge.UNAVAILABLE_MISSING_MODELS
                flag = {"available_locally": False, "unavailable_reason": reason, "missing_models": len(missing)}
                marks.append((pick, flag))
    except templates_cmd.TemplateCheckError as e:
        return e.code
    for pick, mark in marks:
        pick.update(mark)
    return "ok"


@app.command("pick", help="Ranked model picks for a capability; omit it to list every capability.")
@tracking.track_command("knowledge")
def pick_cmd(
    capability: Annotated[
        str | None,
        typer.Argument(help="Capability id (e.g. lipsync, text-to-video). Omit to list every capability."),
    ] = None,
    check_local: Annotated[
        bool,
        typer.Option(
            "--check-local",
            help=(
                "Check each oss pick's template against the local ComfyUI's model folders and flag "
                "the picks it cannot run. Fetches uncached template workflows and calls the local server."
            ),
        ),
    ] = False,
):
    renderer = get_renderer()
    bundle = _require_bundle(renderer)
    if capability is None or not capability.strip():
        _emit_capabilities(renderer, bundle)
        return
    logged = [capability.strip()[: knowledge.MAX_QUERY_CHARS]]
    cap = knowledge.pick(bundle, capability)
    if cap is None:
        knowledge.log_query("knowledge pick", logged, hit_ids=[], zero_hit=True, bundle=bundle)
        _emit_capabilities(renderer, bundle, query=logged[0])
        return
    capability_id = _text(cap.get("id")) or capability.strip().lower()
    knowledge.log_query("knowledge pick", logged, hit_ids=[f"cap:{capability_id}"], zero_hit=False, bundle=bundle)
    picks = []
    for p in cap["picks"]:
        entry = knowledge.pick_entry(bundle, p)
        if "fits" in p:
            entry["fits"] = p["fits"]
        picks.append(entry)
    payload = {
        "capability": capability_id,
        "zero_hit": False,
        "description": _text(cap.get("description")),
        "as_of": _text(cap.get("as_of")),
        "picks": picks,
        "bundle_version": bundle.version,
        "compiled_at": bundle.compiled_at,
        "stale": bundle.stale,
    }
    if check_local:
        payload["local_check"] = _check_picks_locally(picks)
    if renderer.is_pretty():
        from rich.table import Table

        columns = ("rank", "model", "route", "template", "status", "caveat", "best_for")
        if check_local:
            columns += ("unavailable_reason", "local_check")
        tbl = Table(show_header=True, header_style="bold")
        for col in columns:
            tbl.add_column(col)
        for p in picks:
            cells = {**p, "best_for": ", ".join(p.get("best_for") or [])}
            tbl.add_row(*(sanitize_markup("" if cells.get(c) is None else cells[c]) for c in columns))
        renderer.console().print(tbl)
        if check_local and payload["local_check"] != "ok":
            rprint(f"[yellow]local check did not finish:[/yellow] {payload['local_check']}")
    renderer.emit(payload, command="knowledge pick")
