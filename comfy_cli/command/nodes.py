"""``comfy nodes`` — node-class introspection.

Agent-facing wrappers over the same object_info source CQL consumes, but
with flag-based interfaces an LLM can pattern-match without learning a
grammar. Three primitives:

    comfy nodes ls   [--produces T] [--accepts T] [--category PAT] [--limit N]
    comfy nodes show <NodeClass>
    comfy nodes search <text>

All three resolve the graph in this order:
    1. ``--input <path>`` to an object_info dump (offline mode)
    2. a live ComfyUI server, addressed exactly as ``comfy run`` addresses it:
       ``--host``/``--port`` > ``COMFY_LOCAL_URL`` > the persisted
       ``config.background`` server > 127.0.0.1:8188

Backed by the pure-Python CQL engine (``comfy_cli.cql.engine.Graph``).
"""

from __future__ import annotations

import difflib
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.cql.engine import Graph, LoadError
from comfy_cli.output import get_renderer, rprint
from comfy_cli.output.sanitize import sanitize_markup

app = typer.Typer(no_args_is_help=True, help="Introspect ComfyUI node classes (inputs, outputs, categories).")


# ---------------------------------------------------------------------------
# graph resolution — shared across ls/show/search
# ---------------------------------------------------------------------------


def _resolved_where(where: str | None) -> str:
    """Apply the full precedence chain: per-command flag > env > config > default."""
    from comfy_cli import where as where_module

    # Mirror comfy_cli.target.resolve_target()'s defensive fallback: a corrupt
    # config must not take the whole `comfy nodes *` surface down with a
    # traceback before the structured renderer ever runs (resolve_default reads
    # the persisted where_default defensively for the same reason).
    try:
        decision = where_module.resolve_default(flag=where)
    except ValueError:
        # An invalid persisted where_default shouldn't be fatal; fall back to
        # the flag (if valid) or auto-detect with the bad config value dropped.
        decision = where_module.resolve(flag=where, config_value=None)
    return decision.target.value  # "local" | "cloud"


def _get_graph(
    input_path: str | None,
    host: str | None,
    port: int | None,
    where: str | None = None,
    on_stale=None,
) -> Graph:
    """Load the Graph for ``comfy nodes`` commands.

    Routing follows the standard precedence: explicit ``--where`` > env
    (``COMFY_WHERE``) > config (``where_default``) > local default. The
    ``--input <path>`` flag short-circuits everything (offline mode). For a
    live local target the address is resolved with ``resolve_host_port`` —
    ``--host``/``--port`` > ``COMFY_LOCAL_URL`` > ``config.background`` >
    127.0.0.1:8188 — so discovery reads the same server ``comfy run`` submits to.

    ``on_stale``, if provided, is forwarded to ``resilient_load_object_info``
    and fired when a stale-cache fallback occurs (see loader for signature).
    """
    mode = _resolved_where(where)
    # Every `comfy nodes` verb routes through here, so this is the one place
    # the target is decided — stamp it on the renderer so the envelopes emitted
    # downstream (the `cql_no_graph` error below included) carry `where`
    # instead of `null`. `--input` reads a dump instead of talking to the
    # backend, but `mode` still says which catalog the dump is interpreted as,
    # which is what `where` reports (see the field's schema description).
    get_renderer().where = mode
    # Resolve the local server the same way `comfy run` / `comfy jobs` do —
    # flag > COMFY_LOCAL_URL > config.background > 127.0.0.1:8188. `resolve_target`
    # deliberately skips the `config.background` step (other callers must not
    # honor it), so callers that do resolve it upstream. Without this, an agent
    # discovering nodes here would read a different server's object_info than the
    # one `comfy run` submits to whenever ComfyUI was launched in the background
    # on a non-default port (BE-6299).
    if input_path is None and mode == "local":
        from comfy_cli.host_port import report_usage_error, resolve_host_port

        # A rejected `--host`/`--port` raises `typer.BadParameter`, which click
        # turns into a stderr usage panel + exit 2 with nothing on stdout.
        # Emit the terminating envelope first so JSON/NDJSON consumers get a
        # parseable final line; the exception still escapes, so exit stays 2.
        with report_usage_error(get_renderer()):
            host, port = resolve_host_port(host, port)
    try:
        if input_path is not None:
            # Explicit offline dump — let Graph.load read + annotate it.
            return Graph.load(
                mode=mode,
                input_path=input_path,
                host=host,
                port=port,
            )
        # Live fetch goes through the resilient loader: auto-cache on success,
        # refresh-and-retry once, then fall back to the last cached dump (with
        # a stderr warning) when the server/session is briefly unreachable.
        from comfy_cli.cql.loader import resilient_load_object_info

        raw = resilient_load_object_info(
            mode=mode,
            host=host,
            port=port,
            on_stale=on_stale,
        )
        graph = Graph.from_object_info(raw)
        graph._try_default_annotations()
        return graph
    except LoadError as e:
        renderer = get_renderer()
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <path>, or start the server with `comfy launch`"),
            details=e.details,
        )
        raise typer.Exit(code=1) from e


def _category_matches(category: str | None, pat: str) -> bool:
    """Glob-style match on category: ``loaders%`` matches ``loaders/anything``."""
    if not isinstance(category, str):
        return False
    # Support both SQL-style `%` and standard glob `*` so agents can use either.
    pat_norm = pat.replace("%", "*")
    import fnmatch

    return fnmatch.fnmatchcase(category, pat_norm)


# ---------------------------------------------------------------------------
# ls
# ---------------------------------------------------------------------------


@app.command(
    "ls",
    help="List node classes. Filter via --produces/--accepts/--category/--pack/--label or boolean flags.",
)
@tracking.track_command("nodes")
def ls_cmd(
    produces: Annotated[
        str | None,
        typer.Option("--produces", help="Only nodes whose outputs include this type (e.g. MODEL, IMAGE)."),
    ] = None,
    accepts: Annotated[
        str | None,
        typer.Option("--accepts", help="Only nodes with at least one input of this type."),
    ] = None,
    category: Annotated[
        str | None,
        typer.Option("--category", help="Glob match on category path (e.g. 'loaders*', 'sampling/*')."),
    ] = None,
    pack: Annotated[
        str | None,
        typer.Option("--pack", help="Filter by custom-node pack name (e.g. 'core', 'comfyui-impact-pack')."),
    ] = None,
    label: Annotated[
        str | None,
        typer.Option("--label", help="Filter by behavioral label (e.g. 'WritesToDisk', 'NetworkAccess')."),
    ] = None,
    cloud_disabled: Annotated[
        bool,
        typer.Option("--cloud-disabled/--cloud-enabled", show_default=False, help="Filter by cloud availability."),
    ] = False,
    api_only: Annotated[
        bool,
        typer.Option("--api-only", show_default=False, help="Only partner API nodes."),
    ] = False,
    output_only: Annotated[
        bool,
        typer.Option("--output-only", show_default=False, help="Only terminal output nodes."),
    ] = False,
    exclude_deprecated: Annotated[
        bool,
        typer.Option("--exclude-deprecated", show_default=False, help="Exclude deprecated nodes."),
    ] = False,
    limit: Annotated[
        int | None,
        typer.Option(show_default=False, help="Cap output to N rows."),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a local object_info JSON (offline mode)."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            show_default=False, help="ComfyUI host (defaults to COMFY_LOCAL_URL, the background server, or 127.0.0.1)."
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            show_default=False, help="ComfyUI port (defaults to COMFY_LOCAL_URL, the background server, or 8188)."
        ),
    ] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )

    produces_upper = produces.upper() if produces else None
    accepts_upper = accepts.upper() if accepts else None

    nodes = []
    for m in graph.all_nodes():
        if produces_upper and not m.has_output(produces_upper):
            continue
        if accepts_upper and not m.has_input(accepts_upper):
            continue
        if category and not _category_matches(m.category, category):
            continue
        if pack and m.pack.lower() != pack.lower():
            continue
        if label and label not in m.labels:
            continue
        if cloud_disabled and not m.cloud_disabled:
            continue
        if api_only and not m.is_api_node:
            continue
        if output_only and not m.is_output_node:
            continue
        if exclude_deprecated and m.deprecated:
            continue
        nodes.append(m)

    nodes.sort(key=lambda m: m.id)

    # Note: cloud servers pre-filter disabled nodes from object_info, so
    # --cloud-disabled will always return 0 results against a cloud target.
    cloud_note = None
    if cloud_disabled and not nodes:
        mode = _resolved_where(where)
        if mode == "cloud":
            cloud_note = "Cloud server pre-filters disabled nodes; query a local server to see what would be blocked."

    total_matched = len(nodes)
    if limit is not None:
        nodes = nodes[: max(0, limit)]

    payload = {
        "filter": {
            "produces": produces,
            "accepts": accepts,
            "category": category,
            "pack": pack,
            "label": label,
            "cloud_disabled": cloud_disabled if cloud_disabled else None,
            "api_only": api_only if api_only else None,
            "output_only": output_only if output_only else None,
            "exclude_deprecated": exclude_deprecated if exclude_deprecated else None,
        },
        "total": total_matched,
        "count": len(nodes),
        "rows": [
            {
                "name": m.id,
                "category": m.category,
                "display_name": m.display_name,
                "output_types": m.output_types(),
                "output_node": m.is_output_node,
                # Paid partner-API node vs free open-weights node. JSON only —
                # the pretty table is deliberately left unchanged.
                "is_api_node": m.is_api_node,
            }
            for m in nodes
        ],
    }

    if cloud_note:
        payload["cloud_note"] = cloud_note

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        if not nodes:
            rprint("[dim]0 nodes matched.[/dim]")
            if cloud_note:
                rprint(f"[yellow]{cloud_note}[/yellow]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("outputs")
            for m in nodes:
                # object_info text into a markup sink; the em-dash fallback is
                # ours, so only the joined server values are escaped.
                outs = sanitize_markup(", ".join(m.output_types())) or "[dim]—[/dim]"
                tbl.add_row(sanitize_markup(m.id), sanitize_markup(m.category or ""), outs)
            renderer.console().print(tbl)
            rprint(f"[dim]{len(nodes)} node(s)[/dim]")
    renderer.emit(payload, command="nodes ls")


# ---------------------------------------------------------------------------
# show
# ---------------------------------------------------------------------------


@app.command("show", help="Show the full schema for one node class: inputs, outputs, defaults, constraints.")
@tracking.track_command("nodes")
def show_cmd(
    name: Annotated[str, typer.Argument(help="Node class name (case-sensitive), e.g. 'KSampler'.")],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a local object_info JSON (offline mode)."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            show_default=False, help="ComfyUI host (defaults to COMFY_LOCAL_URL, the background server, or 127.0.0.1)."
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            show_default=False, help="ComfyUI port (defaults to COMFY_LOCAL_URL, the background server, or 8188)."
        ),
    ] = None,
    select: Annotated[
        str | None,
        typer.Option(
            "--select",
            show_default=False,
            help="Project the payload: dot path (inputs.0.name), wildcard (inputs.#.name), comma multi-select.",
        ),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )

    m = graph.node(name)
    if m is None:
        # A subgraph instance's `type` is its definition UUID, and ls-nodes
        # prints that verbatim — so callers ask show for a "class" the catalog
        # can never have. `workflow add-node` already names this shape
        # (UnknownNodeType subgraph_id); show was left behind with the generic
        # miss, and prod agents retried it verbatim. Say what the UUID is and
        # which surface CAN inspect it. difflib against a UUID is pure noise.
        from comfy_cli.workflow_ops import _UUID_RE

        if _UUID_RE.match(name.strip()):
            renderer.error(
                code="node_not_found",
                message=(
                    f"{name!r} is a subgraph type id, not a node class — `ls-nodes` prints a subgraph instance's "
                    "definition uuid as its type, and the catalog has no schema for it."
                ),
                hint=(
                    "inspect the instance's editable inputs with `comfy workflow slots <file>` / `ls-nodes`; "
                    "interior nodes are addressed `<instance>/<interior>` and written with `set-widget`."
                ),
                details={"requested": name, "subgraph_id": True},
            )
            raise typer.Exit(code=1)
        # Surface near-matches so the agent can self-correct from the error.
        all_names = [n.id for n in graph.all_nodes()]
        close = difflib.get_close_matches(name, all_names, n=5, cutoff=0.6)
        renderer.error(
            code="node_not_found",
            message=f"Node class {name!r} not found in the loaded environment.",
            hint=(
                f"did you mean: {', '.join(close)}?"
                if close
                else "run `comfy nodes ls` or `comfy nodes search <text>` to find available classes."
            ),
            details={"requested": name, "close_matches": close},
        )
        raise typer.Exit(code=1)

    payload = graph.morphism_to_dict(m)

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if select is not None:
        from comfy_cli.selector import emit_selected

        return emit_selected(renderer, payload, select, command="nodes show")

    if renderer.is_pretty():
        from rich.table import Table

        rprint(
            f"[bold]{sanitize_markup(payload['name'])}[/bold]"
            + (
                f"  [dim]({sanitize_markup(payload['display_name'])})[/dim]"
                if payload["display_name"] and payload["display_name"] != payload["name"]
                else ""
            )
        )
        if payload["category"]:
            rprint(f"[dim]category[/dim]  {sanitize_markup(payload['category'])}")
        if payload["description"]:
            rprint(f"[dim]{sanitize_markup(payload['description'])}[/dim]")
        outs = sanitize_markup(", ".join(payload["output_types"])) or "(none)"
        rprint(f"[dim]outputs[/dim]   {outs}")
        rprint("")
        if payload["inputs"]:
            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("input")
            tbl.add_column("type")
            tbl.add_column("section", style="dim")
            tbl.add_column("default", style="dim")
            for i in payload["inputs"]:
                opts = i.get("options") or {}
                default = opts.get("default")
                tbl.add_row(
                    sanitize_markup(i.get("name") or ""),
                    sanitize_markup(i.get("type") or ""),
                    sanitize_markup(i.get("section") or ""),
                    "" if default is None else sanitize_markup(default),
                )
            renderer.console().print(tbl)
    renderer.emit(payload, command="nodes show")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command(
    "search",
    help=(
        "Search node classes by name, display name, category, or description "
        "(case-insensitive, word-order-independent; falls back to close-name matches)."
    ),
)
@tracking.track_command("nodes")
def search_cmd(
    query: Annotated[
        str,
        typer.Argument(
            help=("Text to search for (case-insensitive, word-order-independent; falls back to close-name matches).")
        ),
    ],
    limit: Annotated[int, typer.Option(help="Cap output to N rows.")] = 20,
    expand_top: Annotated[
        int,
        typer.Option(
            "--expand-top",
            metavar="N",
            help=(
                "Also attach the full `nodes show` schema (inputs, defaults, enum choices, outputs) "
                "for the top-N hits under `expanded`, so no follow-up `show` calls are needed. "
                "0 (the default) leaves the output unchanged."
            ),
        ),
    ] = 0,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a local object_info JSON (offline mode)."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(
            show_default=False, help="ComfyUI host (defaults to COMFY_LOCAL_URL, the background server, or 127.0.0.1)."
        ),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(
            show_default=False, help="ComfyUI port (defaults to COMFY_LOCAL_URL, the background server, or 8188)."
        ),
    ] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )

    # Token-AND matching: every whitespace-separated token must be present, in
    # any order. `q_joined` additionally lets a spaced query hit a CamelCase
    # class name ('ksampler advanced' -> 'ksampleradvanced' == KSamplerAdvanced).
    # A query with no tokens (empty or all-whitespace) has nothing to match on.
    # Don't fall back to the raw string: `" "` would then be a substring of every
    # blob and match the entire catalog, and `all(...)` over an empty token list
    # is vacuously true, which does the same. Both mean "no match".
    q = query.lower()
    tokens = q.split()
    q_joined = "".join(tokens)
    scored: list[tuple[int, Any]] = []
    for m in graph.all_nodes() if tokens else ():
        name_l = m.id.lower()
        display_l = m.display_name.lower()
        desc_l = m.description.lower()
        cat_l = (m.category or "").lower()
        blob = " ".join((name_l, display_l, desc_l, cat_l))
        # Tiered: exact name > name prefix > all tokens in name > display > category > anywhere.
        if name_l == q or name_l == q_joined:
            score = 0
        elif name_l.startswith(q) or name_l.startswith(q_joined):
            score = 1
        elif all(t in name_l for t in tokens):
            score = 2
        elif all(t in display_l for t in tokens):
            score = 3
        elif all(t in cat_l for t in tokens):
            score = 4
        elif all(t in blob for t in tokens):
            score = 5
        else:
            continue
        scored.append((score, m))

    scored.sort(key=lambda x: (x[0], x[1].id))
    close_match = False
    if scored:
        total_matched = len(scored)
        matched = [m for _, m in scored[: max(0, limit)]]
    else:
        # Zero hits: fall back to the closest node names, so a typo
        # ('KSampeler') still points the caller at 'KSampler'. Ids are bucketed
        # by their lowered form (difflib needs unique candidates) but every node
        # in a colliding bucket is surfaced — a pack may register both
        # 'LoadImage' and 'loadimage', and dropping one hides a real suggestion.
        by_lower: dict[str, list[Any]] = {}
        for m in graph.all_nodes():
            by_lower.setdefault(m.id.lower(), []).append(m)
        close = difflib.get_close_matches(q_joined, list(by_lower), n=max(1, limit), cutoff=0.6) if q_joined else []
        candidates = [m for name_l in close for m in by_lower[name_l]]
        # Count before truncating, like every other path here (`ls`, `upstream`,
        # and the scored branch above) — otherwise `--limit 0` reports total 0
        # and silently erases the fact that a close match exists.
        total_matched = len(candidates)
        matched = candidates[: max(0, limit)]
        close_match = bool(candidates)

    payload = {
        "query": query,
        "total": total_matched,
        "count": len(matched),
        # Top-level too, not just per-row: a caller that gates on `count == 0` to
        # mean "no such node" would otherwise have to inspect every row to notice
        # the search actually found nothing and is guessing. Always present, so
        # `data["close_match"]` is a stable check rather than a key-exists probe.
        "close_match": close_match,
        "rows": [
            {
                "name": m.id,
                "category": m.category,
                "display_name": m.display_name,
                "description": m.description,
                "output_types": m.output_types(),
                # Paid partner-API node vs free open-weights node — two nodes can
                # share a display name and differ only here (MiniMax H3). JSON
                # only; the pretty table is deliberately left unchanged.
                "is_api_node": m.is_api_node,
                **({"close_match": True} if close_match else {}),
            }
            for m in matched
        ],
    }

    # --expand-top N: kill the search → show × N loop (measured on prod agent
    # traces: the follow-up `show` args are overwhelmingly a verbatim copy of the
    # hit name). The top-N returned rows are re-resolved through the SAME catalog
    # path `nodes show` uses (graph.node → morphism_to_dict), so `expanded[i]` is
    # exactly the show payload plus a `class_type` key to join back on the row.
    # A per-hit miss degrades to a per-hit error entry — it never fails the
    # search, since the rows themselves are still perfectly good results.
    if expand_top > 0:
        expanded: list[dict[str, Any]] = []
        for m in matched[: max(0, expand_top)]:
            resolved = graph.node(m.id)
            if resolved is None:
                expanded.append(
                    {
                        "class_type": m.id,
                        "error": {
                            "code": "expand_miss",
                            "message": f"search matched {m.id!r} but the catalog could not resolve its schema",
                        },
                    }
                )
                continue
            expanded.append({"class_type": m.id, **graph.morphism_to_dict(resolved)})
        payload["expanded"] = expanded

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        # The query is echoed back into a markup-interpreting sink, so escape it
        # for the same reason the table cells below do.
        query_safe = sanitize_markup(repr(query))
        # Key the empty-state on the pre-slice count, not on `matched`: with
        # `--limit 0` the slice is empty even though the search found hits, and
        # printing "no nodes match" there contradicts the JSON's `total`.
        if not total_matched:
            rprint(f"[dim]No nodes match {query_safe}.[/dim]")
        elif not matched and close_match:
            # Guesses, not matches — say so, or --limit 0 would report the
            # fallback's finds as real hits and undo the distinction above.
            rprint(
                f"[dim]No nodes match {query_safe}; {total_matched} close name match(es) "
                f"found but --limit {limit} returned none.[/dim]"
            )
        elif not matched:
            rprint(f"[dim]{total_matched} node(s) match {query_safe}; --limit {limit} returned none.[/dim]")
        else:
            if close_match:
                rprint(f"[dim]No nodes match {query_safe} — showing close name matches.[/dim]")
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("description", style="dim")
            for m in matched:
                # Truncate first, then escape, so the escapes stay balanced.
                desc = sanitize_markup(m.description[:60])
                tbl.add_row(sanitize_markup(m.id), sanitize_markup(m.category or ""), desc)
            renderer.console().print(tbl)
            rprint(f"[dim]{len(matched)} node(s)[/dim]")
    renderer.emit(payload, command="nodes search")


# ---------------------------------------------------------------------------
# graph traversal: upstream / downstream / path
# ---------------------------------------------------------------------------


def _morphism_row(m) -> dict[str, Any]:
    """Project a Morphism into our agent-friendly row shape."""
    return {
        "name": m.id,
        "category": m.category,
        "display_name": m.display_name,
        "output_types": m.output_types(),
    }


@app.command("upstream", help="List nodes whose outputs can feed into <name>'s link inputs.")
@tracking.track_command("nodes")
def upstream_cmd(
    name: Annotated[str, typer.Argument(help="Node class name, e.g. 'KSampler'.")],
    limit: Annotated[int | None, typer.Option(show_default=False, help="Cap output to N rows.")] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )
    nodes = graph.upstream(name)

    total_upstream = len(nodes)
    if limit is not None:
        nodes = nodes[: max(0, limit)]
    rows = [_morphism_row(m) for m in nodes]
    payload = {"name": name, "total": total_upstream, "count": len(rows), "rows": rows}

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        if not rows:
            rprint(f"[dim]No upstream nodes for {name!r}.[/dim]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("outputs")
            for r in rows:
                outs = sanitize_markup(", ".join(r["output_types"])) or "[dim]—[/dim]"
                tbl.add_row(sanitize_markup(r["name"] or ""), sanitize_markup(r["category"] or ""), outs)
            renderer.console().print(tbl)
            tail = f" of {total_upstream}" if total_upstream != len(rows) else ""
            rprint(f"[dim]{len(rows)} upstream node(s){tail}[/dim]")
    renderer.emit(payload, command="nodes upstream")


@app.command("downstream", help="List nodes that accept any of <name>'s output types.")
@tracking.track_command("nodes")
def downstream_cmd(
    name: Annotated[str, typer.Argument(help="Node class name, e.g. 'CheckpointLoaderSimple'.")],
    limit: Annotated[int | None, typer.Option(show_default=False, help="Cap output to N rows.")] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )
    nodes = graph.downstream(name)

    total_downstream = len(nodes)
    if limit is not None:
        nodes = nodes[: max(0, limit)]
    rows = [_morphism_row(m) for m in nodes]
    payload = {"name": name, "total": total_downstream, "count": len(rows), "rows": rows}

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        if not rows:
            rprint(f"[dim]No downstream nodes for {name!r}.[/dim]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("outputs")
            for r in rows:
                outs = sanitize_markup(", ".join(r["output_types"])) or "[dim]—[/dim]"
                tbl.add_row(sanitize_markup(r["name"] or ""), sanitize_markup(r["category"] or ""), outs)
            renderer.console().print(tbl)
            tail = f" of {total_downstream}" if total_downstream != len(rows) else ""
            rprint(f"[dim]{len(rows)} downstream node(s){tail}[/dim]")
    renderer.emit(payload, command="nodes downstream")


def _alias_slug(class_type: str) -> str:
    """A spec-batch alias for a class name — the same slugging
    ``workflow_ops.capture_recipe`` uses, so the two surfaces mint identical
    alias vocabulary."""
    import re

    return re.sub(r"[^a-z0-9]+", "_", str(class_type or "node").lower()).strip("_") or "node"


def _emit_path_ops(graph, steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Project a routed path (``steps`` from ``find_paths``/``exact_paths``)
    into a ready-to-apply spec batch in the frozen edit vocabulary.

    THE CONTRACT IS THE ROUND-TRIP: the returned specs must pass
    ``workflow_ops.apply_specs`` unchanged. Shape:

    * one ``add_node`` per step, with a deterministic dedup-suffixed ``as:``
      alias (``tinysampler``, ``tinysampler_2``, …);
    * one ``connect`` per step whose consumed type is produced by an earlier
      step — from the NEAREST prior producer's matching output to this step's
      first link input accepting that type. Alias references are emitted as
      bare names (the freeze vocabulary's valid form; the ``$``-canonical
      sugar lands with the vocabulary-freeze PR).

    The path's seed FROM type is produced by nothing in the path, so the first
    step's input deliberately stays unbound — never a phantom connect.
    """
    from comfy_cli.workflow_ops import _types_compatible

    # Deterministic dedup-suffixed aliases, one add_node per step.
    counts: dict[str, int] = {}
    aliases: list[str] = []
    specs: list[dict[str, Any]] = []
    for step in steps:
        class_type = str(step.get("node") or "")
        slug = _alias_slug(class_type)
        counts[slug] = counts.get(slug, 0) + 1
        alias = slug if counts[slug] == 1 else f"{slug}_{counts[slug]}"
        aliases.append(alias)
        specs.append({"op": "add_node", "class_type": class_type, "as": alias})

    def _producer_output(class_type: str, want: str) -> str | None:
        m = graph.node(class_type)
        for p in m.outputs if m is not None else ():
            types = {t.strip() for t in str(p.type).split(",") if t.strip()}
            if want in types or "*" in types:
                return p.name
        return None

    def _consumer_input(class_type: str, want: str) -> str | None:
        m = graph.node(class_type)
        ports = [p for p in (m.inputs if m is not None else ()) if p.is_link]
        # Required inputs first — that's the slot the plan is routing through.
        for p in sorted(ports, key=lambda p: not p.required):
            if _types_compatible(want, p.type):
                return p.name
        return None

    for j, step in enumerate(steps):
        want = str(step.get("input_type") or "")
        if not want:
            continue  # step consumes nothing from the path (e.g. a loader)
        # Nearest prior step whose node actually produces `want` — in exact
        # mode the recorded per-step output_type is one of possibly several
        # outputs, so resolve against the schema, not just the step record.
        for k in range(j - 1, -1, -1):
            src_class = str(steps[k].get("node") or "")
            out_name = _producer_output(src_class, want)
            if out_name is None:
                continue
            in_name = _consumer_input(str(step.get("node") or ""), want)
            if in_name is not None:
                specs.append({"op": "connect", "from": f"{aliases[k]}.{out_name}", "to": f"{aliases[j]}.{in_name}"})
            break
    return specs


@app.command("path", help="Routed paths from one type to another (e.g. MODEL -> IMAGE).")
@tracking.track_command("nodes")
def path_cmd(
    from_type: Annotated[str, typer.Argument(metavar="FROM", help="Starting type, e.g. MODEL.")],
    to_type: Annotated[str, typer.Argument(metavar="TO", help="Target type, e.g. IMAGE.")],
    max_depth: Annotated[int, typer.Option("--max-depth", help="Maximum path length.")] = 6,
    max_paths: Annotated[int, typer.Option("--max-paths", help="Maximum number of paths to return.")] = 10,
    exact: Annotated[
        bool,
        typer.Option(
            "--exact/--loose",
            help="Exact: every step's other required link inputs must be satisfiable (reported per path as 'support'). Loose: any routed sequence.",
        ),
    ] = True,
    emit_ops: Annotated[
        bool,
        typer.Option(
            "--emit-ops",
            help=(
                "Attach each path's plan as a ready-to-apply spec batch under `paths[].ops` "
                "(frozen add_node/connect vocabulary with `as:` aliases) — feed it straight "
                "to `comfy workflow apply --ops`."
            ),
        ),
    ] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    """Envelope contract (``data``):

    - ``mode`` — the *requested* matching mode, ``"exact"`` or ``"loose"``. It
      echoes ``--exact/--loose`` and says nothing about completeness.
    - ``exact`` — the exhaustiveness claim, and deliberately NOT the flag echoed
      back: true only when the listed paths are the complete, type-constrained
      answer. Exact mode that stopped early (``truncated``), was still expanding
      at the bound (``depth_limited``), or dropped an alternate route into an
      already-explored state (``collapsed``) withholds the claim, as does loose
      mode always. ``exact: true`` with ``count: 0`` is therefore a proof that
      no route exists; ``exact: false`` means "these paths, maybe not all".
    - ``truncated`` / ``truncated_by`` / ``depth_limited`` / ``collapsed`` /
      ``not_searched`` — the individual reasons the claim was withheld, so a
      caller can widen the right bound instead of guessing.
    - ``not_searched`` / ``not_searched_reason`` — the walk declined the query
      and never ran, so the empty result is an abstention, not an answer. No
      reason is reachable from this command: the only shape ``search_paths``
      declines is a bound below 1, which is rejected up front with
      ``path_bounds_invalid`` before the graph is even loaded. Same-type queries
      (``MODEL MODEL``) are searched like any other and return the real
      self-returning routes. The fields stay in the envelope so a caller can
      never mistake an abstention the engine adds later for a proof of
      unreachability — an abstention always reports ``exact: false``.
    """
    renderer = get_renderer()

    # A bound below 1 admits no path at all, so the search would return an empty
    # result with every flag false — i.e. `exact: true, count: 0`, a proof that
    # no route exists. That proof would come from the typo, not from a walk, so
    # refuse the bound instead of emitting it.
    if max_depth < 1 or max_paths < 1:
        renderer.error(
            code="path_bounds_invalid",
            message="--max-depth and --max-paths must be at least 1.",
            hint="retry with `--max-depth 6 --max-paths 10`",
            details={"max_depth": max_depth, "max_paths": max_paths},
        )
        raise typer.Exit(code=1)

    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )

    result = graph.search_paths(from_type, to_type, exact=exact, max_depth=max_depth, max_paths=max_paths)
    paths = result["paths"]
    truncated = bool(result["truncated"])
    depth_limited = bool(result["depth_limited"])
    collapsed = bool(result["collapsed"])
    not_searched = bool(result["not_searched"])

    payload = {
        "from": from_type,
        "to": to_type,
        "mode": "exact" if exact else "loose",
        # Not the flag echoed back: the honest claim that these paths are the
        # complete, type-constrained answer. Any early stop (max_paths, the
        # internal state budget), a frontier still expanding at max_depth, an
        # intermediate state reached by a second route that was not re-explored,
        # or a query the walk declined outright means paths may be missing, so
        # the claim is withheld.
        "exact": bool(exact and not truncated and not depth_limited and not collapsed and not not_searched),
        "truncated": truncated,
        "truncated_by": result["truncated_by"],
        "depth_limited": depth_limited,
        "collapsed": collapsed,
        "not_searched": not_searched,
        "not_searched_reason": result["not_searched_reason"],
        "max_depth": max_depth,
        "max_paths": max_paths,
        "count": len(paths),
        "paths": [
            {
                "from": p.get("from"),
                "to": p.get("to"),
                "steps": [
                    {
                        "node": s.get("node"),
                        "from_type": s.get("input_type"),
                        "to_type": s.get("output_type"),
                    }
                    for s in (p.get("steps") or [])
                ],
                "support": list(p.get("support") or []),
                # --emit-ops: the plan as a ready-to-apply spec batch (round-trips
                # through workflow_ops.apply_specs unchanged). Absent without the
                # flag so the default output stays byte-identical.
                **({"ops": _emit_path_ops(graph, list(p.get("steps") or []))} if emit_ops else {}),
            }
            for p in paths
        ],
    }

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        if not paths:
            rprint(
                f"[dim]No {'exact' if exact else 'routed'} paths from {from_type} to {to_type} within depth {max_depth}.[/dim]"
            )
        else:
            for p in paths:
                chain = " [dim]→[/dim] ".join(
                    f"[bold]{sanitize_markup(s.get('node'))}[/bold]" for s in (p.get("steps") or [])
                )
                rprint(
                    f"[cyan]{sanitize_markup(p.get('from'))}[/cyan]  {chain}  "
                    f"[cyan]{sanitize_markup(p.get('to'))}[/cyan]"
                )
                needs = ", ".join(
                    f"{sanitize_markup(s.get('type'))} from {sanitize_markup(s.get('node'))}"
                    for s in (p.get("support") or [])
                )
                if needs:
                    rprint(f"  [dim]also needs: {needs}[/dim]")
            rprint(f"[dim]{len(paths)} path(s)[/dim]")
        if truncated:
            rprint(f"[dim]Partial result — stopped at {payload['truncated_by']}; more paths may exist.[/dim]")
        elif depth_limited:
            rprint(f"[dim]Searched to depth {max_depth}; longer paths were not explored.[/dim]")
        elif collapsed:
            rprint("[dim]Equivalent alternate routes were collapsed; this is a sample, not every path.[/dim]")
    renderer.emit(payload, command="nodes path")


# ---------------------------------------------------------------------------
# browse: types / categories
# ---------------------------------------------------------------------------


@app.command("types", help="List all connection types in the loaded environment, ranked by connectivity.")
@tracking.track_command("nodes")
def types_cmd(
    limit: Annotated[int | None, typer.Option(show_default=False, help="Cap output to N types.")] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )
    types = graph.list_types()

    if limit is not None:
        types = types[: max(0, limit)]
    payload = {"count": len(types), "types": list(types)}

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        from rich.columns import Columns

        renderer.console().print(Columns([f"[cyan]{sanitize_markup(t)}[/cyan]" for t in types], expand=True))
        rprint(f"[dim]{len(types)} type(s)[/dim]")
    renderer.emit(payload, command="nodes types")


def _flatten_category_tree(tree: dict[str, Any]) -> list[tuple[str, int]]:
    """Walk the CategoryTree → flat [(full_path, count)].

    Shape: every node has ``FullPath``, ``Count``, and ``Children`` (a dict
    keyed by name). The root sits under ``Root``.
    """
    out: list[tuple[str, int]] = []
    if not isinstance(tree, dict):
        return out
    root = tree.get("Root")
    if not isinstance(root, dict):
        return out

    def walk(node: dict[str, Any]) -> None:
        children = node.get("Children")
        if not isinstance(children, dict):
            return
        for child in children.values():
            if not isinstance(child, dict):
                continue
            full = str(child.get("FullPath") or "")
            count = int(child.get("Count") or 0)
            if full:
                out.append((full, count))
            walk(child)

    walk(root)
    return out


@app.command("categories", help="Browse the category tree.")
@tracking.track_command("nodes")
def categories_cmd(
    prefix: Annotated[
        str | None,
        typer.Option("--prefix", show_default=False, help="Only categories starting with this path."),
    ] = None,
    limit: Annotated[int | None, typer.Option(show_default=False, help="Cap output to N rows.")] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
):
    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )
    tree = graph.category_tree()

    flat = _flatten_category_tree(tree)
    if prefix:
        flat = [(p, c) for p, c in flat if p.startswith(prefix)]
    flat.sort(key=lambda x: x[0])
    if limit is not None:
        flat = flat[: max(0, limit)]

    payload = {
        "prefix": prefix,
        "count": len(flat),
        "rows": [{"category": p, "node_count": c} for p, c in flat],
    }

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if renderer.is_pretty():
        if not flat:
            rprint(f"[dim]No categories{' matching ' + prefix if prefix else ''}.[/dim]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("category")
            tbl.add_column("nodes", justify="right", style="dim")
            for p, c in flat:
                tbl.add_row(sanitize_markup(p), sanitize_markup(c))
            renderer.console().print(tbl)
            rprint(f"[dim]{len(flat)} categories[/dim]")
    renderer.emit(payload, command="nodes categories")


# ---------------------------------------------------------------------------
# widget-catalog — the derived name↔index projection the CRDT applier needs
# ---------------------------------------------------------------------------


@app.command(
    "widget-catalog",
    help=(
        "Emit the widget catalog: per-class widget order (plus autogrow/inputcount families) "
        "with a content-hash catalog_version. The projection of object_info a name<->index "
        "widget converter needs."
    ),
)
@tracking.track_command("nodes")
def widget_catalog_cmd(
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="'cloud' to query Comfy Cloud's catalog; default is local."),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a local object_info JSON (offline mode)."),
    ] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    select: Annotated[
        str | None,
        typer.Option(
            "--select",
            show_default=False,
            help="Project the payload: dot path (types.KSampler.widget_order), comma multi-select.",
        ),
    ] = None,
):
    """Export ``{types: {<class_type>: {widget_order, ...}}}`` + ``catalog_version``.

    A ComfyUI workflow stores widget values POSITIONALLY (``widgets_values``);
    the CRDT document the cloud agent and the frontend co-edit stores them BY
    NAME. Converting between the two needs the widget order — which
    ``cql.engine.Graph`` already computes for every ``set-widget`` in this CLI,
    including the two shapes a naive projection gets wrong (the synthetic
    ``control_after_generate`` slot, and dynamic-combo sub-widget expansion).
    Exporting it from here means there is one implementation of widget order,
    not one per consumer; see ``comfy_cli.cql.widget_catalog``.

    Offline is the normal case for a server-side host: with
    ``COMFY_OBJECT_INFO_FILE`` set (or ``--input``) this reads a baked dump and
    never touches the network or a credential.
    """
    from comfy_cli.cql.widget_catalog import build_catalog

    renderer = get_renderer()
    _stale: dict = {}
    graph = _get_graph(
        input_path,
        host,
        port,
        where=where,
        on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err),
    )

    payload = build_catalog(graph)

    if _stale:
        # A stale catalog is still a usable catalog — the version pins WHICH one
        # it is, so a consumer that cached a different version re-fetches. Warn,
        # don't fail: the alternative is no catalog at all.
        payload["stale"] = True
        payload["warnings"] = [
            {
                "code": "object_info_stale",
                "message": f"served from cache ({_stale['source']}): {_stale['reason']}",
            }
        ]

    if select is not None:
        from comfy_cli.selector import emit_selected

        return emit_selected(renderer, payload, select, command="nodes widget-catalog")

    if renderer.is_pretty():
        from rich.table import Table

        rprint(f"[bold]{payload['class_count']}[/bold] class(es)  [dim]{payload['catalog_version']}[/dim]")
        table = Table(show_header=True, header_style="bold")
        table.add_column("class_type")
        table.add_column("widgets")
        table.add_column("widget_order")
        for class_type, entry in sorted(payload["types"].items()):
            order = entry["widget_order"]
            table.add_row(sanitize_markup(class_type), str(len(order)), sanitize_markup(", ".join(order)))
        renderer.console().print(table)

    renderer.emit(payload, command="nodes widget-catalog")


# ---------------------------------------------------------------------------
# refresh — object_info is fetched live; the annotation data is what's cached
# ---------------------------------------------------------------------------


@app.command(
    "refresh",
    help=(
        "Re-fetch node annotation data (pack/labels/cloud_disabled) from Comfy-Org/comfy-complete. "
        "Set COMFY_CLI_NO_REMOTE_REFRESH=1 to keep every `nodes` command off the network."
    ),
)
@tracking.track_command("nodes")
def refresh_cmd(
    where: Annotated[
        str | None,
        typer.Option(
            "--where",
            show_default=False,
            hidden=True,
            help="Deprecated and ignored — annotation data is the same for local and cloud.",
        ),
    ] = None,
):
    """Force-refresh the node annotation cache from the public comfy-complete repo.

    ``object_info`` itself is fetched live from the server on every command, so
    there is nothing to refresh there. The *annotations* (which custom-node pack
    a node belongs to, its behavioral labels, and whether it's disabled on
    cloud) come from Comfy-Org/comfy-complete and are cached locally with a TTL;
    this command pulls the latest copy immediately.

    ``--where`` is accepted and ignored. It steered nothing even when this
    command was a no-op, and the annotation files are routing-independent — but
    it was in the CLI's own error hints and in two shipped skill docs, so
    rejecting it would turn "you followed the hint" into ``No such option``
    (exit 2) for anyone on an older doc. Hidden from ``--help`` so nothing new
    learns it.
    """
    renderer = get_renderer()
    from comfy_cli.cql import annotations_source

    results = annotations_source.refresh_annotations()
    ok = all(r["source"] == "remote" for r in results)
    if renderer.is_pretty():
        for r in results:
            if r["source"] == "remote":
                # A remote fetch that couldn't be persisted still refreshed this
                # run's data; say so, and say why it won't survive to the next.
                dest = r["path"] or f"not cached ({r.get('cache_error', 'cache unavailable')})"
                rprint(f"[green]✓[/green] {r['name']} ({r['bytes']:,} bytes) → {dest}")
            elif r["source"] == "bundled":
                # Not necessarily a failure: COMFY_CLI_NO_REMOTE_REFRESH lands
                # here by design, so let the reason carry the why.
                rprint(
                    f"[yellow]![/yellow] {r['name']}: using bundled snapshot "
                    f"([dim]{r.get('error') or 'remote unavailable'}[/dim])"
                )
            else:
                rprint(f"[red]✗[/red] {r['name']}: unavailable ([dim]{r.get('error') or 'no source'}[/dim])")
    renderer.emit({"refreshed": ok, "files": results}, command="nodes refresh")
