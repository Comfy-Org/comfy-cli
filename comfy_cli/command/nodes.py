"""``comfy nodes`` — node-class introspection.

Agent-facing wrappers over the same object_info source CQL consumes, but
with flag-based interfaces an LLM can pattern-match without learning a
grammar. Three primitives:

    comfy nodes ls   [--produces T] [--accepts T] [--category PAT] [--limit N]
    comfy nodes show <NodeClass>
    comfy nodes search <text>

All three resolve the graph in this order:
    1. ``--input <path>`` to an object_info dump (offline mode)
    2. ``--host:--port`` live ComfyUI server (default 127.0.0.1:8188)

Backed by the pure-Python CQL engine (``comfy_cli.cql.engine.Graph``).
"""

from __future__ import annotations

import difflib
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.cql.engine import Graph, LoadError
from comfy_cli.output import get_renderer, rprint

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
    ``--input <path>`` flag short-circuits everything (offline mode).

    ``on_stale``, if provided, is forwarded to ``resilient_load_object_info``
    and fired when a stale-cache fallback occurs (see loader for signature).
    """
    mode = _resolved_where(where)
    try:
        if input_path is not None:
            # Explicit offline dump — let Graph.load read + annotate it.
            return Graph.load(
                mode=mode,
                input_path=input_path,
                host=host or "127.0.0.1",
                port=port or 8188,
            )
        # Live fetch goes through the resilient loader: auto-cache on success,
        # refresh-and-retry once, then fall back to the last cached dump (with
        # a stderr warning) when the server/session is briefly unreachable.
        from comfy_cli.cql.loader import resilient_load_object_info

        raw = resilient_load_object_info(
            mode=mode,
            host=host or "127.0.0.1",
            port=port or 8188,
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
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
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
                outs = ", ".join(m.output_types()) or "[dim]—[/dim]"
                tbl.add_row(m.id, m.category or "", outs)
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
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
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

    if renderer.is_pretty():
        from rich.table import Table

        rprint(
            f"[bold]{payload['name']}[/bold]"
            + (
                f"  [dim]({payload['display_name']})[/dim]"
                if payload["display_name"] and payload["display_name"] != payload["name"]
                else ""
            )
        )
        if payload["category"]:
            rprint(f"[dim]category[/dim]  {payload['category']}")
        if payload["description"]:
            rprint(f"[dim]{payload['description']}[/dim]")
        outs = ", ".join(payload["output_types"]) or "(none)"
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
                    str(i.get("name") or ""),
                    str(i.get("type") or ""),
                    str(i.get("section") or ""),
                    "" if default is None else str(default),
                )
            renderer.console().print(tbl)
    renderer.emit(payload, command="nodes show")


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command("search", help="Fuzzy-search node classes by name, display name, or description.")
@tracking.track_command("nodes")
def search_cmd(
    query: Annotated[str, typer.Argument(help="Text to search for (case-insensitive substring).")],
    limit: Annotated[int, typer.Option(help="Cap output to N rows.")] = 20,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a local object_info JSON (offline mode)."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
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

    q = query.lower()
    scored: list[tuple[int, Any]] = []
    for m in graph.all_nodes():
        name_l = m.id.lower()
        display_l = m.display_name.lower()
        desc_l = m.description.lower()
        # Simple scoring: exact name hit > prefix > substring in name > display > description.
        if name_l == q:
            score = 0
        elif name_l.startswith(q):
            score = 1
        elif q in name_l:
            score = 2
        elif q in display_l:
            score = 3
        elif q in desc_l:
            score = 4
        else:
            continue
        scored.append((score, m))

    scored.sort(key=lambda x: (x[0], x[1].id))
    total_matched = len(scored)
    matched = [m for _, m in scored[: max(0, limit)]]

    payload = {
        "query": query,
        "total": total_matched,
        "count": len(matched),
        "rows": [
            {
                "name": m.id,
                "category": m.category,
                "display_name": m.display_name,
                "description": m.description,
                "output_types": m.output_types(),
            }
            for m in matched
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
        if not matched:
            rprint(f"[dim]No nodes match {query!r}.[/dim]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("description", style="dim")
            for m in matched:
                desc = m.description[:60]
                tbl.add_row(m.id, m.category or "", desc)
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
                outs = ", ".join(r["output_types"]) or "[dim]—[/dim]"
                tbl.add_row(r["name"] or "", r["category"] or "", outs)
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
                outs = ", ".join(r["output_types"]) or "[dim]—[/dim]"
                tbl.add_row(r["name"] or "", r["category"] or "", outs)
            renderer.console().print(tbl)
            tail = f" of {total_downstream}" if total_downstream != len(rows) else ""
            rprint(f"[dim]{len(rows)} downstream node(s){tail}[/dim]")
    renderer.emit(payload, command="nodes downstream")


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
            help="Exact: every step's required link inputs must be satisfiable from the path so far. Loose: any routed sequence.",
        ),
    ] = True,
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

    finder = graph.exact_paths if exact else graph.find_paths
    paths = finder(from_type, to_type, max_depth=max_depth, max_paths=max_paths)

    payload = {
        "from": from_type,
        "to": to_type,
        "exact": exact,
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
                chain = " [dim]→[/dim] ".join(f"[bold]{s.get('node')}[/bold]" for s in (p.get("steps") or []))
                rprint(f"[cyan]{p.get('from')}[/cyan]  {chain}  [cyan]{p.get('to')}[/cyan]")
            rprint(f"[dim]{len(paths)} path(s)[/dim]")
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

        renderer.console().print(Columns([f"[cyan]{t}[/cyan]" for t in types], expand=True))
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
                tbl.add_row(p, str(c))
            renderer.console().print(tbl)
            rprint(f"[dim]{len(flat)} categories[/dim]")
    renderer.emit(payload, command="nodes categories")


# ---------------------------------------------------------------------------
# refresh — object_info is fetched live; nothing to cache
# ---------------------------------------------------------------------------


@app.command(
    "refresh",
    help="object_info is fetched live from the server on each command — nothing to refresh.",
)
@tracking.track_command("nodes")
def refresh_cmd(
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Override the resolved routing mode."),
    ] = None,
):
    """Explain that object_info is fetched live and exit."""
    renderer = get_renderer()
    rprint("[dim]object_info is fetched live from the server on each command — nothing to refresh.[/dim]")
    renderer.emit({"refreshed": False, "reason": "live_fetch"}, command="nodes refresh")
