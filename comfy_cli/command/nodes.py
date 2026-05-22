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

Backed today by the Python CQL stub (``comfy_cli.cql.loader``). If/when we
bind to upstream ``comfygraph`` the user-facing surface here stays the
same — only the backing call changes.
"""

from __future__ import annotations

from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.cql.errors import CQLRuntimeError
from comfy_cli.cql.loader import load_graph
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(no_args_is_help=True, help="Introspect ComfyUI node classes (inputs, outputs, categories).")


# ---------------------------------------------------------------------------
# graph resolution — shared across ls/show/search
# ---------------------------------------------------------------------------


def _resolved_where(where: str | None) -> str:
    """Apply the full precedence chain: per-command flag > env > config > default."""
    from comfy_cli import where as where_module
    from comfy_cli.config_manager import ConfigManager

    cfg = ConfigManager()
    decision = where_module.resolve(
        flag=where,
        config_value=cfg.get(where_module.CONFIG_KEY_WHERE_DEFAULT),
    )
    return decision.target.value  # "local" | "cloud"


def _resolve_graph(
    input_path: str | None,
    host: str | None,
    port: int | None,
    where: str | None = None,
) -> dict[str, Any]:
    """Load the normalized graph for `comfy nodes ls / show / search`.

    Routing follows the standard precedence: explicit ``--where`` > env
    (``COMFY_WHERE``) > config (``where_default``) > local default. The
    ``--input <path>`` flag short-circuits everything (offline mode).
    """
    if input_path:
        return load_graph(input_path=input_path, host=host, port=port)
    if _resolved_where(where) == "cloud":
        raw = _load_raw_object_info(None, None, None, mode="cloud")
        from comfy_cli.cql.loader import normalize

        return normalize(raw)
    return load_graph(
        input_path=None,
        host=host or "127.0.0.1",
        port=port or 8188,
    )


def _inputs_index(graph: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Group inputs by node name for O(1) lookup."""
    index: dict[str, list[dict[str, Any]]] = {}
    for row in graph.get("inputs", []):
        if not isinstance(row, dict):
            continue
        node = row.get("node")
        if isinstance(node, str):
            index.setdefault(node, []).append(row)
    return index


def _resolve_choices(input_row: dict[str, Any]) -> list[Any]:
    """Return the enum choices for an input, regardless of where the graph put them.

    Local ENUM inputs surface their values in `choices`. Cloud-API COMBO inputs
    surface theirs at `options.options`. From the caller's point of view both
    are the same idea ("what values can I pick?") so we normalize to one field.
    """
    direct = input_row.get("choices")
    if isinstance(direct, list) and direct:
        return list(direct)
    options = input_row.get("options") or {}
    nested = options.get("options") if isinstance(options, dict) else None
    if isinstance(nested, list):
        return list(nested)
    return []


def _node_inputs(inputs_index: dict[str, list[dict[str, Any]]], name: str) -> list[dict[str, Any]]:
    """Return all inputs for a node, sorted by section (required first) then name."""
    rows = inputs_index.get(name, [])
    return sorted(
        rows,
        key=lambda r: (
            0 if r.get("section") == "required" else 1 if r.get("section") == "optional" else 2,
            str(r.get("name") or ""),
        ),
    )


def _node_accepts_type(inputs_index: dict[str, list[dict[str, Any]]], name: str, type_: str) -> bool:
    """True if this node has at least one input of `type_` (case-insensitive)."""
    target = type_.upper()
    for inp in inputs_index.get(name, []):
        t = inp.get("type")
        if isinstance(t, str) and t.upper() == target:
            return True
    return False


def _node_produces_type(node: dict[str, Any], type_: str) -> bool:
    """True if this node's output_types contains `type_` (case-insensitive)."""
    target = type_.upper()
    outs = node.get("output_types") or []
    return any(isinstance(t, str) and t.upper() == target for t in outs)


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


def _ls_via_query(
    renderer,
    query: str,
    input_path: str | None,
    host: str | None,
    port: int | None,
    limit: int | None,
) -> None:
    """Run a CQL grammar query through comfygraph (wasm) and emit the result."""
    from comfy_cli import comfygraph

    try:
        obj_info = _load_raw_object_info(input_path, host, port)
    except comfygraph.ComfygraphError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <path>, or start the server with `comfy launch`"),
        )
        raise typer.Exit(code=1) from e

    try:
        result = comfygraph.run_query(query, obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(
            code="cql_query_invalid",
            message=str(e),
            hint="check the grammar — see `comfy nodes ls --help` for examples",
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    nodes = result.get("Nodes") or []
    if limit is not None:
        nodes = nodes[: max(0, limit)]

    payload = {
        "query": query,
        "count": len(nodes),
        "total": result.get("Total", len(nodes)),
        "counted": bool(result.get("Counted")),
        "limited": bool(result.get("Limited")),
        "hints": list(result.get("Hints") or []),
        "rows": [
            {
                "name": n.get("id"),
                "category": n.get("category"),
                "display_name": n.get("display_name"),
                "description": n.get("description"),
                "output_types": [
                    str(p.get("type") if isinstance(p, dict) else p)
                    for p in (n.get("outputs") or [])
                    if (p.get("type") if isinstance(p, dict) else p)
                ],
            }
            for n in nodes
        ],
    }

    if renderer.is_pretty():
        from rich.table import Table

        if not nodes:
            rprint("[dim]0 nodes matched.[/dim]")
        else:
            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("description", style="dim")
            for n in nodes:
                desc = (n.get("description") or "")[:60]
                tbl.add_row(str(n.get("id") or ""), n.get("category") or "", desc)
            renderer.console().print(tbl)
            rprint(
                f"[dim]{len(nodes)} node(s){' (of ' + str(result.get('Total')) + ')' if result.get('Total') and result.get('Total') != len(nodes) else ''}[/dim]"
            )
        for h in result.get("Hints") or []:
            rprint(f"[dim]hint:[/dim] {h}")
    renderer.emit(payload, command="nodes ls")


def _load_raw_object_info(
    input_path: str | None,
    host: str | None,
    port: int | None,
    *,
    mode: str = "local",
) -> dict[str, Any]:
    """Load raw object_info, delegating to the unified loader."""
    from comfy_cli.comfygraph import load_object_info

    return load_object_info(
        mode=mode,
        input_path=input_path,
        host=host or "127.0.0.1",
        port=port or 8188,
    )


@app.command(
    "ls",
    help="List node classes. Filter via --produces/--accepts/--category, or pass --query for the full CQL grammar.",
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
    query: Annotated[
        str | None,
        typer.Option(
            "--query",
            "-q",
            show_default=False,
            help="Power-user: a CQL grammar query (e.g. 'produces IMAGE | NOT deprecated | sort connections'). Bypasses the flag filters.",
        ),
    ] = None,
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
):
    renderer = get_renderer()

    # Power-user path: agent passes a real CQL query. Route through the wasm
    # grammar (comfygraph) rather than the flag-based filters.
    if query is not None:
        return _ls_via_query(renderer, query, input_path, host, port, limit)

    try:
        graph = _resolve_graph(input_path, host, port)
    except CQLRuntimeError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint="pass --input <path>, or start the server with `comfy launch`",
            details=e.as_details(),
        )
        raise typer.Exit(code=1)

    inputs_index = _inputs_index(graph) if accepts else {}
    nodes: list[dict[str, Any]] = []
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        name = n.get("name")
        if not isinstance(name, str):
            continue
        if produces and not _node_produces_type(n, produces):
            continue
        if accepts and not _node_accepts_type(inputs_index, name, accepts):
            continue
        if category and not _category_matches(n.get("category"), category):
            continue
        nodes.append(n)

    nodes.sort(key=lambda r: str(r.get("name") or ""))
    total_matched = len(nodes)
    if limit is not None:
        nodes = nodes[: max(0, limit)]

    payload = {
        "filter": {
            "produces": produces,
            "accepts": accepts,
            "category": category,
        },
        "total": total_matched,
        "count": len(nodes),
        "rows": [
            {
                "name": n.get("name"),
                "category": n.get("category"),
                "display_name": n.get("display_name"),
                "output_types": list(n.get("output_types") or []),
                "output_node": bool(n.get("output_node")),
            }
            for n in nodes
        ],
    }

    if renderer.is_pretty():
        if not nodes:
            rprint("[dim]0 nodes matched.[/dim]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("outputs")
            for n in nodes:
                outs = ", ".join(n.get("output_types") or []) or "[dim]—[/dim]"
                tbl.add_row(n.get("name") or "", n.get("category") or "", outs)
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
    try:
        graph = _resolve_graph(input_path, host, port, where=where)
    except CQLRuntimeError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint="pass --input <path>, or start the server with `comfy launch`",
            details=e.as_details(),
        )
        raise typer.Exit(code=1)

    node = next(
        (n for n in graph.get("nodes", []) if isinstance(n, dict) and n.get("name") == name),
        None,
    )
    if node is None:
        # Surface near-matches so the agent can self-correct from the error.
        import difflib

        all_names = [n.get("name") for n in graph.get("nodes", []) if isinstance(n, dict)]
        all_names = [n for n in all_names if isinstance(n, str)]
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

    inputs = _node_inputs(_inputs_index(graph), name)
    payload = {
        "name": node.get("name"),
        "display_name": node.get("display_name"),
        "category": node.get("category"),
        "description": node.get("description"),
        "output_node": bool(node.get("output_node")),
        "output_types": list(node.get("output_types") or []),
        "inputs": [
            {
                "name": i.get("name"),
                "type": i.get("type"),
                "section": i.get("section"),
                "choices": _resolve_choices(i),
                "options": i.get("options") or {},
            }
            for i in inputs
        ],
    }

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
        if inputs:
            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("input")
            tbl.add_column("type")
            tbl.add_column("section", style="dim")
            tbl.add_column("default", style="dim")
            for i in inputs:
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
):
    renderer = get_renderer()
    try:
        graph = _resolve_graph(input_path, host, port)
    except CQLRuntimeError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint="pass --input <path>, or start the server with `comfy launch`",
            details=e.as_details(),
        )
        raise typer.Exit(code=1)

    q = query.lower()
    scored: list[tuple[int, dict[str, Any]]] = []
    for n in graph.get("nodes", []):
        if not isinstance(n, dict):
            continue
        name = str(n.get("name") or "")
        display = str(n.get("display_name") or "")
        desc = str(n.get("description") or "")
        # Simple scoring: exact name hit > prefix > substring in name > display > description.
        if name.lower() == q:
            score = 0
        elif name.lower().startswith(q):
            score = 1
        elif q in name.lower():
            score = 2
        elif q in display.lower():
            score = 3
        elif q in desc.lower():
            score = 4
        else:
            continue
        scored.append((score, n))

    scored.sort(key=lambda x: (x[0], str(x[1].get("name") or "")))
    total_matched = len(scored)
    nodes = [n for _, n in scored[: max(0, limit)]]

    payload = {
        "query": query,
        "total": total_matched,
        "count": len(nodes),
        "rows": [
            {
                "name": n.get("name"),
                "category": n.get("category"),
                "display_name": n.get("display_name"),
                "description": n.get("description"),
                "output_types": list(n.get("output_types") or []),
            }
            for n in nodes
        ],
    }

    if renderer.is_pretty():
        if not nodes:
            rprint(f"[dim]No nodes match {query!r}.[/dim]")
        else:
            from rich.table import Table

            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("name")
            tbl.add_column("category", style="dim")
            tbl.add_column("description", style="dim")
            for n in nodes:
                desc = (n.get("description") or "")[:60]
                tbl.add_row(n.get("name") or "", n.get("category") or "", desc)
            renderer.console().print(tbl)
            rprint(f"[dim]{len(nodes)} node(s)[/dim]")
    renderer.emit(payload, command="nodes search")


# ---------------------------------------------------------------------------
# graph traversal: upstream / downstream / path
# ---------------------------------------------------------------------------


def _wasm_load_or_fail(renderer, input_path, host, port, *, mode="local"):
    """Fetch object_info (raw) for any wasm-backed command. Returns the dict or exits."""
    from comfy_cli.comfygraph import ComfygraphError

    try:
        return _load_raw_object_info(input_path, host, port, mode=mode)
    except ComfygraphError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <path> or start the server with `comfy launch`"),
        )
        raise typer.Exit(code=1) from e


def _morphism_row(m: dict[str, Any]) -> dict[str, Any]:
    """Project a Morphism returned by the wasm into our agent-friendly row shape."""
    return {
        "name": m.get("id"),
        "category": m.get("category"),
        "display_name": m.get("display_name"),
        "output_types": [
            str(p.get("type") if isinstance(p, dict) else p)
            for p in (m.get("outputs") or [])
            if (p.get("type") if isinstance(p, dict) else p)
        ],
    }


@app.command("upstream", help="List nodes that can feed into <name>'s required link inputs.")
@tracking.track_command("nodes")
def upstream_cmd(
    name: Annotated[str, typer.Argument(help="Node class name, e.g. 'KSampler'.")],
    limit: Annotated[int | None, typer.Option(show_default=False, help="Cap output to N rows.")] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    obj_info = _wasm_load_or_fail(renderer, input_path, host, port)
    try:
        nodes = comfygraph.upstream(name, obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(code="cql_query_invalid", message=str(e), details=e.details)
        raise typer.Exit(code=1) from e

    if limit is not None:
        nodes = nodes[: max(0, limit)]
    rows = [_morphism_row(n) for n in nodes]
    payload = {"name": name, "count": len(rows), "rows": rows}

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
            rprint(f"[dim]{len(rows)} upstream node(s)[/dim]")
    renderer.emit(payload, command="nodes upstream")


@app.command("downstream", help="List nodes that accept any of <name>'s output types.")
@tracking.track_command("nodes")
def downstream_cmd(
    name: Annotated[str, typer.Argument(help="Node class name, e.g. 'CheckpointLoaderSimple'.")],
    limit: Annotated[int | None, typer.Option(show_default=False, help="Cap output to N rows.")] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    obj_info = _wasm_load_or_fail(renderer, input_path, host, port)
    try:
        nodes = comfygraph.downstream(name, obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(code="cql_query_invalid", message=str(e), details=e.details)
        raise typer.Exit(code=1) from e

    if limit is not None:
        nodes = nodes[: max(0, limit)]
    rows = [_morphism_row(n) for n in nodes]
    payload = {"name": name, "count": len(rows), "rows": rows}

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
            rprint(f"[dim]{len(rows)} downstream node(s)[/dim]")
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
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    obj_info = _wasm_load_or_fail(renderer, input_path, host, port)
    try:
        finder = comfygraph.exact_paths if exact else comfygraph.find_paths
        paths = finder(from_type, to_type, obj_info, max_depth=max_depth, max_paths=max_paths)
    except comfygraph.ComfygraphError as e:
        renderer.error(code="cql_query_invalid", message=str(e), details=e.details)
        raise typer.Exit(code=1) from e

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
                        "input_type": s.get("input_type"),
                        "output_type": s.get("output_type"),
                    }
                    for s in (p.get("steps") or [])
                ],
            }
            for p in paths
        ],
    }

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
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    obj_info = _wasm_load_or_fail(renderer, input_path, host, port)
    try:
        types = comfygraph.list_types(obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(code="cql_query_invalid", message=str(e), details=e.details)
        raise typer.Exit(code=1) from e

    if limit is not None:
        types = types[: max(0, limit)]
    payload = {"count": len(types), "types": list(types)}

    if renderer.is_pretty():
        from rich.columns import Columns

        renderer.console().print(Columns([f"[cyan]{t}[/cyan]" for t in types], expand=True))
        rprint(f"[dim]{len(types)} type(s)[/dim]")
    renderer.emit(payload, command="nodes types")


def _flatten_category_tree(tree: dict[str, Any]) -> list[tuple[str, int]]:
    """Walk the wasm CategoryTree → flat [(full_path, count)].

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
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    obj_info = _wasm_load_or_fail(renderer, input_path, host, port)
    try:
        tree = comfygraph.category_tree(obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(code="cql_query_invalid", message=str(e), details=e.details)
        raise typer.Exit(code=1) from e

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
# refresh — fetch live object_info for the resolved --where, update the cache
# ---------------------------------------------------------------------------


@app.command(
    "refresh",
    help=(
        "Fetch a fresh object_info from the resolved --where (cloud writes to "
        "the user cache; local always reads live and needs no refresh)."
    ),
)
@tracking.track_command("nodes")
def refresh_cmd(
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Override the resolved routing mode."),
    ] = None,
):
    """Refresh the catalog cache for the resolved routing mode.

    Cloud → fetch live ``object_info`` from ``cloud.comfy.org`` via the
    auth-aware client, write to ``<config>/cache/object_info-cloud.json``.
    Subsequent ``comfy nodes …`` calls read from this cache, taking
    priority over the bundled snapshot.

    Local → no cache, always reads live from ``127.0.0.1:8188``. This
    command surfaces that and exits successfully; nothing to refresh.
    """
    from comfy_cli import comfygraph

    renderer = get_renderer()
    mode = _resolved_where(where)

    if mode == "local":
        if renderer.is_pretty():
            rprint("[dim]Local routing reads `/object_info` live on every call — nothing to cache.[/dim]")
        renderer.emit(
            {"mode": "local", "refreshed": False, "reason": "live_local"},
            command="nodes refresh",
        )
        return

    # cloud
    try:
        cache_file, n_bytes = comfygraph.refresh_cloud_snapshot()
    except comfygraph.ComfygraphError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint="check `comfy auth whoami` — needs X-API-Key or OAuth",
            details=e.details,
        )
        raise typer.Exit(code=1) from e
    except (OSError, ValueError) as e:
        renderer.error(
            code="cloud_http_error",
            message=f"refresh failed: {e}",
            hint="check the network and `comfy auth whoami`",
        )
        raise typer.Exit(code=1) from e

    if renderer.is_pretty():
        from rich.text import Text

        rprint(
            Text.from_markup(
                f"[bold green]✓[/bold green] Cloud catalog refreshed → "
                f"[dim]{cache_file}[/dim] ([cyan]{n_bytes / 1024:.0f} KiB[/cyan])"
            )
        )
    renderer.emit(
        {
            "mode": "cloud",
            "refreshed": True,
            "cache_file": str(cache_file),
            "bytes": n_bytes,
        },
        command="nodes refresh",
        changed=True,
    )
