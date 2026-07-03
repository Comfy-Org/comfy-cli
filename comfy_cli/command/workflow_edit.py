"""``comfy workflow add-node/connect/set-widget/delete`` — structured,
CRDT-ready edit primitives for frontend-format ComfyUI workflows.

Each command mutates the workflow file in place (or ``--stdout``) AND emits a
replayable operation in the envelope's ``data.op``. The op is what a CRDT/merge
consumer (the cloud agent) applies; the file write is the single-writer local
path. Both come from the same ``comfy_cli.workflow_ops`` core, so a local edit
and a server-side merge stay in lock-step.

Node/link identity is leaderless (random 53-bit ints) so concurrent edits never
collide; widgets are addressed by name, not array index. See ``workflow_ops``.
"""

from __future__ import annotations

import json
from typing import Annotated, Any

import typer

from comfy_cli import tracking, workflow_ops
from comfy_cli.command.workflow import (
    _atomic_write_text,
    _get_graph,
    _load_workflow_or_fail,
    _parse_value,
)
from comfy_cli.output import get_renderer, rprint


def _split_addr(addr: str, renderer) -> tuple[Any, str]:
    """Split ``<node_id>.<name>`` → (node_id, name). node_id is int when numeric."""
    if "." not in addr:
        renderer.error(
            code="workflow_edit_invalid",
            message=f"expected `<node_id>.<name>`, got {addr!r}",
            hint="example: `3.steps` — run `comfy workflow slots <file>` to list node ids",
        )
        raise typer.Exit(code=1)
    node_str, _, name = addr.partition(".")
    node_str = node_str.strip()
    node_id: Any = int(node_str) if node_str.lstrip("-").isdigit() else node_str
    return node_id, name.strip()


def _finish(renderer, p, workflow: dict, op: dict, base_version: int, stdout: bool, command: str) -> None:
    """Serialize the mutated workflow (file or stdout) and emit the op envelope."""
    workflow_ops.strip_internal(workflow)
    serialized = json.dumps(workflow, indent=2)
    wrote: str | None = None
    if stdout:
        import sys

        sys.stdout.write(serialized)
        sys.stdout.write("\n")
    else:
        _atomic_write_text(p, serialized)
        wrote = str(p)
    payload = {
        "workflow": str(p),
        "op": op,
        "base_version": base_version,
        "version": base_version + 1,
        "wrote": wrote,
    }
    if op.get("warnings"):
        payload["warnings"] = op["warnings"]
    if renderer.is_pretty():
        rprint(f"[bold green]✓[/bold green] {op['op']} → [dim]{p}[/dim]")
    renderer.emit(payload, command=command, changed=True)


def _graph_or_exit(input_path, host, port, renderer, where=None):
    return _get_graph(input_path, host, port, where=where)


# ---------------------------------------------------------------------------
# add-node
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def add_node_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    class_type: Annotated[str, typer.Argument(help="Node class_type, e.g. KSampler.")],
    at: Annotated[
        str | None,
        typer.Option("--at", show_default=False, help="Canvas position 'x,y' for the new node."),
    ] = None,
    actor: Annotated[str, typer.Option("--actor", help="Op author id (for CRDT stamping).")] = "cli",
    base_version: Annotated[int, typer.Option("--base-version", help="Draft version this edit is based on.")] = 0,
    stdout: Annotated[bool, typer.Option("--stdout/--in-place", show_default=False)] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    renderer = get_renderer()
    renderer.command = "workflow add-node"
    p, workflow = _load_workflow_or_fail(renderer, file)
    graph = _graph_or_exit(input_path, host, port, renderer, where)
    pos = None
    if at:
        try:
            pos = [float(x) for x in at.split(",", 1)]
        except ValueError as e:
            renderer.error(code="workflow_edit_invalid", message=f"--at must be 'x,y': {e}")
            raise typer.Exit(code=1) from e
    try:
        workflow, op = workflow_ops.add_node(
            workflow, graph, class_type, pos=pos, actor=actor, base_version=base_version
        )
    except ValueError as e:
        renderer.error(code="workflow_edit_invalid", message=str(e), hint="run `comfy nodes types` to list class_types")
        raise typer.Exit(code=1) from e
    _finish(renderer, p, workflow, op, base_version, stdout, "workflow add-node")


# ---------------------------------------------------------------------------
# set-widget
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def set_widget_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    addr: Annotated[str, typer.Argument(help="Widget address `<node_id>.<widget_name>`.")],
    value: Annotated[str, typer.Argument(help="New value (parsed as JSON, else literal string).")],
    actor: Annotated[str, typer.Option("--actor", help="Op author id (for CRDT stamping).")] = "cli",
    base_version: Annotated[int, typer.Option("--base-version", help="Draft version this edit is based on.")] = 0,
    stdout: Annotated[bool, typer.Option("--stdout/--in-place", show_default=False)] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    renderer = get_renderer()
    renderer.command = "workflow set-widget"
    p, workflow = _load_workflow_or_fail(renderer, file)
    graph = _graph_or_exit(input_path, host, port, renderer, where)
    node_id, widget = _split_addr(addr, renderer)
    # Subgraph addresses — a promoted input on a subgraph instance (flat
    # ``57.text``, exactly what `slots` advertises) or an interior node (nested
    # ``57/27.text``) — are resolved inside ``workflow_ops.set_widget`` against
    # the same CQL resolver `slots` uses, and emit a replayable op that writes
    # back into the subgraph definition.
    try:
        workflow, op = workflow_ops.set_widget(
            workflow, graph, node_id, widget, _parse_value(value), actor=actor, base_version=base_version
        )
    except ValueError as e:
        renderer.error(
            code="workflow_edit_invalid",
            message=str(e),
            hint="run `comfy workflow slots <file>` to list widget addresses",
        )
        raise typer.Exit(code=1) from e
    _finish(renderer, p, workflow, op, base_version, stdout, "workflow set-widget")


# ---------------------------------------------------------------------------
# connect
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def connect_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    source: Annotated[str, typer.Argument(help="Source `<node_id>.<output_slot>` (slot name or index).")],
    target: Annotated[str, typer.Argument(help="Target `<node_id>.<input_slot>` (slot name or index).")],
    actor: Annotated[str, typer.Option("--actor", help="Op author id (for CRDT stamping).")] = "cli",
    base_version: Annotated[int, typer.Option("--base-version", help="Draft version this edit is based on.")] = 0,
    stdout: Annotated[bool, typer.Option("--stdout/--in-place", show_default=False)] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    renderer = get_renderer()
    renderer.command = "workflow connect"
    p, workflow = _load_workflow_or_fail(renderer, file)
    graph = _graph_or_exit(input_path, host, port, renderer, where)
    from_node, from_slot = _split_addr(source, renderer)
    to_node, to_slot = _split_addr(target, renderer)
    try:
        workflow, op = workflow_ops.connect(
            workflow, graph, from_node, from_slot, to_node, to_slot, actor=actor, base_version=base_version
        )
    except ValueError as e:
        renderer.error(code="workflow_edit_invalid", message=str(e))
        raise typer.Exit(code=1) from e
    _finish(renderer, p, workflow, op, base_version, stdout, "workflow connect")


# ---------------------------------------------------------------------------
# delete
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def delete_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    node: Annotated[str, typer.Argument(help="Node id to delete.")],
    actor: Annotated[str, typer.Option("--actor", help="Op author id (for CRDT stamping).")] = "cli",
    base_version: Annotated[int, typer.Option("--base-version", help="Draft version this edit is based on.")] = 0,
    stdout: Annotated[bool, typer.Option("--stdout/--in-place", show_default=False)] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    renderer = get_renderer()
    renderer.command = "workflow delete-node"
    p, workflow = _load_workflow_or_fail(renderer, file)
    graph = _graph_or_exit(input_path, host, port, renderer, where)
    node_id: Any = int(node) if node.lstrip("-").isdigit() else node
    try:
        workflow, op = workflow_ops.delete_node(workflow, graph, node_id, actor=actor, base_version=base_version)
    except ValueError as e:
        renderer.error(code="workflow_edit_invalid", message=str(e))
        raise typer.Exit(code=1) from e
    _finish(renderer, p, workflow, op, base_version, stdout, "workflow delete")


# ---------------------------------------------------------------------------
# ls-nodes — recover node ids/types (so an agent can address minted nodes)
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def ls_nodes_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
):
    renderer = get_renderer()
    renderer.command = "workflow ls-nodes"
    p, workflow = _load_workflow_or_fail(renderer, file)
    rows = []
    for n in workflow.get("nodes") or []:
        if not isinstance(n, dict):
            continue
        rows.append(
            {
                "id": n.get("id"),
                "type": n.get("type"),
                "title": n.get("title") or (n.get("properties") or {}).get("Node name for S&R"),
            }
        )
    payload = {"workflow": str(p), "count": len(rows), "nodes": rows}
    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("id", no_wrap=True)
        tbl.add_column("type")
        tbl.add_column("title", style="dim")
        for r in rows:
            tbl.add_row(str(r["id"]), str(r["type"]), str(r["title"] or ""))
        renderer.console().print(tbl)
    renderer.emit(payload, command="workflow ls-nodes")


# ---------------------------------------------------------------------------
# capture — project a graph into a reusable recipe (the decompose analog)
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def capture_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON to capture.")],
    name: Annotated[str | None, typer.Option("--name", show_default=False, help="Recipe name.")] = None,
    param: Annotated[
        list[str] | None,
        typer.Option(
            "--param",
            show_default=False,
            help="Lift a widget to a recipe param: `<node_id>.<widget>=<param_name>` (repeatable).",
        ),
    ] = None,
    out: Annotated[
        str | None,
        typer.Option("--out", "-o", show_default=False, help="Write the recipe JSON here (else stdout)."),
    ] = None,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    """Project a workflow into a reusable recipe — the op-batch that rebuilds it.
    `apply` that recipe onto an empty graph to reproduce the workflow; edit a value
    to a `${param}` to make it parameterized."""
    from pathlib import Path

    renderer = get_renderer()
    renderer.command = "workflow capture"
    p, workflow = _load_workflow_or_fail(renderer, file)
    graph = _graph_or_exit(input_path, host, port, renderer, where)
    lift: dict[tuple[Any, str], str] = {}
    for spec in param or []:
        if "=" not in spec or "." not in spec.split("=", 1)[0]:
            renderer.error(
                code="workflow_edit_invalid",
                message=f"--param must be `<node_id>.<widget>=<param_name>`, got {spec!r}",
            )
            raise typer.Exit(code=1)
        target, _, pname = spec.partition("=")
        node_str, _, widget = target.partition(".")  # node id has no dot; widget may (model.resolution)
        node_id: Any = int(node_str) if node_str.lstrip("-").isdigit() else node_str
        lift[(node_id, widget)] = pname.strip()
    try:
        recipe = workflow_ops.capture_recipe(workflow, graph, name=name or p.stem, lift=lift)
    except workflow_ops.RecipeError as e:
        renderer.error(code="workflow_edit_invalid", message=str(e))
        raise typer.Exit(code=1) from e

    serialized = json.dumps(recipe, indent=2)
    wrote: str | None = None
    if out:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(out_path, serialized)
        wrote = str(out_path)
    elif renderer.is_pretty():
        import sys

        sys.stdout.write(serialized + "\n")
    payload = {
        "recipe": recipe["recipe"],
        "op_count": len(recipe["ops"]),
        "out": wrote or "stdout",
        "recipe_doc": recipe,
    }
    if renderer.is_pretty() and wrote:
        rprint(f"[bold green]✓[/bold green] captured {len(recipe['ops'])} ops → [dim]{wrote}[/dim]")
    renderer.emit(payload, command="workflow capture")


# ---------------------------------------------------------------------------
# apply — batch: one object_info load, many edits, aliases for just-made nodes
# ---------------------------------------------------------------------------


@tracking.track_command("workflow")
def apply_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    ops_file: Annotated[
        str,
        typer.Option("--ops", help="Recipe file (JSON array of ops, or {params, ops}), or '-' for stdin."),
    ],
    param: Annotated[
        list[str] | None,
        typer.Option("--param", show_default=False, help="Recipe param as key=value; repeatable."),
    ] = None,
    actor: Annotated[str, typer.Option("--actor", help="Op author id (for CRDT stamping).")] = "cli",
    base_version: Annotated[int, typer.Option("--base-version", help="Draft version this batch is based on.")] = 0,
    stdout: Annotated[bool, typer.Option("--stdout/--in-place", show_default=False)] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    """Apply a batch of edits in one pass — the catalog loads once, and an
    `add_node` spec may set `"as": "<alias>"` so later specs reference the
    minted node by alias instead of a captured id."""
    renderer = get_renderer()
    renderer.command = "workflow apply"
    p, workflow = _load_workflow_or_fail(renderer, file)
    graph = _graph_or_exit(input_path, host, port, renderer, where)

    if ops_file == "-":
        import sys

        raw = sys.stdin.read()
    else:
        from pathlib import Path

        try:
            raw = Path(ops_file).expanduser().read_text(encoding="utf-8")
        except OSError as e:
            renderer.error(code="workflow_edit_invalid", message=f"cannot read --ops file: {e}")
            raise typer.Exit(code=1) from e
    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as e:
        renderer.error(code="workflow_edit_invalid", message=f"--ops is not valid JSON: {e}")
        raise typer.Exit(code=1) from e

    # A recipe is `{params?, ops}` (or a bare op list); `${param}` holes are filled
    # from --param with strict validation (no silent blanks).
    provided: dict[str, str] = {}
    for kv in param or []:
        if "=" not in kv:
            renderer.error(code="workflow_edit_invalid", message=f"--param must be key=value, got {kv!r}")
            raise typer.Exit(code=1)
        k, _, v = kv.partition("=")
        provided[k.strip()] = v
    try:
        specs, params_decl = workflow_ops.parse_recipe(doc)
        params = workflow_ops.resolve_params(params_decl, provided)
        specs = workflow_ops.substitute_params(specs, params)
    except workflow_ops.RecipeError as e:
        renderer.error(code="workflow_edit_invalid", message=str(e))
        raise typer.Exit(code=1) from e

    try:
        workflow, ops, aliases = workflow_ops.apply_specs(
            workflow, graph, specs, actor=actor, base_version=base_version
        )
    except (ValueError, KeyError) as e:
        # Atomic batch: nothing is written if any spec fails.
        renderer.error(code="workflow_edit_invalid", message=f"batch failed: {e}")
        raise typer.Exit(code=1) from e

    workflow_ops.strip_internal(workflow)
    serialized = json.dumps(workflow, indent=2)
    wrote: str | None = None
    if stdout:
        import sys

        sys.stdout.write(serialized + "\n")
    else:
        _atomic_write_text(p, serialized)
        wrote = str(p)
    payload = {
        "workflow": str(p),
        "count": len(ops),
        "ops": ops,
        "aliases": aliases,
        "base_version": base_version,
        "version": base_version + len(ops),
        "wrote": wrote,
    }
    if renderer.is_pretty():
        rprint(f"[bold green]✓[/bold green] applied {len(ops)} edit(s) → [dim]{p}[/dim]")
    renderer.emit(payload, command="workflow apply", changed=True)


# ---------------------------------------------------------------------------
# foreach — instantiate a recipe over N param-sets → N ready-to-run workflows
# ---------------------------------------------------------------------------


def _load_param_sets(raw: str, renderer) -> list[dict]:
    """Param-sets are a JSON array of objects, a single object, or JSONL."""
    raw = raw.strip()
    try:
        doc = json.loads(raw)
        return doc if isinstance(doc, list) else [doc]
    except json.JSONDecodeError:
        pass
    sets: list[dict] = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            sets.append(json.loads(ln))
        except json.JSONDecodeError as e:
            renderer.error(code="workflow_edit_invalid", message=f"--params line is not JSON: {e}")
            raise typer.Exit(code=1) from e
    return sets


@tracking.track_command("workflow")
def foreach_cmd(
    recipe_file: Annotated[str, typer.Argument(help="Recipe file: {params, ops}.")],
    params_file: Annotated[
        str,
        typer.Option("--params", help="Param-sets: a JSON array of objects, one object, or JSONL; '-' for stdin."),
    ],
    out_dir: Annotated[str, typer.Option("--out-dir", help="Directory to write the N materialized workflows.")],
    actor: Annotated[str, typer.Option("--actor")] = "cli",
    base_version: Annotated[int, typer.Option("--base-version")] = 0,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False, help="Catalog target: local | cloud.")] = None,
):
    """Instantiate a recipe over N param-sets → N ready-to-run workflows (bulk).
    Run them with `comfy run --workflow <each> --where cloud`."""
    import sys
    from pathlib import Path

    renderer = get_renderer()
    renderer.command = "workflow foreach"
    graph = _graph_or_exit(input_path, host, port, renderer, where)
    try:
        doc = json.loads(Path(recipe_file).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        renderer.error(code="workflow_edit_invalid", message=f"cannot read recipe: {e}")
        raise typer.Exit(code=1) from e
    raw = sys.stdin.read() if params_file == "-" else _read_text_or_exit(renderer, params_file)
    param_sets = _load_param_sets(raw, renderer)
    if not param_sets:
        renderer.error(code="workflow_edit_invalid", message="--params yielded no param-sets")
        raise typer.Exit(code=1)

    try:
        specs_template, params_decl = workflow_ops.parse_recipe(doc)
    except workflow_ops.RecipeError as e:
        renderer.error(code="workflow_edit_invalid", message=str(e))
        raise typer.Exit(code=1) from e

    name = (doc.get("recipe") if isinstance(doc, dict) else None) or Path(recipe_file).expanduser().stem
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    try:
        for i, pset in enumerate(param_sets):
            if not isinstance(pset, dict):
                raise workflow_ops.RecipeError(f"param-set #{i} must be a JSON object")
            params = workflow_ops.resolve_params(params_decl, {k: str(v) for k, v in pset.items()})
            specs = workflow_ops.substitute_params(specs_template, params)
            wf: dict = {"nodes": [], "links": [], "last_node_id": 0, "last_link_id": 0}
            wf, _ops, _aliases = workflow_ops.apply_specs(wf, graph, specs, actor=actor, base_version=base_version)
            workflow_ops.strip_internal(wf)
            target = out / f"{name}_{i:03d}.json"
            _atomic_write_text(target, json.dumps(wf, indent=2))
            written.append(str(target))
    except (workflow_ops.RecipeError, ValueError, KeyError) as e:
        renderer.error(code="workflow_edit_invalid", message=f"foreach failed: {e}")
        raise typer.Exit(code=1) from e

    payload = {"recipe": name, "count": len(written), "out_dir": str(out), "written": written}
    if renderer.is_pretty():
        rprint(f"[bold green]✓[/bold green] materialized {len(written)} workflow(s) → [dim]{out}[/dim]")
        rprint("[dim]run each: comfy run --workflow <file> --where cloud[/dim]")
    renderer.emit(payload, command="workflow foreach", changed=bool(written))


def _read_text_or_exit(renderer, path: str) -> str:
    from pathlib import Path

    try:
        return Path(path).expanduser().read_text(encoding="utf-8")
    except OSError as e:
        renderer.error(code="workflow_edit_invalid", message=f"cannot read --params file: {e}")
        raise typer.Exit(code=1) from e
