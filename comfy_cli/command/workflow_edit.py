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

_SUBGRAPH_SEP = "/"


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
    if isinstance(node_id, str) and _SUBGRAPH_SEP in node_id:
        # Interior-of-subgraph edits aren't part of the top-level op/CRDT model
        # (subgraph authoring is deferred by design). Route to the proven path.
        renderer.error(
            code="workflow_edit_invalid",
            message=f"nested subgraph address {node_id!r} is not supported by set-widget",
            hint=f"use `comfy workflow set-slot <file> '{addr}=<value>'` for values inside a subgraph",
        )
        raise typer.Exit(code=1)
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
# apply — batch: one object_info load, many edits, aliases for just-made nodes
# ---------------------------------------------------------------------------


def _resolve_ref(ref: Any, aliases: dict[str, Any]) -> Any:
    """Map an alias to its minted id; pass ints/unknown strings through."""
    if isinstance(ref, str):
        if ref in aliases:
            return aliases[ref]
        if ref.lstrip("-").isdigit():
            return int(ref)
    return ref


def _split_ref_slot(spec_val: str, aliases: dict[str, Any]) -> tuple[Any, Any]:
    """Split `<node_or_alias>.<slot>` and resolve the node part."""
    node_part, _, slot = str(spec_val).partition(".")
    return _resolve_ref(node_part, aliases), slot


@tracking.track_command("workflow")
def apply_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    ops_file: Annotated[
        str,
        typer.Option("--ops", help="JSON array of edit specs, or '-' to read from stdin."),
    ],
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
        specs = json.loads(raw)
    except json.JSONDecodeError as e:
        renderer.error(code="workflow_edit_invalid", message=f"--ops is not valid JSON: {e}")
        raise typer.Exit(code=1) from e
    if not isinstance(specs, list):
        renderer.error(code="workflow_edit_invalid", message="--ops must be a JSON array of edit specs")
        raise typer.Exit(code=1)

    aliases: dict[str, Any] = {}
    ops: list[dict] = []
    try:
        for i, spec in enumerate(specs):
            if not isinstance(spec, dict) or "op" not in spec:
                raise ValueError(f"spec #{i} must be an object with an 'op' field")
            kind = spec["op"]
            if kind == "add_node":
                workflow, op = workflow_ops.add_node(
                    workflow, graph, spec["class_type"], pos=spec.get("at"), actor=actor, base_version=base_version
                )
                if spec.get("as"):
                    aliases[spec["as"]] = op["node_id"]
            elif kind == "connect":
                fn, fs = _split_ref_slot(spec["from"], aliases)
                tn, ts = _split_ref_slot(spec["to"], aliases)
                workflow, op = workflow_ops.connect(
                    workflow, graph, fn, fs, tn, ts, actor=actor, base_version=base_version
                )
            elif kind == "set_widget":
                workflow, op = workflow_ops.set_widget(
                    workflow,
                    graph,
                    _resolve_ref(spec["node"], aliases),
                    spec["widget"],
                    spec["value"],
                    actor=actor,
                    base_version=base_version,
                )
            elif kind == "delete_node":
                workflow, op = workflow_ops.delete_node(
                    workflow, graph, _resolve_ref(spec["node"], aliases), actor=actor, base_version=base_version
                )
            else:
                raise ValueError(f"spec #{i}: unknown op {kind!r}")
            ops.append(op)
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
