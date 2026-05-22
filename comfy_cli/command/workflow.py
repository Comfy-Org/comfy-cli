"""``comfy workflow`` — slot-based editing of ComfyUI frontend-format workflows.

Three primitives, each thin over a CQL/comfygraph wasm call:

    comfy workflow slots <file>                        # what can I tweak?
    comfy workflow set-slot <file> ADDR=VALUE [...]    # tweak one or more
    comfy workflow vary <file> --slot ADDR='[v1,v2]'   # produce N variants

Workflows must be **frontend-format** (the regular ComfyUI save — has
``nodes[]`` / ``links[]``, may contain subgraphs). API-format (the export
that ``comfy run`` consumes) is rejected with a clean envelope and a hint.

Slot addresses follow CQL's format: ``<instance_id>.<input_name>``. Run
``slots`` first to discover them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.output import get_renderer, rprint

app = typer.Typer(no_args_is_help=True, help="Slot-based editing of frontend-format ComfyUI workflows.")


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _is_frontend_format(data: Any) -> bool:
    """Heuristic: frontend format has ``nodes`` as a list. API format has it as a dict keyed by IDs."""
    return isinstance(data, dict) and isinstance(data.get("nodes"), list)


def _load_workflow_or_fail(renderer, path: str) -> tuple[Path, dict[str, Any]]:
    """Read + parse + format-check a workflow file. Exit with envelope on any failure."""
    p = Path(path).expanduser()
    if not p.is_file():
        renderer.error(
            code="workflow_not_found",
            message=f"Workflow file not found: {path}",
            hint="check the path",
        )
        raise typer.Exit(code=1)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except OSError as e:
        renderer.error(code="workflow_not_found", message=f"Unable to read workflow file: {e}")
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        renderer.error(
            code="workflow_invalid_json",
            message=f"Workflow file is not valid JSON: {e}",
            hint="check the file or re-export from ComfyUI",
        )
        raise typer.Exit(code=1) from e
    if not _is_frontend_format(data):
        renderer.error(
            code="workflow_not_frontend_format",
            message="`comfy workflow` requires the frontend-format workflow (with `nodes[]` / `links[]`).",
            hint="in ComfyUI, use `File > Save (As)` to export the editing format. "
            "The `File > Export (API)` output is for `comfy run`, not for editing.",
            details={"path": str(p)},
        )
        raise typer.Exit(code=1)
    return p, data


def _load_object_info_or_fail(renderer, input_path: str | None, host: str | None, port: int | None) -> dict[str, Any]:
    """Fetch object_info via the unified loader, with error envelope on failure."""
    from comfy_cli import where as where_module
    from comfy_cli.comfygraph import ComfygraphError, load_object_info
    from comfy_cli.config_manager import ConfigManager

    # Resolve mode from global routing chain
    decision = where_module.resolve(
        flag=None,
        config_value=ConfigManager().get(where_module.CONFIG_KEY_WHERE_DEFAULT),
    )
    mode = "cloud" if decision.target is where_module.WhereTarget.CLOUD else "local"

    try:
        return load_object_info(
            mode=mode,
            input_path=input_path,
            host=host or "127.0.0.1",
            port=port or 8188,
        )
    except ComfygraphError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <path>, start the server with `comfy launch`, or switch to cloud"),
        )
        raise typer.Exit(code=1) from e


def _atomic_write_text(path: Path, content: str) -> None:
    """Write via tmp + rename so SIGINT mid-write can't leave a half-written file."""
    import os

    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def _parse_value(raw: str) -> Any:
    """Parse a CLI-supplied value as JSON; fall back to the literal string."""
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def _split_addr_value(arg: str, renderer) -> tuple[str, Any]:
    """Split ``addr=value`` and parse value as JSON-or-string."""
    if "=" not in arg:
        renderer.error(
            code="workflow_slot_invalid",
            message=f"Expected `ADDR=VALUE`, got {arg!r}",
            hint='example: `positive_prompt.text="a cat"` (run `comfy workflow slots <file>` for addresses)',
        )
        raise typer.Exit(code=1)
    addr, _, raw = arg.partition("=")
    return addr.strip(), _parse_value(raw)


# ---------------------------------------------------------------------------
# slots
# ---------------------------------------------------------------------------


@app.command("slots", help="List the agent-tweakable slots a workflow exposes.")
@tracking.track_command("workflow")
def slots_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a saved object_info JSON (offline)."),
    ] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    template_id: Annotated[
        str,
        typer.Option("--id", show_default=False, help="Template ID label; cosmetic only — defaults to the filename."),
    ] = "",
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    obj_info = _load_object_info_or_fail(renderer, input_path, host, port)

    template_id = template_id or p.stem
    try:
        schema = comfygraph.get_template_schema(template_id, workflow, obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(
            code="workflow_slot_invalid",
            message=f"Could not extract slots: {e}",
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    payload = {
        "workflow": str(p),
        "id": schema.get("id"),
        "count": len(schema.get("slots") or []),
        "slots": schema.get("slots") or [],
    }

    if renderer.is_pretty():
        from rich.table import Table

        slots = payload["slots"]
        if not slots:
            rprint("[dim]No tweakable slots in this workflow.[/dim]")
        else:
            tbl = Table(show_header=True, header_style="bold")
            tbl.add_column("address", no_wrap=True)
            tbl.add_column("type", style="dim", no_wrap=True)
            tbl.add_column("current", style="dim", overflow="fold")
            for s in slots:
                if not isinstance(s, dict):
                    continue
                addr = s.get("address") or s.get("name") or ""
                t = s.get("type") or ""
                val = s.get("current_value")
                tbl.add_row(str(addr), str(t), "" if val is None else str(val)[:80])
            renderer.console().print(tbl)
            rprint(f"[dim]{len(slots)} slot(s) · run `comfy workflow set-slot {p} <addr>=<value>`[/dim]")
    renderer.emit(payload, command="workflow slots")


# ---------------------------------------------------------------------------
# set-slot
# ---------------------------------------------------------------------------


@app.command("set-slot", help="Apply one or more slot overrides to a workflow in place (or --stdout).")
@tracking.track_command("workflow")
def set_slot_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    overrides: Annotated[list[str], typer.Argument(metavar="ADDR=VALUE...", help="One or more ADDR=VALUE pairs.")],
    stdout: Annotated[
        bool,
        typer.Option(
            "--stdout/--in-place",
            show_default=False,
            help="Print the result to stdout instead of writing back to <file>.",
        ),
    ] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    obj_info = _load_object_info_or_fail(renderer, input_path, host, port)

    overrides_dict: dict[str, Any] = {}
    for raw in overrides:
        addr, value = _split_addr_value(raw, renderer)
        overrides_dict[addr] = value

    try:
        result = comfygraph.apply_slots(workflow, overrides_dict, obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(
            code="workflow_slot_invalid",
            message=str(e),
            hint="run `comfy workflow slots <file>` to see valid addresses + types",
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    new_workflow = result.get("workflow")
    warnings = result.get("warnings") or []
    serialized = json.dumps(new_workflow, indent=2)

    if stdout:
        # Direct write to the renderer's stdout stream — bypasses the
        # envelope deliberately so `--stdout` actually pipes the JSON.
        import sys

        sys.stdout.write(serialized)
        sys.stdout.write("\n")
        return

    _atomic_write_text(p, serialized)

    payload = {
        "workflow": str(p),
        "applied": list(overrides_dict.keys()),
        "warnings": warnings,
        "wrote": str(p),
    }
    if renderer.is_pretty():
        rprint(f"[bold green]✓[/bold green] applied {len(overrides_dict)} slot(s) → [dim]{p}[/dim]")
        for addr in overrides_dict:
            rprint(f"  [dim]·[/dim] {addr}")
        for w in warnings:
            rprint(f"  [yellow]warning:[/yellow] {w}")
    renderer.emit(payload, command="workflow set-slot", changed=True)


# ---------------------------------------------------------------------------
# vary
# ---------------------------------------------------------------------------


@app.command("vary", help="Produce N workflow variants from a per-slot value list. Emits NDJSON.")
@tracking.track_command("workflow")
def vary_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
    slot: Annotated[
        list[str],
        typer.Option(
            "--slot",
            help="ADDR='[v1,v2,...]' — repeat per slot. Lists are zipped, so all --slot args must have the same length.",
        ),
    ],
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
    out_dir: Annotated[
        str | None,
        typer.Option(
            "--out-dir",
            show_default=False,
            help="If set, write each variation to <out-dir>/<stem>_<N>.json. Otherwise emit NDJSON to stdout.",
        ),
    ] = None,
):
    from comfy_cli import comfygraph

    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    obj_info = _load_object_info_or_fail(renderer, input_path, host, port)

    # Parse each --slot ADDR='[a,b,c]'. Each value must be a JSON list.
    by_addr: dict[str, list[Any]] = {}
    for raw in slot:
        addr, value = _split_addr_value(raw, renderer)
        if not isinstance(value, list):
            renderer.error(
                code="workflow_variation_invalid",
                message=f"--slot {addr}: value must be a JSON array (got {type(value).__name__}).",
                hint='example: --slot positive_prompt.text=\'["a cat","a dog"]\'',
            )
            raise typer.Exit(code=1)
        by_addr[addr] = value

    if not by_addr:
        renderer.error(
            code="workflow_variation_invalid",
            message="vary needs at least one --slot ADDR='[v1,v2,...]'",
            hint='example: --slot positive_prompt.text=\'["a cat","a dog"]\'',
        )
        raise typer.Exit(code=1)

    lengths = {addr: len(vals) for addr, vals in by_addr.items()}
    n = next(iter(lengths.values()))
    if any(length != n for length in lengths.values()):
        renderer.error(
            code="workflow_variation_invalid",
            message=f"All --slot lists must have the same length (zip semantics). Got: {lengths}",
            hint="trim the lists to a matching length, or call vary multiple times",
        )
        raise typer.Exit(code=1)

    # Zip into N variation dicts.
    variations: list[dict[str, Any]] = [{addr: vals[i] for addr, vals in by_addr.items()} for i in range(n)]

    try:
        result = comfygraph.expand_variations(workflow, variations, obj_info)
    except comfygraph.ComfygraphError as e:
        renderer.error(
            code="workflow_variation_invalid",
            message=str(e),
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    workflows = result.get("variations") or []
    warnings = result.get("warnings") or []

    written: list[str] = []
    if out_dir:
        out = Path(out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for i, wf in enumerate(workflows):
            target = out / f"{p.stem}_{i:03d}.json"
            _atomic_write_text(target, json.dumps(wf, indent=2))
            written.append(str(target))
    else:
        # NDJSON to stdout — one workflow per line.
        import sys

        for wf in workflows:
            sys.stdout.write(json.dumps(wf))
            sys.stdout.write("\n")
        sys.stdout.flush()

    payload = {
        "workflow": str(p),
        "count": len(workflows),
        "variations_summary": [list(v.keys()) for v in variations],
        "warnings": warnings,
        "out_dir": str(Path(out_dir).expanduser()) if out_dir else None,
        "written": written,
    }
    if renderer.is_pretty():
        rprint(f"[bold green]✓[/bold green] produced {len(workflows)} variation(s)")
        if written:
            for path in written[:5]:
                rprint(f"  [dim]→[/dim] {path}")
            if len(written) > 5:
                rprint(f"  [dim]… and {len(written) - 5} more[/dim]")
        for w in warnings:
            rprint(f"  [yellow]warning:[/yellow] {w}")
    renderer.emit(payload, command="workflow vary", changed=bool(written))
