"""``comfy workflow compose`` and ``comfy workflow fragment`` — the Typer surface.

The composition engine lives in :mod:`comfy_cli.fragments`; this module is the
thin I/O shell that wraps it for the CLI — it renders envelopes and maps the
domain exceptions (``FragmentError`` / ``BlueprintError``) onto error codes.
See ``comfy_cli/fragments.py`` for the fragment/blueprint format and model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.fragments import (
    BlueprintError,
    FragmentError,
    RefResolutionError,
    compose_blueprints,
    decompose_workflow,
    load_fragment,
    parse_fragment,
    resolve_fragment_name,
)
from comfy_cli.output import get_renderer, rprint

# ---------------------------------------------------------------------------
# Typer surface
# ---------------------------------------------------------------------------

fragment_app = typer.Typer(no_args_is_help=True, help="Inspect and validate workflow fragments.")


def _default_lib_dir(override: str | None) -> Path:
    """Resolve ``--lib`` → Path. Default is ``./fragments`` in cwd."""
    if override:
        return Path(override).expanduser()
    return Path.cwd() / "fragments"


@tracking.track_command("workflow")
def compose_cmd(
    blueprint: Annotated[Path, typer.Argument(help="Blueprint YAML file.")],
    out: Annotated[
        Path | None,
        typer.Option(
            "--out", "-o", show_default=False, help="Output workflow JSON path. Defaults to <blueprint>.compiled.json"
        ),
    ] = None,
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Fragment library directory. Defaults to ./fragments"),
    ] = None,
):
    """Compose a YAML blueprint of fragments into a single API-format workflow."""
    renderer = get_renderer()
    if not blueprint.is_file():
        renderer.error(code="blueprint_not_found", message=f"Blueprint file not found: {blueprint}")
        raise typer.Exit(code=1)

    try:
        import yaml
    except ImportError as e:  # pragma: no cover
        renderer.error(
            code="blueprint_yaml_unavailable",
            message="PyYAML is required for `compose`",
            hint="install with: pip install pyyaml",
        )
        raise typer.Exit(code=1) from e

    try:
        blueprint_data = yaml.safe_load(blueprint.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        renderer.error(code="blueprint_invalid_yaml", message=f"Blueprint is not valid YAML: {e}")
        raise typer.Exit(code=1) from e
    except OSError as e:
        renderer.error(code="compose_io_error", message=f"Unable to read blueprint file: {e}")
        raise typer.Exit(code=1) from e

    # `$asset.<name>` refs resolve through the governing project's push lock
    # and `$var.<name>` refs through its comfy.yaml `vars:` block (anchored
    # at the blueprint's dir, like the journal). No project, no resolvers —
    # fragments.py stays project-unaware and errors with the hint. The var
    # resolver is wrapped to record which vars this compilation actually
    # referenced, so `_meta.vars` can snapshot the values used (provenance).
    from comfy_cli.project import find_project, make_asset_resolver, make_var_resolver

    project = find_project(blueprint.resolve().parent)
    asset_resolver = make_asset_resolver(project) if project is not None else None
    var_resolver = None
    used_vars: dict[str, Any] = {}
    if project is not None:
        _resolve_var = make_var_resolver(project)

        def var_resolver(name: str) -> Any:
            used_vars[name] = _resolve_var(name)
            return used_vars[name]

    lib_dir = _default_lib_dir(lib)
    try:
        graphs = compose_blueprints(
            blueprint_data,
            lib_dir=lib_dir,
            blueprint_dir=blueprint.parent,
            asset_resolver=asset_resolver,
            var_resolver=var_resolver,
        )
    except FragmentError as e:
        renderer.error(code="fragment_invalid", message=str(e), hint=e.hint or "", details={"path": e.path})
        raise typer.Exit(code=1) from e
    except RefResolutionError as e:
        # Before BlueprintError — AssetError/VarError subclass it and carry their own code.
        renderer.error(code=e.code, message=str(e), hint=e.hint or "")
        raise typer.Exit(code=1) from e
    except BlueprintError as e:
        renderer.error(
            code="blueprint_invalid", message=str(e), hint=e.hint or "", details={"step_alias": e.step_alias}
        )
        raise typer.Exit(code=1) from e

    base_out = out or blueprint.with_suffix(".compiled.json")
    try:
        base_out.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        renderer.error(code="compose_io_error", message=f"Failed preparing output path: {e}")
        raise typer.Exit(code=1) from e

    # Provenance block embedded in every written workflow (`compose/1`). The
    # one artifact that travels (the compiled JSON) carries which blueprint
    # produced it and — for foreach — which item produced which nodes.
    # `comfy run` strips `_meta` before submitting (see run/loader.py).
    blueprint_abs = str(blueprint.resolve())

    def _meta_block(summary: dict) -> dict:
        meta: dict = {"schema": "compose/1", "blueprint": blueprint_abs}
        if summary.get("item_map"):
            meta["items"] = summary["item_map"]
        # Vars this compilation referenced, with the values used — provenance,
        # same spirit as `items`. Only referenced names; raw scalars.
        if used_vars:
            meta["vars"] = dict(used_vars)
        return meta

    # A single graph keeps the simple `<out>` name; `chunk:` fan-out writes one
    # numbered file per graph (`<stem>.000.json`, `<stem>.001.json`, ...).
    written: list[str] = []
    try:
        if len(graphs) == 1:
            workflow, summary = graphs[0]
            workflow["_meta"] = _meta_block(summary)
            base_out.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
            written.append(str(base_out))
            single_out: str | None = str(base_out)
        else:
            # Chunked fan-out: numbered files only. Remove any stale unnumbered file
            # from a prior single-graph compose so `comfy run <out>` can't execute it.
            if base_out.exists():
                base_out.unlink()
            for i, (workflow, summary) in enumerate(graphs):
                workflow["_meta"] = _meta_block(summary)  # each file: only ITS batch's items
                target = base_out.with_suffix(f".{i:03d}{base_out.suffix}")
                target.write_text(json.dumps(workflow, indent=2), encoding="utf-8")
                written.append(str(target))
            single_out = None  # no single runnable file; consumers must read `written`
    except OSError as e:
        renderer.error(code="compose_io_error", message=f"Failed writing composed workflow: {e}")
        raise typer.Exit(code=1) from e

    # Provenance journal: one line into the governing project, if any
    # (anchored at the blueprint's dir, not cwd). Best-effort by contract.
    _journal_compose(blueprint, written)

    first_summary = graphs[0][1]
    total_nodes = sum(s["nodes"] for _, s in graphs)
    fragments_used = sorted({f for _, s in graphs for f in s["fragments_used"]})
    payload = {
        "blueprint": str(blueprint),
        "out": single_out,
        "graphs": len(graphs),
        "written": written,
        "steps": first_summary["steps"],
        "nodes": total_nodes,
        "fragments_used": fragments_used,
    }
    if "total_items" in first_summary:
        payload["items"] = first_summary["total_items"]
    # Union of every graph's per-item provenance (keys are unique across
    # batches, so a plain merge is lossless).
    item_map = {k: v for _, s in graphs for k, v in (s.get("item_map") or {}).items()}
    if item_map:
        payload["item_map"] = item_map
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] composed [bold]{len(graphs)} graph(s)[/bold]")
        for path in written:
            rprint(f"  [dim]→[/dim] {path}")
        if "total_items" in first_summary:
            rprint(f"  items     : {first_summary['total_items']}")
        rprint(f"  steps     : {first_summary['steps']}")
        rprint(f"  nodes     : {total_nodes}")
        rprint(f"  fragments : {', '.join(fragments_used)}")
    renderer.emit(payload, command="workflow compose")


def _journal_compose(blueprint: Path, written: list[str]) -> None:
    """Append the compose event to the governing project's run journal.
    Wrapped end-to-end: a journaling failure can never fail the compose."""
    try:
        from comfy_cli import project as project_module

        p = project_module.find_project(blueprint.resolve().parent)
        if p is not None:
            project_module.journal(p, cmd="compose", blueprint=str(blueprint), written=list(written))
    except Exception:  # noqa: BLE001 — best-effort by contract
        pass


@tracking.track_command("workflow")
def decompose_cmd(
    workflow: Annotated[Path, typer.Argument(help="Workflow JSON to project — API format, or frontend (UI) format.")],
    name: Annotated[
        str | None,
        typer.Option("--name", show_default=False, help="Fragment name. Defaults to the workflow file stem."),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", "-o", show_default=False, help="Output path. Defaults to <lib>/<name>.json"),
    ] = None,
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Fragment library directory. Defaults to ./fragments"),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option(
            "--input", show_default=False, help="Offline object_info.json (needed to convert frontend format)."
        ),
    ] = None,
    host: Annotated[str | None, typer.Option("--host", show_default=False, help="Comfy server host.")] = None,
    port: Annotated[int | None, typer.Option("--port", show_default=False, help="Comfy server port.")] = None,
):
    """Project a workflow into a reusable fragment (the inverse of `compose`).

    Loaders become typed inputs, the terminal save's producer an output, and
    every scalar widget a named param — so a blueprint overrides values by name
    instead of editing the compiled graph by hand. Frontend-format workflows are
    flattened to API format first (subgraphs expanded), which needs object_info
    from a running/cloud server or `--input <object_info.json>`.
    """
    renderer = get_renderer()
    if not workflow.is_file():
        renderer.error(code="workflow_not_found", message=f"Workflow file not found: {workflow}", hint="check the path")
        raise typer.Exit(code=1)
    try:
        data = json.loads(workflow.read_text(encoding="utf-8"))
    except OSError as e:
        renderer.error(code="workflow_not_found", message=f"Unable to read workflow file: {e}")
        raise typer.Exit(code=1) from e
    except json.JSONDecodeError as e:
        renderer.error(code="workflow_invalid_json", message=f"Workflow file is not valid JSON: {e}")
        raise typer.Exit(code=1) from e

    from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api, is_api_format

    # Frontend (UI) format needs object_info to flatten subgraphs + name widgets.
    if not is_api_format(data):
        object_info = _load_object_info(renderer, input_path=input_path, host=host, port=port)
        try:
            api_workflow = convert_ui_to_api(data, object_info)
        except WorkflowConversionError as e:
            renderer.error(
                code="workflow_conversion_failed",
                message=f"Could not convert frontend workflow to API format: {e}",
                hint="re-export from ComfyUI, or pass a matching --input object_info.json",
            )
            raise typer.Exit(code=1) from e
    else:
        api_workflow = data

    frag_name = name or _slug_name(workflow.stem)
    frag_json = decompose_workflow(api_workflow, name=frag_name, source=str(workflow))
    try:
        frag = parse_fragment(frag_json, source_path=str(workflow))
    except FragmentError as e:  # pragma: no cover — projection always emits valid fragments
        renderer.error(code="fragment_invalid", message=str(e), hint=e.hint or "", details={"path": str(workflow)})
        raise typer.Exit(code=1) from e

    target = out or (_default_lib_dir(lib) / f"{frag_name}.json")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(frag_json, indent=2), encoding="utf-8")
    except OSError as e:
        renderer.error(code="decompose_io_error", message=f"Failed writing fragment: {e}")
        raise typer.Exit(code=1) from e

    payload = {
        "workflow": str(workflow),
        "out": str(target),
        "name": frag.name,
        "node_count": len(frag.nodes),
        "ports": {"inputs": len(frag.inputs), "outputs": len(frag.outputs), "params": len(frag.params)},
        "inputs": sorted(frag.inputs.keys()),
        "outputs": sorted(frag.outputs.keys()),
        "params": sorted(frag.params.keys()),
    }
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] fragment [bold]{frag.name}[/bold] → {target}")
        rprint(f"  nodes   : {len(frag.nodes)}")
        rprint(f"  inputs  : {', '.join(sorted(frag.inputs)) or '—'}")
        rprint(f"  outputs : {', '.join(sorted(frag.outputs)) or '—'}")
        rprint(f"  params  : {len(frag.params)}")
    renderer.emit(payload, command="workflow decompose")


def _slug_name(text: str) -> str:
    """Filesystem stem → a safe bare fragment name."""
    import re

    s = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return s or "fragment"


def _load_object_info(renderer, *, input_path: str | None, host: str | None, port: int | None) -> dict:
    """Fetch raw object_info for frontend→API conversion (offline dump or live server)."""
    try:
        if input_path is not None:
            p = Path(input_path).expanduser()
            return json.loads(p.read_text(encoding="utf-8"))
        from comfy_cli import where as where_module
        from comfy_cli.cql.loader import resilient_load_object_info

        decision = where_module.resolve_default()
        mode = "cloud" if decision.target is where_module.WhereTarget.CLOUD else "local"
        return resilient_load_object_info(mode=mode, host=host or "127.0.0.1", port=port or 8188)
    except (OSError, json.JSONDecodeError) as e:
        renderer.error(
            code="object_info_unavailable",
            message=f"Could not load object_info: {e}",
            hint="pass --input <object_info.json>, sign into cloud, or start the server with `comfy launch`",
        )
        raise typer.Exit(code=1) from e


@fragment_app.command("ls", help="List fragments in a library directory.")
@tracking.track_command("workflow")
def fragment_ls_cmd(
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Library dir. Defaults to ./fragments"),
    ] = None,
):
    renderer = get_renderer()
    lib_dir = _default_lib_dir(lib)
    if not lib_dir.is_dir():
        renderer.error(
            code="fragment_lib_not_found",
            message=f"Fragment library directory not found: {lib_dir}",
            hint="create ./fragments/ or pass --lib <dir>",
        )
        raise typer.Exit(code=1)

    rows: list[dict] = []
    errors: list[dict] = []
    for path in sorted(lib_dir.glob("*.json")):
        try:
            frag = load_fragment(path)
        except FragmentError as e:
            errors.append({"path": str(path), "error": str(e)})
            continue
        rows.append(
            {
                "name": frag.name,
                "version": frag.version,
                "description": frag.description,
                "inputs": list(frag.inputs.keys()),
                "outputs": list(frag.outputs.keys()),
                "params": list(frag.params.keys()),
                "terminal": frag.terminal,
                "path": str(path),
            }
        )

    payload = {"lib": str(lib_dir), "count": len(rows), "fragments": rows, "errors": errors}
    if renderer.is_pretty():
        if not rows and not errors:
            rprint("[dim]No fragments found.[/dim]")
        for f in rows:
            rprint(
                f"[bold]{f['name']}[/bold]  v{f['version']}  "
                f"in={','.join(f['inputs']) or '∅'}  "
                f"out={','.join(f['outputs']) or '∅'}  "
                f"params={','.join(f['params']) or '∅'}"
            )
            if f["description"]:
                rprint(f"  [dim]{f['description']}[/dim]")
        for e in errors:
            rprint(f"[red]✗ {e['path']}: {e['error']}[/red]")
    renderer.emit(payload, command="workflow fragment ls")


@fragment_app.command("show", help="Show a fragment's metadata, ports, and interior node count.")
@tracking.track_command("workflow")
def fragment_show_cmd(
    fragment: Annotated[str, typer.Argument(help="Fragment name (looked up in --lib) or path to .json.")],
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Library dir. Defaults to ./fragments"),
    ] = None,
):
    renderer = get_renderer()
    lib_dir = _default_lib_dir(lib)
    path = resolve_fragment_name(fragment, lib_dir)
    try:
        frag = load_fragment(path)
    except FragmentError as e:
        renderer.error(code="fragment_invalid", message=str(e), hint=e.hint or "", details={"path": e.path})
        raise typer.Exit(code=1) from e

    payload = {
        "path": str(path),
        "name": frag.name,
        "version": frag.version,
        "description": frag.description,
        "terminal": frag.terminal,
        "inputs": {n: {"type": p.type, "binds": p.binds} for n, p in frag.inputs.items()},
        "outputs": {n: {"type": p.type, "from": p.from_node, "port": p.port} for n, p in frag.outputs.items()},
        "params": {
            n: {"type": p.type, "binds": p.binds, **({"default": p.default} if p.has_default else {})}
            for n, p in frag.params.items()
        },
        "node_count": len(frag.nodes),
    }
    if renderer.is_pretty():
        rprint(f"[bold]{frag.name}[/bold]  v{frag.version}")
        if frag.description:
            rprint(f"  [dim]{frag.description}[/dim]")
        rprint(f"  terminal: {frag.terminal}  |  interior nodes: {len(frag.nodes)}")
        if frag.inputs:
            rprint("  [bold]inputs[/bold]")
            for n, p in frag.inputs.items():
                rprint(f"    {n}: {p.type}  → {p.binds}")
        if frag.outputs:
            rprint("  [bold]outputs[/bold]")
            for n, p in frag.outputs.items():
                rprint(f"    {n}: {p.type}  ← {p.from_node}[{p.port}]")
        if frag.params:
            rprint("  [bold]params[/bold]")
            for n, p in frag.params.items():
                d = f"  (default={p.default!r})" if p.has_default else ""
                rprint(f"    {n}: {p.type}  → {p.binds}{d}")
    renderer.emit(payload, command="workflow fragment show")


@fragment_app.command("validate", help="Validate that a fragment file is well-formed.")
@tracking.track_command("workflow")
def fragment_validate_cmd(
    fragment: Annotated[str, typer.Argument(help="Fragment name (looked up in --lib) or path to .json.")],
    lib: Annotated[
        str | None,
        typer.Option("--lib", show_default=False, help="Library dir. Defaults to ./fragments"),
    ] = None,
):
    renderer = get_renderer()
    lib_dir = _default_lib_dir(lib)
    path = resolve_fragment_name(fragment, lib_dir)
    try:
        frag = load_fragment(path)
    except FragmentError as e:
        renderer.error(
            code="fragment_invalid",
            message=str(e),
            hint=e.hint or "",
            details={"path": str(path)},
        )
        raise typer.Exit(code=1) from e

    payload = {
        "path": str(path),
        "valid": True,
        "name": frag.name,
        "node_count": len(frag.nodes),
        "ports": {"inputs": len(frag.inputs), "outputs": len(frag.outputs), "params": len(frag.params)},
    }
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] {path}  ({frag.name} v{frag.version}, {len(frag.nodes)} nodes)")
    renderer.emit(payload, command="workflow fragment validate")
