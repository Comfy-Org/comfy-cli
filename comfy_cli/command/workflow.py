"""``comfy workflow`` — slot-based editing of ComfyUI frontend-format workflows.

Three primitives:

    comfy workflow slots <file>                        # what can I tweak?
    comfy workflow set-slot <file> ADDR=VALUE [...]    # tweak one or more
    comfy workflow vary <file> --slot ADDR='[v1,v2]'   # produce N variants

Plus one read-only reader that needs no object_info at all:

    comfy workflow notes <file>                        # what did the author write?

Workflows must be **frontend-format** (the regular ComfyUI save — has
``nodes[]`` / ``links[]``, may contain subgraphs). API-format (the export
that ``comfy run`` consumes) is rejected with a clean envelope and a hint.

Slot addresses follow CQL's format: ``<instance_id>.<input_name>``. Run
``slots`` first to discover them.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Annotated, Any

import typer

from comfy_cli import tracking
from comfy_cli.file_utils import atomic_write_text

# Aliased at module scope rather than lazy-imported: a class used in ``except``
# clauses at module scope cannot be resolved lazily. ``comfy_cli.http`` is
# stdlib-only and tiny, and ``search.py``/``jobs.py`` already pull
# ``urllib.request`` in at import time, so the precedent exists.
from comfy_cli.http import ResponseTooLarge as _ResponseTooLarge
from comfy_cli.output import get_renderer, rprint
from comfy_cli.output.sanitize import sanitize_markup

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


def _get_graph(input_path: str | None, host: str | None, port: int | None, on_stale=None, where: str | None = None):
    """Build a Graph from the resolved object_info source.

    The live (non-``--input``) fetch goes through ``resilient_load_object_info``,
    which auto-caches successful fetches, retries once after a session refresh,
    and falls back to the last cached dump (with a stderr warning) when the
    server/session is briefly unreachable.

    ``on_stale``, if provided, is fired when a stale-cache fallback occurs:
    ``on_stale(host_key, error_str)``.
    """
    from comfy_cli.cql.engine import Graph, LoadError

    renderer = get_renderer()
    try:
        if input_path is not None:
            # Explicit offline dump — Graph.load reads + annotates it.
            return Graph.load(input_path=input_path, host=host, port=port)
        # Live fetch: resolve mode from global routing chain, then use resilient loader.
        from comfy_cli import where as where_module

        # Honor an explicit --where (threaded from the agent edit commands).
        # ``resolve_default_or_exit`` is main's shared wrapper and emits exactly
        # the ``where_invalid`` envelope this hand-rolled block used to build,
        # and it takes the flag — so the branch keeps its --where threading and
        # drops the duplicate error handling.
        decision = where_module.resolve_default_or_exit(where)
        mode = "cloud" if decision.target is where_module.WhereTarget.CLOUD else "local"
        # Routing resolved — stamp it so the `cql_no_graph` envelope below names
        # the catalog these verbs annotated against, matching what `nodes` does
        # from its own `_get_graph`. The `where_invalid` raised just above stays
        # `where: null`: it *is* the failed decision.
        renderer.where = mode
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
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <path>, or start the server with `comfy launch`"),
        )
        raise typer.Exit(code=1) from e


def _atomic_write_text(path: Path, content: str) -> None:
    """Write via tmp + rename so SIGINT mid-write can't leave a half-written file.

    Branch-local helper: main removed the two call sites this served when it
    reworked those paths, which took the definition with it, but the CRDT edit
    commands in ``workflow_edit`` still write drafts through it.
    """
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
            hint='example: `6.text="a cat"` — run `comfy workflow slots <file>` first to list real addresses (`<node_id>.<input>`)',
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
    select: Annotated[
        str | None,
        typer.Option(
            "--select",
            show_default=False,
            help="Project the payload: dot path (slots.0.address), wildcard (slots.#.address), comma multi-select.",
        ),
    ] = None,
):
    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    _stale: dict = {}
    graph = _get_graph(
        input_path, host, port, on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err)
    )

    template_id = template_id or p.stem
    try:
        schema = graph.get_template_schema(template_id, workflow)
    except (ValueError, KeyError) as e:
        renderer.error(code="workflow_slot_invalid", message=f"Could not extract slots: {e}")
        raise typer.Exit(code=1) from e

    payload = {
        "workflow": str(p),
        "id": schema.get("id"),
        "count": len(schema.get("slots") or []),
        "slots": schema.get("slots") or [],
    }

    if _stale:
        payload["stale"] = True
        payload["warnings"] = [
            {"code": "object_info_stale", "message": f"served from cache ({_stale['source']}): {_stale['reason']}"}
        ]

    if select is not None:
        from comfy_cli.selector import emit_selected

        return emit_selected(renderer, payload, select, command="workflow slots")

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
            help="Return the result instead of writing back to <file>: `data.workflow_json` in the "
            "envelope under --json, or the raw workflow on stdout with --no-json. Redirecting "
            "stdout selects JSON mode, so `--stdout > new.json` needs --no-json to get a raw workflow.",
        ),
    ] = False,
    input_path: Annotated[str | None, typer.Option("--input", show_default=False)] = None,
    host: Annotated[str | None, typer.Option(show_default=False)] = None,
    port: Annotated[int | None, typer.Option(show_default=False)] = None,
):
    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    _stale: dict = {}
    graph = _get_graph(
        input_path, host, port, on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err)
    )

    overrides_dict: dict[str, Any] = {}
    for raw in overrides:
        addr, value = _split_addr_value(raw, renderer)
        overrides_dict[addr] = value

    try:
        new_workflow, warnings = graph.apply_slots(workflow, overrides_dict)
    except ValueError as e:
        renderer.error(
            code="workflow_slot_invalid",
            message=str(e),
            hint="run `comfy workflow slots <file>` to see valid addresses + types",
        )
        raise typer.Exit(code=1) from e

    # Fold the stale-cache note into `warnings` up front so every exit path —
    # including the `--stdout` early return below — reports it.
    if _stale:
        warnings = list(warnings) + [
            {"code": "object_info_stale", "message": f"served from cache ({_stale['source']}): {_stale['reason']}"}
        ]

    # `--stdout` in human mode is a pipe target: print the raw workflow so
    # `comfy workflow set-slot ... --stdout --no-json > new.json` keeps working.
    # `--no-json` is required there: a redirect makes stdout a non-TTY, which
    # `Renderer.resolve` reads as JSON mode. In JSON mode stdout is reserved
    # for the envelope (see docs/json-output.md), so the modified workflow
    # rides in `data.workflow_json` instead — a bare workflow object is not an
    # `envelope/1` and machine callers reject it.
    if stdout and renderer.is_pretty():
        import sys

        sys.stdout.write(json.dumps(new_workflow, indent=2))
        sys.stdout.write("\n")
        sys.stdout.flush()
        # stdout now holds exactly the workflow; warnings would corrupt it, so
        # they go to stderr rather than being dropped.
        for w in warnings:
            renderer.stderr_console().print(f"[yellow]warning:[/yellow] {w}")
        return

    if not stdout:
        atomic_write_text(p, json.dumps(new_workflow, indent=2))

    payload: dict[str, Any] = {
        "workflow": str(p),
        "applied": list(overrides_dict.keys()),
        "warnings": warnings,
        "wrote": None if stdout else str(p),
    }
    if stdout:
        payload["out"] = "stdout"
        payload["workflow_json"] = new_workflow
    if _stale:
        payload["stale"] = True
    if renderer.is_pretty():
        rprint(f"[bold green]✓[/bold green] applied {len(overrides_dict)} slot(s) → [dim]{p}[/dim]")
        for addr in overrides_dict:
            rprint(f"  [dim]·[/dim] {addr}")
        for w in warnings:
            rprint(f"  [yellow]warning:[/yellow] {w}")
    renderer.emit(payload, command="workflow set-slot", changed=not stdout)


# ---------------------------------------------------------------------------
# vary
# ---------------------------------------------------------------------------


@app.command(
    "vary",
    help="Produce N workflow variants from a per-slot value list. Emits NDJSON (or `data.variants` under --json).",
)
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
            help="If set, write each variation to <out-dir>/<stem>_<N>.json. Otherwise return the "
            "variants: `data.variants` in the envelope under --json, or NDJSON on stdout with "
            "--no-json. Redirecting stdout selects JSON mode, so `> out.ndjson` needs --no-json.",
        ),
    ] = None,
):
    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    _stale: dict = {}
    graph = _get_graph(
        input_path, host, port, on_stale=lambda key, err: _stale.update(stale=True, source=key, reason=err)
    )

    # Parse each --slot ADDR='[a,b,c]'. Each value must be a JSON list.
    by_addr: dict[str, list[Any]] = {}
    for raw in slot:
        addr, value = _split_addr_value(raw, renderer)
        if not isinstance(value, list):
            renderer.error(
                code="workflow_slot_invalid",
                message=f"--slot {addr}: value must be a JSON array (got {type(value).__name__}).",
                hint='example: --slot \'6.text=["a cat","a dog"]\' — run `comfy workflow slots <file>` first to list real addresses (`<node_id>.<input>`)',
            )
            raise typer.Exit(code=1)
        by_addr[addr] = value

    if not by_addr:
        renderer.error(code="workflow_slot_invalid", message="vary needs at least one --slot")
        raise typer.Exit(code=1)

    lengths = {addr: len(vals) for addr, vals in by_addr.items()}
    n = next(iter(lengths.values()))
    if any(length != n for length in lengths.values()):
        renderer.error(
            code="workflow_slot_invalid",
            message=f"All --slot lists must have the same length. Got: {lengths}",
        )
        raise typer.Exit(code=1)

    variations = [{addr: vals[i] for addr, vals in by_addr.items()} for i in range(n)]

    try:
        workflows, warnings = graph.expand_variations(workflow, variations)
    except ValueError as e:
        renderer.error(code="workflow_slot_invalid", message=str(e))
        raise typer.Exit(code=1) from e

    if _stale:
        warnings = list(warnings) + [
            {"code": "object_info_stale", "message": f"served from cache ({_stale['source']}): {_stale['reason']}"}
        ]

    written: list[str] = []
    # Same envelope contract as set-slot --stdout: raw NDJSON on stdout is the
    # human/pipe form; in JSON mode stdout belongs to the envelope, so the
    # variants ride in `data.variants` instead.
    variants: list[dict[str, Any]] | None = None
    # True once stdout carries the NDJSON stream — the human summary below must
    # then go to stderr, or `comfy workflow vary ... --no-json > out.ndjson`
    # ends with a non-JSON line and breaks strict line-delimited consumers.
    piped_ndjson = False
    if out_dir:
        out = Path(out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for i, wf in enumerate(workflows):
            target = out / f"{p.stem}_{i:03d}.json"
            atomic_write_text(target, json.dumps(wf, indent=2))
            written.append(str(target))
    elif renderer.is_pretty():
        import sys

        for wf in workflows:
            sys.stdout.write(json.dumps(wf))
            sys.stdout.write("\n")
        sys.stdout.flush()
        piped_ndjson = True
    else:
        variants = list(workflows)

    payload: dict[str, Any] = {
        "workflow": str(p),
        "count": len(workflows),
        "warnings": warnings,
        "out_dir": str(Path(out_dir).expanduser()) if out_dir else None,
        "written": written,
        "variants": variants,
    }
    if _stale:
        payload["stale"] = True
    if renderer.is_pretty():
        say = renderer.stderr_console().print if piped_ndjson else rprint
        say(f"[bold green]✓[/bold green] produced {len(workflows)} variation(s)")
        if written:
            for path in written[:5]:
                say(f"  [dim]→[/dim] {path}")
            if len(written) > 5:
                say(f"  [dim]… and {len(written) - 5} more[/dim]")
        for w in warnings:
            say(f"  [yellow]warning:[/yellow] {w}")
    renderer.emit(payload, command="workflow vary", changed=bool(written))


# ---------------------------------------------------------------------------
# notes
# ---------------------------------------------------------------------------

# The two documentation-note types the ComfyUI frontend registers
# (``src/extensions/core/noteNode.ts``). Both are UI-only virtual nodes — they
# carry no schema and are stripped by API conversion (see
# ``workflow_to_api._UI_ONLY_NODE_TYPES``), so reading them is pure JSON
# parsing: no object_info, no running server.
_NOTE_NODE_TYPES = frozenset({"Note", "MarkdownNote"})


def _str_or_none(value: Any) -> str | None:
    """Coerce an untrusted workflow value to the schema's ``string | null``.

    ``notes[].title`` and ``notes[].subgraph.name`` are declared ``string|null``
    in ``schemas/workflow.json``, but a hand-edited file can carry any JSON type
    there. Stringify rather than drop, so the value still reaches the caller.
    """
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _extract_notes(workflow: dict) -> list[dict]:
    """Collect Note/MarkdownNote nodes from the top-level graph and subgraph defs.

    Note text is serialized at ``widgets_values[0]`` (the sole widget both note
    types register — see ComfyUI_frontend ``src/extensions/core/noteNode.ts``).

    Every container is type-checked before it is walked: ``_is_frontend_format``
    only vouches for the top-level ``nodes`` list, so ``definitions.subgraphs``,
    a subgraph's own ``nodes``, and each node's ``type`` are all attacker-shaped
    for a malformed-but-parseable file. A wrong type there must degrade to "no
    notes found", never to an uncaught ``TypeError``.
    """
    out: list[dict] = []

    def _scan(nodes, subgraph):
        if not isinstance(nodes, list):
            return
        for n in nodes:
            if not isinstance(n, dict):
                continue
            # Guard the type before the frozenset membership test: an unhashable
            # value (list/dict) would raise TypeError rather than miss the set.
            node_type = n.get("type")
            if not isinstance(node_type, str) or node_type not in _NOTE_NODE_TYPES:
                continue
            wv = n.get("widgets_values")
            text = wv[0] if isinstance(wv, list) and wv and isinstance(wv[0], str) else ""
            out.append(
                {
                    "id": n.get("id"),
                    "type": node_type,
                    "title": _str_or_none(n.get("title")),
                    "text": text,
                    "pos": n.get("pos"),
                    "size": n.get("size"),
                    "subgraph": subgraph,
                }
            )

    _scan(workflow.get("nodes"), None)
    definitions = workflow.get("definitions")
    subgraphs = definitions.get("subgraphs") if isinstance(definitions, dict) else None
    if isinstance(subgraphs, list):
        for sg in subgraphs:
            if isinstance(sg, dict):
                _scan(sg.get("nodes"), {"id": sg.get("id"), "name": _str_or_none(sg.get("name"))})
    return out


def _safe_console_text(value: Any) -> str:
    """Render an untrusted workflow-file value safely for the pretty console.

    Strips terminal control sequences (so note text can't emit ANSI/OSC escapes)
    and then neutralizes rich markup (so a literal ``[/]`` in a note renders as
    itself instead of raising ``MarkupError`` or spoofing styled output).
    """
    from rich.markup import escape

    return escape(_strip_terminal_controls(str(value)))


@app.command("notes", help="List the documentation notes (Note/MarkdownNote nodes) a workflow carries.")
@tracking.track_command("workflow")
def notes_cmd(
    file: Annotated[str, typer.Argument(help="Frontend-format workflow JSON.")],
):
    """Read the authored notes out of a workflow — offline, read-only.

    Notes are where template authors put the human-facing documentation an
    agent otherwise has to ``grep`` the raw JSON for (LoRA trigger words, model
    download links, usage caveats). API-format files are rejected: the
    conversion drops note nodes entirely, so there is nothing to read there.
    """
    renderer = get_renderer()
    p, workflow = _load_workflow_or_fail(renderer, file)
    notes = _extract_notes(workflow)
    payload = {"workflow": str(p), "count": len(notes), "notes": notes}
    if renderer.is_pretty():
        if not notes:
            rprint("[dim]No notes in this workflow.[/dim]")
        else:
            # Every field interpolated below is untrusted file content, so it
            # goes through _safe_console_text().
            for n in notes:
                sg = n["subgraph"]
                # A note may carry no id, and a subgraph def no name/id — omit the
                # fragment rather than printing a literal "#None" / "(subgraph None)".
                sg_label = (sg.get("name") or sg.get("id")) if sg else None
                where = f" (subgraph {_safe_console_text(sg_label)})" if sg_label is not None else ""
                id_frag = f"#{_safe_console_text(n['id'])}" if n["id"] is not None else ""
                heading = _safe_console_text(n["title"] or n["type"] or "Note")
                meta = f" [dim]{id_frag}{where}[/dim]" if (id_frag or where) else ""
                rprint(f"[bold]{heading}[/bold]{meta}")
                rprint(_safe_console_text(n["text"]))
                rprint()
    renderer.emit(payload, command="workflow notes")


# ---------------------------------------------------------------------------
# Saved workflows — list, get, save, delete.
# ---------------------------------------------------------------------------
#
# These four subcommands route through ``--where``:
#
#   cloud  → Comfy Cloud's ``/api/workflows`` store (UUID-keyed, versioned).
#   local  → the running ComfyUI's ``/userdata`` file store, under the same
#            ``workflows/`` dir the ComfyUI frontend uses. A workflow's id on
#            the local path is its path *relative to* ``workflows/`` (e.g.
#            ``flux.json`` or ``sub/dir/flux.json``) — that same string is what
#            ``get``/``delete`` take and what ``save`` returns.
#
# The two paths share the ``--json`` envelope shape as far as feasible; the
# per-verb docstrings + PR note the deltas (local has no versioning, no
# server-side description, and reports raw file ``size``/``modified``/``created``
# epoch-ms timestamps instead of cloud's ISO ``created_at``/``updated_at``).


# The userdata subdirectory the ComfyUI frontend stores saved workflows in.
_WORKFLOWS_DIR = "workflows"

# Cap on a single ``/userdata`` response we buffer into memory. We read one byte
# past the cap so we can *detect* truncation and fail loudly, rather than
# silently writing a partial workflow and reporting success.
_USERDATA_MAX_BYTES = 64 * 1024 * 1024

# Same cap for a single cloud API response. Kept separate from
# ``_USERDATA_MAX_BYTES`` so the two surfaces can diverge without surprise.
_HTTP_MAX_BYTES = 64 * 1024 * 1024


# Per-operation guidance for an oversize cloud response. ``save``/``delete``
# have already sent their request by the time the response is read, so the
# server-side write may well have landed — say so rather than implying it did not.
_TOO_LARGE_HINTS = {
    "list": "narrow the result set with `--limit` or `--name`",
    "get": "the saved workflow is unexpectedly large; inspect it directly in the cloud UI",
    "save": "the workflow may still have been saved; confirm with `comfy --json workflow list`",
    "delete": "the workflow may still have been deleted; confirm with `comfy --json workflow list`",
}


class _ResponseUnparseable(Exception):
    """A non-empty 200 body could not be decoded as JSON — surface it as a loud
    error instead of a misleading empty/success result (an empty body is still a
    legitimate ``None`` and is *not* this)."""


# What an oversize ``/userdata`` response means depends on the verb, so the hint
# has to as well. ``list`` fetches the whole ``workflows/`` listing, so oversize
# means *too many* workflows — not one big one. ``save``/``delete`` have already
# sent their request by the time the body is read, so the write may well have
# landed: their hints must not imply a clean failure the user should retry.
_LOCAL_TOO_LARGE_HINTS = {
    "list": "too many saved workflows on the server; prune the ComfyUI `workflows/` userdata directory "
    "(`--limit`/`--name` filter client-side, so they cannot shrink the response)",
    "get": "the saved workflow is unexpectedly large; inspect it directly on the server",
    "save": "the workflow may still have been saved; confirm with `comfy --json --where local workflow list`",
    "delete": "the workflow may still have been deleted; confirm with `comfy --json --where local workflow list`",
}


# Map the cloud ``--sort`` fields onto local FileInfo keys (client-side sort;
# ComfyUI's /userdata listing has no server-side sort/limit/filter).
_LOCAL_SORT_KEYS = {"create_time": "created", "update_time": "modified", "name": "path"}


def _resolve_where_target(where: str | None):
    """Resolve the routing Target for a saved-workflow verb (cloud or local).

    This is the single point where ``workflow list/get/save/delete`` decide
    local-vs-cloud, so it is also where the routed target gets stamped on the
    renderer: every error envelope emitted downstream then carries ``where``
    instead of ``null``. Explicit ``emit(..., where=...)`` arguments still win
    (they resolve as ``where or self.where``), so the success envelopes are
    unchanged.
    """
    from comfy_cli.target import resolve_target

    target = resolve_target(where=where)
    get_renderer().where = target.kind
    return target


# Unicode categories that survive the C0/C1 filter but still let untrusted text
# lie about what it says on a terminal. Matching by category rather than by a
# hand-rolled codepoint list keeps this exhaustive as Unicode grows:
#   Cf  format characters — zero-width space/non-joiner/joiner, LRM/RLM/ALM,
#       bidi embeddings + overrides + isolates (Trojan Source reordering), word
#       joiner, BOM, soft hyphen, the invisible math operators, and the
#       U+E0020–U+E007F tag block
#   Zl  U+2028 line separator, and Zp U+2029 paragraph separator — not newlines,
#       but some terminals honour them as line breaks
_SPOOFING_CATEGORIES = frozenset({"Cf", "Zl", "Zp"})


def _strip_terminal_controls(text: str) -> str:
    """Drop everything untrusted workflow content could use to spoof a terminal.

    Removes C0/C1 control chars (keeping only tab and newline) so the text can't
    emit ANSI/OSC escape sequences, and drops every invisible Unicode codepoint
    in ``_SPOOFING_CATEGORIES``.

    Carriage return is dropped too, not kept: a lone ``\\r`` returns the cursor to
    column 0, letting a note overwrite text this command already printed —
    the same output-spoofing the escape stripping exists to prevent. Dropping it
    is lossless for CRLF content, which keeps its ``\\n``.
    """
    # ASCII is settled by range alone — no Cf/Zl/Zp lives below U+00A0 — so the
    # category lookup only runs for non-ASCII, keeping large payloads cheap.
    return "".join(
        ch
        for ch in text
        if (ch in "\t\n" or 0x20 <= ord(ch) < 0x7F)
        or (ord(ch) >= 0xA0 and unicodedata.category(ch) not in _SPOOFING_CATEGORIES)
    )


def _reject_unsafe_workflow_key(renderer, key: str) -> str:
    """Validate a local workflow id/name as a safe relative path under ``workflows/``.

    Subdirectories are allowed (``sub/flux.json``), but traversal
    (``..``), absolute paths, home refs, and backslashes are rejected so a
    hostile id can't escape the userdata dir. Returns the cleaned key.

    Components are checked after stripping trailing dots and spaces, because a
    Windows ComfyUI server strips those from filenames — so ``.. `` or ``...``
    would collapse to ``..`` and escape ``workflows/`` if we only matched the
    literal ``..``.
    """
    cleaned = key.strip()
    parts = cleaned.split("/")
    if (
        not cleaned
        or cleaned.startswith("/")
        or cleaned.startswith("~")
        or "\\" in cleaned
        # Catches "" (leading/trailing/double slash), ".", "..", "...", ".. ", etc.
        or any(p.rstrip(" .") in ("", "..") for p in parts)
    ):
        renderer.error(
            code="invalid_argument",
            message=f"workflow id {key!r} is not a valid path under the local workflows/ dir",
            hint="use a relative name like `flux.json` or `sub/flux.json` (no `..`, no leading `/`)",
        )
        raise typer.Exit(code=1)
    return cleaned


def _userdata_request(
    url: str,
    target,
    *,
    method: str = "GET",
    data: bytes | None = None,
    content_type: str | None = None,
    timeout: float = 30.0,
) -> tuple[int, bytes]:
    """Authed HTTP call to a ComfyUI ``/userdata`` endpoint returning (status, raw_bytes).

    Raises urllib errors verbatim so callers can map them to envelope codes.
    Local ComfyUI needs no auth; ``authed_urlopen`` attaches no header when the
    Target carries no credential.
    """
    from comfy_cli.http import authed_urlopen

    with authed_urlopen(url, target, method=method, data=data, content_type=content_type, timeout=timeout) as resp:
        status = resp.status
        # Read one byte past the cap so we can tell a full body from a truncated one.
        raw = resp.read(_USERDATA_MAX_BYTES + 1)
    if len(raw) > _USERDATA_MAX_BYTES:
        raise _ResponseTooLarge()
    return status, raw


def _handle_local_http_error(renderer, e, *, operation: str, workflow_id: str | None = None) -> typer.Exit:
    """Map local ``/userdata`` failures to envelope codes. Returns an Exit to ``raise from``.

    A *reachable* server that answers with an HTTP error or an unparseable body
    gets a distinct code (``server_error`` / ``client_error`` / ``invalid_response``)
    so the user isn't wrongly told to `comfy launch` — that hint is reserved for a
    genuinely unreachable server (URLError / OSError).
    """
    import urllib.error

    if isinstance(e, _ResponseTooLarge):
        renderer.error(
            code="workflow_too_large",
            message=f"local ComfyUI /userdata response during {operation} exceeded the "
            f"{_USERDATA_MAX_BYTES // (1024 * 1024)} MiB cap",
            hint=_LOCAL_TOO_LARGE_HINTS.get(operation, "the local response was unexpectedly large"),
            details={"operation": operation, "limit_bytes": _USERDATA_MAX_BYTES},
        )
    elif isinstance(e, urllib.error.HTTPError) and e.code == 404:
        renderer.error(
            code="workflow_not_found",
            message=f"no saved workflow with id {workflow_id!r}"
            if workflow_id
            else f"workflow not found ({operation})",
            hint="list available workflows via `comfy --json --where local workflow list`",
            details={"workflow_id": workflow_id, "operation": operation},
        )
    elif isinstance(e, urllib.error.HTTPError) and 500 <= e.code < 600:
        renderer.error(
            code="server_error",
            message=f"HTTP {e.code} during {operation} against local ComfyUI /userdata",
            hint="check the ComfyUI server logs",
            details={"status": e.code, "operation": operation},
        )
    elif isinstance(e, urllib.error.HTTPError):
        renderer.error(
            code="client_error",
            message=f"HTTP {e.code} during {operation} against local ComfyUI /userdata",
            hint="the server rejected the request; check the workflow id and the server version",
            details={"status": e.code, "operation": operation},
        )
    elif isinstance(e, json.JSONDecodeError):
        renderer.error(
            code="invalid_response",
            message=f"local ComfyUI returned an unparseable body during {operation}",
            hint="check that the host:port really is a ComfyUI server",
            details={"operation": operation},
        )
    else:
        renderer.error(
            code="server_not_running",
            message=f"could not reach local ComfyUI during {operation}: {e}",
            hint="run `comfy launch` to start a local server",
        )
    return typer.Exit(code=1)


def _userdata_file_url(target, key: str, query: dict | None = None) -> str:
    """Build the ``/userdata/<encoded workflows/key>`` URL. The whole relative
    path is percent-encoded into a single segment (``/`` → ``%2F``), exactly as
    the ComfyUI frontend does, so subdir keys survive aiohttp's ``{file}`` route."""
    import urllib.parse

    encoded = urllib.parse.quote(f"{_WORKFLOWS_DIR}/{key}", safe="")
    url = target.url("userdata", encoded)
    if query:
        url += "?" + urllib.parse.urlencode(query)
    return url


def _http_request(
    url: str, target, *, method: str = "GET", body: dict | None = None, timeout: float = 30.0
) -> tuple[int, dict | list | None]:
    """Authed HTTP call returning (status, parsed_json_or_none). Raises
    urllib errors verbatim so callers can surface the right error code, and
    ``_ResponseTooLarge`` when the body exceeds ``_HTTP_MAX_BYTES`` — an
    oversize body must not masquerade as an unparseable one."""
    from comfy_cli.http import authed_urlopen

    data = json.dumps(body).encode("utf-8") if body is not None else None
    ct = "application/json" if data is not None else None
    with authed_urlopen(url, target, method=method, data=data, content_type=ct, timeout=timeout) as resp:
        status = resp.status
        # Read one byte past the cap so we can tell a full body from a truncated one.
        raw = resp.read(_HTTP_MAX_BYTES + 1)
    if len(raw) > _HTTP_MAX_BYTES:
        raise _ResponseTooLarge()
    if not raw:
        return status, None
    try:
        return status, json.loads(raw.decode("utf-8"))
    except (ValueError, RecursionError) as e:
        # A non-empty body that won't decode as JSON is a *malformed* response, not
        # "no data": returning ``None`` here would let callers report an empty list
        # or a null id as success. Raise instead so it surfaces as a loud, mapped
        # error. Decode as UTF-8 *explicitly* first: handed raw bytes, ``json.loads``
        # auto-detects UTF-16/32 (RFC 4627) and would silently accept a non-UTF-8 body
        # the contract treats as malformed. Non-UTF-8 bytes -> ``UnicodeDecodeError``;
        # valid-UTF-8 non-JSON text -> ``JSONDecodeError``; both subclass ``ValueError``.
        # Catch the ``ValueError`` base to also map the parser's *other* rejections that
        # aren't ``JSONDecodeError`` — e.g. a JSON integer past CPython's 4300-digit
        # int/str limit raises a bare ``ValueError`` — plus ``RecursionError`` from
        # pathologically nested input (the 64 MiB cap permits deep nesting). Otherwise
        # those escape the mapping and crash the CLI with a raw traceback.
        raise _ResponseUnparseable() from e


def _handle_cloud_http_error(renderer, e, *, operation: str, workflow_id: str | None = None) -> typer.Exit:
    """Map HTTP failures to envelope codes. Returns an Exit to ``raise from``.

    Thin wrapper over the shared cloud-error mapper (BE-3266) that supplies the
    ``workflow``-specific 404 envelope and the oversize/unparseable-response
    checks; everything else is shared with ``jobs``.
    """
    if isinstance(e, _ResponseUnparseable):
        renderer.error(
            code="workflow_unparseable",
            message=f"cloud returned a non-empty but unparseable (non-JSON) response during {operation}",
            hint="the server sent a malformed body; retry, and report it if it persists",
            details={"operation": operation, "workflow_id": workflow_id},
        )
        return typer.Exit(code=1)
    if isinstance(e, _ResponseTooLarge):
        renderer.error(
            code="workflow_too_large",
            message=f"cloud API response during {operation} exceeded the {_HTTP_MAX_BYTES // (1024 * 1024)} MiB cap",
            hint=_TOO_LARGE_HINTS.get(operation, "the cloud response was unexpectedly large"),
            details={"operation": operation, "workflow_id": workflow_id, "limit_bytes": _HTTP_MAX_BYTES},
        )
        return typer.Exit(code=1)

    from comfy_cli.command._cloud_errors import handle_cloud_http_error

    not_found_message = (
        f"no saved workflow with id {workflow_id!r}" if workflow_id else f"workflow not found ({operation})"
    )
    return handle_cloud_http_error(
        renderer,
        e,
        operation=operation,
        not_found_code="workflow_not_found",
        not_found_message=not_found_message,
        not_found_hint="list available workflows via `comfy --json workflow list`",
        id_label="workflow_id",
        resource_id=workflow_id,
    )


# ---------------------------------------------------------------------------
# Local ``/userdata`` implementations of the four saved-workflow verbs.
# ---------------------------------------------------------------------------


def _local_list(renderer, target, *, name: str | None, limit: int, sort: str, order: str) -> None:
    import urllib.error
    import urllib.parse

    params = {"dir": _WORKFLOWS_DIR, "recurse": "true", "split": "false", "full_info": "true"}
    url = target.url("userdata") + "?" + urllib.parse.urlencode(params)
    try:
        _, raw = _userdata_request(url, target)
        rows = json.loads(raw) if raw else []
    except urllib.error.HTTPError as e:
        if e.code == 404:
            rows = []  # the workflows/ dir doesn't exist yet → no saved workflows
        else:
            raise _handle_local_http_error(renderer, e, operation="list") from e
    except (urllib.error.URLError, OSError, json.JSONDecodeError, _ResponseTooLarge) as e:
        raise _handle_local_http_error(renderer, e, operation="list") from e

    rows = [r for r in rows if isinstance(r, dict) and isinstance(r.get("path"), str)]
    if name:
        needle = name.lower()
        rows = [r for r in rows if needle in r["path"].lower()]

    sort_key = _LOCAL_SORT_KEYS.get(sort, "created")
    reverse = order != "asc"
    if sort_key == "path":
        rows.sort(key=lambda r: r["path"].lower(), reverse=reverse)
    else:
        rows.sort(key=lambda r: r.get(sort_key) or 0, reverse=reverse)
    rows = rows[: min(max(limit, 1), 100)]

    workflows = [
        {
            "id": r["path"],
            "name": r["path"],
            "size": r.get("size"),
            "modified": r.get("modified"),
            "created": r.get("created"),
        }
        for r in rows
    ]
    payload = {"count": len(workflows), "workflows": workflows}
    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("id")
        tbl.add_column("size", justify="right", style="dim")
        for r in workflows[:50]:
            # Both cells are fields of the `/userdata` listing the server
            # returned, and `Table.add_row` parses markup in a `str` cell.
            tbl.add_row(
                sanitize_markup(r["id"]),
                sanitize_markup(r["size"]) if r["size"] is not None else "",
            )
        renderer.console().print(tbl)
        rprint(f"[dim]{len(workflows)} workflow(s) (local)[/dim]")
    renderer.emit(payload, command="workflow list", where="local")


def _local_get(renderer, target, workflow_id: str, out: str | None) -> None:
    import urllib.error

    key = _reject_unsafe_workflow_key(renderer, workflow_id)
    url = _userdata_file_url(target, key)
    try:
        _, raw = _userdata_request(url, target)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, _ResponseTooLarge) as e:
        raise _handle_local_http_error(renderer, e, operation="get", workflow_id=workflow_id) from e

    try:
        # ``json.loads`` decodes bytes itself and raises ``UnicodeDecodeError`` (not a
        # ``JSONDecodeError``) on non-UTF-8 input, so catch both.
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        data = None
    if _is_frontend_format(data):
        node_count = len(data["nodes"])
    elif isinstance(data, dict):
        node_count = len(data)
    else:
        node_count = None

    # A valid-UTF-8-but-not-JSON body (e.g. an HTML proxy/error page returned 200) or
    # non-UTF-8 bytes still get written verbatim; warn so a corrupt fetch isn't silent.
    warnings: list[dict[str, str]] = []
    if data is None:
        warnings.append(
            {
                "code": "workflow_content_not_json",
                "message": "fetched content is not parseable JSON; wrote the raw bytes unchanged",
            }
        )

    if out:
        out_path = Path(out).expanduser()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(raw)
        except OSError as e:
            renderer.error(
                code="workflow_write_error",
                message=f"could not write workflow to {out_path}: {e}",
                hint="check the --out path is writable and the disk has space",
            )
            raise typer.Exit(code=1) from e
        target_repr = str(out_path)
    else:
        if renderer.is_pretty():
            import sys

            # Strip control chars so untrusted content can't emit ANSI/OSC escapes
            # that spoof or manipulate the terminal.
            sys.stdout.write(_strip_terminal_controls(raw.decode("utf-8", "replace")))
            sys.stdout.write("\n")
        target_repr = "stdout"

    payload: dict[str, Any] = {
        "workflow_id": key,
        "out": target_repr,
        "bytes": len(raw),
        "node_count": node_count,
    }
    if warnings:
        payload["warnings"] = warnings
    if renderer.is_pretty() and out:
        rprint(f"[green]✓[/green] wrote {len(raw):,} bytes to {target_repr}")
    renderer.emit(payload, command="workflow get", where="local")


def _local_save(renderer, target, workflow_file: str, name: str, description: str | None) -> None:
    import urllib.error

    path = Path(workflow_file).expanduser()
    if not path.is_file():
        renderer.error(
            code="workflow_not_found",
            message=f"local workflow file not found: {path}",
            hint="check the path",
        )
        raise typer.Exit(code=1)
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        renderer.error(
            code="workflow_read_error",
            message=f"could not read {path}: {e}",
            hint="check file permissions and encoding",
        )
        raise typer.Exit(code=1) from e
    try:
        workflow_json = json.loads(text)
    except json.JSONDecodeError as e:
        renderer.error(
            code="workflow_invalid_json",
            message=f"{path} is not valid JSON: {e}",
            hint="re-export the workflow from ComfyUI",
        )
        raise typer.Exit(code=1) from e
    if not isinstance(workflow_json, dict):
        renderer.error(
            code="workflow_not_api_format",
            message="workflow_json must be a JSON object",
            hint="use ComfyUI's `File > Save` to export",
        )
        raise typer.Exit(code=1)

    key = name if name.lower().endswith(".json") else f"{name}.json"
    key = _reject_unsafe_workflow_key(renderer, key)
    url = _userdata_file_url(target, key, query={"overwrite": "true", "full_info": "true"})
    try:
        _, raw = _userdata_request(
            url, target, method="POST", data=text.encode("utf-8"), content_type="application/json"
        )
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, _ResponseTooLarge) as e:
        raise _handle_local_http_error(renderer, e, operation="save", workflow_id=key) from e

    info = None
    try:
        info = json.loads(raw)
    except json.JSONDecodeError:
        pass
    # ComfyUI returns FileInfo.path as "workflows/<key>"; strip the prefix back
    # to the id the other verbs use. Fall back to the key we sent.
    stored_id = key
    if isinstance(info, dict) and isinstance(info.get("path"), str):
        stored = info["path"]
        prefix = f"{_WORKFLOWS_DIR}/"
        stored_id = stored[len(prefix) :] if stored.startswith(prefix) else stored

    payload: dict[str, Any] = {
        "workflow_id": stored_id,
        "name": stored_id,
        "source": str(path),
        "size": info.get("size") if isinstance(info, dict) else None,
        "modified": info.get("modified") if isinstance(info, dict) else None,
    }
    if description:
        # Local file-backed userdata has no metadata store for a description.
        payload["warnings"] = [
            {"code": "description_ignored", "message": "--description is ignored on the local path (no metadata store)"}
        ]
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] saved [dim]{stored_id}[/dim]")
    renderer.emit(payload, command="workflow save", where="local", changed=True)


def _local_delete(renderer, target, workflow_id: str) -> None:
    import urllib.error

    key = _reject_unsafe_workflow_key(renderer, workflow_id)
    url = _userdata_file_url(target, key)
    try:
        _userdata_request(url, target, method="DELETE")
    except (urllib.error.HTTPError, urllib.error.URLError, OSError, _ResponseTooLarge) as e:
        raise _handle_local_http_error(renderer, e, operation="delete", workflow_id=workflow_id) from e

    payload = {"workflow_id": key, "deleted": True}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] deleted [dim]{key}[/dim]")
    renderer.emit(payload, command="workflow delete", where="local", changed=True)


@app.command("list", help="List saved workflows (cloud store, or local ComfyUI /userdata with --where local).")
@tracking.track_command("workflow")
def list_cmd(
    name: Annotated[
        str | None,
        typer.Option("--name", show_default=False, help="Case-insensitive substring match on workflow name."),
    ] = None,
    limit: Annotated[int, typer.Option("--limit", help="Cap rows returned (max 100).")] = 20,
    sort: Annotated[
        str,
        typer.Option("--sort", help="Sort field: create_time | update_time | name."),
    ] = "create_time",
    order: Annotated[
        str,
        typer.Option("--order", help="Sort direction: asc | desc."),
    ] = "desc",
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.error
    import urllib.parse

    renderer = get_renderer()

    # Validate the free-form sort/order options up front (both routes) so a typo like
    # `--order ASC` errors loudly instead of silently sorting the wrong way.
    order_norm = order.lower()
    if order_norm not in ("asc", "desc"):
        renderer.error(
            code="invalid_argument",
            message=f"--order must be 'asc' or 'desc', got {order!r}",
            hint="pass `--order asc` or `--order desc`",
        )
        raise typer.Exit(code=1)
    if sort not in _LOCAL_SORT_KEYS:
        renderer.error(
            code="invalid_argument",
            message=f"--sort must be one of {', '.join(_LOCAL_SORT_KEYS)}, got {sort!r}",
            hint="pass `--sort create_time|update_time|name`",
        )
        raise typer.Exit(code=1)

    target = _resolve_where_target(where)
    if not target.is_cloud:
        return _local_list(renderer, target, name=name, limit=limit, sort=sort, order=order_norm)

    params: dict[str, Any] = {"limit": min(max(limit, 1), 100), "sort": sort, "order": order_norm}
    if name:
        params["name"] = name
    url = target.url("workflows") + "?" + urllib.parse.urlencode(params)

    try:
        _, body = _http_request(url, target)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        _ResponseUnparseable,
        _ResponseTooLarge,
    ) as e:
        raise _handle_cloud_http_error(renderer, e, operation="list") from e

    # A non-empty body decodes here only if it was valid JSON; guard the shape the
    # way ``get``/``save`` do. An empty body is a legitimate ``None`` (→ no rows), but
    # a valid-JSON *non-dict* 200 (an array like ``[1, 2, 3]`` or a scalar) is malformed:
    # ``(body).get("data")`` would raise a raw ``AttributeError``, and coercing it to an
    # empty list would masquerade the malformed shape as a genuinely-empty listing.
    if body is not None and not isinstance(body, dict):
        renderer.error(
            code="cloud_http_error",
            message="unexpected response shape from /api/workflows (expected a JSON object)",
            details={"got_type": type(body).__name__},
        )
        raise typer.Exit(code=1)

    # A missing/empty ``data`` is a legitimately-empty listing, but a present non-list
    # ``data`` is malformed the same way a non-dict body is: a scalar (``{"data": 42}``)
    # would raise a raw ``TypeError`` in the comprehension below, and a str/dict would
    # iterate silently and masquerade as an empty listing. Reject it with the same envelope.
    rows = (body or {}).get("data")
    if rows is None:
        rows = []
    elif not isinstance(rows, list):
        renderer.error(
            code="cloud_http_error",
            message="unexpected response shape from /api/workflows (data must be a JSON array)",
            details={"got_type": type(rows).__name__},
        )
        raise typer.Exit(code=1)
    payload = {
        "count": len(rows),
        "workflows": [
            {
                "id": r.get("id"),
                "name": r.get("name"),
                "description": r.get("description"),
                "default_view": r.get("default_view"),
                "latest_version": r.get("latest_version"),
                "created_at": r.get("created_at"),
                "updated_at": r.get("updated_at"),
            }
            for r in rows
            if isinstance(r, dict)
        ],
    }
    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(show_header=True, header_style="bold")
        tbl.add_column("id", style="dim")
        tbl.add_column("name")
        tbl.add_column("ver", justify="right", style="dim")
        tbl.add_column("updated", style="dim")
        for r in payload["workflows"][:50]:
            # The cloud workflow catalog is server-supplied end to end, same as
            # the local `/userdata` listing `_local_list` renders. `str()` before
            # the slices: `sanitize_markup` coerces, but the truncation runs
            # first, and a numeric `id`/`updated_at` in the JSON would raise
            # `TypeError: 'int' object is not subscriptable` before it got there.
            tbl.add_row(
                sanitize_markup(str(r["id"])[:8] + "…" if r["id"] else ""),
                sanitize_markup(r["name"] or "(untitled)"),
                sanitize_markup(r["latest_version"] or ""),
                sanitize_markup(str(r["updated_at"] or "")[:10]),
            )
        renderer.console().print(tbl)
        rprint(f"[dim]{len(rows)} workflow(s)[/dim]")
    renderer.emit(payload, command="workflow list", where="cloud")


@app.command(
    "get",
    help="Fetch a saved workflow's content (cloud, or local with --where local); writes JSON to --out or stdout.",
)
@tracking.track_command("workflow")
def get_cmd(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow id: cloud UUID, or local path under workflows/ (e.g. flux.json)."),
    ],
    out: Annotated[
        str | None,
        typer.Option("--out", "-o", show_default=False, help="Write JSON to this file instead of stdout."),
    ] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.error

    renderer = get_renderer()
    target = _resolve_where_target(where)
    if not target.is_cloud:
        return _local_get(renderer, target, workflow_id, out)

    import urllib.parse as _up

    # Encode the id so a malformed or hostile value can't escape the path
    # segment. Cloud rejects malformed UUIDs upstream too, but encode at
    # the client for defense in depth (e.g. ``../foo`` → ``%2E%2E%2Ffoo``).
    url = target.url("workflows", _up.quote(workflow_id, safe=""), "content")
    try:
        _, body = _http_request(url, target)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        _ResponseUnparseable,
        _ResponseTooLarge,
    ) as e:
        raise _handle_cloud_http_error(renderer, e, operation="get", workflow_id=workflow_id) from e

    if not isinstance(body, dict) or "workflow_json" not in body:
        renderer.error(
            code="cloud_http_error",
            message=f"unexpected response shape from /api/workflows/{workflow_id}/content",
            details={"workflow_id": workflow_id, "got_keys": list(body.keys()) if isinstance(body, dict) else None},
        )
        raise typer.Exit(code=1)

    workflow_bytes = json.dumps(body["workflow_json"], indent=2).encode("utf-8")
    if out:
        out_path = Path(out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(workflow_bytes)
        target_repr = str(out_path)
    else:
        if renderer.is_pretty():
            import sys

            sys.stdout.write(workflow_bytes.decode("utf-8"))
            sys.stdout.write("\n")
        target_repr = "stdout"

    payload = {
        "workflow_id": workflow_id,
        "version_id": body.get("id"),
        "version": body.get("version"),
        "out": target_repr,
        "bytes": len(workflow_bytes),
        "node_count": len(body["workflow_json"]) if isinstance(body["workflow_json"], dict) else None,
    }
    if renderer.is_pretty() and out:
        rprint(f"[green]✓[/green] wrote {len(workflow_bytes):,} bytes to {target_repr}")
    renderer.emit(payload, command="workflow get", where="cloud")


@app.command(
    "save",
    help="Save a workflow JSON to the saved-workflow store (cloud, or local ComfyUI /userdata with --where local).",
)
@tracking.track_command("workflow")
def save_cmd(
    workflow_file: Annotated[str, typer.Argument(help="Path to a workflow JSON file.")],
    name: Annotated[
        str,
        typer.Option(
            "--name", help="Cloud: display name. Local: filename under workflows/ ('.json' appended if absent)."
        ),
    ],
    description: Annotated[
        str | None,
        typer.Option("--description", show_default=False, help="Optional description (cloud only; ignored on local)."),
    ] = None,
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.error

    renderer = get_renderer()
    target = _resolve_where_target(where)
    if not target.is_cloud:
        return _local_save(renderer, target, workflow_file, name, description)

    path = Path(workflow_file).expanduser()
    if not path.is_file():
        renderer.error(
            code="workflow_not_found",
            message=f"local workflow file not found: {path}",
            hint="check the path",
        )
        raise typer.Exit(code=1)
    try:
        workflow_json = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        renderer.error(
            code="workflow_invalid_json",
            message=f"{path} is not valid JSON: {e}",
            hint="re-export the workflow from ComfyUI",
        )
        raise typer.Exit(code=1) from e
    if not isinstance(workflow_json, dict):
        renderer.error(
            code="workflow_not_api_format",
            message="workflow_json must be a JSON object",
            hint="use ComfyUI's `File > Save` to export",
        )
        raise typer.Exit(code=1)

    body: dict[str, Any] = {"name": name, "workflow_json": workflow_json}
    if description:
        body["description"] = description
    url = target.url("workflows")
    try:
        _, resp = _http_request(url, target, method="POST", body=body)
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        _ResponseUnparseable,
        _ResponseTooLarge,
    ) as e:
        raise _handle_cloud_http_error(renderer, e, operation="save") from e

    workflow_id = (resp or {}).get("id") if isinstance(resp, dict) else None
    payload = {
        "workflow_id": workflow_id,
        "name": name,
        "latest_version": (resp or {}).get("latest_version") if isinstance(resp, dict) else None,
        "source": str(path),
    }
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] saved {name!r} → [dim]{workflow_id}[/dim]")
    renderer.emit(payload, command="workflow save", where="cloud", changed=True)


@app.command("delete", help="Delete a saved workflow (cloud, or local ComfyUI /userdata with --where local).")
@tracking.track_command("workflow")
def delete_cmd(
    workflow_id: Annotated[
        str,
        typer.Argument(help="Workflow id to delete: cloud UUID, or local path under workflows/ (e.g. flux.json)."),
    ],
    where: Annotated[str | None, typer.Option("--where", show_default=False)] = None,
):
    import urllib.error

    renderer = get_renderer()
    target = _resolve_where_target(where)
    if not target.is_cloud:
        return _local_delete(renderer, target, workflow_id)

    import urllib.parse as _up

    url = target.url("workflows", _up.quote(workflow_id, safe=""))
    try:
        _, _body = _http_request(url, target, method="DELETE")
    except (
        urllib.error.HTTPError,
        urllib.error.URLError,
        OSError,
        _ResponseUnparseable,
        _ResponseTooLarge,
    ) as e:
        raise _handle_cloud_http_error(renderer, e, operation="delete", workflow_id=workflow_id) from e

    payload = {"workflow_id": workflow_id, "deleted": True}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] deleted [dim]{workflow_id}[/dim]")
    renderer.emit(payload, command="workflow delete", where="cloud", changed=True)


# ---------------------------------------------------------------------------
# validate — API-format workflow validation
# ---------------------------------------------------------------------------
# The canonical home for API-format workflow validation. The top-level
# `comfy validate` is kept as a hidden deprecated alias that delegates to the
# shared implementation below (see cmdline.py).


def validate_api_workflow(
    workflow: str,
    *,
    where: str | None = None,
    host: str | None = None,
    port: int | None = None,
    input_path: str | None = None,
    command: str = "workflow validate",
) -> None:
    """Validate an API-format workflow without submitting it.

    Shared implementation behind ``comfy workflow validate`` (canonical) and the
    deprecated top-level ``comfy validate`` alias. Checks class_types, input
    shapes, enum values, and edge wiring against object_info loaded from the run
    target. ``command`` labels the emitted envelope so each entry point reports
    its own path.
    """
    from comfy_cli import where as where_module
    from comfy_cli.command.run import is_ui_workflow
    from comfy_cli.command.run.preflight import _detect_partner_nodes
    from comfy_cli.cql.engine import Graph, LoadError
    from comfy_cli.env_checker import _bracket_host, _unbracket_host
    from comfy_cli.workflow_to_api import WorkflowConversionError, convert_ui_to_api

    renderer = get_renderer()

    # Load workflow
    wf_path = Path(workflow).expanduser()
    if not wf_path.is_file():
        renderer.error(code="workflow_not_found", message=f"Workflow file not found: {workflow}", hint="check the path")
        raise typer.Exit(code=1)
    try:
        wf_data = json.loads(wf_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        renderer.error(code="workflow_invalid_json", message=f"Invalid JSON: {e}", hint="re-export from ComfyUI")
        raise typer.Exit(code=1) from e
    except (OSError, UnicodeDecodeError) as e:
        # e.g. a non-UTF-8 file, permission denied, or a TOCTOU race if the file
        # vanished after the is_file() check above. Report structurally instead
        # of crashing with a raw traceback.
        renderer.error(
            code="workflow_read_error",
            message=f"Unable to read workflow file: {e}",
            hint="check file permissions and encoding",
        )
        raise typer.Exit(code=1) from e
    if not isinstance(wf_data, dict):
        renderer.error(
            code="workflow_not_api_format", message="Workflow must be a JSON object", hint="use File > Export (API)"
        )
        raise typer.Exit(code=1)

    # Load graph. Route through the shared resolver rather than trusting the raw
    # `--where` string: it normalizes case/whitespace (`--where LOCAL` is the
    # same target as `local`) and emits a `where_invalid` envelope instead of
    # letting an unknown value escape as a bare ValueError traceback — but only
    # for an *explicit* bad `--where`; a bad env/project/config default has no
    # user to blame, so it drops to the local default instead of breaking the
    # command, matching the pre-existing routing behavior of `nodes`/`jobs`.
    target = where_module.WhereTarget.LOCAL
    try:
        target = where_module.resolve_default(flag=where).target
    except ValueError as e:
        if where:
            renderer.error(
                code="where_invalid",
                message=str(e),
                hint="use `--where local` or `--where cloud`",
            )
            raise typer.Exit(code=1) from e
        # A bad env/project/config value with no explicit flag never breaks the
        # command — drop to the local default, as before.
    mode = target.value
    # Routing resolved — stamp it so the envelopes below (the object_info load
    # failure and the UI-conversion errors) name the target this validate ran
    # against. The file-read errors above stay `where: null`: they precede the
    # decision, as does the `where_invalid` envelope raised just above.
    renderer.where = mode

    # Resolve the local object_info server the same way `comfy run` does —
    # flag > COMFY_LOCAL_URL > config.background > 127.0.0.1:8188. Without the
    # `config.background` step validate would consult whatever answers on the
    # default port while `run` submits to the background server comfy-cli
    # launched on another one, making the verdict meaningless for the server
    # that will actually execute the workflow (BE-6299). `resolve_target` does
    # not consult `config.background` on purpose (other callers, e.g. transfer
    # and system, must not), so — as its docstring says — the callers that do
    # honor it resolve upstream, here.
    is_local_fetch = input_path is None and target is where_module.WhereTarget.LOCAL
    if is_local_fetch:
        from comfy_cli.host_port import parse_host_port_arg, report_usage_error, resolve_host_port

        # `host is not None` (not `if host:`): `--host ""` must reach the parser
        # and be rejected, not be read as "no --host given". Likewise the port
        # merge tests `is None`, so an explicit `--port 0` isn't silently
        # overridden by a port embedded in the combined `--host h:p` form —
        # `resolve_host_port` rejects it as out of range instead.
        # `report_usage_error` gives JSON/NDJSON consumers a terminating
        # envelope for that rejection instead of an empty stdout (exit stays 2).
        with report_usage_error(renderer, command=command):
            if host is not None:
                host, parsed_port = parse_host_port_arg(host)
                if port is None and parsed_port is not None:
                    port = parsed_port
            host, port = resolve_host_port(host, port)

    try:
        # Through the resilient loader — NOT a direct `Graph.load` — so validate
        # honors the same chain as every other consumer: `--input` >
        # COMFY_OBJECT_INFO_FILE > cloud TTL cache > live fetch (+forced-refresh
        # retry) > stale cache. A direct load silently dropped the env-pinned
        # offline catalog and every fallback for the one command an agent runs
        # before every submit.
        from comfy_cli.cql.loader import resilient_load_object_info

        raw = resilient_load_object_info(mode=mode, host=host, port=port, input_path=input_path)
        graph = Graph.from_object_info(raw)
        graph._try_default_annotations()
    except LoadError as e:
        renderer.error(
            code="cql_no_graph",
            message=str(e),
            hint=e.details.get("hint", "pass --input <object_info.json>, or start the server"),
            details=e.details,
        )
        raise typer.Exit(code=1) from e

    # Detect a UI-export (frontend/canvas) workflow and lower it to API format
    # before validating — exactly as `comfy run` does. Without this the wrapper
    # keys (`nodes`, `links`, `groups`, `config`, …) each emit a `non_node_key`
    # warning, zero nodes are checked, and the result is a vacuous `valid:true`.
    # The converter reuses the object_info the graph was already built from
    # (`graph.object_info`), so offline `--input` works and no second fetch happens.
    converted_from_ui = False
    if is_ui_workflow(wf_data):
        if renderer.is_pretty():
            rprint("[yellow]Detected UI-format workflow, converting to API format...[/yellow]")
        try:
            converted = convert_ui_to_api(wf_data, graph.object_info)
        except WorkflowConversionError as e:
            renderer.error(
                code="workflow_not_api_format",
                message=f"Workflow is a UI export that could not be converted to API format: {e}",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1) from e
        except Exception as e:  # noqa: BLE001 — never leak a raw traceback to the agent flow
            renderer.error(
                code="conversion_crash",
                message=f"Workflow conversion crashed unexpectedly: {type(e).__name__}: {e}",
                hint="report this at https://github.com/Comfy-Org/comfy-cli/issues",
                details={"exception_type": type(e).__name__},
            )
            raise typer.Exit(code=1) from e
        if not converted:
            renderer.error(
                code="workflow_not_api_format",
                message="Workflow is a UI export that converted to no executable nodes",
                hint="use ComfyUI's 'File > Export (API)' to save as API format",
            )
            raise typer.Exit(code=1)
        wf_data = converted
        converted_from_ui = True

    result = graph.validate_workflow(wf_data)

    # When the caller handed us a CANVAS graph, they have never seen the
    # flattened ids the lowering mints for subgraph interiors (`57:3`) — their
    # edit surface (slots / set-widget) speaks `57/3`. Key every issue by the
    # editable address so a validate error can be acted on directly; keep the
    # raw API id alongside for anyone correlating with server node_errors. An
    # already-API input skips this: its ids address the document as given.
    if converted_from_ui:
        for issue in (*result["errors"], *result["warnings"]):
            nid = str(issue.get("node_id", ""))
            if ":" in nid:
                issue["api_node_id"] = nid
                issue["node_id"] = nid.replace(":", "/")

    # Preview credit spend: partner-API (paid) nodes spend Comfy credits when the
    # workflow is run. This is the same detection `comfy run` uses (authoritative
    # `api_node: true`, `partner/...` category fallback), surfaced here read-only
    # so agents can answer "will this spend credits?" without running. `wf_data`
    # is API format at this point (any UI export already converted above), which
    # is the format the detector reads. Purely informational — no exit-code gate.
    partner_nodes = _detect_partner_nodes(wf_data, graph.object_info)

    payload = {
        "workflow": str(wf_path),
        "valid": result["valid"],
        "error_count": len(result["errors"]),
        "warning_count": len(result["warnings"]),
        "errors": result["errors"],
        "warnings": result["warnings"],
        "partner_nodes": partner_nodes,
        "spends_credits": bool(partner_nodes),
        # Name the server (or file) the verdict was computed against, so an
        # agent comparing `validate` with `run` can see whether they consulted
        # the same object_info. Populated from the values resolved above, so a
        # local run reports the concrete host/port actually queried. `host` is
        # reported unbracketed, matching `Target.host` — brackets belong to the
        # URL composed for display, not to the address itself.
        "object_info_source": (
            {"mode": "file", "path": str(input_path)}
            if input_path is not None
            else {"mode": mode, "host": _unbracket_host(host), "port": port}
            if is_local_fetch
            else {"mode": mode}
        ),
    }
    if converted_from_ui:
        # Signal that validation ran against the converted graph, not the file's
        # literal bytes, and report how many nodes the conversion produced.
        payload["converted_from_ui"] = True
        payload["converted_node_count"] = len(wf_data)

    if renderer.is_pretty():
        # Workflow-supplied strings (node ids, messages, enum/input values echoed
        # in messages) flow into rprint, which parses Rich markup. Escape them so a
        # crafted file can't inject markup to spoof/hide output (e.g. fake a green ✓).
        from rich.markup import escape

        # Name the object_info source in one dim line. A file path (and, in
        # principle, a hostname) can contain Rich-markup metacharacters, so
        # escape it — same reason the partner-node line below does.
        #
        # Branch on the same flags that built the payload, not on its "mode"
        # string: "file" is an offline sentinel that shares a key with the
        # routing targets, so a mode-string branch couples display to a value
        # the routing layer also owns.
        source = payload["object_info_source"]
        if input_path is not None:
            where_oi = escape(source["path"])
        elif is_local_fetch:
            where_oi = escape(f"http://{_bracket_host(source['host'])}:{source['port']}")
        else:
            where_oi = escape(source["mode"])
        rprint(f"[dim]object_info from {where_oi}[/dim]")
        if result["valid"]:
            rprint(f"[bold green]✓[/bold green] workflow is valid ({len(wf_data)} nodes)")
            for w in result["warnings"]:
                rprint(f"  [yellow]⚠[/yellow] {escape(str(w.get('message', '')))}")
        else:
            rprint(f"[bold red]✗[/bold red] {len(result['errors'])} error(s)")
            for e in result["errors"]:
                msg = str(e.get("message", ""))
                suggestions = e.get("suggestions", [])
                if suggestions:
                    msg += f" (did you mean: {', '.join(str(s) for s in suggestions[:3])}?)"
                rprint(f"  [red]•[/red] node {escape(str(e.get('node_id', '?')))}: {escape(msg)}")
            for w in result["warnings"]:
                rprint(f"  [yellow]⚠[/yellow] {escape(str(w.get('message', '')))}")
        if partner_nodes:
            rprint(
                f"[yellow]⚠ uses partner-API (paid) nodes that spend Comfy credits: "
                f"{', '.join(escape(n) for n in partner_nodes)}[/yellow]"
            )
    renderer.emit(payload, command=command, ok=result["valid"])

    if not result["valid"]:
        raise typer.Exit(code=1)


@app.command(
    "validate",
    help="Validate an API-format workflow without submitting. Checks class_types, input shapes, enum values, and edge wiring.",
)
@tracking.track_command("workflow")
def validate_cmd(
    workflow: Annotated[
        str,
        typer.Option(help="Path to the API-format workflow JSON file."),
    ],
    where: Annotated[
        str | None,
        typer.Option("--where", show_default=False, help="Routing target for object_info: 'local' or 'cloud'."),
    ] = None,
    host: Annotated[
        str | None,
        typer.Option(show_default=False, help="ComfyUI host (default 127.0.0.1)."),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option(show_default=False, help="ComfyUI port (default 8188)."),
    ] = None,
    input_path: Annotated[
        str | None,
        typer.Option("--input", show_default=False, help="Path to a saved object_info JSON (offline mode)."),
    ] = None,
):
    validate_api_workflow(
        workflow, where=where, host=host, port=port, input_path=input_path, command="workflow validate"
    )


# ---------------------------------------------------------------------------
# compose / fragment — fragment-based workflow composition
# ---------------------------------------------------------------------------
# Implemented in workflow_fragments.py; mounted here so the surface stays
# under `comfy workflow`. compose is a single command; fragment is a sub-typer
# of inspectors (ls/show/validate).

from comfy_cli.command import workflow_fragments as _wfrag  # noqa: E402

app.command(
    "compose",
    help="Compose a YAML blueprint of fragments into a single API-format workflow.",
)(_wfrag.compose_cmd)
app.command(
    "decompose",
    help="Project a workflow (template or API JSON) into a reusable fragment — the inverse of compose.",
)(_wfrag.decompose_cmd)
app.add_typer(_wfrag.fragment_app, name="fragment")


# ---------------------------------------------------------------------------
# Structured, CRDT-ready edit primitives (add-node / connect / set-widget /
# delete). Implemented in workflow_edit.py; mounted here so the surface stays
# under `comfy workflow`. Each emits a replayable op in `data.op`.
# ---------------------------------------------------------------------------

from comfy_cli.command import workflow_edit as _wedit  # noqa: E402

app.command("add-node", help="Add a node to the graph; emits an add_node op.")(_wedit.add_node_cmd)
app.command("connect", help="Wire an output slot to an input slot; emits a connect op.")(_wedit.connect_cmd)
app.command("set-widget", help="Set a widget by name (`<id>.<widget>`); emits a set_widget op.")(_wedit.set_widget_cmd)
app.command("delete-node", help="Delete a node and its links; emits a delete_node op.")(_wedit.delete_cmd)
app.command(
    "delete-nodes",
    help="Delete N nodes in one atomic write; emits one delete_node op per id (all-or-nothing).",
)(_wedit.delete_nodes_cmd)
app.command("clear", help="Remove every node, link, and group; emits one clear op.")(_wedit.clear_cmd)
app.command(
    "reset-doc",
    help="Reset the document to the empty baseline — nodes, ids AND replay history. Requires --confirm.",
)(_wedit.reset_doc_cmd)
app.command("ls-nodes", help="List nodes (id/type/title) in a workflow file.")(_wedit.ls_nodes_cmd)
app.command("apply", help="Apply a recipe / batch of edits in one pass; supports node aliases + --param.")(
    _wedit.apply_cmd
)
app.command("capture", help="Project a workflow into a reusable recipe (the op-batch that rebuilds it).")(
    _wedit.capture_cmd
)
app.command("foreach", help="Instantiate a recipe over N param-sets → N workflows (bulk generation).")(
    _wedit.foreach_cmd
)
