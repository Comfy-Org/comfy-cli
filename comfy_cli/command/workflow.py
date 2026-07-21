"""``comfy workflow`` — slot-based editing of ComfyUI frontend-format workflows.

Three primitives:

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


def _get_graph(input_path: str | None, host: str | None, port: int | None, on_stale=None):
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

        decision = where_module.resolve_default()
        mode = "cloud" if decision.target is where_module.WhereTarget.CLOUD else "local"
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

    serialized = json.dumps(new_workflow, indent=2)

    if stdout:
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
    if _stale:
        payload["stale"] = True
        payload["warnings"] = list(warnings) + [
            {"code": "object_info_stale", "message": f"served from cache ({_stale['source']}): {_stale['reason']}"}
        ]
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

    written: list[str] = []
    if out_dir:
        out = Path(out_dir).expanduser()
        out.mkdir(parents=True, exist_ok=True)
        for i, wf in enumerate(workflows):
            target = out / f"{p.stem}_{i:03d}.json"
            _atomic_write_text(target, json.dumps(wf, indent=2))
            written.append(str(target))
    else:
        import sys

        for wf in workflows:
            sys.stdout.write(json.dumps(wf))
            sys.stdout.write("\n")
        sys.stdout.flush()

    payload = {
        "workflow": str(p),
        "count": len(workflows),
        "warnings": warnings,
        "out_dir": str(Path(out_dir).expanduser()) if out_dir else None,
        "written": written,
    }
    if _stale:
        payload["stale"] = True
        payload["warnings"] = list(warnings) + [
            {"code": "object_info_stale", "message": f"served from cache ({_stale['source']}): {_stale['reason']}"}
        ]
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


class _ResponseTooLarge(Exception):
    """A ``/userdata`` response exceeded ``_USERDATA_MAX_BYTES`` — refuse to truncate."""


# Map the cloud ``--sort`` fields onto local FileInfo keys (client-side sort;
# ComfyUI's /userdata listing has no server-side sort/limit/filter).
_LOCAL_SORT_KEYS = {"create_time": "created", "update_time": "modified", "name": "path"}


def _resolve_where_target(where: str | None):
    """Resolve the routing Target for a saved-workflow verb (cloud or local)."""
    from comfy_cli.target import resolve_target

    return resolve_target(where=where)


def _strip_terminal_controls(text: str) -> str:
    """Drop C0/C1 control chars (keeping tab / newline / carriage return) so
    untrusted workflow content printed to a TTY can't emit ANSI/OSC escape
    sequences that spoof output or manipulate the terminal."""
    return "".join(ch for ch in text if ch in "\t\n\r" or (0x20 <= ord(ch) < 0x7F) or ord(ch) >= 0xA0)


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
    Local ComfyUI needs no auth; ``_authed_request`` is a no-op on the headers
    when the Target carries no credential.
    """
    import urllib.request

    req = _authed_request(url, target, method=method, data=data, content_type=content_type)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
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
            hint="the saved workflow is unexpectedly large; inspect it directly on the server",
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


def _authed_request(
    url: str, target, *, method: str = "GET", data: bytes | None = None, content_type: str | None = None
):
    """Build an authenticated urllib Request. The return type is annotated
    loosely to keep urllib out of the module's top-level imports."""
    import urllib.request

    req = urllib.request.Request(url, data=data, method=method)
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")
    if content_type:
        req.add_header("Content-Type", content_type)
    return req


def _http_request(
    url: str, target, *, method: str = "GET", body: dict | None = None, timeout: float = 30.0
) -> tuple[int, dict | None]:
    """Authed HTTP call returning (status, parsed_json_or_none). Raises
    urllib errors verbatim so callers can surface the right error code."""
    import urllib.request

    data = json.dumps(body).encode("utf-8") if body is not None else None
    ct = "application/json" if data is not None else None
    req = _authed_request(url, target, method=method, data=data, content_type=ct)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = resp.status
        raw = resp.read(64 * 1024 * 1024)  # 64 MiB cap
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except json.JSONDecodeError:
        return status, None


def _handle_cloud_http_error(renderer, e, *, operation: str, workflow_id: str | None = None) -> typer.Exit:
    """Map HTTP failures to envelope codes. Returns an Exit to ``raise from``."""
    import urllib.error

    if isinstance(e, urllib.error.HTTPError):
        body = (e.read() or b"")[:1000].decode("utf-8", "replace")
        if e.code == 404:
            renderer.error(
                code="workflow_not_found",
                message=f"no saved workflow with id {workflow_id!r}"
                if workflow_id
                else f"workflow not found ({operation})",
                hint="list available workflows via `comfy --json workflow list`",
                details={"workflow_id": workflow_id, "operation": operation},
            )
        elif e.code in (401, 403):
            renderer.error(
                code="cloud_unauthorized",
                message=f"HTTP {e.code} during {operation}",
                hint="re-run `comfy cloud login`",
                details={"status": e.code},
            )
        else:
            renderer.error(
                code="cloud_http_error",
                message=f"HTTP {e.code} during {operation}",
                hint="check `details.body` for the server's message",
                details={"status": e.code, "body": body, "operation": operation},
            )
    else:
        renderer.error(
            code="cloud_http_error",
            message=f"{operation} failed: {e}",
            hint="check network / `comfy auth whoami`",
        )
    return typer.Exit(code=1)


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
            tbl.add_row(r["id"], str(r["size"]) if r["size"] is not None else "")
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
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        raise _handle_cloud_http_error(renderer, e, operation="list") from e

    rows = (body or {}).get("data") or []
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
            tbl.add_row(
                (r["id"] or "")[:8] + "…" if r["id"] else "",
                r["name"] or "(untitled)",
                str(r["latest_version"] or ""),
                (r["updated_at"] or "")[:10],
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
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
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
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
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
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        raise _handle_cloud_http_error(renderer, e, operation="delete", workflow_id=workflow_id) from e

    payload = {"workflow_id": workflow_id, "deleted": True}
    if renderer.is_pretty():
        rprint(f"[green]✓[/green] deleted [dim]{workflow_id}[/dim]")
    renderer.emit(payload, command="workflow delete", where="cloud", changed=True)


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
