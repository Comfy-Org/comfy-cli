"""``comfy jobs`` — list, status, and live-watch ComfyUI prompts.

The ComfyUI server already speaks WebSocket: every node-execution event is
pushed to every connected ``/ws?clientId=…`` client, tagged with the
``prompt_id`` it belongs to. We use that channel as the live "push" feed —
no daemon, no polling.

Three subcommands:

- ``comfy jobs ls``        — combine ``/queue`` (running + pending) and
                              ``/history`` (recent completions) into one
                              ordered list.
- ``comfy jobs status``    — one-shot snapshot of a single prompt_id from
                              ``/history``.
- ``comfy jobs watch``     — live-tail the WS feed, filter on prompt_id,
                              emit events as they arrive (pretty progress
                              bar or NDJSON depending on mode).
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Any

import typer
from websocket import WebSocket, WebSocketException, WebSocketTimeoutException

from comfy_cli import cancellation, execution_errors, tracking
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.host_port import resolve_host_port as _resolve_host_port
from comfy_cli.output import get_renderer
from comfy_cli.where import cloud_preflight_or_exit

app = typer.Typer(no_args_is_help=True, help="List, inspect, and live-watch ComfyUI prompts.")


def _is_pid_alive(pid: int) -> bool:
    """Check if a process with the given PID is still running."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)  # signal 0 = existence check, no actual signal sent
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we can't signal it — still alive.
        return True


# Host/port resolution (`resolve_host_port`) is shared with `comfy run` via
# `comfy_cli.host_port`; imported above as `_resolve_host_port` to preserve the
# call sites in this module unchanged.


def _server_or_error(host: str, port: int, *, raise_on_missing: bool = True) -> bool:
    """Return True if the server is up. If ``raise_on_missing`` is True (the
    default) we emit the error envelope and exit; if False, we return False so
    the caller can fall back to a different source (e.g. on-disk state files).
    """
    if check_comfy_server_running(port, host):
        return True
    if not raise_on_missing:
        return False
    renderer = get_renderer()
    renderer.error(
        code="server_not_running",
        message=f"ComfyUI not running on {host}:{port}",
        hint="run: comfy launch",
        details={"host": host, "port": port},
    )
    raise typer.Exit(code=1)


def _http_get_json(url: str, *, timeout: float = 10.0) -> Any:
    req = urllib.request.Request(url)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.URLError as e:
        raise RuntimeError(f"failed to GET {url}: {e}") from e


# ---------------------------------------------------------------------------
# Terminal job verdict — shared helper for `jobs watch` (local + cloud)
# ---------------------------------------------------------------------------

# Terminal job status -> (ok, error_code, exit_code). A failed or cancelled job
# must surface ok:false + non-zero exit so `comfy --json jobs watch $ID && next`
# stops. Mirrors `run --wait` (command/run/__init__.py).
_TERMINAL_VERDICT: dict[str, tuple[bool, str | None, int]] = {
    "completed": (True, None, 0),
    "error": (False, "execution_error", 1),
    "cancelled": (False, "cancelled", 130),
}


def _emit_terminal(renderer, payload: dict, *, command: str, where: str | None = None) -> None:
    """Emit a job's terminal envelope with ok/exit derived from its status.

    completed -> ok:true exit 0; error -> ok:false exit 1; cancelled -> exit 130.
    Unknown statuses default to ok:true exit 0 (non-terminal/best-effort).
    """
    status = str(payload.get("status") or "unknown")
    ok, code, exit_code = _TERMINAL_VERDICT.get(status, (True, None, 0))
    if ok:
        renderer.emit(payload, command=command, where=where)
        return
    err = payload.get("error")
    raw = err.get("message") if isinstance(err, dict) and err.get("message") else None
    if not raw:
        # Cloud snapshots (_cloud_status_snapshot) carry the failure text at
        # top-level `error_message`; the local WS path carries the decoded
        # execution_error event dict under `details`.
        raw = payload.get("error_message") or payload.get("details")
    message = raw if isinstance(raw, str) else None
    hint = None
    if status == "error":
        verdict = execution_errors.classify(raw)
        code, message, hint = verdict["code"], verdict["message"], verdict["hint"]
        # The raw server text repeats the full traceback; keep the envelope to
        # the one-line cause + structured tail and leave the full record to
        # `jobs status`.
        if payload.get("error_message"):
            payload["error_message"] = verdict["message"]
        if isinstance(payload.get("details"), dict) and "traceback" in payload["details"]:
            payload["details"] = verdict["details"]
        if isinstance(err, dict):
            payload["error"] = {**err, "code": code, "message": verdict["message"], "details": verdict["details"]}
    renderer.error(
        code=code or "execution_error",
        message=message or f"job {payload.get('prompt_id')} ended in status {status!r}",
        hint=hint,
        details=payload,
        exit_code=exit_code,
        command=command,
    )
    raise typer.Exit(code=exit_code)


# ---------------------------------------------------------------------------
# `jobs ls` — combine /queue + /history
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobRow:
    prompt_id: str
    status: str  # "running" | "pending" | "completed" | "error" | "cancelled" | "allocated" | "executing"
    queue_position: int | None
    elapsed_seconds: float | None
    workflow_size: int | None  # number of nodes
    outputs: int
    where: str = "local"  # "local" | "cloud"
    workflow_path: str | None = None  # set when sourced from a state file
    updated_at: str | None = None  # ISO timestamp, set for state-file rows


def _gather_local_state_files(*, limit: int, orphaned_only: bool = False) -> list[JobRow]:
    """Read every state file in the jobs state dir → JobRow.

    This is the canonical "what did *I* submit via this CLI" view —
    independent of whether the server is reachable. Surfaces async submits
    that the user otherwise wouldn't see in `jobs ls`.

    When ``orphaned_only`` is True, return only rows whose state file
    has ``error.code == "watcher_crashed"`` — jobs where the
    background watcher died and was reaped. Useful for cleanup.
    """
    import re as _re

    from comfy_cli import jobs_state

    # Reasonable prompt_ids are alphanumeric + dashes + underscores (UUIDs,
    # short hex IDs). Anything wilder (e.g. legacy MagicMock leak from tests)
    # is filtered so ``jobs ls`` stays clean.
    _SANE_ID = _re.compile(r"^[A-Za-z0-9_-]{1,128}$")
    rows: list[JobRow] = []
    state_dir = jobs_state.state_dir()
    for path in sorted(state_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not _SANE_ID.match(path.stem):
            continue
        state = jobs_state.read(path.stem)
        if state is None:
            continue
        # Reap stale watchers: if the job is non-terminal and the watcher
        # PID is recorded but dead, mark the job as errored so it doesn't
        # sit as "running" forever.
        if (
            not state.is_terminal
            and state.watcher_pid is not None
            and state.watcher_pid > 0
            and not _is_pid_alive(state.watcher_pid)
        ):
            state.status = "error"
            state.error = {
                "code": "watcher_crashed",
                "message": f"Background watcher (pid {state.watcher_pid}) is no longer running.",
                "hint": "re-submit the workflow, or check `comfy jobs status <id>` against the server",
            }
            state.watcher_pid = None
            jobs_state.write(state)
        if orphaned_only:
            err = state.error or {}
            if not (isinstance(err, dict) and err.get("code") == "watcher_crashed"):
                continue
        rows.append(
            JobRow(
                prompt_id=state.prompt_id,
                status=state.status,
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=None,
                outputs=len(state.outputs or []),
                where=state.where,
                workflow_path=state.workflow,
                updated_at=state.updated_at,
            )
        )
        if len(rows) >= limit:
            break
    return rows


def _parse_epoch(ts: str | None) -> float:
    """Parse an ISO ``updated_at`` to epoch seconds; 0.0 if missing/unparseable."""
    if not ts:
        return 0.0
    try:
        return datetime.fromisoformat(ts).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _merge_jobs(state_rows: list[JobRow], server_rows: list[JobRow]) -> list[JobRow]:
    """Server's view wins for prompts it knows about (fresher status); state
    files fill in everything else (jobs the server doesn't see, e.g. cloud
    jobs viewed from a local-only `jobs ls`).
    """
    by_id: dict[str, JobRow] = {r.prompt_id: r for r in state_rows}
    for r in server_rows:
        by_id[r.prompt_id] = r

    # Sort: non-terminal first (running/pending/allocated/executing), then
    # terminal ones by updated_at desc (freshest completions first, so the
    # caller's slice keeps the newest results).
    def sort_key(r: JobRow) -> tuple[int, float | str]:
        terminal = r.status in {"completed", "error", "cancelled"}
        if terminal:
            return (1, -_parse_epoch(r.updated_at))
        return (0, "" if not r.updated_at else r.updated_at)

    return sorted(by_id.values(), key=sort_key, reverse=False)


def _gather_jobs(host: str, port: int, *, limit: int) -> list[JobRow]:
    """Pull running + pending + recent history; merge into a single ordered list."""
    rows: list[JobRow] = []
    try:
        queue = _http_get_json(f"http://{host}:{port}/queue")
    except RuntimeError:
        queue = {"queue_running": [], "queue_pending": []}

    for i, entry in enumerate(queue.get("queue_running") or []):
        prompt_id, wf = _safe_queue_entry(entry)
        rows.append(
            JobRow(
                prompt_id=prompt_id,
                status="running",
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=len(wf) if isinstance(wf, dict) else None,
                outputs=0,
            )
        )
    for i, entry in enumerate(queue.get("queue_pending") or []):
        prompt_id, wf = _safe_queue_entry(entry)
        rows.append(
            JobRow(
                prompt_id=prompt_id,
                status="pending",
                queue_position=i + 1,
                elapsed_seconds=None,
                workflow_size=len(wf) if isinstance(wf, dict) else None,
                outputs=0,
            )
        )

    try:
        history = _http_get_json(f"http://{host}:{port}/history")
    except RuntimeError:
        history = {}
    if not isinstance(history, dict):
        history = {}

    history_items = list(history.items())
    # /history is keyed by prompt_id; values include prompt + outputs. Order
    # isn't documented but recent entries are typically last — pull the tail.
    history_items = history_items[-limit:]
    for prompt_id, body in reversed(history_items):
        if not isinstance(body, dict):
            continue
        status_obj = body.get("status") or {}
        completed = status_obj.get("completed")
        # error: status_str == "error" OR any message of type execution_error
        status_str = "completed" if completed else "error"
        messages = status_obj.get("messages") or []
        for msg in messages:
            if isinstance(msg, list) and msg and msg[0] == "execution_error":
                status_str = "error"
                break
        outputs = body.get("outputs") or {}
        output_count = sum(
            len(items)
            for v in outputs.values()
            if isinstance(v, dict)
            for key in ("images", "gifs", "videos", "audio", "files")
            for items in [v.get(key) or []]
            if isinstance(items, list)
        )
        wf = body.get("prompt") or [None, None, None, None]
        wf_dict = wf[2] if isinstance(wf, list) and len(wf) > 2 else None
        rows.append(
            JobRow(
                prompt_id=str(prompt_id),
                status=status_str,
                queue_position=None,
                elapsed_seconds=None,
                workflow_size=len(wf_dict) if isinstance(wf_dict, dict) else None,
                outputs=output_count,
            )
        )
    return rows


def _safe_queue_entry(entry: Any) -> tuple[str, Any]:
    """ComfyUI /queue rows are [<num>, <prompt_id>, <prompt_dict>, ...]."""
    if isinstance(entry, list) and len(entry) >= 3:
        return str(entry[1]), entry[2]
    return ("?", None)


@app.command("ls", help="List jobs: locally-tracked async submits + server queue/history.")
@tracking.track_command("jobs")
def ls_cmd(
    host: Annotated[str | None, typer.Option(help="Server host (defaults to background or 127.0.0.1).")] = None,
    port: Annotated[int | None, typer.Option(help="Server port (defaults to background or 8188).")] = None,
    limit: Annotated[int, typer.Option(help="How many history entries to include.")] = 10,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (default) or 'cloud'. Cloud requires `comfy auth login`."),
    ] = None,
    local_only: Annotated[
        bool,
        typer.Option(
            "--local-only",
            show_default=False,
            help="Only read the on-disk state files; skip the server query. Useful when offline.",
        ),
    ] = False,
    orphaned: Annotated[
        bool,
        typer.Option(
            "--orphaned",
            show_default=False,
            help=(
                "Show only jobs whose background watcher died (error.code == "
                "watcher_crashed). Implies --local-only because the server "
                "doesn't track watcher liveness."
            ),
        ),
    ] = False,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch",
            show_default=False,
            help="Live-refresh the table every 2s (pretty mode only). Ctrl-C to exit.",
        ),
    ] = False,
):
    renderer = get_renderer()

    if watch:
        if not renderer.is_pretty():
            renderer.error(
                code="json_incompatible",
                message="--watch requires pretty mode (TTY). For JSON, poll with a shell loop.",
                hint="drop --json, or run `while true; do comfy --json jobs ls; sleep 2; done`",
            )
            raise typer.Exit(code=1)
        _watch_ls(host=host, port=port, limit=limit, where=where, local_only=local_only)
        return

    state_rows = _gather_local_state_files(limit=limit, orphaned_only=orphaned)

    # --orphaned only makes sense for state files (the server doesn't know
    # whether a watcher crashed), so skip the server query in that mode.
    if orphaned:
        local_only = True

    server_rows: list[JobRow] = []
    h, p = _resolve_host_port(host, port)
    if not local_only:
        if _is_cloud(where):
            try:
                cloud_preflight_or_exit()
                client = _cloud_client()
                server_rows = [_cloud_job_to_row(j) for j in client.list_jobs(limit=limit)]
            except typer.Exit:
                # Preflight surfaced an error envelope already. Fall through
                # to state-only view; the local files are still useful.
                pass
        else:
            try:
                _server_or_error(h, p, raise_on_missing=False)
                server_rows = _gather_jobs(h, p, limit=limit)
            except RuntimeError:
                # Server unreachable — that's fine, state files cover us.
                pass

    rows = _merge_jobs(state_rows, server_rows)[:limit]

    if renderer.is_pretty():
        _render_jobs_pretty(rows, host=h if not _is_cloud(where) else "cloud.comfy.org", port=p)
    renderer.emit(
        {
            "host": h,
            "port": p,
            "where": "cloud" if _is_cloud(where) else "local",
            "count": len(rows),
            "jobs": [_row_to_dict(r) for r in rows],
        },
        command="jobs ls",
    )


def _watch_ls(*, host, port, limit, where, local_only):
    """Rich Live refresh of the jobs table every 2s until Ctrl-C."""
    import time

    from rich.live import Live
    from rich.table import Table

    from comfy_cli.output.glyphs import status_glyph

    renderer = get_renderer()
    console = renderer.console()
    h, p = _resolve_host_port(host, port)

    def build_table() -> Table:
        state_rows = _gather_local_state_files(limit=limit)
        server_rows: list[JobRow] = []
        if not local_only:
            try:
                if _is_cloud(where):
                    client = _cloud_client()
                    server_rows = [_cloud_job_to_row(j) for j in client.list_jobs(limit=limit)]
                else:
                    server_rows = _gather_jobs(h, p, limit=limit)
            except (RuntimeError, Exception):  # noqa: BLE001 — best effort, keep refreshing
                pass
        rows = _merge_jobs(state_rows, server_rows)[:limit]

        title_loc = "cloud.comfy.org" if _is_cloud(where) else f"{h}:{p}"
        tbl = Table(
            title=f"Jobs ({title_loc}) — refreshing every 2s · Ctrl-C to exit",
            show_header=True,
            header_style="bold magenta",
            border_style="cyan",
            pad_edge=False,
        )
        tbl.add_column("prompt_id", style="bold white", no_wrap=True, overflow="fold")
        tbl.add_column("status", no_wrap=True)
        tbl.add_column("where", style="dim", no_wrap=True)
        tbl.add_column("outputs", no_wrap=True, justify="right")
        tbl.add_column("workflow", style="dim", overflow="fold")
        for r in rows:
            wf_display = ""
            if r.workflow_path:
                from pathlib import Path

                wf_display = Path(r.workflow_path).name
            tbl.add_row(
                r.prompt_id[:8] + "…" if len(r.prompt_id) > 8 else r.prompt_id,
                status_glyph(r.status),
                r.where,
                str(r.outputs) if r.outputs else "—",
                wf_display,
            )
        if not rows:
            tbl.add_row("[dim]no jobs[/dim]", "", "", "", "")
        return tbl

    try:
        with Live(build_table(), console=console, refresh_per_second=2) as live:
            while True:
                time.sleep(2)
                live.update(build_table())
    except KeyboardInterrupt:
        # Clean exit — Rich Live tears down automatically.
        return


def _row_to_dict(r: JobRow) -> dict:
    return {
        "prompt_id": r.prompt_id,
        "status": r.status,
        "queue_position": r.queue_position,
        "workflow_size": r.workflow_size,
        "outputs": r.outputs,
        "where": r.where,
        "workflow_path": r.workflow_path,
        "updated_at": r.updated_at,
    }


def _render_jobs_pretty(rows: list[JobRow], *, host: str, port: int) -> None:
    from rich.table import Table

    from comfy_cli.config_manager import ConfigManager
    from comfy_cli.output.branding import branded_panel
    from comfy_cli.output.glyphs import status_glyph

    renderer = get_renderer()
    is_cloud = str(host).startswith("http") or host == "cloud.comfy.org"
    where_label = "cloud" if is_cloud else "local"
    host_label = host if is_cloud else f"{host}:{port}"

    if not rows:
        empty = "[dim]No jobs.[/dim]\n[dim]→ comfy run --workflow X.json[/dim]"
        renderer.console().print(
            branded_panel(
                empty,
                title="jobs",
                version=ConfigManager().get_cli_version(),
                where=where_label,
                host=host_label,
            )
        )
        return

    tbl = Table(
        show_header=True,
        header_style="bold magenta",
        border_style="dim",
        pad_edge=False,
        expand=True,
    )
    tbl.add_column("prompt_id", style="bold white", no_wrap=True, overflow="fold")
    tbl.add_column("status", no_wrap=True)
    tbl.add_column("queue", no_wrap=True, justify="right")
    tbl.add_column("nodes", no_wrap=True, justify="right")
    tbl.add_column("outputs", no_wrap=True, justify="right")
    for r in rows:
        tbl.add_row(
            r.prompt_id[:8] + "…" if len(r.prompt_id) > 8 else r.prompt_id,
            status_glyph(r.status),
            str(r.queue_position) if r.queue_position is not None else "—",
            str(r.workflow_size) if r.workflow_size is not None else "—",
            str(r.outputs) if r.outputs else "—",
        )

    renderer.console().print(
        branded_panel(
            tbl,
            title="jobs",
            version=ConfigManager().get_cli_version(),
            where=where_label,
            host=host_label,
        )
    )


# ---------------------------------------------------------------------------
# `jobs status` — single prompt_id from /history (or /queue if still in flight)
# ---------------------------------------------------------------------------


@app.command("status", help="Show the status of a single prompt_id (local or --where cloud).")
@tracking.track_command("jobs")
def status_cmd(
    prompt_id: Annotated[str, typer.Argument(help="The prompt_id returned by `comfy run`.")],
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (default) or 'cloud'."),
    ] = None,
):
    renderer = get_renderer()
    if _is_cloud(where):
        return _cloud_status(prompt_id)

    h, p = _resolve_host_port(host, port)
    _server_or_error(h, p)

    snapshot = _snapshot(h, p, prompt_id)
    if snapshot is None:
        renderer.error(
            code="prompt_not_found",
            message=f"No prompt with id {prompt_id!r} on {h}:{p}.",
            hint="check `comfy jobs ls`; very old prompts may have been pruned from /history",
            details={"prompt_id": prompt_id, "host": h, "port": p},
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        _render_status_pretty(snapshot, host=h, port=p)
    renderer.emit(snapshot, command="jobs status")


def _snapshot(host: str, port: int, prompt_id: str) -> dict | None:
    # First: is it in the queue?
    try:
        q = _http_get_json(f"http://{host}:{port}/queue")
    except RuntimeError:
        q = {}
    for state, key in (("running", "queue_running"), ("pending", "queue_pending")):
        for entry in q.get(key) or []:
            pid, wf = _safe_queue_entry(entry)
            if pid == prompt_id:
                return {
                    "prompt_id": prompt_id,
                    "status": state,
                    "workflow_size": len(wf) if isinstance(wf, dict) else None,
                    "outputs": [],
                    "outputs_by_node": {},
                    "outputs_by_item": {},
                    "host": host,
                    "port": port,
                }

    # Then: history.
    try:
        h = _http_get_json(f"http://{host}:{port}/history/{prompt_id}")
    except RuntimeError:
        return None
    if not isinstance(h, dict) or prompt_id not in h:
        return None
    body = h[prompt_id]
    if not isinstance(body, dict):
        return None
    status_obj = body.get("status") or {}
    completed = bool(status_obj.get("completed"))
    error_detail = None
    interrupted = False
    for msg in status_obj.get("messages") or []:
        if isinstance(msg, list) and msg:
            if msg[0] == "execution_error":
                error_detail = msg[1] if len(msg) > 1 else None
            elif msg[0] == "execution_interrupted":
                interrupted = True
    # Flatten the node-keyed /history outputs into URL entries that keep
    # their producing-node association — same flatten the cloud snapshot
    # uses, so the grouped keys match the cloud envelope shape exactly.
    from comfy_cli import jobs_state
    from comfy_cli.comfy_client import _group_outputs, extract_output_entries

    node_outputs: list[dict] = []
    for entry in extract_output_entries(body):
        q = urllib.parse.urlencode({k: entry[k] for k in ("filename", "subfolder", "type")})
        node_outputs.append({**entry, "url": f"http://{host}:{port}/view?{q}"})
    output_urls = [o["url"] for o in node_outputs]
    # The compose item_map (foreach item -> node ids) lives on the job state
    # file, written at submit time by `comfy run`.
    try:
        job = jobs_state.read(prompt_id)
    except ValueError:  # unsafe prompt_id — no state file to join against
        job = None
    item_map = job.item_map if job is not None else None
    outputs_by_node, outputs_by_item = _group_outputs(node_outputs, item_map)

    return {
        "prompt_id": prompt_id,
        "status": ("error" if error_detail else "completed" if completed else "cancelled" if interrupted else "queued"),
        "workflow_size": None,
        "outputs": output_urls,
        "outputs_by_node": outputs_by_node,
        "outputs_by_item": outputs_by_item,
        "error": error_detail,
        "host": host,
        "port": port,
    }


def _render_status_pretty(snap: dict, *, host: str, port: int) -> None:
    from rich.console import Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    renderer = get_renderer()
    status = snap["status"]
    badge = {
        "running": Text.assemble(("● ", "bold green"), ("running", "bold green")),
        "pending": Text.assemble(("◌ ", "bold yellow"), ("pending", "bold yellow")),
        "completed": Text.assemble(("✓ ", "bold green"), ("completed", "bold green")),
        "queued": Text.assemble(("◌ ", "dim"), ("queued", "dim")),
        "error": Text.assemble(("✗ ", "bold red"), ("error", "bold red")),
    }.get(status, Text(status))

    tbl = Table.grid(padding=(0, 2), expand=False)
    tbl.add_column(justify="right", style="dim", no_wrap=True)
    tbl.add_column(overflow="fold")
    tbl.add_row("prompt_id", snap["prompt_id"])
    tbl.add_row("status", badge)
    if snap.get("outputs"):
        tbl.add_row("outputs", "\n".join(snap["outputs"]))
    if snap.get("error"):
        tbl.add_row("error", str(snap["error"])[:600])

    renderer.console().print(
        Panel(
            Group(tbl),
            title=Text(f"job on {host}:{port}", style="bold cyan"),
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
    )


# ---------------------------------------------------------------------------
# `jobs wait` — block until N prompt_ids are all terminal (multi-job wait)
# ---------------------------------------------------------------------------


def _wait_fetch_snapshot(
    prompt_id: str, *, cloud: bool, host: str | None, port: int | None, server_up: bool
) -> dict | None:
    """Best-effort single-job status snapshot for the wait loop.

    cloud -> /api/jobs/<id>; local -> live /history when the server is up,
    else fall back to the on-disk state file the async watcher maintains.
    """
    if cloud:
        return _cloud_status_snapshot(prompt_id)
    if server_up and host is not None and port is not None:
        snap = _snapshot(host, port, prompt_id)
        if snap is not None:
            return snap
    from comfy_cli import jobs_state

    try:
        st = jobs_state.read(prompt_id)
    except ValueError:
        st = None
    if st is None:
        return None
    err_msg = st.error.get("message") if isinstance(st.error, dict) else None
    return {
        "prompt_id": prompt_id,
        "status": st.status,
        "outputs": list(st.outputs or []),
        "error_message": err_msg,
    }


def _wait_loop(prompt_ids, fetch, *, poll_interval: float, deadline: float, renderer):
    """Poll ``fetch(pid)`` for each id until all are terminal or the deadline
    passes. Emits a ``settled`` NDJSON event as each job finishes. Returns
    ``(snapshots, still_pending)``.
    """
    from comfy_cli import cancellation, jobs_state

    pending = list(prompt_ids)
    snapshots: dict[str, dict] = {}
    total = len(pending)
    cancel_token = cancellation.get_token()
    while pending:
        still: list[str] = []
        for pid in pending:
            snap = fetch(pid)
            status = (snap or {}).get("status")
            if status in jobs_state.TERMINAL_STATUSES:
                snapshots[pid] = snap if isinstance(snap, dict) else {"prompt_id": pid, "status": status}
                renderer.event("settled", prompt_id=pid, status=status, settled=len(snapshots), total=total)
            else:
                still.append(pid)
        pending = still
        if not pending:
            break
        if time.time() >= deadline or (cancel_token is not None and cancel_token.is_set()):
            break
        time.sleep(max(0.0, min(poll_interval, deadline - time.time())))
    return snapshots, pending


def _gather_waitable_ids(cloud: bool) -> list[str]:
    """Every non-terminal locally-tracked prompt_id whose ``where`` matches routing."""
    from comfy_cli import jobs_state

    want = "cloud" if cloud else "local"
    out: list[str] = []
    try:
        paths = sorted(jobs_state.state_dir().glob("*.json"))
    except OSError:
        return out
    for path in paths:
        st = jobs_state.read(path.stem)
        if st is None or st.where != want or st.is_terminal:
            continue
        out.append(st.prompt_id)
    return out


def _render_wait_pretty(summary: dict) -> None:
    from rich.table import Table
    from rich.text import Text

    badge = {
        "completed": ("✓", "bold green"),
        "error": ("✗", "bold red"),
        "cancelled": ("⊘", "bold yellow"),
        "timed_out": ("⏱", "bold yellow"),
    }
    tbl = Table(
        title=f"waited on {summary['total']} job(s) — {summary['elapsed_seconds']}s",
        border_style="cyan",
        show_header=True,
    )
    tbl.add_column("prompt_id", style="dim", no_wrap=True)
    tbl.add_column("status")
    for r in summary["jobs"]:
        glyph, style = badge.get(r["status"], ("•", "white"))
        tbl.add_row(r["prompt_id"], Text(f"{glyph} {r['status']}", style=style))
    get_renderer().console().print(tbl)


@app.command("wait", help="Block until ALL given prompt_ids reach a terminal state; emit a summary.")
@tracking.track_command("jobs")
def wait_cmd(
    prompt_ids: Annotated[
        list[str] | None,
        typer.Argument(help="prompt_ids to wait on (omit and use --all to wait on every tracked job)."),
    ] = None,
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    where: Annotated[str | None, typer.Option("--where", help="'local' (default) or 'cloud'.")] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", help="Seconds between status polls (these are long jobs; don't hammer)."),
    ] = 5.0,
    timeout: Annotated[float, typer.Option("--timeout", help="Give up after this many seconds total.")] = 1800.0,
    wait_all: Annotated[bool, typer.Option("--all", help="Wait on all locally-tracked non-terminal jobs.")] = False,
):
    renderer = get_renderer()
    cloud = _is_cloud(where)

    ids = list(prompt_ids or [])
    if wait_all:
        ids.extend(_gather_waitable_ids(cloud))
    ids = list(dict.fromkeys(ids))  # de-dup, preserve order
    if not ids:
        renderer.error(
            code="no_prompt_ids",
            message="no prompt_ids to wait on",
            hint="pass one or more prompt_ids, or --all to wait on every tracked job",
        )
        raise typer.Exit(code=2)

    h: str | None = None
    p: int | None = None
    server_up = False
    if cloud:
        cloud_preflight_or_exit()
    else:
        h, p = _resolve_host_port(host, port)
        server_up = _server_or_error(h, p, raise_on_missing=False)

    start = time.time()
    deadline = start + timeout

    def fetch(pid: str) -> dict | None:
        return _wait_fetch_snapshot(pid, cloud=cloud, host=h, port=p, server_up=server_up)

    if renderer.is_pretty():
        renderer.console().print(
            f"[bold]Waiting on {len(ids)} job(s)[/bold] [dim](poll {poll_interval}s, timeout {timeout:.0f}s)[/dim]"
        )

    snapshots, pending = _wait_loop(ids, fetch, poll_interval=poll_interval, deadline=deadline, renderer=renderer)

    jobs_list: list[dict] = []
    for pid in ids:
        if pid in pending:
            jobs_list.append({"prompt_id": pid, "status": "timed_out", "ok": False})
            continue
        snap = snapshots.get(pid) or {"prompt_id": pid, "status": "unknown"}
        status = str(snap.get("status") or "unknown")
        row: dict = {"prompt_id": pid, "status": status, "ok": status == "completed"}
        if snap.get("outputs"):
            row["outputs"] = snap["outputs"]
        if snap.get("error_message"):
            row["error_message"] = snap["error_message"]
        jobs_list.append(row)

    completed = sum(1 for r in jobs_list if r["status"] == "completed")
    failed = sum(1 for r in jobs_list if r["status"] == "error")
    cancelled = sum(1 for r in jobs_list if r["status"] == "cancelled")
    timed_out = sum(1 for r in jobs_list if r["status"] == "timed_out")
    summary = {
        "total": len(ids),
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "timed_out": timed_out,
        "elapsed_seconds": round(time.time() - start, 2),
        "jobs": jobs_list,
    }
    where_label = "cloud" if cloud else "local"

    if renderer.is_pretty():
        _render_wait_pretty(summary)

    if failed == 0 and cancelled == 0 and timed_out == 0:
        renderer.emit(summary, command="jobs wait", where=where_label)
        return

    # Literal codes (not a variable) so the error-code registry ratchet can
    # AST-scan them. execution_error/cancelled are already registered; wait_timeout
    # is registered alongside no_prompt_ids in comfy_cli/error_codes.py.
    msg = f"{completed}/{len(ids)} completed — {failed} failed, {cancelled} cancelled, {timed_out} timed out"
    if failed:
        renderer.error(code="execution_error", message=msg, details=summary, exit_code=1, command="jobs wait")
        raise typer.Exit(code=1)
    if cancelled:
        renderer.error(code="cancelled", message=msg, details=summary, exit_code=130, command="jobs wait")
        raise typer.Exit(code=130)
    renderer.error(
        code="wait_timeout",
        message=msg,
        hint="the jobs may still be running — raise `--timeout`, or check `comfy jobs status <id>`",
        details=summary,
        exit_code=1,
        command="jobs wait",
    )
    raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# `jobs cancel` — stop a running or pending prompt, locally or on cloud
# ---------------------------------------------------------------------------


@app.command(
    "cancel",
    help="Cancel a job. Idempotent — calling on an already-terminal prompt returns ok.",
)
@tracking.track_command("jobs")
def cancel_cmd(
    prompt_id: Annotated[str, typer.Argument(help="The prompt_id to cancel.")],
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (default) or 'cloud'."),
    ] = None,
):
    if _is_cloud(where):
        return _cloud_cancel(prompt_id)
    h, p = _resolve_host_port(host, port)
    _server_or_error(h, p)
    return _local_cancel(prompt_id, h, p)


def _local_cancel(prompt_id: str, host: str, port: int) -> None:
    """Cancel a local prompt by removing it from the pending queue AND
    interrupting any in-flight execution. ComfyUI splits these into two
    endpoints; we hit both so the call works regardless of phase.

    Returns 200 (ok) regardless of whether the prompt was actually
    queued/running — mirrors cloud's idempotent behavior.
    """
    renderer = get_renderer()
    base = f"http://{host}:{port}"

    # 1. Remove from the pending queue (no-op if not pending).
    queue_body = json.dumps({"delete": [prompt_id]}).encode("utf-8")
    queue_req = urllib.request.Request(
        f"{base}/queue", data=queue_body, method="POST", headers={"Content-Type": "application/json"}
    )
    queue_ok = True
    try:
        with urllib.request.urlopen(queue_req, timeout=10) as resp:
            _ = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        # Server refused the delete; common when the prompt isn't in queue.
        # Don't fail the whole command — try the interrupt next.
        queue_ok = False

    # 2. Interrupt only if THIS prompt is the one currently executing.
    #    /interrupt takes NO prompt_id — it kills whatever is running — so
    #    blindly posting it after a pending-job delete would also abort an
    #    unrelated running job ("cancel B" silently cancelling A). Gate on
    #    /queue's queue_running list; the queue delete above already covers
    #    pending jobs.
    interrupt_ok = True
    try:
        queue = _http_get_json(f"{base}/queue")
        queue_reachable = True
    except RuntimeError:
        queue = {}
        queue_reachable = False
    running_ids = {str(_safe_queue_entry(entry)[0]) for entry in (queue.get("queue_running") or [])}
    if prompt_id in running_ids:
        interrupt_req = urllib.request.Request(f"{base}/interrupt", method="POST")
        try:
            with urllib.request.urlopen(interrupt_req, timeout=10) as resp:
                _ = resp.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            interrupt_ok = False

    if not queue_ok and not queue_reachable:
        renderer.error(
            code="cancel_failed",
            message=f"both /queue delete and /queue status failed on {host}:{port}",
            hint="check the server is still reachable",
            details={"host": host, "port": port, "prompt_id": prompt_id},
        )
        raise typer.Exit(code=1)

    payload = {
        "prompt_id": prompt_id,
        "where": "local",
        "host": host,
        "port": port,
        "queue_delete_ok": queue_ok,
        "interrupt_ok": interrupt_ok,
    }
    from comfy_cli import jobs_state

    existing = jobs_state.read(prompt_id)
    if existing is not None:
        existing.status = "cancelled"
        jobs_state.write(existing)

    if renderer.is_pretty():
        from rich.text import Text

        msg = Text.from_markup(f"  [bold green]✓[/bold green]  cancel sent for [cyan]{prompt_id[:8]}…[/cyan]")
        renderer.console().print(msg)
    renderer.emit(payload, command="jobs cancel")


def _cloud_cancel(prompt_id: str) -> None:
    """Cancel a cloud job via ``POST /api/jobs/<id>/cancel`` — idempotent."""
    cloud_preflight_or_exit()
    renderer = get_renderer()

    from comfy_cli.target import resolve_target

    target = resolve_target(where="cloud")
    # Quote prompt_id into the path segment so a hostile/malformed value can't
    # escape (e.g. ``../foo`` → ``%2E%2E%2Ffoo``). Cloud rejects bad UUIDs
    # upstream too; encoding here is defense in depth.
    url = target.url("jobs", urllib.parse.quote(prompt_id, safe=""), "cancel")
    req = urllib.request.Request(url, data=b"", method="POST")
    if target.api_key:
        req.add_header("X-API-Key", target.api_key)
    elif target.auth_token:
        req.add_header("Authorization", f"Bearer {target.auth_token}")

    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
    except urllib.error.HTTPError as e:
        body_text = (e.read() or b"")[:1000].decode("utf-8", "replace")
        if e.code == 404:
            renderer.error(
                code="prompt_not_found",
                message=f"no cloud job with id {prompt_id!r}",
                hint="check `comfy jobs ls --where cloud`",
                details={"prompt_id": prompt_id},
            )
        else:
            renderer.error(
                code="cloud_http_error",
                message=f"HTTP {e.code} cancelling {prompt_id}",
                hint="check auth and that the job exists",
                details={"status": e.code, "body": body_text, "prompt_id": prompt_id},
            )
        raise typer.Exit(code=1) from e
    except (urllib.error.URLError, OSError) as e:
        renderer.error(
            code="cloud_http_error",
            message=f"cancel failed: {e}",
            hint="check network / `comfy auth whoami`",
        )
        raise typer.Exit(code=1) from e

    parsed: dict | None
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = None
    payload = {
        "prompt_id": prompt_id,
        "where": "cloud",
        "base_url": target.base_url,
        "response": parsed if isinstance(parsed, dict) else None,
    }
    if renderer.is_pretty():
        from rich.text import Text

        renderer.console().print(
            Text.from_markup(f"  [bold green]✓[/bold green]  cancel sent for [cyan]{prompt_id[:8]}…[/cyan]")
        )
    renderer.emit(payload, command="jobs cancel", where="cloud")


# ---------------------------------------------------------------------------
# `jobs watch` — tail WS events live, filtered on prompt_id
# ---------------------------------------------------------------------------


@app.command("watch", help="Tail live execution events for a prompt_id (WS local / polling cloud).")
@tracking.track_command("jobs")
def watch_cmd(
    prompt_id: Annotated[str, typer.Argument(help="The prompt_id returned by `comfy run`.")],
    host: Annotated[str | None, typer.Option()] = None,
    port: Annotated[int | None, typer.Option()] = None,
    timeout: Annotated[int, typer.Option(help="Per-recv (or per-poll) timeout in seconds.")] = 30,
    where: Annotated[
        str | None,
        typer.Option("--where", help="'local' (WebSocket) or 'cloud' (HTTP polling)."),
    ] = None,
    poll_interval: Annotated[
        float,
        typer.Option("--poll-interval", help="Cloud-only: seconds between status polls."),
    ] = 1.5,
    max_wait: Annotated[
        float,
        typer.Option("--max-wait", help="Cloud-only: give up after this many seconds total."),
    ] = 600.0,
):
    renderer = get_renderer()
    if _is_cloud(where):
        return _cloud_watch(prompt_id, poll_interval=poll_interval, max_wait=max_wait)

    h, p = _resolve_host_port(host, port)
    _server_or_error(h, p)

    # If the job already finished, just print status and return — there will
    # be no more WS events.
    snap = _snapshot(h, p, prompt_id)
    if snap and snap["status"] in {"completed", "error", "cancelled"}:
        if renderer.is_pretty():
            renderer.console().print(f"[dim]Prompt {prompt_id} already {snap['status']}; nothing more to watch.[/dim]")
            _render_status_pretty(snap, host=h, port=p)
        _emit_terminal(renderer, snap, command="jobs watch")
        return

    ws = WebSocket()
    client_id = str(uuid.uuid4())
    try:
        ws.connect(f"ws://{h}:{p}/ws?clientId={client_id}")
    except (WebSocketException, ConnectionError, OSError) as e:
        renderer.error(
            code="ws_disconnected",
            message=f"Could not open WebSocket: {e}",
            hint="check the server is reachable; try `comfy jobs status` instead",
        )
        raise typer.Exit(code=1)

    token = cancellation.get_token()
    token.on_cancel(lambda: _safe_close_ws(ws))

    ws.settimeout(timeout)

    completed_nodes: set[str] = set()
    outputs: list[str] = []
    saw_any_event = False
    missing_deadline: float | None = None
    end_reason: str | None = None
    end_details: Any = None
    start = time.time()

    if renderer.is_pretty():
        renderer.console().print(f"[bold]Watching prompt[/bold] {prompt_id} on {h}:{p}   [dim](Ctrl-C to stop)[/dim]")

    try:
        while True:
            try:
                raw = ws.recv()
            except WebSocketTimeoutException:
                # If the job moved to completed between recvs, exit cleanly.
                snap = _snapshot(h, p, prompt_id)
                if snap and snap["status"] in {"completed", "error", "cancelled"}:
                    end_reason = snap["status"]
                    end_details = snap
                    outputs.extend(snap.get("outputs") or [])
                    break
                # Bounded wait for an unknown prompt: if the server has never
                # heard of this prompt_id (no snapshot) and no events have
                # arrived, don't loop forever on a typoed/already-pruned id —
                # mirror the cloud path's deadline and surface prompt_not_found.
                if snap is None and not saw_any_event:
                    if missing_deadline is None:
                        missing_deadline = time.time() + max(timeout, 1)
                    elif time.time() >= missing_deadline:
                        # The enclosing `finally` closes the socket.
                        renderer.error(
                            code="prompt_not_found",
                            message=f"prompt {prompt_id} not found on {h}:{p}",
                            hint="check the prompt_id; it may be a typo or already pruned",
                            details={"prompt_id": prompt_id, "host": h, "port": p},
                        )
                        raise typer.Exit(code=1)
                else:
                    missing_deadline = None
                continue
            except (WebSocketException, ConnectionError, OSError) as e:
                # Cancellation closes the socket out from under recv(). Check
                # the token before classifying as "server disconnected".
                if token.is_set():
                    end_reason = "cancelled"
                    break
                renderer.error(
                    code="ws_disconnected",
                    message=f"Lost connection while watching {prompt_id}: {e}",
                    hint="re-run `comfy jobs status` to check final state",
                )
                raise typer.Exit(code=1) from e
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            data = msg.get("data") or {}
            if data.get("prompt_id") != prompt_id:
                continue
            saw_any_event = True
            t = msg.get("type")
            if t == "executing":
                node = data.get("node")
                if node is None:
                    end_reason = "completed"
                    break
                if renderer.is_pretty():
                    renderer.console().print(f"[dim]→[/dim] executing node [bold]{node}[/bold]")
                renderer.event("executing", node=str(node), prompt_id=prompt_id)
            elif t == "execution_cached":
                nodes = data.get("nodes") or []
                for n in nodes:
                    completed_nodes.add(str(n))
                if renderer.is_pretty():
                    renderer.console().print(f"[dim]✓[/dim] cached: {len(nodes)} node(s)")
                renderer.event(
                    "execution_cached",
                    nodes=[str(n) for n in nodes],
                    prompt_id=prompt_id,
                )
            elif t == "progress":
                renderer.throttled_event(
                    f"progress:{data.get('node')}",
                    "progress",
                    max_hz=10,
                    node=str(data.get("node")),
                    completed=data.get("value"),
                    total=data.get("max"),
                    prompt_id=prompt_id,
                )
            elif t == "executed":
                node = str(data.get("node"))
                completed_nodes.add(node)
                output = data.get("output") or {}
                for key in ("images", "gifs", "videos", "audio", "files"):
                    for item in output.get(key) or []:
                        if isinstance(item, dict) and "filename" in item:
                            q = urllib.parse.urlencode(
                                {k: item[k] for k in ("filename", "subfolder", "type") if k in item}
                            )
                            url = f"http://{h}:{p}/view?{q}"
                            outputs.append(url)
                            if renderer.is_pretty():
                                renderer.console().print(f"[bold green]✓[/bold green] output: [cyan]{url}[/cyan]")
                            renderer.event("output", url=url, prompt_id=prompt_id)
                renderer.event("executed", node=node, prompt_id=prompt_id)
            elif t == "execution_error":
                end_reason = "error"
                end_details = data
                break
    finally:
        _safe_close_ws(ws)

    elapsed = time.time() - start
    final_status = end_reason or ("completed" if completed_nodes else "unknown")
    if renderer.is_pretty():
        from rich.text import Text

        if final_status == "completed":
            renderer.console().print(
                Text.assemble(("\n✓ ", "bold green"), ("completed", "bold green"), (f"  in {elapsed:.1f}s", "dim"))
            )
        elif final_status == "error":
            renderer.console().print(Text.assemble(("\n✗ ", "bold red"), ("error", "bold red")))
        elif final_status == "cancelled":
            renderer.console().print(Text.assemble(("\n⊘ ", "yellow"), ("cancelled", "yellow")))

    payload = {
        "prompt_id": prompt_id,
        "status": final_status,
        "outputs": outputs,
        "completed_nodes": sorted(completed_nodes),
        "elapsed_seconds": elapsed,
        "host": h,
        "port": p,
    }
    if end_details is not None:
        payload["details"] = end_details if isinstance(end_details, dict) else {"raw": end_details}
    if not saw_any_event and final_status == "unknown":
        payload["hint"] = "watch returned without events; the prompt may already have completed"
    _emit_terminal(renderer, payload, command="jobs watch")


# ---------------------------------------------------------------------------
# Cloud handlers — /api/jobs, /api/jobs/<id>, /api/history_v2/<id>
# ---------------------------------------------------------------------------


def _is_cloud(where: str | None) -> bool:
    """Resolve the routing target using the same precedence as the rest of
    the CLI: per-command ``--where`` flag > ``COMFY_WHERE`` env var >
    persisted ``where_default`` config > default ``local``.

    Honoring the env var matters because the top-level ``comfy --where
    cloud`` flag is sugar for ``COMFY_WHERE=cloud``: ``cmdline.py`` sets
    the env so every subcommand inherits the routing decision without
    repeating the flag. A previous implementation looked only at the
    per-command parameter, which silently dropped the top-level flag
    for ``jobs ls/status/watch``.
    """
    from comfy_cli import where as where_module

    try:
        decision = where_module.resolve_default(flag=where)
    except ValueError:
        # Invalid value — fall back to local; the validating command
        # (cmdline.py top-level option) will surface ``where_invalid``.
        return False
    return decision.target is where_module.WhereTarget.CLOUD


def _cloud_job_to_row(j: dict) -> JobRow:
    """Map a /api/jobs entry to our JobRow shape."""
    status_map = {"completed": "completed", "success": "completed", "failed": "error", "error": "error"}
    raw_status = (j.get("status") or "").lower()
    status = status_map.get(raw_status, raw_status or "pending")
    outputs = int(j.get("outputs_count") or 0)
    return JobRow(
        prompt_id=str(j.get("id") or ""),
        status=status,
        queue_position=None,
        elapsed_seconds=None,
        workflow_size=None,
        outputs=outputs,
    )


def _cloud_client():
    """Construct a unified Client targeting cloud. Raises if not signed in.

    Observer commands (status/ls/watch snapshots) must never clear the shared
    OAuth session on a fatal refresh error: batch workloads run dozens of these
    concurrently, and one spurious invalid_grant wiping the login mid-run turns
    a transient hiccup into a hard logout. Session lifecycle belongs to
    login/logout and the foreground submit path.
    """
    from comfy_cli.comfy_client import Client, Unauthenticated
    from comfy_cli.target import resolve_target

    target = resolve_target(where="cloud")
    try:
        return Client(target, clear_session_on_auth_failure=False)
    except Unauthenticated as e:
        renderer = get_renderer()
        renderer.error(code="cloud_unauthorized", message=str(e), hint="run: comfy auth login")
        raise typer.Exit(code=1) from e


def _cloud_ls(*, limit: int) -> None:
    from comfy_cli.comfy_client import HTTPError

    cloud_preflight_or_exit()
    renderer = get_renderer()
    client = _cloud_client()
    try:
        jobs = client.list_jobs(limit=limit)
    except HTTPError as e:
        renderer.error(
            code="cloud_http_error",
            message=f"Cloud /api/jobs failed (HTTP {e.status}): {e.message}",
            details={"status": e.status, "body": e.body[:1000]},
        )
        raise typer.Exit(code=1)

    rows = [_cloud_job_to_row(j) for j in jobs[:limit]]
    if renderer.is_pretty():
        _render_jobs_pretty(rows, host=client.target.base_url, port=0)
    renderer.emit(
        {
            "base_url": client.target.base_url,
            "count": len(rows),
            "jobs": [_row_to_dict(r) for r in rows],
        },
        command="jobs ls",
        where="cloud",
    )


def _cloud_status_snapshot(prompt_id: str) -> dict | None:
    """Compose a cloud snapshot from /api/jobs/<id> + /api/history_v2/<id>."""
    from comfy_cli import jobs_state
    from comfy_cli.comfy_client import _group_outputs

    client = _cloud_client()
    status = client.get_job_status(prompt_id)
    if status is None:
        return None
    raw = (status.get("status") or "").lower()
    state = {
        "success": "completed",
        "completed": "completed",
        "failed": "error",
        "error": "error",
        "non_retryable_error": "error",
        "lost": "error",
    }.get(raw, raw or "pending")

    outputs: list[str] = []
    outputs_by_node: dict[str, list[str]] = {}
    outputs_by_item: dict[str, list[str]] = {}
    if state == "completed":
        record = client.get_history(prompt_id)
        if record:
            node_outputs = client.extract_outputs(record)
            outputs = [o["url"] for o in node_outputs]
            # The compose item_map (foreach item -> node ids) lives on the
            # job state file, written at submit time by `comfy run`.
            job = jobs_state.read(prompt_id)
            item_map = job.item_map if job is not None else None
            outputs_by_node, outputs_by_item = _group_outputs(node_outputs, item_map)

    return {
        "prompt_id": prompt_id,
        "status": state,
        "outputs": outputs,
        "outputs_by_node": outputs_by_node,
        "outputs_by_item": outputs_by_item,
        "assigned_inference": status.get("assigned_inference"),
        "error_message": status.get("error_message"),
        "created_at": status.get("created_at"),
        "updated_at": status.get("updated_at"),
        "base_url": client.target.base_url,
    }


def _cloud_status(prompt_id: str) -> None:
    cloud_preflight_or_exit()
    renderer = get_renderer()
    snap = _cloud_status_snapshot(prompt_id)
    if snap is None:
        renderer.error(
            code="prompt_not_found",
            message=f"No cloud prompt with id {prompt_id!r}.",
            hint="check `comfy jobs ls --where cloud`",
            details={"prompt_id": prompt_id},
        )
        raise typer.Exit(code=1)

    if renderer.is_pretty():
        from rich.table import Table

        tbl = Table(title=f"Cloud prompt {prompt_id[:8]}…", border_style="cyan", show_header=False)
        tbl.add_column(style="bold cyan")
        tbl.add_column()
        tbl.add_row("status", snap["status"])
        if snap.get("assigned_inference"):
            tbl.add_row("inference", snap["assigned_inference"])
        if snap.get("created_at"):
            tbl.add_row("created", snap["created_at"])
        if snap.get("updated_at"):
            tbl.add_row("updated", snap["updated_at"])
        if snap.get("error_message"):
            tbl.add_row("error", snap["error_message"])
        for u in snap.get("outputs") or []:
            tbl.add_row("output", u)
        renderer.console().print(tbl)
    renderer.emit(snap, command="jobs status", where="cloud")


def _cloud_watch(prompt_id: str, *, poll_interval: float, max_wait: float) -> None:
    """Poll cloud's job status, emit NDJSON events on each transition."""
    cloud_preflight_or_exit()
    renderer = get_renderer()
    base_url = _cloud_client().target.base_url

    cancel_token = cancellation.get_token()
    deadline = time.time() + max_wait
    last_state: str | None = None
    start = time.time()

    if renderer.is_pretty():
        renderer.console().print(
            f"[bold]Watching cloud prompt[/bold] {prompt_id}   [dim]({base_url}, Ctrl-C to stop)[/dim]"
        )

    final_snap: dict | None = None
    while not cancel_token.is_set():
        snap = _cloud_status_snapshot(prompt_id)
        if snap is None:
            # Not yet known to the cloud — wait briefly.
            if time.time() >= deadline:
                renderer.error(
                    code="prompt_not_found",
                    message=f"prompt {prompt_id} not found on cloud after {max_wait}s",
                    details={"prompt_id": prompt_id, "base_url": base_url},
                )
                raise typer.Exit(code=1)
            time.sleep(min(poll_interval, deadline - time.time()))
            continue

        if snap["status"] != last_state:
            last_state = snap["status"]
            if renderer.is_pretty():
                renderer.console().print(f"[dim]→[/dim] state [bold]{last_state}[/bold]")
            renderer.event("state", prompt_id=prompt_id, status=last_state)

        if snap["status"] in {"completed", "error", "cancelled"}:
            for u in snap.get("outputs") or []:
                renderer.event("output", url=u, prompt_id=prompt_id)
                if renderer.is_pretty():
                    renderer.console().print(f"[bold green]✓[/bold green] output: [cyan]{u}[/cyan]")
            final_snap = snap
            break

        if time.time() >= deadline:
            renderer.error(
                code="cloud_timeout",
                message=f"prompt {prompt_id} still {snap['status']} after {max_wait}s",
                hint="raise --max-wait or re-run with --where cloud",
                details=snap,
            )
            raise typer.Exit(code=1)

        time.sleep(min(poll_interval, deadline - time.time()))

    payload = final_snap or {"prompt_id": prompt_id, "status": "cancelled"}
    payload["elapsed_seconds"] = time.time() - start
    _emit_terminal(renderer, payload, command="jobs watch", where="cloud")


def _safe_close_ws(ws) -> None:
    try:
        ws.close()
    except Exception:  # noqa: BLE001
        pass
