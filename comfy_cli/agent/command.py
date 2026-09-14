"""``comfy agent`` — the Typer surface over :mod:`comfy_cli.agent`.

``permissions`` shows the agent's requests waiting for a human, what the local
agent may reach beyond its defaults, and whether one is running; ``allow
--approve <id>`` / ``deny <id>`` answer a request the agent's ``request_path``
/ ``request_host`` tools recorded (the model cannot answer its own request);
``allow --path`` / ``--host`` grants a folder or a host outright.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.agent import (
    DISCOVERY_FILE,
    StateError,
    UnknownRequestError,
    allow_host,
    allow_path,
    approve,
    data_dir,
    deny,
    read_state,
    vet_host,
    vet_path,
)
from comfy_cli.http import build_http_only_opener
from comfy_cli.output import get_renderer, rprint
from comfy_cli.output.sanitize import sanitize_markup as esc

app = typer.Typer(
    no_args_is_help=True,
    help="The local comfy agent: what it may reach, and how to let it reach more.",
)

_DATA_DIR_OPT = Annotated[
    str | None,
    typer.Option(
        "--data-dir", show_default=False, help="The agent's data dir (default: AGENT_DATA_DIR, else ~/.comfy-agent)."
    ),
]

PICKUP_NOTE = "a running agent picks this up within about 20 seconds; otherwise it applies at the agent's next start"
REFUSED_HINT = (
    "a narrower folder that holds only what is needed may be allowed; credential stores, the home folder "
    "and the whole disk never are. A host must be a full name or a routable address"
)
REFUSED_REQUEST_HINT = (
    "the request stays pending and nothing was approved: `comfy agent deny <id>` clears it, or "
    "`comfy agent allow --path <narrower folder>` grants what is actually needed. " + REFUSED_HINT
)
UNKNOWN_REQUEST_HINT = "run `comfy agent permissions` to see the requests waiting; each shows its id"
GRANT_HINTS = {
    "approve": "comfy agent allow --approve <id>",
    "deny": "comfy agent deny <id>",
    "path": 'comfy agent allow --path "<folder>" --reason "<why>"',
    "host": "comfy agent allow --host <host> --reason <why>",
}


def _age(requested_at: object, now: datetime | None = None) -> str:
    """How long ago an RFC 3339 timestamp was, for a human; ``?`` when it does not parse."""
    if not isinstance(requested_at, str):
        return "?"
    try:
        then = datetime.fromisoformat(requested_at.replace("Z", "+00:00"))
    except ValueError:
        return "?"
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    seconds = max(0, int(((now or datetime.now(timezone.utc)) - then).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _no_agent_warning(root) -> str:
    # No agent has ever published itself here. A running agent started with
    # another AGENT_DATA_DIR (the launcher's, or a typo in --data-dir) will
    # never read this file; say so rather than promise a pick-up.
    return (
        f"no {DISCOVERY_FILE} in {root}: no agent has run from this data dir. A running agent that uses "
        "another data dir will not see this approval; pass --data-dir or set AGENT_DATA_DIR to match it"
    )


def _open_health(url: str):
    """Open the agent's /health on loopback, ignoring any HTTP(S)_PROXY in the environment.

    The global ``urlopen`` honours ``HTTP_PROXY``, which a shell behind the
    agent's own egress proxy exports; routing a loopback probe through it
    made a live agent look absent. An empty ``ProxyHandler`` skips the
    environment.
    """
    return build_http_only_opener(urllib.request.ProxyHandler({})).open(url, timeout=2)


def _sandbox_status(port: int | None) -> dict | None:
    if not port:
        return None
    try:
        with _open_health(f"http://127.0.0.1:{port}/health") as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        # HTTPException is not an OSError: a stale port taken by something
        # that is not HTTP (an SSH banner) raises BadStatusLine here.
        return None
    if not isinstance(body, dict):
        return None  # a stale port answered by something that is not the agent
    # The envelope schema says sandbox is an object or null and comfy_path a
    # string or null; a stale or foreign server's shapes are not passed on.
    sandbox = body.get("sandbox")
    out: dict = {"sandbox": sandbox if isinstance(sandbox, dict) else None}
    cli = body.get("cli")
    if isinstance(cli, dict) and isinstance(cli.get("workspace"), str) and cli["workspace"]:
        out["comfy_path"] = cli["workspace"]
    return out


@app.command(
    "permissions",
    help="Show the agent's requests waiting for approval, the approved folders and hosts, and a running agent's sandbox.",
)
@tracking.track_command("agent")
def permissions_cmd(data_dir_opt: _DATA_DIR_OPT = None):
    renderer = get_renderer()
    root = data_dir(data_dir_opt)
    try:
        state = read_state(root)
    except StateError as exc:
        renderer.error(code="agent_state_unreadable", message=str(exc), hint="fix or remove the file and try again")
        raise typer.Exit(code=1) from exc
    health = _sandbox_status(state.port) if state.running else None
    payload = {
        "data_dir": str(root),
        "agent": {"running": state.running and health is not None, "port": state.port},
        "sandbox": (health or {}).get("sandbox"),
        "comfy_path": (health or {}).get("comfy_path"),
        "pending": state.pending,
        "paths": state.paths,
        "hosts": state.hosts,
        "grant": dict(GRANT_HINTS),
    }
    if renderer.is_pretty():
        # Everything interpolated below came from files on disk or a local
        # server; it is escaped so a "[/]" in a reason cannot crash the print.
        rprint(f"[bold]data dir[/bold]  {esc(root)}")
        status = f"running on port {state.port}" if payload["agent"]["running"] else "not running"
        rprint(f"[bold]agent[/bold]     {status}")
        if payload["sandbox"]:
            rprint(f"[bold]sandbox[/bold]   {esc(json.dumps(payload['sandbox']))}")
        if payload["comfy_path"]:
            rprint(f"[bold]ComfyUI[/bold]   {esc(payload['comfy_path'])}")
        rprint("[bold]requests waiting for your approval[/bold]" + ("" if state.pending else "  (none)"))
        for e in state.pending:
            rprint(
                f"  {esc(e['id'])}  {e['kind']:<4} {esc(e['target'])}  — {esc(e.get('reason', ''))}"
                f"  [dim]({_age(e.get('requested_at'))} ago)[/dim]"
            )
        if state.pending:
            first = state.pending[0]["id"]
            rprint(f"  [dim]comfy agent allow --approve {first}    comfy agent deny {first}[/dim]")
        rprint("[bold]folders the user approved[/bold]" + ("" if state.paths else "  (none)"))
        for e in state.paths:
            rprint(f"  {esc(e.get('path'))}  — {esc(e.get('reason', ''))}")
        rprint("[bold]hosts the user approved[/bold]" + ("" if state.hosts else "  (none)"))
        for e in state.hosts:
            rprint(f"  {esc(e.get('host'))}  — {esc(e.get('reason', ''))}")
        rprint("\n[dim]" + "\n".join(GRANT_HINTS.values()) + "[/dim]")
    renderer.emit(payload, command="agent permissions")


def _approve_request(root, request_id: str) -> None:
    """``comfy agent allow --approve <id>``: answer one request the agent recorded."""
    renderer = get_renderer()
    try:
        state = read_state(root)
        record, kind = approve(root, request_id)
    except StateError as exc:
        renderer.error(code="agent_state_unreadable", message=str(exc), hint="fix or remove the file and try again")
        raise typer.Exit(code=1) from exc
    except UnknownRequestError as exc:
        renderer.error(code="agent_unknown_request", message=str(exc), hint=UNKNOWN_REQUEST_HINT)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        renderer.error(code="agent_refused", message=str(exc), hint=REFUSED_REQUEST_HINT)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        renderer.error(
            code="agent_state_unwritable",
            message=f"{root}: cannot write: {exc}",
            hint="check the data dir's permissions, or pass --data-dir for the dir the agent uses",
        )
        raise typer.Exit(code=1) from exc
    approved = {"id": request_id, "kind": kind, "target": record[kind]}
    payload: dict = {"data_dir": str(root), "approved": approved, "note": PICKUP_NOTE}
    if not state.running:
        payload["warning"] = _no_agent_warning(root)
    if renderer.is_pretty():
        rprint(f"[green]approved[/green] {kind}: {esc(approved['target'])}  (request {esc(request_id)})")
        rprint(f"[dim]{PICKUP_NOTE}[/dim]")
        if "warning" in payload:
            renderer.warn(payload["warning"])
    renderer.emit(payload, command="agent allow")


@app.command(
    "allow",
    help="Approve one of the agent's pending requests (--approve <id>), or let it reach a folder (--path) or host (--host).",
)
@tracking.track_command("agent")
def allow_cmd(
    approve_id: Annotated[
        str | None,
        typer.Option(
            "--approve", show_default=False, help="The id of a pending request (see `comfy agent permissions`)."
        ),
    ] = None,
    path: Annotated[
        str | None, typer.Option("--path", show_default=False, help="An absolute, existing folder.")
    ] = None,
    host: Annotated[str | None, typer.Option("--host", show_default=False, help="A host name, no scheme.")] = None,
    reason: Annotated[
        str, typer.Option("--reason", help="Why; recorded with the approval.")
    ] = "allowed with comfy agent allow",
    data_dir_opt: _DATA_DIR_OPT = None,
):
    renderer = get_renderer()
    if approve_id and (path or host):
        # A request names its own target and reason; --path/--host beside it
        # would leave which target got approved to the reader's guess.
        renderer.error(
            code="agent_bad_args",
            message="--approve answers one recorded request and cannot be combined with --path or --host",
            hint="run `comfy agent allow --approve <id>` on its own, then `--path`/`--host` as a separate command",
        )
        raise typer.Exit(code=2)
    if not path and not host and not approve_id:
        renderer.error(
            code="agent_bad_args",
            message="nothing to allow",
            hint="pass --approve <id> for a pending request, or --path <folder> and/or --host <host>",
        )
        raise typer.Exit(code=2)
    root = data_dir(data_dir_opt)
    if approve_id:
        _approve_request(root, approve_id.strip())
        return
    added: dict = {}
    try:
        # Read every state file and vet every value before the first write, so
        # a refused --host or a broken egress-allow.json does not leave a
        # --path approved behind a failure exit.
        state = read_state(root)
        if path:
            vet_path(path, root=root)
        if host:
            vet_host(host)
        if path:
            folder, is_new = allow_path(root, path, reason)
            added["path"] = {"path": str(folder), "added": is_new}
        if host:
            h, is_new = allow_host(root, host, reason)
            added["host"] = {"host": h, "added": is_new}
    except StateError as exc:
        renderer.error(code="agent_state_unreadable", message=str(exc), hint="fix or remove the file and try again")
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        renderer.error(code="agent_refused", message=str(exc), hint=REFUSED_HINT)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        renderer.error(
            code="agent_state_unwritable",
            message=f"{root}: cannot write: {exc}",
            hint="check the data dir's permissions, or pass --data-dir for the dir the agent uses",
        )
        raise typer.Exit(code=1) from exc
    payload: dict = {"data_dir": str(root), **added, "note": PICKUP_NOTE}
    if not state.running:
        payload["warning"] = _no_agent_warning(root)
    if renderer.is_pretty():
        for kind, entry in added.items():
            rprint(f"[green]{'allowed' if entry['added'] else 'already allowed'}[/green] {kind}: {esc(entry[kind])}")
        rprint(f"[dim]{PICKUP_NOTE}[/dim]")
        if "warning" in payload:
            renderer.warn(payload["warning"])
    renderer.emit(payload, command="agent allow")


@app.command("deny", help="Refuse one of the agent's pending requests; nothing is approved and the request is dropped.")
@tracking.track_command("agent")
def deny_cmd(
    request_id: Annotated[str, typer.Argument(help="The id of a pending request (see `comfy agent permissions`).")],
    data_dir_opt: _DATA_DIR_OPT = None,
):
    renderer = get_renderer()
    root = data_dir(data_dir_opt)
    request_id = request_id.strip()
    try:
        req = deny(root, request_id)
    except StateError as exc:
        renderer.error(code="agent_state_unreadable", message=str(exc), hint="fix or remove the file and try again")
        raise typer.Exit(code=1) from exc
    except UnknownRequestError as exc:
        renderer.error(code="agent_unknown_request", message=str(exc), hint=UNKNOWN_REQUEST_HINT)
        raise typer.Exit(code=1) from exc
    except OSError as exc:
        renderer.error(
            code="agent_state_unwritable",
            message=f"{root}: cannot write: {exc}",
            hint="check the data dir's permissions, or pass --data-dir for the dir the agent uses",
        )
        raise typer.Exit(code=1) from exc
    denied = {"id": request_id, "kind": req["kind"], "target": req["target"]}
    payload = {"data_dir": str(root), "denied": denied}
    if renderer.is_pretty():
        rprint(f"[yellow]denied[/yellow] {denied['kind']}: {esc(denied['target'])}  (request {esc(request_id)})")
    renderer.emit(payload, command="agent deny")
