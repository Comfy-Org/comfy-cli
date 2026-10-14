"""``comfy agent`` — the Typer surface over :mod:`comfy_cli.agent`.

``permissions`` shows the agent's requests waiting for a human, what the local
agent may reach beyond its defaults, and whether one is running. It only reads:
the user approves or denies a request in the agent panel, never from this CLI,
since the agent's model can run this CLI and must not answer its own request.
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
from comfy_cli.agent import StateError, data_dir, read_state
from comfy_cli.http import build_http_only_opener
from comfy_cli.output import get_renderer, rprint
from comfy_cli.output.sanitize import sanitize_markup as esc

app = typer.Typer(
    no_args_is_help=True,
    help="The local comfy agent: what it may reach. Approve its requests in the agent panel.",
)


@app.callback()
def _agent_group() -> None:
    # With one command and no callback, Typer would collapse this app into that
    # command when invoked on its own; the callback keeps ``permissions`` a
    # subcommand everywhere.
    pass


_DATA_DIR_OPT = Annotated[
    str | None,
    typer.Option(
        "--data-dir", show_default=False, help="The agent's data dir (default: AGENT_DATA_DIR, else ~/.comfy-agent)."
    ),
]

NOTE = (
    "approve or deny a pending request on its card in the agent panel; the agent cannot approve its own "
    "requests. A request whose card closed unanswered gets a new card when the agent asks for it again"
)


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
        "note": NOTE,
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
            rprint(f"  [dim]{NOTE}[/dim]")
        rprint("[bold]folders the user approved[/bold]" + ("" if state.paths else "  (none)"))
        for e in state.paths:
            rprint(f"  {esc(e.get('path'))}  — {esc(e.get('reason', ''))}")
        rprint("[bold]hosts the user approved[/bold]" + ("" if state.hosts else "  (none)"))
        for e in state.hosts:
            rprint(f"  {esc(e.get('host'))}  — {esc(e.get('reason', ''))}")
    renderer.emit(payload, command="agent permissions")
