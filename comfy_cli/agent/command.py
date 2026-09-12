"""``comfy agent`` — the Typer surface over :mod:`comfy_cli.agent`.

``permissions`` shows what the local agent may reach beyond its defaults and
whether one is running; ``allow`` grants a folder or a host from a terminal,
the way the agent's ``allow_path`` / ``allow_host`` tools do from chat.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Annotated

import typer

from comfy_cli import tracking
from comfy_cli.agent import allow_host, allow_path, data_dir, read_state, vet_host, vet_path
from comfy_cli.output import get_renderer, rprint

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


def _sandbox_status(port: int | None) -> dict | None:
    if not port:
        return None
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:  # noqa: S310 - loopback
            body = json.loads(resp.read().decode("utf-8"))
    except (OSError, ValueError, urllib.error.URLError):
        return None
    if not isinstance(body, dict):
        return None  # a stale port answered by something that is not the agent
    out = {"sandbox": body.get("sandbox")}
    cli = body.get("cli") or {}
    if isinstance(cli, dict) and cli.get("workspace"):
        out["comfy_path"] = cli["workspace"]
    return out


@app.command("permissions", help="Show the approved folders and hosts, and the sandbox status of a running agent.")
@tracking.track_command("agent")
def permissions_cmd(data_dir_opt: _DATA_DIR_OPT = None):
    renderer = get_renderer()
    root = data_dir(data_dir_opt)
    try:
        state = read_state(root)
    except ValueError as exc:
        renderer.error(code="agent_state_unreadable", message=str(exc), hint="fix or remove the file and try again")
        raise typer.Exit(code=1) from exc
    health = _sandbox_status(state.port) if state.running else None
    payload = {
        "data_dir": str(root),
        "agent": {"running": state.running and health is not None, "port": state.port},
        "sandbox": (health or {}).get("sandbox"),
        "comfy_path": (health or {}).get("comfy_path"),
        "paths": state.paths,
        "hosts": state.hosts,
        "grant": {
            "path": 'comfy agent allow --path "<folder>" --reason "<why>"',
            "host": "comfy agent allow --host <host> --reason <why>",
        },
    }
    if renderer.is_pretty():
        rprint(f"[bold]data dir[/bold]  {root}")
        rprint(
            f"[bold]agent[/bold]     {'running on port ' + str(state.port) if payload['agent']['running'] else 'not running'}"
        )
        if payload["sandbox"]:
            rprint(f"[bold]sandbox[/bold]   {json.dumps(payload['sandbox'])}")
        if payload["comfy_path"]:
            rprint(f"[bold]ComfyUI[/bold]   {payload['comfy_path']}")
        rprint("[bold]folders the user approved[/bold]" + ("" if state.paths else "  (none)"))
        for e in state.paths:
            rprint(f"  {e.get('path')}  — {e.get('reason', '')}")
        rprint("[bold]hosts the user approved[/bold]" + ("" if state.hosts else "  (none)"))
        for e in state.hosts:
            rprint(f"  {e.get('host')}  — {e.get('reason', '')}")
        rprint(f"\n[dim]{payload['grant']['path']}\n{payload['grant']['host']}[/dim]")
    renderer.emit(payload, command="agent permissions")


@app.command("allow", help="Let the local agent reach one more folder (--path) or host (--host).")
@tracking.track_command("agent")
def allow_cmd(
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
    if not path and not host:
        renderer.error(code="bad_args", message="nothing to allow", hint="pass --path <folder> and/or --host <host>")
        raise typer.Exit(code=2)
    root = data_dir(data_dir_opt)
    added: dict = {}
    try:
        # Vet everything before the first write so a refused --host does not
        # leave a --path approved behind a failure exit.
        if path:
            vet_path(path)
        if host:
            vet_host(host)
        if path:
            folder, is_new = allow_path(root, path, reason)
            added["path"] = {"path": str(folder), "added": is_new}
        if host:
            h, is_new = allow_host(root, host, reason)
            added["host"] = {"host": h, "added": is_new}
    except ValueError as exc:
        renderer.error(
            code="refused",
            message=str(exc),
            hint="a narrower folder that holds only what is needed may be allowed; credential stores and the whole disk never are",
        )
        raise typer.Exit(code=1) from exc
    if renderer.is_pretty():
        for kind, entry in added.items():
            what = entry[kind]
            rprint(f"[green]{'allowed' if entry['added'] else 'already allowed'}[/green] {kind}: {what}")
        rprint(f"[dim]{PICKUP_NOTE}[/dim]")
    renderer.emit({"data_dir": str(root), **added, "note": PICKUP_NOTE}, command="agent allow")
