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


# The keys only the agent's own /health carries. The standalone binary serves
# the plain string "ok" until it has something to report and a JSON object
# once it has, and its startup reports sandbox and cli together, so a JSON
# object carrying none of these is some other service on a reused port (cloud
# services/agent/local/internal/server/server.go, healthPayload). `status` is
# deliberately not in the list: every other health endpoint has one too.
# Checked against that struct's JSON tags at cloud commit 6c54e206, and pinned
# by test_health_markers_are_fields_the_agent_actually_sends so a rename on the
# agent's side fails here instead of reading a live agent as gone.
_AGENT_HEALTH_KEYS = (
    "sandbox",
    "skills",
    "crdt",
    "doc_host",
    "prompt",
    "cli",
    "assets",
    "memory",
    "node_trees",
)


# What a /health probe on the discovery port established: GONE is only for what
# the probe *proved* is not this agent, and everything it could not settle is
# UNCONFIRMED. `_sandbox_status` collapses both onto None for display — either
# way there is no running agent or sandbox to show.
_AGENT_LIVE = "live"
_AGENT_GONE = "gone"
_AGENT_UNCONFIRMED = "unconfirmed"

# The bare answer the agent's health handler falls through to while every
# reporter is still nil, in both forms it could arrive in: the server sends the
# two bytes as text/plain, a JSON encoder would quote them.
#
# No agent can serve it on the port this probe reaches. `app.go` calls
# ReportSandbox/ReportCLI/ReportNodeTrees/ReportAssets/ReportMemory/
# ReportPrompt/ReportSkills/ReportCRDT (lines 233-255 on cloud main) BEFORE
# net.Listen (257) and before WriteDiscoveryWith puts the port in agent.json
# (263), so by the time anything can connect, or even learn the port, eight
# reporters are non-nil and the handler serves the JSON payload. The
# fall-through is live code on a path nothing can observe from here.
#
# So this body means "something else holds the port", not "the agent, early".
_AGENT_EARLY_OK = ("ok", '"ok"')


def _probe_agent(port: int | None) -> tuple[str, dict | None]:
    """Ask the discovery port whether this agent is answering on it.

    Returns the verdict and, when the answer carried them, the sandbox and
    workspace fields `permissions` displays.
    """
    if not port:
        return _AGENT_GONE, None
    try:
        with _open_health(f"http://127.0.0.1:{port}/health") as resp:
            raw = resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError:
        # Something serves HTTP here and refused the probe. Whether it is the
        # agent is exactly what this failed to establish.
        return _AGENT_UNCONFIRMED, None
    except urllib.error.URLError as exc:
        # urllib wraps the transport error; a refused connection is the one
        # that proves nothing is listening, so agent.json is stale.
        return (_AGENT_GONE if isinstance(exc.reason, ConnectionRefusedError) else _AGENT_UNCONFIRMED), None
    except ConnectionRefusedError:
        return _AGENT_GONE, None
    except (http.client.RemoteDisconnected, http.client.IncompleteRead):
        # Both reach the HTTPException clause below and must not: they come
        # from a peer that accepted the connection and then failed mid-exchange
        # -- a live agent shutting down, crashing or overloaded does exactly
        # this. urllib wraps only h.request() in URLError, not h.getresponse(),
        # so a RemoteDisconnected (a BadStatusLine as well as a
        # ConnectionResetError) arrives here unwrapped; an IncompleteRead comes
        # from resp.read() on a short body. Neither establishes that the
        # listener is not this agent.
        return _AGENT_UNCONFIRMED, None
    except http.client.HTTPException:
        # HTTPException is not an OSError: a stale port taken by something
        # that is not HTTP (an SSH banner) raises BadStatusLine here. Proof
        # enough that this is not the agent.
        return _AGENT_GONE, None
    except (OSError, ValueError):
        # A timeout (the 2 s budget in _open_health) or a reset mid-read.
        # Something is there; what it is is unsettled.
        return _AGENT_UNCONFIRMED, None
    if raw.strip() in _AGENT_EARLY_OK:
        # Unsettled, and deliberately not either certainty. It cannot be this
        # agent answering early (see `_AGENT_EARLY_OK`: the port does not exist
        # until eight reporters do), so "live" would assert something false.
        # But it is not proof of absence either — something is listening on
        # the port agent.json names. The JSON-quoted form is matched too, so a
        # later switch to `c.JSON` does not change the verdict.
        return _AGENT_UNCONFIRMED, None
    try:
        body = json.loads(raw)
    except ValueError:
        # Not the early answer, and not JSON either. Something is there; what
        # it is is unsettled.
        return _AGENT_UNCONFIRMED, None
    if not isinstance(body, dict):
        # Neither the agent's own early answer nor anything judgeable.
        return _AGENT_UNCONFIRMED, None
    if not any(key in body for key in _AGENT_HEALTH_KEYS):
        # A reused port held by an unrelated local service whose /health
        # answers `{"status": "ok"}`. Established: not this agent.
        return _AGENT_GONE, None
    # The envelope schema says sandbox is an object or null and comfy_path a
    # string or null; a stale or foreign server's shapes are not passed on.
    sandbox = body.get("sandbox")
    out: dict = {"sandbox": sandbox if isinstance(sandbox, dict) else None}
    cli = body.get("cli")
    if isinstance(cli, dict) and isinstance(cli.get("workspace"), str) and cli["workspace"]:
        out["comfy_path"] = cli["workspace"]
    return _AGENT_LIVE, out


def _sandbox_status(port: int | None) -> dict | None:
    """The sandbox and workspace to display, or ``None`` when there are none.

    Two-valued on purpose: a caller that only renders state has nothing to do
    differently for a port it could not reach and one it reached and dismissed.
    """
    verdict, detail = _probe_agent(port)
    return detail if verdict == _AGENT_LIVE else None


@app.command("allow", hidden=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
@app.command("deny", hidden=True, context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def _moved_to_the_panel(ctx: typer.Context):
    """Kept hidden so an older agent's instructions get an answer: approving here was removed, and changes nothing."""
    get_renderer().error(
        code="agent_approval_moved",
        message=f"comfy agent {ctx.info_name} was removed: a script the agent ran could approve its own requests",
        hint="approve or deny the request on its card in the agent panel",
    )
    raise typer.Exit(code=1)


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
