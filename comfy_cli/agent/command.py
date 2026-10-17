"""``comfy agent`` — the Typer surface over :mod:`comfy_cli.agent`.

``permissions`` shows the agent's requests waiting for a human, what the local
agent may reach beyond its defaults, and whether one is running; ``allow
--approve <id>`` / ``deny <id>`` answer a request the agent's ``request_path``
/ ``request_host`` tools recorded (the model cannot answer its own request);
``allow --path`` / ``--host`` grants a folder or a host outright.

Every one of those writes refuses while an agent is answering on its port: a
running agent is the only writer of its permission files, so the write would
be ignored and then overwritten. While one runs, requests are answered in the
agent panel or on the in-chat card, which call the agent's own API.
"""

from __future__ import annotations

import http.client
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
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

# A running agent is the only writer of its permission files: it loads them at
# start and saves its whole state on every change, so an edit made here while
# it runs is never read and is then overwritten by the agent's next save.
# Hence NEXT_START_NOTE, and the refusals below. The rule is the agent's, and
# is stated in its own `services/agent/local/README.md`.
NEXT_START_NOTE = "this applies at the agent's next start; a running agent does not read it"
AGENT_RUNNING_MESSAGE = (
    "an agent is answering on port {port} from {root}: a running agent is the only writer of its "
    "permission files, so this edit would be ignored and then overwritten"
)
ANSWER_IN_PANEL = (
    "a running agent is the only writer of these files: answer a waiting request in the agent panel or "
    "on the card in chat, or stop the agent to edit them from here"
)
AGENT_RUNNING_HINT = (
    "answer a waiting request in the agent panel or on the card in chat, which call the running agent's "
    "own API; to write these files from here instead, stop the agent first"
)
AGENT_UNCONFIRMED_MESSAGE = (
    "something is answering on port {port}, which {root} names as its agent's, but it did not answer as that "
    "agent: a write here is lost without a trace if an agent is in fact running, so none was made"
)
AGENT_UNCONFIRMED_HINT = (
    "if an agent is running, answer the request in the agent panel or on the card in chat; if none is, delete "
    "the stale agent.json in that data dir and run this again"
)
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
    "approve": "{cli} agent allow{dir} --approve <id>",
    "deny": "{cli} agent deny{dir} <id>",
    "path": '{cli} agent allow{dir} --path "<folder>" --reason "<why>"',
    "host": "{cli} agent allow{dir} --host <host> --reason <why>",
}

# The entry points pyproject installs. Anything else in argv[0] -- a test
# runner, an embedding host -- is not a way to call this CLI, so the hints name
# the installed command rather than whatever started the process.
_CLI_NAMES = frozenset({"comfy", "comfy-cli", "comfycli"})


def _cli_invocation() -> str:
    """How to call this CLI, as this process was started.

    ``comfy`` for an installed entry point (``comfy.exe`` folded to the same),
    ``<python> -m comfy_cli`` when started as a module, and ``comfy`` for
    anything else -- a hint is for the user's terminal, and a path that only
    works inside the agent's sandbox is exactly what sent users to paste a venv
    path that answered "Access is denied".
    """
    argv0 = Path(sys.argv[0] or "")
    if argv0.name == "__main__.py":
        return f"{Path(sys.executable).name} -m comfy_cli"
    name = argv0.name[:-4] if argv0.name.lower().endswith(".exe") else argv0.name
    return name if name in _CLI_NAMES else "comfy"


def grant_hints(root: Path) -> dict[str, str]:
    """The exact commands that approve or grant against ``root``.

    ``--data-dir`` rides along whenever the agent's dir is not the one a bare
    terminal resolves (``AGENT_DATA_DIR``, else ``~/.comfy-agent``): the user's
    shell is a different process, and without it ``--approve`` answers
    ``agent_unknown_request`` while the request is still waiting. A relative
    dir is emitted absolute: the user's shell need not share this process's cwd.
    """
    flag = "" if root.is_absolute() and root == data_dir(None) else f' --data-dir "{root.absolute()}"'
    cli = _cli_invocation()
    return {k: v.format(cli=cli, dir=flag) for k, v in GRANT_HINTS.items()}


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


def _refuse_while_live(root, state) -> None:
    """Refuse a write to the permission files unless the agent is known gone.

    ``state.running`` only says a discovery file exists; a crashed agent
    leaves one behind, so this probes ``/health`` rather than trusting the
    file — otherwise a stale file would lock the user out of the only way to
    edit these files.

    It refuses on ``unconfirmed`` as well as ``live``, which is the whole
    reason :func:`_probe_agent` has three outcomes rather than two. The two
    mistakes are not symmetric: writing while an agent runs is overwritten by
    its next save and the user never learns the approval was lost, while a
    refusal names the data dir and what to do about it. Only what the probe
    *establishes* is not this agent permits the write.
    """
    if not state.running:
        return
    verdict, _ = _probe_agent(state.port)
    if verdict == _AGENT_GONE:
        return
    renderer = get_renderer()
    if verdict == _AGENT_LIVE:
        renderer.error(
            code="agent_running",
            message=AGENT_RUNNING_MESSAGE.format(port=state.port, root=root),
            hint=AGENT_RUNNING_HINT,
        )
    else:
        renderer.error(
            code="agent_unconfirmed",
            message=AGENT_UNCONFIRMED_MESSAGE.format(port=state.port, root=root),
            hint=AGENT_UNCONFIRMED_HINT,
        )
    raise typer.Exit(code=1)


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


# What a /health probe on the discovery port established. Three outcomes, not
# two, because the write guard needs the distinction: GONE is only for what the
# probe *proved* is not this agent, and everything it could not settle is
# UNCONFIRMED. `_sandbox_status` collapses both onto None, which is right for
# display — either way there is no sandbox to show — and wrong for a refusal.
_AGENT_LIVE = "live"
_AGENT_GONE = "gone"
_AGENT_UNCONFIRMED = "unconfirmed"


def _probe_agent(port: int | None) -> tuple[str, dict | None]:
    """Ask the discovery port whether this agent is answering on it.

    Returns the verdict and, when the answer carried them, the sandbox and
    workspace fields `permissions` displays.
    """
    if not port:
        return _AGENT_GONE, None
    try:
        with _open_health(f"http://127.0.0.1:{port}/health") as resp:
            body = json.loads(resp.read().decode("utf-8"))
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
    except http.client.HTTPException:
        # HTTPException is not an OSError: a stale port taken by something
        # that is not HTTP (an SSH banner) raises BadStatusLine here. Proof
        # enough that this is not the agent.
        return _AGENT_GONE, None
    except (OSError, ValueError):
        # A timeout (the 2 s budget in _open_health), a reset mid-read, or a
        # body that is not JSON. Something is there; what it is is unsettled.
        return _AGENT_UNCONFIRMED, None
    if body == "ok":
        # The agent's own server answers the plain string "ok" until it has
        # something to report (cloud services/agent/local/internal/server/
        # server.go, healthPayload), so this is the agent, early — not a
        # foreign service. Reading it as "gone" was how a write could still
        # be made to files a running agent was about to overwrite.
        return _AGENT_LIVE, None
    if not isinstance(body, dict):
        # Neither the agent's own early answer nor anything judgeable.
        return _AGENT_UNCONFIRMED, None
    if not any(key in body for key in _AGENT_HEALTH_KEYS):
        # A reused port held by an unrelated local service whose /health
        # answers `{"status": "ok"}`. Established: not this agent, so the
        # user is not locked out of editing the files at all.
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
    A caller that *writes* must use :func:`_probe_agent` directly.
    """
    verdict, detail = _probe_agent(port)
    return detail if verdict == _AGENT_LIVE else None


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
    # One probe serves both readings: whether to show the agent as running,
    # and whether a `grant` command offered here would be refused by
    # `_refuse_while_live`. Offering one it would refuse was the contradiction.
    verdict, health = _probe_agent(state.port) if state.running else (_AGENT_GONE, None)
    live = verdict == _AGENT_LIVE
    writable = verdict == _AGENT_GONE
    payload = {
        "data_dir": str(root),
        "agent": {"running": live, "port": state.port},
        "sandbox": (health or {}).get("sandbox"),
        "comfy_path": (health or {}).get("comfy_path"),
        "pending": state.pending,
        "paths": state.paths,
        "hosts": state.hosts,
        # Offered only when a write from here would actually land. While an
        # agent is answering it is the only writer of these files, and when
        # the probe could not settle that, `allow`/`deny` refuse too — so an
        # empty `grant` is the honest answer, not a missing one.
        "grant": grant_hints(root) if writable else {},
    }
    if not writable:
        payload["answer_in"] = ANSWER_IN_PANEL
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
        if state.pending and writable:
            first = state.pending[0]["id"]
            approve_cmd, deny_cmd = (payload["grant"][k].replace("<id>", first) for k in ("approve", "deny"))
            rprint(f"  [dim]{esc(approve_cmd)}    {esc(deny_cmd)}[/dim]")
        elif state.pending:
            rprint(f"  [dim]{esc(ANSWER_IN_PANEL)}[/dim]")
        rprint("[bold]folders the user approved[/bold]" + ("" if state.paths else "  (none)"))
        for e in state.paths:
            rprint(f"  {esc(e.get('path'))}  — {esc(e.get('reason', ''))}")
        rprint("[bold]hosts the user approved[/bold]" + ("" if state.hosts else "  (none)"))
        for e in state.hosts:
            rprint(f"  {esc(e.get('host'))}  — {esc(e.get('reason', ''))}")
        if writable:
            rprint("\n[dim]" + "\n".join(esc(h) for h in payload["grant"].values()) + "[/dim]")
        else:
            rprint(f"\n[dim]{esc(ANSWER_IN_PANEL)}[/dim]")
    renderer.emit(payload, command="agent permissions")


def _approve_request(root, request_id: str) -> None:
    """``comfy agent allow --approve <id>``: answer one request the agent recorded."""
    renderer = get_renderer()
    try:
        state = read_state(root)
        _refuse_while_live(root, state)
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
    payload: dict = {"data_dir": str(root), "approved": approved, "note": NEXT_START_NOTE}
    if not state.running:
        payload["warning"] = _no_agent_warning(root)
    if renderer.is_pretty():
        rprint(f"[green]approved[/green] {kind}: {esc(approved['target'])}  (request {esc(request_id)})")
        rprint(f"[dim]{NEXT_START_NOTE}[/dim]")
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
        _refuse_while_live(root, state)
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
    payload: dict = {"data_dir": str(root), **added, "note": NEXT_START_NOTE}
    if not state.running:
        payload["warning"] = _no_agent_warning(root)
    if renderer.is_pretty():
        for kind, entry in added.items():
            rprint(f"[green]{'allowed' if entry['added'] else 'already allowed'}[/green] {kind}: {esc(entry[kind])}")
        rprint(f"[dim]{NEXT_START_NOTE}[/dim]")
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
        _refuse_while_live(root, read_state(root))
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
