"""``comfy stop --port <p>`` — verified stop of an *untracked* local ComfyUI.

Plain ``comfy stop`` can only stop a server this CLI started: it reads the
recorded ``(host, port, pid)`` out of the config and kills that. The most
common real-world restart case is the opposite one — a ComfyUI somebody else
started (the desktop app, a bare ``python main.py``, a wedged server) is
holding the port and comfy-cli has no handle on it at all.

This module is that handle, and every decision in it is about *not* killing the
wrong process:

- The listener is found with the **per-process** ``Process.net_connections()``
  over ``psutil.process_iter()``. The system-wide ``psutil.net_connections()``
  is root-only on macOS, so it is never used here.
- Identity is proven from the **process table first**: a python-ish ``argv[0]``
  running a ``main.py``. Only then do we ask the server itself
  (``GET /system_stats``). A port that answers but disagrees with the process
  table is refused — that is the reverse-proxy false positive.
- A port that does not answer at all is *still* stopped, on the cmdline
  evidence alone: a wedged ComfyUI is precisely what this command exists for.
- We never stop anything on HTTP evidence alone.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import psutil
import typer

from comfy_cli.config_manager import ConfigManager
from comfy_cli.output import Renderer
from comfy_cli.output.sanitize import sanitize_markup

# ~2s, per the ticket: long enough for a healthy server to answer on loopback,
# short enough that probing a wedged one doesn't stall the command.
HTTP_PROBE_TIMEOUT = 2.0

# Bounded wait for each half of the terminate -> kill escalation.
TERMINATE_TIMEOUT = 5.0
KILL_TIMEOUT = 3.0

# `python`, `python3`, `python3.12`, `pythonw`, `py`, and the macOS framework
# `Python` — but not `uv`, `node`, `nginx`, or anything else that merely has a
# `main.py` somewhere on its command line.
_PYTHON_ARGV0 = re.compile(r"^(python|pythonw|py)[0-9._]*$", re.IGNORECASE)

_EXPECTED_ERRORS = (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError)


@dataclass(frozen=True)
class Listener:
    """Whatever we could read about the process holding the port."""

    pid: int
    cmdline: list[str] = field(default_factory=list)
    cwd: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class Probe:
    """Outcome of ``GET /system_stats``.

    ``answered`` distinguishes "the port produced an HTTP response" (even a 404
    from a proxy, or a non-JSON body) from "refused / timed out", which is the
    wedged-server case we deliberately proceed on.
    """

    answered: bool
    stats: dict[str, Any] | None = None


@dataclass(frozen=True)
class Verdict:
    verified: bool
    reason: str
    http: str


# --------------------------------------------------------------------------- #
# Finding the listener
# --------------------------------------------------------------------------- #


def _is_listen_on(conn: Any, port: int) -> bool:
    if conn.status != psutil.CONN_LISTEN:
        return False
    laddr = getattr(conn, "laddr", None)
    return bool(laddr) and getattr(laddr, "port", None) == port


def _safe_cwd(proc: psutil.Process) -> str | None:
    try:
        return proc.cwd()
    except _EXPECTED_ERRORS:
        # macOS denies cwd() for other users' processes. Not knowing it is fine
        # — it is reported for the operator's benefit, never used as evidence.
        return None
    except NotImplementedError:  # pragma: no cover - platform-dependent
        return None


def find_listener(port: int) -> Listener | None:
    """Return the process LISTENing on ``port``, or None.

    Processes we are not allowed to inspect are skipped rather than fatal: on
    macOS every process owned by another user raises ``AccessDenied`` from
    ``net_connections()``, and one of those must not hide the one we can read.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            conns = proc.net_connections(kind="inet")
        except _EXPECTED_ERRORS:
            continue
        if not any(_is_listen_on(c, port) for c in conns):
            continue
        info = proc.info or {}
        return Listener(
            pid=proc.pid,
            # AccessDenied on the cmdline leaves it empty — we still report the
            # pid, so the caller can say *something* about what it refused.
            cmdline=[str(tok) for tok in (info.get("cmdline") or [])],
            cwd=_safe_cwd(proc),
            name=info.get("name"),
        )
    return None


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #


def _is_python_ish(argv0: str) -> bool:
    base = os.path.basename(argv0)
    if base.lower().endswith(".exe"):
        base = base[:-4]
    return bool(_PYTHON_ARGV0.match(base))


def _script_tail(cmdline: Sequence[str]) -> list[str] | None:
    """Return the command line from the ``main.py`` token onward, or None.

    That tail is what the server's own ``sys.argv`` should look like, so it is
    both the cmdline identity check and the input to the ``/system_stats``
    cross-check.
    """
    for i, tok in enumerate(cmdline[1:], start=1):
        if os.path.basename(tok) == "main.py":
            return list(cmdline[i:])
    return None


def looks_like_comfyui(cmdline: Sequence[str]) -> bool:
    """cmdline evidence: a python-ish ``argv[0]`` running some ``main.py``.

    Deliberately shape-based, not path-based: `comfy launch` runs
    ``[python, "main.py"]`` from the workspace, a hand-started server uses an
    absolute path, and the desktop app uses its bundled interpreter — all three
    are the same shape.
    """
    if len(cmdline) < 2 or not _is_python_ish(cmdline[0]):
        return False
    return _script_tail(cmdline) is not None


def probe_system_stats(port: int, *, timeout: float = HTTP_PROBE_TIMEOUT) -> Probe:
    """``GET http://127.0.0.1:<port>/system_stats`` with a short timeout."""
    url = f"http://127.0.0.1:{port}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - fixed loopback http URL
            body = resp.read()
    except urllib.error.HTTPError:
        # Something is serving this port and it is not ComfyUI's /system_stats.
        return Probe(answered=True)
    except (urllib.error.URLError, OSError):
        # Refused, reset, or timed out — the wedged-server case.
        return Probe(answered=False)
    try:
        payload = json.loads(body.decode("utf-8", "replace"))
    except (ValueError, UnicodeDecodeError):
        return Probe(answered=True)
    return Probe(answered=True, stats=payload if isinstance(payload, dict) else None)


def _argv_agrees(server_argv: Sequence[Any], cmdline: Sequence[str]) -> bool:
    """Does the server's own ``sys.argv`` match the process table's cmdline?

    The server reports ``argv[0]`` as it was spelled on the command line, which
    may be relative (``main.py``) or absolute, so only the basename is compared;
    the arguments after it must match exactly.
    """
    tail = _script_tail(cmdline)
    if not tail or not server_argv:
        return False
    if os.path.basename(str(server_argv[0])) != os.path.basename(tail[0]):
        return False
    return [str(a) for a in server_argv[1:]] == list(tail[1:])


def verify_listener(listener: Listener, port: int, *, probe: Probe | None = None) -> Verdict:
    """Decide whether ``listener`` is positively identifiable as ComfyUI.

    Order matters: the cmdline is a precondition, so an HTTP answer alone can
    never authorize a kill. The probe can only *withdraw* that authorization.
    """
    if not looks_like_comfyui(listener.cmdline):
        return Verdict(
            verified=False,
            reason="the process holding the port is not a python `main.py` command line",
            http="not_probed",
        )

    if probe is None:
        probe = probe_system_stats(port)

    if not probe.answered:
        # Wedged or still starting: the process table already says ComfyUI, and
        # a server that cannot answer is exactly what this command is for.
        return Verdict(verified=True, reason="identified from the process command line", http="unreachable")

    stats = probe.stats
    system = stats.get("system") if isinstance(stats, dict) else None
    if not isinstance(system, dict) or not system.get("comfyui_version"):
        return Verdict(
            verified=False,
            reason="the port answered but did not return a ComfyUI /system_stats payload",
            http="disagreed",
        )

    server_argv = system.get("argv")
    if isinstance(server_argv, list) and server_argv:
        if not _argv_agrees(server_argv, listener.cmdline):
            return Verdict(
                verified=False,
                reason="the server on this port reports a different command line than the process holding it",
                http="disagreed",
            )
        return Verdict(verified=True, reason="command line corroborated by /system_stats", http="agreed")

    # Older servers omit `argv`. `comfyui_version` plus the cmdline is still two
    # independent signals, so this corroborates — it just can't cross-check.
    return Verdict(
        verified=True,
        reason="command line corroborated by /system_stats (server reported no argv)",
        http="agreed_no_argv",
    )


# --------------------------------------------------------------------------- #
# Stopping
# --------------------------------------------------------------------------- #


def kill_process_tree(pid: int, *, terminate_timeout: float = TERMINATE_TIMEOUT, kill_timeout: float = KILL_TIMEOUT):
    """Terminate ``pid`` and its recursive children, escalating to SIGKILL.

    ``utils.kill_all`` is deliberately not reused: it kills only the *children*
    of the pid it is given, which is right for the recorded wrapper pid but
    would leave a directly-identified listener running.

    Returns ``(ok, survivors)`` — ``ok`` is True only once nothing is left.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return True, []

    try:
        targets = parent.children(recursive=True)
    except _EXPECTED_ERRORS:
        targets = []
    # Children first so a supervising parent can't respawn one mid-teardown.
    targets.append(parent)

    for proc in targets:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            continue
        except _EXPECTED_ERRORS:
            # Can't signal it (not ours). Leave it in the list so the wait below
            # reports it as a survivor rather than silently claiming success.
            continue

    _, alive = psutil.wait_procs(targets, timeout=terminate_timeout)
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            continue
        except _EXPECTED_ERRORS:
            continue

    _, survivors = psutil.wait_procs(alive, timeout=kill_timeout)
    return not survivors, [p.pid for p in survivors]


def _dry_run_payload(listener: Listener, port: int) -> dict[str, Any]:
    data: dict[str, Any] = {
        # `stopped` is required by schemas/stop.json and is what a caller
        # branches on; `dry_run` says why it is False.
        "stopped": False,
        "dry_run": True,
        "verified": True,
        "untracked": True,
        "pid": listener.pid,
        "port": port,
        "cmdline": list(listener.cmdline),
    }
    if listener.cwd:
        data["cwd"] = listener.cwd
    return data


def stop_port_execute(renderer: Renderer, *, port: int, dry_run: bool) -> None:
    """Body of ``comfy stop --port <p>``. Raises ``typer.Exit`` on refusal."""
    listener = find_listener(port)
    if listener is None:
        renderer.error(
            code="port_not_listening",
            message=f"Nothing is listening on port {port}.",
            command="stop",
            details={"port": port},
        )
        raise typer.Exit(code=1)

    verdict = verify_listener(listener, port)
    if not verdict.verified:
        renderer.error(
            code="unverified_process",
            message=(
                f"Refusing to stop pid {listener.pid} on port {port}: it cannot be identified as ComfyUI "
                f"({verdict.reason})."
            ),
            command="stop",
            details={
                "port": port,
                "pid": listener.pid,
                "cmdline": list(listener.cmdline),
                "cwd": listener.cwd,
                "name": listener.name,
                "reason": verdict.reason,
                "http": verdict.http,
            },
        )
        raise typer.Exit(code=1)

    if dry_run:
        # The cmdline is a foreign process's argv — Rich would happily render
        # `[red]`/`[link=…]` out of it, or crash on an unbalanced `[/]`.
        rendered = sanitize_markup(" ".join(listener.cmdline))
        renderer.print(
            f"[bold yellow]Would stop ComfyUI on port {port}[/bold yellow] (pid={listener.pid})\n  {rendered}"
        )
        renderer.emit(_dry_run_payload(listener, port), command="stop", changed=False)
        return

    ok, survivors = kill_process_tree(listener.pid)
    if not ok:
        renderer.error(
            code="stop_failed",
            message=f"Failed to stop pid {listener.pid} on port {port}.",
            command="stop",
            details={"port": port, "pid": listener.pid, "survivors": survivors},
        )
        raise typer.Exit(code=1)

    _forget_background_if_same_pid(listener.pid)

    renderer.print(f"[bold yellow]ComfyUI on port {port} is stopped.[/bold yellow] (pid={listener.pid})")
    renderer.emit(
        {"stopped": True, "pid": listener.pid, "port": port, "untracked": True},
        command="stop",
        changed=True,
    )


def _forget_background_if_same_pid(pid: int) -> None:
    """Drop the recorded background server when we just stopped that exact pid.

    Only on an exact pid match: a port collision alone doesn't prove the record
    describes the process we killed, and wrongly dropping it would orphan a
    still-running server this CLI does own.
    """
    config = ConfigManager()
    bg_info = config.background
    if bg_info and bg_info[2] == pid:
        config.remove_background()
