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
  (``GET /system_stats``, on the address it actually bound). A port that answers
  but disagrees with the process table is refused — that is the reverse-proxy
  false positive.
- A port that does *not* answer at all — wedged, still starting, or speaking
  TLS — is stopped only if the ``main.py`` on that command line also lives in a
  ComfyUI checkout on disk. The cmdline shape alone matches any ``python
  main.py`` service, and killing one of those is exactly what this must not do.
- We never stop anything on HTTP evidence alone.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
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

# `/system_stats` is a small JSON document. `timeout` on urlopen bounds each
# socket operation, not the whole transfer, so without a cap whatever holds the
# port could trickle an unbounded body into memory instead.
MAX_PROBE_BYTES = 1 << 20
_PROBE_CHUNK = 64 * 1024

# `python`, `python3`, `python3.12`, `pythonw`, `py`, and the macOS framework
# `Python` — but not `uv`, `node`, `nginx`, or anything else that merely has a
# `main.py` somewhere on its command line.
_PYTHON_ARGV0 = re.compile(r"^(python|pythonw|py)[0-9._]*$", re.IGNORECASE)

_EXPECTED_ERRORS = (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess, OSError)

# Files and directories that sit next to `main.py` in every ComfyUI checkout —
# source, portable, and the desktop app's bundle alike. Used as the second
# signal when the server cannot speak for itself.
_COMFYUI_TREE_MARKERS = ("nodes.py", "execution.py", "folder_paths.py", "server.py", "comfy", "comfy_extras")
_COMFYUI_TREE_MIN_MARKERS = 3


@dataclass(frozen=True)
class Listener:
    """Whatever we could read about the process holding the port."""

    pid: int
    cmdline: list[str] = field(default_factory=list)
    cwd: str | None = None
    name: str | None = None
    # The local address of the LISTEN socket we matched, so the probe below
    # reaches the server we are actually about to stop.
    laddr_ip: str | None = None
    # Pid + create_time is the only stable process identity; a bare pid can be
    # recycled between discovery and teardown.
    create_time: float | None = None


@dataclass(frozen=True)
class Probe:
    """Outcome of ``GET /system_stats``.

    ``answered`` distinguishes "the port produced an HTTP response" (even a 404
    from a proxy, or a non-JSON body) from "refused / timed out / not HTTP",
    which is the wedged-server case.
    """

    answered: bool
    stats: dict[str, Any] | None = None


@dataclass(frozen=True)
class Verdict:
    verified: bool
    reason: str
    http: str


@dataclass(frozen=True)
class Teardown:
    """Result of :func:`kill_process_tree`."""

    ok: bool
    survivors: list[int]
    # False when the child list could not be read, so `ok` cannot speak for
    # anything but the parent — reported rather than silently claimed.
    children_enumerated: bool = True


# --------------------------------------------------------------------------- #
# Finding the listener
# --------------------------------------------------------------------------- #


def _proc_net_connections(proc: Any, kind: str = "inet"):
    """``Process.net_connections()``, tolerating psutil < 6.

    psutil 6.0.0 renamed ``Process.connections()`` to ``net_connections()``.
    ``pyproject.toml`` pins the floor, but an already-provisioned environment
    can still resolve 5.x, where the missing attribute would escape as a raw
    ``AttributeError`` traceback instead of a structured error envelope.
    """
    getter = getattr(proc, "net_connections", None) or proc.connections
    return getter(kind=kind)


def _listen_match(conns: Iterable[Any], port: int) -> Any | None:
    """The first LISTEN socket on ``port``, or None."""
    for conn in conns:
        if conn.status != psutil.CONN_LISTEN:
            continue
        laddr = getattr(conn, "laddr", None)
        if laddr and getattr(laddr, "port", None) == port:
            return conn
    return None


def _safe_cwd(proc: psutil.Process) -> str | None:
    try:
        return proc.cwd()
    except _EXPECTED_ERRORS:
        # macOS denies cwd() for other users' processes. Not knowing it is fine
        # — it is reported for the operator's benefit, and only ever used to
        # resolve a relative script path (whose absence just means "refuse").
        return None
    except NotImplementedError:  # pragma: no cover - platform-dependent
        return None


def _safe_create_time(proc: psutil.Process) -> float | None:
    try:
        return proc.create_time()
    except _EXPECTED_ERRORS:
        return None


def find_listener(port: int) -> Listener | None:
    """Return the process LISTENing on ``port``, or None.

    Processes we are not allowed to inspect are skipped rather than fatal: on
    macOS every process owned by another user raises ``AccessDenied`` from
    ``net_connections()``, and one of those must not hide the one we can read.
    """
    for proc in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            conns = _proc_net_connections(proc)
        except _EXPECTED_ERRORS:
            continue
        conn = _listen_match(conns, port)
        if conn is None:
            continue
        info = proc.info or {}
        laddr = getattr(conn, "laddr", None)
        return Listener(
            pid=proc.pid,
            # AccessDenied on the cmdline leaves it empty — we still report the
            # pid, so the caller can say *something* about what it refused.
            cmdline=[str(tok) for tok in (info.get("cmdline") or [])],
            cwd=_safe_cwd(proc),
            name=info.get("name"),
            laddr_ip=getattr(laddr, "ip", None),
            create_time=_safe_create_time(proc),
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
    are the same shape. It is a *necessary* condition, never a sufficient one.
    """
    if len(cmdline) < 2 or not _is_python_ish(cmdline[0]):
        return False
    return _script_tail(cmdline) is not None


def _script_path(listener: Listener) -> str | None:
    """Absolute path of the ``main.py`` this process is running, if resolvable."""
    tail = _script_tail(listener.cmdline)
    if not tail:
        return None
    script = tail[0]
    if os.path.isabs(script):
        return script
    if not listener.cwd:
        return None
    return os.path.join(listener.cwd, script)


def looks_like_comfyui_tree(listener: Listener) -> bool:
    """Does this process's ``main.py`` live in a ComfyUI checkout?

    The second, independent signal for a listener that cannot corroborate
    itself over HTTP. Every ComfyUI checkout keeps ``main.py`` alongside
    ``nodes.py`` / ``execution.py`` / ``folder_paths.py`` / ``comfy/``; an
    unrelated ``python main.py`` service does not.
    """
    script = _script_path(listener)
    if not script:
        return False
    root = os.path.dirname(script) or "."
    try:
        hits = sum(1 for marker in _COMFYUI_TREE_MARKERS if os.path.exists(os.path.join(root, marker)))
    except OSError:  # pragma: no cover - unreadable directory
        return False
    return hits >= _COMFYUI_TREE_MIN_MARKERS


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects on the probe.

    A hostile local listener must not be able to point the probe off-box —
    either to source a corroborating ``/system_stats`` payload from elsewhere,
    or to an unreachable URL that would authorize its own termination.
    Returning None here leaves the 3xx unhandled, so urllib raises ``HTTPError``
    and the response reads as "answered, but not ComfyUI".
    """

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _open_url(url: str, timeout: float):
    """Open ``url`` with an opener that cannot be steered off the local machine.

    ``ProxyHandler({})`` because the *default* opener honors
    ``http_proxy``/``ALL_PROXY``, and urllib's ``proxy_bypass`` consults only
    ``no_proxy`` — there is no implicit localhost exemption, so without this the
    probe can leave the box entirely.
    """
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect)
    return opener.open(url, timeout=timeout)  # noqa: S310 - callers build a fixed http:// URL


def probe_host(laddr_ip: str | None) -> str:
    """Host component to probe, given the address the listener actually bound.

    ``find_listener`` matches a LISTEN socket on *any* local address, so
    hardcoding ``127.0.0.1`` would read a server started with ``--listen
    <lan-ip>``, or bound only to ``::1``, as unreachable — skipping the
    ``/system_stats`` cross-check exactly where it matters most.
    """
    ip = (laddr_ip or "").strip()
    # A zone id (`fe80::1%en0`) needs RFC 6874 `%25` escaping that urllib does
    # not accept; loopback is the safe fallback, and a non-answer is handled
    # conservatively downstream anyway.
    if not ip or ip == "*" or "%" in ip:
        return "127.0.0.1"
    try:
        parsed = ipaddress.ip_address(ip)
    except ValueError:
        # Scoped (`fe80::1%en0`) or otherwise unparseable. Fall back to
        # loopback; a non-answer here is handled conservatively anyway.
        return "127.0.0.1"
    if parsed.version == 4:
        return "127.0.0.1" if parsed.is_unspecified else ip
    mapped = parsed.ipv4_mapped
    if mapped is not None:
        return "127.0.0.1" if mapped.is_unspecified else str(mapped)
    return "[::1]" if parsed.is_unspecified else f"[{ip}]"


def _read_capped(resp: Any, *, limit: int, deadline: float) -> tuple[bytes, bool]:
    """Read at most ``limit`` bytes, giving up at ``deadline``.

    Returns ``(body, oversized)``. Raises ``TimeoutError`` (an ``OSError``, so
    it reads as "did not answer") if the peer trickles past the deadline.
    """
    chunks: list[bytes] = []
    total = 0
    while total <= limit:
        if time.monotonic() > deadline:
            raise TimeoutError("timed out reading /system_stats")
        chunk = resp.read(min(_PROBE_CHUNK, limit + 1 - total))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks), total > limit


def probe_system_stats(port: int, *, host: str = "127.0.0.1", timeout: float = HTTP_PROBE_TIMEOUT) -> Probe:
    """``GET http://<host>:<port>/system_stats`` with a short, bounded timeout."""
    url = f"http://{host}:{port}/system_stats"
    deadline = time.monotonic() + timeout
    try:
        with _open_url(url, timeout) as resp:
            body, oversized = _read_capped(resp, limit=MAX_PROBE_BYTES, deadline=deadline)
    except urllib.error.HTTPError:
        # Something is serving this port and it is not ComfyUI's /system_stats.
        return Probe(answered=True)
    except (urllib.error.URLError, http.client.HTTPException, OSError):
        # Refused, reset, timed out, or not speaking HTTP at all (a TLS-enabled
        # ComfyUI lands here) — the server cannot corroborate itself.
        # `http.client.HTTPException` is neither a URLError nor an OSError:
        # urllib only wraps OSError from the request, not from reading it.
        return Probe(answered=False)
    if oversized:
        # No `/system_stats` is a megabyte long; whatever this is, it isn't one.
        return Probe(answered=True)
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
    never authorize a kill. The probe can only *withdraw* that authorization —
    or, when it cannot be had at all, hand the decision to the on-disk checkout.
    """
    if not looks_like_comfyui(listener.cmdline):
        return Verdict(
            verified=False,
            reason="the process holding the port is not a python `main.py` command line",
            http="not_probed",
        )

    if probe is None:
        probe = probe_system_stats(port, host=probe_host(listener.laddr_ip))

    if not probe.answered:
        # Wedged, still starting, or serving TLS. The process table alone is a
        # weak signal here — every `python main.py` matches it — so require the
        # ComfyUI checkout on disk before signalling anything.
        if not looks_like_comfyui_tree(listener):
            return Verdict(
                verified=False,
                reason=(
                    "the port did not answer and the `main.py` it is running is not in a ComfyUI checkout "
                    "(no nodes.py/execution.py/comfy/ beside it)"
                ),
                http="unreachable",
            )
        return Verdict(
            verified=True,
            reason="identified from the process command line and the ComfyUI checkout it is running",
            http="unreachable",
        )

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


def _is_same_process(proc: psutil.Process, create_time: float) -> bool:
    """Is ``proc`` still the process we identified, or a pid recycled onto it?"""
    try:
        return abs(proc.create_time() - create_time) < 0.001
    except _EXPECTED_ERRORS:
        return False


def kill_process_tree(
    pid: int,
    *,
    create_time: float | None = None,
    terminate_timeout: float = TERMINATE_TIMEOUT,
    kill_timeout: float = KILL_TIMEOUT,
) -> Teardown:
    """Terminate ``pid`` and its recursive children, escalating to SIGKILL.

    ``utils.kill_all`` is deliberately not reused: it kills only the *children*
    of the pid it is given, which is right for the recorded wrapper pid but
    would leave a directly-identified listener running.

    ``create_time`` (when known) pins process identity: discovery is separated
    from teardown by an HTTP probe that can block for the full timeout, and a
    bare pid can be recycled onto an unrelated process in that window.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return Teardown(ok=True, survivors=[])

    if create_time is not None and not _is_same_process(parent, create_time):
        # The process we verified exited on its own and the OS handed its pid to
        # a stranger. The target is gone; signalling now would hit the stranger.
        return Teardown(ok=True, survivors=[])

    children_enumerated = True
    try:
        targets = parent.children(recursive=True)
    except _EXPECTED_ERRORS:
        # We cannot see the children, so `ok` below can only speak for the
        # parent. Say so rather than reporting a clean stop over live workers.
        children_enumerated = False
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
    return Teardown(
        ok=not survivors,
        survivors=[p.pid for p in survivors],
        children_enumerated=children_enumerated,
    )


def _ancestor_pids(pid: int) -> list[int]:
    try:
        return [p.pid for p in psutil.Process(pid).parents()]
    except _EXPECTED_ERRORS:
        return []


def _is_launch_wrapper(pid: int) -> bool:
    """Is ``pid`` a ``comfy ... launch`` process?

    Guards the ancestor match below against pid recycling: a recorded pid that
    has been reused by some unrelated ancestor (a login shell, say) must not
    make us signal it or forget a background server we still own.
    """
    try:
        cmdline = psutil.Process(pid).cmdline()
    except _EXPECTED_ERRORS:
        return False
    if not cmdline:
        return False
    argv0 = os.path.basename(str(cmdline[0])).lower()
    if argv0.endswith(".exe"):
        argv0 = argv0[:-4]
    return argv0 == "comfy" and "launch" in [str(tok) for tok in cmdline[1:]]


def _tracked_wrapper_pid(listener: Listener) -> int | None:
    """The recorded background pid, when *this* listener is that server.

    `comfy launch --background` records the pid of the outer ``comfy ... launch``
    wrapper, not the ``python main.py`` it spawns — so an exact-pid match against
    the listener essentially never fires for a server this CLI started. The
    wrapper is matched as an ancestor too, but only when it really is a
    ``comfy ... launch`` process.
    """
    bg_info = ConfigManager().background
    if not bg_info:
        return None
    bg_pid = bg_info[2]
    if bg_pid == listener.pid:
        return bg_pid
    if bg_pid in _ancestor_pids(listener.pid) and _is_launch_wrapper(bg_pid):
        return bg_pid
    return None


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

    # If this listener is in fact the server `comfy launch --background` started,
    # tear down from the recorded wrapper: it is the pid `comfy stop` owns, and
    # its output-redirector threads never exit on their own, so stopping only the
    # inner python would leak it and leave the background record behind.
    wrapper_pid = _tracked_wrapper_pid(listener)
    if wrapper_pid is not None and wrapper_pid != listener.pid:
        result = kill_process_tree(wrapper_pid)
    else:
        result = kill_process_tree(listener.pid, create_time=listener.create_time)

    survivors = list(result.survivors)
    # The identified process is not necessarily the only thing bound to the port
    # (SO_REUSEPORT, split v4/v6 sockets), and children we could not enumerate
    # may still hold it. Confirm the port actually came free.
    still = find_listener(port)
    if still is not None and still.pid not in survivors:
        survivors.append(still.pid)

    if not result.ok or still is not None:
        message = (
            f"Stopped pid {listener.pid}, but port {port} is still held by pid {still.pid}."
            if result.ok and still is not None
            else f"Failed to stop pid {listener.pid} on port {port}."
        )
        renderer.error(
            code="stop_failed",
            message=message,
            command="stop",
            details={"port": port, "pid": listener.pid, "survivors": survivors},
        )
        raise typer.Exit(code=1)

    if wrapper_pid is not None:
        ConfigManager().remove_background()

    renderer.print(f"[bold yellow]ComfyUI on port {port} is stopped.[/bold yellow] (pid={listener.pid})")
    data: dict[str, Any] = {"stopped": True, "pid": listener.pid, "port": port, "untracked": True}
    if not result.children_enumerated:
        # Honest about the gap: the parent is gone and the port is free, but the
        # child list was unreadable, so we cannot claim the whole tree is down.
        data["children_unknown"] = True
    renderer.emit(data, command="stop", changed=True)
