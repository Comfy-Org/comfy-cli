from __future__ import annotations

import asyncio
import glob
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from stat import S_ISREG

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel

from comfy_cli import constants, utils
from comfy_cli.command.custom_nodes.cm_cli_util import find_cm_cli, resolve_manager_gui_mode
from comfy_cli.config_manager import ConfigManager
from comfy_cli.env_checker import _bracket_host, check_comfy_server_running
from comfy_cli.output import get_renderer
from comfy_cli.output import rprint as print  # context-aware print: stderr in JSON mode
from comfy_cli.resolve_python import resolve_workspace_python
from comfy_cli.workspace_manager import WorkspaceManager, WorkspaceType

workspace_manager = WorkspaceManager()
console = Console()


def _hard_exit(code: int) -> None:
    """`os._exit(code)`, but drain telemetry on the way out.

    Every exit path in this module uses `os._exit` (plain `sys.exit` doesn't
    work once the redirector threads are running), which skips atexit handlers —
    so `comfy_cli.tracking`'s shutdown drain never runs here. That cost nothing
    while Mixpanel sent inline from `track()`: `@track_command` fires the
    `launch` event *before* the command body runs, so it was already delivered.
    Now that dispatch is queue-and-drain (BE-5868) the event is still queued at
    this point, and without an explicit drain every `comfy launch` — background
    success included — silently drops its own telemetry.

    The drain is bounded (~5s worst case, same budget as the atexit hook) and
    best-effort; the process exits with `code` regardless.
    """
    try:
        from comfy_cli import tracking

        tracking.flush_for_hard_exit()
    except BaseException:  # noqa: BLE001  # pragma: no cover - defensive
        pass
    os._exit(code)


def _get_manager_flags() -> list[str]:
    """Get manager flags based on config mode."""
    mode = resolve_manager_gui_mode(not_installed_value=None)

    if mode is None or mode == "disable":
        return []

    # For enable-* modes, verify cm-cli is available
    if not find_cm_cli():
        print(
            "[bold yellow]Warning: ComfyUI-Manager (cm-cli) not found. "
            "Manager flags will not be injected.[/bold yellow]"
        )
        return []

    if mode == "enable-gui":
        return ["--enable-manager"]
    elif mode == "disable-gui":
        return ["--enable-manager", "--disable-manager-ui"]
    elif mode == "enable-legacy-gui":
        return ["--enable-manager", "--enable-manager-legacy-ui"]
    else:
        print(f"[bold yellow]Warning: Unknown manager mode '{mode}'. Falling back to --enable-manager.[/bold yellow]")
        return ["--enable-manager"]  # fallback to default


def _format_url(listen, port) -> str:
    """Build the ComfyUI ``http://host:port`` URL, bracketing IPv6 literals.

    An IPv6 ``--listen`` (e.g. ``::1`` or ``::``) must be wrapped in brackets or
    the bare ``host:port`` join yields an invalid URL like ``http://::1:8188``.
    Delegates the bracketing to the shared ``_bracket_host`` choke point.
    """
    return f"http://{_bracket_host(str(listen))}:{port}"


def _emit_launch_success(listen, port, pid) -> None:
    """Emit the background-launch success ``envelope/1`` for ``--json`` callers.

    Extracted from ``launch_and_monitor``'s success handler (which ``os._exit``s
    immediately after) so the terminal success envelope is unit-testable. No-op
    in pretty mode, keeping human output unchanged.
    """
    renderer = get_renderer()
    if renderer.is_json():
        renderer.emit(
            {
                "background": True,
                "listen": listen,
                "port": port,
                "url": _format_url(listen, port),
                "pid": pid,
            },
            command="launch",
            changed=True,
        )


def launch_comfyui(extra, frontend_pr=None, python=sys.executable):
    reboot_path = None

    new_env = os.environ.copy()

    session_path = os.path.join(ConfigManager().get_config_path(), "tmp", str(uuid.uuid4()))
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["PYTHONENCODING"] = "utf-8"

    # To minimize the possibility of leaving residue in the tmp directory, use files instead of directories.
    reboot_path = os.path.join(session_path + ".reboot")

    extra = extra if extra is not None else []

    # Handle temporary frontend PR
    if frontend_pr:
        from comfy_cli.command.install import handle_temporary_frontend_pr

        try:
            frontend_path = handle_temporary_frontend_pr(frontend_pr)
            if frontend_path:
                # Check if --front-end-root is not already specified
                if not any(arg.startswith("--front-end-root") for arg in extra):
                    extra = ["--front-end-root", frontend_path] + extra
        except Exception as e:
            print(f"[bold red]Failed to prepare frontend PR: {e}[/bold red]")
            # Continue with default frontend

    process = None

    if "COMFY_CLI_BACKGROUND" not in os.environ:
        # If not running in background mode, there's no need to use popen. This can prevent the issue of linefeeds occurring with tqdm.
        # Under --json the child inherits stdout by default, so ComfyUI's raw
        # non-JSON output would land on stdout ahead of our envelope and break
        # the single-envelope-on-stdout contract for machine callers. Redirect
        # the child's stdout to stderr in JSON mode; leave it inherited in pretty
        # mode so humans still see ComfyUI's output on the terminal.
        child_stdout = sys.stderr if get_renderer().is_json() else None
        while True:
            res = subprocess.run([python, "main.py"] + extra, env=new_env, check=False, stdout=child_stdout)

            if reboot_path is None:
                print("[bold red]ComfyUI is not installed.[/bold red]\n")
                exit(res.returncode)

            if not os.path.exists(reboot_path):
                # Foreground server exited (Ctrl-C, crash, or clean stop). Emit
                # the lifecycle envelope so a `--json` caller gets a terminal
                # verdict instead of nothing; no-op in pretty mode.
                renderer = get_renderer()
                if renderer.is_json():
                    if res.returncode == 0:
                        renderer.emit(
                            {"background": False, "returncode": res.returncode},
                            command="launch",
                            changed=True,
                        )
                    else:
                        renderer.error(
                            code="launch_failed",
                            message=f"ComfyUI exited with code {res.returncode}",
                            command="launch",
                            details={"returncode": res.returncode},
                        )
                exit(res.returncode)

            os.remove(reboot_path)
    else:
        # If running in background mode without using a popen, broken pipe errors may occur when flushing stdout/stderr.
        def redirector_stderr():
            while True:
                if process is not None and process.stderr is not None:
                    print(process.stderr.readline(), end="")

        def redirector_stdout():
            while True:
                if process is not None and process.stdout is not None:
                    print(process.stdout.readline(), end="")

        threading.Thread(target=redirector_stderr).start()
        threading.Thread(target=redirector_stdout).start()

        try:
            while True:
                if sys.platform == "win32":
                    process = subprocess.Popen(
                        [python, "main.py"] + extra,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=new_env,
                        encoding="utf-8",
                        shell=True,  # win32 only
                        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # win32 only
                    )
                else:
                    process = subprocess.Popen(
                        [python, "main.py"] + extra,
                        text=True,
                        env=new_env,
                        encoding="utf-8",
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )

                process.wait()

                if reboot_path is None:
                    print("[bold red]ComfyUI is not installed.[/bold red]\n")
                    _hard_exit(1)

                if not os.path.exists(reboot_path):
                    _hard_exit(process.returncode)

                os.remove(reboot_path)
        except KeyboardInterrupt:
            if process is not None:
                _hard_exit(1)


def launch(
    background: bool = False,
    extra: list[str] | None = None,
    frontend_pr: str | None = None,
):
    resolved_workspace = workspace_manager.workspace_path

    if not resolved_workspace:
        print(
            "\nComfyUI is not available.\nTo install ComfyUI, you can run:\n\n\tcomfy install\n\n",
            file=sys.stderr,
        )
        renderer = get_renderer()
        if renderer.is_json():
            renderer.error(
                code="not_in_workspace",
                message="ComfyUI is not available.",
                hint="run `comfy install`, or pass `--workspace`",
                command="launch",
            )
        raise typer.Exit(code=1)

    if (extra is None or len(extra) == 0) and workspace_manager.workspace_type == WorkspaceType.DEFAULT:
        launch_extras = workspace_manager.config_manager.config["DEFAULT"].get(
            constants.CONFIG_KEY_DEFAULT_LAUNCH_EXTRAS, ""
        )

        if launch_extras != "":
            extra = launch_extras.split(" ")

    print(f"\nLaunching ComfyUI from: {resolved_workspace}\n")

    # Update the recent workspace
    workspace_manager.set_recent_workspace(resolved_workspace)

    os.chdir(resolved_workspace)
    python = resolve_workspace_python(resolved_workspace)

    # Inject manager flags based on config mode
    manager_flags = _get_manager_flags()
    if manager_flags:
        extra = (extra or []) + manager_flags

    if background:
        background_launch(extra, frontend_pr)
    else:
        launch_comfyui(extra, frontend_pr, python=python)


def background_launch(extra, frontend_pr=None):
    renderer = get_renderer()
    config_background = ConfigManager().background
    if config_background is not None and utils.is_running(config_background[2]):
        print(
            "[bold red]ComfyUI is already running in background.\nYou cannot start more than one background service.[/bold red]\n"
        )
        if renderer.is_json():
            renderer.error(
                code="server_already_running",
                message="ComfyUI is already running in background.",
                hint="run `comfy stop` before launching another background service",
                command="launch",
            )
        raise typer.Exit(code=1)

    port = 8188
    listen = "127.0.0.1"

    if extra is not None:
        # Accept both the two-token (``--port 9000``) and the ``--port=9000``
        # forms; the latter is common and was previously ignored, leaving port
        # at the 8188 default and silently disagreeing with what ComfyUI binds.
        for i, tok in enumerate(extra):
            if tok == "--port" and i + 1 < len(extra):
                port = extra[i + 1]
            elif tok.startswith("--port="):
                port = tok[len("--port=") :]
            elif tok == "--listen" and i + 1 < len(extra):
                listen = extra[i + 1]
            elif tok.startswith("--listen="):
                listen = tok[len("--listen=") :]

        if len(extra) > 0:
            extra = ["--"] + extra
    else:
        extra = []

    # Validate --port as an integer in the valid TCP range. It flows into the log
    # path (``comfyui_<port>.log``); a non-integer value like ``../../etc/x``
    # would otherwise escape the workspace when the logfile is created, and an
    # out-of-range value (e.g. -1 or 70000) would degrade to a generic
    # launch_failed instead of this precise port_invalid verdict.
    try:
        port_int = int(port)
    except (TypeError, ValueError):
        port_int = None
    if port_int is None or not (1 <= port_int <= 65535):
        print(f"[bold red]Invalid --port value {port!r}; expected an integer in 1-65535.[/bold red]\n")
        if renderer.is_json():
            renderer.error(
                code="port_invalid",
                message=f"Invalid --port value {port!r}; expected an integer in 1-65535.",
                command="launch",
                details={"port": port},
            )
        raise typer.Exit(code=1)
    port = port_int

    if check_comfy_server_running(port):
        print(f"[bold red]The {port} port is already in use. A new ComfyUI server cannot be launched.\n[bold red]\n")
        if renderer.is_json():
            renderer.error(
                code="port_in_use",
                message=f"The {port} port is already in use. A new ComfyUI server cannot be launched.",
                command="launch",
                details={"port": port},
            )
        raise typer.Exit(code=1)

    cmd = [
        "comfy",
        f"--workspace={os.path.abspath(os.getcwd())}",
        "launch",
    ]

    # Add frontend PR option if specified
    if frontend_pr:
        cmd.extend(["--frontend-pr", frontend_pr])

    cmd.extend(extra)

    log = asyncio.run(launch_and_monitor(cmd, listen, port))

    # Reaching here means the monitor returned without seeing the success line
    # (the success path emits its envelope and _hard_exit(0)s inside the monitor).
    if log is not None:
        print(
            Panel(
                "".join(log),
                title="[bold red]Error log during ComfyUI execution[/bold red]",
                border_style="bright_red",
            )
        )

    print("\n[bold red]Execution error: failed to launch ComfyUI[/bold red]\n")
    if renderer.is_json():
        renderer.error(
            code="launch_failed",
            message="Execution error: failed to launch ComfyUI",
            command="launch",
            details={"log": _bounded_log(log)} if log else None,
        )
    # NOTE: os.exit(0) doesn't work
    _hard_exit(1)


def background_log_path(port, workspace: str | None = None) -> str:
    """Path to the persisted background ComfyUI log for ``port``.

    ``<workspace>/user/comfyui_<port>.log`` — ``<port>`` disambiguates multiple
    installs that share one comfy-cli config. Truncated on each background launch
    (a fresh run starts a fresh log). ``workspace`` defaults to the current
    working directory, which ``launch`` has already ``chdir``'d to the resolved
    workspace by the time the background monitor runs.
    """
    if workspace is None:
        workspace = os.path.abspath(os.getcwd())
    return os.path.join(workspace, "user", f"comfyui_{port}.log")


def _open_log_for_write(log_path: str):
    """Open ``log_path`` for a truncating write, refusing to follow a symlink.

    On a shared host an attacker with write access to ``<workspace>/user`` could
    pre-place ``comfyui_<port>.log`` as a symlink so ``open(..., "w")`` clobbers
    the link target. ``O_NOFOLLOW`` makes the open fail (``ELOOP``) instead. The
    flag is absent on some platforms (older Windows); there ``getattr`` yields 0
    and we fall back to a plain truncating open. The file is created owner-only
    (``0o600``) — consistent with the shared-host threat model above, the log
    isn't meant to be world-readable.
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(log_path, flags, 0o600)
    return os.fdopen(fd, "w", encoding="utf-8")


async def launch_and_monitor(cmd, listen, port):
    """
    Monitor the process during the background launch.

    ComfyUI's stdout/stderr are redirected straight onto the child's own file
    descriptors pointing at a workspace logfile (``<workspace>/user/comfyui_<port>.log``,
    truncate-on-launch). Because the redirect lives on the child's fds — not on a
    monitor thread — every line still lands in the file AFTER this monitor exits
    on the success signal (the ComfyUI child outlives the monitor). The monitor
    tails that same file to detect the "To see the GUI go to:" success line.

    If a success message is captured, record the background info and exit;
    otherwise, return the log in case of failure.
    """
    logging_flag = False
    log = []

    # NOTE: To prevent encoding error on Windows platform
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env["COMFY_CLI_BACKGROUND"] = "true"
    # Flush the child's stdout per line so the success marker reaches the logfile
    # promptly instead of sitting in a block buffer (stdout is a file, not a tty).
    env["PYTHONUNBUFFERED"] = "1"

    log_path = background_log_path(port)

    # Truncate-on-launch: each background launch starts a fresh log. The child
    # inherits its own dup of this fd, so writes continue after we (the monitor)
    # close our handle and after we os._exit on success. Failing to create the
    # log (read-only/permission-restricted workspace) is reported cleanly rather
    # than aborting launch with a raw traceback.
    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logfh = _open_log_for_write(log_path)
    except OSError as e:
        print(f"[bold red]Could not open background log file {log_path}: {e}[/bold red]\n")
        renderer = get_renderer()
        if renderer.is_json():
            renderer.error(
                code="launch_failed",
                message=f"Could not open background log file {log_path}: {e}",
                command="launch",
                details={"log_path": log_path},
            )
        _hard_exit(1)

    # Record the log path up front so `comfy logs` can surface a crash log even
    # when startup fails before the success marker below (where the running
    # background info is recorded). A fresh ConfigManager re-reads this on the
    # success path.
    cfg = ConfigManager()
    cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND_LOG] = log_path
    cfg.write_config()

    try:
        if sys.platform == "win32":
            process = subprocess.Popen(
                cmd,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                env=env,
                shell=True,  # win32 only
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,  # win32 only
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=logfh,
                stderr=subprocess.STDOUT,
                env=env,
            )
    finally:
        # The child holds its own fd now; drop the monitor's copy so tailing sees
        # a stable, child-owned writer.
        logfh.close()

    def _handle(line):
        nonlocal logging_flag
        if "Launching ComfyUI from:" in line:
            logging_flag = True
        if "To see the GUI go to:" in line:
            print(
                f"[bold yellow]ComfyUI is successfully launched in the background.[/bold yellow]\nTo see the GUI go to: {_format_url(listen, port)}"
            )
            # CONFIG_KEY_BACKGROUND_LOG was already recorded before launch; here
            # we add the running background info now that startup succeeded.
            cfg = ConfigManager()
            cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND] = f"{(listen, port, process.pid)}"
            cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND_LOG] = log_path
            cfg.write_config()

            _emit_launch_success(listen, port, process.pid)

            # NOTE: os.exit(0) doesn't work.
            _hard_exit(0)
        if logging_flag:
            log.append(line)

    # Tail the logfile the child is writing, reassembling whole lines (a
    # concurrent writer can leave a trailing partial line without a newline).
    with open(log_path, encoding="utf-8", errors="replace") as reader:
        pending = ""
        while True:
            chunk = reader.readline()
            if chunk:
                pending += chunk
                if pending.endswith("\n") or process.poll() is not None:
                    _handle(pending)
                    pending = ""
                # else: partial line — wait for the rest before acting.
            else:
                if process.poll() is not None:
                    if pending:
                        _handle(pending)
                    break
                time.sleep(0.1)

    return log


# Output caps for `comfy logs`, so `--json` payloads stay bounded even if the
# caller asks for a huge --tail against a long-running server's log.
LOGS_MAX_LINES = 2000
LOGS_MAX_BYTES = 256 * 1024


def read_log_tail(
    path: str,
    n: int,
    *,
    max_lines: int = LOGS_MAX_LINES,
    max_bytes: int = LOGS_MAX_BYTES,
) -> tuple[list[str], bool]:
    """Return ``(lines, truncated)`` — the last ``n`` lines of ``path``.

    Bounded so machine output stays small: at most ``max_lines`` lines and
    ``max_bytes`` bytes (whichever binds first), trimmed from the top.
    ``truncated`` is True when a cap dropped lines the caller would otherwise
    have received (NOT for the ordinary case of a tail omitting earlier lines,
    nor for a file shorter than ``n``).
    """
    n = max(0, n)
    want = min(n, max_lines)

    # deque(maxlen) keeps only the last ``want`` lines in memory regardless of
    # file size. Count total lines in the same pass to decide truncation.
    tail: deque[str] = deque(maxlen=want) if want > 0 else deque(maxlen=0)
    total = 0
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            tail.append(line)
            total += 1

    lines = list(tail)
    # The line cap actually dropped content only if the caller asked for more
    # than the cap AND the file had more than the cap to give.
    truncated = n > max_lines and total > max_lines

    size = sum(len(line.encode("utf-8")) for line in lines)
    while len(lines) > 1 and size > max_bytes:
        size -= len(lines.pop(0).encode("utf-8"))
        truncated = True

    # If the sole remaining line is itself larger than the byte cap, keep a
    # byte-truncated tail of it rather than dropping all output for a non-empty
    # logfile (e.g. a single huge newline-less line).
    if lines and size > max_bytes:
        lines[-1] = lines[-1].encode("utf-8")[-max_bytes:].decode("utf-8", errors="replace")
        truncated = True

    return lines, truncated


def _bounded_log(lines, *, max_lines: int = LOGS_MAX_LINES, max_bytes: int = LOGS_MAX_BYTES) -> str:
    """Join an in-memory captured log (list of lines) for a JSON envelope,
    applying the same ``LOGS_MAX_LINES``/``LOGS_MAX_BYTES`` caps as
    ``read_log_tail``. A verbose failing launch can otherwise produce an
    unbounded JSON payload (and leak any secrets present in the log) to machine
    consumers. Keeps the last ``max_lines`` lines, then trims from the top until
    within ``max_bytes``.
    """
    tail = list(lines)[-max_lines:]
    size = sum(len(s.encode("utf-8")) for s in tail)
    while len(tail) > 1 and size > max_bytes:
        size -= len(tail.pop(0).encode("utf-8"))
    if tail and size > max_bytes:
        tail[-1] = tail[-1].encode("utf-8")[-max_bytes:].decode("utf-8", errors="replace")
    return "".join(tail)


# ComfyUI-Manager rotates its logfile on startup: `comfyui.log` becomes
# `comfyui.prev.log`, and the previous `.prev` becomes `.prev2`. Those rotations
# are stale by construction, so the glob fallback must never serve one.
_ROTATED_LOG_RE = re.compile(r"\.prev\d*\.log$")

# `<workspace>/user/comfyui_<port>.log` — the suffix carries the port the server
# was serving, which is what makes a port mismatch detectable after the fact.
_PORTED_LOG_RE = re.compile(r"^comfyui_(\d+)\.log$")

DEFAULT_LOG_PORT = 8188


def unsuffixed_log_path(workspace: str) -> str:
    """``<workspace>/user/comfyui.log`` — ComfyUI-Manager's own logfile.

    Manager only appends ``_<port>`` when ``--port`` is in ComfyUI's argv, so a
    server started outside ``comfy launch --background`` (foreground, desktop
    app, no explicit ``--port``) logs here and nowhere else. `comfy install`
    installs Manager, so this file exists on a default install.
    """
    return os.path.join(workspace, "user", "comfyui.log")


def _path_within(path: str, directory: str) -> bool:
    """True when ``path`` lies inside ``directory``, compared lexically.

    Deliberately does NOT touch the filesystem (no ``realpath``): the caller
    only needs to know whether a *recorded* path claims to belong to the current
    workspace, and that question must stay answerable for a file that no longer
    exists.
    """
    try:
        target = os.path.normcase(os.path.abspath(path))
        root = os.path.normcase(os.path.abspath(directory))
    except (OSError, ValueError):
        return False
    return target == root or target.startswith(root.rstrip(os.sep) + os.sep)


def candidate_log_paths(port: int | None = None) -> list[tuple[str, str]]:
    """The ordered candidates `comfy logs` considers, existing or not.

    Each entry is ``(path, source)``; the ``fallback_glob`` entry's "path" is a
    glob PATTERN rather than a concrete file. Kept separate from resolution so
    the miss path can report everything that was checked.

    With ``port`` given (``comfy logs --port N``) the list is restricted to that
    port's logfile plus the unsuffixed Manager file — the latter covers a server
    on port N that was started without an explicit ``--port`` argv flag, which is
    exactly when Manager omits the suffix.
    """
    workspace = workspace_manager.workspace_path

    if port is not None:
        if not workspace:
            return []
        return [
            (background_log_path(port, workspace), "explicit_port"),
            (unsuffixed_log_path(workspace), "fallback_unsuffixed"),
        ]

    cfg = ConfigManager()
    recorded = cfg.get(constants.CONFIG_KEY_BACKGROUND_LOG)

    if not workspace:
        # Nothing local to prefer it over, and nothing else to check.
        return [(recorded, "recorded")] if recorded else []

    live_port = cfg.background[1] if cfg.background else None
    derived_port = live_port if live_port is not None else DEFAULT_LOG_PORT
    local: list[tuple[str, str]] = [
        (
            background_log_path(derived_port, workspace),
            "derived_port" if live_port is not None else "default_port",
        ),
        (unsuffixed_log_path(workspace), "fallback_unsuffixed"),
        (os.path.join(workspace, "user", "comfyui_*.log"), "fallback_glob"),
    ]

    if not recorded:
        return local

    # The recorded pointer outranks the workspace-local candidates only when it
    # is still relevant HERE: either the background server it describes is live,
    # or it names a file inside the current workspace (the crash-log case — the
    # pointer deliberately survives `comfy stop` and dead-pid cleanup).
    #
    # Otherwise it is a pointer into some OTHER workspace left over from an
    # earlier run, and serving it ahead of this workspace's own `comfyui_*.log`
    # would silently show a cross-workspace log with no live server to raise
    # `port_mismatch`. Demote it to last so it is still a usable last resort
    # when this workspace has no log at all, but never shadows a local one.
    if live_port is not None or _path_within(recorded, workspace):
        return [(recorded, "recorded"), *local]
    return [*local, (recorded, "recorded")]


def _newest_globbed_log(pattern: str) -> str | None:
    """Newest-mtime match of ``pattern``, ignoring ComfyUI-Manager's rotations.

    Only the BASENAME of ``pattern`` is a glob; its directory is escaped, so a
    workspace path containing glob metacharacters (``/Users/a[1]/comfy``) still
    matches instead of silently globbing nothing.

    Symlinks are skipped: on a shared host an attacker with write access to
    ``<workspace>/user`` could plant ``comfyui_<n>.log`` as a link to a file
    outside the workspace and backdate/forward-date it to win the newest-mtime
    pick, making `comfy logs` disclose the target. ``lstat`` + ``S_ISREG`` is the
    read-side counterpart of ``_open_log_for_write``'s ``O_NOFOLLOW``.
    """
    directory, filename = os.path.split(pattern)
    newest: str | None = None
    newest_mtime = float("-inf")
    for path in glob.glob(os.path.join(glob.escape(directory), filename)):
        if _ROTATED_LOG_RE.search(os.path.basename(path)):
            continue
        try:
            # lstat, not stat: a symlink must not be resolved to its target here.
            st = os.lstat(path)
            if not S_ISREG(st.st_mode):
                continue
            mtime = st.st_mtime
        except OSError:
            # Raced away between glob and lstat, or unreadable — not a candidate.
            continue
        if mtime > newest_mtime:
            newest, newest_mtime = path, mtime
    return newest


def resolve_background_log_path(port: int | None = None) -> tuple[str, str] | None:
    """Locate a ComfyUI logfile, returning ``(path, source)`` for the first
    candidate of :func:`candidate_log_paths` that actually exists.

    Returns None when no candidate exists — including the "no workspace resolves
    and nothing was recorded" case, where there is nothing to check at all.
    """
    for path, source in candidate_log_paths(port):
        if source == "fallback_glob":
            match = _newest_globbed_log(path)
            if match:
                return match, source
        elif os.path.isfile(path):
            return path, source
    return None


def _served_log_port(path: str) -> int | None:
    """The port encoded in a ``comfyui_<port>.log`` filename, else None."""
    match = _PORTED_LOG_RE.match(os.path.basename(path))
    return int(match.group(1)) if match else None


def logs(tail: int = 200, where: str | None = None, port: int | None = None):
    """Print the tail of a captured ComfyUI log.

    Resolution walks :func:`candidate_log_paths` — the path recorded by
    `comfy launch --background`, the port-derived logfile, ComfyUI-Manager's
    unsuffixed ``user/comfyui.log``, then the newest ``user/comfyui_*.log`` —
    so a server started outside `comfy launch --background` is still readable.
    ``port`` restricts that walk to a single port's log (plus the unsuffixed
    file, which is where a server on that port lands when it was started without
    an explicit ``--port`` argv flag).
    """
    from comfy_cli import where as where_mod

    renderer = get_renderer()

    # Honor the same routing precedence as the rest of the CLI (flag, COMFY_WHERE,
    # project comfy.yaml, persisted where_default) instead of only the --where flag,
    # so `comfy logs` errors when routing is *explicitly* pointed at cloud. The
    # cloud-credentials auto-detect (source="auto") is deliberately NOT treated as
    # an explicit choice: `comfy logs` is a local-only command, so simply having
    # cloud creds configured shouldn't force a --where local on every invocation.
    try:
        resolution = where_mod.resolve_default(flag=where)
    except ValueError as e:
        renderer.error(
            code="where_invalid",
            message=str(e),
            hint="pass --where local, or set routing to local",
            command="logs",
        )
        raise typer.Exit(code=1)
    if resolution.target is not where_mod.WhereTarget.LOCAL and resolution.source != "auto":
        renderer.error(
            code="where_invalid",
            message="`comfy logs` reads a local logfile; only `local` routing is supported.",
            hint="pass --where local, or set routing to local",
            command="logs",
        )
        raise typer.Exit(code=1)

    resolved = resolve_background_log_path(port)
    if resolved is None:
        # Report EVERY candidate that was checked, not just the last one — with
        # four candidates "no log file" is otherwise unactionable.
        checked = [path for path, _ in candidate_log_paths(port)]
        renderer.error(
            code="no_log_file",
            message="No captured ComfyUI log was found." + (f" Looked for: {', '.join(checked)}" if checked else ""),
            hint="start ComfyUI with `comfy launch` so its output is captured",
            command="logs",
        )
        raise typer.Exit(code=1)
    log_path, source = resolved

    # Pretty output goes to a human terminal: honor the requested --tail. The
    # line/byte caps exist to keep JSON payloads bounded, so apply them only in
    # machine mode (matching the --tail help text).
    if renderer.is_pretty():
        read_kwargs = {"max_lines": max(tail, LOGS_MAX_LINES), "max_bytes": sys.maxsize}
    else:
        read_kwargs = {}

    try:
        lines, truncated = read_log_tail(log_path, tail, **read_kwargs)
    except OSError as e:
        # The isfile() check above is best-effort; the file can vanish or become
        # unreadable in the TOCTOU window. Emit a clean error, not a raw traceback.
        renderer.error(
            code="log_read_failed",
            message=f"Could not read log file {log_path}: {e}",
            hint="check the file still exists and is readable",
            command="logs",
        )
        raise typer.Exit(code=1)

    # Staleness metadata: which candidate won, how old/big the served file is,
    # and whether it belongs to a different port than the live background server
    # (the wrong-port-empty-log case a failed launch attempt leaves behind).
    try:
        st = os.stat(log_path)
        mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
        size = st.st_size
    except (OSError, ValueError, OverflowError):
        # OSError: the same TOCTOU window as the read above. ValueError/OverflowError:
        # an out-of-range or corrupt st_mtime that fromtimestamp can't represent
        # (notably any negative value on Windows). Metadata is best-effort.
        mtime, size = None, None

    background = ConfigManager().background
    served_port = _served_log_port(log_path)
    # Suppressed when --port was passed: there the served file is *by definition*
    # the port the user asked for, so reporting it as a mismatch — and advising
    # them to retry with the live port they deliberately did not ask for — would
    # contradict the request they just made.
    port_mismatch = bool(port is None and background and served_port is not None and served_port != background[1])

    if renderer.is_pretty():
        if port_mismatch:
            # Without this a human just sees an empty/stale file and no reason why.
            print(
                f"[bold yellow]Warning: showing {escape(log_path)} (port {served_port}), but the running "
                f"background server is on port {background[1]}. "
                f"Try `comfy logs --port {background[1]}`.[/bold yellow]"
            )
        elif port is not None and source == "fallback_unsuffixed":
            # An explicit --port that resolved to the unsuffixed Manager log is a
            # deliberate fallback (a server on that port started without a --port
            # argv flag logs only there) — but that file encodes no port, so it
            # can equally be some other port's log and `port_mismatch` cannot
            # detect it. Say so rather than answering the request silently.
            print(
                f"[bold yellow]Warning: no comfyui_{port}.log was found; showing "
                f"{escape(log_path)}, ComfyUI-Manager's unsuffixed log, which does not "
                f"record which port it served.[/bold yellow]"
            )
        # Write raw so ComfyUI log text (which can contain '[...]') isn't
        # reinterpreted as Rich markup, and byte-for-byte matches the file.
        renderer.pretty_stream.write("".join(lines))

    renderer.emit(
        {
            "lines": lines,
            "path": log_path,
            "truncated": truncated,
            "source": source,
            "mtime": mtime,
            "size": size,
            "port_mismatch": port_mismatch,
        },
        command="logs",
        where="local",
    )
