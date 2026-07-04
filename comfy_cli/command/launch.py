from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque

import typer
from rich.console import Console
from rich.panel import Panel

from comfy_cli import constants, utils
from comfy_cli.command.custom_nodes.cm_cli_util import find_cm_cli, resolve_manager_gui_mode
from comfy_cli.config_manager import ConfigManager
from comfy_cli.env_checker import check_comfy_server_running
from comfy_cli.output import rprint as print  # context-aware print: stderr in JSON mode
from comfy_cli.resolve_python import resolve_workspace_python
from comfy_cli.workspace_manager import WorkspaceManager, WorkspaceType

workspace_manager = WorkspaceManager()
console = Console()


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
        while True:
            res = subprocess.run([python, "main.py"] + extra, env=new_env, check=False)

            if reboot_path is None:
                print("[bold red]ComfyUI is not installed.[/bold red]\n")
                exit(res.returncode)

            if not os.path.exists(reboot_path):
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
                    os._exit(1)

                if not os.path.exists(reboot_path):
                    os._exit(process.returncode)

                os.remove(reboot_path)
        except KeyboardInterrupt:
            if process is not None:
                os._exit(1)


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
    config_background = ConfigManager().background
    if config_background is not None and utils.is_running(config_background[2]):
        print(
            "[bold red]ComfyUI is already running in background.\nYou cannot start more than one background service.[/bold red]\n"
        )
        raise typer.Exit(code=1)

    port = 8188
    listen = "127.0.0.1"

    if extra is not None:
        for i in range(len(extra) - 1):
            if extra[i] == "--port":
                port = extra[i + 1]
            if extra[i] == "--listen":
                listen = extra[i + 1]

        if len(extra) > 0:
            extra = ["--"] + extra
    else:
        extra = []

    if check_comfy_server_running(port):
        print(f"[bold red]The {port} port is already in use. A new ComfyUI server cannot be launched.\n[bold red]\n")
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

    if log is not None:
        print(
            Panel(
                "".join(log),
                title="[bold red]Error log during ComfyUI execution[/bold red]",
                border_style="bright_red",
            )
        )

    print("\n[bold red]Execution error: failed to launch ComfyUI[/bold red]\n")
    # NOTE: os.exit(0) doesn't work
    os._exit(1)


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
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Truncate-on-launch: each background launch starts a fresh log. The child
    # inherits its own dup of this fd, so writes continue after we (the monitor)
    # close our handle and after we os._exit on success.
    logfh = open(log_path, "w", encoding="utf-8")
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
                f"[bold yellow]ComfyUI is successfully launched in the background.[/bold yellow]\nTo see the GUI go to: http://{listen}:{port}"
            )
            cfg = ConfigManager()
            cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND] = f"{(listen, port, process.pid)}"
            cfg.config["DEFAULT"][constants.CONFIG_KEY_BACKGROUND_LOG] = log_path
            cfg.write_config()

            # NOTE: os.exit(0) doesn't work.
            os._exit(0)
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
    while lines and size > max_bytes:
        size -= len(lines.pop(0).encode("utf-8"))
        truncated = True

    return lines, truncated


def resolve_background_log_path() -> str | None:
    """Locate the background logfile: the path recorded at launch, else the
    default derived from the resolved workspace and background/default port.

    Returns None when no workspace can be resolved and nothing was recorded.
    """
    cfg = ConfigManager()
    recorded = cfg.get(constants.CONFIG_KEY_BACKGROUND_LOG)
    if recorded:
        return recorded

    workspace = workspace_manager.workspace_path
    if not workspace:
        return None

    port = cfg.background[1] if cfg.background else 8188
    return background_log_path(port, workspace)


def logs(tail: int = 200, where: str | None = None):
    """Print the tail of the background ComfyUI log captured by `comfy launch`."""
    from comfy_cli.output import get_renderer

    renderer = get_renderer()

    if where is not None and where.strip().lower() != "local":
        renderer.error(
            code="where_invalid",
            message="`comfy logs` reads a local logfile; only `--where local` is supported.",
            hint="drop --where, or pass --where local",
            command="logs",
        )
        raise typer.Exit(code=1)

    log_path = resolve_background_log_path()
    if not log_path or not os.path.isfile(log_path):
        renderer.error(
            code="no_log_file",
            message="No captured ComfyUI log was found." + (f" Looked for: {log_path}" if log_path else ""),
            hint="start ComfyUI with `comfy launch` so its output is captured",
            command="logs",
        )
        raise typer.Exit(code=1)

    lines, truncated = read_log_tail(log_path, tail)

    if renderer.is_pretty():
        # Write raw so ComfyUI log text (which can contain '[...]') isn't
        # reinterpreted as Rich markup, and byte-for-byte matches the file.
        renderer.pretty_stream.write("".join(lines))

    renderer.emit(
        {"lines": lines, "path": log_path, "truncated": truncated},
        command="logs",
        where="local",
    )
