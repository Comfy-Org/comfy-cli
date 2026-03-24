from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import uuid
from functools import lru_cache

import typer
from rich import print

from comfy_cli.config_manager import ConfigManager
from comfy_cli.resolve_python import resolve_workspace_python
from comfy_cli.uv import DependencyCompiler
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

workspace_manager = WorkspaceManager()

# set of commands that invalidate (ie require an update of) dependencies after they are run
_dependency_cmds = {
    "install",
    "reinstall",
}


@lru_cache(maxsize=1)
def find_cm_cli() -> bool:
    """Check if cm_cli module is available in the workspace Python.

    First checks the workspace venv Python (primary path — matches the Python
    used by execute_cm_cli). Falls back to the current Python environment only
    when the workspace Python is the same as sys.executable.

    Results are cached for the session lifetime.
    """
    ws = workspace_manager.workspace_path
    if ws:
        python = resolve_workspace_python(ws)
        if python != sys.executable:
            # Workspace uses a different Python — check that one
            try:
                result = subprocess.run(
                    [python, "-c", "import cm_cli"],
                    capture_output=True,
                    timeout=10,
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, OSError):
                return False

    # Same Python or no workspace — check current environment
    return importlib.util.find_spec("cm_cli") is not None


def resolve_manager_gui_mode(not_installed_value: str | None = None) -> str | None:
    """Resolve manager GUI mode from config, with legacy migration.

    Priority: CONFIG_KEY_MANAGER_GUI_MODE > CONFIG_KEY_MANAGER_GUI_ENABLED > auto-detect.

    Args:
        not_installed_value: Value to return when manager is not installed and no config exists.
            Callers use None (launch — means "no flags") or "not-installed" (display).
    """
    from comfy_cli import constants

    config_manager = ConfigManager()
    mode = config_manager.get(constants.CONFIG_KEY_MANAGER_GUI_MODE)

    if mode is not None:
        return mode

    # Legacy migration
    old_value = config_manager.get(constants.CONFIG_KEY_MANAGER_GUI_ENABLED)
    if old_value is not None:
        old_str = str(old_value).lower()
        if old_str in ("false", "0", "off"):
            return "disable"
        if old_str in ("true", "1", "on"):
            return "enable-gui"

    # No config at all — check manager availability
    if not find_cm_cli():
        return not_installed_value
    return "enable-gui"


def execute_cm_cli(
    args, channel=None, fast_deps=False, no_deps=False, uv_compile=False, mode=None, raise_on_error=False
) -> str | None:
    _config_manager = ConfigManager()

    workspace_path = workspace_manager.workspace_path

    if not workspace_path:
        print("\n[bold red]ComfyUI path is not resolved.[/bold red]\n", file=sys.stderr)
        raise typer.Exit(code=1)

    if not check_comfy_repo(workspace_path)[0]:
        print(
            f"\n[bold red]'{workspace_path}' is not a valid ComfyUI workspace.[/bold red]\n"
            "Run [bold]comfy install[/bold] to set up ComfyUI, or use [bold]--workspace <path>[/bold] to specify a valid path.\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    if not find_cm_cli():
        print(
            "\n[bold red]ComfyUI-Manager not found. 'cm-cli' command is not available.[/bold red]\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    python = resolve_workspace_python(workspace_path)
    cmd = [python, "-m", "cm_cli"] + args

    if channel is not None:
        cmd += ["--channel", channel]

    if uv_compile:
        cmd += ["--uv-compile"]
    elif fast_deps or no_deps:
        cmd += ["--no-deps"]

    if mode is not None:
        cmd += ["--mode", mode]

    new_env = os.environ.copy()
    session_path = os.path.join(_config_manager.get_config_path(), "tmp", str(uuid.uuid4()))
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["COMFYUI_PATH"] = workspace_path
    new_env["PYTHONUNBUFFERED"] = "1"

    print(f"Execute from: {workspace_path}")
    print(f"Command: {cmd}")
    try:
        process = subprocess.Popen(
            cmd,
            env=new_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        # Read stderr in a background thread to avoid pipe deadlock on Windows.
        # Windows pipe buffers are small (4 KB); if stderr fills up while the main
        # thread is blocked reading stdout line-by-line, the child process blocks
        # on stderr writes and never closes stdout — classic deadlock.
        stderr_lines: list[str] = []

        def _drain_stderr():
            for line in process.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        stdout_lines = []
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            stdout_lines.append(line)

        stderr_thread.join(timeout=10)
        return_code = process.wait()
        stdout_output = "".join(stdout_lines)
        stderr_output = "".join(stderr_lines)
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=stdout_output, stderr=stderr_output)

        if fast_deps and args[0] in _dependency_cmds:
            # we're using the fast_deps behavior and just ran a command that invalidated the dependencies
            depComp = DependencyCompiler(cwd=workspace_path, executable=python)
            depComp.compile_deps()
            depComp.install_deps()

        workspace_manager.set_recent_workspace(workspace_path)
        return stdout_output
    except subprocess.CalledProcessError as e:
        if raise_on_error:
            raise e

        if e.returncode == 1:
            print(f"\n[bold red]Execution error: {cmd}[/bold red]\n", file=sys.stderr)
            return None

        if e.returncode == 2:
            return None

        raise e
