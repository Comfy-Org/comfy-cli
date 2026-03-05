from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import uuid
from functools import lru_cache

import typer
from rich import print

from comfy_cli.config_manager import ConfigManager
from comfy_cli.uv import DependencyCompiler
from comfy_cli.workspace_manager import WorkspaceManager

workspace_manager = WorkspaceManager()

# set of commands that invalidate (ie require an update of) dependencies after they are run
_dependency_cmds = {
    "install",
    "reinstall",
}


@lru_cache(maxsize=1)
def find_cm_cli() -> bool:
    """Check if cm_cli module is available in the current Python environment.

    Only checks the currently activated Python environment.
    Does NOT fallback to PATH lookup to avoid using cm-cli from different environments.

    Results are cached for the session lifetime.

    Returns:
        True if cm_cli module is importable, False otherwise.
    """
    return importlib.util.find_spec("cm_cli") is not None


def execute_cm_cli(args, channel=None, fast_deps=False, no_deps=False, mode=None, raise_on_error=False) -> str | None:
    _config_manager = ConfigManager()

    workspace_path = workspace_manager.workspace_path

    if not workspace_path:
        print("\n[bold red]ComfyUI path is not resolved.[/bold red]\n", file=sys.stderr)
        raise typer.Exit(code=1)

    if not find_cm_cli():
        print(
            "\n[bold red]ComfyUI-Manager not found. 'cm-cli' command is not available.[/bold red]\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    cmd = [sys.executable, "-m", "cm_cli"] + args

    if channel is not None:
        cmd += ["--channel", channel]

    if fast_deps or no_deps:
        cmd += ["--no-deps"]

    if mode is not None:
        cmd += ["--mode", mode]

    new_env = os.environ.copy()
    session_path = os.path.join(_config_manager.get_config_path(), "tmp", str(uuid.uuid4()))
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["COMFYUI_PATH"] = workspace_path

    print(f"Execute from: {workspace_path}")
    print(f"Command: {cmd}")
    try:
        result = subprocess.run(
            cmd, env=new_env, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        print(result.stdout)

        if fast_deps and args[0] in _dependency_cmds:
            # we're using the fast_deps behavior and just ran a command that invalidated the dependencies
            depComp = DependencyCompiler(cwd=workspace_path)
            depComp.compile_deps()
            depComp.install_deps()

        return result.stdout
    except subprocess.CalledProcessError as e:
        if raise_on_error:
            if e.stdout:
                print(e.stdout)
            if e.stderr:
                print(e.stderr, file=sys.stderr)
            raise e

        if e.returncode == 1:
            print(f"\n[bold red]Execution error: {cmd}[/bold red]\n", file=sys.stderr)
            return None

        if e.returncode == 2:
            return None

        raise e
    finally:
        workspace_manager.set_recent_workspace(workspace_path)
