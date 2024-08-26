from __future__ import annotations

import os
import subprocess
import sys
import uuid

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


def execute_cm_cli(args, channel=None, fast_deps=False, mode=None) -> str | None:
    _config_manager = ConfigManager()

    workspace_path = workspace_manager.workspace_path

    if not workspace_path:
        print("\n[bold red]ComfyUI path is not resolved.[/bold red]\n", file=sys.stderr)
        raise typer.Exit(code=1)

    cm_cli_path = os.path.join(workspace_path, "custom_nodes", "ComfyUI-Manager", "cm-cli.py")
    if not os.path.exists(cm_cli_path):
        print(
            f"\n[bold red]ComfyUI-Manager not found: {cm_cli_path}[/bold red]\n",
            file=sys.stderr,
        )
        raise typer.Exit(code=1)

    cmd = [sys.executable, cm_cli_path] + args

    if channel is not None:
        cmd += ["--channel", channel]

    if fast_deps:
        cmd += ["--no-deps"]

    if mode is not None:
        cmd += ["--mode", mode]

    new_env = os.environ.copy()
    session_path = os.path.join(_config_manager.get_config_path(), "tmp", str(uuid.uuid4()))
    new_env["__COMFY_CLI_SESSION__"] = session_path
    new_env["COMFYUI_PATH"] = workspace_path

    print(f"Execute from: {workspace_path}")

    try:
        result = subprocess.run(cmd, env=new_env, check=True, capture_output=True, text=True)
        print(result.stdout)

        if fast_deps and args[0] in _dependency_cmds:
            # we're using the fast_deps behavior and just ran a command that invalidated the dependencies
            depComp = DependencyCompiler(cwd=workspace_path)
            depComp.compile_deps()
            depComp.install_deps()

        return result.stdout
    except subprocess.CalledProcessError as e:
        if e.returncode == 1:
            print(f"\n[bold red]Execution error: {cmd}[/bold red]\n", file=sys.stderr)
            return None

        if e.returncode == 2:
            return None

        raise e
    finally:
        workspace_manager.set_recent_workspace(workspace_path)
