from __future__ import annotations

import os
import platform
import subprocess
import sys

from rich import print as rprint


def _get_python_binary(env_path: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(env_path, "Scripts", "python.exe")
    return os.path.join(env_path, "bin", "python")


def resolve_workspace_python(workspace_path: str | None = None) -> str:
    if virtual_env := os.environ.get("VIRTUAL_ENV"):
        python = _get_python_binary(virtual_env)
        if os.path.isfile(python):
            return python

    if conda_prefix := os.environ.get("CONDA_PREFIX"):
        python = _get_python_binary(conda_prefix)
        if os.path.isfile(python):
            return python

    if workspace_path is not None:
        for venv_name in (".venv", "venv"):
            venv_dir = os.path.join(workspace_path, venv_name)
            if os.path.isdir(venv_dir):
                python = _get_python_binary(venv_dir)
                if os.path.isfile(python):
                    return python

    return sys.executable


def create_workspace_venv(workspace_path: str) -> str:
    venv_dir = os.path.join(workspace_path, ".venv")
    rprint(f"Creating workspace virtual environment at [bold]{venv_dir}[/bold]")
    subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
    python = _get_python_binary(venv_dir)
    if not os.path.isfile(python):
        raise RuntimeError(f"Failed to create venv: {python} not found after creation")
    return python


def ensure_workspace_python(workspace_path: str) -> str:
    if os.environ.get("VIRTUAL_ENV") or os.environ.get("CONDA_PREFIX"):
        return resolve_workspace_python(workspace_path)

    for venv_name in (".venv", "venv"):
        venv_dir = os.path.join(workspace_path, venv_name)
        if os.path.isdir(venv_dir):
            python = _get_python_binary(venv_dir)
            if os.path.isfile(python):
                return python

    # Running from the system/global Python (e.g. Docker root installs, global pip installs)
    # Safe to install deps directly — no venv needed.
    if sys.prefix == sys.base_prefix:
        return sys.executable

    # Running from an isolated tool environment (pipx, uv tool, etc.)
    # Must create a workspace venv to avoid polluting the tool's env
    return create_workspace_venv(workspace_path)
