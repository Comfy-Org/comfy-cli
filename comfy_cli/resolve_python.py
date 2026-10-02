from __future__ import annotations

import os
import platform
import subprocess
import sys
import sysconfig

from comfy_cli.output import rprint


def _get_python_binary(env_path: str) -> str:
    if platform.system() == "Windows":
        return os.path.join(env_path, "Scripts", "python.exe")
    return os.path.join(env_path, "bin", "python")


def _is_externally_managed() -> bool:
    """Detect PEP 668 externally-managed Python (e.g. Ubuntu 24.04 system Python)."""
    stdlib = sysconfig.get_path("stdlib")
    return bool(stdlib) and os.path.isfile(os.path.join(stdlib, "EXTERNALLY-MANAGED"))


def _find_portable_python(workspace_path: str) -> str | None:
    """Return the official Windows portable build's interpreter, if workspace_path is one.

    The portable archive ships its own interpreter in a ``python_embeded``
    directory. That directory is a sibling of the ``ComfyUI`` workspace (the
    layout ``comfy install`` records in config), but is also a child of the
    workspace when the workspace is the portable root itself. Both are probed.

    The embedded interpreter is a plain ``python.exe`` directly inside the
    directory, not a venv, so ``_get_python_binary`` does not apply here.

    Returns ``None`` when the layout is absent, so callers fall back to their
    existing resolution order unchanged.
    """
    bases = [workspace_path]
    parent = os.path.dirname(workspace_path)
    if parent and parent != workspace_path:
        bases.append(parent)

    for base in bases:
        python = os.path.join(base, "python_embeded", "python.exe")
        if os.path.isfile(python):
            return python

    return None


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
        # A portable workspace carries its own interpreter; prefer it over any
        # venv/virtualenv so workspace operations run in ComfyUI's environment.
        if portable := _find_portable_python(workspace_path):
            return portable

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

    # Portable workspaces already have an interpreter with ComfyUI's
    # dependencies; never create a venv beside it or fall through to the
    # system Python comfy-cli itself runs on.
    if portable := _find_portable_python(workspace_path):
        return portable

    for venv_name in (".venv", "venv"):
        venv_dir = os.path.join(workspace_path, venv_name)
        if os.path.isdir(venv_dir):
            python = _get_python_binary(venv_dir)
            if os.path.isfile(python):
                return python

    # Running from the system/global Python (e.g. Docker root installs, global pip installs).
    if sys.prefix == sys.base_prefix:
        if _is_externally_managed():
            # PEP 668: system Python is locked (e.g. Ubuntu 24.04), need a workspace venv.
            return create_workspace_venv(workspace_path)
        return sys.executable

    # Running from an isolated tool environment (pipx, uv tool, etc.)
    # Must create a workspace venv to avoid polluting the tool's env
    return create_workspace_venv(workspace_path)
