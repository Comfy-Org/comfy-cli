from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import threading
import uuid
from functools import lru_cache

import typer

from comfy_cli.config_manager import ConfigManager
from comfy_cli.output import rprint as print  # noqa: A001 - context-aware: stderr in JSON mode
from comfy_cli.resolve_python import resolve_workspace_python
from comfy_cli.uv import DependencyCompiler
from comfy_cli.workspace_manager import WorkspaceManager, check_comfy_repo

workspace_manager = WorkspaceManager()

# set of commands that invalidate (ie require an update of) dependencies after they are run
_dependency_cmds = {
    "install",
    "reinstall",
}


@lru_cache(maxsize=8)
def _probe_cm_cli(workspace_path: str | None) -> bool:
    """Probe for the ``cm_cli`` module for one workspace. Cached per workspace path."""
    if workspace_path:
        python = resolve_workspace_python(workspace_path)
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


def find_cm_cli() -> bool:
    """Check if cm_cli module is available in the workspace Python.

    First checks the workspace venv Python (primary path — matches the Python
    used by execute_cm_cli). Falls back to the current Python environment only
    when the workspace Python is the same as sys.executable.

    Results are cached per workspace path, so resolving (or switching) the
    workspace after a first call re-probes instead of returning a stale answer.
    """
    ws = workspace_manager.workspace_path
    return _probe_cm_cli(str(ws) if ws else None)


# Preserve the ``lru_cache`` API callers rely on to force a re-probe.
find_cm_cli.cache_clear = _probe_cm_cli.cache_clear


@lru_cache(maxsize=8)
def _scan_for_legacy_manager_clone(workspace_path: str) -> bool:
    """Scan one workspace's ``custom_nodes/`` for a Manager clone. Cached per path."""
    custom_nodes = os.path.join(workspace_path, "custom_nodes")
    try:
        entries = os.listdir(custom_nodes)
    except OSError:
        return False

    for name in entries:
        if name.endswith(".disabled"):
            continue
        node_dir = os.path.join(custom_nodes, name)
        try:
            if not os.path.isdir(node_dir):
                continue
            if os.path.isfile(os.path.join(node_dir, "glob", "manager_core.py")) or os.path.isfile(
                os.path.join(node_dir, "cm-cli.py")
            ):
                return True
        except OSError:
            continue

    return False


def find_legacy_manager_clone() -> bool:
    """Check if ComfyUI-Manager exists as a plain git clone under ``custom_nodes/``.

    This is the pre-pip-package install shape: the Manager repo cloned straight
    into ``custom_nodes/``. It is invisible to :func:`find_cm_cli`, which only
    tests whether the ``cm_cli`` module imports in the workspace Python.

    Detection is by marker file, not directory name — users clone the Manager
    under arbitrary names. Directories ending in ``.disabled`` (the Manager's own
    disable convention) are skipped.

    Results are cached per workspace path, so an unresolved workspace on the
    first call does not poison the answer for the rest of the session.
    """
    ws = workspace_manager.workspace_path
    if not ws:
        return False
    return _scan_for_legacy_manager_clone(str(ws))


# Preserve the ``lru_cache`` API callers rely on to force a re-scan.
find_legacy_manager_clone.cache_clear = _scan_for_legacy_manager_clone.cache_clear


def detect_manager_installation() -> str:
    """Report how ComfyUI-Manager is installed for the current workspace.

    Returns ``"venv-package"`` (the pip ``comfyui_manager`` package — cm-cli
    usable), ``"legacy-clone"`` (an on-disk clone under ``custom_nodes/`` — cm-cli
    integration unavailable), or ``"none"``.
    """
    if find_cm_cli():
        return "venv-package"
    if find_legacy_manager_clone():
        return "legacy-clone"
    return "none"


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


def normalize_cm_cli_exit_code(returncode: int) -> int:
    """Map a ``CalledProcessError.returncode`` from :func:`execute_cm_cli` onto an
    exit code that survives ``sys.exit``.

    ``Popen.wait()`` reports ``-N`` when the child is killed by signal N (e.g. -9
    when the OOM killer reaps cm-cli mid dependency build), and Windows can return
    values far above 255. POSIX truncates to the low byte, so -9 would surface as a
    fabricated 247 and any multiple of 256 as 0 — a failure reporting success.
    Normalize to the shell's 128+N signal convention, and never return 0.

    A low byte of 2 is remapped because Click already uses 2 for its own usage
    errors; leaving it through (whether as a literal 2 or a wide status like 258
    that ``sys.exit`` truncates back to 2) would make "you invoked comfy wrong"
    indistinguishable from "cm-cli exited 2". Callers should keep the raw status
    in their error ``details``.
    """
    if returncode < 0:
        return min(128 + abs(returncode), 255)
    if returncode % 256 in (0, 2):
        return 1
    return returncode


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
            assert process.stderr is not None
            for line in process.stderr:
                sys.stderr.write(line)
                sys.stderr.flush()
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=_drain_stderr, daemon=True)
        stderr_thread.start()

        assert process.stdout is not None
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
