"""
Module for checking various env and state conditions.
"""

import os
import sys

import requests
from rich.console import Console

from comfy_cli.config_manager import ConfigManager
from comfy_cli.utils import singleton

console = Console()


def format_python_version(version_info):
    """
    Formats the Python version string to display the major and minor version numbers.

    If the minor version is greater than 8, the version is displayed in normal text.
    If the minor version is 8 or less, the version is displayed in bold red text to indicate an older version.

    Args:
        version_info (sys.version_info): The Python version information

    Returns:
        str: The formatted Python version string.
    """
    if version_info.major == 3 and version_info.minor > 8:
        return f"{version_info.major}.{version_info.minor}.{version_info.micro}"
    return f"[bold red]{version_info.major}.{version_info.minor}.{version_info.micro}[/bold red]"


def _bracket_host(host: str) -> str:
    """Bracket a bare IPv6 literal (``::1`` -> ``[::1]``) for use in a URL.

    Idempotent: an already-bracketed host (as returned by
    ``host_port.resolve_host_port``) and hostnames / IPv4 (no ``:``) pass
    through unchanged, so it's safe to apply at a shared choke point regardless
    of whether the caller pre-bracketed.
    """
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def check_comfy_server_running(port=8188, host="localhost", timeout: float = 5.0):
    """
    Checks if the Comfy server is running by making a GET request to the /history endpoint.

    `timeout` bounds the probe so a TCP-reachable but unresponsive server
    (e.g., stuck in a CUDA kernel) doesn't hang the caller.

    Returns:
        bool: True if the Comfy server is running, False otherwise.
    """
    try:
        response = requests.get(f"http://{_bracket_host(host)}:{port}/history", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _resolved_local_address() -> tuple[str, int]:
    """The local ComfyUI ``(host, port)`` ``comfy env`` should probe + report.

    Honors the same precedence as every other local command minus the
    per-command flag (``comfy env`` takes none): ``COMFY_LOCAL_URL`` env >
    ``config.background`` > ``127.0.0.1:8188``.
    """
    from comfy_cli.local_address import resolve_local_host_port

    return resolve_local_host_port(None, None, background=ConfigManager().background)


def _display_url(host: str, port: int) -> str:
    """``http://host:port`` with IPv6 literals bracketed for a valid URL."""
    return f"http://{_bracket_host(host)}:{port}"


@singleton
class EnvChecker:
    """
    Provides an `EnvChecker` class to check the current environment and print information about it.

    - `virtualenv_path`: The path to the current virtualenv, or "Not Used" if not in a virtualenv.
    - `conda_env`: The name of the current conda environment, or "Not Used" if not in a conda environment.
    - `python_version`: The version information for the current Python installation.
    - `currently_in_comfy_repo`: A boolean indicating whether the current directory is part of the Comfy repository.

    The `EnvChecker` class is a singleton that checks the current environment
    and stores information about the Python version, virtualenv path, conda
    environment, and whether the current directory is part of the Comfy
    repository.


    The `print()` method of the `EnvChecker` class displays the collected
    environment information in a formatted table.
    """

    def __init__(self):
        self.virtualenv_path = None
        self.conda_env = None
        self.python_version = sys.version_info
        self.check()

    def is_isolated_env(self):
        return self.virtualenv_path or self.conda_env

    def get_isolated_env(self):
        if self.virtualenv_path:
            return self.virtualenv_path

        if self.conda_env:
            return self.conda_env

        return None

    def check(self):
        self.virtualenv_path = os.environ.get("VIRTUAL_ENV") if os.environ.get("VIRTUAL_ENV") else None
        self.conda_env = os.environ.get("CONDA_DEFAULT_ENV") if os.environ.get("CONDA_DEFAULT_ENV") else None

    def fill_print_table(self):
        data = []
        data.append(("Python Version", format_python_version(sys.version_info)))
        data.append(("Python Executable", sys.executable))
        data.append(
            (
                "Virtualenv Path",
                self.virtualenv_path if self.virtualenv_path else "Not Used",
            )
        )
        data.append(("Conda Env", self.conda_env if self.conda_env else "Not Used"))

        config_data = ConfigManager().get_env_data()
        data.extend(config_data)

        host, port = _resolved_local_address()
        if check_comfy_server_running(port=port, host=host):
            data.append(
                (
                    "Comfy Server Running",
                    f"[bold green]Yes[/bold green]\n{_display_url(host, port)}",
                )
            )
        else:
            data.append(("Comfy Server Running", "[bold red]No[/bold red]"))

        return data

    def fill_data(self) -> dict:
        """Structured snapshot of the environment, used by ``comfy env --json``.

        Distinct from ``fill_print_table`` (which returns Rich-formatted tuples
        for display): this returns a JSON-serializable dict that validates
        against ``schemas/env.json``.
        """
        cm = ConfigManager()
        host, port = _resolved_local_address()
        server_running = check_comfy_server_running(port=port, host=host)
        return {
            "python": {
                "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                "executable": sys.executable,
                "virtualenv": self.virtualenv_path,
                "conda_env": self.conda_env,
            },
            "config": cm.get_data(),
            "server": {
                "running": server_running,
                "url": _display_url(host, port) if server_running else None,
            },
        }
