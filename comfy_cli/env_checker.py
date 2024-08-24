"""
Module for checking various env and state conditions.
"""

import os
import sys

import requests
from rich.console import Console
from rich.table import Table

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
        return "{}.{}.{}".format(version_info.major, version_info.minor, version_info.micro)
    return "[bold red]{}.{}.{}[/bold red]".format(version_info.major, version_info.minor, version_info.micro)


def check_comfy_server_running(port=8188, host="localhost"):
    """
    Checks if the Comfy server is running by making a GET request to the /history endpoint.

    Returns:
        bool: True if the Comfy server is running, False otherwise.
    """
    try:
        response = requests.get(f"http://{host}:{port}/history")
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@singleton
class EnvChecker(object):
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

    # TODO: use ui.display_table
    def fill_print_table(self):
        table = Table(":laptop_computer: Environment", "Value")
        table.add_row("Python Version", format_python_version(sys.version_info))
        table.add_row("Python Executable", sys.executable)
        table.add_row(
            "Virtualenv Path",
            self.virtualenv_path if self.virtualenv_path else "Not Used",
        )
        table.add_row("Conda Env", self.conda_env if self.conda_env else "Not Used")

        ConfigManager().fill_print_env(table)

        if check_comfy_server_running():
            table.add_row(
                "Comfy Server Running",
                "[bold green]Yes[/bold green]\nhttp://localhost:8188",
            )
        else:
            table.add_row("Comfy Server Running", "[bold red]No[/bold red]")

        return table
