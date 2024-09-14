import sys
from importlib.metadata import metadata

import requests
from packaging import version
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_for_newer_pypi_version(package_name: str, current_version: str, timeout: float) -> tuple[bool, str]:
    """
    Checks if a newer version of the specified package is available on PyPI.

    :param package_name: The name of the package to check.
    :param current_version: The current version of the package.
    :param timeout: Timeout in seconds for the request to PyPI.
    :return: A tuple where the first value indicates if a newer version is available,
             and the second value is the latest version (or the current version if no update is found).
    """
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred
        latest_version = response.json()["info"]["version"]

        if version.parse(latest_version) > version.parse(current_version):
            return True, latest_version

        return False, current_version
    except requests.RequestException:
        # Fail quietly on timeout or any request exception
        return False, current_version


def check_for_updates(timeout: float = 10) -> None:
    """
    Checks for updates to the 'comfy-cli' package by comparing the current version
    to the latest version on PyPI. If a newer version is available, a notification
    is displayed.

    :param timeout: (default 10) Timeout in seconds for the request to check for updates.
                    If not provided, no timeout is enforced.
    """
    current_version = get_version_from_pyproject()
    has_newer, newer_version = check_for_newer_pypi_version("comfy-cli", current_version, timeout=timeout)

    if has_newer:
        notify_update(current_version, newer_version)


def get_version_from_pyproject() -> str:
    package_metadata = metadata("comfy-cli")
    return package_metadata["Version"]


def notify_update(current_version: str, newer_version: str) -> None:
    """
    Notifies the user that a newer version of the 'comfy-cli' package is available.

    :param current_version: The current version of the package.
    :param newer_version: The newer version available on PyPI.
    """
    message = (
        f":sparkles: Newer version of [bold magenta]comfy-cli[/bold magenta] is available: [bold green]{newer_version}[/bold green].\n"
        f"Current version: [bold cyan]{current_version}[/bold cyan]\n"
        f"Update by running: [bold yellow]'pip install --upgrade comfy-cli'[/bold yellow] :arrow_up:"
    )

    if sys.platform == "win32":
        # windows cannot display emoji characters.
        bell = ""
        message = message.replace(":sparkles:", "")
        message = message.replace(":arrow_up:", "")
    else:
        bell = ":bell:"

    console.print(
        Panel(
            message,
            title=f"[bold red]{bell} Update Available![/bold red]",
            border_style="bright_blue",
        )
    )
