import sys
from importlib.metadata import metadata

import requests
from packaging import version
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_for_newer_pypi_version(package_name, current_version):
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred
        latest_version = response.json()["info"]["version"]

        if version.parse(latest_version) > version.parse(current_version):
            return True, latest_version

        return False, current_version
    except requests.RequestException:
        # print(f"Error checking latest version: {e}")
        return False, current_version


def check_for_updates():
    current_version = get_version_from_pyproject()
    has_newer, newer_version = check_for_newer_pypi_version("comfy-cli", current_version)

    if has_newer:
        notify_update(current_version, newer_version)


def get_version_from_pyproject():
    package_metadata = metadata("comfy-cli")
    return package_metadata["Version"]


def notify_update(current_version: str, newer_version: str):
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
