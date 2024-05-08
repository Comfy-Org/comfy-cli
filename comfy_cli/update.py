import requests
import tomlkit
from rich import print
import tomlkit.exceptions
from rich.console import Console
from rich.panel import Panel

console = Console()


def check_for_newer_pypi_version(package_name, current_version):
    url = f"https://pypi.org/pypi/{package_name}/json"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises stored HTTPError, if one occurred
        latest_version = response.json()["info"]["version"]
        return latest_version != current_version, latest_version
    except requests.RequestException as e:
        # print(f"Error checking latest version: {e}")
        return False, current_version


def check_for_updates():
    current_version = get_version_from_pyproject()
    has_newer, newer_version = check_for_newer_pypi_version(
        "comfy-cli", current_version
    )

    if has_newer:
        notify_update(current_version, newer_version)


def get_version_from_pyproject(file_path="pyproject.toml"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            toml_content = file.read()
            parsed_toml = tomlkit.parse(toml_content)
            # Accessing the project version under [project]
            if "project" in parsed_toml:
                version = parsed_toml["project"]["version"]
            else:
                raise KeyError(
                    "Version key not found under [project] in pyproject.toml"
                )
            return version
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except tomlkit.exceptions.TOMLKitError as e:
        print(f"Error parsing TOML: {e}")
    except KeyError as e:
        print(f"Error accessing version in TOML: {e}")


def notify_update(current_version: str, newer_version: str):
    message = (
        f":sparkles: Newer version of [bold magenta]comfy-cli[/bold magenta] is available: [bold green]{newer_version}[/bold green].\n"
        f"Current version: [bold cyan]{current_version}[/bold cyan]\n"
        f"Update by running: [bold yellow]'pip install --upgrade comfy-cli'[/bold yellow] :arrow_up:"
    )
    console.print(
        Panel(
            message,
            title="[bold red]:bell: Update Available![/bold red]",
            border_style="bright_blue",
        )
    )
