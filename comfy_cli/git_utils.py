import os
import subprocess

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()


def git_checkout_tag(repo_path: str, tag: str) -> bool:
    """
    Checkout a specific Git tag in the given repository.

    :param repo_path: Path to the Git repository
    :param tag: The tag to checkout
    :return: The output of the git command if successful, None if an error occurred
    """
    original_dir = os.getcwd()
    try:
        # Change to the repository directory

        os.chdir(repo_path)

        # Fetch the latest tags
        subprocess.run(["git", "fetch", "--tags"], check=True, capture_output=True, text=True)

        # Checkout the specified tag
        subprocess.run(["git", "checkout", tag], check=True, capture_output=True, text=True)

        console.print(f"[bold green]Successfully checked out tag: [cyan]{tag}[/cyan][/bold green]")

        return True
    except subprocess.CalledProcessError as e:
        error_message = Text()
        error_message.append("Git Checkout Error", style="bold red on white")
        error_message.append("\n\nFailed to checkout tag: ", style="bold yellow")
        error_message.append(f"[cyan]{tag}[/cyan]")
        error_message.append("\n\nError details:", style="bold red")
        error_message.append(f"\n{str(e)}", style="italic")

        if e.stderr:
            error_message.append("\n\nError output:", style="bold red")
            error_message.append(f"\n{e.stderr}", style="italic yellow")

        console.print(
            Panel(
                error_message,
                title="[bold white on red]Git Checkout Failed[/bold white on red]",
                border_style="red",
                expand=False,
            )
        )

        return False
    finally:
        # Ensure we always return to the original directory
        os.chdir(original_dir)
