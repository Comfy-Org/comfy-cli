from enum import Enum
import questionary
import typer
from rich.console import Console
from rich.progress import Progress
from rich.table import Table
from typing import List, Tuple

console = Console()


def show_progress(iterable, total, description="Downloading..."):
    """
    Display progress bar for iterable processes, especially useful for file downloads.
    Each item in the iterable should be a chunk of data, and the progress bar will advance
    by the size of each chunk.

    Args:
        iterable (Iterable[bytes]): An iterable that yields chunks of data.
        total (int): The total size of the data (e.g., total number of bytes) to be downloaded.
        description (str): Description text for the progress bar.

    Yields:
        bytes: Chunks of data as they are processed.
    """
    with Progress() as progress:
        task = progress.add_task(description, total=total)
        for chunk in iterable:
            yield chunk
            progress.update(task, advance=len(chunk))


def prompt_select(question: str, choices: list) -> str:
    """
    Asks a single select question using questionary and returns the selected response.

    Args:
        question (str): The question to display to the user.
        choices (list): A list of string choices for the user to select from.

    Returns:
        str: The selected choice from the user.
    """
    return questionary.select(question, choices=choices).ask()


def prompt_select_enum(question: str, choices: list) -> str:
    """
    Asks a single select question using questionary and returns the selected response.

    Args:
        question (str): The question to display to the user.
        choices (list): A list of Enum choices for the user to select from.

    Returns:
        str: The selected choice from the user.
    """
    choice_map = {choice.value: choice for choice in choices}
    display_choices = list(choice_map.keys())

    selected = questionary.select(question, choices=display_choices).ask()

    return choice_map[selected]


def prompt_input(question: str, default: str = "") -> str:
    """
    Asks the user for an input using questionary.

    Args:
        question (str): The question to display to the user.
        default (str): The default value for the input.

    Returns:
        str: The user's input.

    Raises:
        KeyboardInterrupt: If the user interrupts the input.
    """
    return questionary.text(question, default=default).ask()


def prompt_multi_select(prompt: str, choices: List[str]) -> List[str]:
    """
    Prompts the user to select multiple items from a list of choices.

    Args:
        prompt (str): The message to display to the user.
        choices (List[str]): A list of choices from which the user can select.

    Returns:
        List[str]: A list of the selected items.
    """
    selections = questionary.checkbox(
        prompt, choices=choices
    ).ask()  # returns list of selected items
    return selections if selections else []


def prompt_confirm_action(prompt: str) -> bool:
    """
    Prompts the user for confirmation before proceeding with an action.

    Args:
        prompt (str): The confirmation message to display to the user.

    Returns:
        bool: True if the user confirms, False otherwise.
    """

    return typer.confirm(prompt)


def display_table(data: List[Tuple], column_names: List[str], title: str = "") -> None:
    """
    Displays a list of tuples in a table format using Rich.

    Args:
        data (List[Tuple]): A list of tuples, where each tuple represents a row.
        column_names (List[str]): A list of column names for the table.
        title (str): The title of the table.
    """
    table = Table(title=title)

    for name in column_names:
        table.add_column(name, overflow="fold")

    for row in data:
        table.add_row(*[str(item) for item in row])

    console.print(table)
