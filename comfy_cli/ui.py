from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import questionary
import typer
from questionary import Choice
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from comfy_cli.workspace_manager import WorkspaceManager

console = Console()
workspace_manager = WorkspaceManager()


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


ChoiceType = Union[str, Choice, Dict[str, Any]]


def prompt_autocomplete(
    question: str, choices: List[ChoiceType], default: ChoiceType = "", force_prompting: bool = False
) -> Optional[ChoiceType]:
    """
    Asks a single select question using questionary and returns the selected response.

    Args:
        question (str): The question to display to the user.
        choices (List[ChoiceType]): A list of choices the user can autocomplete from.
        default (ChoiceType): Default choice.
        force_prompting (bool): Whether to force prompting even if skip_prompting is set.

    Returns:
        Optional[ChoiceType]: The selected choice from the user, or None if skipping prompts.
    """
    if workspace_manager.skip_prompting and not force_prompting:
        return None
    return questionary.autocomplete(question, choices=choices, default=default).ask()


def prompt_select(
    question: str, choices: List[ChoiceType], default: ChoiceType = "", force_prompting: bool = False
) -> Optional[ChoiceType]:
    """
    Asks a single select question using questionary and returns the selected response.

    Args:
        question (str): The question to display to the user.
        choices (List[ChoiceType]): A list of choices for the user to select from.
        default (ChoiceType): Default choice.
        force_prompting (bool): Whether to force prompting even if skip_prompting is set.

    Returns:
        Optional[ChoiceType]: The selected choice from the user, or None if skipping prompts.
    """
    if workspace_manager.skip_prompting and not force_prompting:
        return None
    return questionary.select(question, choices=choices, default=default).ask()


E = TypeVar("E", bound=Enum)


def prompt_select_enum(question: str, choices: List[E], force_prompting: bool = False) -> Optional[E]:
    """
    Asks a single select question using questionary and returns the selected response.

    Args:
        question (str): The question to display to the user.
        choices (List[E]): A list of Enum choices for the user to select from.
        force_prompting (bool): Whether to force prompting even if skip_prompting is set.

    Returns:
        Optional[E]: The selected Enum choice from the user, or None if skipping prompts.
    """
    if workspace_manager.skip_prompting and not force_prompting:
        return None

    choice_map = {choice.value: choice for choice in choices}
    display_choices = list(choice_map.keys())

    selected = questionary.select(question, choices=display_choices).ask()

    return choice_map[selected] if selected is not None else None


def prompt_input(question: str, default: str = "", force_prompting: bool = False) -> str:
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
    if workspace_manager.skip_prompting and not force_prompting:
        return default
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
    selections = questionary.checkbox(prompt, choices=choices).ask()  # returns list of selected items
    return selections if selections else []


def prompt_confirm_action(prompt: str, default: bool) -> bool:
    """
    Prompts the user for confirmation before proceeding with an action.

    Args:
        prompt (str): The confirmation message to display to the user.

    Returns:
        bool: True if the user confirms, False otherwise.
    """
    if workspace_manager.skip_prompting:
        return default

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


def display_error_message(message: str) -> None:
    """
    Displays an error message to the user in red text.

    Args:
        message (str): The error message to display.
    """
    console.print(f"[red]{message}[/]")
