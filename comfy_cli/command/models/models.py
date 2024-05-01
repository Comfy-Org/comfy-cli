import pathlib
from typing import List, Optional

import typer

from typing_extensions import Annotated

from comfy_cli import tracking, ui
from comfy_cli.constants import DEFAULT_COMFY_MODEL_PATH
from comfy_cli.workspace_manager import WorkspaceManager

app = typer.Typer()
workspace_manager = WorkspaceManager()


def get_workspace(ctx: typer.Context) -> pathlib.Path:
    workspace_path = workspace_manager.get_workspace_comfy_path(ctx)
    return pathlib.Path(workspace_path)


@app.command()
@tracking.track_command("model")
def download(
    ctx: typer.Context,
    url: Annotated[
        str,
        typer.Option(
            help="The URL from which to download the model", show_default=False
        ),
    ],
    relative_path: Annotated[
        Optional[str],
        typer.Option(
            help="The relative path from the current workspace to install the model.",
            show_default=True,
        ),
    ] = DEFAULT_COMFY_MODEL_PATH,
):
    """Download a model to a specified relative path if it is not already downloaded."""
    # Convert relative path to absolute path based on the current working directory
    local_filename = url.split("/")[-1]
    local_filepath = get_workspace(ctx) / relative_path / local_filename

    # Check if the file already exists
    if local_filepath.exists():
        typer.echo(f"File already exists: {local_filepath}")
        return

    # File does not exist, proceed with download
    typer.echo(f"Start downloading URL: {url} into {local_filepath}")
    download_file(url, local_filepath)


@app.command()
@tracking.track_command("model")
def remove(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
    model_names: Optional[List[str]] = typer.Option(
        None,
        help="List of model filenames to delete, separated by spaces",
        show_default=False,
    ),
):
    """Remove one or more downloaded models, either by specifying them directly or through an interactive selection."""
    model_dir = get_workspace(ctx) / relative_path
    available_models = list_models(model_dir)

    if not available_models:
        typer.echo("No models found to remove.")
        return

    to_delete = []
    # Scenario #1: User provided model names to delete
    if model_names:
        # Validate and filter models to delete based on provided names
        missing_models = []
        for name in model_names:
            model_path = model_dir / name
            if model_path.exists():
                to_delete.append(model_path)
            else:
                missing_models.append(name)

        if missing_models:
            typer.echo(
                "The following models were not found and cannot be removed: "
                + ", ".join(missing_models)
            )
            if not to_delete:
                return  # Exit if no valid models were found

        return

    # Scenario #2: User did not provide model names, prompt for selection
    else:
        selections = ui.prompt_multi_select(
            "Select models to delete:", [model.name for model in available_models]
        )
        if not selections:
            typer.echo("No models selected for deletion.")
            return
        to_delete = [model_dir / selection for selection in selections]

    # Confirm deletion
    if to_delete and ui.prompt_confirm_action(
        "Are you sure you want to delete the selected files?"
    ):
        for model_path in to_delete:
            model_path.unlink()
            typer.echo(f"Deleted: {model_path}")
    else:
        typer.echo("Deletion canceled.")


@app.command()
@tracking.track_command("model")
def list(
    ctx: typer.Context,
    relative_path: str = typer.Option(
        DEFAULT_COMFY_MODEL_PATH,
        help="The relative path from the current workspace where the models are stored.",
        show_default=True,
    ),
):
    """Display a list of all models currently downloaded in a table format."""
    model_dir = get_workspace(ctx) / relative_path
    models = list_models(model_dir)

    if not models:
        typer.echo("No models found.")
        return

    # Prepare data for table display
    data = [(model.name, f"{model.stat().st_size // 1024} KB") for model in models]
    column_names = ["Model Name", "Size"]
    ui.display_table(data, column_names)


def download_file(url: str, local_filepath: pathlib.Path):
    """Helper function to download a file."""

    import httpx

    local_filepath.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure the directory exists

    with httpx.stream("GET", url, follow_redirects=True) as response:
        total = int(response.headers["Content-Length"])
        with open(local_filepath, "wb") as f:
            for data in ui.show_progress(response.iter_bytes(), total):
                f.write(data)


def list_models(path: pathlib.Path) -> list:
    """List all models in the specified directory."""
    return [file for file in path.iterdir() if file.is_file()]
